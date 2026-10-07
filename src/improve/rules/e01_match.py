# 역할: E-01 일치 계산. replay.py 가 쓴 조합별 알람 목록과 판정 불가 구간을 읽어 docs/improvement/PREREG_E-01.md 5~7항의 정의대로 센다.
#   - 구간별 창 안 알람 수, 구간별 일치 건수 = min(창 안 알람 수, 그 구간의 사건 수), 전체 일치 건수(최대 6)
#   - 창 안 초과 알람(창 안 알람 수 - 일치 건수), 창 밖 추가 알람(시각, 지속 시간, 구간과의 거리)
#   - 판정 불가 구간 수, 총 길이, 최장 길이, 해석 구간 문구
# 입력: --replay <이름>=<폴더> (이름은 detected 또는 detected+tracked, 폴더는 replay.py 의 출력 폴더). 이름마다 임계값 0.15, 0.21, 0.25 의
#       alarms_thr_<임계값>.csv, episodes_thr_<임계값>.csv, unjudgeable_thr_<임계값>.csv 가 있어야 한다.
# 출력: --out-dir(필수, 저장소 안의 기본값 없음) 아래에 e01_match.json, e01_match.txt. 이미 있으면 덮어쓰지 않고 중단한다.
# 구간, 사건 수, 허용 오차, 창, 해석 구간은 아래 상수이며 PREREG_E-01.md 본문과 같아야 한다(시험이 본문에서 읽어 확인한다). 값은 바꾸지 않는다.
#
# 문서가 정하지 않아 "제안"으로 정한 해석:
#   (제안 1) 경계 포함: 창의 시작과 끝 시각은 창 안으로 센다(이상, 이하). 부동소수점 오차 허용 1e-9초.
#   (제안 2) 추가 알람의 "구간과의 거리"는 가장 가까운 구간 창 경계까지의 초 거리이다(창 안이면 0). 구간 근접 여부는 알람 시각이 구간 시각
#            (창 허용 오차를 뺀 영상 시각) 안에 있는지로 나눈다: "구간 안(창 밖)" 또는 "구간 밖".
#   (제안 3) 알람이 둘 이상의 창에 걸리면 앞의 창에 센다(창은 겹치지 않으므로 실제로는 일어나지 않는다).
#   (제안 4) 해석 문구는 모든 조합에 계산하되 주 결과는 (임계값 0.21, detected 만) 조합이다.
import argparse
import csv
import json
import os
import sys

EPS = 1e-9

# PREREG_E-01.md 2항: (구간 번호, 영상 시각 시작 초, 끝 초, 사건 수). 0:12~0:19, 0:45~1:05, 1:09~1:22.
INTERVALS = ((1, 12.0, 19.0, 1), (2, 45.0, 65.0, 2), (3, 69.0, 82.0, 3))
WINDOW_START_OFFSET_S = 2.0     # 구간 창 = 구간 시작 +2.0초 ~ 끝 +1.0초(PREREG 5항)
WINDOW_END_OFFSET_S = 1.0
THRESHOLDS = (0.15, 0.21, 0.25)
SOURCE_SET_NAMES = ("detected", "detected+tracked")
MAIN_COMBINATION = (0.21, "detected")
# PREREG_E-01.md 6항 해석 구간: (최소 건수, 최대 건수, 문구)
INTERPRETATION = ((5, 6, "이 영상에서 대부분 일치"), (3, 4, "부분 일치, 원인 분석 필요(미확정)"), (0, 2, "불일치, 규칙 정의 재검토"))


def windows():
    """구간별 창 목록 [(구간 번호, 창 시작 초, 창 끝 초, 사건 수)]."""
    return [(index, start + WINDOW_START_OFFSET_S, end + WINDOW_END_OFFSET_S, events) for index, start, end, events in INTERVALS]


def total_events():
    return sum(events for _i, _s, _e, events in INTERVALS)


def interpretation(match_count):
    """전체 일치 건수에 대한 PREREG 6항의 해석 문구."""
    if isinstance(match_count, bool) or not isinstance(match_count, int) or not 0 <= match_count <= total_events():
        raise ValueError("일치 건수는 0~%d 의 정수여야 합니다: %r" % (total_events(), match_count))
    for low, high, text in INTERPRETATION:
        if low <= match_count <= high:
            return text
    raise ValueError("해석 구간에 없는 건수: %r" % (match_count,))


def format_time(seconds):
    """초를 m:ss.s 로 쓴다."""
    minutes = int(seconds // 60)
    return "%d:%04.1f" % (minutes, seconds - 60 * minutes)


def in_window(window, time_s):
    """알람 시각이 창 안인지(경계 포함, 제안 1)."""
    _index, start, end, _events = window
    return start - EPS <= time_s <= end + EPS


def distance_to_window(window, time_s):
    """창 경계까지의 초 거리(창 안이면 0)."""
    _index, start, end, _events = window
    if time_s < start:
        return start - time_s
    if time_s > end:
        return time_s - end
    return 0.0


def describe_extra(alarm, episodes):
    """창 밖 알람 하나의 설명(시각, 지속 시간, 가장 가까운 구간, 거리, 구간 시각 안 여부)."""
    time_s = alarm["alarm_time_s"]
    best = None
    for window in windows():
        distance = distance_to_window(window, time_s)
        if best is None or distance < best[1] - EPS:
            best = (window[0], distance)
    inside_interval = any(start - EPS <= time_s <= end + EPS for _i, start, end, _e in INTERVALS)
    episode = episodes.get(alarm["episode_id"], {})
    return {"alarm_time_s": time_s, "alarm_time": format_time(time_s), "episode_id": alarm["episode_id"],
            "episode_start_s": episode.get("start_s"), "episode_end_s": episode.get("end_s"),
            "episode_length_s": episode.get("length_s"), "episode_status": episode.get("status"),
            "nearest_interval": best[0], "distance_to_window_s": best[1],
            "relation": "구간 안(창 밖)" if inside_interval else "구간 밖"}


def compute_combination(alarms, episodes, unjudgeable):
    """조합 하나의 일치 계산.

    입력: alarms - dict 목록(episode_id, alarm_time_s ...), episodes - {episode_id: dict}, unjudgeable - dict 목록(length_s ...)
    반환: dict(구간별 창 안 알람 수, 일치 건수, 전체 일치 건수, 창 안 초과 알람, 추가 알람, 판정 불가 집계, 해석 문구)
    """
    wins = windows()
    in_counts = [0] * len(wins)
    in_times = [[] for _ in wins]
    extras = []
    for alarm in sorted(alarms, key=lambda row: row["alarm_time_s"]):
        for position, window in enumerate(wins):
            if in_window(window, alarm["alarm_time_s"]):
                in_counts[position] += 1
                in_times[position].append(alarm["alarm_time_s"])
                break
        else:
            extras.append(describe_extra(alarm, episodes))
    matches = [min(count, window[3]) for count, window in zip(in_counts, wins)]
    excess = [count - match for count, match in zip(in_counts, matches)]
    lengths = [row["length_s"] for row in unjudgeable]
    total = sum(matches)
    return {"alarms_in_window": in_counts, "alarm_times_in_window": in_times, "matches": matches, "total_match": total,
            "excess_in_window": excess, "extra_alarms": extras,
            "unjudgeable": {"n": len(lengths), "total_length_s": sum(lengths), "longest_s": max(lengths) if lengths else 0.0},
            "interpretation": interpretation(total)}


def _to_float(text):
    text = (text or "").strip()
    return float(text) if text else None


def read_alarms(path):
    """replay.py 의 alarms_thr_*.csv 를 읽는다."""
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return [{"episode_id": int(row["episode_id"]), "alarm_time_s": float(row["alarm_time_s"]),
                 "episode_start_s": _to_float(row["episode_start_s"]),
                 "duration_at_alarm_s": _to_float(row["duration_at_alarm_s"])} for row in csv.DictReader(handle)]


def read_episodes(path):
    """replay.py 의 episodes_thr_*.csv 를 {episode_id: dict} 로 읽는다."""
    result = {}
    with open(path, "r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            result[int(row["episode_id"])] = {"start_s": _to_float(row["start_s"]), "end_s": _to_float(row["end_s"]),
                                              "length_s": _to_float(row["length_s"]), "status": row["status"]}
    return result


def read_unjudgeable(path):
    """replay.py 의 unjudgeable_thr_*.csv 를 읽는다."""
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return [{"start_s": float(row["start_s"]), "end_s": float(row["end_s"]), "length_s": float(row["length_s"])}
                for row in csv.DictReader(handle)]


def threshold_label(threshold):
    return "%.2f" % threshold


def compute_all(replay_dirs):
    """이름 -> replay 폴더 매핑에서 6개 조합(임계값 3 x 유효 출처 2)을 모두 계산한다."""
    combinations = []
    for name in SOURCE_SET_NAMES:
        if name not in replay_dirs:
            raise ValueError("replay 폴더가 필요합니다: %s" % name)
        for threshold in THRESHOLDS:
            label = threshold_label(threshold)
            folder = replay_dirs[name]
            result = compute_combination(read_alarms(os.path.join(folder, "alarms_thr_%s.csv" % label)),
                                         read_episodes(os.path.join(folder, "episodes_thr_%s.csv" % label)),
                                         read_unjudgeable(os.path.join(folder, "unjudgeable_thr_%s.csv" % label)))
            result["threshold"] = threshold
            result["valid_sources"] = name
            result["is_main"] = (threshold, name) == MAIN_COMBINATION
            combinations.append(result)
    return combinations


def format_text(combinations):
    """사람이 읽는 텍스트(해석 문장 없이 건수와 구간만)."""
    lines = ["E-01 일치 계산(PREREG_E-01.md 5~7항 정의). 구간 창(초): " +
             ", ".join("구간%d %.1f~%.1f(사건 %d)" % (i, s, e, n) for i, s, e, n in windows()), ""]
    for item in combinations:
        lines.append("임계값 %s, 유효 출처 %s%s" % (threshold_label(item["threshold"]), item["valid_sources"],
                                               " (주 결과)" if item["is_main"] else ""))
        lines.append("  구간별 창 안 알람 수: %s, 구간별 일치 건수: %s, 전체 일치 건수: %d/%d" % (
            item["alarms_in_window"], item["matches"], item["total_match"], total_events()))
        lines.append("  창 안 초과 알람: %s" % item["excess_in_window"])
        lines.append("  창 밖 추가 알람 %d개" % len(item["extra_alarms"]))
        for extra in item["extra_alarms"]:
            lines.append("    %s (%.3f초), 구간 %d까지 %.3f초, %s, 구간 시작 %s, 길이 %s초" % (
                extra["alarm_time"], extra["alarm_time_s"], extra["nearest_interval"], extra["distance_to_window_s"],
                extra["relation"], "-" if extra["episode_start_s"] is None else format_time(extra["episode_start_s"]),
                "-" if extra["episode_length_s"] is None else "%.3f" % extra["episode_length_s"]))
        unj = item["unjudgeable"]
        lines.append("  판정 불가 구간: %d개, 총 %.3f초, 최장 %.3f초" % (unj["n"], unj["total_length_s"], unj["longest_s"]))
        lines.append("  해석 구간 문구: %s" % item["interpretation"])
        lines.append("")
    return "\n".join(lines) + "\n"


def run(replay_dirs, out_dir):
    """6개 조합을 계산해 out_dir 에 쓴다. 쓴 파일 이름 list 를 돌려준다."""
    if not out_dir:
        raise ValueError("출력 폴더가 필요합니다(기본값 없음).")
    combinations = compute_all(replay_dirs)
    os.makedirs(out_dir, exist_ok=True)
    payload = {"intervals": [{"index": i, "start_s": s, "end_s": e, "events": n} for i, s, e, n in INTERVALS],
               "window_start_offset_s": WINDOW_START_OFFSET_S, "window_end_offset_s": WINDOW_END_OFFSET_S,
               "boundary_policy": "창의 시작과 끝은 창 안(이상, 이하), 제안", "combinations": combinations}
    with open(os.path.join(out_dir, "e01_match.json"), "x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    with open(os.path.join(out_dir, "e01_match.txt"), "x", encoding="utf-8") as handle:
        handle.write(format_text(combinations))
    return ["e01_match.json", "e01_match.txt"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="replay.py 결과에서 E-01 일치를 계산한다.")
    parser.add_argument("--replay", action="append", required=True, metavar="이름=폴더",
                        help="이름은 detected 또는 detected+tracked, 폴더는 replay.py 출력 폴더(두 이름 모두 필요)")
    parser.add_argument("--out-dir", dest="out_dir", required=True, help="출력 폴더(필수, 기본값 없음)")
    return parser.parse_args(argv)


def main(argv=None):
    arguments = parse_args(argv)
    replay_dirs = {}
    for text in arguments.replay:
        name, separator, folder = text.partition("=")
        if not separator or name not in SOURCE_SET_NAMES or not folder:
            sys.exit("--replay 는 이름=폴더 형식이고 이름은 %s 중 하나여야 합니다: %r" % (", ".join(SOURCE_SET_NAMES), text))
        replay_dirs[name] = folder
    for name in run(replay_dirs, arguments.out_dir):
        print(name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
