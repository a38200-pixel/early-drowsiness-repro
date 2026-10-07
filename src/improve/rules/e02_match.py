# 역할: E-02 일치 계산. replay.py 가 쓴 조합별 알람 목록과 정답(e02_truth.json)을 읽어 docs/improvement/PREREG_E-02.md 5~7항의 정의대로 센다.
#   영상 2: 사건마다 창 [시작 + 2.0초, 끝 + 1.0초](경계 포함)에서 양성 일치, 창 안 초과, 음성 창 알람, 경계 창 알람, 구간별 추가 알람,
#           판정 불가 수·총 길이·최장, 해석 구간 문구.
#   영상 1: 알람 개수와 시각, EAR 중앙값(detected 만, detected+tracked), I-09 판정.
# 입력: --truth(정답 JSON), --video2-replay 이름=폴더, --video1-replay 이름=폴더(이름은 detected 또는 detected+tracked, 폴더는 replay.py 출력
#       폴더이며 임계값 0.15, 0.21, 0.25 의 alarms_thr_*.csv, episodes_thr_*.csv, unjudgeable_thr_*.csv 가 있어야 함),
#       --video1-dump(영상 1 프레임별 EAR CSV), --min-reference-s(I-09 기준 구간 최소 길이, 기본값 없음), --out-dir(필수, 기본값 없음).
# 출력: <out-dir>/e02_match.json, e02_match.txt, e02_video2_summary.csv. 이미 있으면 덮어쓰지 않고 중단한다. 결과는 결정적이다.
# 영상, 프레임, 이미지는 읽거나 쓰지 않는다. 이 도구는 영상에 실행하지 않은 상태로 커밋한다(E-02 사전 등록).
#
# 문서가 정하지 않아 "제안"으로 정한 해석:
#   (제안 1) 창 경계 포함: 시작과 끝 시각은 창 안(이상, 이하), 부동소수점 오차 허용 1e-9초.
#   (제안 2) 알람이 둘 이상의 창에 걸리면 시작이 이른 창에 귀속하고 겹침을 "미확정"으로 보고한다.
#   (제안 3) 구간 귀속은 구간 시작과 끝 시각을 포함하고 F 의 끝(null)은 영상 끝으로 본다. 구간에 속하지 않으면 "공백(앞 구간 끝~뒤 구간 시작초)".
#   (제안 4) 영상 1의 I-09 판정은 영상 1 전체를 기준 구간으로 하고, detected+tracked 는 tracked 프레임을 detected 로 보고 같은 검사를 한다.
import argparse
import collections
import csv
import json
import math
import os
import sys

from . import e01_match as E1
from . import replay
from . import validity

EPS = 1e-9

# PREREG_E-02.md 4~5항 상수(시험이 본문에서 읽어 같은지 확인한다). 값은 바꾸지 않는다.
WINDOW_START_OFFSET_S = 2.0
WINDOW_END_OFFSET_S = 1.0
POSITIVE_MIN_LENGTH_S = 3.5
NEGATIVE_MAX_LENGTH_S = 2.5
THRESHOLDS = (0.15, 0.21, 0.25)
SOURCE_SET_NAMES = ("detected", "detected+tracked")
MAIN_COMBINATION = (0.21, "detected")
EYE_CLOSURE_KIND = "눈 감기"
CLASS_POSITIVE = "양성"
CLASS_NEGATIVE = "음성"
CLASS_BORDERLINE = "경계"
CLASS_NONE = "비대상"
INTERP_ALL = "이 영상에서 일치"
INTERP_ONE_SHORT = "부분 일치, 원인 분석 필요(미확정)"
INTERP_TWO_SHORT = "불일치, 규칙 정의 재검토"
OVERLAP_NONE = "없음"
OVERLAP_UNDECIDED = "미확정"


def classify_length(length_s):
    """사건 길이(초)의 분류: 3.5초 이상 양성, 2.5초 이하 음성, 그 사이 경계."""
    if length_s >= POSITIVE_MIN_LENGTH_S - EPS:
        return CLASS_POSITIVE
    if length_s <= NEGATIVE_MAX_LENGTH_S + EPS:
        return CLASS_NEGATIVE
    return CLASS_BORDERLINE


def interpretation(matched, positives):
    """양성 일치 건수에 대한 PREREG 7항의 해석 문구(전부 일치 / 1건 모자람 / 2건 이상 모자람)."""
    for name, value in (("matched", matched), ("positives", positives)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("%s 는 0 이상의 정수여야 합니다: %r" % (name, value))
    if matched > positives:
        raise ValueError("일치 건수(%d)가 양성 건수(%d)보다 클 수 없습니다." % (matched, positives))
    short = positives - matched
    if short == 0:
        return INTERP_ALL
    if short == 1:
        return INTERP_ONE_SHORT
    return INTERP_TWO_SHORT


def load_truth(path):
    """정답 JSON 을 읽고 검증한다(길이 = 끝 - 시작, 눈 감기 사건의 분류가 분류 규칙과 같아야 함)."""
    with open(path, "r", encoding="utf-8") as handle:
        truth = json.load(handle)
    return validate_truth(truth)


def validate_truth(truth):
    for key in ("intervals", "events"):
        if key not in truth or not truth[key]:
            raise ValueError("정답에 %s 가 없습니다." % key)
    for event in truth["events"]:
        if abs(event["length_s"] - (event["end_s"] - event["start_s"])) > EPS:
            raise ValueError("사건 %s 의 길이가 끝 - 시작과 다릅니다." % event["id"])
        if event["kind"] == EYE_CLOSURE_KIND and event["class"] != classify_length(event["length_s"]):
            raise ValueError("사건 %s 의 분류가 분류 규칙과 다릅니다." % event["id"])
    return truth


def closure_windows(truth):
    """눈 감기 사건마다 창 [시작 + 2.0, 끝 + 1.0]을 만든다(시작이 이른 순서)."""
    wins = []
    for event in truth["events"]:
        if event["kind"] == EYE_CLOSURE_KIND:
            wins.append({"id": event["id"], "interval": event["interval"], "class": event["class"],
                         "start": event["start_s"], "end": event["end_s"],
                         "ws": event["start_s"] + WINDOW_START_OFFSET_S, "we": event["end_s"] + WINDOW_END_OFFSET_S})
    return sorted(wins, key=lambda w: (w["ws"], w["id"]))


def window_overlaps(wins):
    """겹치는 창 쌍의 id 목록."""
    pairs = []
    for i in range(len(wins)):
        for j in range(i + 1, len(wins)):
            if wins[j]["ws"] <= wins[i]["we"] + EPS:
                pairs.append([wins[i]["id"], wins[j]["id"]])
    return pairs


def _num(value):
    return "%g" % value


def locate_interval(time_s, truth):
    """알람 시각이 속한 구간 id, 어느 구간에도 없으면 "공백(앞 구간 끝~뒤 구간 시작초)"."""
    intervals = sorted(truth["intervals"], key=lambda item: item["start_s"])
    for item in intervals:
        end = item["end_s"] if item["end_s"] is not None else float("inf")
        if item["start_s"] - EPS <= time_s <= end + EPS:
            return item["id"]
    previous = None
    following = None
    for item in intervals:
        end = item["end_s"] if item["end_s"] is not None else float("inf")
        if end < time_s:
            previous = item
        elif item["start_s"] > time_s and following is None:
            following = item
    if previous is not None and following is not None:
        return "공백(%s~%s초)" % (_num(previous["end_s"]), _num(following["start_s"]))
    return "공백(앞)" if previous is None else "공백(뒤)"


def compute_combination(alarms, episodes, unjudgeable, truth):
    """영상 2의 조합 하나를 계산한다.

    입력: alarms - dict 목록(episode_id, alarm_time_s), episodes - {episode_id: dict}, unjudgeable - dict 목록(length_s), truth - 정답
    반환: OrderedDict
    """
    wins = closure_windows(truth)
    in_window = collections.OrderedDict((w["id"], []) for w in wins)
    extras = []
    for alarm in sorted(alarms, key=lambda row: (row["alarm_time_s"], row["episode_id"])):
        time_s = alarm["alarm_time_s"]
        for win in wins:
            if win["ws"] - EPS <= time_s <= win["we"] + EPS:
                in_window[win["id"]].append(time_s)
                break
        else:
            episode = episodes.get(alarm["episode_id"], {})
            extras.append(collections.OrderedDict([
                ("alarm_time_s", time_s), ("alarm_time", E1.format_time(time_s)), ("segment", locate_interval(time_s, truth)),
                ("episode_id", alarm["episode_id"]), ("episode_length_s", episode.get("length_s")),
                ("episode_status", episode.get("status"))]))
    events = []
    positives = matched = excess = negative_alarms = borderline_alarms = 0
    for win in wins:
        times = in_window[win["id"]]
        record = collections.OrderedDict([("id", win["id"]), ("class", win["class"]), ("window_start_s", win["ws"]),
                                          ("window_end_s", win["we"]), ("alarm_times_in_window", times),
                                          ("matched", None), ("excess_in_window", 0)])
        if win["class"] == CLASS_POSITIVE:
            positives += 1
            record["matched"] = len(times) >= 1
            record["excess_in_window"] = max(0, len(times) - 1)
            matched += 1 if record["matched"] else 0
            excess += record["excess_in_window"]
        elif win["class"] == CLASS_NEGATIVE:
            negative_alarms += len(times)
        elif win["class"] == CLASS_BORDERLINE:
            borderline_alarms += len(times)
        events.append(record)
    segments = collections.OrderedDict((item["id"], []) for item in sorted(truth["intervals"], key=lambda i: i["start_s"]))
    for extra in extras:
        segments.setdefault(extra["segment"], []).append(extra["alarm_time_s"])
    overlaps = window_overlaps(wins)
    lengths = [row["length_s"] for row in unjudgeable]
    return collections.OrderedDict([
        ("events", events), ("positives", positives), ("matched_positives", matched), ("excess_in_window", excess),
        ("negative_window_alarms", negative_alarms), ("borderline_window_alarms", borderline_alarms),
        ("extra_alarms_total", len(extras)),
        ("extra_alarms_by_segment", collections.OrderedDict((k, {"count": len(v), "times": v, "times_text": [E1.format_time(t) for t in v]})
                                                           for k, v in segments.items())),
        ("extra_alarms", extras), ("window_overlaps", overlaps),
        ("window_overlap_status", OVERLAP_UNDECIDED if overlaps else OVERLAP_NONE),
        ("unjudgeable", collections.OrderedDict([("n", len(lengths)), ("total_length_s", sum(lengths)),
                                                 ("longest_s", max(lengths) if lengths else 0.0)])),
        ("interpretation", interpretation(matched, positives))])


def summarize_video1(alarms, samples, threshold, source_set, min_reference_s):
    """영상 1의 조합 하나: 알람 개수와 시각, EAR 중앙값, I-09 판정."""
    if source_set == "detected":
        reference = list(samples)
        valid = ("detected",)
    else:
        reference = [(ts, ear, "detected" if src == "tracked" else src) for ts, ear, src in samples]
        valid = ("detected", "tracked")
    values = [ear for _ts, ear, src in samples if src in valid and ear is not None]
    check = validity.check_reference(reference, threshold, min_reference_s)
    times = sorted(alarm["alarm_time_s"] for alarm in alarms)
    return collections.OrderedDict([
        ("alarm_count", len(times)), ("alarm_times_s", times), ("alarm_times_text", [E1.format_time(t) for t in times]),
        ("ear_median", validity.median(values) if values else None), ("ear_frames", len(values)),
        ("i09_status", check["status"])])


def _read_dir(folder, threshold):
    label = E1.threshold_label(threshold)
    return (E1.read_alarms(os.path.join(folder, "alarms_thr_%s.csv" % label)),
            E1.read_episodes(os.path.join(folder, "episodes_thr_%s.csv" % label)),
            E1.read_unjudgeable(os.path.join(folder, "unjudgeable_thr_%s.csv" % label)))


def compute_all(truth, video2_dirs, video1_dirs, video1_samples, min_reference_s):
    """영상 2의 6개 조합과 영상 1의 6개 조합을 계산한다."""
    for name in SOURCE_SET_NAMES:
        if name not in video2_dirs or name not in video1_dirs:
            raise ValueError("영상 1과 영상 2의 replay 폴더가 모두 필요합니다: %s" % name)
    video2 = []
    video1 = []
    for name in SOURCE_SET_NAMES:
        for threshold in THRESHOLDS:
            alarms, episodes, unjudgeable = _read_dir(video2_dirs[name], threshold)
            item = compute_combination(alarms, episodes, unjudgeable, truth)
            item["threshold"] = threshold
            item["valid_sources"] = name
            item["is_main"] = (threshold, name) == MAIN_COMBINATION
            video2.append(item)
            alarms1, _episodes1, unjudgeable1 = _read_dir(video1_dirs[name], threshold)
            summary = summarize_video1(alarms1, video1_samples, threshold, name, min_reference_s)
            summary["threshold"] = threshold
            summary["valid_sources"] = name
            summary["unjudgeable_n"] = len(unjudgeable1)
            video1.append(summary)
    return collections.OrderedDict([("video2", video2), ("video1", video1)])


def format_text(result, truth):
    lines = ["E-02 일치 계산(PREREG_E-02.md 5~7항). 창 = 눈 감기 사건마다 [시작 + %.1f초, 끝 + %.1f초], 경계 포함." % (
        WINDOW_START_OFFSET_S, WINDOW_END_OFFSET_S), ""]
    lines.append("== 영상 2 ==")
    for item in result["video2"]:
        lines.append("임계값 %s, 유효 출처 %s%s" % (E1.threshold_label(item["threshold"]), item["valid_sources"],
                                               " (1차 조건)" if item["is_main"] else ""))
        lines.append("  양성 일치 %d/%d, 창 안 초과 %d, 음성 창 알람 %d, 경계 창 알람 %d, 창 겹침 %s" % (
            item["matched_positives"], item["positives"], item["excess_in_window"], item["negative_window_alarms"],
            item["borderline_window_alarms"], item["window_overlap_status"]))
        for event in item["events"]:
            lines.append("    %s(%s) 창 %.1f~%.1f초: 알람 %s" % (event["id"], event["class"], event["window_start_s"], event["window_end_s"],
                                                           ", ".join(E1.format_time(t) for t in event["alarm_times_in_window"]) or "없음"))
        lines.append("  추가 알람(사용자 확인 전) %d개: %s" % (item["extra_alarms_total"], "; ".join(
            "%s %d개%s" % (seg, data["count"], (" [" + ", ".join(data["times_text"]) + "]") if data["count"] else "")
            for seg, data in item["extra_alarms_by_segment"].items())))
        unj = item["unjudgeable"]
        lines.append("  판정 불가 %d개, 총 %.3f초, 최장 %.3f초" % (unj["n"], unj["total_length_s"], unj["longest_s"]))
        lines.append("  해석 구간 문구: %s" % item["interpretation"])
        lines.append("")
    lines.append("== 영상 1 ==")
    for item in result["video1"]:
        median = "-" if item["ear_median"] is None else "%.6f" % item["ear_median"]
        lines.append("임계값 %s, 유효 출처 %s: 알람 %d개 %s, EAR 중앙값 %s(프레임 %d개), I-09 판정 %s, 판정 불가 %d개" % (
            E1.threshold_label(item["threshold"]), item["valid_sources"], item["alarm_count"],
            "[" + ", ".join(item["alarm_times_text"]) + "]" if item["alarm_count"] else "", median, item["ear_frames"],
            item["i09_status"], item["unjudgeable_n"]))
    return "\n".join(lines) + "\n"


def summary_rows(result, truth):
    segment_ids = [item["id"] for item in sorted(truth["intervals"], key=lambda i: i["start_s"])]
    header = (["threshold", "valid_sources", "matched_positives", "positives", "excess_in_window", "negative_window_alarms",
               "borderline_window_alarms", "extra_alarms_total"] + ["extra_%s" % s for s in segment_ids] +
              ["extra_gap", "unjudgeable_n", "unjudgeable_total_s", "unjudgeable_longest_s", "interpretation"])
    rows = []
    for item in result["video2"]:
        by_segment = item["extra_alarms_by_segment"]
        gap = sum(data["count"] for seg, data in by_segment.items() if seg not in segment_ids)
        rows.append([E1.threshold_label(item["threshold"]), item["valid_sources"], item["matched_positives"], item["positives"],
                     item["excess_in_window"], item["negative_window_alarms"], item["borderline_window_alarms"],
                     item["extra_alarms_total"]] + [by_segment[s]["count"] for s in segment_ids] +
                    [gap, item["unjudgeable"]["n"], "%.6f" % item["unjudgeable"]["total_length_s"],
                     "%.6f" % item["unjudgeable"]["longest_s"], item["interpretation"]])
    return header, rows


def run(truth_path, video2_dirs, video1_dirs, video1_dump, min_reference_s, out_dir):
    """6개 조합(영상 2)과 6개 조합(영상 1)을 계산해 out_dir 에 쓴다. 쓴 파일 이름 list 를 돌려준다."""
    if not out_dir:
        raise ValueError("출력 폴더가 필요합니다(기본값 없음).")
    truth = load_truth(truth_path)
    samples = replay.read_ear_csv(video1_dump)
    result = compute_all(truth, video2_dirs, video1_dirs, samples, min_reference_s)
    os.makedirs(out_dir, exist_ok=True)
    payload = collections.OrderedDict([("video", truth.get("video")), ("min_reference_s", min_reference_s),
                                       ("window_start_offset_s", WINDOW_START_OFFSET_S), ("window_end_offset_s", WINDOW_END_OFFSET_S),
                                       ("result", result)])
    with open(os.path.join(out_dir, "e02_match.json"), "x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    with open(os.path.join(out_dir, "e02_match.txt"), "x", encoding="utf-8") as handle:
        handle.write(format_text(result, truth))
    header, rows = summary_rows(result, truth)
    with open(os.path.join(out_dir, "e02_video2_summary.csv"), "x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    return ["e02_match.json", "e02_match.txt", "e02_video2_summary.csv"]


def _parse_dirs(values, option):
    folders = {}
    for text in values:
        name, separator, folder = text.partition("=")
        if not separator or name not in SOURCE_SET_NAMES or not folder:
            sys.exit("%s 는 이름=폴더 형식이고 이름은 %s 중 하나여야 합니다: %r" % (option, ", ".join(SOURCE_SET_NAMES), text))
        folders[name] = folder
    return folders


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="replay.py 결과에서 E-02 일치를 계산한다.")
    parser.add_argument("--truth", required=True, help="정답 JSON(e02_truth.json)")
    parser.add_argument("--video2-replay", dest="video2", action="append", required=True, metavar="이름=폴더")
    parser.add_argument("--video1-replay", dest="video1", action="append", required=True, metavar="이름=폴더")
    parser.add_argument("--video1-dump", dest="video1_dump", required=True, help="영상 1 프레임별 EAR CSV")
    parser.add_argument("--min-reference-s", dest="min_reference_s", type=float, required=True,
                        help="I-09 기준 구간 최소 길이(초, 기본값 없음)")
    parser.add_argument("--out-dir", dest="out_dir", required=True, help="출력 폴더(필수, 기본값 없음)")
    return parser.parse_args(argv)


def main(argv=None):
    arguments = parse_args(argv)
    for name in run(arguments.truth, _parse_dirs(arguments.video2, "--video2-replay"),
                    _parse_dirs(arguments.video1, "--video1-replay"), arguments.video1_dump,
                    arguments.min_reference_s, arguments.out_dir):
        print(name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
