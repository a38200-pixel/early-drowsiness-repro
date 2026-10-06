# 역할: 오프라인 재생(C2). 프레임별 EAR CSV 를 읽어 임계값(기본 0.15 / 0.21 / 0.25)마다 상태기계를 돌리고,
#       임계값별 알람 목록, 판정 불가 구간, 감김 구간 목록, 요약을 출력 폴더에 쓴다(I-10, RULE_SPEC 8절).
#       이 도구는 실험이 아니다. 실험 E-01 은 사전 등록 뒤에 이 도구로 돌린다.
# 입력: --input CSV(열 timestamp_s, ear_left, ear_right, source). source 는 detected / tracked / none.
#       ear_left, ear_right 가 둘 다 있으면 평균을 EAR 로 쓰고, 하나라도 비어 있거나 source 가 none 이면 EAR 없음으로 본다.
# 출력: --out-dir(필수, 저장소 안의 기본값 없음) 아래에
#       alarms_thr_<임계값>.csv, episodes_thr_<임계값>.csv, unjudgeable_thr_<임계값>.csv(판정 불가 구간, 비어 있어도 쓴다),
#       summary.json, summary.txt. 이미 있는 파일은 덮어쓰지 않고 중단한다.
# 사용: (src 폴더에서 또는 PYTHONPATH=src) python -m improve.rules.replay --input <CSV> --out-dir <출력 폴더>
# 영상, 프레임, 이미지는 읽거나 쓰지 않는다. 파이프라인과 탐지기를 조정하지 않는다.
import argparse
import csv
import json
import math
import os
import sys

from . import episode_machine as EM
from .validity import median

DEFAULT_THRESHOLDS = (0.15, 0.21, 0.25)
CSV_COLUMNS = ("timestamp_s", "ear_left", "ear_right", "source")
# 감김 구간 길이 분포의 구간 경계(초). 경계 선택은 제안이다(사전 등록에서 확정).
LENGTH_BINS = ((0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 5.0), (5.0, None))


def _to_float(text, row_number, column):
    """CSV 칸을 float 로 바꾼다. 빈 칸은 None, 유한하지 않은 값은 오류이다."""
    text = (text or "").strip()
    if text == "":
        return None
    try:
        value = float(text)
    except ValueError:
        raise ValueError("CSV %d번째 줄의 %s 칸이 숫자가 아닙니다." % (row_number, column))
    if not math.isfinite(value):
        raise ValueError("CSV %d번째 줄의 %s 칸이 유한한 수가 아닙니다." % (row_number, column))
    return value


def read_ear_csv(path):
    """프레임별 EAR CSV 를 읽어 (timestamp_s, ear_mean 또는 None, source) 목록으로 만든다.

    입력: path - CSV 경로
    반환: list of (float, float 또는 None, str)
    """
    samples = []
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        names = reader.fieldnames or []
        missing = [column for column in CSV_COLUMNS if column not in names]
        if missing:
            raise ValueError("CSV 에 열이 없습니다: %s" % ", ".join(missing))
        for row_number, row in enumerate(reader, start=2):
            timestamp = _to_float(row["timestamp_s"], row_number, "timestamp_s")
            if timestamp is None:
                raise ValueError("CSV %d번째 줄의 timestamp_s 칸이 비어 있습니다." % row_number)
            source = (row["source"] or "").strip()
            if source not in EM.SOURCES:
                raise ValueError("CSV %d번째 줄의 source 가 %s 중 하나가 아닙니다: %r" % (row_number, ", ".join(EM.SOURCES), source))
            left = _to_float(row["ear_left"], row_number, "ear_left")
            right = _to_float(row["ear_right"], row_number, "ear_right")
            ear = None if (source == "none" or left is None or right is None) else (left + right) / 2.0
            samples.append((timestamp, ear, source))
    return samples


def threshold_label(threshold):
    """파일 이름에 쓰는 임계값 표기(소수 둘째 자리)."""
    return "%.2f" % threshold


def replay_thresholds(samples, thresholds=DEFAULT_THRESHOLDS, valid_sources=EM.DEFAULT_VALID_SOURCES,
                      reliability=EM.DEFAULT_RELIABILITY):
    """같은 샘플에 임계값만 바꿔 상태기계를 다시 돌린다. 3.0초와 0.3초 기준은 바꾸지 않는다.

    입력: samples - read_ear_csv 의 반환. thresholds - 임계값 목록. valid_sources - 유효 출처. reliability - 알람 신뢰 표시
    반환: {임계값: Result} (입력 순서 유지를 위해 list of (임계값, Result))
    """
    return [(threshold, EM.run_samples(samples, threshold=threshold, valid_sources=valid_sources,
                                       reliability=reliability)) for threshold in thresholds]


def _length_distribution(lengths):
    """감김 구간 길이의 개수, 최소, 중앙값, 최대와 구간별 개수를 만든다."""
    counts = []
    for low, high in LENGTH_BINS:
        label = "%.1f~%s" % (low, "%.1f" % high if high is not None else "")
        counts.append({"bin": label, "count": sum(1 for value in lengths if value >= low and (high is None or value < high))})
    stats = {"n": len(lengths), "min": min(lengths) if lengths else None, "median": median(lengths) if lengths else None,
             "max": max(lengths) if lengths else None}
    return {"stats": stats, "bins": counts}


def summarize(result):
    """Result 하나를 요약한다(알람 수, 감김 구간 길이 분포, 판정 불가 구간)."""
    kept = (EM.STATUS_RELEASED, EM.STATUS_KEPT_AT_END)
    lengths = [episode.length_s for episode in result.episodes if episode.status in kept]
    status_counts = {}
    for episode in result.episodes:
        status_counts[episode.status] = status_counts.get(episode.status, 0) + 1
    return {"threshold": result.threshold, "params": result.params, "n_alarms": len(result.alarms),
            "episode_status_counts": status_counts, "episode_length_distribution": _length_distribution(lengths),
            "unjudgeable": {"n": len(result.unjudgeable),
                            "total_length_s": sum(item.length_s for item in result.unjudgeable),
                            "intervals": [[item.start_s, item.end_s, item.length_s] for item in result.unjudgeable]}}


def _cell(value):
    """CSV 칸 값: None 은 빈 칸, float 은 소수 6자리."""
    if value is None:
        return ""
    if isinstance(value, float):
        return "%.6f" % value
    return value


def _write_csv(path, header, rows):
    """CSV 를 새로 쓴다(이미 있으면 'x' 모드로 중단)."""
    with open(path, "x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row in rows:
            writer.writerow([_cell(value) for value in row])


def write_outputs(out_dir, replays):
    """임계값별 결과 파일과 요약을 출력 폴더에 쓴다.

    입력: out_dir - 출력 폴더(호출자가 지정, 없으면 만듦). replays - replay_thresholds 의 반환
    반환: 쓴 파일 이름 list
    """
    if not out_dir:
        raise ValueError("출력 폴더가 필요합니다(기본값 없음).")
    os.makedirs(out_dir, exist_ok=True)
    written = []
    summaries = []
    for threshold, result in replays:
        label = threshold_label(threshold)
        name = "alarms_thr_%s.csv" % label
        _write_csv(os.path.join(out_dir, name), Alarm_HEADER, [tuple(alarm) for alarm in result.alarms])
        written.append(name)
        name = "episodes_thr_%s.csv" % label
        _write_csv(os.path.join(out_dir, name), EM.Episode._fields, [tuple(item) for item in result.episodes])
        written.append(name)
        name = "unjudgeable_thr_%s.csv" % label
        _write_csv(os.path.join(out_dir, name), EM.Unjudgeable._fields, [tuple(item) for item in result.unjudgeable])
        written.append(name)
        summaries.append(summarize(result))
    with open(os.path.join(out_dir, "summary.json"), "x", encoding="utf-8") as handle:
        json.dump(summaries, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    written.append("summary.json")
    with open(os.path.join(out_dir, "summary.txt"), "x", encoding="utf-8") as handle:
        for item in summaries:
            handle.write(format_summary(item))
    written.append("summary.txt")
    return written


Alarm_HEADER = EM.Alarm._fields


def format_summary(item):
    """요약 하나를 사람이 읽는 텍스트로 만든다. 판정 불가 구간은 알람 수 옆에 반드시 쓴다(RULE_SPEC 7.1절)."""
    stats = item["episode_length_distribution"]["stats"]
    unj = item["unjudgeable"]
    lines = ["임계값 %s" % threshold_label(item["threshold"]),
             "  알람 수: %d" % item["n_alarms"],
             "  판정 불가 구간: %d개, 합계 %.6f초" % (unj["n"], unj["total_length_s"]),
             "  감김 구간(해제됨 또는 영상 끝에서 유지): %d개, 최소 %s, 중앙값 %s, 최대 %s" % (
                 stats["n"], _cell(stats["min"]), _cell(stats["median"]), _cell(stats["max"])),
             "  구간 상태별 개수: %s" % json.dumps(item["episode_status_counts"], ensure_ascii=False, sort_keys=True)]
    for entry in item["episode_length_distribution"]["bins"]:
        lines.append("    길이 %s초: %d개" % (entry["bin"], entry["count"]))
    lines.append("")
    return "\n".join(lines) + "\n"


def parse_args(argv=None):
    """명령줄 인자를 읽는다. --input 과 --out-dir 은 필수이다(저장소 안의 기본값을 두지 않는다)."""
    parser = argparse.ArgumentParser(description="프레임별 EAR CSV 를 임계값별로 오프라인 재생한다.")
    parser.add_argument("--input", required=True, help="CSV(열: timestamp_s, ear_left, ear_right, source)")
    parser.add_argument("--out-dir", dest="out_dir", required=True, help="출력 폴더(필수, 기본값 없음)")
    parser.add_argument("--thresholds", nargs="+", type=float, default=list(DEFAULT_THRESHOLDS), help="닫힘 임계값 목록")
    parser.add_argument("--valid-sources", dest="valid_sources", nargs="+", choices=["detected", "tracked"],
                        default=list(EM.DEFAULT_VALID_SOURCES), help="유효 EAR 로 보는 출처(기본 detected, 제안)")
    parser.add_argument("--reliability", default=EM.DEFAULT_RELIABILITY, help="알람 신뢰 표시(제안)")
    return parser.parse_args(argv)


def main(argv=None):
    """재생을 실행하고 쓴 파일 이름을 출력한다."""
    arguments = parse_args(argv)
    samples = read_ear_csv(arguments.input)
    replays = replay_thresholds(samples, arguments.thresholds, tuple(arguments.valid_sources), arguments.reliability)
    for name in write_outputs(arguments.out_dir, replays):
        print(name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
