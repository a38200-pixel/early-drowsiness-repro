# summarize_batch.py
# 역할: extract_blinks_batch.py 가 남긴 batch_manifest.csv 를 읽어 일괄 추출의 진행 상황을 콘솔에 요약한다.
#       읽기 전용이다. 어떤 파일도 쓰지 않고 영상도 열지 않는다.
# 사용법: BLINK_OUT_DIR 를 설정한 창에서 저장소 루트에서 실행한다.
#           python src/repro/summarize_batch.py [--recent-minutes N]        (N 의 기본값은 30)
# 출력: (1) 상태별 영상 수와 전체 처리 목록 대비 진행률 (2) 처리된 프레임 합과 비율
#       (3) 경과 시간, 평균 합산 처리량(fps), 최근 N분 합산 처리량, 남은 시간 추정
#       (4) 회전별 face_frames/frame_count 비율의 평균과 최소 (5) 비율이 0.5 미만인 영상 (6) no_output 과 failed 영상
# 입력: <BLINK_OUT_DIR>/batch_manifest.csv (# 로 시작하는 주석 줄은 건너뜀), configs/video_inventory.csv(frame_count),
#       configs/video_orientation_overrides.csv, configs/video_orientation_survey.csv(resolved_orientation), configs/video_manual_flags.csv
# 용어: "전체 처리 목록" = manual_flags 영상을 뺀(배치 도구의 기본 순서와 같음) 영상 중 회전이 정해진 영상.
#       "처리된 프레임" = 종료 코드 0 으로 끝난 영상(ok, no_output)의 frame_count. 출력 파일만 있고 매니페스트에 없는 영상은 세지 않는다.
"""일괄 추출 진행 요약 도구(Python 3.6 호환, 읽기 전용)."""
import argparse
import csv
import os
import sys
from datetime import datetime, timedelta

from extract_blinks_batch import (FLAGS_PATH, INVENTORY_PATH, MANIFEST_COLUMNS, MANIFEST_NAME, OVERRIDES_PATH,
                                  SURVEY_PATH, TIME_FORMAT, entry_key, is_flagged, load_flag_keys, load_overrides,
                                  load_survey_orientations, read_inventory, rotation_label)

LOW_RATIO = 0.5
ROTATIONS = ("none", "cw90", "ccw90", "180")


def parse_arguments():
    """명령줄 옵션을 해석한다.

    반환: argparse.Namespace (recent_minutes: 최근 처리량을 보는 구간의 길이(분))
    """
    parser = argparse.ArgumentParser(description="일괄 추출 진행 요약(읽기 전용)")
    parser.add_argument("--recent-minutes", type=int, default=30, dest="recent_minutes",
                        help="최근 처리량을 계산할 구간(분, 기본 30)")
    arguments = parser.parse_args()
    if arguments.recent_minutes < 1:
        parser.error("--recent-minutes 는 1 이상이어야 합니다.")
    return arguments


def require_manifest_path():
    """BLINK_OUT_DIR 아래의 매니페스트 경로를 돌려준다. 환경변수가 없거나 파일이 없으면 안내 후 중단한다.

    반환: batch_manifest.csv 경로
    """
    out_dir = os.environ.get("BLINK_OUT_DIR")
    if not out_dir:
        sys.exit("환경변수 BLINK_OUT_DIR 를 설정하세요(일괄 추출의 출력 루트).")
    path = os.path.join(out_dir, MANIFEST_NAME)
    if not os.path.isfile(path):
        sys.exit("batch_manifest.csv 가 없습니다. 아직 일괄 추출을 실행하지 않았거나 BLINK_OUT_DIR 가 다릅니다.")
    return path


def parse_run(row):
    """매니페스트 한 행을 형 변환한 dict 로 바꾼다.

    입력: row - csv.DictReader 의 행
    반환: dict(key, status, return_code, elapsed_s, started, ended, rotate, face_frames). 값을 읽지 못하면 ValueError
    """
    key = (int(row["fold"]), int(row["part"]), int(row["subject_id"]), int(row["class_label"]), row["variant"])
    face_frames = int(row["face_frames"]) if row["face_frames"].strip() != "" else None
    return {"key": key, "status": row["status"], "return_code": int(row["return_code"]),
            "elapsed_s": float(row["elapsed_s"]), "started": datetime.strptime(row["started"], TIME_FORMAT),
            "ended": datetime.strptime(row["ended"], TIME_FORMAT), "rotate": row["rotate"] or "?",
            "face_frames": face_frames}


def read_manifest(path):
    """매니페스트를 읽어 실행 기록 list 로 돌려준다(# 주석 줄 제외). 열 이름이 다르면 중단한다.

    입력: path - batch_manifest.csv 경로
    반환: parse_run 결과 list(파일 순서)
    """
    data_lines = []
    with open(path, "r", newline="", encoding="utf-8") as file:
        for line in file:
            if not line.startswith("#"):
                data_lines.append(line)
    reader = csv.DictReader(data_lines)
    if reader.fieldnames != MANIFEST_COLUMNS:
        sys.exit("batch_manifest.csv 의 열 이름이 현재 도구와 다릅니다. extract_blinks_batch.py 와 같은 버전의 매니페스트만 요약합니다.")
    runs = []
    for number, row in enumerate(reader, 1):
        try:
            runs.append(parse_run(row))
        except (ValueError, KeyError):
            sys.exit("batch_manifest.csv 의 %d번째 데이터 행을 읽을 수 없습니다." % number)
    return runs


def latest_by_video(runs):
    """영상마다 시작 시각이 가장 늦은 기록(마지막 시도)만 남긴다.

    입력: runs - read_manifest 결과
    반환: {키: 실행 기록}
    """
    latest = {}
    for run in sorted(runs, key=lambda r: r["started"]):
        latest[run["key"]] = run
    return latest


def build_targets(entries, flag_keys, survey_map, overrides):
    """전체 처리 목록을 만든다: manual_flags 영상과 회전이 정해지지 않은(review) 영상을 뺀 영상.

    입력: entries - inventory 항목, flag_keys - manual_flags 키, survey_map/overrides - 회전 정보
    반환: (목록, 제외한 manual_flags 영상 수, 제외한 review 영상 수)
    """
    targets, flagged, review = [], 0, 0
    for entry in entries:
        if is_flagged(entry, flag_keys):
            flagged += 1
        elif rotation_label(entry, survey_map, overrides) == "review":
            review += 1
        else:
            targets.append(entry)
    return targets, flagged, review


def percent(part, whole):
    """비율을 '12.3%' 형태 문자열로 만든다. 분모가 0이면 '-'.

    입력: part - 분자, whole - 분모
    반환: 문자열
    """
    return "-" if whole == 0 else "%.1f%%" % (100.0 * part / whole)


def format_duration(seconds):
    """초를 'H시간 M분 S초' 문자열로 바꾼다.

    입력: seconds - 초(실수)
    반환: 문자열
    """
    total = int(round(seconds))
    return "%d시간 %d분 %d초" % (total // 3600, (total % 3600) // 60, total % 60)


def print_status_summary(latest, targets, flagged, review):
    """(1) 상태별 영상 수와 전체 처리 목록 대비 진행률을 출력한다.

    입력: latest - 영상별 마지막 기록, targets - 전체 처리 목록, flagged/review - 목록에서 뺀 영상 수
    """
    target_keys = set(entry_key(entry) for entry in targets)
    counts = {"ok": 0, "no_output": 0, "failed": 0}
    for run in latest.values():
        counts[run["status"]] = counts.get(run["status"], 0) + 1
    in_target = [run for key, run in latest.items() if key in target_keys]
    done = sum(1 for run in in_target if run["status"] == "ok")
    print("== (1) 상태별 영상 수 (영상마다 마지막 기록 기준) ==")
    print("기록된 영상 %d개: ok %d, no_output %d, failed %d" % (len(latest), counts["ok"], counts["no_output"], counts["failed"]))
    print("전체 처리 목록 %d개 (manual_flags 제외 %d개, 회전 미정 review 제외 %d개)" % (len(targets), flagged, review))
    print("진행률: 완료(ok) %d/%d = %s | 기록된 것(ok+no_output+failed) %d/%d = %s | 미처리 %d개" % (
        done, len(targets), percent(done, len(targets)), len(in_target), len(targets),
        percent(len(in_target), len(targets)), len(targets) - len(in_target)))
    print("전체 처리 목록 밖의 기록: %d개 (예: manual_flags 영상)" % (len(latest) - len(in_target)))


def print_frame_summary(latest, targets, frame_of):
    """(2) 처리된 프레임 합과 비율, 얼굴이 검출된 프레임 합과 비율을 출력한다.

    입력: latest - 영상별 마지막 기록, targets - 전체 처리 목록, frame_of - {키: frame_count}
    """
    target_keys = set(entry_key(entry) for entry in targets)
    total_frames = sum(entry["frame_count"] for entry in targets)
    processed, face_sum, face_base = 0, 0, 0
    for key, run in latest.items():
        if key not in target_keys:
            continue
        if run["return_code"] == 0:
            processed += frame_of[key]
        if run["face_frames"] is not None:
            face_sum += run["face_frames"]
            face_base += frame_of[key]
    print("== (2) 처리된 프레임 (전체 처리 목록 기준) ==")
    print("종료 코드 0(ok, no_output) 영상의 frame_count 합 %d / 목록 전체 %d = %s" % (
        processed, total_frames, percent(processed, total_frames)))
    print("face_frames 합 %d / 그 영상들의 frame_count 합 %d = %s (face_frames 가 기록된 영상 한정)" % (
        face_sum, face_base, percent(face_sum, face_base)))


def window_frames(runs, frame_of, window_start, window_end):
    """구간 [window_start, window_end] 에서 처리된 프레임 수를 센다. 긴 영상은 실행 구간과 겹친 비율만큼만 센다.

    입력: runs - 실행 기록 전체, frame_of - {키: frame_count}, window_start/window_end - datetime
    반환: 실수. 종료 코드가 0 이 아닌 실행은 세지 않는다
    """
    total = 0.0
    for run in runs:
        duration = (run["ended"] - run["started"]).total_seconds()
        if run["return_code"] != 0 or duration <= 0 or run["key"] not in frame_of:
            continue
        overlap = (min(run["ended"], window_end) - max(run["started"], window_start)).total_seconds()
        if overlap > 0:
            total += frame_of[run["key"]] * overlap / duration
    return total


def print_time_summary(runs, latest, targets, frame_of, recent_minutes):
    """(3) 경과 시간, 평균·최근 합산 처리량(fps), 남은 시간 추정을 출력한다.

    입력: runs - 실행 기록 전체, latest - 영상별 마지막 기록, targets - 전체 처리 목록,
          frame_of - {키: frame_count}, recent_minutes - 최근 구간(분)
    """
    first_start = min(run["started"] for run in runs)
    last_end = max(run["ended"] for run in runs)
    elapsed = (last_end - first_start).total_seconds()
    done_frames = sum(frame_of[key] for key, run in latest.items() if run["return_code"] == 0 and key in frame_of)
    average = done_frames / elapsed if elapsed > 0 else 0.0
    window = min(recent_minutes * 60.0, elapsed)
    recent = window_frames(runs, frame_of, last_end - timedelta(seconds=window), last_end) / window if window > 0 else 0.0
    remaining = sum(entry["frame_count"] for entry in targets if entry_key(entry) not in latest)
    print("== (3) 시간과 처리량 ==")
    print("첫 시작 %s ~ 마지막 종료 %s, 경과 %s (중간에 멈춘 시간 포함)" % (
        first_start.strftime(TIME_FORMAT), last_end.strftime(TIME_FORMAT), format_duration(elapsed)))
    print("마지막 기록 후 지금까지 %s" % format_duration((datetime.now() - last_end).total_seconds()))
    print("평균 합산 처리량 %.1f fps (완료 영상 frame_count 합 %d / 경과)" % (average, done_frames))
    print("최근 %.0f분 합산 처리량 %.1f fps (각 영상은 실행 구간과 겹친 비율만큼 계산)" % (window / 60.0, recent))
    print("미처리 영상의 frame_count 합 %d" % remaining)
    print("남은 시간 추정: 최근 처리량 기준 %s | 평균 처리량 기준 %s" % (
        format_duration(remaining / recent) if recent > 0 else "추정 불가",
        format_duration(remaining / average) if average > 0 else "추정 불가"))


def find_rotation_mismatches(latest, entry_by_key, survey_map, overrides):
    """매니페스트에 기록된 회전과 현재 override/조사 분류가 다른 영상을 찾는다.

    입력: latest - 영상별 마지막 기록, entry_by_key - {키: inventory 항목}, survey_map/overrides - 회전 정보
    반환: (rel_path, 기록된 회전, 현재 분류) 튜플 list
    """
    mismatches = []
    for key, run in latest.items():
        entry = entry_by_key.get(key)
        if entry is None:
            continue
        label = rotation_label(entry, survey_map, overrides)
        if label != run["rotate"]:
            mismatches.append((entry["rel_path"], run["rotate"], label))
    return mismatches


def print_rotation_stats(latest, frame_of, mismatches):
    """(4) 회전별 face_frames/frame_count 비율의 평균과 최소를 출력한다.

    입력: latest - 영상별 마지막 기록, frame_of - {키: frame_count}, mismatches - 분류가 달라진 영상 목록
    """
    groups = dict((name, []) for name in ROTATIONS)
    for key, run in latest.items():
        if run["face_frames"] is not None and frame_of.get(key):
            groups.setdefault(run["rotate"], []).append(run["face_frames"] / float(frame_of[key]))
    print("== (4) 회전별 face_frames/frame_count 비율 (기록된 회전 기준) ==")
    for name in list(ROTATIONS) + [n for n in groups if n not in ROTATIONS]:
        ratios = groups.get(name, [])
        if ratios:
            print("  %-6s 영상 %d개, 평균 %.4f, 최소 %.4f" % (name, len(ratios), sum(ratios) / len(ratios), min(ratios)))
        else:
            print("  %-6s 기록 없음" % name)
    print("  매니페스트의 회전과 현재 override/조사 분류가 다른 영상: %d개" % len(mismatches))
    for rel_path, recorded, current in mismatches:
        print("    %s 기록 %s, 현재 %s" % (rel_path, recorded, current))


def print_low_ratio(latest, entry_by_key, frame_of):
    """(5) face_frames/frame_count 비율이 0.5 미만인 영상 목록을 출력한다.

    입력: latest - 영상별 마지막 기록, entry_by_key - {키: inventory 항목}, frame_of - {키: frame_count}
    """
    print("== (5) face_frames/frame_count < %.1f 인 영상 ==" % LOW_RATIO)
    found = 0
    for key, run in sorted(latest.items(), key=lambda item: item[1]["started"]):
        if run["face_frames"] is None or not frame_of.get(key):
            continue
        ratio = run["face_frames"] / float(frame_of[key])
        if ratio < LOW_RATIO:
            found += 1
            print("  %s 회전 %s, face_frames %d / frame_count %d = %.4f (%s)" % (
                entry_by_key[key]["rel_path"], run["rotate"], run["face_frames"], frame_of[key], ratio, run["status"]))
    if found == 0:
        print("  없음")


def print_problem_videos(latest, entry_by_key):
    """(6) 마지막 기록이 no_output 이거나 failed 인 영상 목록을 출력한다.

    입력: latest - 영상별 마지막 기록, entry_by_key - {키: inventory 항목}
    """
    print("== (6) no_output / failed 영상 (마지막 기록 기준) ==")
    found = 0
    for key, run in sorted(latest.items(), key=lambda item: item[1]["started"]):
        if run["status"] in ("no_output", "failed"):
            found += 1
            rel_path = entry_by_key[key]["rel_path"] if key in entry_by_key else str(key)
            print("  %s %s 종료코드 %d, 회전 %s, 소요 %.1f초" % (rel_path, run["status"], run["return_code"],
                                                              run["rotate"], run["elapsed_s"]))
    if found == 0:
        print("  없음")


def main():
    """전체 흐름: 입력 읽기, 전체 처리 목록 만들기, (1)~(6) 요약 출력. 어떤 파일도 쓰지 않는다."""
    arguments = parse_arguments()
    runs = read_manifest(require_manifest_path())
    if not runs:
        print("매니페스트에 기록된 영상이 없습니다.")
        return
    entries = read_inventory(INVENTORY_PATH)
    survey_map = load_survey_orientations(SURVEY_PATH)
    overrides = load_overrides(OVERRIDES_PATH)
    targets, flagged, review = build_targets(entries, load_flag_keys(FLAGS_PATH), survey_map, overrides)
    entry_by_key = dict((entry_key(entry), entry) for entry in entries)
    frame_of = dict((key, entry["frame_count"]) for key, entry in entry_by_key.items())
    latest = latest_by_video(runs)
    print_status_summary(latest, targets, flagged, review)
    print_frame_summary(latest, targets, frame_of)
    print_time_summary(runs, latest, targets, frame_of, arguments.recent_minutes)
    print_rotation_stats(latest, frame_of, find_rotation_mismatches(latest, entry_by_key, survey_map, overrides))
    print_low_ratio(latest, entry_by_key, frame_of)
    print_problem_videos(latest, entry_by_key)


if __name__ == "__main__":
    main()
