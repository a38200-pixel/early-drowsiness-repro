# survey_orientation.py
# 역할: 영상별로 dlib 정면 얼굴 검출이 어느 회전 방향(0도, 시계 90도, 180도, 반시계 90도)에서 되는지 조사한다.
#       영상에서 5%~95% 구간을 고르게 나눈 8개 프레임 위치만 읽고(전체 디코딩 없음), 프레임이나 이미지는 어디에도 저장하지 않는다.
#       --resolve-only 는 영상을 읽지 않고 기존 CSV 의 카운트만으로 규칙 v2 의 resolved_orientation 열을 추가한다.
# 전처리 순서(blink_video.py 와 같음, 회전만 맨 앞에 추가): 원본 프레임 -> 회전 -> imutils.resize(width=450)
#       -> 그레이스케일 -> 감마 1.5 -> detector(gray, 0). 회전은 resize 전에 적용한다.
# 규칙 v1(best_orientation): 얼굴 수 최댓값 방향. 전부 0이면 none, 1위가 2위의 2배 미만이면 ambiguous.
# 규칙 v2(resolved_orientation, 탐색적): faces_0 >= 6 이면 0. 아니면 네 방향 중 최댓값이 5 이상이고 2위의 2배 이상이면
#       그 방향. 그 외는 review(육안 확인). 사전 등록: docs/REPRODUCTION_LOG.md 의 "방향 판정 규칙 v2 사전 등록".
# 사용법: conda 환경 ghoddoosian_tf1 을 활성화하고 저장소 루트에서 실행한다.
#           python src/repro/survey_orientation.py --only Fold2_part1/17/0.mp4 --only Fold4_part2/45/10.mp4
#           python src/repro/survey_orientation.py --subset zero     (inventory 의 faces_orig==0 영상만)
#           python src/repro/survey_orientation.py                   (전체)   옵션: --limit K
#           python src/repro/survey_orientation.py --resolve-only    (영상을 읽지 않음, RLDD_DATA_DIR 불필요)
# 환경변수: RLDD_DATA_DIR (조사 실행 시 영상 루트). dlib .dat 는 쓰지 않고 dlib.get_frontal_face_detector() 만 쓴다.
# 입력: configs/video_inventory.csv, configs/video_manual_flags.csv (--resolve-only 는 video_orientation_survey.csv)
# 출력: configs/video_orientation_survey.csv (실행할 때마다 덮어쓴다. 부분 실행의 결과도 같은 파일에 쓰인다)
"""영상 회전 방향별 얼굴 검출 조사 스크립트(Python 3.6 호환, 읽기 전용)."""
import argparse
import csv
import os
import sys
import time

import cv2
import dlib
import imutils

from inventory_videos import build_gamma_table, load_manual_flags
from orientation_common import SURVEY_ORIENTATIONS, rotate_frame

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INVENTORY_PATH = os.path.join(REPO, "configs", "video_inventory.csv")
FLAGS_PATH = os.path.join(REPO, "configs", "video_manual_flags.csv")
OUTPUT_PATH = os.path.join(REPO, "configs", "video_orientation_survey.csv")
SAMPLE_COUNT = 8
FIRST_PERCENT = 5.0
LAST_PERCENT = 95.0
DETECT_WIDTH = 450
RESOLVE_ZERO_MIN = 6      # 규칙 v2: faces_0 가 이 값 이상이면 0
RESOLVE_TOP_MIN = 5       # 규칙 v2: 최댓값이 이 값 이상이어야 방향을 정한다
RESOLVE_RATIO = 2         # 규칙 v2: 최댓값이 2위의 이 배수 이상이어야 한다
PREDICTIONS = [("Fold1_part2/10/0.MOV", "cw90"), ("Fold4_part2/45/10.mp4", "180"), ("Fold2_part1/17/0.mp4", "0"),
               ("Fold1_part1/06/0.mp4", "0"), ("Fold1_part1/06/5.mp4", "0"), ("Fold1_part1/06/10.mp4", "0")]
COLUMNS = ["fold", "part", "subject_id", "class_label", "variant", "rel_path", "width", "height",
           "frames_sampled", "faces_0", "faces_cw90", "faces_180", "faces_ccw90", "best_orientation", "manual_flag",
           "resolved_orientation"]


def parse_arguments():
    """명령줄 옵션을 해석한다.

    반환: argparse.Namespace (only, subset, limit, resolve_only). --resolve-only 는 다른 옵션과 함께 쓸 수 없다
    """
    parser = argparse.ArgumentParser(description="영상 회전 방향별 얼굴 검출 조사")
    parser.add_argument("--only", action="append", default=None, help="이 rel_path 영상만(여러 번 지정 가능)")
    parser.add_argument("--subset", choices=["zero"], default=None, help="zero: inventory 의 faces_orig==0 영상만")
    parser.add_argument("--limit", type=int, default=None, help="앞 K개만")
    parser.add_argument("--resolve-only", action="store_true", dest="resolve_only",
                        help="영상을 읽지 않고 기존 CSV 의 카운트만으로 resolved_orientation 열을 추가")
    arguments = parser.parse_args()
    if arguments.limit is not None and arguments.limit < 0:
        parser.error("--limit 는 0 이상이어야 합니다.")
    if arguments.resolve_only and (arguments.only or arguments.subset or arguments.limit is not None):
        parser.error("--resolve-only 는 --only, --subset, --limit 와 함께 쓸 수 없습니다.")
    return arguments


def load_inventory_rows(inventory_path):
    """영상 목록 CSV 의 행을 dict 의 list 로 읽는다.

    입력: inventory_path - video_inventory.csv 경로
    반환: 행 dict 의 list (CSV 순서 그대로)
    """
    with open(inventory_path, "r", newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def select_rows(rows, arguments):
    """옵션에 맞는 행을 고른다. --only 로 지정한 영상이 목록에 없으면 중단한다.

    입력: rows - inventory 행, arguments - 명령줄 옵션
    반환: 선택된 행 list (--limit 적용 후)
    """
    chosen = rows
    if arguments.only is not None:
        known = set(row["rel_path"] for row in rows)
        for rel_path in arguments.only:
            if rel_path not in known:
                sys.exit("inventory 에 없는 영상입니다: %s" % rel_path)
        chosen = [row for row in rows if row["rel_path"] in arguments.only]
    if arguments.subset == "zero":
        chosen = [row for row in chosen if row["faces_orig"] == "0"]
    if arguments.limit is not None:
        chosen = chosen[:arguments.limit]
    return chosen


def sample_positions(frame_count):
    """5%~95% 구간을 고르게 나눈 8개 프레임 번호를 계산한다.

    입력: frame_count - 전체 프레임 수
    반환: 프레임 번호 list (frame_count 가 1 미만이면 빈 list)
    """
    if frame_count < 1:
        return []
    step = (LAST_PERCENT - FIRST_PERCENT) / (SAMPLE_COUNT - 1)
    positions = []
    for index in range(SAMPLE_COUNT):
        percent = FIRST_PERCENT + step * index
        positions.append(min(int(frame_count * percent / 100.0), frame_count - 1))
    return positions


def read_sample_frames(capture, positions):
    """지정한 위치의 프레임만 읽는다(전체 디코딩 아님). 읽지 못한 위치는 건너뛴다.

    입력: capture - 열린 VideoCapture, positions - 프레임 번호 list
    반환: 읽힌 BGR 프레임 list
    """
    frames = []
    for position in positions:
        capture.set(cv2.CAP_PROP_POS_FRAMES, position)
        grabbed, frame = capture.read()
        if grabbed and frame is not None:
            frames.append(frame)
    return frames


def count_faces(frame, detector, gamma_table):
    """blink_video.py 와 같은 전처리(resize width=450, 그레이스케일, 감마 1.5)를 거쳐 얼굴 수를 센다.

    입력: frame - 이미 회전된 BGR 프레임, detector - dlib 정면 얼굴 검출기, gamma_table - 감마 변환표
    반환: 검출된 얼굴 수(정수)
    """
    resized = imutils.resize(frame, width=DETECT_WIDTH)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gray = cv2.LUT(gray, gamma_table)
    return len(detector(gray, 0))


def count_by_orientation(frames, detector, gamma_table):
    """읽힌 프레임 전체에서 방향별 얼굴 수의 합을 구한다(회전 정의는 orientation_common 를 쓴다).

    입력: frames - 읽힌 프레임 list, detector/gamma_table - 공용 객체
    반환: {"0": 수, "cw90": 수, "180": 수, "ccw90": 수}
    """
    counts = {}
    for name in SURVEY_ORIENTATIONS:
        total = 0
        for frame in frames:
            total += count_faces(rotate_frame(frame, name), detector, gamma_table)
        counts[name] = total
    return counts


def choose_best(counts):
    """규칙 v1: 얼굴 수가 가장 많은 방향을 고른다. 전부 0이면 none, 1위가 2위의 2배 미만이면 ambiguous.

    입력: counts - count_by_orientation 결과
    반환: 방향 이름 문자열
    """
    ranked = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    top_name, top_count = ranked[0]
    second_count = ranked[1][1]
    if top_count == 0:
        return "none"
    if top_count < 2 * second_count:
        return "ambiguous"
    return top_name


def resolve_orientation(counts):
    """규칙 v2(탐색적): faces_0 가 6 이상이면 0, 아니면 최댓값이 5 이상이고 2위의 2배 이상인 방향, 그 외는 review.

    입력: counts - {"0": 수, "cw90": 수, "180": 수, "ccw90": 수}
    반환: "0", "cw90", "ccw90", "180", "review" 중 하나
    """
    if counts["0"] >= RESOLVE_ZERO_MIN:
        return "0"
    ranked = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    top_name, top_count = ranked[0]
    second_count = ranked[1][1]
    if top_count >= RESOLVE_TOP_MIN and top_count >= RESOLVE_RATIO * second_count:
        return top_name
    return "review"


def survey_video(row, data_dir, detector, gamma_table, flag_keys):
    """영상 1개를 조사해 출력 행(dict)을 만든다. 열지 못하거나 예외가 나면 frames_sampled 를 0으로 둔다.

    입력: row - inventory 행, data_dir - 영상 루트, detector/gamma_table - 공용 객체, flag_keys - manual_flags 키
    반환: COLUMNS 열을 가진 dict
    """
    counts = dict((name, 0) for name in SURVEY_ORIENTATIONS)
    width, height, sampled = "", "", 0
    capture = cv2.VideoCapture(os.path.join(data_dir, *row["rel_path"].split("/")))
    try:
        if capture.isOpened():
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            positions = sample_positions(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
            frames = read_sample_frames(capture, positions)
            sampled = len(frames)
            counts = count_by_orientation(frames, detector, gamma_table)
    except Exception as error:  # 영상 하나의 실패가 전체 조사를 멈추지 않게 종류만 알린다
        print("  예외: %s" % type(error).__name__)
    finally:
        capture.release()
    key = (int(row["fold"]), int(row["part"]), int(row["subject_id"]), int(row["class_label"]))
    return {"fold": row["fold"], "part": row["part"], "subject_id": row["subject_id"],
            "class_label": row["class_label"], "variant": row["variant"], "rel_path": row["rel_path"],
            "width": width, "height": height, "frames_sampled": sampled,
            "faces_0": counts["0"], "faces_cw90": counts["cw90"], "faces_180": counts["180"],
            "faces_ccw90": counts["ccw90"], "best_orientation": choose_best(counts),
            "manual_flag": 1 if key in flag_keys else 0, "resolved_orientation": resolve_orientation(counts)}


def write_survey(rows, output_path):
    """조사 결과를 CSV 로 저장한다(UTF-8 BOM, 줄바꿈 LF).

    입력: rows - 조사 결과 행 list(COLUMNS 열), output_path - 저장 경로
    """
    with open(output_path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def print_summary(results, elapsed):
    """방향별 영상 수(v1)와 총 소요 시간을 출력한다.

    입력: results - 조사 결과 행 list, elapsed - 총 소요(초)
    """
    tally = {}
    for result in results:
        tally[result["best_orientation"]] = tally.get(result["best_orientation"], 0) + 1
    print("== 요약: best_orientation(v1) 별 영상 수 (총 %d개) ==" % len(results))
    for name in ["0", "cw90", "180", "ccw90", "ambiguous", "none"]:
        print("  %-9s %d" % (name, tally.get(name, 0)))
    print("총 소요 %.1f초 (%.1f분)" % (elapsed, elapsed / 60.0))


def run_survey(arguments):
    """영상을 읽어 방향별 얼굴 수를 조사하고 CSV 를 저장한다.

    입력: arguments - 명령줄 옵션
    """
    data_dir = os.environ.get("RLDD_DATA_DIR")
    if not data_dir or not os.path.isdir(data_dir):
        sys.exit("환경변수 RLDD_DATA_DIR 를 UTA-RLDD 영상 폴더로 설정하세요.")
    rows = select_rows(load_inventory_rows(INVENTORY_PATH), arguments)
    flag_keys = load_manual_flags(FLAGS_PATH)
    detector = dlib.get_frontal_face_detector()
    gamma_table = build_gamma_table()
    results = []
    began = time.time()
    for number, row in enumerate(rows, 1):
        clock = time.time()
        result = survey_video(row, data_dir, detector, gamma_table, flag_keys)
        results.append(result)
        print("[%d/%d] %s frames=%s faces(0/cw90/180/ccw90)=%s/%s/%s/%s best=%s resolved=%s 소요 %.1f초" % (
            number, len(rows), row["rel_path"], result["frames_sampled"], result["faces_0"], result["faces_cw90"],
            result["faces_180"], result["faces_ccw90"], result["best_orientation"], result["resolved_orientation"],
            time.time() - clock))
        sys.stdout.flush()
    write_survey(results, OUTPUT_PATH)
    print("저장: configs/video_orientation_survey.csv (%d행)" % len(results))
    print_summary(results, time.time() - began)


def counts_from_row(row):
    """CSV 행의 faces_* 열에서 방향별 얼굴 수 dict 를 만든다.

    입력: row - video_orientation_survey.csv 의 행
    반환: {"0": 수, "cw90": 수, "180": 수, "ccw90": 수}
    """
    return {"0": int(row["faces_0"]), "cw90": int(row["faces_cw90"]),
            "180": int(row["faces_180"]), "ccw90": int(row["faces_ccw90"])}


def evaluate_predictions(rows):
    """사전 등록한 예측(고정 영상 6개와 v1 ambiguous 중 0°에서 정상 검출된 영상 묶음)의 적중 여부를 계산한다.

    입력: rows - resolved_orientation 이 채워진 행 list
    반환: (라벨, 기대값, 대상 수, 적중 수) 튜플 list. 묶음은 faces_0 가 읽은 프레임 수와 같은 v1 ambiguous 영상
    """
    by_path = dict((row["rel_path"], row) for row in rows)
    results = []
    for rel_path, expected in PREDICTIONS:
        row = by_path.get(rel_path)
        hit = 1 if row is not None and row["resolved_orientation"] == expected else 0
        results.append((rel_path, expected, 1, hit))
    group = [row for row in rows if row["best_orientation"] == "ambiguous"
             and int(row["frames_sampled"]) > 0 and int(row["faces_0"]) == int(row["frames_sampled"])]
    hits = sum(1 for row in group if row["resolved_orientation"] == "0")
    results.append(("v1 ambiguous 이고 faces_0 == 읽은 프레임 수", "0", len(group), hits))
    return results


def count_changes(rows):
    """v1 대비 바뀐 영상 수를 센다. v1 의 ambiguous 와 none 은 방향이 정해지지 않은 상태로 본다.

    입력: rows - resolved_orientation 이 채워진 행 list
    반환: (v1 에서 정해졌는데 v2 가 다른 값인 수, v1 에서 미정이었는데 v2 가 방향을 정한 수)
    """
    decided_changed = 0
    undecided_resolved = 0
    for row in rows:
        v1_decided = row["best_orientation"] not in ("ambiguous", "none")
        if v1_decided and row["resolved_orientation"] != row["best_orientation"]:
            decided_changed += 1
        if not v1_decided and row["resolved_orientation"] != "review":
            undecided_resolved += 1
    return decided_changed, undecided_resolved


def print_resolve_report(rows):
    """--resolve-only 의 보고: 예측 적중, 분류별 영상 수, v1 대비 변화, review 영상 목록.

    입력: rows - resolved_orientation 이 채워진 행 list
    """
    print("== 예측 적중 여부 (규칙 v2) ==")
    for label, expected, total, hits in evaluate_predictions(rows):
        print("  %s 기대 %s: %d/%d 적중" % (label, expected, hits, total))
    tally = {}
    for row in rows:
        tally[row["resolved_orientation"]] = tally.get(row["resolved_orientation"], 0) + 1
    print("== resolved_orientation 별 영상 수 (총 %d개) ==" % len(rows))
    for name in ["0", "cw90", "ccw90", "180", "review"]:
        print("  %-7s %d" % (name, tally.get(name, 0)))
    decided_changed, undecided_resolved = count_changes(rows)
    print("== v1 대비: 정해졌던 영상 중 값이 바뀐 수 %d, 미정(ambiguous/none)이었다가 방향이 정해진 수 %d, 합 %d ==" % (
        decided_changed, undecided_resolved, decided_changed + undecided_resolved))
    print("== review 영상 목록 (faces 0/cw90/180/ccw90) ==")
    for row in rows:
        if row["resolved_orientation"] == "review":
            print("  %s  %s/%s/%s/%s  v1=%s" % (row["rel_path"], row["faces_0"], row["faces_cw90"],
                                                row["faces_180"], row["faces_ccw90"], row["best_orientation"]))


def run_resolve_only():
    """영상을 읽지 않고 기존 조사 CSV 의 카운트만으로 resolved_orientation 열을 추가해 다시 저장한다."""
    if not os.path.isfile(OUTPUT_PATH):
        sys.exit("configs/video_orientation_survey.csv 가 없습니다. 먼저 조사를 실행하세요.")
    with open(OUTPUT_PATH, "r", newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        row["resolved_orientation"] = resolve_orientation(counts_from_row(row))
    write_survey(rows, OUTPUT_PATH)
    print("저장: configs/video_orientation_survey.csv (%d행, resolved_orientation 열 추가, 영상은 읽지 않음)" % len(rows))
    print_resolve_report(rows)


def main():
    """전체 흐름: --resolve-only 이면 CSV 만 갱신하고, 아니면 영상 조사를 실행한다."""
    arguments = parse_arguments()
    if arguments.resolve_only:
        run_resolve_only()
        return
    run_survey(arguments)


if __name__ == "__main__":
    main()
