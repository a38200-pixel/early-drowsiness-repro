# review_orientation.py
# 역할: 방향이 정해지지 않은(review) 영상을 사용자가 눈으로 보고 올바른 방향을 고르게 하는 대화형 도구.
#       영상마다 25%/50%/75% 위치의 프레임을 읽어(OpenCV 는 회전 태그를 적용하지 않은 원본 프레임) 4방향으로 돌린 모습을
#       2x2 로 한 창에 보여 주고, 얼굴이 바로 선 칸의 방향을 고르면 configs/video_orientation_overrides.csv 에 한 줄 추가한다.
#       blink 추출은 하지 않고, 영상 전체를 디코딩하지 않으며, 프레임이나 이미지를 파일로 저장하지 않는다(화면에 표시만 한다).
# 사용법: Anaconda Prompt 에서 conda activate ghoddoosian_tf1 후 저장소 루트에서 실행한다(메모를 터미널에 입력한다).
#           python src/repro/review_orientation.py                      (review 이고 override 가 아직 없는 영상 전부)
#           python src/repro/review_orientation.py --only Fold5_part2/59/0.MOV   (영상 1개만)
# 키(창을 클릭해 포커스를 둔 뒤): 0=0, 1=cw90, 2=ccw90, 3=180 (얼굴이 바로 선 칸의 방향), 스페이스=다음 프레임 위치,
#       s=이 영상 건너뛰기(기록 안 함), q=종료. 방향 키를 누르면 터미널에서 메모를 묻는다(Enter 만 치면 빈 메모).
#       얼굴이 어느 칸에서도 보이지 않으면 0 을 고르고 메모에 "얼굴 안 보임"을 남긴다.
# 환경변수: RLDD_DATA_DIR (영상 루트)
# 입력: configs/video_orientation_survey.csv (resolved_orientation 이 review 인 영상), configs/video_orientation_overrides.csv
# 출력: configs/video_orientation_overrides.csv 에 한 줄 추가(rel_path, orientation, observer, date, note). 이미 값이 있는 영상은 건너뛴다.
#       중간에 q 로 끝냈다가 다시 실행하면 이어서 한다.
"""영상 방향 육안 판정 도구(Python 3.6 호환)."""
import argparse
import csv
import datetime
import os
import sys

import cv2
import numpy as np

from orientation_common import normalize_orientation, rotate_frame

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SURVEY_PATH = os.path.join(REPO, "configs", "video_orientation_survey.csv")
OVERRIDES_PATH = os.path.join(REPO, "configs", "video_orientation_overrides.csv")
OVERRIDE_COLUMNS = ["rel_path", "orientation", "observer", "date", "note"]
OBSERVER = "사용자 육안"
REVIEW_PERCENTS = (25, 50, 75)
MAX_CELL_HEIGHT = 360
HEADER_HEIGHT = 30
WINDOW_NAME = "orientation review"
# 칸 배치(왼쪽 위부터 오른쪽으로): (표시 키, 방향 이름)
CELLS = [("0", "0"), ("1", "cw90"), ("2", "ccw90"), ("3", "180")]
KEY_ACTIONS = {ord("0"): "0", ord("1"): "cw90", ord("2"): "ccw90", ord("3"): "180", 32: "next",
               ord("s"): "skip", ord("S"): "skip", ord("q"): "quit", ord("Q"): "quit"}


def parse_arguments():
    """명령줄 옵션을 해석한다.

    반환: argparse.Namespace (only: rel_path 또는 None)
    """
    parser = argparse.ArgumentParser(description="영상 방향 육안 판정")
    parser.add_argument("--only", default=None, help="이 rel_path 영상 1개만 본다")
    return parser.parse_args()


def load_decided_paths(overrides_path):
    """overrides CSV 에서 이미 방향이 기록된(orientation 이 비어 있지 않은) 영상의 rel_path 집합을 만든다.

    입력: overrides_path - video_orientation_overrides.csv 경로
    반환: set (파일이 없으면 빈 set)
    """
    decided = set()
    if not os.path.isfile(overrides_path):
        return decided
    with open(overrides_path, "r", newline="", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            if (row.get("orientation") or "").strip() != "":
                decided.add(row["rel_path"])
    return decided


def load_review_targets(survey_path, decided, only):
    """육안 판정할 영상 목록을 만든다: review 이고 아직 기록이 없는 영상. --only 이면 그 영상 1개(기록이 있으면 빈 목록).

    입력: survey_path - 조사 CSV 경로, decided - 이미 기록된 rel_path 집합, only - rel_path 또는 None
    반환: rel_path 문자열 list (조사 CSV 순서). --only 영상이 조사 CSV 에 없으면 중단한다
    """
    if not os.path.isfile(survey_path):
        sys.exit("configs/video_orientation_survey.csv 가 없습니다. survey_orientation.py 로 조사를 먼저 실행하세요.")
    with open(survey_path, "r", newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    if only is not None:
        if only not in [row["rel_path"] for row in rows]:
            sys.exit("조사 CSV 에 없는 영상입니다: %s" % only)
        return [] if only in decided else [only]
    return [row["rel_path"] for row in rows
            if row.get("resolved_orientation") == "review" and row["rel_path"] not in decided]


def read_review_frames(video_path):
    """25%/50%/75% 위치의 프레임만 읽는다(전체 디코딩 아님). 읽지 못한 위치는 건너뛴다.

    입력: video_path - 영상 경로
    반환: (위치 퍼센트, BGR 프레임) 튜플 list. 영상을 열지 못하면 빈 list
    """
    capture = cv2.VideoCapture(video_path)
    frames = []
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) if capture.isOpened() else 0
        for percent in REVIEW_PERCENTS:
            if frame_count < 1:
                break
            capture.set(cv2.CAP_PROP_POS_FRAMES, min(int(frame_count * percent / 100.0), frame_count - 1))
            grabbed, frame = capture.read()
            if grabbed and frame is not None:
                frames.append((percent, frame))
    finally:
        capture.release()
    return frames


def fit_height(frame, max_height):
    """프레임의 높이가 max_height 보다 크면 비율을 유지하며 줄인다(작으면 그대로).

    입력: frame - BGR 프레임, max_height - 최대 높이(px)
    반환: 크기가 조정된 프레임
    """
    height, width = frame.shape[:2]
    if height <= max_height:
        return frame
    new_width = max(1, int(width * max_height / float(height)))
    return cv2.resize(frame, (new_width, max_height), interpolation=cv2.INTER_AREA)


def build_cell(frame, key, orientation):
    """프레임을 해당 방향으로 돌려 높이를 줄이고, 왼쪽 위에 '[키] 방향이름' 라벨을 적은 칸 이미지를 만든다.

    입력: frame - 원본 BGR 프레임, key - 표시할 키 문자, orientation - 방향 이름(orientation_common 의 정의)
    반환: 라벨이 그려진 칸 이미지(원본 프레임은 바꾸지 않는다)
    """
    cell = fit_height(rotate_frame(frame, orientation), MAX_CELL_HEIGHT).copy()
    text = "[%s] %s" % (key, orientation)
    cv2.putText(cell, text, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(cell, text, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
    return cell


def build_canvas(frame, header_text):
    """4방향 칸을 2x2 로 배치한 한 장의 표시용 이미지를 만든다(파일로 저장하지 않는다).

    입력: frame - 원본 BGR 프레임, header_text - 맨 위에 적을 안내 문구(ASCII)
    반환: 2x2 캔버스 이미지(칸 크기는 가장 큰 칸에 맞추고 남는 곳은 검게 둔다)
    """
    cells = [build_cell(frame, key, orientation) for key, orientation in CELLS]
    cell_height = max(cell.shape[0] for cell in cells)
    cell_width = max(cell.shape[1] for cell in cells)
    canvas = np.zeros((HEADER_HEIGHT + 2 * cell_height, 2 * cell_width, 3), dtype=np.uint8)
    for index, cell in enumerate(cells):
        top = HEADER_HEIGHT + (index // 2) * cell_height
        left = (index % 2) * cell_width
        canvas[top:top + cell.shape[0], left:left + cell.shape[1]] = cell
    cv2.putText(canvas, header_text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def wait_for_action():
    """허용된 키(0~3, 스페이스, s, q)가 눌릴 때까지 기다리고 해당 동작 이름을 돌려준다.

    반환: "0", "cw90", "ccw90", "180", "next", "skip", "quit" 중 하나
    """
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key in KEY_ACTIONS:
            return KEY_ACTIONS[key]


def ask_note():
    """터미널에서 메모를 묻는다. 입력이 막혀 있거나 Enter 만 치면 빈 문자열.

    반환: 메모 문자열(앞뒤 공백 제거)
    """
    try:
        return input("  메모(없으면 Enter, 얼굴이 어느 칸에도 안 보이면 '얼굴 안 보임'): ").strip()
    except EOFError:
        return ""


def row_prefix(overrides_path):
    """overrides CSV 에 한 줄을 덧붙이기 전에 먼저 써야 할 문자열을 정한다.

    입력: overrides_path - video_orientation_overrides.csv 경로
    반환: 파일이 없거나 비어 있으면 헤더 줄, 마지막이 줄바꿈이 아니면 "\\n", 아니면 빈 문자열
    """
    if not os.path.isfile(overrides_path) or os.path.getsize(overrides_path) == 0:
        return ",".join(OVERRIDE_COLUMNS) + "\n"
    with open(overrides_path, "rb") as file:
        file.seek(-1, os.SEEK_END)
        return "" if file.read(1) == b"\n" else "\n"


def append_override(overrides_path, rel_path, orientation, note):
    """overrides CSV 에 육안 판정 결과 한 줄을 추가한다(기존 줄은 건드리지 않는다).

    입력: overrides_path - CSV 경로, rel_path - 영상, orientation - 표준 방향 이름, note - 메모
    """
    canonical = normalize_orientation(orientation)
    prefix = row_prefix(overrides_path)
    with open(overrides_path, "a", newline="", encoding="utf-8") as file:
        file.write(prefix)
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow([rel_path, canonical, OBSERVER, datetime.date.today().isoformat(), note])


def review_video(rel_path, data_dir, label):
    """영상 1개를 2x2 창으로 보여 주고 사용자의 선택을 기다린다. 스페이스로 25%/50%/75% 위치를 돌려 본다.

    입력: rel_path - 영상, data_dir - 영상 루트, label - 진행 표시 문자열(예: "3/14")
    반환: ("chosen", 방향 이름) 또는 ("skip", None) 또는 ("quit", None). 프레임을 읽지 못하면 ("skip", None)
    """
    frames = read_review_frames(os.path.join(data_dir, *rel_path.split("/")))
    if not frames:
        print("  프레임을 읽지 못해 건너뜁니다: %s" % rel_path)
        return "skip", None
    position = 0
    while True:
        percent, frame = frames[position]
        header = "%s  %s  %d%%  [0-3] choose  [space] next frame  [s] skip  [q] quit" % (label, rel_path, percent)
        cv2.imshow(WINDOW_NAME, build_canvas(frame, header))
        action = wait_for_action()
        if action == "next":
            position = (position + 1) % len(frames)
        elif action in ("skip", "quit"):
            return action, None
        else:
            return "chosen", action


def run_review(targets, data_dir):
    """대상 영상을 차례로 보여 주고 선택된 방향을 overrides CSV 에 기록한다.

    입력: targets - 판정할 rel_path list, data_dir - 영상 루트
    반환: (판정한 수, 건너뛴 수)
    """
    chosen_count, skipped = 0, 0
    for number, rel_path in enumerate(targets, 1):
        print("[%d/%d] %s" % (number, len(targets), rel_path))
        action, orientation = review_video(rel_path, data_dir, "%d/%d" % (number, len(targets)))
        if action == "quit":
            break
        if action == "skip":
            skipped += 1
            continue
        append_override(OVERRIDES_PATH, rel_path, orientation, ask_note())
        chosen_count += 1
        print("  기록: %s -> %s" % (rel_path, orientation))
    return chosen_count, skipped


def main():
    """전체 흐름: 환경 확인, 대상 선택, 창으로 판정, 요약 출력."""
    arguments = parse_arguments()
    data_dir = os.environ.get("RLDD_DATA_DIR")
    if not data_dir or not os.path.isdir(data_dir):
        sys.exit("환경변수 RLDD_DATA_DIR 를 UTA-RLDD 영상 폴더로 설정하세요.")
    targets = load_review_targets(SURVEY_PATH, load_decided_paths(OVERRIDES_PATH), arguments.only)
    print("판정할 영상: %d개 (이미 기록된 영상은 건너뜀)" % len(targets))
    if not targets:
        return
    try:
        chosen_count, skipped = run_review(targets, data_dir)
    finally:
        cv2.destroyAllWindows()
    print("요약: 판정 %d개, 건너뜀 %d개, 남은 영상 %d개" % (chosen_count, skipped, len(targets) - chosen_count - skipped))


if __name__ == "__main__":
    main()
