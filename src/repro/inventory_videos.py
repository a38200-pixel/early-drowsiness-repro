# inventory_videos.py
# 역할: 압축을 푼 UTA-RLDD 영상 폴더를 읽어 영상별 컨테이너 메타와 표본 프레임 5장의 통계·얼굴 검출 결과를
#       configs/video_inventory.csv 로 저장한다. 영상은 전체 디코딩하지 않고, 프레임을 파일로 저장하지도 않는다.
# 사용법: conda 환경 ghoddoosian_tf1 의 python 으로 저장소 루트에서 실행한다.
#           python src/repro/inventory_videos.py --subset flagged   (육안 점검 8개 + 대조군 3개만, 시험 실행)
#           python src/repro/inventory_videos.py                    (전체 영상)
# 환경변수: RLDD_DATA_DIR (필수) - FoldN_partM 폴더가 들어 있는 UTA-RLDD 루트
# 입력: configs/video_manual_flags.csv (사용자 육안 점검 기록, 있으면 manual_flag 열에 반영)
# 출력: configs/video_inventory.csv (UTF-8 BOM, 절대경로 없음. rel_path 는 RLDD_DATA_DIR 기준 상대경로)
"""UTA-RLDD 영상 목록 작성 스크립트(Python 3.6 호환)."""
import argparse
import csv
import os
import re
import sys
import time

import cv2
import dlib
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FLAGS_RELATIVE = "configs/video_manual_flags.csv"
INVENTORY_RELATIVE = "configs/video_inventory.csv"
FOLDER_PATTERN = re.compile(r"^Fold(\d+)_part(\d+)$")
NAME_PATTERN = re.compile(r"^(0|5|10)(?:_(\w+))?$")
SAMPLE_PERCENTS = (10, 30, 50, 70, 90)
DETECT_WIDTH = 450        # 원본 blink_video.py 가 얼굴 검출 전에 줄이는 가로 크기
GAMMA = 1.5               # 원본 blink_video.py 의 adjust_gamma 값
COLUMNS = ["fold", "part", "subject_id", "class_label", "variant", "extension", "size_bytes", "fourcc",
           "frame_count", "fps", "width", "height", "orientation_meta", "rel_path", "frames_read",
           "gray_mean", "gray_std_max", "faces_orig", "faces_rot180", "note", "manual_flag"]


def parse_arguments():
    """명령줄 인자를 해석한다.

    반환: argparse.Namespace (subset: "all" 또는 "flagged")
    """
    parser = argparse.ArgumentParser(description="UTA-RLDD 영상 목록 작성")
    parser.add_argument("--subset", choices=["all", "flagged"], default="all",
                        help="flagged: 육안 점검 영상 + 정상 대조군 3개만 처리")
    return parser.parse_args()


def parse_video_name(file_name):
    """파일명에서 클래스·변형·확장자를 분리한다.

    입력: file_name - 예: "10_1.mp4"
    반환: (클래스 정수 또는 None, 변형 문자열, 확장자(대소문자 유지)). 형식이 다르면 클래스는 None
    """
    stem, extension = os.path.splitext(file_name)
    matched = NAME_PATTERN.match(stem)
    if matched is None:
        return None, "", extension
    return int(matched.group(1)), matched.group(2) or "", extension


def scan_subject_folder(data_dir, folder_name, subject_name, fold, part):
    """피험자 폴더 하나의 영상 파일을 훑어 기본 항목(파일 메타만) 목록을 만든다.

    입력: data_dir - RLDD 루트, folder_name - FoldN_partM, subject_name - 피험자 폴더명, fold/part - 정수
    반환: 항목 dict 의 list (영상은 열지 않는다)
    """
    entries = []
    subject_path = os.path.join(data_dir, folder_name, subject_name)
    for file_name in sorted(os.listdir(subject_path)):
        full_path = os.path.join(subject_path, file_name)
        if not os.path.isfile(full_path):
            continue
        class_label, variant, extension = parse_video_name(file_name)
        entries.append({
            "fold": fold, "part": part, "subject_id": int(subject_name), "class_label": class_label,
            "variant": variant, "extension": extension, "size_bytes": os.path.getsize(full_path),
            "rel_path": "/".join([folder_name, subject_name, file_name]),
            "note_list": [] if class_label is not None else ["파일명이 클래스 형식과 다름"],
        })
    return entries


def scan_videos(data_dir):
    """RLDD 루트 아래 FoldN_partM/<피험자ID>/ 구조를 모두 훑는다.

    입력: data_dir - RLDD 루트
    반환: (fold, part, 피험자, 클래스, 변형) 순으로 정렬된 항목 list. 형식이 다른 폴더는 건너뛴다
    """
    entries = []
    for folder_name in sorted(os.listdir(data_dir)):
        matched = FOLDER_PATTERN.match(folder_name)
        if matched is None or not os.path.isdir(os.path.join(data_dir, folder_name)):
            continue
        fold, part = int(matched.group(1)), int(matched.group(2))
        for subject_name in sorted(os.listdir(os.path.join(data_dir, folder_name))):
            if subject_name.isdigit():
                entries.extend(scan_subject_folder(data_dir, folder_name, subject_name, fold, part))
    entries.sort(key=lambda e: (e["fold"], e["part"], e["subject_id"],
                                -1 if e["class_label"] is None else e["class_label"], e["variant"]))
    return entries


def load_manual_flags(flags_path):
    """육안 점검 기록 CSV 를 읽어 (fold, part, subject_id, class_label) 집합으로 돌려준다.

    입력: flags_path - video_manual_flags.csv 경로
    반환: set. 파일이 없으면 빈 set
    """
    keys = set()
    if not os.path.isfile(flags_path):
        return keys
    with open(flags_path, "r", newline="", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            keys.add((int(row["fold"]), int(row["part"]), int(row["subject_id"]), int(row["class_label"])))
    return keys


def mark_manual_flags(entries, flag_keys):
    """항목마다 manual_flag(0/1)를 채우고, 어떤 항목과도 맞지 않은 기록 수를 돌려준다.

    입력: entries - scan_videos 결과, flag_keys - load_manual_flags 결과
    반환: 대응하는 영상이 없는 기록의 수
    """
    matched_keys = set()
    for entry in entries:
        key = (entry["fold"], entry["part"], entry["subject_id"], entry["class_label"])
        entry["manual_flag"] = 1 if key in flag_keys else 0
        if key in flag_keys:
            matched_keys.add(key)
    return len(flag_keys - matched_keys)


def select_flagged_subset(entries):
    """시험 실행용 부분집합을 고른다: manual_flag 영상 전부 + 클래스 0/5/10 별 정상 영상 첫 1개씩.

    입력: entries - 정렬된 항목 list
    반환: 항목 list (정렬 순서 유지)
    """
    chosen_ids = set(id(entry) for entry in entries if entry["manual_flag"] == 1)
    for class_label in (0, 5, 10):
        for entry in entries:
            if entry["manual_flag"] == 0 and entry["class_label"] == class_label and not entry["variant"]:
                chosen_ids.add(id(entry))
                break
    return [entry for entry in entries if id(entry) in chosen_ids]


def fourcc_to_text(value):
    """CAP_PROP_FOURCC 정수를 4글자 문자열로 바꾼다. 읽을 수 없는 글자는 '?', 값이 0이면 빈 문자열.

    입력: value - capture.get(CAP_PROP_FOURCC) 결과
    반환: 문자열
    """
    code = int(value)
    if code == 0:
        return ""
    letters = []
    for shift in (0, 8, 16, 24):
        char_code = (code >> shift) & 0xFF
        letters.append(chr(char_code) if 32 <= char_code < 127 else "?")
    return "".join(letters)


def read_container_meta(capture):
    """열린 VideoCapture 에서 컨테이너 메타(코덱·프레임 수·fps·크기·회전 메타)를 읽는다.

    입력: capture - cv2.VideoCapture
    반환: dict. orientation_meta 는 이 OpenCV 에 해당 상수가 없으면 빈 문자열
    """
    meta = {
        "fourcc": fourcc_to_text(capture.get(cv2.CAP_PROP_FOURCC)),
        "frame_count": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": round(capture.get(cv2.CAP_PROP_FPS), 3),
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "orientation_meta": "",
    }
    orientation_prop = getattr(cv2, "CAP_PROP_ORIENTATION_META", None)
    if orientation_prop is not None:
        meta["orientation_meta"] = int(capture.get(orientation_prop))
    return meta


def sample_positions(frame_count):
    """영상 길이의 10/30/50/70/90% 위치에 해당하는 프레임 번호를 계산한다.

    입력: frame_count - 전체 프레임 수
    반환: 프레임 번호 list (frame_count 가 1 미만이면 빈 list)
    """
    if frame_count < 1:
        return []
    positions = []
    for percent in SAMPLE_PERCENTS:
        positions.append(min(int(frame_count * percent / 100.0), frame_count - 1))
    return positions


def read_sample_frames(capture, positions, notes):
    """지정한 위치의 프레임만 읽는다(전체 디코딩 아님). 읽지 못한 위치는 notes 에 기록한다.

    입력: capture - 열린 VideoCapture, positions - 프레임 번호 list, notes - 메시지를 덧붙일 list
    반환: 읽힌 프레임(BGR 배열)의 list
    """
    frames = []
    for percent, position in zip(SAMPLE_PERCENTS, positions):
        capture.set(cv2.CAP_PROP_POS_FRAMES, position)
        grabbed, frame = capture.read()
        if grabbed and frame is not None:
            frames.append(frame)
        else:
            notes.append("%d%% 위치(프레임 %d) 읽기 실패" % (percent, position))
    return frames


def build_gamma_table():
    """원본 blink_video.py 의 adjust_gamma 와 같은 256칸 변환표를 만든다.

    반환: uint8 배열(길이 256)
    """
    values = []
    for index in range(256):
        values.append(((index / 255.0) ** (1.0 / GAMMA)) * 255)
    return np.array(values).astype("uint8")


def prepare_gray(frame):
    """프레임을 가로 450 으로 줄여 회색조로 바꾼다(원본 blink_video.py 의 전처리 순서).

    입력: frame - BGR 프레임
    반환: 회색조 배열(감마 보정 전)
    """
    height, width = frame.shape[:2]
    new_height = int(height * DETECT_WIDTH / float(width))
    resized = cv2.resize(frame, (DETECT_WIDTH, new_height), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)


def analyze_frames(frames, detector, gamma_table):
    """읽힌 프레임들의 회색조 평균·최대 표준편차와 얼굴 검출 프레임 수(원본·180도 회전)를 계산한다.

    입력: frames - BGR 프레임 list, detector - dlib 정면 얼굴 검출기, gamma_table - build_gamma_table 결과
    반환: dict(gray_mean, gray_std_max, faces_orig, faces_rot180). 프레임이 없으면 값은 빈 문자열
    """
    result = {"gray_mean": "", "gray_std_max": "", "faces_orig": "", "faces_rot180": ""}
    if not frames:
        return result
    means, stds, found_orig, found_rot = [], [], 0, 0
    for frame in frames:
        gray = prepare_gray(frame)
        means.append(float(gray.mean()))
        stds.append(float(gray.std()))
        corrected = cv2.LUT(gray, gamma_table)
        if len(detector(corrected, 0)) > 0:
            found_orig += 1
        if len(detector(cv2.flip(corrected, -1), 0)) > 0:
            found_rot += 1
    result.update({"gray_mean": round(sum(means) / len(means), 2), "gray_std_max": round(max(stds), 2),
                   "faces_orig": found_orig, "faces_rot180": found_rot})
    return result


def inspect_video(video_path, detector, gamma_table):
    """영상 하나를 열어 메타와 표본 프레임 분석 결과를 모은다. 실패해도 예외를 밖으로 내보내지 않는다.

    입력: video_path - 영상 경로, detector/gamma_table - 재사용할 검출기와 변환표
    반환: dict(메타 열 + frames_read + 분석 열 + note 문자열)
    """
    notes = []
    row = {"frames_read": 0, "fourcc": "", "frame_count": "", "fps": "", "width": "", "height": "",
           "orientation_meta": "", "gray_mean": "", "gray_std_max": "", "faces_orig": "", "faces_rot180": ""}
    capture = cv2.VideoCapture(video_path)
    try:
        if not capture.isOpened():
            notes.append("cv2 로 열지 못함")
        else:
            row.update(read_container_meta(capture))
            positions = sample_positions(row["frame_count"])
            if not positions:
                notes.append("frame_count 가 1 미만이라 표본 위치를 정할 수 없음")
            frames = read_sample_frames(capture, positions, notes)
            row["frames_read"] = len(frames)
            row.update(analyze_frames(frames, detector, gamma_table))
    except Exception as error:  # 영상 하나의 실패가 전체 목록 작성을 멈추지 않게 종류만 기록한다
        notes.append("예외 발생: %s" % type(error).__name__)
    finally:
        capture.release()
    row["note"] = "; ".join(notes)
    return row


def inspect_all(entries, data_dir, detector, gamma_table):
    """항목 전체에 inspect_video 를 적용해 완성된 행 list 를 만든다. 진행 상황은 경로 없이 출력한다.

    입력: entries - 처리할 항목 list, data_dir - RLDD 루트, detector/gamma_table - 공용 객체
    반환: COLUMNS 열을 모두 가진 dict 의 list
    """
    rows = []
    for number, entry in enumerate(entries, 1):
        video_path = os.path.join(data_dir, *entry["rel_path"].split("/"))
        row = dict(entry)
        measured = inspect_video(video_path, detector, gamma_table)
        extra_note = "; ".join(entry["note_list"])
        row.update(measured)
        row["note"] = "; ".join(part for part in (extra_note, measured["note"]) if part)
        rows.append(row)
        print("[%d/%d] Fold%d_part%d 피험자 %d 클래스 %s frames_read=%s" % (
            number, len(entries), entry["fold"], entry["part"], entry["subject_id"],
            entry["class_label"], row["frames_read"]))
    return rows


def write_inventory(rows, output_path):
    """행 list 를 CSV 로 저장한다(UTF-8 BOM, 줄바꿈 LF).

    입력: rows - inspect_all 결과, output_path - 저장 경로
    """
    with open(output_path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def print_summary(entries):
    """fold 별 영상 수·피험자 수·총 용량(GB 와 GiB)과 피험자별 영상 수 분포를 출력한다.

    입력: entries - scan_videos 결과(전체)
    """
    print("== fold별 요약 ==")
    for fold in sorted(set(e["fold"] for e in entries)):
        in_fold = [e for e in entries if e["fold"] == fold]
        size = sum(e["size_bytes"] for e in in_fold)
        print("Fold%d 영상 %d개, 피험자 %d명, 용량 %.2f GB (%.2f GiB)" % (
            fold, len(in_fold), len(set(e["subject_id"] for e in in_fold)), size / 1e9, size / 2.0 ** 30))
    total = sum(e["size_bytes"] for e in entries)
    print("전체 영상 %d개, 피험자 %d명, 용량 %.2f GB (%.2f GiB)" % (
        len(entries), len(set(e["subject_id"] for e in entries)), total / 1e9, total / 2.0 ** 30))
    counts = {}
    for entry in entries:
        counts[entry["subject_id"]] = counts.get(entry["subject_id"], 0) + 1
    print("피험자별 영상 수가 3개가 아닌 ID:", sorted((k, v) for k, v in counts.items() if v != 3))


def main():
    """전체 흐름: 폴더 훑기, 육안 기록 대조, 요약 출력, (부분) 분석, CSV 저장, 소요 시간 출력."""
    arguments = parse_arguments()
    data_dir = os.environ.get("RLDD_DATA_DIR")
    if not data_dir or not os.path.isdir(data_dir):
        sys.exit("환경변수 RLDD_DATA_DIR 를 UTA-RLDD 루트 폴더로 설정하세요.")
    entries = scan_videos(data_dir)
    unmatched = mark_manual_flags(entries, load_manual_flags(os.path.join(REPO, *FLAGS_RELATIVE.split("/"))))
    print_summary(entries)
    print("육안 기록 중 대응 영상이 없는 행: %d" % unmatched)
    targets = entries if arguments.subset == "all" else select_flagged_subset(entries)
    started = time.time()
    rows = inspect_all(targets, data_dir, dlib.get_frontal_face_detector(), build_gamma_table())
    elapsed = time.time() - started
    write_inventory(rows, os.path.join(REPO, *INVENTORY_RELATIVE.split("/")))
    print("저장: %s (%d행)" % (INVENTORY_RELATIVE, len(rows)))
    print("걸린 시간 %.1f초, 영상당 평균 %.2f초, 전체 %d개 예상 %.1f분" % (
        elapsed, elapsed / len(rows), len(entries), elapsed / len(rows) * len(entries) / 60.0))


if __name__ == "__main__":
    main()
