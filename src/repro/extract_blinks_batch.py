# extract_blinks_batch.py
# 역할: configs/video_inventory.csv 의 영상을 reproduction_blink_headless.py 로 일괄 추출하는 관리 도구.
#       영상 1개마다 별도 프로세스를 띄워 전역 상태를 격리하고, 재시작·실패 기록·진행 로그·매니페스트를 관리한다.
# 사용법: conda 환경 ghoddoosian_tf1 을 활성화하고 저장소 루트에서 실행한다.
#           python src/repro/extract_blinks_batch.py --list                  (실행하지 않고 대상과 순서만 출력)
#           python src/repro/extract_blinks_batch.py --limit 3 --workers 1   (앞 3개만 시험)
#           python src/repro/extract_blinks_batch.py --workers 4             (전체)
#         옵션: --workers N, --list, --only-fold F, --limit K, --flagged-only, --retry-failed
# 회전: 영상마다 configs/video_orientation_overrides.csv(육안 확인) 값이 있으면 그 값, 없으면
#       configs/video_orientation_survey.csv 의 resolved_orientation 을 자식 프로세스의 --rotate 로 넘긴다.
#       resolved_orientation 이 review 이고 override 도 없는 영상은 처리하지 않고 목록으로 출력한다.
# 환경변수: RLDD_DATA_DIR (영상 루트), BLINK_OUT_DIR (출력 루트, 필수·저장소 밖·기본값 없음),
#           DLIB_LANDMARK_PATH (실행 시 필수)
"""blink 일괄 추출 관리 도구(Python 3.6 호환).

출력: <BLINK_OUT_DIR>/Fold<N>_part<M>/<피험자폴더>/<클래스>[_<변형>].txt (영상 파일 이름과 같은 규칙)
매니페스트: <BLINK_OUT_DIR>/batch_manifest.csv (영상이 끝날 때마다 한 줄 추가, 저장소 밖)
기본 순서: manual_flags 영상을 뺀 뒤 용량이 작은 것부터. manual_flags 영상은 --flagged-only 로 따로 처리한다.
재시작: 최종 출력 파일이 있으면 건너뛰고 덮어쓰지 않는다. 자식 프로세스는 <이름>.txt.part 에 쓰고
        성공했을 때만 최종 이름으로 바꾸므로, 중간에 끊겨도 반쪽 출력이 완료로 오인되지 않는다.
실패: 종료 코드가 0이 아니거나 출력이 없으면 매니페스트에 기록만 하고 계속한다.
      기본 실행은 실패로 기록된 영상을 건너뛰고, --retry-failed 는 그 영상들만 다시 처리한다.
--limit K 는 이번에 실제로 처리할 목록의 앞 K개이다(0이면 아무것도 처리하지 않는다).
회전: override(육안 확인)가 있으면 그 값, 없으면 resolved_orientation 을 쓴다. review 이면서 override 가 없는 영상은
      처리하지 않고 카운트와 함께 목록으로 출력한다. 자식 프로세스에는 --rotate 를 넘긴다.
매니페스트 열에는 rotate 와 face_frames(자식 출력에서 "not grabbed" 다음 줄의 숫자, 얼굴이 검출된 프레임 수)가 있다.
      기존 매니페스트의 열 이름이 다르면 이어 쓰지 않고 중단한다.
"""
import argparse
import csv
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from orientation_common import normalize_orientation, to_rotate_option
from reproduction_blink_headless import count_blink_rows

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HEADLESS_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reproduction_blink_headless.py")
INVENTORY_PATH = os.path.join(REPO, "configs", "video_inventory.csv")
FLAGS_PATH = os.path.join(REPO, "configs", "video_manual_flags.csv")
SURVEY_PATH = os.path.join(REPO, "configs", "video_orientation_survey.csv")
OVERRIDES_PATH = os.path.join(REPO, "configs", "video_orientation_overrides.csv")
MANIFEST_NAME = "batch_manifest.csv"
MANIFEST_COLUMNS = ["fold", "part", "subject_id", "class_label", "variant", "status", "return_code",
                    "elapsed_s", "blink_rows", "started", "ended", "rotate", "face_frames"]
FAILURE_STATUSES = ("failed", "no_output")
TIME_FORMAT = "%Y-%m-%d %H:%M:%S"


def parse_arguments():
    """명령줄 옵션을 해석한다.

    반환: argparse.Namespace (workers, list_only, only_fold, limit, flagged_only, retry_failed)
    """
    parser = argparse.ArgumentParser(description="blink 일괄 추출 관리 도구")
    parser.add_argument("--workers", type=int, default=1, help="동시에 돌릴 프로세스 수(기본 1)")
    parser.add_argument("--list", action="store_true", dest="list_only", help="실행하지 않고 대상만 출력")
    parser.add_argument("--only-fold", type=int, choices=[1, 2, 3, 4, 5], default=None, dest="only_fold")
    parser.add_argument("--limit", type=int, default=None, help="이번에 처리할 목록의 앞 K개")
    parser.add_argument("--flagged-only", action="store_true", dest="flagged_only", help="manual_flags 영상만")
    parser.add_argument("--retry-failed", action="store_true", dest="retry_failed", help="실패 기록 영상만 재처리")
    arguments = parser.parse_args()
    if arguments.workers < 1:
        parser.error("--workers 는 1 이상이어야 합니다.")
    if arguments.limit is not None and arguments.limit < 0:
        parser.error("--limit 는 0 이상이어야 합니다.")
    return arguments


def is_inside(path, parent):
    """path 가 parent 폴더 자신이거나 그 안인지 판단한다(심볼릭 링크 해석, Windows 대소문자 무시).

    입력: path - 검사할 경로, parent - 기준 폴더
    반환: 안에 있으면 True
    """
    real_path = os.path.normcase(os.path.realpath(path))
    real_parent = os.path.normcase(os.path.realpath(parent))
    return real_path == real_parent or real_path.startswith(real_parent + os.sep)


def require_output_dir():
    """BLINK_OUT_DIR 을 읽고 검사한다. 설정이 없거나 저장소 폴더 안이면 중단한다(기본값 없음).

    반환: 출력 루트의 실제 경로 문자열
    """
    raw_value = os.environ.get("BLINK_OUT_DIR")
    if not raw_value:
        sys.exit("환경변수 BLINK_OUT_DIR 를 출력 루트 폴더로 설정하세요(저장소 밖, 기본값 없음).")
    out_dir = os.path.realpath(raw_value)
    if is_inside(out_dir, REPO):
        sys.exit("BLINK_OUT_DIR 이 저장소 폴더 안입니다. 저장소 밖 폴더를 지정하세요.")
    return out_dir


def require_run_environment():
    """실제 추출에 필요한 환경변수(RLDD_DATA_DIR, DLIB_LANDMARK_PATH)를 확인한다. 문제가 있으면 중단한다."""
    problems = []
    data_dir = os.environ.get("RLDD_DATA_DIR")
    if not data_dir or not os.path.isdir(data_dir):
        problems.append("RLDD_DATA_DIR 를 UTA-RLDD 영상 폴더로 설정하세요")
    dlib_path = os.environ.get("DLIB_LANDMARK_PATH")
    if not dlib_path or not os.path.isfile(dlib_path):
        problems.append("DLIB_LANDMARK_PATH 를 shape_predictor_68_face_landmarks.dat 로 설정하세요")
    if problems:
        sys.exit("; ".join(problems))


def read_inventory(inventory_path):
    """영상 목록 CSV 를 읽어 항목 dict 의 list 로 돌려준다. 클래스를 알 수 없는 행은 건너뛰고 알린다.

    입력: inventory_path - video_inventory.csv 경로
    반환: dict(fold, part, subject_id, class_label, variant, size_bytes, frame_count, rel_path) 의 list
    """
    entries = []
    with open(inventory_path, "r", newline="", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            if row["class_label"] == "":
                print("경고: 클래스를 알 수 없어 건너뜀: %s" % row["rel_path"])
                continue
            entries.append({
                "fold": int(row["fold"]), "part": int(row["part"]), "subject_id": int(row["subject_id"]),
                "class_label": int(row["class_label"]), "variant": row["variant"],
                "size_bytes": int(row["size_bytes"]),
                "frame_count": int(row["frame_count"]) if row["frame_count"] != "" else 0,
                "rel_path": row["rel_path"],
            })
    return entries


def load_flag_keys(flags_path):
    """육안 점검 기록 CSV 에서 (fold, part, subject_id, class_label) 집합을 만든다.

    입력: flags_path - video_manual_flags.csv 경로
    반환: set (파일이 없으면 빈 set)
    """
    keys = set()
    if not os.path.isfile(flags_path):
        return keys
    with open(flags_path, "r", newline="", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            keys.add((int(row["fold"]), int(row["part"]), int(row["subject_id"]), int(row["class_label"])))
    return keys


def is_flagged(entry, flag_keys):
    """영상이 육안 점검 기록(manual_flags)에 있는 영상인지 알려 준다.

    입력: entry - 영상 항목, flag_keys - load_flag_keys 결과
    반환: bool
    """
    return (entry["fold"], entry["part"], entry["subject_id"], entry["class_label"]) in flag_keys


def select_targets(entries, flag_keys, arguments):
    """옵션에 맞는 영상을 골라 처리 순서(용량이 작은 것부터)로 정렬한다.

    입력: entries - 전체 항목, flag_keys - manual_flags 키, arguments - 명령줄 옵션
    반환: 정렬된 항목 list. 기본은 manual_flags 영상 제외, --flagged-only 이면 그 영상만
    """
    chosen = []
    for entry in entries:
        if is_flagged(entry, flag_keys) != arguments.flagged_only:
            continue
        if arguments.only_fold is not None and entry["fold"] != arguments.only_fold:
            continue
        chosen.append(entry)
    chosen.sort(key=lambda e: (e["size_bytes"], e["fold"], e["part"], e["subject_id"], e["class_label"], e["variant"]))
    return chosen


def entry_key(entry):
    """영상 항목을 매니페스트와 대조할 키 튜플로 바꾼다.

    입력: entry - 영상 항목
    반환: (fold, part, subject_id, class_label, variant)
    """
    return (entry["fold"], entry["part"], entry["subject_id"], entry["class_label"], entry["variant"])


def load_survey_orientations(survey_path):
    """회전 조사 CSV 에서 영상별 resolved_orientation 과 방향별 얼굴 수를 읽는다.

    입력: survey_path - video_orientation_survey.csv 경로
    반환: {rel_path: {"resolved": 값, "counts": (faces_0, faces_cw90, faces_180, faces_ccw90) 문자열 튜플}}.
          파일이 없거나 resolved_orientation 열이 없으면 안내 후 중단한다
    """
    if not os.path.isfile(survey_path):
        sys.exit("configs/video_orientation_survey.csv 가 없습니다. survey_orientation.py 로 조사를 먼저 실행하세요.")
    survey_map = {}
    with open(survey_path, "r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if "resolved_orientation" not in (reader.fieldnames or []):
            sys.exit("video_orientation_survey.csv 에 resolved_orientation 열이 없습니다. "
                     "survey_orientation.py --resolve-only 를 먼저 실행하세요.")
        for row in reader:
            counts = (row["faces_0"], row["faces_cw90"], row["faces_180"], row["faces_ccw90"])
            survey_map[row["rel_path"]] = {"resolved": row["resolved_orientation"], "counts": counts}
    return survey_map


def load_overrides(overrides_path):
    """육안 확인 결과 CSV(rel_path, orientation, ...)를 읽어 {rel_path: 표준 방향 이름} 으로 돌려준다.

    입력: overrides_path - video_orientation_overrides.csv 경로
    반환: dict. 파일이 없으면 빈 dict. orientation 이 빈 행은 건너뛰고(아직 못 본 영상),
          허용되지 않는 값(허용: 0, none, cw90, ccw90, 180)이나 같은 영상의 중복 행이 있으면 안내 후 중단한다
    """
    overrides = {}
    if not os.path.isfile(overrides_path):
        return overrides
    with open(overrides_path, "r", newline="", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            value = (row["orientation"] or "").strip()
            if value == "":
                continue
            canonical = normalize_orientation(value)
            if canonical is None:
                sys.exit("video_orientation_overrides.csv 의 orientation 값이 올바르지 않습니다 (%s: %s). "
                         "허용: 0, none, cw90, ccw90, 180" % (row["rel_path"], value))
            if row["rel_path"] in overrides:
                sys.exit("video_orientation_overrides.csv 에 같은 영상이 두 번 있습니다: %s" % row["rel_path"])
            overrides[row["rel_path"]] = canonical
    return overrides


def rotation_label(entry, survey_map, overrides):
    """영상에 쓸 회전을 정한다. override(육안 확인)가 있으면 그 값, 없으면 resolved_orientation 을 쓴다.

    입력: entry - 영상 항목, survey_map - load_survey_orientations 결과, overrides - load_overrides 결과
    반환: --rotate 옵션 이름("none", "cw90", "ccw90", "180") 또는 "review"(정할 수 없음, 조사 결과 없음 포함)
    """
    rel_path = entry["rel_path"]
    if rel_path in overrides:
        return to_rotate_option(overrides[rel_path])
    surveyed = survey_map.get(rel_path)
    if surveyed is None or surveyed["resolved"] == "review":
        return "review"
    return to_rotate_option(surveyed["resolved"])


def split_review(entries, survey_map, overrides):
    """처리 후보를 회전이 정해진 영상과 review(override 없음) 영상으로 나눈다. 정해진 항목에는 rotate 를 기록한다.

    입력: entries - 처리 후보 항목 list, survey_map/overrides - 회전 정보
    반환: (ready, review) 항목 list 2개(순서 유지)
    """
    ready, review = [], []
    for entry in entries:
        label = rotation_label(entry, survey_map, overrides)
        if label == "review":
            review.append(entry)
        else:
            entry["rotate"] = label
            ready.append(entry)
    return ready, review


def print_review_list(review, survey_map):
    """방향이 정해지지 않아(review, override 없음) 처리하지 않는 영상의 목록을 카운트와 함께 출력한다.

    입력: review - split_review 의 review 목록, survey_map - 조사 결과
    """
    if not review:
        return
    print("review(방향 미정, override 없음)라 처리하지 않는 영상: %d개 (faces 0/cw90/180/ccw90)" % len(review))
    for entry in review:
        surveyed = survey_map.get(entry["rel_path"])
        counts = "조사 결과 없음" if surveyed is None else "/".join(surveyed["counts"])
        print("  %s  %s" % (entry["rel_path"], counts))
    print("  육안으로 확인한 뒤 configs/video_orientation_overrides.csv 에 orientation 을 기록하면 다음 실행부터 처리합니다.")


def print_rotation_overview(targets, survey_map, overrides):
    """--list 용: 선택된 대상의 회전별 영상 수와 review 개수, override 적용 개수를 출력한다.

    입력: targets - 선택된 대상, survey_map/overrides - 회전 정보
    """
    tally = {}
    overridden = 0
    for entry in targets:
        label = rotation_label(entry, survey_map, overrides)
        tally[label] = tally.get(label, 0) + 1
        if entry["rel_path"] in overrides:
            overridden += 1
    print("회전별 영상 수(선택된 대상): none %d, cw90 %d, ccw90 %d, 180 %d | review(처리 안 함) %d | override 적용 %d" % (
        tally.get("none", 0), tally.get("cw90", 0), tally.get("ccw90", 0), tally.get("180", 0),
        tally.get("review", 0), overridden))


def parse_face_frames(output_bytes):
    """자식 출력에서 blink_video.py 가 영상 끝에 찍는 'not grabbed' 다음 줄의 숫자(얼굴이 검출된 프레임 수)를 읽는다.

    입력: output_bytes - 합쳐진 stdout/stderr 바이트
    반환: 숫자 문자열. 영상 끝까지 가지 못했거나 숫자를 찾지 못하면 빈 문자열
    """
    lines = output_bytes.decode("utf-8", "replace").split("\n")
    for index, line in enumerate(lines):
        if line.strip() == "not grabbed":
            for later in lines[index + 1:]:
                if later.strip() != "":
                    return later.strip() if later.strip().isdigit() else ""
    return ""


def check_manifest_header(manifest_path):
    """기존 매니페스트의 열 이름이 현재 도구와 다르면 이어 쓰지 않고 중단한다. 파일이 없으면 아무것도 하지 않는다.

    입력: manifest_path - batch_manifest.csv 경로
    """
    if not os.path.isfile(manifest_path):
        return
    header = None
    with open(manifest_path, "r", newline="", encoding="utf-8") as file:
        for line in file:
            if not line.startswith("#"):
                header = next(csv.reader([line]))
                break
    if header != MANIFEST_COLUMNS:
        sys.exit("기존 batch_manifest.csv 의 열 이름이 현재 도구와 다릅니다. 이어 쓰지 않고 중단합니다. "
                 "기존 파일을 다른 이름으로 옮기거나 BLINK_OUT_DIR 을 바꾸세요.")


def output_path_for(out_dir, entry):
    """영상의 최종 출력 txt 경로를 만든다(영상 파일 이름과 같은 규칙, 피험자 폴더 이름은 원본 그대로).

    입력: out_dir - 출력 루트, entry - 영상 항목
    반환: 경로 문자열. 예: <out>/Fold1_part1/06/10_1.txt
    """
    parts = entry["rel_path"].split("/")
    name = str(entry["class_label"])
    if entry["variant"]:
        name = name + "_" + entry["variant"]
    return os.path.join(out_dir, "Fold%d_part%d" % (entry["fold"], entry["part"]), parts[1], name + ".txt")


def read_status_map(manifest_path):
    """매니페스트를 읽어 영상별 마지막 상태를 돌려준다. 주석(#) 줄은 건너뛴다.

    입력: manifest_path - batch_manifest.csv 경로
    반환: {키 튜플: status}. 파일이 없으면 빈 dict
    """
    statuses = {}
    if not os.path.isfile(manifest_path):
        return statuses
    data_lines = []
    with open(manifest_path, "r", newline="", encoding="utf-8") as file:
        for line in file:
            if not line.startswith("#"):
                data_lines.append(line)
    for row in csv.DictReader(data_lines):
        key = (int(row["fold"]), int(row["part"]), int(row["subject_id"]), int(row["class_label"]), row["variant"])
        statuses[key] = row["status"]
    return statuses


def split_by_state(targets, out_dir, status_map):
    """대상을 완료(최종 출력 있음), 실패 기록, 미처리로 나눈다.

    입력: targets - 정렬된 대상, out_dir - 출력 루트, status_map - read_status_map 결과
    반환: (done, failed, pending) 항목 list 3개(순서 유지)
    """
    done, failed, pending = [], [], []
    for entry in targets:
        if os.path.exists(output_path_for(out_dir, entry)):
            done.append(entry)
        elif status_map.get(entry_key(entry)) in FAILURE_STATUSES:
            failed.append(entry)
        else:
            pending.append(entry)
    return done, failed, pending


def sum_frames(entries):
    """항목들의 frame_count 합을 구한다.

    입력: entries - 항목 list
    반환: 정수
    """
    total = 0
    for entry in entries:
        total += entry["frame_count"]
    return total


def print_first_ten(run_list):
    """이번 실행의 처리 순서 앞 10개를 출력한다.

    입력: run_list - 이번에 처리할 항목 list
    """
    print("처리 순서 앞 10개:")
    for number, entry in enumerate(run_list[:10], 1):
        print("  %2d. %s  (용량 %.1f MB, 프레임 %d)" % (number, entry["rel_path"], entry["size_bytes"] / 1e6, entry["frame_count"]))


def print_list(entries, flag_keys, targets, states, run_list, arguments):
    """--list 출력: 전체·대상·상태별 영상 수와 프레임 합, manual_flags 제외 확인, 앞 10개.

    입력: entries - 전체, flag_keys - manual_flags 키, targets - 선택된 대상,
          states - (done, failed, pending), run_list - 이번 처리 목록, arguments - 옵션
    """
    done, failed, pending = states
    flagged_all = 0
    for entry in entries:
        if is_flagged(entry, flag_keys):
            flagged_all += 1
    flagged_in_targets = 0
    for entry in targets:
        if is_flagged(entry, flag_keys):
            flagged_in_targets += 1
    print("CSV 전체: 영상 %d개, 프레임 합 %d" % (len(entries), sum_frames(entries)))
    print("manual_flags 영상: %d개 | 이번 선택에 포함된 manual_flags 영상: %d개" % (flagged_all, flagged_in_targets))
    print("선택된 대상: %d개, 프레임 합 %d" % (len(targets), sum_frames(targets)))
    print("상태: 완료(건너뜀) %d, 실패 기록 %d, 미처리 %d" % (len(done), len(failed), len(pending)))
    print("이번 처리 목록: %d개, 예상 총 프레임 %d%s" % (len(run_list), sum_frames(run_list),
          " (--retry-failed)" if arguments.retry_failed else ""))
    print_first_ten(run_list)


def get_tool_versions():
    """python, cv2, dlib, sklearn 버전을 깨끗한 별도 프로세스에서 읽어 한 줄 문자열로 돌려준다.

    반환: 'python=... cv2=... dlib=... sklearn=...' (읽지 못한 항목은 '?')
    """
    code = "import cv2, dlib, sklearn; print(cv2.__version__, dlib.__version__, sklearn.__version__)"
    versions = ["?", "?", "?"]
    try:
        result = subprocess.run([sys.executable, "-c", code], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        words = result.stdout.decode("utf-8", "replace").split()
        if result.returncode == 0 and len(words) == 3:
            versions = words
    except OSError:
        pass
    return "python=%s cv2=%s dlib=%s sklearn=%s" % (sys.version.split()[0], versions[0], versions[1], versions[2])


def append_manifest_row(manifest_path, values):
    """매니페스트에 한 줄을 추가한다. 파일이 새로 만들어질 때만 버전 주석 줄과 열 이름 줄을 먼저 쓴다.

    입력: manifest_path - batch_manifest.csv 경로, values - MANIFEST_COLUMNS 순서의 값 list
    """
    is_new = not os.path.exists(manifest_path)
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, lineterminator="\n")
        if is_new:
            file.write("# " + get_tool_versions() + "\n")
            writer.writerow(MANIFEST_COLUMNS)
        writer.writerow(values)


def judge_result(return_code, part_path, final_path):
    """자식 프로세스 결과를 판정하고, 성공이면 .part 를 최종 이름으로 바꾼다(덮어쓰지 않는다).

    입력: return_code - 종료 코드, part_path - 임시 출력, final_path - 최종 출력
    반환: (status, blink_rows). status 는 ok / failed / no_output
    """
    if return_code != 0:
        return "failed", 0
    if not os.path.exists(part_path):
        return "no_output", 0
    blink_rows = count_blink_rows(part_path)
    os.rename(part_path, final_path)
    return "ok", blink_rows


def last_output_line(output_bytes):
    """자식 프로세스 출력의 마지막 비어 있지 않은 줄을 돌려준다(실패 원인 표시용).

    입력: output_bytes - 합쳐진 stdout/stderr 바이트
    반환: 문자열(없으면 빈 문자열)
    """
    lines = output_bytes.decode("utf-8", "replace").split("\n")
    for line in reversed(lines):
        if line.strip() != "":
            return line.strip()
    return ""


def process_video(entry, out_dir):
    """영상 1개를 별도 프로세스로 처리한다. 작업 스레드에서 호출되며 공유 상태를 건드리지 않는다.

    입력: entry - 영상 항목, out_dir - 출력 루트
    반환: dict(entry, status, return_code, elapsed_s, blink_rows, started, ended, message, rotate, face_frames)
    """
    final_path = output_path_for(out_dir, entry)
    part_path = final_path + ".part"
    started = datetime.now()
    clock = time.time()
    return_code, message, status, blink_rows, face_frames = -1, "", "failed", 0, ""
    try:
        os.makedirs(os.path.dirname(final_path), exist_ok=True)
        if os.path.exists(part_path):
            os.remove(part_path)     # 이전 실행이 끊겨 남은 미완료 파일만 지운다
        environment = dict(os.environ)
        environment["PYTHONIOENCODING"] = "utf-8"
        command = [sys.executable, HEADLESS_SCRIPT, entry["rel_path"], part_path, "--rotate", entry["rotate"]]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=environment)
        return_code = result.returncode
        face_frames = parse_face_frames(result.stdout)
        status, blink_rows = judge_result(return_code, part_path, final_path)
        message = last_output_line(result.stdout) if status != "ok" else ""
    except OSError as error:
        message = "OSError: %s" % type(error).__name__
    return {"entry": entry, "status": status, "return_code": return_code, "blink_rows": blink_rows,
            "elapsed_s": time.time() - clock, "started": started.strftime(TIME_FORMAT),
            "ended": datetime.now().strftime(TIME_FORMAT), "message": message,
            "rotate": entry["rotate"], "face_frames": face_frames}


def manifest_values(result):
    """process_video 결과를 매니페스트 한 줄(MANIFEST_COLUMNS 순서)로 바꾼다.

    입력: result - process_video 반환 dict
    반환: 값 list
    """
    entry = result["entry"]
    return [entry["fold"], entry["part"], entry["subject_id"], entry["class_label"], entry["variant"],
            result["status"], result["return_code"], "%.1f" % result["elapsed_s"], result["blink_rows"],
            result["started"], result["ended"], result["rotate"], result["face_frames"]]


def print_progress(number, total, result):
    """영상 1개가 끝날 때 한 줄 진행 로그를 출력한다.

    입력: number - 완료 순번, total - 이번 처리 총수, result - process_video 반환 dict
    """
    entry = result["entry"]
    line = "[%d/%d] %s %s rotate=%s (소요 %.1f초, blink %d행, face_frames=%s)" % (
        number, total, entry["rel_path"], result["status"], result["rotate"], result["elapsed_s"],
        result["blink_rows"], result["face_frames"] or "-")
    if result["status"] != "ok":
        line += " 종료코드 %s: %s" % (result["return_code"], result["message"])
    print(line)
    sys.stdout.flush()


def run_batch(run_list, skipped_count, out_dir, manifest_path, workers):
    """이번 처리 목록을 작업 스레드 풀로 실행하고(영상마다 별도 프로세스), 끝날 때마다 기록한다.

    입력: run_list - 처리할 항목, skipped_count - 건너뛴 영상 수, out_dir - 출력 루트,
          manifest_path - 매니페스트 경로, workers - 동시 처리 수
    반환: 종료 코드(실패가 있으면 1)
    """
    began = time.time()
    counts = {"ok": 0, "fail": 0}
    print("건너뜀(최종 출력 있음): %d개 | 이번 처리: %d개 (workers=%d)" % (skipped_count, len(run_list), workers))
    executor = ThreadPoolExecutor(max_workers=workers)
    futures = []
    for entry in run_list:
        futures.append(executor.submit(process_video, entry, out_dir))
    try:
        for number, future in enumerate(as_completed(futures), 1):
            result = future.result()
            append_manifest_row(manifest_path, manifest_values(result))
            counts["ok" if result["status"] == "ok" else "fail"] += 1
            print_progress(number, len(run_list), result)
    except KeyboardInterrupt:
        for future in futures:
            future.cancel()
        print("중단됨: 대기 중인 영상은 취소했습니다. 남은 .part 파일은 다음 실행에서 정리됩니다.")
    executor.shutdown(wait=False)
    print("요약: 성공 %d, 실패 %d, 건너뜀 %d, 총 소요 %.1f초" % (counts["ok"], counts["fail"], skipped_count,
                                                          time.time() - began))
    return 1 if counts["fail"] > 0 else 0


def main():
    """전체 흐름: 환경 확인, 대상 선택, 상태 분류, --list 출력 또는 일괄 실행."""
    arguments = parse_arguments()
    out_dir = require_output_dir()
    entries = read_inventory(INVENTORY_PATH)
    flag_keys = load_flag_keys(FLAGS_PATH)
    targets = select_targets(entries, flag_keys, arguments)
    manifest_path = os.path.join(out_dir, MANIFEST_NAME)
    check_manifest_header(manifest_path)
    survey_map = load_survey_orientations(SURVEY_PATH)
    overrides = load_overrides(OVERRIDES_PATH)
    done, failed, pending = split_by_state(targets, out_dir, read_status_map(manifest_path))
    ready, review = split_review(failed if arguments.retry_failed else pending, survey_map, overrides)
    run_list = ready if arguments.limit is None else ready[:arguments.limit]
    if arguments.list_only:
        print_list(entries, flag_keys, targets, (done, failed, pending), run_list, arguments)
        print_rotation_overview(targets, survey_map, overrides)
        print_review_list(review, survey_map)
        return 0
    require_run_environment()
    print_review_list(review, survey_map)
    return run_batch(run_list, len(done), out_dir, manifest_path, arguments.workers)


if __name__ == "__main__":
    sys.exit(main())
