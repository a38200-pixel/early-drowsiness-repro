# extract_blinks_batch.py
# 역할: configs/video_inventory.csv 의 영상을 reproduction_blink_headless.py 로 일괄 추출하는 관리 도구.
#       영상 1개마다 별도 프로세스를 띄워 전역 상태를 격리하고, 재시작·실패 기록·진행 로그·매니페스트를 관리한다.
# 사용법: conda 환경 ghoddoosian_tf1 을 활성화하고 저장소 루트에서 실행한다.
#           python src/repro/extract_blinks_batch.py --list                  (실행하지 않고 대상과 순서만 출력)
#           python src/repro/extract_blinks_batch.py --limit 3 --workers 1   (앞 3개만 시험)
#           python src/repro/extract_blinks_batch.py --workers 4             (전체)
#         옵션: --workers N, --list, --only-fold F, --limit K, --flagged-only, --retry-failed
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
"""
import argparse
import csv
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from reproduction_blink_headless import count_blink_rows

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HEADLESS_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reproduction_blink_headless.py")
INVENTORY_PATH = os.path.join(REPO, "configs", "video_inventory.csv")
FLAGS_PATH = os.path.join(REPO, "configs", "video_manual_flags.csv")
MANIFEST_NAME = "batch_manifest.csv"
MANIFEST_COLUMNS = ["fold", "part", "subject_id", "class_label", "variant", "status", "return_code",
                    "elapsed_s", "blink_rows", "started", "ended"]
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
    반환: dict(entry, status, return_code, elapsed_s, blink_rows, started, ended, message)
    """
    final_path = output_path_for(out_dir, entry)
    part_path = final_path + ".part"
    started = datetime.now()
    clock = time.time()
    return_code, message, status, blink_rows = -1, "", "failed", 0
    try:
        os.makedirs(os.path.dirname(final_path), exist_ok=True)
        if os.path.exists(part_path):
            os.remove(part_path)     # 이전 실행이 끊겨 남은 미완료 파일만 지운다
        environment = dict(os.environ)
        environment["PYTHONIOENCODING"] = "utf-8"
        command = [sys.executable, HEADLESS_SCRIPT, entry["rel_path"], part_path]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=environment)
        return_code = result.returncode
        status, blink_rows = judge_result(return_code, part_path, final_path)
        message = last_output_line(result.stdout) if status != "ok" else ""
    except OSError as error:
        message = "OSError: %s" % type(error).__name__
    return {"entry": entry, "status": status, "return_code": return_code, "blink_rows": blink_rows,
            "elapsed_s": time.time() - clock, "started": started.strftime(TIME_FORMAT),
            "ended": datetime.now().strftime(TIME_FORMAT), "message": message}


def manifest_values(result):
    """process_video 결과를 매니페스트 한 줄(MANIFEST_COLUMNS 순서)로 바꾼다.

    입력: result - process_video 반환 dict
    반환: 값 list
    """
    entry = result["entry"]
    return [entry["fold"], entry["part"], entry["subject_id"], entry["class_label"], entry["variant"],
            result["status"], result["return_code"], "%.1f" % result["elapsed_s"], result["blink_rows"],
            result["started"], result["ended"]]


def print_progress(number, total, result):
    """영상 1개가 끝날 때 한 줄 진행 로그를 출력한다.

    입력: number - 완료 순번, total - 이번 처리 총수, result - process_video 반환 dict
    """
    entry = result["entry"]
    line = "[%d/%d] %s %s (소요 %.1f초, blink %d행)" % (number, total, entry["rel_path"], result["status"],
                                                        result["elapsed_s"], result["blink_rows"])
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
    done, failed, pending = split_by_state(targets, out_dir, read_status_map(manifest_path))
    run_list = failed if arguments.retry_failed else pending
    if arguments.limit is not None:
        run_list = run_list[:arguments.limit]
    if arguments.list_only:
        print_list(entries, flag_keys, targets, (done, failed, pending), run_list, arguments)
        return 0
    require_run_environment()
    return run_batch(run_list, len(done), out_dir, manifest_path, arguments.workers)


if __name__ == "__main__":
    sys.exit(main())
