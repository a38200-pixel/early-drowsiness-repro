# compare_with_author_npy.py
# 역할: 우리 시퀀스 산출물(SEQ_OUT_DIR 의 npy, metadata)과 저자 원본 테스트 .npy 를 읽기 전용으로 비교한다.
#       (작업 0) 우리 산출물 정합성 점검, (작업 1) 저자 테스트 y 의 라벨 런으로 영상 경계 복원과 start_indices 대조,
#       (작업 2) (fold, 피험자 순서, 클래스 0/5/10 순서)의 위치 대응으로 영상별 시퀀스 수·값 비교, 회전별 집계, 동일 블록 쌍 검사.
# 사용법: 저장소 루트에서 conda 환경 ghoddoosian_tf1 의 python 으로 실행한다.
#           python src/repro/compare_with_author_npy.py --check-only   (표시만, 아무 파일도 쓰지 않음)
#           python src/repro/compare_with_author_npy.py                (SEQ_OUT_DIR/compare/ 에 표와 CSV 를 새로 씀)
#         옵션: --author-dir <폴더> (기본 third_party/ghoddoosian, 읽기 전용)
# 환경변수: SEQ_OUT_DIR (우리 산출물 루트, 저장소 밖), BLINK_OUT_DIR (배치 manifest 의 rotate 열을 읽기 위한 입력 루트, 읽기 전용)
# 규칙: 저자 npy 는 mmap_mode='r' 로만 연다. 정렬 규칙을 바꿔 일치율을 올리지 않는다(어긋나는 위치만 보고하고 재정렬하지 않는다).
#       compare 폴더가 이미 있으면 덮어쓰지 않고 중단한다.
"""저자 .npy 대비 비교 도구(Python 3.6 호환)."""
import argparse
import csv
import hashlib
import json
import os
import re
import sys

import numpy as np

from extract_blinks_batch import MANIFEST_NAME
from reproduction_build_sequences import REPO, is_inside, read_inventory_rows

DEFAULT_AUTHOR_DIR = os.path.join(REPO, "third_party", "ghoddoosian")
FOLDS = (1, 2, 3, 4, 5)
CLASS_ORDER = (0, 5, 10)
EXPECTED_VIDEOS = {1: 33, 2: 36, 3: 36, 4: 36, 5: 33}
ALIGN_COLUMNS = ["fold", "position", "author_start", "author_n_seq", "author_label", "our_video_id", "our_class",
                 "status", "rotation", "our_n_seq", "count_equal", "allclose", "max_abs_diff", "y_equal"]
DUP_COLUMNS = ["n_seq", "fold_a", "position_a", "our_id_a", "fold_b", "position_b", "our_id_b"]


def parse_arguments():
    """명령줄 옵션을 해석한다.

    반환: argparse.Namespace (check_only, author_dir)
    """
    parser = argparse.ArgumentParser(description="저자 .npy 대비 비교(읽기 전용)")
    parser.add_argument("--check-only", action="store_true", dest="check_only", help="아무 파일도 쓰지 않고 결과만 출력")
    parser.add_argument("--author-dir", default=DEFAULT_AUTHOR_DIR, dest="author_dir", help="저자 .npy 폴더(읽기 전용)")
    return parser.parse_args()


def require_dirs():
    """SEQ_OUT_DIR, BLINK_OUT_DIR 를 읽고 검사한다. 없거나 SEQ_OUT_DIR 가 저장소 안이면 중단한다.

    반환: (seq_dir, blink_dir) 실제 경로 튜플
    """
    seq_raw, blink_raw = os.environ.get("SEQ_OUT_DIR"), os.environ.get("BLINK_OUT_DIR")
    if not seq_raw or not blink_raw:
        sys.exit("환경변수 SEQ_OUT_DIR 와 BLINK_OUT_DIR 를 설정하세요.")
    seq_dir, blink_dir = os.path.realpath(seq_raw), os.path.realpath(blink_raw)
    if is_inside(seq_dir, REPO):
        sys.exit("SEQ_OUT_DIR 이 저장소 폴더 안입니다.")
    if not os.path.isdir(os.path.join(seq_dir, "npy")):
        sys.exit("SEQ_OUT_DIR 에 npy 폴더가 없습니다.")
    return seq_dir, blink_dir


def read_csv_rows(path):
    """CSV 를 dict list 로 읽는다(BOM 허용).

    입력: path - CSV 경로
    반환: dict list
    """
    with open(path, "r", newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def load_npy(path):
    """npy 를 읽기 전용 mmap 으로 연다.

    입력: path - npy 경로
    반환: numpy 배열(mmap, 읽기 전용)
    """
    return np.load(path, mmap_mode="r")


def check_our_fold(seq_dir, fold):
    """우리 산출물 한 fold 의 shape, dtype, 메타데이터 행 수, start_indices 정합성을 점검한다.

    입력: seq_dir - 산출물 루트, fold - fold 번호
    반환: (보고 문자열, 통과 여부)
    """
    npy_dir, meta_dir = os.path.join(seq_dir, "npy"), os.path.join(seq_dir, "metadata")
    arrays = {}
    for name in ("Blinks", "Labels", "BlinksTest", "LabelsTest"):
        arrays[name] = load_npy(os.path.join(npy_dir, "%s_30_Fold%d.npy" % (name, fold)))
    train_rows = len(read_csv_rows(os.path.join(meta_dir, "sequence_metadata_fold%d_train.csv" % fold)))
    test_meta = read_csv_rows(os.path.join(meta_dir, "sequence_metadata_fold%d_test.csv" % fold))
    with open(os.path.join(meta_dir, "start_indices_fold%d.json" % fold), "r", encoding="utf-8") as file:
        starts = json.load(file)
    n_test = len(arrays["BlinksTest"])
    lengths = [b - a for a, b in zip(starts, starts[1:] + [n_test])]
    videos = len(set(row["video_id"] for row in test_meta))
    checks = [len(arrays["Blinks"]) == len(arrays["Labels"]) == train_rows, n_test == len(arrays["LabelsTest"]) == len(test_meta),
              starts[0] == 0 and min(lengths) >= 1 and sum(lengths) == n_test, len(starts) == videos,
              all(a.dtype == np.float64 for a in arrays.values()), arrays["Blinks"].shape[1:] == (30, 4)]
    text = "Fold%d: train %s/%s 메타 %d, test %s/%s 메타 %d, start_indices %d개 합 %d, 영상 %d, dtype %s -> %s" % (
        fold, arrays["Blinks"].shape, arrays["Labels"].shape, train_rows, arrays["BlinksTest"].shape,
        arrays["LabelsTest"].shape, len(test_meta), len(starts), sum(lengths), videos, arrays["Blinks"].dtype,
        "OK" if all(checks) else "불일치 %s" % checks)
    return text, all(checks)


def parse_training_start_indices(path):
    """Training.py 를 텍스트로 읽어 fold 별 start_indices(주석 아닌 줄)를 꺼낸다(코드를 실행하지 않는다).

    입력: path - Training.py 경로
    반환: {fold 번호(1부터): 정수 list}
    """
    result, current = {}, None
    with open(path, "r", encoding="utf-8", errors="replace") as file:
        for line in file.read().split("\n")[439:499]:
            match = re.match(r"\s*if i==(\d):", line)
            if match:
                current = int(match.group(1)) + 1
                result[current] = []
            elif current is not None and line.strip().startswith("start_indices="):
                result[current] = [int(v) for v in re.findall(r"\d+", line.split("#")[0].split("=", 1)[1])]
            elif current is not None and re.match(r"\s+\d[\d, ]*\]", line):
                result[current].extend(int(v) for v in re.findall(r"\d+", line.split("#")[0]))
    return result


def label_runs(labels):
    """y 를 라벨이 같은 연속 구간(런)으로 나눈다.

    입력: labels - (N,1) 라벨 배열
    반환: (시작 위치, 길이, 라벨) 튜플 list
    """
    flat = np.asarray(labels).reshape(-1)
    runs, start = [], 0
    for index in range(1, len(flat) + 1):
        if index == len(flat) or flat[index] != flat[start]:
            runs.append((start, index - start, int(flat[start])))
            start = index
    return runs


def check_author_structure(fold, runs, expected_starts):
    """저자 한 fold 의 런 구조를 (0,5,10) 반복, 영상 수, start_indices 일치로 검사한다.

    입력: fold - fold 번호, runs - label_runs 결과, expected_starts - Training.py 의 start_indices
    반환: (보고 문자열, 통과 여부)
    """
    labels = [run[2] for run in runs]
    pattern = labels == [CLASS_ORDER[i % 3] for i in range(len(labels))]
    starts = [run[0] for run in runs]
    same_count = len(runs) == EXPECTED_VIDEOS[fold] == len(expected_starts)
    same_starts = starts == list(expected_starts)
    text = "Fold%d 저자: 런 %d개(기대 %d, Training.py %d개), (0,5,10) 반복 %s, 경계 == Training.py start_indices %s" % (
        fold, len(runs), EXPECTED_VIDEOS[fold], len(expected_starts), pattern, same_starts)
    return text, pattern and same_count and same_starts


def our_slots(fold, inventory, summary):
    """우리 쪽 한 fold 의 영상 칸(피험자 (part, id) 오름차순 x 클래스 0/5/10)을 만든다.

    입력: fold - fold 번호, inventory - read_inventory_rows 결과, summary - video_summary 행의 {video_id: dict}
    반환: dict(video_id, class_label, reason, n_seq, key) list. 변형이 있으면 video_id 를 '|' 로 잇는다
    """
    subjects = sorted(set((v["part"], v["subject"]) for v in inventory if v["fold"] == fold))
    slots = []
    for part, subject in subjects:
        for label in CLASS_ORDER:
            videos = [v for v in inventory if v["fold"] == fold and v["part"] == part and v["subject"] == subject
                      and v["class_label"] == label]
            rows = [summary[v["video_id"]] for v in videos]
            reason = "|".join(sorted(set(r["reason"] for r in rows if r["reason"])))
            slots.append({"video_id": "|".join(v["video_id"] for v in videos), "class_label": label, "reason": reason,
                          "n_seq": int(rows[0]["n_sequences"]) if len(rows) == 1 else 0,
                          "key": (fold, part, int(subject), label, videos[0]["variant"])})
    return slots


def our_block_offsets(seq_dir, fold):
    """우리 테스트 메타데이터에서 영상별 (시작 위치, 시퀀스 수)를 구한다.

    입력: seq_dir - 산출물 루트, fold - fold 번호
    반환: {video_id: (시작, 개수)}
    """
    rows = read_csv_rows(os.path.join(seq_dir, "metadata", "sequence_metadata_fold%d_test.csv" % fold))
    offsets = {}
    for row in rows:
        start, count = offsets.get(row["video_id"], (int(row["seq_index"]), 0))
        offsets[row["video_id"]] = (start, count + 1)
    return offsets


def read_rotations(blink_dir):
    """배치 manifest 에서 영상별 적용 회전(rotate 열, 같은 영상이 여러 줄이면 마지막 줄)을 읽는다.

    입력: blink_dir - 입력 루트
    반환: {(fold, part, 피험자 정수, 클래스, 변형): 회전 문자열}
    """
    result = {}
    with open(os.path.join(blink_dir, MANIFEST_NAME), "r", newline="", encoding="utf-8-sig") as file:
        lines = [line for line in file if not line.startswith("#")]
    for row in csv.DictReader(lines):
        if row["fold"] == "fold":
            continue
        key = (int(row["fold"]), int(row["part"]), int(row["subject_id"]), int(row["class_label"]), row["variant"])
        result[key] = row["rotate"] or "none"
    return result


def compare_blocks(author_x, author_y, our_x, our_y):
    """저자 영상 블록과 우리 영상 블록을 비교한다(시퀀스 수가 같을 때만 값 비교).

    입력: author_x, author_y, our_x, our_y - 영상 하나의 X, y 블록
    반환: dict(count_equal, allclose, max_abs_diff, y_equal)
    """
    result = {"count_equal": len(author_x) == len(our_x), "allclose": "", "max_abs_diff": "", "y_equal": ""}
    if result["count_equal"]:
        result["allclose"] = bool(np.allclose(author_x, our_x, atol=1e-6))
        result["max_abs_diff"] = "%.6g" % float(np.max(np.abs(np.asarray(author_x) - np.asarray(our_x))))
        result["y_equal"] = bool(np.array_equal(author_y, our_y))
    return result


def align_one(fold, position, run, slot, author, ours, rotations):
    """위치 하나(저자 런, 우리 칸)의 대응 행을 만든다. 라벨이 다르면 값을 비교하지 않는다.

    입력: run - 저자 (시작, 길이, 라벨) 또는 None, slot - 우리 칸 또는 None, author - (X, y), ours - (X, y, offsets),
          rotations - read_rotations 결과
    반환: ALIGN_COLUMNS 의 dict
    """
    row = dict((name, "") for name in ALIGN_COLUMNS)
    row.update({"fold": fold, "position": position})
    if run is not None:
        row.update({"author_start": run[0], "author_n_seq": run[1], "author_label": run[2]})
    if slot is not None:
        row.update({"our_video_id": slot["video_id"], "our_class": slot["class_label"], "our_n_seq": slot["n_seq"],
                    "rotation": rotations.get(slot["key"], "unknown")})
    if run is None or slot is None:
        row["status"] = "ours_only" if run is None else "author_only"
    elif slot["reason"]:
        row["status"] = "ours_excluded:" + slot["reason"]
    elif run[2] != slot["class_label"]:
        row["status"] = "label_mismatch"
    else:
        start, count = ours[2][slot["video_id"]]
        row.update(compare_blocks(author[0][run[0]:run[0] + run[1]], author[1][run[0]:run[0] + run[1]],
                                  ours[0][start:start + count], ours[1][start:start + count]))
        row["status"] = "compared"
    return row


def align_fold(fold, runs, author, ours, slots, rotations):
    """한 fold 의 저자 런과 우리 칸을 위치로 대응시켜 행 list 를 만든다(재정렬하지 않는다).

    입력: runs - 저자 런, author - (X, y), ours - (X, y, offsets), slots - 우리 칸 list, rotations - 회전 사전
    반환: ALIGN_COLUMNS 의 dict list
    """
    rows = []
    for position in range(max(len(runs), len(slots))):
        run = runs[position] if position < len(runs) else None
        slot = slots[position] if position < len(slots) else None
        rows.append(align_one(fold, position, run, slot, author, ours, rotations))
    return rows


def first_positions(rows):
    """한 fold 의 대응 행에서 처음 어긋나는 위치(라벨 불일치, 제외 칸, 시퀀스 수 불일치)를 찾는다.

    입력: rows - align_fold 결과
    반환: 보고 문자열
    """
    names = (("label_mismatch", lambda r: r["status"] == "label_mismatch"),
             ("우리 제외 칸", lambda r: r["status"].startswith("ours_excluded")),
             ("시퀀스 수 불일치", lambda r: r["status"] == "compared" and not r["count_equal"]),
             ("한쪽에만 있음", lambda r: r["status"] in ("author_only", "ours_only")))
    parts = []
    for name, test in names:
        found = [r for r in rows if test(r)]
        parts.append("%s %s" % (name, "없음" if not found else "첫 위치 %d(%s)" % (found[0]["position"], found[0]["our_video_id"])))
    return "; ".join(parts)


def rate_text(compared, name):
    """비교된 행 묶음의 count_equal, allclose 비율 문자열을 만든다.

    입력: compared - status 가 compared 인 행 list, name - 묶음 이름
    반환: 문자열
    """
    equal = [r for r in compared if r["count_equal"]]
    close = [r for r in equal if r["allclose"]]
    return "%s: 비교 %d, count_equal %d(%.3f), allclose %d(count_equal 중 %.3f)" % (
        name, len(compared), len(equal), len(equal) / float(max(len(compared), 1)), len(close),
        len(close) / float(max(len(equal), 1)))


def rotation_lines(rows):
    """회전별과 전체 count_equal, allclose 비율 줄을 만든다.

    입력: rows - 모든 fold 의 대응 행
    반환: 문자열 list
    """
    compared = [r for r in rows if r["status"] == "compared"]
    lines = [rate_text(compared, "전체")]
    for rotation in sorted(set(r["rotation"] for r in compared)):
        lines.append(rate_text([r for r in compared if r["rotation"] == rotation], "회전 " + rotation))
    return lines


def find_duplicates(blocks, labels_by_block):
    """저자 테스트 배열 전체에서 X 블록이 np.array_equal 로 같은 서로 다른 영상 쌍을 찾는다.

    입력: blocks - [(fold, position, X 블록)], labels_by_block - {(fold, position): 우리 영상 ID}
    반환: DUP_COLUMNS 의 dict list
    """
    groups = {}
    for fold, position, block in blocks:
        digest = hashlib.sha1(np.ascontiguousarray(block).tobytes()).hexdigest()
        groups.setdefault((block.shape, digest), []).append((fold, position, block))
    pairs = []
    for members in groups.values():
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a, b = members[i], members[j]
                if np.array_equal(a[2], b[2]):
                    pairs.append({"n_seq": a[2].shape[0], "fold_a": a[0], "position_a": a[1],
                                  "our_id_a": labels_by_block.get((a[0], a[1]), ""), "fold_b": b[0], "position_b": b[1],
                                  "our_id_b": labels_by_block.get((b[0], b[1]), "")})
    return pairs


def write_csv(path, columns, rows):
    """CSV 를 새 파일로만 쓴다(이미 있으면 예외).

    입력: path - 경로, columns - 열 이름 list, rows - dict list
    """
    with open(path, "x", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_author_fold(author_dir, fold):
    """저자 테스트 X, y 를 읽기 전용으로 연다.

    입력: author_dir - 저자 npy 폴더, fold - fold 번호
    반환: (X, y)
    """
    return (load_npy(os.path.join(author_dir, "BlinksTest_30_Fold%d.npy" % fold)),
            load_npy(os.path.join(author_dir, "LabelsTest_30_Fold%d.npy" % fold)))


def report_author_arrays(authors):
    """저자 테스트 배열의 이름, shape, dtype 줄을 만든다.

    입력: authors - {fold: (X, y)}
    반환: 문자열 list
    """
    lines = []
    for fold in FOLDS:
        x_array, y_array = authors[fold]
        lines.append("BlinksTest_30_Fold%d.npy %s %s / LabelsTest_30_Fold%d.npy %s %s" % (
            fold, x_array.shape, x_array.dtype, fold, y_array.shape, y_array.dtype))
    return lines


def run_structure(authors, expected):
    """모든 fold 의 저자 런 구조를 검사한다.

    입력: authors - {fold: (X, y)}, expected - Training.py 의 start_indices 사전
    반환: (보고 줄 list, 통과 여부, {fold: runs})
    """
    lines, passed, all_runs = [], True, {}
    for fold in FOLDS:
        all_runs[fold] = label_runs(authors[fold][1])
        text, ok = check_author_structure(fold, all_runs[fold], expected[fold])
        lines.append(text)
        passed = passed and ok
    return lines, passed, all_runs


def run_alignment(seq_dir, blink_dir, authors, all_runs):
    """모든 fold 를 위치 대응으로 비교한다.

    입력: authors - 저자 배열 사전, all_runs - 저자 런 사전
    반환: (대응 행 list, 보고 줄 list)
    """
    inventory = read_inventory_rows(os.path.join(REPO, "configs", "video_inventory.csv"))
    summary = dict((r["video_id"], r) for r in read_csv_rows(os.path.join(seq_dir, "metadata", "video_summary.csv")))
    rotations = read_rotations(blink_dir)
    rows, lines = [], []
    for fold in FOLDS:
        ours = (load_npy(os.path.join(seq_dir, "npy", "BlinksTest_30_Fold%d.npy" % fold)),
                load_npy(os.path.join(seq_dir, "npy", "LabelsTest_30_Fold%d.npy" % fold)), our_block_offsets(seq_dir, fold))
        slots = our_slots(fold, inventory, summary)
        fold_rows = align_fold(fold, all_runs[fold], authors[fold], ours, slots, rotations)
        lines.append("Fold%d: 저자 영상 %d, 우리 칸 %d(피험자 %d명) | %s" % (
            fold, len(all_runs[fold]), len(slots), len(slots) // 3, first_positions(fold_rows)))
        rows.extend(fold_rows)
    return rows, lines


def run_duplicates(authors, all_runs, rows):
    """저자 테스트 영상 블록 전체에서 동일 블록 쌍을 찾는다.

    입력: authors - 저자 배열 사전, all_runs - 저자 런 사전, rows - 대응 행(영상 ID 표시용)
    반환: DUP_COLUMNS 의 dict list
    """
    blocks, names = [], {}
    for row in rows:
        names[(row["fold"], row["position"])] = row["our_video_id"]
    for fold in FOLDS:
        for position, run in enumerate(all_runs[fold]):
            blocks.append((fold, position, np.asarray(authors[fold][0][run[0]:run[0] + run[1]])))
    return find_duplicates(blocks, names)


def main():
    """전체 흐름: 작업 0 점검, 저자 구조 검사(불일치면 중단), 위치 대응 비교, 동일 블록 검사, (쓰기 모드면) compare 폴더에 저장."""
    arguments = parse_arguments()
    seq_dir, blink_dir = require_dirs()
    print("== 작업 0: 우리 산출물 ==")
    for fold in FOLDS:
        print(check_our_fold(seq_dir, fold)[0])
    authors = dict((fold, load_author_fold(arguments.author_dir, fold)) for fold in FOLDS)
    print("== 작업 1: 저자 테스트 배열 ==")
    print("\n".join(report_author_arrays(authors)))
    expected = parse_training_start_indices(os.path.join(REPO, "third_party", "ghoddoosian", "Training.py"))
    lines, passed, all_runs = run_structure(authors, expected)
    print("\n".join(lines))
    if not passed:
        sys.exit("저자 구조가 start_indices 와 맞지 않아 이후 비교를 멈춥니다.")
    print("== 작업 2: 위치 대응 비교 ==")
    rows, align_lines = run_alignment(seq_dir, blink_dir, authors, all_runs)
    duplicates = run_duplicates(authors, all_runs, rows)
    report = align_lines + rotation_lines(rows) + ["저자 테스트 배열의 동일 X 블록 쌍: %d개" % len(duplicates)]
    print("\n".join(report))
    if arguments.check_only:
        print("--check-only: 아무 파일도 쓰지 않았습니다.")
        return
    compare_dir = os.path.join(seq_dir, "compare")
    os.makedirs(compare_dir, exist_ok=False)
    write_csv(os.path.join(compare_dir, "author_vs_ours.csv"), ALIGN_COLUMNS, rows)
    write_csv(os.path.join(compare_dir, "duplicate_pairs.csv"), DUP_COLUMNS, duplicates)
    with open(os.path.join(compare_dir, "summary.txt"), "x", encoding="utf-8") as file:
        file.write("\n".join(lines + report) + "\n")
    print("compare 폴더에 썼습니다.")


if __name__ == "__main__":
    main()
