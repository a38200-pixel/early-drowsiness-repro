# reproduction_build_sequences.py
# 역할: 우리가 재추출한 blink txt 에서 저자 Preprocessing.py(PP)와 같은 계산(피험자별 정규화, 길이 30·폭 2 윈도우)으로
#       .npy 시퀀스(Blinks_30_Fold<k>.npy 등 20개)와 메타데이터를 만든다. 저자 코드는 exec 하지 않고 같은 계산을 직접 구현한다.
# 사용법: 저장소 루트에서 conda 환경 ghoddoosian_tf1 의 python 으로 실행한다.
#           python src/repro/reproduction_build_sequences.py --check-only   (입력만 읽고 coverage 와 제외 목록을 출력, 파일을 쓰지 않음)
#           python src/repro/reproduction_build_sequences.py                (SEQ_OUT_DIR 에 산출물을 쓴다)
#         옵션: --inventory <CSV>  (기본 configs/video_inventory.csv)
# 환경변수: BLINK_OUT_DIR (입력 루트, 읽기 전용, 필수), SEQ_OUT_DIR (출력 루트, 저장소 밖, 기본값 없음, --check-only 에서는 불필요)
# 입력: <BLINK_OUT_DIR>/Fold<N>_part<M>/<피험자>/<클래스>[_<변형>].txt (5열: 누적 blink 수, 100x빈도, 진폭, 지속 시간, 속도, 첫 줄은 빈 줄)
# 출력(SEQ_OUT_DIR): npy/ 20개, metadata/ (sequence_metadata_fold<k>_<train|test>.csv, video_summary.csv, exclusions.csv,
#       start_indices_fold<k>.json, coverage.txt). 이미 있는 출력 파일은 덮어쓰지 않고 중단한다.
# PP 와 같은 점: 쓰는 열은 txt 의 1~4열(0열은 쓰지 않음), 기준 통계는 alert 의 마지막 n//3 개(n<3 이면 전체), 표준편차 0이면 전체 표준편차로 대체,
#       alert 는 앞쪽 n-n//3 개만 데이터로 쓰고 나머지는 전부 쓴다, n<=30 이면 앞쪽 0 패딩 시퀀스 1개, 아니면 (n-30)//2+1 개.
# PP 와 다른 점: 제외 정책 5가지(영상별 사유 기록, 앞 피험자 값으로 대체하지 않음. degenerate_reference 는 저자 코드가
#       예외를 내거나 비유한 값을 만드는 경우로 한정하고, n=2~5 와 표준편차 0 대체는 저자 동작을 따르며 author_fallback_used 에 기록),
#       메타데이터와 start_indices 출력,
#       훈련 세트는 fold 번호를 시드로 고정해 섞음(PP 는 시드 없음), 변형 파일(10_1 등)은 이번 단계에서 보류.
"""blink txt 에서 시퀀스 .npy 를 만드는 스크립트(Python 3.6 호환)."""
import argparse
import csv
import json
import os
import sys

import numpy as np

from extract_blinks_batch import REPO, is_inside

WINDOW_SIZE = 30
STRIDE = 2
CLASS_ORDER = (0, 5, 10)
FEATURE_COLUMNS = (1, 2, 3, 4)
REF_SMALL_LIMIT = 10
INVENTORY_PATH = os.path.join(REPO, "configs", "video_inventory.csv")
SEQUENCE_COLUMNS = ["seq_index", "fold", "split", "subject_id", "class_label", "variant", "video_id", "seq_in_video",
                    "n_blinks_video", "n_blinks_used", "start_blink", "end_blink", "padded"]
SUMMARY_COLUMNS = ["video_id", "subject_id", "class_label", "status", "reason", "n_blink_rows", "stats_blink_range",
                   "ref_n", "ref_small", "author_fallback_used", "norm_mean_json", "norm_std_json", "n_sequences"]
EXCLUSION_COLUMNS = ["video_id", "subject_id", "class_label", "level", "reason", "detail"]
REASONS = ("variant_deferred", "not_extracted", "no_blink_file", "no_alert_reference", "degenerate_reference")
SUBJECT_LEVEL_REASONS = ("no_alert_reference", "degenerate_reference")


def parse_arguments():
    """명령줄 옵션을 해석한다.

    반환: argparse.Namespace (check_only, inventory)
    """
    parser = argparse.ArgumentParser(description="blink txt 에서 시퀀스 npy 와 메타데이터 만들기")
    parser.add_argument("--check-only", action="store_true", dest="check_only",
                        help="입력만 읽고 coverage 와 제외 목록을 출력한다(아무 파일도 쓰지 않음)")
    parser.add_argument("--inventory", default=INVENTORY_PATH, help="영상 인벤토리 CSV(기본 configs/video_inventory.csv)")
    return parser.parse_args()


def require_blink_dir():
    """BLINK_OUT_DIR(입력 루트)를 읽는다. 없거나 폴더가 아니면 안내 후 중단한다.

    반환: 입력 루트의 실제 경로
    """
    raw_value = os.environ.get("BLINK_OUT_DIR")
    if not raw_value:
        sys.exit("환경변수 BLINK_OUT_DIR 를 입력 루트(blink txt 폴더)로 설정하세요.")
    path = os.path.realpath(raw_value)
    if not os.path.isdir(path):
        sys.exit("BLINK_OUT_DIR 가 폴더가 아닙니다.")
    return path


def require_seq_dir(blink_dir):
    """SEQ_OUT_DIR(출력 루트)를 읽고 검사한다. 설정이 없거나, 저장소 안이거나, 입력 루트 안이면 중단한다(기본값 없음).

    입력: blink_dir - 입력 루트의 실제 경로
    반환: 출력 루트의 실제 경로
    """
    raw_value = os.environ.get("SEQ_OUT_DIR")
    if not raw_value:
        sys.exit("환경변수 SEQ_OUT_DIR 를 출력 루트로 설정하세요(저장소 밖, 기본값 없음).")
    path = os.path.realpath(raw_value)
    if is_inside(path, REPO):
        sys.exit("SEQ_OUT_DIR 이 저장소 폴더 안입니다. 저장소 밖 폴더를 지정하세요.")
    if is_inside(path, blink_dir):
        sys.exit("SEQ_OUT_DIR 이 BLINK_OUT_DIR 와 같거나 그 안입니다. 입력 폴더에는 쓰지 않습니다.")
    return path


def read_inventory_rows(inventory_path):
    """인벤토리 CSV 를 읽어 영상 목록(dict)으로 만든다. 피험자 id 는 폴더 이름 그대로(문자열)이다.

    입력: inventory_path - video_inventory.csv 경로
    반환: dict(fold, part, subject, class_label, variant, manual_flag, video_id, txt_parts) 의 list
    """
    videos = []
    with open(inventory_path, "r", newline="", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            parts = row["rel_path"].split("/")
            if row["class_label"] == "":
                sys.exit("인벤토리에 클래스를 알 수 없는 영상이 있습니다: %s" % row["rel_path"])
            name = row["class_label"] + ("_" + row["variant"] if row["variant"] else "")
            videos.append({"fold": int(row["fold"]), "part": int(row["part"]), "subject": parts[1],
                           "class_label": int(row["class_label"]), "variant": row["variant"],
                           "manual_flag": int(row["manual_flag"] or 0), "video_id": "%s/%s/%s" % (parts[0], parts[1], name),
                           "txt_parts": (parts[0], parts[1], name + ".txt")})
    return videos


def parse_blink_file(path, label):
    """blink txt 를 읽는다. 첫 빈 줄과 줄 끝 공백은 명시적으로 건너뛰고, 열 수가 5가 아니면 중단한다.

    입력: path - txt 경로, label - 오류 메시지에 쓸 영상 이름
    반환: 파일이 없으면 None, 있으면 (n, 5) float64 배열(blink 가 없으면 (0, 5))
    """
    if not os.path.isfile(path):
        return None
    values = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            text = line.strip()
            if text == "":
                continue
            fields = text.split()
            if len(fields) != 5:
                raise AssertionError("열 수가 5가 아닙니다(%d열): %s" % (len(fields), label))
            values.append([float(field) for field in fields])
    if not values:
        return np.zeros((0, 5))
    return np.array(values, dtype=np.float64)


def load_videos(videos, blink_dir):
    """각 영상의 txt 를 읽어 video["table"] 에 붙인다. 변형 파일은 이번 단계에서 읽지 않는다.

    입력: videos - read_inventory_rows 결과, blink_dir - 입력 루트
    반환: 같은 list (table 키가 추가됨. txt 가 없거나 읽지 않으면 None)
    """
    for video in videos:
        video["table"] = None
        if not video["variant"]:
            path = os.path.join(blink_dir, *video["txt_parts"])
            video["table"] = parse_blink_file(path, video["video_id"])
    return videos


def classify_video(video):
    """영상 하나 자체의 제외 사유를 정한다(피험자 단위 사유는 나중에 붙인다).

    입력: video - table 이 붙은 영상 dict
    반환: variant_deferred, not_extracted, no_blink_file 중 하나 또는 None(자체로는 문제 없음)
    """
    if video["variant"]:
        return "variant_deferred"
    if video["table"] is None:
        return "not_extracted" if video["manual_flag"] == 1 else "no_blink_file"
    if len(video["table"]) == 0:
        return "no_blink_file"
    return None


def compute_reference(table):
    """alert 표에서 저자 PP 와 같은 방식으로 정규화 기준(평균, 표준편차)을 계산한다.

    기준 구간은 마지막 n//3 개이고 n<3 이면 [-0:] 처럼 전체이다. 표준편차가 0이면 전체 표준편차로 대체한다.
    입력: table - (n, 5) alert 표
    반환: dict(ref_n, range, means, stds, fallback, reason). reason 은 degenerate 사유 문자열 또는 None
    """
    count = len(table)
    bunch = count // 3
    ref_n = bunch if bunch > 0 else count
    means, stds, fallback = [], [], 0
    with np.errstate(all="ignore"):
        for column in FEATURE_COLUMNS:
            values = np.ascontiguousarray(table[:, column])
            mean = np.mean(values[-bunch:])
            sigma = np.std(values[-bunch:])
            if sigma == 0:
                sigma = np.std(values)
                fallback += 1
            means.append(mean)
            stds.append(sigma)
    first = count - ref_n
    reference = {"ref_n": ref_n, "range": "%d-%d" % (first, count - 1), "means": means, "stds": stds,
                 "fallback": fallback, "fallback_kind": fallback_kind(count, fallback), "reason": None}
    reference["reason"] = degenerate_reason(reference, count)
    return reference


def fallback_kind(alert_rows, fallback_columns):
    """저자 PP 가 쓴 대체 방식을 문자열로 돌려준다.

    입력: alert_rows - alert 의 blink 행 수, fallback_columns - 표준편차 0 이라 전체 표준편차로 바꾼 열 수
    반환: whole_array(n<3 이라 전체 배열 통계) / whole_std(전체 열 표준편차로 대체) / none
    """
    if alert_rows < 3:
        return "whole_array"
    if fallback_columns > 0:
        return "whole_std"
    return "none"


def degenerate_reason(reference, alert_rows):
    """저자 코드가 예외를 내거나 비유한 값을 만드는 경우의 사유를 문자열로 돌려준다.

    alert 1행(저자 TypeError), 대체 후에도 표준편차가 0 또는 비유한(저자 NaN)일 때만 해당한다.
    입력: reference - compute_reference 의 dict, alert_rows - alert 의 blink 행 수
    반환: 사유 문자열 또는 None
    """
    if alert_rows < 2:
        return "alert %d행: 저자 코드는 TypeError" % alert_rows
    for mean, sigma in zip(reference["means"], reference["stds"]):
        if sigma == 0 or not np.isfinite(sigma) or not np.isfinite(mean):
            return "표준편차가 0이거나 비유한 (alert %d행)" % alert_rows
    return None


def normalize_table(table, reference, n_used):
    """표의 앞쪽 n_used 개 blink 를 기준 통계로 정규화한다(열마다 (값-평균)/표준편차, PP 의 normalize_blinks 와 같음).

    입력: table - (n, 5) 표, reference - compute_reference 결과, n_used - 데이터로 쓸 blink 수
    반환: (n_used, 4) float64 배열(열 순서: 빈도x100, 진폭, 지속 시간, 속도)
    """
    normalized = np.zeros([n_used, 4])
    for position, column in enumerate(FEATURE_COLUMNS):
        values = np.ascontiguousarray(table[:, column])
        normalized[:, position] = (values[0:n_used] - reference["means"][position]) / reference["stds"][position]
    return normalized


def unroll_in_time(in_data):
    """PP 의 unroll_in_time 과 같다: 길이 30 윈도우를 폭 2 로 민다. 30 이하이면 앞쪽을 0 으로 채운 시퀀스 1개이다.

    입력: in_data - (n, 4) 정규화된 blink
    반환: (N, 30, 4) float64 배열
    """
    count = len(in_data)
    if count <= WINDOW_SIZE:
        out_data = np.zeros([1, WINDOW_SIZE, 4])
        out_data[0, -count:, :] = in_data
        return out_data
    total = ((count - WINDOW_SIZE) // STRIDE) + 1
    out_data = np.zeros([total, WINDOW_SIZE, 4])
    for index in range(total):
        out_data[index, :, :] = in_data[index * STRIDE:index * STRIDE + WINDOW_SIZE, :]
    return out_data


def sequence_ranges(n_used):
    """시퀀스마다 사용한 blink 행 번호의 범위(0부터, 끝 포함)를 계산한다. 앞쪽 0 패딩 행은 범위에 넣지 않는다.

    입력: n_used - 데이터로 쓴 blink 수
    반환: (start_blink, end_blink) 튜플 list
    """
    if n_used <= WINDOW_SIZE:
        return [(0, n_used - 1)]
    ranges = []
    for index in range(((n_used - WINDOW_SIZE) // STRIDE) + 1):
        ranges.append((index * STRIDE, index * STRIDE + WINDOW_SIZE - 1))
    return ranges


def attach_sequences(video, reference):
    """포함된 영상의 정규화와 시퀀스 생성을 수행해 video 에 X, ranges, n_used 를 붙인다.

    입력: video - 포함된 영상 dict, reference - 피험자의 정규화 기준
    """
    count = len(video["table"])
    n_used = count - count // 3 if video["class_label"] == 0 else count
    video["X"] = unroll_in_time(normalize_table(video["table"], reference, n_used))
    video["ranges"] = sequence_ranges(n_used)
    video["n_used"] = n_used
    if len(video["ranges"]) != len(video["X"]):
        raise AssertionError("시퀀스 수 계산이 맞지 않습니다: %s" % video["video_id"])


def find_alert(subject):
    """피험자의 alert 영상(클래스 0, 변형 아님)을 찾는다.

    입력: subject - 피험자 dict
    반환: 영상 dict 또는 None
    """
    for video in subject["videos"]:
        if video["class_label"] == 0 and not video["variant"]:
            return video
    return None


def process_subject(subject):
    """피험자 하나의 제외 사유를 정하고 포함된 영상의 시퀀스를 만든다. 앞 피험자의 값으로 대체하지 않는다.

    입력: subject - 피험자 dict (videos 의 table 이 채워져 있어야 한다)
    """
    for video in subject["videos"]:
        video["reason"] = classify_video(video)
    alert = find_alert(subject)
    state, reference = "no_reference", None
    if alert is not None and alert["reason"] is None:
        reference = compute_reference(alert["table"])
        state = "degenerate" if reference["reason"] else "ok"
    subject["state"], subject["reference"] = state, reference
    for video in subject["videos"]:
        if video["reason"] is None and state != "ok":
            video["reason"] = "no_alert_reference" if state == "no_reference" else "degenerate_reference"
        if video["reason"] is None:
            attach_sequences(video, reference)


def group_subjects(videos):
    """영상 목록을 피험자별로 묶는다(피험자 id 는 폴더 이름 문자열).

    입력: videos - load_videos 결과
    반환: {피험자: dict(id, fold, part, videos, state, reference)}
    """
    subjects = {}
    for video in videos:
        subject = subjects.setdefault(video["subject"], {"id": video["subject"], "fold": video["fold"],
                                                          "part": video["part"], "videos": [], "state": None,
                                                          "reference": None})
        subject["videos"].append(video)
    return subjects


def make_meta_rows(video, fold, split):
    """포함된 영상의 시퀀스마다 메타데이터 행(seq_index 는 나중에 채움)을 만든다.

    입력: video - 포함된 영상 dict, fold - fold 번호, split - train 또는 test
    반환: dict list
    """
    rows = []
    for position, (start, end) in enumerate(video["ranges"]):
        rows.append({"seq_index": 0, "fold": fold, "split": split, "subject_id": video["subject"],
                     "class_label": video["class_label"], "variant": video["variant"], "video_id": video["video_id"],
                     "seq_in_video": position, "n_blinks_video": len(video["table"]), "n_blinks_used": video["n_used"],
                     "start_blink": start, "end_blink": end, "padded": 1 if video["n_used"] < WINDOW_SIZE else 0})
    return rows


def collect_split(subject_list, fold, split):
    """피험자 목록의 포함된 영상을 순서대로(alert, semisleepy, sleepy) 이어 붙여 한 split 의 배열과 메타데이터를 만든다.

    입력: subject_list - 정렬된 피험자 dict list, fold - fold 번호, split - train 또는 test
    반환: dict(X, y, meta, starts). starts 는 영상별 시작 시퀀스 번호 list
    """
    arrays, labels, meta, starts = [], [], [], []
    offset = 0
    for subject in subject_list:
        for label in CLASS_ORDER:
            for video in subject["videos"]:
                if video["class_label"] != label or video["variant"] or video["reason"] is not None:
                    continue
                count = len(video["X"])
                starts.append(offset)
                arrays.append(video["X"])
                labels.append(label * np.ones([count, 1]))
                meta.extend(make_meta_rows(video, fold, split))
                offset += count
    if not arrays:
        sys.exit("Fold%d %s 에 포함된 영상이 없습니다." % (fold, split))
    for index, row in enumerate(meta):
        row["seq_index"] = index
    return {"X": np.concatenate(arrays, axis=0), "y": np.concatenate(labels, axis=0), "meta": meta, "starts": starts}


def shuffle_split(block, fold):
    """훈련 split 을 fold 번호를 시드로 고정해 섞는다(PP 의 unison_shuffled_copies 와 같은 순열 방식, 시드만 고정).

    입력: block - collect_split 결과, fold - fold 번호(시드)
    반환: 섞인 dict(X, y, meta, starts=[]). meta 의 seq_index 는 섞인 뒤의 행 번호이다
    """
    permutation = np.random.RandomState(fold).permutation(len(block["X"]))
    meta = [dict(block["meta"][index]) for index in permutation]
    for index, row in enumerate(meta):
        row["seq_index"] = index
    return {"X": block["X"][permutation], "y": block["y"][permutation], "meta": meta, "starts": []}


def assemble_fold(fold, subjects):
    """fold 하나의 훈련·테스트 배열을 만든다. 테스트는 그 fold 의 피험자 전체, 훈련은 나머지 fold 의 피험자이다.

    입력: fold - fold 번호, subjects - group_subjects 결과
    반환: dict(train, test, start_indices). start_indices 는 테스트 영상 경계(합이 테스트 시퀀스 수와 같음을 단언)
    """
    ordered = sorted(subjects.values(), key=lambda s: (s["fold"], s["part"], s["id"]))
    test_subjects = [s for s in ordered if s["fold"] == fold]
    train_subjects = [s for s in ordered if s["fold"] != fold]
    test = collect_split(test_subjects, fold, "test")
    train = shuffle_split(collect_split(train_subjects, fold, "train"), fold)
    starts = test["starts"]
    counts = [b - a for a, b in zip(starts, starts[1:] + [len(test["X"])])]
    if starts[0] != 0 or sum(counts) != len(test["X"]) or min(counts) < 1:
        raise AssertionError("start_indices 가 테스트 시퀀스 수와 맞지 않습니다: Fold%d" % fold)
    return {"train": train, "test": test, "start_indices": starts}


def build_pipeline(blink_dir, inventory_path):
    """입력을 읽어 모든 영상·피험자 처리와 fold 별 배열 조립까지 수행한다(파일은 쓰지 않는다).

    입력: blink_dir - 입력 루트, inventory_path - 인벤토리 CSV 경로
    반환: dict(videos, subjects, folds). folds 는 {fold 번호: assemble_fold 결과}
    """
    videos = load_videos(read_inventory_rows(inventory_path), blink_dir)
    subjects = group_subjects(videos)
    for subject in subjects.values():
        process_subject(subject)
    folds = {}
    for fold in sorted(set(subject["fold"] for subject in subjects.values())):
        folds[fold] = assemble_fold(fold, subjects)
    return {"videos": videos, "subjects": subjects, "folds": folds}


def summary_rows(pipeline):
    """영상마다 video_summary.csv 의 한 행을 만든다. 피험자 단위 값(ref_n 등)은 그 피험자의 모든 영상 행에 같게 넣는다.

    입력: pipeline - build_pipeline 결과
    반환: dict list (SUMMARY_COLUMNS 열)
    """
    rows = []
    for video in pipeline["videos"]:
        subject = pipeline["subjects"][video["subject"]]
        reference = subject["reference"]
        included = video["reason"] is None
        row = {"video_id": video["video_id"], "subject_id": video["subject"], "class_label": video["class_label"],
               "status": "included" if included else "excluded", "reason": video["reason"] or "",
               "n_blink_rows": "" if video["table"] is None else len(video["table"]), "stats_blink_range": "",
               "ref_n": "", "ref_small": "", "author_fallback_used": "", "norm_mean_json": "", "norm_std_json": "",
               "n_sequences": len(video["X"]) if included else 0}
        if reference is not None:
            row["stats_blink_range"], row["ref_n"] = reference["range"], reference["ref_n"]
            row["ref_small"] = "true" if reference["ref_n"] < REF_SMALL_LIMIT else "false"
            row["author_fallback_used"] = reference["fallback_kind"]
        if subject["state"] == "ok":
            row["norm_mean_json"] = json.dumps([float(x) for x in reference["means"]])
            row["norm_std_json"] = json.dumps([float(x) for x in reference["stds"]])
        rows.append(row)
    return rows


def exclusion_rows(pipeline):
    """제외된 영상마다 exclusions.csv 의 한 행을 만든다. 피험자 단위 사유는 level 이 subject 이다.

    입력: pipeline - build_pipeline 결과
    반환: dict list (EXCLUSION_COLUMNS 열)
    """
    rows = []
    for video in pipeline["videos"]:
        reason = video["reason"]
        if reason is None:
            continue
        subject = pipeline["subjects"][video["subject"]]
        detail = ""
        if reason == "degenerate_reference":
            detail = subject["reference"]["reason"]
        elif reason == "not_extracted":
            detail = "txt 없음, 인벤토리 manual_flag=1"
        elif reason == "no_blink_file":
            detail = "txt 없음 또는 blink 0행"
        rows.append({"video_id": video["video_id"], "subject_id": video["subject"], "class_label": video["class_label"],
                     "level": "subject" if reason in SUBJECT_LEVEL_REASONS else "video", "reason": reason,
                     "detail": detail})
    return rows


def coverage_video_lines(pipeline):
    """coverage.txt 의 영상 수 부분(전체, 제외 사유별, fold 별·클래스별 포함 수)을 만든다.

    입력: pipeline - build_pipeline 결과
    반환: 문자열 list
    """
    videos = pipeline["videos"]
    included = [v for v in videos if v["reason"] is None]
    lines = ["== 영상 수 ==", "인벤토리 영상 %d개: 포함 %d, 제외 %d" % (len(videos), len(included), len(videos) - len(included))]
    lines.append("== 제외 사유별 개수 ==")
    for reason in REASONS:
        lines.append("  %-20s %d" % (reason, sum(1 for v in videos if v["reason"] == reason)))
    lines.append("== fold 별 클래스별 포함 영상 수 ==")
    for fold in sorted(pipeline["folds"]):
        counts = [sum(1 for v in included if v["fold"] == fold and v["class_label"] == label) for label in CLASS_ORDER]
        lines.append("  Fold%d: 클래스 0 %d, 5 %d, 10 %d" % (fold, counts[0], counts[1], counts[2]))
    return lines


def coverage_sequence_lines(pipeline):
    """coverage.txt 의 시퀀스 수 부분(fold 별 훈련·테스트)을 만든다.

    입력: pipeline - build_pipeline 결과
    반환: 문자열 list
    """
    lines = ["== fold 별 시퀀스 수 =="]
    for fold in sorted(pipeline["folds"]):
        result = pipeline["folds"][fold]
        lines.append("  Fold%d: 훈련 %d, 테스트 %d, 테스트 영상 %d개" % (
            fold, len(result["train"]["X"]), len(result["test"]["X"]), len(result["start_indices"])))
    return lines


def coverage_reference_lines(pipeline):
    """coverage.txt 의 정규화 기준 부분(ref_n 분포, ref_small 피험자, 표준편차 대체 피험자)을 만든다.

    입력: pipeline - build_pipeline 결과
    반환: 문자열 list
    """
    usable = [s for s in pipeline["subjects"].values() if s["state"] == "ok"]
    ref_values = sorted(s["reference"]["ref_n"] for s in usable)
    small = sorted(s["id"] for s in usable if s["reference"]["ref_n"] < REF_SMALL_LIMIT)
    fallback = sorted(s["id"] for s in usable if s["reference"]["fallback"] > 0)
    lines = ["== 정규화 기준(ref_n) 분포 (사용 가능한 피험자 %d명) ==" % len(usable)]
    if ref_values:
        lines.append("  최소 %d, 중앙값 %d, 최대 %d" % (ref_values[0], ref_values[len(ref_values) // 2], ref_values[-1]))
    lines.append("  ref_small(ref_n < %d): %d명 %s" % (REF_SMALL_LIMIT, len(small), small))
    lines.append("  기준 구간 표준편차 0 이라 전체 표준편차로 대체한 피험자: %d명 %s" % (len(fallback), fallback))
    for kind in ("none", "whole_array", "whole_std"):
        members = sorted(s["id"] for s in usable if s["reference"]["fallback_kind"] == kind)
        lines.append("  author_fallback_used=%s: %d명 %s" % (kind, len(members), members if kind != "none" else ""))
    return lines


def coverage_text(pipeline):
    """coverage.txt 전체 문자열을 만든다.

    입력: pipeline - build_pipeline 결과
    반환: 줄바꿈으로 이어 붙인 문자열(끝에 줄바꿈 포함)
    """
    lines = coverage_video_lines(pipeline) + coverage_sequence_lines(pipeline) + coverage_reference_lines(pipeline)
    return "\n".join(lines) + "\n"


def output_paths(out_dir, folds):
    """쓸 모든 출력 파일의 경로를 (상대 이름, 전체 경로) 튜플 list 로 만든다.

    입력: out_dir - 출력 루트, folds - fold 번호 list
    반환: 튜플 list
    """
    names = []
    for fold in folds:
        for pattern in ("Blinks_30_Fold%d.npy", "Labels_30_Fold%d.npy", "BlinksTest_30_Fold%d.npy", "LabelsTest_30_Fold%d.npy"):
            names.append("npy/" + pattern % fold)
        names.append("metadata/sequence_metadata_fold%d_train.csv" % fold)
        names.append("metadata/sequence_metadata_fold%d_test.csv" % fold)
        names.append("metadata/start_indices_fold%d.json" % fold)
    names.extend(["metadata/video_summary.csv", "metadata/exclusions.csv", "metadata/coverage.txt"])
    return [(name, os.path.join(out_dir, *name.split("/"))) for name in names]


def write_csv(path, columns, rows):
    """CSV 를 새 파일로만 쓴다(이미 있으면 예외). UTF-8, 줄바꿈 LF.

    입력: path - 경로, columns - 열 이름 list, rows - dict list
    """
    with open(path, "x", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_fold_files(out_dir, fold, result):
    """fold 하나의 npy 4개와 메타데이터 3개를 새 파일로 쓴다.

    입력: out_dir - 출력 루트, fold - fold 번호, result - assemble_fold 결과
    """
    npy_dir, meta_dir = os.path.join(out_dir, "npy"), os.path.join(out_dir, "metadata")
    for name, array in (("Blinks_30_Fold%d.npy" % fold, result["train"]["X"]), ("Labels_30_Fold%d.npy" % fold, result["train"]["y"]),
                        ("BlinksTest_30_Fold%d.npy" % fold, result["test"]["X"]),
                        ("LabelsTest_30_Fold%d.npy" % fold, result["test"]["y"])):
        with open(os.path.join(npy_dir, name), "xb") as file:
            np.save(file, array)
    write_csv(os.path.join(meta_dir, "sequence_metadata_fold%d_train.csv" % fold), SEQUENCE_COLUMNS, result["train"]["meta"])
    write_csv(os.path.join(meta_dir, "sequence_metadata_fold%d_test.csv" % fold), SEQUENCE_COLUMNS, result["test"]["meta"])
    with open(os.path.join(meta_dir, "start_indices_fold%d.json" % fold), "x", encoding="utf-8") as file:
        file.write(json.dumps([int(value) for value in result["start_indices"]]) + "\n")


def write_outputs(out_dir, pipeline):
    """모든 산출물을 쓴다. 쓰기 전에 이미 있는 출력 파일이 하나라도 있으면 아무것도 쓰지 않고 중단한다.

    입력: out_dir - 출력 루트(저장소 밖), pipeline - build_pipeline 결과
    """
    paths = output_paths(out_dir, sorted(pipeline["folds"]))
    existing = [name for name, path in paths if os.path.exists(path)]
    if existing:
        sys.exit("이미 있는 출력 파일 %d개가 있어 덮어쓰지 않고 중단합니다(예: %s)." % (len(existing), existing[0]))
    os.makedirs(os.path.join(out_dir, "npy"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "metadata"), exist_ok=True)
    for fold, result in sorted(pipeline["folds"].items()):
        write_fold_files(out_dir, fold, result)
    write_csv(os.path.join(out_dir, "metadata", "video_summary.csv"), SUMMARY_COLUMNS, summary_rows(pipeline))
    write_csv(os.path.join(out_dir, "metadata", "exclusions.csv"), EXCLUSION_COLUMNS, exclusion_rows(pipeline))
    with open(os.path.join(out_dir, "metadata", "coverage.txt"), "x", encoding="utf-8") as file:
        file.write(coverage_text(pipeline))


def print_exclusions(pipeline):
    """제외된 영상 목록(사유 포함)을 콘솔에 출력한다.

    입력: pipeline - build_pipeline 결과
    """
    rows = exclusion_rows(pipeline)
    print("== 제외 목록 (%d개) ==" % len(rows))
    for row in rows:
        print("  %s  %s  (%s) %s" % (row["video_id"], row["reason"], row["level"], row["detail"]))


def main():
    """전체 흐름: 환경 확인, 파이프라인 실행, --check-only 이면 coverage 와 제외 목록만 출력하고 아니면 SEQ_OUT_DIR 에 쓴다."""
    arguments = parse_arguments()
    blink_dir = require_blink_dir()
    out_dir = None if arguments.check_only else require_seq_dir(blink_dir)
    pipeline = build_pipeline(blink_dir, arguments.inventory)
    print(coverage_text(pipeline))
    print_exclusions(pipeline)
    if arguments.check_only:
        print("--check-only: 아무 파일도 쓰지 않았습니다.")
        return
    write_outputs(out_dir, pipeline)
    print("산출물을 SEQ_OUT_DIR 에 썼습니다.")


if __name__ == "__main__":
    main()
