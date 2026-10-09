# stage2.py
# 역할: 2단계 전역 표준화 통계(결정 I-W-20, I-30 D2)를 폴드 학습 배열에서 계산하고, 저장소 안의 작은 JSON 설정 파일(stage2_stats.json)로
#       저장/로드하고, 배열에서 다시 계산해 JSON 과 대조한다. 계산식은 저자 Training.py 420~431행과 같다:
#       특징 k 마다 np.mean(Blinks[:,:,k]), np.std(Blinks[:,:,k]) (ddof=0, 0 패딩 창의 0 도 포함), 학습 배열은 폴드 k 의 Blinks_30_Fold{k}.npy.
# JSON 형식: folds["1"~"5"] = {file, n, shape, dtype, sha256_12(배열 파일 SHA-256 앞 12자), mean[4], std[4]}, 열 순서 [Freq, Amp, Dur, Vel].
#       float 은 repr 전체 자릿수로 저장해 반올림 없이 정확히 복원된다. 절대경로는 쓰지 않고 파일 이름만 적는다.
# 사용법: (저장소 폴더에서 conda activate ghoddoosian_tf1, set PYTHONPATH=src)
#           python -m improve.hmlstm.stage2 --verify --data-dir <SEQ_DIR>/npy                       (JSON 과 배열 재계산 대조, 아무 것도 쓰지 않음)
#           python -m improve.hmlstm.stage2 --write --data-dir <SEQ_DIR>/npy --ckpt-root <CKPT_DIR>  (stage2_stats.json 을 새로 만든다, 이미 있으면 중단)
#         --ckpt-root 를 주면 <CKPT_DIR>/b1_seed1/seed1/fold{k}/run_config.json 의 input_sha256 과 배열 해시를 대조한다(불일치면 쓰지 않고 보고).
# 종료 코드: 0 = 일치, 1 = 불일치, 2 = 입력 오류
"""2단계 통계 계산, 저장, 검증(Python 3.6 호환)."""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

from .windows import FEATURE_NAMES

FOLDS = (1, 2, 3, 4, 5)
ARRAY_NAME = "Blinks_30_Fold%d.npy"
HASH_CHARS = 12
DEFAULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stage2_stats.json")


def file_sha256(path):
    """파일 전체의 SHA-256 16진 문자열(메모리 안에서만 쓰고 문서와 JSON 에는 앞 12자만 적는다)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_stats(array):
    """(평균 4개, 표준편차 4개) python float list. Training.py 420~431행과 같은 식(array[:,:,k] 전체, ddof=0)."""
    return [float(np.mean(array[:, :, k])) for k in range(4)], [float(np.std(array[:, :, k])) for k in range(4)]


def describe_fold(array_path, fold):
    """폴드 학습 배열 파일에서 JSON 한 항목(dict)과 전체 해시를 만든다. 반환: (항목, 전체 해시)."""
    array = np.load(array_path)
    if array.ndim != 3 or array.shape[1:] != (30, 4):
        raise ValueError("배열 모양이 (N, 30, 4) 가 아닙니다: %s" % (array.shape,))
    means, stds = array_stats(array)
    full_hash = file_sha256(array_path)
    entry = {"file": ARRAY_NAME % fold, "n": int(array.shape[0]), "shape": [int(v) for v in array.shape], "dtype": str(array.dtype),
             "sha256_12": full_hash[:HASH_CHARS], "mean": means, "std": stds}
    return entry, full_hash


def read_run_config_hash(ckpt_root, seed, fold):
    """학습 실행의 run_config.json 에서 그 폴드 학습 배열의 input_sha256 전체 값을 읽는다(없으면 None)."""
    path = os.path.join(ckpt_root, "b1_seed%d" % seed, "seed%d" % seed, "fold%d" % fold, "run_config.json")
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle).get("input_sha256", {}).get(ARRAY_NAME % fold)


def build_document(data_dir, ckpt_root=None, seed=1):
    """폴드 1~5 의 JSON 문서를 만든다. 반환: (문서 dict, 해시 대조 결과 list). 대조 항목: (폴드, 'match'|'mismatch'|'no_run_config')."""
    folds, checks = {}, []
    for fold in FOLDS:
        entry, full_hash = describe_fold(os.path.join(data_dir, ARRAY_NAME % fold), fold)
        folds[str(fold)] = entry
        if ckpt_root is not None:
            recorded = read_run_config_hash(ckpt_root, seed, fold)
            checks.append((fold, "no_run_config" if recorded is None else ("match" if recorded == full_hash else "mismatch")))
    document = {"schema": 1, "description": "2단계 표준화 통계: 폴드 k 학습 배열 Blinks_30_Fold{k}.npy 의 array[:,:,j] 전체 np.mean, np.std(ddof=0)",
                "feature_order": list(FEATURE_NAMES), "folds": folds}
    return document, checks


def write_document(document, path):
    """JSON 을 새 파일로만 쓴다(이미 있으면 FileExistsError)."""
    with open(path, "x", encoding="utf-8", newline="\n") as handle:
        json.dump(document, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def load_document(path=None):
    """JSON 문서를 읽고 형식을 검사한다. 키나 길이가 맞지 않으면 ValueError."""
    with open(path or DEFAULT_PATH, "r", encoding="utf-8") as handle:
        document = json.load(handle)
    if document.get("feature_order") != list(FEATURE_NAMES):
        raise ValueError("feature_order 가 %s 이 아닙니다" % (list(FEATURE_NAMES),))
    for fold in FOLDS:
        entry = document.get("folds", {}).get(str(fold))
        if entry is None:
            raise ValueError("폴드 %d 항목이 없습니다" % fold)
        for key in ("file", "n", "shape", "dtype", "sha256_12", "mean", "std"):
            if key not in entry:
                raise ValueError("폴드 %d 항목에 %s 가 없습니다" % (fold, key))
        if len(entry["mean"]) != 4 or len(entry["std"]) != 4:
            raise ValueError("폴드 %d 의 mean, std 는 길이 4 여야 합니다" % fold)
    return document


def load_stats(path=None):
    """{폴드: (평균 배열(4), 표준편차 배열(4))} float64 를 돌려준다."""
    document = load_document(path)
    return dict((fold, (np.array(document["folds"][str(fold)]["mean"], dtype=np.float64),
                        np.array(document["folds"][str(fold)]["std"], dtype=np.float64))) for fold in FOLDS)


def standardize(windows, means, stds):
    """2단계 표준화: 특징 k 마다 (x - 평균[k]) / 표준편차[k] (Training.py 420~431행과 같은 식). 입력은 바꾸지 않고 float64 복사본을 돌려준다."""
    out = np.array(windows, dtype=np.float64, copy=True)
    for k in range(4):
        out[:, :, k] = (out[:, :, k] - means[k]) / stds[k]
    return out


def verify_document(document, data_dir):
    """JSON 의 각 폴드를 배열 파일에서 다시 계산해 대조한다. 반환: 폴드별 dict(fold, shape_match, n_match, hash_match, max_abs_diff, exact)."""
    rows = []
    for fold in FOLDS:
        saved = document["folds"][str(fold)]
        entry, _full = describe_fold(os.path.join(data_dir, saved["file"]), fold)
        diffs = [abs(a - b) for a, b in zip(saved["mean"] + saved["std"], entry["mean"] + entry["std"])]
        rows.append({"fold": fold, "shape_match": saved["shape"] == entry["shape"] and saved["dtype"] == entry["dtype"],
                     "n_match": saved["n"] == entry["n"], "hash_match": saved["sha256_12"] == entry["sha256_12"],
                     "max_abs_diff": max(diffs), "exact": max(diffs) == 0.0})
    return rows


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="2단계 표준화 통계 JSON 생성과 검증")
    parser.add_argument("--write", action="store_true", help="stage2_stats.json 을 새로 만든다(이미 있으면 중단)")
    parser.add_argument("--verify", action="store_true", help="JSON 과 배열 재계산을 대조한다(아무 것도 쓰지 않음)")
    parser.add_argument("--data-dir", dest="data_dir", help="Blinks_30_Fold{k}.npy 가 있는 폴더(<SEQ_DIR>/npy)")
    parser.add_argument("--ckpt-root", dest="ckpt_root", default=None, help="학습 출력 루트(<CKPT_DIR>), run_config.json 해시 대조용")
    parser.add_argument("--seed", type=int, default=1, help="run_config.json 을 읽을 시드(기본 1)")
    parser.add_argument("--json", default=DEFAULT_PATH, help="JSON 경로(기본: 이 패키지 안의 stage2_stats.json)")
    return parser.parse_args(argv)


def main(argv=None):
    options = parse_args(argv)
    if options.write == options.verify or not options.data_dir:
        print("--write 와 --verify 중 하나와 --data-dir 이 필요합니다.")
        return 2
    try:
        if options.write:
            document, checks = build_document(options.data_dir, options.ckpt_root, options.seed)
            for fold, state in checks:
                print("fold %d run_config input_sha256 vs array: %s" % (fold, state))
            if any(state == "mismatch" for _fold, state in checks):
                print("해시가 일치하지 않아 쓰지 않았습니다.")
                return 1
            write_document(document, options.json)
            print("wrote %s" % os.path.basename(options.json))
            for fold in FOLDS:
                entry = document["folds"][str(fold)]
                print("fold %d: N=%d shape=%s dtype=%s sha256_12=%s" % (fold, entry["n"], entry["shape"], entry["dtype"], entry["sha256_12"]))
            return 0
        rows = verify_document(load_document(options.json), options.data_dir)
    except (OSError, ValueError) as error:
        print("%s: %s" % (type(error).__name__, error))
        return 2
    for row in rows:
        print("fold %d: shape/dtype_match=%s n_match=%s hash12_match=%s max_abs_diff=%.3e exact=%s" % (
            row["fold"], row["shape_match"], row["n_match"], row["hash_match"], row["max_abs_diff"], row["exact"]))
    ok = all(row["shape_match"] and row["n_match"] and row["hash_match"] and row["exact"] for row in rows)
    print("verdict: %s" % ("일치" if ok else "불일치"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
