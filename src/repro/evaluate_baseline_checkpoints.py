# evaluate_baseline_checkpoints.py
# 역할: B1 기준선 학습(reproduction_train_baseline.py)이 남긴 fold별 체크포인트(my_model<k>)를 불러 테스트 영상 전체의
#       영상별 예측(저자의 영상 단위 다수결, 임계값 3.34/6.68)을 만들고, 클래스별 영상 정확도와 혼동행렬을 낸다.
#       체크포인트 로드와 예측만 하고 학습은 하지 않는다. Training.py 는 텍스트로 읽어 메모리에서만 치환한다.
# 사용법: 저장소 루트에서 conda 환경 ghoddoosian_tf1 의 python 으로 실행한다.
#           python src/repro/evaluate_baseline_checkpoints.py --roots <시드1 루트> <시드2 루트> <시드3 루트> --out-dir <집계 폴더> --dry-run
#           python src/repro/evaluate_baseline_checkpoints.py --roots <시드1 루트> <시드2 루트> <시드3 루트> --out-dir <집계 폴더>
#         옵션: --data-dir (기본 SEQ_OUT_DIR/npy), --start-indices-dir (기본 SEQ_OUT_DIR/metadata; start_indices JSON 과
#               sequence_metadata_fold<k>_test.csv 가 있는 폴더), --dry-run(치환과 영상 식별자 검증만, TF 를 import 하지 않음)
# 입력: 각 루트 안의 실행 폴더/fold<k>/ 의 my_model<k>.* 와 run_config.json (읽기만 한다). 경로는 인자 또는 SEQ_OUT_DIR 로만 받는다.
# 출력(집계 폴더, 학습 출력 루트 밖, 저장소 밖): video_predictions_seed<s>.csv, confusion_matrix_seed<s>.csv,
#       class_accuracy_summary.csv, fold_check.csv(재평가 VA 와 metrics.csv 마지막 epoch 값 대조), evaluation_summary.txt.
#       이미 있는 출력 파일은 덮어쓰지 않고 중단한다. 실행 전후로 third_party 와 학습 출력 파일의 해시가 같은지 확인한다.
"""B1 체크포인트 영상별 평가 도구(Python 3.6 호환, TF 1.8).

평가 방식은 reproduction_eval.py 와 같다: Training.py 를 메모리에서 load=True 로 치환해 원본 Train() 이 체크포인트를 복원하고
테스트 세트를 한 번 평가한다. 영상별 결정은 원본 vote_accuracy_per_video 규칙(영상 안 다수결, 동률이면 평균 예측)을 별도로
구현하고, 원본 함수의 VA 와 원본이 출력한 영상별 줄과 일치하는지 검증한다.
"""
import sys

sys.dont_write_bytecode = True  # third_party 에 __pycache__ 가 생기지 않게 한다

import argparse
import csv
import os
import re
import warnings

warnings.filterwarnings("ignore")
import numpy as np

import common
import reproduction_train_baseline as RB
import summarize_training as ST

REPO = ST.REPO
UPSTREAM_DIR = RB.UPSTREAM_DIR
FOLDS = ST.FOLDS
CLASS_POOL = (0, 5, 10)
THRESHOLD = 3.34
PREDICTION_COLUMNS = ["fold", "video_id", "true_class", "pred_class", "regression_mean"]
CHECK_COLUMNS = ["seed", "fold", "n_videos", "VA_eval", "VA_metrics_csv", "VA_abs_diff", "VA_match", "BSA_loss_eval",
                 "BSA_loss_csv", "BSA_acc_eval", "BSA_acc_csv", "BSRE_eval", "BSRE_csv", "VRE_eval", "VRE_csv"]
ACCURACY_COLUMNS = ["seed", "class", "n_videos", "correct", "accuracy"]
CONFUSION_COLUMNS = ["true_label", "pred_0", "pred_5", "pred_10", "n_videos", "class_accuracy"]
VA_TOLERANCE = 1e-6


def parse_arguments():
    """명령줄 옵션을 해석한다.

    반환: argparse.Namespace (roots, out_dir, data_dir, start_dir, dry_run)
    """
    parser = argparse.ArgumentParser(description="B1 체크포인트 영상별 평가(학습 없음)")
    parser.add_argument("--roots", nargs=3, required=True, metavar=("ROOT1", "ROOT2", "ROOT3"), help="학습 출력 루트 3개(시드 1, 2, 3)")
    parser.add_argument("--out-dir", dest="out_dir", required=True, help="집계 폴더(학습 출력 루트 밖, 저장소 밖)")
    parser.add_argument("--data-dir", dest="data_dir", default=None, help="npy 폴더(기본 SEQ_OUT_DIR/npy)")
    parser.add_argument("--start-indices-dir", dest="start_dir", default=None, help="start_indices JSON 과 메타데이터 폴더(기본 SEQ_OUT_DIR/metadata)")
    parser.add_argument("--dry-run", action="store_true", dest="dry_run", help="치환과 영상 식별자 검증만(TF 를 import 하지 않음)")
    return parser.parse_args()


def validate_paths(arguments):
    """학습 출력 루트, 입력 폴더, 집계 폴더를 검사한다(집계 폴더는 루트 밖, 저장소 밖이어야 한다).

    입력: arguments - parse_arguments 결과
    반환: (루트 list, 집계 폴더, npy 폴더, start_indices 폴더) 실제 경로
    """
    roots = [os.path.realpath(root) for root in arguments.roots]
    for root in roots:
        if not os.path.isdir(root):
            sys.exit("학습 출력 루트가 폴더가 아닙니다: %s" % os.path.basename(root))
    out_dir = os.path.realpath(arguments.out_dir)
    if ST.is_inside(out_dir, REPO) or any(ST.is_inside(out_dir, root) or ST.is_inside(root, out_dir) for root in roots):
        sys.exit("집계 폴더는 저장소 밖이어야 하고 학습 출력 루트와 겹치면 안 됩니다.")
    seq_raw = os.environ.get("SEQ_OUT_DIR")
    if (arguments.data_dir is None or arguments.start_dir is None) and not seq_raw:
        sys.exit("--data-dir 와 --start-indices-dir 를 주거나 환경변수 SEQ_OUT_DIR 를 설정하세요.")
    data_dir = arguments.data_dir or os.path.join(seq_raw, "npy")
    start_dir = arguments.start_dir or os.path.join(seq_raw, "metadata")
    return roots, out_dir, os.path.realpath(data_dir), os.path.realpath(start_dir)


def load_run(root):
    """학습 출력 루트 하나에서 fold 폴더, 시드, 마지막 epoch 지표를 읽는다(SMOKE 가 아닌 완전한 실행이 하나여야 함).

    입력: root - 학습 출력 루트
    반환: dict(seed, fold_dirs{fold: 폴더}, last{fold: 마지막 epoch 지표 dict})
    """
    runs, _skipped = ST.find_run(root)
    if len(runs) != 1 or sorted(runs[0][1]) != list(FOLDS):
        sys.exit("루트 %s 에는 SMOKE 가 아닌 완전한 실행 폴더(fold1~5)가 하나여야 합니다." % os.path.basename(root))
    fold_dirs = runs[0][1]
    last = dict((fold, ST.read_metrics_rows(os.path.join(path, "metrics.csv"))[-1]) for fold, path in fold_dirs.items())
    seed = ST.read_json(os.path.join(fold_dirs[1], "run_config.json"))["seed"]
    return {"seed": seed, "fold_dirs": fold_dirs, "last": last}


def snapshot_hashes(roots):
    """학습 출력 루트 안 모든 파일의 SHA-256 을 모은다(실행 전후 불변 확인용).

    입력: roots - 학습 출력 루트 list
    반환: {루트 이름/상대 경로: 해시}
    """
    hashes = {}
    for root in roots:
        for folder, _subfolders, names in os.walk(root):
            for name in sorted(names):
                path = os.path.join(folder, name)
                key = "%s/%s" % (os.path.basename(root), os.path.relpath(path, root).replace("\\", "/"))
                hashes[key] = common.sha256_of_file(path)
    return hashes


def build_substitutions(fold, checkpoint_prefix, data_prefix, starts):
    """체크포인트 복원 평가용 치환 항목을 만든다(줄 번호는 원본 Training.py 기준, 기대 개수는 모두 1).

    입력: fold, checkpoint_prefix - 체크포인트 폴더('/'로 끝남), data_prefix - npy 폴더('/'로 끝남), starts - start_indices
    반환: RB.make_substitution 항목 list
    """
    items = [RB.make_substitution("load_flag(407)", "load=False", "load=True", 1),
             RB.make_substitution("checkpoint_path(316)", "    variable_path='./'\n", "    variable_path=%r\n" % checkpoint_prefix, 1),
             RB.make_substitution("fold_loop(408)", "for i in range(5):", "for i in range(%d, %d):" % (fold - 1, fold), 1)]
    loads = ((414, "Blinks_30_Fold", "Blinks_30_Fold"), (415, "Labels_30_Fold", "Labels_30_Fold"),
             (416, "BlinksTest_30_Fold", "BlinksTest_30_Fold"), (417, "./LabelsTest_30_Fold", "LabelsTest_30_Fold"))
    for number, old_name, new_name in loads:
        items.append(RB.make_substitution("load_%s(%d)" % (new_name, number), "np.load('%s" % old_name,
                                          "np.load(%r + '%s" % (data_prefix, new_name), 1))
    items.append(RB.make_substitution("start_indices(502)", "    start_indices=np.asarray(start_indices)\n",
                                      "    start_indices=%r\n    start_indices=np.asarray(start_indices)\n" % list(starts), 1))
    return items


def split_patched(text):
    """치환된 소스를 정의부(load=True 줄 앞)와 fold 반복부로 나눈다.

    입력: text - 치환된 소스
    반환: (head, tail). 'load=True' 로 시작하는 줄이 하나가 아니면 RuntimeError
    """
    lines = text.split("\n")
    starts = [index for index, line in enumerate(lines) if line.startswith("load=True")]
    if len(starts) != 1:
        raise RuntimeError("치환된 소스에서 'load=True' 줄이 %d개입니다(1개여야 함)" % len(starts))
    return "\n".join(lines[:starts[0]]) + "\n", "\n".join(lines[starts[0]:]) + "\n"


def map_video_ids(start_dir, fold, starts, test_labels):
    """우리 테스트 메타데이터의 영상 순서와 start_indices 구간을 맞춰 영상 식별자를 붙인다. 어긋나면 중단한다.

    입력: start_dir - 메타데이터 폴더, fold, starts - start_indices, test_labels - (N,1) 테스트 라벨
    반환: 영상 식별자 list(start_indices 순서)
    """
    path = os.path.join(start_dir, "sequence_metadata_fold%d_test.csv" % fold)
    first_index, classes = {}, {}
    with open(path, "r", newline="", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            first_index.setdefault(row["video_id"], int(row["seq_index"]))
            classes.setdefault(row["video_id"], set()).add(int(row["class_label"]))
    ids = sorted(first_index, key=lambda name: first_index[name])
    if [first_index[name] for name in ids] != list(starts):
        sys.exit("Fold%d: 메타데이터 영상 시작 위치가 start_indices 와 다릅니다." % fold)
    for name, start in zip(ids, starts):
        if classes[name] != set([int(test_labels[start][0])]):
            sys.exit("Fold%d: %s 의 클래스가 테스트 라벨 구간과 다릅니다." % (fold, name))
    return ids


def author_decision(predicts):
    """원본 vote_accuracy_per_video(99~111줄)와 같은 규칙으로 영상 하나의 결정과 평균 예측을 구한다.

    입력: predicts - 영상 안 시퀀스 예측 배열(원본과 같은 dtype)
    반환: (예측 클래스, 평균 예측)
    """
    labels = np.array(CLASS_POOL)[(predicts // THRESHOLD).astype(np.int8)]
    shares = [np.sum(labels == pool_value) / len(predicts) for pool_value in CLASS_POOL]
    peak = np.max(shares)
    mean = np.mean(predicts)
    if np.sum([share == peak for share in shares]) != 1:
        return CLASS_POOL[int((mean // THRESHOLD).astype(np.int8))], mean
    return CLASS_POOL[int(np.argmax(shares))], mean


def video_results(predictions, labels, starts):
    """영상 구간마다 (실제 클래스, 예측 클래스, 평균 예측)을 계산한다.

    입력: predictions - (N,1) 예측, labels - (N,1) 라벨, starts - start_indices list
    반환: (실제, 예측, 평균) list
    """
    clipped = np.array(predictions)
    np.clip(clipped, 0, 10, out=clipped)
    ends = list(starts[1:]) + [len(clipped)]
    results = []
    for start, end in zip(starts, ends):
        decision, mean = author_decision(clipped[start:end])
        results.append((int(labels[start][0]), decision, float(mean)))
    return results


def make_capture():
    """원본 지표 함수의 입력과 반환값을 모으는 dict 와 print 대체 함수를 만든다.

    반환: (capture dict, print 대체 함수). capture["lines"] 에 원본이 출력한 문자열이 쌓인다
    """
    capture = {"lines": []}

    def capture_print(*values, **options):
        """원본 print 를 대신해 출력 문자열을 모은다(화면에는 내지 않는다).

        입력: print 와 같은 인자
        반환: 없음
        """
        capture["lines"].append(" ".join(str(value) for value in values))
    return capture, capture_print


def prepare_namespace(head, capture, capture_print):
    """정의부를 실행하고 지표 함수와 Train 을 감싸 입력과 반환값을 capture 에 남기게 한다.

    입력: head - 치환된 정의부, capture, capture_print
    반환: exec 용 이름공간 dict
    """
    namespace = {"__name__": "training_inmemory", "print": capture_print}
    exec(compile(head, "Training.py[정의부]", "exec"), namespace)
    original = dict((name, namespace[name]) for name in
                    ("vote_accuracy_per_video", "regression_per_blink", "regression_per_video", "Train"))

    def wrap_vote(labels, predictions, starts_list, idx):
        """원본 VA 함수의 입력을 복사해 두고 그대로 호출한다."""
        capture.update(labels=np.array(labels), predictions=np.array(predictions), starts=list(starts_list))
        capture["va"] = float(original["vote_accuracy_per_video"](labels, predictions, starts_list, idx))
        return capture["va"]

    def wrap_regression(name, key):
        """원본 회귀 함수를 그대로 호출하고 반환값을 기록하는 함수를 만든다."""
        def wrapped(*args, **kwargs):
            """원본을 호출해 반환값을 capture[key] 에 남긴다."""
            capture[key] = float(original[name](*args, **kwargs))
            return capture[key]
        return wrapped

    def wrap_train(*args, **kwargs):
        """원본 Train 을 호출하고 반환 배열(BSA 손실, 정확도, VA)을 기록한다."""
        out = original["Train"](*args, **kwargs)
        capture.update(bsa_loss=float(out[3][0]), bsa_acc=float(out[4][0]), train_va=float(out[5][0]))
        return out
    namespace.update(vote_accuracy_per_video=wrap_vote, regression_per_blink=wrap_regression("regression_per_blink", "bsre"),
                     regression_per_video=wrap_regression("regression_per_video", "vre"), Train=wrap_train)
    return namespace


def check_against_original(capture, results):
    """별도 구현한 영상별 결정이 원본의 VA 와 원본이 출력한 영상별 줄과 일치하는지 검증한다. 어긋나면 RuntimeError.

    입력: capture - 원본 기록, results - video_results 결과
    반환: 없음
    """
    own_va = sum(1 for true, pred, _ in results if true == pred) / float(len(results))
    if abs(own_va - capture["va"]) > 1e-12:
        raise RuntimeError("별도 구현 VA %.9f 와 원본 VA %.9f 가 다릅니다." % (own_va, capture["va"]))
    pattern = re.compile(r"^(\d+): True label is :?\s*(\d+) and the detected label is (\d+)")
    printed = [pattern.match(line) for line in capture["lines"]]
    pairs = [(int(m.group(2)), int(m.group(3))) for m in printed if m]
    if pairs != [(true, pred) for true, pred, _ in results]:
        raise RuntimeError("원본이 출력한 영상별 예측 줄과 별도 구현 결과가 다릅니다(줄 %d개)." % len(pairs))


def rows_for(fold, ids, results):
    """영상별 CSV 행을 만든다.

    입력: fold, ids - 영상 식별자, results - (실제, 예측, 평균) list
    반환: PREDICTION_COLUMNS 의 dict list
    """
    return [{"fold": fold, "video_id": name, "true_class": true, "pred_class": pred, "regression_mean": "%.6f" % mean}
            for name, (true, pred, mean) in zip(ids, results)]


def check_row(run, fold, capture, n_videos):
    """재평가 지표와 metrics.csv 마지막 epoch 값을 나란히 놓은 fold_check 행을 만든다.

    입력: run, fold, capture - 원본 기록, n_videos - 영상 수
    반환: CHECK_COLUMNS 의 dict
    """
    last = run["last"][fold]
    diff = abs(capture["va"] - last["VA"])
    return {"seed": run["seed"], "fold": fold, "n_videos": n_videos, "VA_eval": "%.6f" % capture["va"],
            "VA_metrics_csv": "%.6f" % last["VA"], "VA_abs_diff": "%.2e" % diff, "VA_match": diff < VA_TOLERANCE,
            "BSA_loss_eval": "%.6f" % capture["bsa_loss"], "BSA_loss_csv": "%.6f" % last["BSA_loss"],
            "BSA_acc_eval": "%.6f" % capture["bsa_acc"], "BSA_acc_csv": "%.6f" % last["BSA_acc"],
            "BSRE_eval": "%.6f" % capture["bsre"], "BSRE_csv": "%.6f" % last["BSRE"],
            "VRE_eval": "%.6f" % capture["vre"], "VRE_csv": "%.6f" % last["VRE"]}


def evaluate_fold(context, run, fold):
    """한 시드·fold 의 체크포인트를 복원해 평가하고 영상별 결과와 점검 값을 돌려준다.

    입력: context - 실행 설정 dict, run - load_run 결과, fold - fold 번호
    반환: dict(rows(영상별 CSV 행), check(fold_check 행))
    """
    info = RB.validate_fold_inputs(context["data_dir"], context["start_dir"], fold)
    labels = np.load(os.path.join(context["data_dir"], "LabelsTest_30_Fold%d.npy" % fold), mmap_mode="r")
    ids = map_video_ids(context["start_dir"], fold, info["starts"], labels)
    prefix = run["fold_dirs"][fold].replace("\\", "/") + "/"
    data_prefix = context["data_dir"].replace("\\", "/") + "/"
    text, _report = RB.apply_substitutions(context["source"], build_substitutions(fold, prefix, data_prefix, info["starts"]))
    head, tail = split_patched(text)
    capture, capture_print = make_capture()
    namespace = prepare_namespace(head, capture, capture_print)
    exec(compile(tail, "Training.py[fold 반복부, load=True]", "exec"), namespace)
    results = video_results(capture["predictions"], capture["labels"], capture["starts"])
    check_against_original(capture, results)
    return {"rows": rows_for(fold, ids, results), "check": check_row(run, fold, capture, len(results))}


def confusion_from_rows(rows):
    """영상별 행에서 3x3 혼동행렬(행=정답 0/5/10, 열=예측 0/5/10)을 만든다.

    입력: rows - PREDICTION_COLUMNS 의 dict list
    반환: 3x3 정수 list
    """
    matrix = [[0, 0, 0] for _ in CLASS_POOL]
    for row in rows:
        matrix[CLASS_POOL.index(int(row["true_class"]))][CLASS_POOL.index(int(row["pred_class"]))] += 1
    return matrix


def confusion_csv_rows(matrix):
    """혼동행렬 CSV 행(정답 클래스마다 한 줄)을 만든다.

    입력: matrix - 3x3 list
    반환: CONFUSION_COLUMNS 의 dict list
    """
    accuracy = ST.class_accuracy(matrix)
    return [{"true_label": CLASS_POOL[i], "pred_0": matrix[i][0], "pred_5": matrix[i][1], "pred_10": matrix[i][2],
             "n_videos": sum(matrix[i]), "class_accuracy": "%.6f" % accuracy[i]} for i in range(3)]


def accuracy_csv_rows(label, matrix):
    """클래스별 영상 정확도 요약 CSV 행을 만든다.

    입력: label - 시드 이름('all' 은 합산), matrix - 3x3 list
    반환: ACCURACY_COLUMNS 의 dict list
    """
    accuracy = ST.class_accuracy(matrix)
    return [{"seed": label, "class": CLASS_POOL[i], "n_videos": sum(matrix[i]), "correct": matrix[i][i],
             "accuracy": "%.6f" % accuracy[i]} for i in range(3)]


def summary_lines(checks, matrices, hash_ok):
    """evaluation_summary.txt 줄(재평가 대 metrics.csv VA 대조, 클래스 정확도, 혼동행렬, 해시 확인)을 만든다.

    입력: checks - fold_check 행 list, matrices - {시드: 3x3}, hash_ok - 실행 전후 해시 불변 여부
    반환: 문자열 list
    """
    matched = sum(1 for row in checks if row["VA_match"])
    lines = ["== 재평가 VA 대 metrics.csv 마지막 epoch VA: 일치 %d/%d ==" % (matched, len(checks))]
    for row in checks:
        lines.append("시드%s fold%s: 재평가 %s, metrics.csv %s, 차이 %s, 일치 %s | VRE %s 대 %s, BSA acc %s 대 %s" % (
            row["seed"], row["fold"], row["VA_eval"], row["VA_metrics_csv"], row["VA_abs_diff"], row["VA_match"],
            row["VRE_eval"], row["VRE_csv"], row["BSA_acc_eval"], row["BSA_acc_csv"]))
    lines.append("실행 전후 third_party 와 학습 출력 파일 해시 불변: %s" % hash_ok)
    total = [[0, 0, 0] for _ in CLASS_POOL]
    for seed in sorted(matrices):
        lines.extend(ST.format_confusion("시드%d" % seed, matrices[seed]))
        total = ST.add_matrices(total, matrices[seed])
    lines.extend(ST.format_confusion("시드 합산", total))
    return lines


def write_csv(path, columns, rows):
    """CSV 를 새 파일로만 쓴다(이미 있으면 예외).

    입력: path, columns, rows
    반환: 없음
    """
    with open(path, "x", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def output_names(seeds):
    """쓸 출력 파일 이름 목록을 만든다.

    입력: seeds - 시드 list
    반환: 파일 이름 list
    """
    names = ["class_accuracy_summary.csv", "fold_check.csv", "evaluation_summary.txt"]
    for seed in seeds:
        names += ["video_predictions_seed%d.csv" % seed, "confusion_matrix_seed%d.csv" % seed]
    return names


def write_outputs(out_dir, runs, results, checks, lines):
    """집계 폴더에 모든 출력을 새로 쓴다.

    입력: out_dir, runs - load_run 결과 list, results - {시드: 영상별 행 list}, checks - fold_check 행, lines - 요약 줄
    반환: 없음
    """
    os.makedirs(out_dir, exist_ok=True)
    accuracy_rows, total = [], [[0, 0, 0] for _ in CLASS_POOL]
    for run in runs:
        seed = run["seed"]
        matrix = confusion_from_rows(results[seed])
        total = ST.add_matrices(total, matrix)
        write_csv(os.path.join(out_dir, "video_predictions_seed%d.csv" % seed), PREDICTION_COLUMNS, results[seed])
        write_csv(os.path.join(out_dir, "confusion_matrix_seed%d.csv" % seed), CONFUSION_COLUMNS, confusion_csv_rows(matrix))
        accuracy_rows += accuracy_csv_rows(str(seed), matrix)
    write_csv(os.path.join(out_dir, "class_accuracy_summary.csv"), ACCURACY_COLUMNS, accuracy_rows + accuracy_csv_rows("all", total))
    write_csv(os.path.join(out_dir, "fold_check.csv"), CHECK_COLUMNS, checks)
    with open(os.path.join(out_dir, "evaluation_summary.txt"), "x", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")


def dry_run(context, runs):
    """--dry-run: 모든 시드·fold 의 치환 개수와 영상 식별자 검증을 출력만 한다(TF 를 import 하지 않음).

    입력: context, runs
    반환: 없음
    """
    for run in runs:
        for fold in FOLDS:
            info = RB.validate_fold_inputs(context["data_dir"], context["start_dir"], fold)
            labels = np.load(os.path.join(context["data_dir"], "LabelsTest_30_Fold%d.npy" % fold), mmap_mode="r")
            ids = map_video_ids(context["start_dir"], fold, info["starts"], labels)
            substitutions = build_substitutions(fold, "<체크포인트 폴더>/", "<데이터 폴더>/", info["starts"])
            text, report = RB.apply_substitutions(context["source"], substitutions)
            head, tail = split_patched(text)
            compile(head, "정의부", "exec")
            compile(tail, "반복부", "exec")
            counts = ", ".join("%s %d" % (item["name"], item["found"]) for item in report)
            has_ckpt = os.path.isfile(os.path.join(run["fold_dirs"][fold], "my_model%d.index" % fold))
            print("시드%d fold%d: 영상 %d개 식별자 검증 통과, 체크포인트 있음 %s, 치환 개수 [%s], 컴파일 통과" % (
                run["seed"], fold, len(ids), has_ckpt, counts))
    print("--dry-run: 아무 파일도 쓰지 않았고 TensorFlow 도 import 하지 않았습니다.")


def evaluate_all(context, runs):
    """모든 시드·fold 를 평가한다(체크포인트 로드와 예측만).

    입력: context, runs
    반환: ({시드: 영상별 행 list}, fold_check 행 list)
    """
    results, checks = {}, []
    for run in runs:
        results[run["seed"]] = []
        for fold in FOLDS:
            outcome = evaluate_fold(context, run, fold)
            results[run["seed"]].extend(outcome["rows"])
            checks.append(outcome["check"])
            print("시드%d fold%d 재평가 VA %s (metrics.csv %s)" % (
                run["seed"], fold, outcome["check"]["VA_eval"], outcome["check"]["VA_metrics_csv"]))
            sys.stdout.flush()
    return results, checks


def main():
    """전체 흐름: 입력 검사, (--dry-run 이면 검증만) 실행 전 해시, 평가, 실행 후 해시 대조, 집계 폴더에 저장."""
    arguments = parse_arguments()
    roots, out_dir, data_dir, start_dir = validate_paths(arguments)
    runs = sorted((load_run(root) for root in roots), key=lambda run: run["seed"])
    context = {"data_dir": data_dir, "start_dir": start_dir, "source": RB.read_training_source()}
    common.check_manifest(REPO)
    if arguments.dry_run:
        dry_run(context, runs)
        return
    existing = [name for name in output_names([run["seed"] for run in runs]) if os.path.exists(os.path.join(out_dir, name))]
    if existing:
        sys.exit("이미 있는 출력 파일이 있어 덮어쓰지 않고 중단합니다: %s" % existing)
    before = snapshot_hashes(roots)
    sys.path.insert(0, UPSTREAM_DIR)
    results, checks = evaluate_all(context, runs)
    common.check_manifest(REPO)
    hash_ok = snapshot_hashes(roots) == before
    matrices = dict((seed, confusion_from_rows(rows)) for seed, rows in results.items())
    lines = summary_lines(checks, matrices, hash_ok)
    print("\n".join(lines))
    write_outputs(out_dir, runs, results, checks, lines)
    print("\n집계 폴더에 썼습니다.")


if __name__ == "__main__":
    main()
