# summarize_training.py
# 역할: B1 기준선 학습(reproduction_train_baseline.py, 시드 3개 x 5-fold)의 결과를 읽기 전용으로 집계한다.
#       정상성 점검(맨 앞), 마지막 epoch(80)의 영상 단위 VA/VRE 표, 시드 평균과 표준편차, 낙관적 참고치(최고 epoch),
#       71번째 epoch 참고용 클래스별 정확도와 혼동행렬, 기존 문서에 적힌 비교 값 인용을 만든다.
# 사용법: 저장소 루트에서 conda 환경 ghoddoosian_tf1 의 python 으로 실행한다.
#           python src/repro/summarize_training.py --roots <시드1 출력 루트> <시드2 출력 루트> <시드3 출력 루트> --out-dir <집계 폴더>
#         옵션: --expected-test 1328,2059,1641,1206,1648 (fold별 테스트 시퀀스 수 기대값, 기본값은 우리 start_indices 합)
# 입력: 각 루트 안의 실행 폴더/fold<k>/ 의 metrics.csv, final_metrics.json, run_config.json, train.log (읽기만 한다).
#       SMOKE 표식 파일이 있는 폴더는 건너뛰고 이름을 보고한다. 루트마다 SMOKE 가 아닌 실행 폴더가 하나여야 한다.
# 출력: 집계 폴더(학습 출력 루트 밖, 저장소 밖)에 summary.txt, fold_table.csv, confusion_matrix_epoch71.csv(산출 가능할 때).
#       이미 있는 출력 파일은 덮어쓰지 않고 중단한다. 경로는 인자로만 받는다.
"""B1 기준선 학습 결과 집계 도구(Python 3.6 호환, 표준 라이브러리만 사용).

주 지표는 마지막 epoch 의 값이다. 최고 epoch 값은 테스트 세트로 고른 값이므로 낙관적 참고치로만 따로 표시한다.
metrics.csv 에는 영상별 예측이 없어 마지막 epoch 의 혼동행렬은 산출할 수 없다. 대신 train.log 에 기록된 71번째 epoch
(epoch 번호 70)의 영상별 정답과 예측을 참고용으로 읽는다(기록되지 않은 값은 추정하지 않는다).
"""
import argparse
import csv
import json
import math
import os
import re
import statistics
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FOLDS = (1, 2, 3, 4, 5)
EXPECTED_EPOCHS = 80
LOG_EPOCH_STEP = 10  # 원본이 영상별 예측을 출력하는 epoch 번호(0부터)의 간격
SMOKE_MARKER = "SMOKE"
CLASS_LABELS = (0, 5, 10)
METRIC_COLUMNS = ["epoch", "train_loss_sum", "train_moving_acc", "BSA_loss", "BSA_acc", "VA", "BSRE", "VRE"]
EXPECTED_TOKENS = ("batch_size=64,num_epochs=80,", "hstate_size=[32,32,32,32],num_layers=4,step_size=30,drop_out_p=1.0,",
                   "lr=0.000053,th=1.253,", "0.1 * (tf.nn.l2_loss")
FOLD_TABLE_COLUMNS = ["seed", "fold", "last_epoch", "VA", "VRE", "BSA_acc", "BSA_loss", "BSRE", "best_VA", "best_VA_epoch",
                      "best_VRE", "best_VRE_epoch"]
CONFUSION_COLUMNS = ["seed", "true_label", "pred_0", "pred_5", "pred_10", "n_videos", "class_accuracy"]
# 기존 로그와 문서에 적힌 값(출처: 논문 arXiv 1904.07312v1 Table 2, 3; docs/IMPLEMENTATION_DIFF.md; experiments/02_fold_seeds/aggregate_seeds123.md)
PAPER = {"VA": {1: 0.64, 2: 0.61, 3: 0.70, 4: 0.64, 5: 0.67, "mean": 0.652}, "VRE": {1: 2.42, 2: 1.04, 3: 0.58, 4: 0.85, 5: 0.81, "mean": 1.14},
         "BSA_acc": {"mean": 0.54}, "BSRE": {"mean": 1.90}}
PUBLIC = {"VA": {1: 0.636, 2: 0.611, 3: 0.694, 4: 0.639, 5: 0.667, "mean": 0.6495}, "VRE": {1: 2.420, 2: 1.040, 3: 0.667, 4: 0.850, 5: 0.812, "mean": 1.158},
          "BSA_acc": {"mean": 0.541}, "BSRE": {"mean": 1.901}}
B0 = {"VA": {1: 0.556, 2: 0.602, 3: 0.481, 4: 0.574, 5: 0.566, "mean": 0.556, "sd": 0.036},
      "VRE": {1: 2.743, 2: 1.471, 3: 3.898, 4: 1.005, 5: 0.648, "mean": 1.953, "sd": 0.546},
      "BSA_acc": {"mean": 0.506}, "BSRE": {"mean": 2.874}}


def parse_arguments():
    """명령줄 옵션을 해석한다.

    반환: argparse.Namespace (roots, out_dir, expected_test)
    """
    parser = argparse.ArgumentParser(description="B1 기준선 학습 결과 집계(읽기 전용)")
    parser.add_argument("--roots", nargs=3, required=True, metavar=("ROOT1", "ROOT2", "ROOT3"),
                        help="학습 출력 루트 3개(시드 1, 2, 3)")
    parser.add_argument("--out-dir", dest="out_dir", required=True, help="집계 폴더(학습 출력 루트 밖, 저장소 밖)")
    parser.add_argument("--expected-test", dest="expected_test", default="1328,2059,1641,1206,1648",
                        help="fold별 테스트 시퀀스 수 기대값(쉼표로 구분 5개)")
    return parser.parse_args()


def is_inside(path, base):
    """path 가 base 와 같거나 그 안에 있는지 본다(Windows 대소문자 무시).

    입력: path, base - 경로
    반환: bool
    """
    path = os.path.normcase(os.path.realpath(path))
    base = os.path.normcase(os.path.realpath(base))
    try:
        return os.path.commonpath([path, base]) == base
    except ValueError:
        return False


def validate_paths(arguments):
    """학습 출력 루트와 집계 폴더를 검사한다(집계 폴더는 루트 밖, 저장소 밖이어야 한다).

    입력: arguments - parse_arguments 결과
    반환: (루트 실제 경로 list, 집계 폴더 실제 경로, 기대 시퀀스 수 tuple)
    """
    roots = [os.path.realpath(root) for root in arguments.roots]
    for root in roots:
        if not os.path.isdir(root):
            sys.exit("학습 출력 루트가 폴더가 아닙니다: %s" % os.path.basename(root))
    out_dir = os.path.realpath(arguments.out_dir)
    if is_inside(out_dir, REPO) or any(is_inside(out_dir, root) or is_inside(root, out_dir) for root in roots):
        sys.exit("집계 폴더는 저장소 밖이어야 하고 학습 출력 루트와 겹치면 안 됩니다.")
    try:
        expected = tuple(int(text) for text in arguments.expected_test.split(","))
    except ValueError:
        sys.exit("--expected-test 는 정수 5개를 쉼표로 구분해야 합니다.")
    if len(expected) != 5:
        sys.exit("--expected-test 는 정수 5개여야 합니다.")
    return roots, out_dir, expected


def is_smoke(folder):
    """폴더에 SMOKE 표식 파일이 있는지 본다.

    입력: folder - 폴더 경로
    반환: bool
    """
    return os.path.exists(os.path.join(folder, SMOKE_MARKER))


def find_run(root):
    """루트 안의 실행 폴더에서 SMOKE 가 아닌 fold 폴더를 모으고, 건너뛴 SMOKE 폴더 이름을 기록한다.

    입력: root - 학습 출력 루트
    반환: (실행 list[(실행 이름, {fold: 폴더})], 건너뛴 이름 list)
    """
    runs, skipped = [], []
    for name in sorted(os.listdir(root)):
        run_dir = os.path.join(root, name)
        if not os.path.isdir(run_dir):
            continue
        folds = {}
        for fold in FOLDS:
            fold_dir = os.path.join(run_dir, "fold%d" % fold)
            if not os.path.isdir(fold_dir):
                continue
            if is_smoke(run_dir) or is_smoke(fold_dir):
                skipped.append("%s/fold%d" % (name, fold))
            else:
                folds[fold] = fold_dir
        if folds:
            runs.append((name, folds))
    return runs, skipped


def read_metrics_rows(path):
    """metrics.csv 를 읽어 epoch 별 dict list 로 돌려준다(열 이름이 다르면 KeyError).

    입력: path - metrics.csv 경로
    반환: dict(METRIC_COLUMNS) list (epoch 는 int, 나머지는 float)
    """
    rows = []
    with open(path, "r", newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            values = dict((name, float(row[name])) for name in METRIC_COLUMNS)
            values["epoch"] = int(values["epoch"])
            rows.append(values)
    return rows


def read_json(path):
    """JSON 파일을 읽는다.

    입력: path - 경로
    반환: 파싱한 객체
    """
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_seed(root):
    """학습 출력 루트 하나에서 fold 5개의 지표, 최종 지표, 구성, 로그 경로를 읽는다.

    입력: root - 학습 출력 루트
    반환: dict(root_name, run_name, seed, folds{fold: dict(rows, final, config, log)}, skipped)
    """
    runs, skipped = find_run(root)
    if len(runs) != 1 or sorted(runs[0][1]) != list(FOLDS):
        sys.exit("루트 %s 에는 SMOKE 가 아닌 완전한 실행 폴더(fold1~5)가 하나여야 합니다(찾은 실행: %s)." % (
            os.path.basename(root), [name for name, _ in runs]))
    run_name, fold_dirs = runs[0]
    folds = {}
    for fold, fold_dir in fold_dirs.items():
        folds[fold] = {"rows": read_metrics_rows(os.path.join(fold_dir, "metrics.csv")),
                       "final": read_json(os.path.join(fold_dir, "final_metrics.json")),
                       "config": read_json(os.path.join(fold_dir, "run_config.json")),
                       "log": os.path.join(fold_dir, "train.log")}
    return {"root_name": os.path.basename(root), "run_name": run_name, "seed": folds[1]["config"]["seed"],
            "folds": folds, "skipped": skipped}


def check_row_counts(seeds):
    """점검 1: 모든 시드와 fold 의 metrics.csv 가 epoch 1~80 의 80행인지 본다.

    입력: seeds - load_seed 결과 list
    반환: (이름, 통과 여부, 상세)
    """
    bad = []
    for seed in seeds:
        for fold in FOLDS:
            epochs = [row["epoch"] for row in seed["folds"][fold]["rows"]]
            if epochs != list(range(1, EXPECTED_EPOCHS + 1)):
                bad.append("시드%d fold%d(행 %d)" % (seed["seed"], fold, len(epochs)))
    return "1. 모든 fold의 epoch 행 수 80(epoch 1~80)", not bad, ("문제: %s" % bad) if bad else "15개 모두 80행"


def check_finite(seeds):
    """점검 2: 모든 지표 열(손실 열 포함)에 NaN 과 inf 가 없는지 본다.

    입력: seeds
    반환: (이름, 통과 여부, 상세)
    """
    bad = []
    for seed in seeds:
        for fold in FOLDS:
            for row in seed["folds"][fold]["rows"]:
                broken = [name for name in METRIC_COLUMNS[1:] if not math.isfinite(row[name])]
                if broken:
                    bad.append("시드%d fold%d epoch%d %s" % (seed["seed"], fold, row["epoch"], broken))
    return "2. NaN/inf 없음(손실 열 포함 모든 지표 열)", not bad, ("문제: %s" % bad[:5]) if bad else "없음"


def config_signature(config):
    """run_config.json 에서 시드와 무관해야 하는 항목을 비교용 튜플로 뽑는다(치환된 소스 해시는 제외).

    입력: config - run_config dict
    반환: tuple
    """
    substitutions = tuple((item["name"], item["expected"], item["found"], tuple(item["lines"])) for item in config["substitutions"])
    return (tuple(config["hyperparameter_tokens"]), config["num_epochs"], config["training_py_sha256"], substitutions,
            tuple(sorted(config["input_sha256"].items())), config["n_train_sequences"], config["n_test_sequences"],
            config["start_indices_count"], config["smoke"])


def check_configs(seeds):
    """점검 3: 시드 3개의 구성이 시드만 다르고 저자 하이퍼파라미터(epoch 80, batch 64, lr 0.000053, 4x32)인지 본다.

    입력: seeds
    반환: (이름, 통과 여부, 상세)
    """
    bad = []
    values = [seed["seed"] for seed in seeds]
    if len(set(values)) != len(values):
        bad.append("시드 값이 서로 다르지 않음 %s" % values)
    for fold in FOLDS:
        signatures = [config_signature(seed["folds"][fold]["config"]) for seed in seeds]
        if len(set(signatures)) != 1:
            bad.append("fold%d 구성이 시드마다 다름" % fold)
    for seed in seeds:
        for fold in FOLDS:
            config = seed["folds"][fold]["config"]
            if tuple(config["hyperparameter_tokens"]) != EXPECTED_TOKENS or config["num_epochs"] != EXPECTED_EPOCHS or config["smoke"]:
                bad.append("시드%d fold%d 하이퍼파라미터 또는 스모크 표시 이상" % (seed["seed"], fold))
            if config["seed"] != seed["seed"]:
                bad.append("시드%d fold%d 시드 값 불일치" % (seed["seed"], fold))
    name = "3. 구성 동일(시드만 다름, epoch 80, batch 64, lr 0.000053, 4x32, 원본 해시, 치환 줄 번호, 입력 해시)"
    return name, not bad, ("문제: %s" % bad) if bad else "시드 %s, 5 fold 모두 같음" % sorted(values)


def check_test_counts(seeds, expected):
    """점검 4: 시드별 fold별 테스트 시퀀스 수가 우리 start_indices 합(기대값)과 같은지 본다.

    입력: seeds, expected - fold 1~5 의 기대 시퀀스 수 tuple
    반환: (이름, 통과 여부, 상세)
    """
    bad = []
    for seed in seeds:
        for fold in FOLDS:
            found = seed["folds"][fold]["config"]["n_test_sequences"]
            if found != expected[fold - 1]:
                bad.append("시드%d fold%d: %d(기대 %d)" % (seed["seed"], fold, found, expected[fold - 1]))
    return "4. 테스트 시퀀스 수 == start_indices 합 %s" % (list(expected),), not bad, ("문제: %s" % bad) if bad else "15개 모두 일치"


def check_loss_trend(seeds):
    """점검 5: epoch 1 대비 epoch 80 에서 BSA loss 가 줄었는지 시작과 끝 값만 본다.

    입력: seeds
    반환: (이름, 통과 여부, 상세)
    """
    notes, bad = [], []
    for seed in seeds:
        for fold in FOLDS:
            rows = seed["folds"][fold]["rows"]
            if not rows[-1]["BSA_loss"] < rows[0]["BSA_loss"]:
                bad.append("시드%d fold%d %.3f -> %.3f" % (seed["seed"], fold, rows[0]["BSA_loss"], rows[-1]["BSA_loss"]))
        first = [seed["folds"][fold]["rows"][0]["BSA_loss"] for fold in FOLDS]
        last = [seed["folds"][fold]["rows"][-1]["BSA_loss"] for fold in FOLDS]
        notes.append("시드%d fold별 %s" % (seed["seed"], ", ".join("%.2f->%.2f" % pair for pair in zip(first, last))))
    return "5. BSA loss 감소(epoch 1 -> 80)", not bad, ("줄지 않음: %s" % bad) if bad else "; ".join(notes)


def check_final_json(seeds):
    """참고 점검: final_metrics.json 이 metrics.csv 마지막 행과 같은지 본다.

    입력: seeds
    반환: (이름, 통과 여부, 상세)
    """
    bad = []
    for seed in seeds:
        for fold in FOLDS:
            data = seed["folds"][fold]
            for name in METRIC_COLUMNS:
                if abs(float(data["final"][name]) - data["rows"][-1][name]) > 1e-5:
                    bad.append("시드%d fold%d %s" % (seed["seed"], fold, name))
    return "참고. final_metrics.json == metrics.csv 마지막 행", not bad, ("문제: %s" % bad[:5]) if bad else "15개 모두 일치"


def run_checks(seeds, expected):
    """정상성 점검 1~5 와 참고 점검을 모두 실행한다.

    입력: seeds, expected
    반환: (이름, 통과 여부, 상세) list
    """
    return [check_row_counts(seeds), check_finite(seeds), check_configs(seeds), check_test_counts(seeds, expected),
            check_loss_trend(seeds), check_final_json(seeds)]


def last_values(seed, fold):
    """한 시드·fold 의 마지막 epoch 지표 dict 를 돌려준다.

    입력: seed - load_seed 결과, fold - fold 번호
    반환: dict(METRIC_COLUMNS)
    """
    return seed["folds"][fold]["rows"][-1]


def best_values(seed, fold):
    """한 시드·fold 의 최고 epoch 값(VA 최대, VRE 최소와 각 epoch, 같은 값이면 앞선 epoch)을 구한다(낙관적 참고치).

    입력: seed, fold
    반환: dict(best_VA, best_VA_epoch, best_VRE, best_VRE_epoch)
    """
    rows = seed["folds"][fold]["rows"]
    best_va = max(rows, key=lambda row: (row["VA"], -row["epoch"]))
    best_vre = min(rows, key=lambda row: (row["VRE"], row["epoch"]))
    return {"best_VA": best_va["VA"], "best_VA_epoch": best_va["epoch"], "best_VRE": best_vre["VRE"],
            "best_VRE_epoch": best_vre["epoch"]}


def mean_sd(values):
    """평균과 표본 표준편차(n-1)를 돌려준다.

    입력: values - 숫자 list
    반환: (평균, 표준편차). 값이 1개이면 표준편차는 nan
    """
    values = list(values)
    return statistics.mean(values), (statistics.stdev(values) if len(values) > 1 else float("nan"))


def metric_table_lines(seeds, title, getter):
    """시드별 fold별 값, 시드별 5-fold 평균, fold별 시드 평균과 표준편차를 마크다운 표 줄로 만든다.

    입력: seeds, title - 표 이름, getter - (seed, fold) -> 값 함수
    반환: 문자열 list
    """
    lines = [title, "| 시드 | Fold1 | Fold2 | Fold3 | Fold4 | Fold5 | 5-fold 평균 |", "|---|---|---|---|---|---|---|"]
    seed_means = []
    for seed in seeds:
        values = [getter(seed, fold) for fold in FOLDS]
        seed_means.append(statistics.mean(values))
        lines.append("| %d | %s | %.3f |" % (seed["seed"], " | ".join("%.3f" % v for v in values), seed_means[-1]))
    per_fold = [mean_sd([getter(seed, fold) for seed in seeds]) for fold in FOLDS]
    overall = mean_sd(seed_means)
    lines.append("| 시드 평균 | %s | %.3f |" % (" | ".join("%.3f" % m for m, _ in per_fold), overall[0]))
    lines.append("| 시드 SD(n-1) | %s | %.3f |" % (" | ".join("%.3f" % s for _, s in per_fold), overall[1]))
    return lines


def reference_lines(seeds):
    """기존 문서에 적힌 논문, 공개 체크포인트, 저자 npy 재학습(B0) 값과 B1 5-fold 평균을 나란히 보여 주는 줄을 만든다.

    입력: seeds
    반환: 문자열 list
    """
    lines = ["기존 문서에 적힌 값의 인용(출처: 논문 Table 2, 3, docs/IMPLEMENTATION_DIFF.md, experiments/02_fold_seeds/aggregate_seeds123.md).",
             "B1(우리 재추출 데이터)과 B0(저자 npy 재학습)는 서로 다른 데이터셋이므로 재현 성공/실패로 판정하지 않고 차이만 적는다.",
             "| 항목 | B1 시드 평균 ± SD | B0(저자 npy 재학습) | 공개 체크포인트 | 논문 |", "|---|---|---|---|---|"]
    for metric in ("VA", "VRE", "BSA_acc", "BSRE"):
        mean, sd = mean_sd([statistics.mean([last_values(seed, fold)[metric] for fold in FOLDS]) for seed in seeds])
        extra = " ± %.3f" % B0[metric]["sd"] if "sd" in B0[metric] else ""
        lines.append("| %s 5-fold 평균 | %.3f ± %.3f | %.3f%s | %.4g | %.4g |" % (
            metric, mean, sd, B0[metric]["mean"], extra, PUBLIC[metric]["mean"], PAPER[metric]["mean"]))
    for metric in ("VA", "VRE"):
        lines.append("fold별 %s(논문 / 공개 / B0 3시드 평균): " % metric + "; ".join(
            "Fold%d %.2f / %.3f / %.3f" % (fold, PAPER[metric][fold], PUBLIC[metric][fold], B0[metric][fold]) for fold in FOLDS))
    return lines


def fold_table_rows(seeds):
    """fold_table.csv 의 행(시드 x fold 15행)을 만든다.

    입력: seeds
    반환: FOLD_TABLE_COLUMNS 의 dict list
    """
    rows = []
    for seed in seeds:
        for fold in FOLDS:
            last = last_values(seed, fold)
            row = {"seed": seed["seed"], "fold": fold, "last_epoch": last["epoch"]}
            row.update(dict((name, "%.6f" % last[name]) for name in ("VA", "VRE", "BSA_acc", "BSA_loss", "BSRE")))
            best = best_values(seed, fold)
            row.update({"best_VA": "%.6f" % best["best_VA"], "best_VA_epoch": best["best_VA_epoch"],
                        "best_VRE": "%.6f" % best["best_VRE"], "best_VRE_epoch": best["best_VRE_epoch"]})
            rows.append(row)
    return rows


def optimistic_lines(seeds):
    """낙관적 참고치(fold별 최고 epoch 값, 테스트 기반 선택)와 마지막 epoch 값의 비교 줄을 만든다.

    입력: seeds
    반환: 문자열 list
    """
    lines = ["낙관적 참고치(테스트 세트로 epoch 를 고른 값이며 주 결과가 아니다)"]
    for seed in seeds:
        last_va = statistics.mean([last_values(seed, fold)["VA"] for fold in FOLDS])
        best_va = statistics.mean([best_values(seed, fold)["best_VA"] for fold in FOLDS])
        best_vre = statistics.mean([best_values(seed, fold)["best_VRE"] for fold in FOLDS])
        last_vre = statistics.mean([last_values(seed, fold)["VRE"] for fold in FOLDS])
        epochs = ", ".join("F%d %d" % (fold, best_values(seed, fold)["best_VA_epoch"]) for fold in FOLDS)
        lines.append("시드%d: 5-fold 평균 VA 마지막 %.3f / 최고 %.3f, VRE 마지막 %.3f / 최저 %.3f | VA 최고 epoch %s" % (
            seed["seed"], last_va, best_va, last_vre, best_vre, epochs))
    return lines


def parse_log_blocks(log_path):
    """train.log 에서 영상별 'True label is :x and the detected label is y' 기록을 블록(한 epoch)별로 읽는다.

    입력: log_path - train.log 경로
    반환: 블록 list(블록 = [(정답, 예측), ...]). 파일이 없으면 None
    """
    if not os.path.isfile(log_path):
        return None
    pattern = re.compile(r"^\s*(\d+): True label is :?\s*(\d+) and the detected label is (\d+)")
    blocks = []
    with open(log_path, "r", encoding="utf-8", errors="replace") as file:
        for line in file:
            match = pattern.match(line)
            if not match:
                continue
            if int(match.group(1)) == 1:
                blocks.append([])
            if blocks:
                blocks[-1].append((int(match.group(2)), int(match.group(3))))
    return blocks


def last_logged_pairs(seed, fold):
    """한 시드·fold 의 train.log 마지막 기록 epoch(71번째)의 영상별 (정답, 예측)을 돌려준다.

    입력: seed, fold
    반환: (쌍 list 또는 None, 사유 문자열). 블록 수나 영상 수, VA 가 맞지 않으면 None 과 사유
    """
    data = seed["folds"][fold]
    blocks = parse_log_blocks(data["log"])
    expected_blocks = len(range(0, EXPECTED_EPOCHS, LOG_EPOCH_STEP))
    if blocks is None:
        return None, "train.log 없음"
    if len(blocks) != expected_blocks:
        return None, "기록 블록 %d개(기대 %d개)" % (len(blocks), expected_blocks)
    pairs = blocks[-1]
    if len(pairs) != data["config"]["start_indices_count"]:
        return None, "영상 수 %d(기대 %d)" % (len(pairs), data["config"]["start_indices_count"])
    logged_va = sum(1 for true, pred in pairs if true == pred) / float(len(pairs))
    csv_va = data["rows"][(expected_blocks - 1) * LOG_EPOCH_STEP]["VA"]
    if abs(logged_va - csv_va) > 1e-5:
        return None, "로그 VA %.6f 와 metrics.csv 71번째 epoch VA %.6f 가 다름" % (logged_va, csv_va)
    return pairs, ""


def confusion_matrix(pairs):
    """(정답, 예측) 쌍에서 3x3 혼동행렬(행=정답 0/5/10, 열=예측 0/5/10)을 만든다.

    입력: pairs - (정답, 예측) list
    반환: 3x3 정수 list
    """
    matrix = [[0, 0, 0] for _ in CLASS_LABELS]
    for true, pred in pairs:
        matrix[CLASS_LABELS.index(true)][CLASS_LABELS.index(pred)] += 1
    return matrix


def add_matrices(first, second):
    """3x3 행렬을 원소별로 더한다.

    입력: first, second - 3x3 list
    반환: 합 3x3 list
    """
    return [[first[i][j] + second[i][j] for j in range(3)] for i in range(3)]


def confusion_by_seed(seeds):
    """시드별로 fold 5개의 71번째 epoch 혼동행렬을 합산한다. 하나라도 산출 불가이면 그 시드는 사유와 함께 None.

    입력: seeds
    반환: {시드: (3x3 합산 행렬 또는 None, 사유)}
    """
    result = {}
    for seed in seeds:
        total, reason = [[0, 0, 0] for _ in CLASS_LABELS], ""
        for fold in FOLDS:
            pairs, why = last_logged_pairs(seed, fold)
            if pairs is None:
                total, reason = None, "fold%d: %s" % (fold, why)
                break
            total = add_matrices(total, confusion_matrix(pairs))
        result[seed["seed"]] = (total, reason)
    return result


def class_accuracy(matrix):
    """혼동행렬에서 클래스별 영상 단위 정확도(대각/행 합)를 구한다.

    입력: matrix - 3x3 list
    반환: 정확도 list(영상이 없는 클래스는 nan)
    """
    return [(matrix[i][i] / float(sum(matrix[i])) if sum(matrix[i]) else float("nan")) for i in range(3)]


def format_confusion(label, matrix):
    """혼동행렬 한 개를 보고 줄로 만든다.

    입력: label - 시드 이름, matrix - 3x3 list
    반환: 문자열 list
    """
    accuracy = class_accuracy(matrix)
    lines = ["%s: 클래스별 정확도 0 %.3f, 5 %.3f, 10 %.3f" % (label, accuracy[0], accuracy[1], accuracy[2])]
    for index, label_value in enumerate(CLASS_LABELS):
        lines.append("  정답 %2d -> 예측 0 %3d, 5 %3d, 10 %3d (영상 %d)" % (
            label_value, matrix[index][0], matrix[index][1], matrix[index][2], sum(matrix[index])))
    return lines


def confusion_rows(label, matrix):
    """혼동행렬 한 개의 CSV 행(정답 클래스마다 한 줄)을 만든다.

    입력: label - 시드 이름('all' 은 합산), matrix - 3x3 list
    반환: CONFUSION_COLUMNS 의 dict list
    """
    accuracy = class_accuracy(matrix)
    return [{"seed": label, "true_label": CLASS_LABELS[i], "pred_0": matrix[i][0], "pred_5": matrix[i][1],
             "pred_10": matrix[i][2], "n_videos": sum(matrix[i]), "class_accuracy": "%.6f" % accuracy[i]} for i in range(3)]


def confusion_lines_and_rows(by_seed):
    """혼동행렬 보고 줄과 CSV 행을 만든다(마지막 epoch 는 산출 불가, 71번째 epoch 는 참고).

    입력: by_seed - confusion_by_seed 결과
    반환: (문자열 list, CSV 행 list). 산출 가능한 시드가 없으면 CSV 행은 빈 list
    """
    lines = ["클래스별 영상 단위 정확도와 혼동행렬: 마지막 epoch(80)는 metrics.csv 에 영상별 예측이 없어 산출 불가(추정하지 않음).",
             "참고(train.log 의 71번째 epoch, 마지막 epoch 가 아님), 행=정답 0/5/10, 열=예측 0/5/10, fold 5개 합산"]
    rows, usable = [], []
    for seed_value in sorted(by_seed):
        matrix, reason = by_seed[seed_value]
        if matrix is None:
            lines.append("시드%d: 산출 불가(%s)" % (seed_value, reason))
            continue
        usable.append(matrix)
        lines.extend(format_confusion("시드%d" % seed_value, matrix))
        rows.extend(confusion_rows(str(seed_value), matrix))
    if usable and len(usable) == len(by_seed):
        total = usable[0]
        for matrix in usable[1:]:
            total = add_matrices(total, matrix)
        lines.extend(format_confusion("시드 합산", total))
        rows.extend(confusion_rows("all", total))
    return lines, rows


def build_summary(seeds, checks, by_seed):
    """summary.txt 전체 줄(정상성 점검, 주 결과 표, 낙관적 참고치, 혼동행렬, 인용 값)을 만든다.

    입력: seeds, checks - run_checks 결과, by_seed - confusion_by_seed 결과
    반환: (문자열 list, 혼동행렬 CSV 행 list)
    """
    failed = [name for name, ok, _ in checks[:5] if not ok]
    lines = ["== 정상성 점검 (실패 %d개: %s) ==" % (len(failed), failed or "없음")]
    lines += ["[%s] %s: %s" % ("통과" if ok else "실패", name, detail) for name, ok, detail in checks]
    skipped = [name for seed in seeds for name in seed["skipped"]]
    lines += ["", "건너뛴 SMOKE 폴더: %s" % (skipped or "없음"), "", "== 주 결과: 마지막 epoch(80)의 영상 단위 지표 =="]
    lines += metric_table_lines(seeds, "VA(영상 단위 투표 정확도)", lambda s, f: last_values(s, f)["VA"]) + [""]
    lines += metric_table_lines(seeds, "VRE(영상 단위 회귀 오차)", lambda s, f: last_values(s, f)["VRE"]) + [""]
    lines += metric_table_lines(seeds, "BSA acc(시퀀스 정확도, 참고)", lambda s, f: last_values(s, f)["BSA_acc"]) + [""]
    lines += metric_table_lines(seeds, "BSRE(시퀀스 회귀 오차, 참고)", lambda s, f: last_values(s, f)["BSRE"]) + [""]
    lines += optimistic_lines(seeds) + [""]
    matrix_lines, rows = confusion_lines_and_rows(by_seed)
    return lines + matrix_lines + [""] + reference_lines(seeds), rows


def write_csv(path, columns, rows):
    """CSV 를 새 파일로만 쓴다(이미 있으면 예외).

    입력: path - 경로, columns - 열 이름, rows - dict list
    반환: 없음
    """
    with open(path, "x", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(out_dir, lines, seeds, confusion_rows_list):
    """집계 폴더에 summary.txt, fold_table.csv, (산출 가능하면) 혼동행렬 CSV 를 새로 쓴다. 이미 있으면 아무것도 쓰지 않고 중단한다.

    입력: out_dir, lines - 요약 줄, seeds, confusion_rows_list - 혼동행렬 CSV 행
    반환: 없음
    """
    names = ["summary.txt", "fold_table.csv"] + (["confusion_matrix_epoch71.csv"] if confusion_rows_list else [])
    existing = [name for name in names if os.path.exists(os.path.join(out_dir, name))]
    if existing:
        sys.exit("이미 있는 출력 파일이 있어 덮어쓰지 않고 중단합니다: %s" % existing)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.txt"), "x", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")
    write_csv(os.path.join(out_dir, "fold_table.csv"), FOLD_TABLE_COLUMNS, fold_table_rows(seeds))
    if confusion_rows_list:
        write_csv(os.path.join(out_dir, "confusion_matrix_epoch71.csv"), CONFUSION_COLUMNS, confusion_rows_list)


def main():
    """전체 흐름: 인자 검사, 시드 3개 읽기, 정상성 점검, 집계 보고, 집계 폴더에 저장."""
    arguments = parse_arguments()
    roots, out_dir, expected = validate_paths(arguments)
    seeds = sorted((load_seed(root) for root in roots), key=lambda seed: seed["seed"])
    checks = run_checks(seeds, expected)
    lines, matrix_rows = build_summary(seeds, checks, confusion_by_seed(seeds))
    print("\n".join(lines))
    write_outputs(out_dir, lines, seeds, matrix_rows)
    print("\n집계 폴더에 썼습니다.")


if __name__ == "__main__":
    main()
