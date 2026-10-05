# estimate_baseline_uncertainty.py
# 역할: B1 기준선의 영상 단위 지표가 표본(피험자 57명, 영상 167개) 때문에 얼마나 불확실한지, 피험자를 군집으로 하는
#       부트스트랩(피험자 재표집, 같은 피험자의 영상은 함께)으로 95% 구간을 잰다. 시드별 VA, B1 시드 평균 정답률, 짝지은
#       차이 ΔVA(B1 시드1 대 시드2, B1 대 RF, B1 대 로지스틱), 클래스 5 정확도의 구간을 낸다. 읽기 전용이다.
# 사용법: 저장소 루트에서 conda 환경 ghoddoosian_tf1 의 python 으로 실행한다(numpy 만 사용).
#           python src/repro/estimate_baseline_uncertainty.py --b1-predictions <시드1 CSV> <시드2 CSV> <시드3 CSV>
#                  --diagnose-predictions <진단 CSV> --out-dir <집계 폴더>
# 입력: B1 의 video_predictions_seed<s>.csv 3개(열 fold, video_id, true_class, pred_class, regression_mean)와
#       진단의 video_predictions.csv(열 fold, video_id, true_class, pred_logreg, pred_rf, pred_majority). 읽기만 한다.
# 출력(집계 폴더, 저장소 밖, 입력 폴더 밖): uncertainty_intervals.csv, uncertainty_summary.txt. 이미 있으면 덮어쓰지 않고 중단한다.
#       실행 전후 입력 CSV 의 SHA-256 이 같은지 확인한다. 난수 시드 0, 재표집 10000회, percentile 95% 구간으로 고정이다.
"""피험자 군집 부트스트랩으로 기준선 불확실성을 재는 도구(Python 3.6 호환, numpy 만 사용).

통계는 재표집된 영상 전체(중복 포함)의 정답률(합산)이다. 같은 재표집 인덱스를 모든 모델과 차이에 써서 짝지은 비교가 된다.
이 구간은 같은 테스트 피험자 집합이 표집된다는 불확실성(표본 오차)이며 학습 난수의 변동은 시드 3개의 차이로만 일부 드러난다.
"""
import sys

sys.dont_write_bytecode = True

import argparse
import csv
import os
import re

import numpy as np

import common

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SEED = 0
RESAMPLES = 10000
VIDEO_PATTERN = re.compile(r"^Fold(\d)_part(\d)/(\d{2})/(0|5|10)$")
MODELS = ["b1_seed1", "b1_seed2", "b1_seed3", "logreg", "rf"]
INTERVAL_COLUMNS = ["quantity", "estimate", "ci_low", "ci_high", "half_width", "contains_zero", "n_videos_or_subjects", "note"]


def parse_arguments():
    """명령줄 옵션을 해석한다.

    반환: argparse.Namespace (b1, diagnose, out_dir)
    """
    parser = argparse.ArgumentParser(description="피험자 군집 부트스트랩으로 기준선 불확실성 추정(읽기 전용)")
    parser.add_argument("--b1-predictions", nargs=3, required=True, dest="b1", metavar=("SEED1", "SEED2", "SEED3"),
                        help="B1 의 video_predictions_seed<s>.csv 3개(시드 1, 2, 3 순서)")
    parser.add_argument("--diagnose-predictions", required=True, dest="diagnose", help="진단의 video_predictions.csv")
    parser.add_argument("--out-dir", required=True, dest="out_dir", help="집계 폴더(저장소 밖, 입력 폴더 밖)")
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
    """입력 CSV 와 집계 폴더를 검사한다(집계 폴더는 저장소 밖이고 입력 파일이 있는 폴더 밖이어야 한다).

    입력: arguments - parse_arguments 결과
    반환: (입력 CSV 경로 list(B1 3개 + 진단), 집계 폴더)
    """
    inputs = [os.path.realpath(path) for path in list(arguments.b1) + [arguments.diagnose]]
    for path in inputs:
        if not os.path.isfile(path):
            sys.exit("입력 CSV 가 없습니다: %s" % os.path.basename(path))
    out_dir = os.path.realpath(arguments.out_dir)
    if is_inside(out_dir, REPO) or any(is_inside(out_dir, os.path.dirname(path)) for path in inputs):
        sys.exit("집계 폴더는 저장소 밖이어야 하고 입력 CSV 가 있는 폴더와 겹치면 안 됩니다.")
    return inputs, out_dir


def read_rows(path):
    """CSV 를 dict list 로 읽는다.

    입력: path - CSV 경로
    반환: dict list
    """
    with open(path, "r", newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def load_videos(inputs):
    """입력 CSV 4개를 같은 영상 순서로 맞춰 영상별 정답 여부와 피험자를 만든다. 영상 집합이나 순서가 다르면 중단한다.

    입력: inputs - 경로 list(B1 시드 1, 2, 3, 진단)
    반환: dict(video_ids, subjects, labels, correct{모델: 0/1 배열})
    """
    tables = [read_rows(path) for path in inputs]
    ids = [row["video_id"] for row in tables[0]]
    if any([row["video_id"] for row in table] != ids for table in tables[1:]):
        sys.exit("입력 CSV 들의 영상 집합이나 순서가 다릅니다.")
    bad = [name for name in ids if not VIDEO_PATTERN.match(name)]
    if bad:
        sys.exit("video_id 형식이 예상과 다릅니다: %s" % bad[:3])
    labels = np.array([int(row["true_class"]) for row in tables[0]])
    correct = {}
    for index, name in enumerate(MODELS[:3]):
        correct[name] = np.array([int(row["pred_class"]) == int(row["true_class"]) for row in tables[index]], dtype=float)
    for key, name in (("pred_logreg", "logreg"), ("pred_rf", "rf")):
        correct[name] = np.array([int(row[key]) == int(row["true_class"]) for row in tables[3]], dtype=float)
    return {"video_ids": ids, "subjects": [VIDEO_PATTERN.match(name).group(3) for name in ids], "labels": labels, "correct": correct}


def subject_table(data):
    """피험자별 영상 수, 모델별 맞춘 영상 수, 클래스 5 영상 수와 모델별 맞춘 클래스 5 영상 수를 모은다.

    입력: data - load_videos 결과
    반환: (피험자 list, (피험자 수, 12) 배열). 열 = [영상 수, 모델별 맞춘 수 5개, 클래스 5 영상 수, 모델별 맞춘 클래스 5 수 5개]
    """
    subjects = sorted(set(data["subjects"]))
    table = np.zeros((len(subjects), 12))
    for index, subject in enumerate(subjects):
        mask = np.array([name == subject for name in data["subjects"]])
        five = mask & (data["labels"] == 5)
        table[index, 0] = mask.sum()
        table[index, 6] = five.sum()
        for column, model in enumerate(MODELS):
            table[index, 1 + column] = data["correct"][model][mask].sum()
            table[index, 7 + column] = data["correct"][model][five].sum()
    return subjects, table


def bootstrap_sums(table):
    """피험자를 복원 추출로 재표집해(RandomState(0), 10000회) 재표집마다 열 합계를 구한다.

    입력: table - subject_table 의 배열
    반환: (10000, 12) 배열
    """
    rng = np.random.RandomState(SEED)
    indices = rng.randint(0, len(table), size=(RESAMPLES, len(table)))
    return table[indices].sum(axis=1)


def accuracy_columns(sums):
    """합계(재표집 또는 원 표본)에서 모델별 정답률(VA)과 클래스 5 정확도를 계산한다(B1 시드 평균 포함).

    입력: sums - (행, 12) 배열(원 표본은 1행)
    반환: (va {이름: 배열}, acc5 {이름: 배열})
    """
    va = dict((model, sums[:, 1 + column] / sums[:, 0]) for column, model in enumerate(MODELS))
    acc5 = dict((model, sums[:, 7 + column] / sums[:, 6]) for column, model in enumerate(MODELS))
    va["b1_mean"] = (va["b1_seed1"] + va["b1_seed2"] + va["b1_seed3"]) / 3.0
    acc5["b1_mean"] = (acc5["b1_seed1"] + acc5["b1_seed2"] + acc5["b1_seed3"]) / 3.0
    return va, acc5


def interval_row(name, estimate, samples, count, note, is_delta):
    """한 통계의 점추정과 percentile 95% 구간을 CSV 행으로 만든다.

    입력: name - 이름, estimate - 점추정, samples - 재표집 값 배열, count - 영상 수, note, is_delta - 차이 여부
    반환: INTERVAL_COLUMNS 의 dict
    """
    low, high = np.percentile(samples, [2.5, 97.5])
    return {"quantity": name, "estimate": "%.4f" % estimate, "ci_low": "%.4f" % low, "ci_high": "%.4f" % high,
            "half_width": "%.4f" % ((high - low) / 2.0), "contains_zero": (bool(low <= 0.0 <= high) if is_delta else ""),
            "n_videos_or_subjects": count, "note": note}


def build_intervals(table):
    """요청한 모든 구간(시드별 VA, B1 시드 평균, ΔVA 3개, 클래스 5 정확도)을 만든다.

    입력: table - subject_table 의 배열
    반환: INTERVAL_COLUMNS 의 dict list
    """
    n_videos = int(table[:, 0].sum())
    boot_va, boot_acc5 = accuracy_columns(bootstrap_sums(table))
    est_va, est_acc5 = accuracy_columns(table.sum(axis=0, keepdims=True))
    rows = []
    for name in ("b1_seed1", "b1_seed2", "b1_seed3", "b1_mean", "logreg", "rf"):
        rows.append(interval_row("VA " + name, float(est_va[name][0]), boot_va[name], n_videos, "합산 정답률", False))
    pairs = (("dVA b1_seed1 - b1_seed2", "b1_seed1", "b1_seed2"), ("dVA b1_mean - rf", "b1_mean", "rf"),
             ("dVA b1_mean - logreg", "b1_mean", "logreg"))
    for name, left, right in pairs:
        rows.append(interval_row(name, float(est_va[left][0] - est_va[right][0]), boot_va[left] - boot_va[right], n_videos, "짝지은 차이", True))
    for name in ("b1_seed1", "b1_seed2", "b1_seed3", "b1_mean", "logreg", "rf"):
        rows.append(interval_row("Acc5 " + name, float(est_acc5[name][0]), boot_acc5[name], int(table[:, 6].sum()), "클래스 5 영상 정확도", False))
    return rows


def write_outputs(out_dir, rows, lines):
    """집계 폴더에 구간 CSV 와 요약 txt 를 새로 쓴다(이미 있으면 중단).

    입력: out_dir, rows - 구간 행, lines - 요약 줄
    반환: 없음
    """
    names = ["uncertainty_intervals.csv", "uncertainty_summary.txt"]
    existing = [name for name in names if os.path.exists(os.path.join(out_dir, name))]
    if existing:
        sys.exit("이미 있는 출력 파일이 있어 덮어쓰지 않고 중단합니다: %s" % existing)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, names[0]), "x", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=INTERVAL_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    with open(os.path.join(out_dir, names[1]), "x", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")


def summary_lines(rows, hash_ok, n_subjects, n_videos):
    """요약 txt 줄(방법, 구간 표, 입력 해시 불변)을 만든다.

    입력: rows - 구간 행, hash_ok - 입력 해시 불변 여부, n_subjects - 피험자 수, n_videos - 영상 수
    반환: 문자열 list
    """
    lines = ["== 기준선 불확실성 (피험자 군집 부트스트랩, 재표집 %d회, 시드 %d, percentile 95%% 구간) ==" % (RESAMPLES, SEED),
             "피험자 %d명, 영상 %d개. 통계는 재표집 영상의 합산 정답률이고 같은 재표집 인덱스를 모든 모델과 차이에 쓴다(짝지은 비교)." % (n_subjects, n_videos),
             "점추정은 합산 정답률이며 fold 평균 VA 와는 fold 크기(32~35개) 차이로 조금 다를 수 있다. 입력 CSV 해시 실행 전후 불변: %s" % hash_ok,
             "| 통계 | 점추정 | 95% 구간 | 반폭 | 0 포함 | 비고 |", "|---|---|---|---|---|---|"]
    for row in rows:
        lines.append("| %s | %s | [%s, %s] | %s | %s | %s |" % (row["quantity"], row["estimate"], row["ci_low"], row["ci_high"],
                                                             row["half_width"], row["contains_zero"], row["note"]))
    return lines


def main():
    """전체 흐름: 입력 검사, 입력 해시, 부트스트랩, 입력 해시 대조, 집계 폴더에 저장."""
    arguments = parse_arguments()
    inputs, out_dir = validate_paths(arguments)
    before = [common.sha256_of_file(path) for path in inputs]
    data = load_videos(inputs)
    subjects, table = subject_table(data)
    rows = build_intervals(table)
    hash_ok = [common.sha256_of_file(path) for path in inputs] == before
    lines = summary_lines(rows, hash_ok, len(subjects), int(table[:, 0].sum()))
    print("\n".join(lines))
    write_outputs(out_dir, rows, lines)
    print("\n집계 폴더에 썼습니다.")


if __name__ == "__main__":
    main()
