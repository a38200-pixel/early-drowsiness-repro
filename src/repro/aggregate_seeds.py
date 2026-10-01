# aggregate_seeds.py
# 역할: 재학습 3시드(Fold1은 시드 1~5도 함께)의 epoch 80 지표를 모아 시드별 5-fold 평균, 시드 간 평균±표본 SD(n-1),
#       fold별 (재학습 - 공개 체크포인트) 차이, 공개 체크포인트 값과 논문 Table 2·3의 반올림 일치표를 만든다.
# 사용법: python src\repro\aggregate_seeds.py
#         입력: experiments\01_fold1_seeds\seed<N>\metrics.csv (N=1~5), experiments\02_fold_seeds\fold<F>_seed<S>\metrics.csv (F=2~5, S=1~3),
#               experiments\00_reproduction\regression_eval\pretrained_5fold_results.csv
#         출력: experiments\02_fold_seeds\aggregate_seeds123.md, aggregate_seeds123.csv (실행할 때마다 새로 만든다)
"""재학습 3시드 집계 모듈.

보고값은 각 metrics.csv의 마지막 행(epoch 80)이다. 마지막 epoch가 80이 아니거나 파일이 없으면 값을 고르지 않고 중단한다.
표준편차는 표본 표준편차(n-1)이며 3개 시드로는 참고 수준이다. 결론 문구는 쓰지 않고 숫자와 일치/불일치만 낸다.
표준 라이브러리(csv, statistics)만 사용한다.
"""
import csv
import os
import statistics
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FOLD1_DIR = os.path.join(REPO_ROOT, "experiments", "01_fold1_seeds")
FOLDS_DIR = os.path.join(REPO_ROOT, "experiments", "02_fold_seeds")
PRETRAINED_CSV = os.path.join(REPO_ROOT, "experiments", "00_reproduction", "regression_eval",
                              "pretrained_5fold_results.csv")
OUTPUT_MD = os.path.join(FOLDS_DIR, "aggregate_seeds123.md")
OUTPUT_CSV = os.path.join(FOLDS_DIR, "aggregate_seeds123.csv")
EXPECTED_EPOCHS = 80
FOLDS = [1, 2, 3, 4, 5]
MAIN_SEEDS = [1, 2, 3]
FOLD1_SEEDS = [1, 2, 3, 4, 5]
# (표에 쓸 이름, metrics.csv 열 이름, pretrained_5fold_results.csv 열 이름) - 표 열 순서 그대로
METRICS = [
    ("BSA loss", "BSA_loss", "BSA_loss"),
    ("BSA acc", "BSA_acc", "BSA_accuracy"),
    ("VA", "VA", "VA"),
    ("BSRE", "BSRE", "BSRE"),
    ("VRE", "VRE", "VRE"),
]
# 논문 값. 출처: arXiv 1904.07312v1, Table 3(fold별 VA/VRE)과 Table 2(평균), p.8
PAPER_TABLE3 = {
    1: {"VA": 0.64, "VRE": 2.42},
    2: {"VA": 0.61, "VRE": 1.04},
    3: {"VA": 0.70, "VRE": 0.58},
    4: {"VA": 0.64, "VRE": 0.85},
    5: {"VA": 0.67, "VRE": 0.81},
}
PAPER_TABLE2 = {"BSRE": 1.90, "VRE": 1.14, "BSA acc": 0.54, "VA": 0.652}  # BSA 54%, VA 65.2%
CSV_HEADER = ["표", "행", "BSA loss", "BSA acc", "VA", "BSRE", "VRE", "비고"]
FOLD_ROW_NAMES = [("Fold1", 1), ("Fold1(시드1~5)", 1), ("Fold2", 2), ("Fold3", 3), ("Fold4", 4), ("Fold5", 5)]


def metrics_path_of(fold, seed):
    """fold와 시드에 해당하는 metrics.csv 경로를 돌려준다. Fold1은 01_fold1_seeds, 나머지는 02_fold_seeds에 있다.

    입력: fold - 1~5, seed - 정수 시드
    반환: metrics.csv 경로 문자열
    """
    if fold == 1:
        return os.path.join(FOLD1_DIR, "seed%d" % seed, "metrics.csv")
    return os.path.join(FOLDS_DIR, "fold%d_seed%d" % (fold, seed), "metrics.csv")


def read_final_row(path):
    """metrics.csv의 마지막 행을 읽고, 마지막 epoch가 80인지 확인한다.

    입력: path - metrics.csv 경로
    반환: 열 이름을 키로 하는 dict. 파일이 없거나 마지막 epoch가 80이 아니면 메시지를 출력하고 중단한다
    """
    if not os.path.isfile(path):
        sys.exit("입력 파일이 없습니다: %s" % path)
    last_row = None
    with open(path, "r", newline="") as file:
        for row in csv.DictReader(file):
            last_row = row
    if last_row is None or int(last_row["epoch"]) != EXPECTED_EPOCHS:
        found = "행 없음" if last_row is None else "epoch %s" % last_row["epoch"]
        sys.exit("마지막 epoch가 %d이 아닙니다(%s): %s" % (EXPECTED_EPOCHS, found, path))
    return last_row


def load_run(fold, seed):
    """한 실행(fold, 시드)의 epoch 80 지표를 {표 이름: float}로 읽는다.

    입력: fold - 1~5, seed - 정수 시드
    반환: {"BSA loss": 값, ...} dict
    """
    row = read_final_row(metrics_path_of(fold, seed))
    values = {}
    for label, metrics_column, _pretrained_column in METRICS:
        values[label] = float(row[metrics_column])
    return values


def load_all_runs():
    """필요한 모든 실행을 읽는다. Fold1은 시드 1~5, Fold2~5는 시드 1~3이다.

    입력: 없음
    반환: {(fold, 시드): {표 이름: float}} dict
    """
    runs = {}
    for fold in FOLDS:
        seeds = FOLD1_SEEDS if fold == 1 else MAIN_SEEDS
        for seed in seeds:
            runs[(fold, seed)] = load_run(fold, seed)
    return runs


def read_pretrained():
    """공개 체크포인트 평가 CSV에서 fold별 지표를 읽는다.

    입력: 없음
    반환: {fold(int): {표 이름: float}} dict. fold 1~5가 하나라도 없으면 중단한다
    """
    by_fold = {}
    with open(PRETRAINED_CSV, "r", newline="") as file:
        for row in csv.DictReader(file):
            if row["Fold"].isdigit():
                values = {}
                for label, _metrics_column, pretrained_column in METRICS:
                    values[label] = float(row[pretrained_column])
                by_fold[int(row["Fold"])] = values
    for fold in FOLDS:
        if fold not in by_fold:
            sys.exit("공개 체크포인트 CSV에 Fold%d 행이 없습니다" % fold)
    return by_fold


def average_runs(runs, keys):
    """지정한 실행들의 지표별 평균을 계산한다.

    입력: runs - {키: {표 이름: float}}, keys - 평균에 넣을 키 list
    반환: {표 이름: 평균}
    """
    averages = {}
    for label, _metrics_column, _pretrained_column in METRICS:
        values = []
        for key in keys:
            values.append(runs[key][label])
        averages[label] = statistics.mean(values)
    return averages


def seed_fold_means(runs):
    """시드 1~3 각각의 5-fold 평균을 계산한다(Fold1은 해당 시드 값).

    입력: runs - load_all_runs() 결과
    반환: {시드: {표 이름: 평균}}
    """
    result = {}
    for seed in MAIN_SEEDS:
        keys = []
        for fold in FOLDS:
            keys.append((fold, seed))
        result[seed] = average_runs(runs, keys)
    return result


def mean_and_sd(per_seed):
    """시드별 5-fold 평균들의 평균과 표본 표준편차(n-1)를 계산한다.

    입력: per_seed - seed_fold_means() 결과
    반환: {표 이름: (평균, 표본 SD)}
    """
    result = {}
    for label, _metrics_column, _pretrained_column in METRICS:
        values = []
        for seed in MAIN_SEEDS:
            values.append(per_seed[seed][label])
        result[label] = (statistics.mean(values), statistics.stdev(values))
    return result


def fold_retrained_means(runs):
    """fold별 재학습 평균을 계산한다. Fold1은 시드 1~3 평균과 시드 1~5 평균을 모두 만든다.

    입력: runs - load_all_runs() 결과
    반환: {이름: {표 이름: 평균}} dict. 이름은 'Fold<F>' 또는 'Fold1(시드1~5)'
    """
    result = {}
    for fold in FOLDS:
        keys = []
        for seed in MAIN_SEEDS:
            keys.append((fold, seed))
        result["Fold%d" % fold] = average_runs(runs, keys)
    all_keys = []
    for seed in FOLD1_SEEDS:
        all_keys.append((1, seed))
    result["Fold1(시드1~5)"] = average_runs(runs, all_keys)
    return result


def difference(retrained, public):
    """두 지표 dict의 차이(재학습 - 공개)를 계산한다.

    입력: retrained, public - {표 이름: float}
    반환: {표 이름: 차이}
    """
    result = {}
    for label, _metrics_column, _pretrained_column in METRICS:
        result[label] = retrained[label] - public[label]
    return result


def same_at_two_decimals(paper_value, measured_value):
    """두 값을 소수 둘째 자리로 반올림했을 때 같은지 확인한다.

    입력: paper_value - 논문 값, measured_value - 측정 값
    반환: 같으면 True
    """
    return "%.2f" % paper_value == "%.2f" % measured_value


def make_comparison_item(name, paper, measured):
    """논문 대조 한 칸을 dict로 만든다.

    입력: name - 항목 이름, paper - 논문 값, measured - 공개 체크포인트 값
    반환: name, paper, measured, matched(bool)를 가진 dict
    """
    return {"name": name, "paper": paper, "measured": measured,
            "matched": same_at_two_decimals(paper, measured)}


def paper_comparison(pretrained, public_mean):
    """공개 체크포인트 평가값과 논문 Table 3(fold별 VA·VRE), Table 2(평균)를 반올림 비교한다.

    입력: pretrained - read_pretrained() 결과, public_mean - 5-fold 평균 {표 이름: float}
    반환: 대조 칸 dict의 list(Table 3 10칸 + Table 2 4칸)
    """
    items = []
    for fold in FOLDS:
        for label in ["VA", "VRE"]:
            items.append(make_comparison_item("Fold%d %s" % (fold, label), PAPER_TABLE3[fold][label],
                                              pretrained[fold][label]))
    for label in ["BSRE", "VRE", "BSA acc", "VA"]:
        items.append(make_comparison_item("평균 %s" % label, PAPER_TABLE2[label], public_mean[label]))
    return items


def fmt(value, with_sign=False):
    """지표 값을 소수 3자리 문자열로 바꾼다.

    입력: value - float, with_sign - True면 부호를 붙인다(차이 표용)
    반환: 문자열
    """
    if with_sign:
        return "%+.3f" % value
    return "%.3f" % value


def row_cells(values, with_sign=False):
    """지표 dict를 표 열 순서대로 문자열 칸 list로 바꾼다.

    입력: values - {표 이름: float}, with_sign - 부호 표시 여부
    반환: 문자열 list
    """
    cells = []
    for label, _metrics_column, _pretrained_column in METRICS:
        cells.append(fmt(values[label], with_sign))
    return cells


def metric_labels():
    """표 열 순서대로 지표 이름 list를 돌려준다.

    입력: 없음
    반환: 지표 이름 list
    """
    labels = []
    for label, _metrics_column, _pretrained_column in METRICS:
        labels.append(label)
    return labels


def md_table(header, rows):
    """머리와 행(문자열 list의 list)으로 마크다운 표 줄 목록을 만든다.

    입력: header - 열 이름 list, rows - 행 list
    반환: 문자열 list
    """
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def seed_table_lines(per_seed, stats, public_mean):
    """시드별 5-fold 평균, 평균±SD, 공개, 논문(Table 2)을 나란히 둔 표 줄 목록을 만든다.

    입력: per_seed - 시드별 평균, stats - 평균·SD, public_mean - 공개 5-fold 평균
    반환: 문자열 list
    """
    header = ["지표", "시드1", "시드2", "시드3", "평균 ± SD(n-1, n=3)", "공개 체크포인트", "논문(Table 2)"]
    rows = []
    for label in metric_labels():
        paper = "논문에 없음" if label == "BSA loss" else "%.3f" % PAPER_TABLE2[label]
        mean_value, sd_value = stats[label]
        rows.append([label, fmt(per_seed[1][label]), fmt(per_seed[2][label]), fmt(per_seed[3][label]),
                     "%s ± %s" % (fmt(mean_value), fmt(sd_value)), fmt(public_mean[label]), paper])
    return md_table(header, rows)


def fold_rows(fold_means, pretrained):
    """fold별 표의 행(재학습, 공개, 차이)을 만든다. Fold1은 시드 1~3과 시드 1~5를 모두 넣는다.

    입력: fold_means - fold_retrained_means() 결과, pretrained - read_pretrained() 결과
    반환: 행(문자열 list)의 list
    """
    rows = []
    for fold in FOLDS:
        name = "Fold%d" % fold
        rows.append([name + " 재학습(시드1~3 평균)"] + row_cells(fold_means[name]))
        if fold == 1:
            rows.append(["Fold1 재학습(시드1~5 평균)"] + row_cells(fold_means["Fold1(시드1~5)"]))
        rows.append([name + " 공개"] + row_cells(pretrained[fold]))
        rows.append([name + " 차이(시드1~3 평균 - 공개)"] + row_cells(difference(fold_means[name], pretrained[fold]), True))
        if fold == 1:
            rows.append(["Fold1 차이(시드1~5 평균 - 공개)"] +
                        row_cells(difference(fold_means["Fold1(시드1~5)"], pretrained[1]), True))
    return rows


def paper_table_lines(comparison):
    """논문 대조 결과를 마크다운 표 줄 목록으로 만든다.

    입력: comparison - paper_comparison() 결과
    반환: 문자열 list
    """
    rows = []
    for item in comparison:
        result = "일치" if item["matched"] else "불일치"
        rows.append([item["name"], "%.3f" % item["paper"], fmt(item["measured"]), "%.2f" % item["measured"], result])
    return md_table(["항목", "논문", "공개(원값)", "공개(소수 둘째 자리 반올림)", "판정"], rows)


def unmatched_names(comparison):
    """불일치한 칸의 이름 목록을 만든다.

    입력: comparison - paper_comparison() 결과
    반환: 이름 list
    """
    names = []
    for item in comparison:
        if not item["matched"]:
            names.append(item["name"])
    return names


def build_markdown(per_seed, stats, public_mean, fold_means, pretrained, comparison):
    """집계 보고서 전체를 마크다운 줄 목록으로 만든다.

    입력: 각 집계 결과(시드별 평균, 평균·SD, 공개 평균, fold별 평균, 공개 fold별 값, 논문 대조)
    반환: 문자열 list
    """
    unmatched = unmatched_names(comparison)
    matched_count = len(comparison) - len(unmatched)
    lines = ["# 재학습 3시드 집계", "",
             "- 보고값: 각 실행의 마지막 epoch(80) 값. 최선 epoch 선택은 하지 않았다.",
             "- 시드 1~3: Fold1은 `experiments/01_fold1_seeds/`, Fold2~5는 `experiments/02_fold_seeds/`. Fold1은 시드 1~5도 따로 표시한다.",
             "- SD는 표본 표준편차(n-1)이며 n=3이라 참고 수준이다.",
             "- 공개 체크포인트 값: `experiments/00_reproduction/regression_eval/pretrained_5fold_results.csv`. "
             "논문 값: arXiv 1904.07312v1 Table 2·3(p.8).", "",
             "## 1. 시드별 5-fold 평균", ""]
    lines += seed_table_lines(per_seed, stats, public_mean)
    lines += ["", "## 2. fold별 재학습 평균과 공개 체크포인트 차이", ""]
    lines += md_table(["구분"] + metric_labels(), fold_rows(fold_means, pretrained))
    lines += ["", "## 3. 공개 체크포인트 평가 대 논문 Table 2·3 (소수 둘째 자리 반올림)", ""]
    lines += paper_table_lines(comparison)
    lines += ["", "- 비교 가능한 %d칸 중 %d칸 일치, %d칸 불일치." % (len(comparison), matched_count, len(unmatched)),
              "- 불일치 칸: %s." % ", ".join(unmatched)]
    return lines


def csv_summary_rows(per_seed, stats, public_mean, pretrained):
    """CSV의 앞부분(시드별 평균, 시드 간 평균·SD, 공개 평균, 공개 fold별 값) 행을 만든다.

    입력: 각 집계 결과
    반환: 행(list)의 list. 열은 CSV_HEADER 순서이다
    """
    rows = []
    for seed in MAIN_SEEDS:
        rows.append(["시드별_5fold평균", "시드%d" % seed] + row_cells(per_seed[seed]) + [""])
    mean_cells = []
    sd_cells = []
    for label in metric_labels():
        mean_cells.append(fmt(stats[label][0]))
        sd_cells.append(fmt(stats[label][1]))
    rows.append(["시드간_평균", "평균"] + mean_cells + ["시드 1~3"])
    rows.append(["시드간_평균", "표본SD(n-1)"] + sd_cells + ["시드 1~3"])
    rows.append(["공개", "5-fold 평균"] + row_cells(public_mean) + [""])
    for fold in FOLDS:
        rows.append(["fold별_공개", "Fold%d" % fold] + row_cells(pretrained[fold]) + [""])
    return rows


def csv_fold_and_paper_rows(fold_means, pretrained, comparison):
    """CSV의 뒷부분(fold별 재학습, 차이, 논문 대조) 행을 만든다.

    입력: fold_means, pretrained, comparison - 각 집계 결과
    반환: 행(list)의 list
    """
    rows = []
    for name, fold in FOLD_ROW_NAMES:
        rows.append(["fold별_재학습", name] + row_cells(fold_means[name]) + [""])
        rows.append(["fold별_차이", name] + row_cells(difference(fold_means[name], pretrained[fold]), True) + ["재학습 - 공개"])
    for item in comparison:
        result = "일치" if item["matched"] else "불일치"
        note = "논문 %.3f / 공개 %.3f / 반올림 %.2f / %s" % (item["paper"], item["measured"], item["measured"], result)
        rows.append(["논문대조", item["name"], "", "", "", "", "", note])
    return rows


def write_outputs(markdown_lines, rows):
    """집계 파일(md, csv)을 새로 쓴다.

    입력: markdown_lines - 마크다운 줄 목록, rows - CSV 행 목록
    반환: 없음
    """
    with open(OUTPUT_MD, "w", encoding="utf-8") as file:
        file.write("\n".join(markdown_lines) + "\n")
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(CSV_HEADER)
        for row in rows:
            writer.writerow(row)


def main():
    """모든 입력을 읽어 집계하고 화면에 출력하며 md, csv 파일로 저장한다.

    입력: 없음
    반환: 없음
    """
    runs = load_all_runs()
    pretrained = read_pretrained()
    public_mean = average_runs(pretrained, FOLDS)
    per_seed = seed_fold_means(runs)
    stats = mean_and_sd(per_seed)
    fold_means = fold_retrained_means(runs)
    comparison = paper_comparison(pretrained, public_mean)
    markdown_lines = build_markdown(per_seed, stats, public_mean, fold_means, pretrained, comparison)
    print("\n".join(markdown_lines))
    rows = csv_summary_rows(per_seed, stats, public_mean, pretrained)
    rows += csv_fold_and_paper_rows(fold_means, pretrained, comparison)
    write_outputs(markdown_lines, rows)
    print("\n저장: %s, %s" % (OUTPUT_MD, OUTPUT_CSV))


if __name__ == "__main__":
    main()
