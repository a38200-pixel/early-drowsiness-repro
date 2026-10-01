# summarize_folds.py
# 역할: 시드 하나로 재학습한 5개 fold의 마지막 epoch 지표를 표로 모으고, 공개 체크포인트 값과의 차이(재학습 - 공개)를 함께 낸다.
# 사용법: python src\repro\summarize_folds.py <시드>
#         입력: experiments\01_fold1_seeds\seed<시드>\metrics.csv (Fold1),
#               experiments\02_fold_seeds\fold<F>_seed<시드>\metrics.csv (Fold2~5)
#         출력: 화면 + experiments\02_fold_seeds\summary_seed<시드>.md (매번 새로 만든다)
"""fold별 재학습 결과 요약 모듈.

각 fold의 metrics.csv 마지막 행(마지막 epoch)만 사용한다. 마지막 epoch가 80이 아니거나 파일이 없으면
'미완료'로 표시하고 평균에서 제외한다. 공개 체크포인트 값은 regression_eval/pretrained_5fold_results.csv의
같은 fold 행에서 읽는다(코드에 숫자를 적지 않는다). 결론 문구 없이 숫자만 낸다. csv 모듈만 사용한다.
평균은 완료된 fold만으로 계산하고, 공개 체크포인트 평균과 차이 평균도 같은 fold 집합으로 계산해 맞춘다.
"""
import csv
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FOLD1_SEEDS_DIR = os.path.join(REPO_ROOT, "experiments", "01_fold1_seeds")
FOLD_SEEDS_DIR = os.path.join(REPO_ROOT, "experiments", "02_fold_seeds")
PRETRAINED_CSV = os.path.join(REPO_ROOT, "experiments", "00_reproduction", "regression_eval",
                              "pretrained_5fold_results.csv")
EXPECTED_EPOCHS = 80
FOLDS = [1, 2, 3, 4, 5]
# (표에 쓸 이름, metrics.csv 열 이름, pretrained_5fold_results.csv 열 이름) - 표 열 순서 그대로
METRICS = [
    ("BSA loss", "BSA_loss", "BSA_loss"),
    ("BSA acc", "BSA_acc", "BSA_accuracy"),
    ("VA", "VA", "VA"),
    ("BSRE", "BSRE", "BSRE"),
    ("VRE", "VRE", "VRE"),
]


def metrics_path_of(fold, seed):
    """fold와 시드에 해당하는 metrics.csv 경로를 돌려준다. Fold1은 01_fold1_seeds, 나머지는 02_fold_seeds에 있다.

    입력: fold - 1~5, seed - 정수 시드
    반환: metrics.csv 경로 문자열
    """
    if fold == 1:
        return os.path.join(FOLD1_SEEDS_DIR, "seed%d" % seed, "metrics.csv")
    return os.path.join(FOLD_SEEDS_DIR, "fold%d_seed%d" % (fold, seed), "metrics.csv")


def read_last_row(metrics_path):
    """metrics.csv의 마지막 데이터 행을 읽는다.

    입력: metrics_path - metrics.csv 경로
    반환: 열 이름을 키로 하는 dict. 파일이 없거나 데이터 행이 없으면 None
    """
    if not os.path.isfile(metrics_path):
        return None
    last_row = None
    with open(metrics_path, "r", newline="") as file:
        for row in csv.DictReader(file):
            last_row = row
    return last_row


def read_pretrained_values():
    """공개 체크포인트 5-fold 평가 결과 CSV에서 fold별 지표를 읽는다.

    입력: 없음
    반환: {fold(int): {표 이름: float}} dict. fold 1~5 행 중 하나라도 없으면 RuntimeError
    """
    values_by_fold = {}
    with open(PRETRAINED_CSV, "r", newline="") as file:
        for row in csv.DictReader(file):
            if row["Fold"].isdigit():
                values = {}
                for label, _metrics_column, pretrained_column in METRICS:
                    values[label] = float(row[pretrained_column])
                values_by_fold[int(row["Fold"])] = values
    for fold in FOLDS:
        if fold not in values_by_fold:
            raise RuntimeError("공개 체크포인트 CSV에 Fold%d 행이 없습니다: %s" % (fold, PRETRAINED_CSV))
    return values_by_fold


def collect_fold_results(seed):
    """fold 1~5마다 마지막 행을 읽어 완료 여부와 지표 값을 정리한다.

    입력: seed - 정수 시드
    반환: {fold: {"last_epoch": int 또는 None, "completed": bool, "values": {표 이름: float}}} dict
    """
    results = {}
    for fold in FOLDS:
        last_row = read_last_row(metrics_path_of(fold, seed))
        result = {"last_epoch": None, "completed": False, "values": {}}
        if last_row is not None:
            result["last_epoch"] = int(last_row["epoch"])
            result["completed"] = result["last_epoch"] == EXPECTED_EPOCHS
            for label, metrics_column, _pretrained_column in METRICS:
                result["values"][label] = float(last_row[metrics_column])
        results[fold] = result
    return results


def status_text(result):
    """fold 결과의 상태를 표에 쓸 글자로 바꾼다.

    입력: result - collect_fold_results()의 값 dict
    반환: '완료', '미완료(epoch N)', 'metrics.csv 없음' 중 하나
    """
    if result["last_epoch"] is None:
        return "metrics.csv 없음"
    if result["completed"]:
        return "완료"
    return "미완료(epoch %d)" % result["last_epoch"]


def format_values(values, with_sign):
    """지표 값 dict를 표 열 순서대로 문자열 칸 list로 바꾼다.

    입력: values - {표 이름: float} dict (빠진 지표는 '-'), with_sign - True면 +/- 부호를 붙인다(차이 표용)
    반환: 문자열 list
    """
    pattern = "%+.4f" if with_sign else "%.4f"
    cells = []
    for label, _metrics_column, _pretrained_column in METRICS:
        if label in values:
            cells.append(pattern % values[label])
        else:
            cells.append("-")
    return cells


def average_over_folds(values_by_fold, folds):
    """지정한 fold들의 지표 평균을 계산한다.

    입력: values_by_fold - {fold: {표 이름: float}}, folds - 평균에 넣을 fold 번호 list
    반환: {표 이름: 평균}. folds가 비어 있으면 빈 dict
    """
    averages = {}
    if not folds:
        return averages
    for label, _metrics_column, _pretrained_column in METRICS:
        total = 0.0
        for fold in folds:
            total += values_by_fold[fold][label]
        averages[label] = total / len(folds)
    return averages


def compute_differences(retrained, pretrained, folds):
    """완료된 fold마다 (재학습 - 공개 체크포인트) 차이를 계산한다.

    입력: retrained - {fold: 지표 dict}, pretrained - {fold: 지표 dict}, folds - 차이를 구할 fold 번호 list
    반환: {fold: {표 이름: 차이}}
    """
    differences = {}
    for fold in folds:
        differences[fold] = {}
        for label, _metrics_column, _pretrained_column in METRICS:
            differences[fold][label] = retrained[fold][label] - pretrained[fold][label]
    return differences


def build_table(title, rows, average_row, average_name, with_sign):
    """제목, 표 머리, fold별 행, 평균 행으로 이뤄진 마크다운 표 줄 목록을 만든다.

    입력: title - 표 제목, rows - [(구분 글자, 상태 글자, {표 이름: float})] list,
          average_row - 평균 dict, average_name - 평균 행 이름, with_sign - 부호 표시 여부
    반환: 문자열 list
    """
    header = ["구분", "상태"]
    for label, _metrics_column, _pretrained_column in METRICS:
        header.append(label)
    lines = ["## " + title, "", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for name, status, values in rows:
        lines.append("| " + " | ".join([name, status] + format_values(values, with_sign)) + " |")
    lines.append("| " + " | ".join([average_name, "완료 fold"] + format_values(average_row, with_sign)) + " |")
    lines.append("")
    return lines


def build_fold_rows(results, values_by_fold, status_by_fold):
    """표 한 개에 들어갈 fold별 행 목록을 만든다.

    입력: results - collect_fold_results() 결과(상태 글자에 사용), values_by_fold - {fold: 지표 dict}(없는 fold는 빈 칸),
          status_by_fold - False면 results의 상태, True면 '참고'를 상태로 쓴다
    반환: [(구분 글자, 상태 글자, 지표 dict)] list
    """
    rows = []
    for fold in FOLDS:
        status = "참고" if status_by_fold else status_text(results[fold])
        rows.append(("Fold%d" % fold, status, values_by_fold.get(fold, {})))
    return rows


def build_report(seed):
    """시드 하나에 대한 요약 보고서 전체를 마크다운 줄 목록으로 만든다.

    입력: seed - 정수 시드
    반환: 문자열 list (재학습 표, 공개 체크포인트 표, 차이(재학습 - 공개) 표)
    """
    results = collect_fold_results(seed)
    pretrained = read_pretrained_values()
    done = []
    retrained = {}
    for fold in FOLDS:
        retrained[fold] = results[fold]["values"]
        if results[fold]["completed"]:
            done.append(fold)
    differences = compute_differences(retrained, pretrained, done)
    average_name = "평균(완료 %d개/5개)" % len(done)
    lines = ["# fold별 재학습 요약 (시드 %d)" % seed, "",
             "- 보고값: 각 fold의 마지막 epoch 값. 기대 epoch 수: %d. 평균은 완료된 fold만 사용" % EXPECTED_EPOCHS,
             "- 공개 체크포인트 값 출처: experiments/00_reproduction/regression_eval/pretrained_5fold_results.csv", ""]
    lines += build_table("재학습", build_fold_rows(results, retrained, False),
                         average_over_folds(retrained, done), average_name, False)
    lines += build_table("공개 체크포인트", build_fold_rows(results, pretrained, True),
                         average_over_folds(pretrained, done), average_name, False)
    lines += build_table("차이 (재학습 - 공개)", build_fold_rows(results, differences, False),
                         average_over_folds(differences, done), average_name, True)
    return lines


def main():
    """명령행에서 시드를 받아 보고서를 화면에 출력하고 summary_seed<시드>.md에 새로 쓴다.

    입력: 없음(명령행 인자 사용)
    반환: 없음
    """
    if len(sys.argv) != 2:
        sys.exit("사용법: python src\\repro\\summarize_folds.py <시드>")
    try:
        seed = int(sys.argv[1])
    except ValueError:
        sys.exit("시드는 정수여야 합니다: %s" % sys.argv[1])
    report_text = "\n".join(build_report(seed)) + "\n"
    print(report_text)
    os.makedirs(FOLD_SEEDS_DIR, exist_ok=True)
    summary_path = os.path.join(FOLD_SEEDS_DIR, "summary_seed%d.md" % seed)
    with open(summary_path, "w", encoding="utf-8") as file:
        file.write(report_text)
    print("요약을 새로 썼습니다: %s" % summary_path)


if __name__ == "__main__":
    main()
