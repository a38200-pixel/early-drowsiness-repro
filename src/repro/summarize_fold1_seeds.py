# summarize_fold1_seeds.py
# 역할: Fold1 다중 시드 재학습 결과(시드별 마지막 epoch)를 표로 모으고 공개 체크포인트 값과 나란히 비교한다.
# 사용법: python src\repro\summarize_fold1_seeds.py
#         입력: experiments\01_fold1_seeds\seed<숫자>\metrics.csv (smoke_* 폴더는 읽지 않음)
#         출력: 화면 + experiments\01_fold1_seeds\summary.md (매번 새로 만든다)
"""Fold1 다중 시드 결과 요약 모듈.

각 시드의 metrics.csv 마지막 행(마지막 epoch)만 사용한다. 최선 epoch 선택은 하지 않는다.
마지막 epoch가 80이 아니면 '미완료'로 표시하고 통계에서 뺀다.
공개 체크포인트 Fold1 값은 regression_eval/pretrained_5fold_results.csv에서 읽는다(코드에 숫자를 적지 않는다).
csv 모듈과 numpy만 사용한다.
"""
import csv
import os

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SEEDS_DIR = os.path.join(REPO_ROOT, "experiments", "01_fold1_seeds")
SUMMARY_PATH = os.path.join(SEEDS_DIR, "summary.md")
PRETRAINED_CSV = os.path.join(REPO_ROOT, "experiments", "00_reproduction", "regression_eval",
                              "pretrained_5fold_results.csv")
EXPECTED_EPOCHS = 80
# (표에 쓸 이름, metrics.csv 열 이름, pretrained_5fold_results.csv 열 이름) - 표 열 순서 그대로
METRICS = [
    ("BSA loss", "BSA_loss", "BSA_loss"),
    ("BSA acc", "BSA_acc", "BSA_accuracy"),
    ("VA", "VA", "VA"),
    ("BSRE", "BSRE", "BSRE"),
    ("VRE", "VRE", "VRE"),
]
STATISTIC_NAMES = ["평균", "표준편차(ddof=1)", "최소", "최대"]  # 표에 나오는 통계 행 순서
COMPARED_METRICS = ["VA", "BSA acc"]


def find_seed_dirs():
    """seed<숫자> 이름의 폴더를 찾는다(smoke_* 등 다른 이름은 제외).

    입력: 없음
    반환: (시드 번호, 폴더 이름) 튜플의 list, 시드 번호 순으로 정렬
    """
    found = []
    if not os.path.isdir(SEEDS_DIR):
        return found
    for name in os.listdir(SEEDS_DIR):
        number_part = name[len("seed"):]
        if name.startswith("seed") and number_part.isdigit() and os.path.isdir(os.path.join(SEEDS_DIR, name)):
            found.append((int(number_part), name))
    found.sort()
    return found


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


def read_pretrained_fold1():
    """공개 체크포인트 5-fold 평가 결과 CSV에서 Fold1 행을 읽는다.

    입력: 없음
    반환: {표 이름: 값(float)} dict. 파일이나 Fold1 행이 없으면 RuntimeError
    """
    with open(PRETRAINED_CSV, "r", newline="") as file:
        for row in csv.DictReader(file):
            if row["Fold"] == "1":
                values = {}
                for label, _metrics_column, pretrained_column in METRICS:
                    values[label] = float(row[pretrained_column])
                return values
    raise RuntimeError("Fold1 행을 찾지 못했습니다: %s" % PRETRAINED_CSV)


def collect_seed_results():
    """시드 폴더마다 마지막 행을 읽어 완료 여부와 지표 값을 정리한다.

    입력: 없음
    반환: dict의 list. 각 dict는 name, last_epoch(int 또는 None), completed(bool), values({표 이름: float})
    """
    results = []
    for _seed_number, name in find_seed_dirs():
        last_row = read_last_row(os.path.join(SEEDS_DIR, name, "metrics.csv"))
        result = {"name": name, "last_epoch": None, "completed": False, "values": {}}
        if last_row is not None:
            result["last_epoch"] = int(last_row["epoch"])
            result["completed"] = result["last_epoch"] == EXPECTED_EPOCHS
            for label, metrics_column, _pretrained_column in METRICS:
                result["values"][label] = float(last_row[metrics_column])
        results.append(result)
    return results


def compute_statistics(completed_results):
    """완료된 시드들의 지표별 평균, 표준편차(ddof=1), 최소, 최대를 계산한다.

    입력: completed_results - completed가 True인 결과 dict의 list
    반환: {통계 이름: {표 이름: float}} dict. 시드가 0개면 빈 dict, 1개면 표준편차는 nan
    """
    if not completed_results:
        return {}
    statistics = {}
    for statistic_name in STATISTIC_NAMES:
        statistics[statistic_name] = {}
    for label, _metrics_column, _pretrained_column in METRICS:
        values = np.array([result["values"][label] for result in completed_results])
        statistics["평균"][label] = float(values.mean())
        statistics["표준편차(ddof=1)"][label] = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
        statistics["최소"][label] = float(values.min())
        statistics["최대"][label] = float(values.max())
    return statistics


def format_values(values):
    """지표 값 dict를 표 열 순서대로 문자열 칸 list로 바꾼다.

    입력: values - {표 이름: float} dict (빠진 지표는 '-'로 표시)
    반환: 문자열 list
    """
    cells = []
    for label, _metrics_column, _pretrained_column in METRICS:
        if label in values:
            cells.append("%.4f" % values[label])
        else:
            cells.append("-")
    return cells


def seed_status_text(result):
    """시드 결과의 상태를 표에 쓸 글자로 바꾼다.

    입력: result - collect_seed_results()의 원소 dict
    반환: '완료', '미완료(epoch N)', 'metrics.csv 없음' 중 하나
    """
    if result["last_epoch"] is None:
        return "metrics.csv 없음"
    if result["completed"]:
        return "완료"
    return "미완료(epoch %d)" % result["last_epoch"]


def build_table(seed_results, statistics, pretrained_values):
    """시드별 행, 통계 행, 공개 체크포인트 행을 담은 마크다운 표를 만든다.

    입력: seed_results - collect_seed_results() 결과, statistics - compute_statistics() 결과,
          pretrained_values - read_pretrained_fold1() 결과
    반환: 마크다운 표 줄의 list
    """
    header = ["구분", "상태"]
    for label, _metrics_column, _pretrained_column in METRICS:
        header.append(label)
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for result in seed_results:
        cells = [result["name"], seed_status_text(result)] + format_values(result["values"])
        lines.append("| " + " | ".join(cells) + " |")
    for statistic_name in STATISTIC_NAMES:
        if statistic_name in statistics:
            cells = [statistic_name, "완료 시드"] + format_values(statistics[statistic_name])
            lines.append("| " + " | ".join(cells) + " |")
    cells = ["공개 체크포인트 Fold1", "참고"] + format_values(pretrained_values)
    lines.append("| " + " | ".join(cells) + " |")
    return lines


def compare_with_pretrained(label, pretrained_value, completed_results):
    """공개 체크포인트 값이 시드 몇 개보다 높은지, 시드 범위 안인지 한 줄로 적는다.

    입력: label - 지표 이름(VA 또는 BSA acc), pretrained_value - 공개 체크포인트 값, completed_results - 완료 시드 결과
    반환: 비교 결과 문자열 한 줄
    """
    # 결론 문구('설명 가능/불가')를 쓰지 않는 이유: 같은 분포에서 뽑은 독립 표본 하나가 시드 n개의
    # [최소, 최대] 밖에 놓일 확률은 2/(n+1)이다. n=5면 약 1/3이라, 범위 밖이라는 사실만으로는
    # 공개 체크포인트가 다른 분포에서 왔다고 판정할 수 없다. 그래서 개수와 범위 안/밖만 보고한다.
    seed_values = [result["values"][label] for result in completed_results]
    if not seed_values:
        return "- %s: 완료된 시드가 없어 비교하지 않음" % label
    higher_count = 0
    for value in seed_values:
        if pretrained_value > value:
            higher_count += 1
    lowest, highest = min(seed_values), max(seed_values)
    inside = lowest <= pretrained_value <= highest
    range_text = "안" if inside else "밖 - 결론 보류, 추가 시드 필요"
    return "- %s: 공개 체크포인트 값 %.4f은 시드 %d개 중 %d개보다 높다. 시드 범위 [%.4f, %.4f] %s" % (
        label, pretrained_value, len(seed_values), higher_count, lowest, highest, range_text)


def build_report():
    """요약 보고서 전체를 마크다운 줄 목록으로 만든다.

    입력: 없음
    반환: 문자열 list
    """
    seed_results = collect_seed_results()
    completed_results = [result for result in seed_results if result["completed"]]
    pretrained_values = read_pretrained_fold1()
    statistics = compute_statistics(completed_results)
    lines = ["# Fold1 다중 시드 재학습 요약", "",
             "- 보고값: 각 시드의 마지막 epoch 값(최선 epoch 선택 없음). 기대 epoch 수: %d" % EXPECTED_EPOCHS,
             "- 시드 폴더 %d개, 완료 %d개(미완료는 통계에서 제외)" % (len(seed_results), len(completed_results)),
             "- 공개 체크포인트 값 출처: experiments/00_reproduction/regression_eval/pretrained_5fold_results.csv", ""]
    lines.extend(build_table(seed_results, statistics, pretrained_values))
    lines.extend(["", "## 공개 체크포인트 값과의 비교", ""])
    for label in COMPARED_METRICS:
        lines.append(compare_with_pretrained(label, pretrained_values[label], completed_results))
    return lines


def main():
    """보고서를 만들어 화면에 출력하고 summary.md에 새로 쓴다.

    입력: 없음
    반환: 없음
    """
    report_text = "\n".join(build_report()) + "\n"
    print(report_text)
    os.makedirs(SEEDS_DIR, exist_ok=True)
    with open(SUMMARY_PATH, "w", encoding="utf-8") as file:
        file.write(report_text)
    print("summary.md를 새로 썼습니다: %s" % SUMMARY_PATH)


if __name__ == "__main__":
    main()
