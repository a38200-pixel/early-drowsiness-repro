# compare_rows.py
# 역할: 원본을 메모리에서 실행해 받은 깜빡임 행(훅 행)과 B1 이 파일로 남긴 blinks 파일의 행을 대조한다(결정 I-28 합격 기준).
#       합격 = 행 수 일치 + 열 수 일치 + 모든 값의 차이가 5e-5 이내(소수 4자리 일치). 불일치는 숨기지 않고 첫 불일치 위치와 앞뒤 행을 보인다.
# 파일 형식: 첫 줄이 빈 줄이고, 행마다 [TOTAL_BLINKS, Freq*100, Amp, Dur, Vel] 5개 값이 공백으로 구분되며 소수 4자리, 끝에 공백이 있다.
# 사용: verify_blinks.py 와 시험이 가져다 쓴다. 파일은 읽기만 한다.
"""깜빡임 행 대조(Python 3.6 호환)."""
import numpy as np

TOLERANCE = 5e-5            # 소수 4자리 일치의 허용 차이
COLUMN_NAMES = ("TOTAL_BLINKS", "Freq*100", "Amp", "Dur", "Vel")
_SLACK = 1e-12              # 부동소수 비교 여유


def parse_blink_text(text):
    """blinks 텍스트(str)를 float 배열 목록으로 바꾼다. 빈 줄은 건너뛰고, 쉼표는 공백으로 본다."""
    rows = []
    for line in text.replace(",", " ").split("\n"):
        if line.strip() == "":
            continue
        rows.append(np.array([float(token) for token in line.split()], dtype=float))
    return rows


def read_blink_file(path):
    """blinks 파일을 읽어 행 목록을 돌려준다(읽기 전용, 줄바꿈 변환 없이 바이트로 읽는다)."""
    with open(path, "rb") as handle:
        return parse_blink_text(handle.read().decode("latin-1"))


def _columns(rows):
    return tuple(sorted(set(len(row) for row in rows)))


def compare_rows(hook_rows, reference_rows, tolerance=TOLERANCE, allow_prefix=False):
    """훅 행과 기준 행을 대조해 결과 dict 를 돌려준다.

    verdict: "일치", "불일치"(값이 다름), "행 수 불일치", "열 수 불일치". passed 는 행 수와 열 수가 같고 모든 차이가 tolerance 이내일 때만 True 이다.
    allow_prefix=True 이면 훅 행이 기준보다 적을 때 앞쪽 행만 비교한다(부분 실행 확인용, 이때도 passed 는 False).
    max_abs_diff: 열별 최대 절댓값 차이(비교한 행 기준), rounded_mismatch_rows: 4자리 반올림 값이 다른 행 수,
    first_mismatch: (행 번호(0부터), 열 번호, 훅 값, 기준 값) 또는 None
    """
    hook = [np.asarray(row, dtype=float) for row in hook_rows]
    reference = [np.asarray(row, dtype=float) for row in reference_rows]
    report = {"hook_rows": len(hook), "reference_rows": len(reference), "row_count_equal": len(hook) == len(reference),
              "hook_columns": _columns(hook), "reference_columns": _columns(reference), "compared_rows": 0,
              "max_abs_diff": None, "rounded_mismatch_rows": 0, "first_mismatch": None, "prefix_only": False,
              "verdict": "일치", "passed": False}
    columns_ok = report["hook_columns"] == report["reference_columns"] and len(report["hook_columns"]) <= 1
    if not columns_ok:
        report["verdict"] = "열 수 불일치"
        return report
    prefix_only = allow_prefix and len(hook) < len(reference)
    compared = len(hook) if prefix_only else min(len(hook), len(reference))
    report["compared_rows"] = compared
    report["prefix_only"] = prefix_only
    values_equal = True
    if compared > 0:
        hook_array = np.vstack(hook[:compared])
        reference_array = np.vstack(reference[:compared])
        diff = np.abs(hook_array - reference_array)
        report["max_abs_diff"] = [float(value) for value in diff.max(axis=0)]
        report["rounded_mismatch_rows"] = int(np.any(np.round(hook_array, 4) != np.round(reference_array, 4), axis=1).sum())
        bad = diff > tolerance + _SLACK
        if bad.any():
            values_equal = False
            row, column = [int(value) for value in np.argwhere(bad)[0]]
            report["first_mismatch"] = (row, column, float(hook_array[row, column]), float(reference_array[row, column]))
    if not report["row_count_equal"] and not prefix_only:
        report["verdict"] = "행 수 불일치"
    elif not values_equal:
        report["verdict"] = "불일치"
    report["passed"] = report["row_count_equal"] and values_equal
    return report


def _row_text(row):
    return " ".join("%.4f" % value for value in row)


def format_report(report, hook_rows, reference_rows, context=3):
    """대조 결과를 출력할 줄 목록으로 만든다. 첫 불일치가 있으면 그 행 앞뒤 context 행을 훅/기준 양쪽으로 그대로 보인다."""
    lines = ["verdict: %s (passed=%s)" % (report["verdict"], report["passed"]),
             "rows: hook=%d reference=%d row_count_equal=%s" % (report["hook_rows"], report["reference_rows"], report["row_count_equal"]),
             "columns: hook=%s reference=%s" % (list(report["hook_columns"]), list(report["reference_columns"])),
             "compared_rows: %d%s" % (report["compared_rows"], " (prefix only, not a pass/fail judgment)" if report["prefix_only"] else "")]
    if report["max_abs_diff"] is not None:
        names = COLUMN_NAMES if len(report["max_abs_diff"]) == len(COLUMN_NAMES) else tuple("col%d" % i for i in range(len(report["max_abs_diff"])))
        lines.append("max abs diff per column: " + ", ".join("%s=%.6f" % (name, value) for name, value in zip(names, report["max_abs_diff"])))
        lines.append("rows differing at 4-decimal rounding: %d" % report["rounded_mismatch_rows"])
    mismatch = report["first_mismatch"]
    if mismatch is not None:
        row, column, hook_value, reference_value = mismatch
        lines.append("first mismatch: row=%d column=%d hook=%.4f reference=%.4f diff=%.6f" % (row, column, hook_value, reference_value, abs(hook_value - reference_value)))
        low, high = max(0, row - context), row + context + 1
        for label, rows in (("hook", hook_rows), ("reference", reference_rows)):
            for index in range(low, min(high, len(rows))):
                lines.append("  %s[%d]%s %s" % (label, index, " <-" if index == row else "   ", _row_text(rows[index])))
    elif not report["row_count_equal"]:
        shorter = min(report["hook_rows"], report["reference_rows"])
        lines.append("rows differ in count; first row present on only one side: row=%d" % shorter)
    return lines
