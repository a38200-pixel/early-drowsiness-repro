"""compare_rows 시험: 합성 행만 쓴다(파일 읽기, 쓰기 없음)."""
import unittest

import numpy as np

from improve.hmlstm import compare_rows as CR

B1_TEXT = ("\n1.0000 0.1057 0.0822 20.0000 0.0056 \n"
           "2.0000 0.1957 0.0792 12.0000 0.0123 \n"
           "3.0000 0.1640 0.0362 8.0000 0.0078 ")


def rows(count=10):
    return [np.array([i + 1, 0.1 + i / 100.0, 0.05, 10 + i, 0.005]) for i in range(count)]


class ParseTest(unittest.TestCase):
    def test_leading_blank_line_and_trailing_spaces_are_ignored(self):
        parsed = CR.parse_blink_text(B1_TEXT)
        self.assertEqual(len(parsed), 3)
        np.testing.assert_allclose(parsed[2], [3.0, 0.1640, 0.0362, 8.0, 0.0078])
        self.assertEqual(CR.parse_blink_text("\n\n"), [])


class CompareTest(unittest.TestCase):
    def test_identical_rows_match_and_pass(self):
        report = CR.compare_rows(rows(), rows())
        self.assertEqual(report["verdict"], "일치")
        self.assertTrue(report["passed"])
        self.assertEqual(report["max_abs_diff"], [0.0] * 5)
        self.assertEqual(report["rounded_mismatch_rows"], 0)
        self.assertIsNone(report["first_mismatch"])

    def test_value_mismatch_is_reported_with_the_first_position(self):
        hook = rows()
        hook[4][2] += 0.0001                                                        # 5e-5 보다 큰 차이
        report = CR.compare_rows(hook, rows())
        self.assertEqual(report["verdict"], "불일치")
        self.assertFalse(report["passed"])
        self.assertEqual(report["first_mismatch"][:2], (4, 2))
        self.assertAlmostEqual(report["max_abs_diff"][2], 0.0001, places=9)
        self.assertEqual(report["rounded_mismatch_rows"], 1)

    def test_tolerance_boundary(self):
        hook = rows()
        hook[1][3] += 4e-5
        self.assertTrue(CR.compare_rows(hook, rows())["passed"])                    # 5e-5 이내는 합격
        hook[1][3] += 4e-5                                                          # 합계 8e-5
        self.assertFalse(CR.compare_rows(hook, rows())["passed"])

    def test_row_count_mismatch_is_distinguished_from_value_mismatch(self):
        report = CR.compare_rows(rows(9), rows(10))
        self.assertEqual(report["verdict"], "행 수 불일치")
        self.assertFalse(report["row_count_equal"])
        self.assertFalse(report["passed"])
        self.assertEqual(report["max_abs_diff"], [0.0] * 5)                         # 겹치는 행은 같다

    def test_column_count_mismatch(self):
        hook = [row[:4] for row in rows()]
        report = CR.compare_rows(hook, rows())
        self.assertEqual(report["verdict"], "열 수 불일치")
        self.assertFalse(report["passed"])
        self.assertIsNone(report["max_abs_diff"])

    def test_prefix_mode_compares_only_the_first_rows_and_never_passes(self):
        report = CR.compare_rows(rows(4), rows(10), allow_prefix=True)
        self.assertEqual(report["verdict"], "일치")
        self.assertTrue(report["prefix_only"])
        self.assertFalse(report["passed"])
        self.assertEqual(report["compared_rows"], 4)
        hook = rows(4)
        hook[2][0] = 99.0
        self.assertEqual(CR.compare_rows(hook, rows(10), allow_prefix=True)["verdict"], "불일치")

    def test_both_empty_is_a_match(self):
        report = CR.compare_rows([], [])
        self.assertEqual(report["verdict"], "일치")
        self.assertTrue(report["passed"])


class ReportTest(unittest.TestCase):
    def test_first_mismatch_is_printed_with_three_rows_before_and_after(self):
        hook = rows()
        hook[5][1] += 0.01
        report = CR.compare_rows(hook, rows())
        text = "\n".join(CR.format_report(report, hook, rows(), context=3))
        self.assertIn("first mismatch: row=5 column=1", text)
        for index in range(2, 9):
            self.assertIn("hook[%d]" % index, text)
            self.assertIn("reference[%d]" % index, text)
        self.assertNotIn("hook[1]", text)
        self.assertNotIn("hook[9]", text)
        self.assertIn("hook[5] <-", text)

    def test_row_count_mismatch_report_names_the_first_one_sided_row(self):
        report = CR.compare_rows(rows(7), rows(10))
        text = "\n".join(CR.format_report(report, rows(7), rows(10)))
        self.assertIn("verdict: 행 수 불일치", text)
        self.assertIn("row=7", text)


if __name__ == "__main__":
    unittest.main()
