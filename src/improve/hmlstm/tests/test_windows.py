"""windows 시험: 합성 행과 손으로 계산한 값만 쓴다(외부 데이터, 파일 없음)."""
import unittest

import numpy as np

from improve.hmlstm import windows as W


def ramp(count):
    """행 번호가 값에 드러나는 정규화된 blink: 행 i, 특징 j 의 값이 4*i + j."""
    return np.arange(count * 4, dtype=np.float64).reshape(count, 4)


def table(values):
    """열 1~4 에 values (n, 4) 를 넣은 (n, 5) 행. 0열은 누적 blink 수."""
    values = np.asarray(values, dtype=np.float64)
    rows = np.zeros((len(values), 5))
    rows[:, 0] = np.arange(1, len(values) + 1)
    rows[:, 1:] = values
    return rows


class UnrollTest(unittest.TestCase):
    def test_exactly_30_rows_make_one_window(self):
        out = W.unroll_in_time(ramp(30))
        self.assertEqual(out.shape, (1, 30, 4))
        np.testing.assert_array_equal(out[0], ramp(30))

    def test_31_rows_make_one_window_and_drop_the_last_row(self):
        out = W.unroll_in_time(ramp(31))
        self.assertEqual(out.shape, (1, 30, 4))
        np.testing.assert_array_equal(out[0], ramp(31)[:30])                      # 마지막 불완전 창(행 30 까지)은 버린다

    def test_32_rows_make_two_windows_with_stride_two(self):
        out = W.unroll_in_time(ramp(32))
        self.assertEqual(out.shape, (2, 30, 4))
        np.testing.assert_array_equal(out[1], ramp(32)[2:32])
        self.assertEqual(W.unroll_in_time(ramp(34)).shape[0], 3)                  # (34-30)//2 + 1
        self.assertEqual(W.unroll_in_time(ramp(33)).shape[0], 2)

    def test_29_rows_wait_by_default_and_pad_in_front_when_asked(self):
        waiting = W.unroll_in_time(ramp(29))
        self.assertEqual(waiting.shape, (0, 30, 4))
        padded = W.unroll_in_time(ramp(29), pad_short=True)
        self.assertEqual(padded.shape, (1, 30, 4))
        np.testing.assert_array_equal(padded[0, :1], np.zeros((1, 4)))            # 앞쪽 1행이 0 패딩
        np.testing.assert_array_equal(padded[0, 1:], ramp(29))

    def test_empty_input_never_makes_a_window(self):
        self.assertEqual(W.unroll_in_time(np.zeros((0, 4)), pad_short=True).shape, (0, 30, 4))

    def test_window_ranges_match_window_count(self):
        for count in (0, 1, 29, 30, 31, 32, 45, 100):
            for pad in (False, True):
                self.assertEqual(len(W.window_ranges(count, pad)), W.unroll_in_time(ramp(count), pad).shape[0], (count, pad))
        self.assertEqual(W.window_ranges(32), [(0, 29), (2, 31)])
        self.assertEqual(W.window_ranges(5, pad_short=True), [(0, 4)])
        self.assertEqual(W.window_ranges(5), [])


class ReferenceTest(unittest.TestCase):
    def test_last_third_of_the_rows_is_the_reference(self):
        values = [[float(i), float(2 * i), float(i * i), float(10 - i)] for i in range(9)]
        details = W.reference_details(table(values))
        self.assertEqual((details.ref_n, details.first), (3, 6))
        last = np.array(values[-3:])
        np.testing.assert_allclose(details.means, last.mean(axis=0), rtol=0, atol=1e-12)
        np.testing.assert_allclose(details.stds, last.std(axis=0), rtol=0, atol=1e-12)      # ddof=0
        self.assertEqual((details.fallback, details.fallback_kind, details.reason), (0, "none", None))
        means, stds = W.compute_reference(table(values))
        np.testing.assert_array_equal(means, details.means)

    def test_zero_std_falls_back_to_the_whole_column_std(self):
        values = [[float(i), 5.0 if i >= 6 else float(i), float(i * i), float(10 - i)] for i in range(9)]    # 열 1 은 마지막 3개가 모두 같은 값
        details = W.reference_details(table(values))
        self.assertEqual((details.fallback, details.fallback_kind), (1, "whole_std"))
        self.assertAlmostEqual(details.stds[1], float(np.std([v[1] for v in values])), places=12)      # 특징 1(Amp)이 대체됨
        self.assertIsNone(details.reason)

    def test_fewer_than_two_rows_have_no_reference(self):
        self.assertIsNone(W.compute_reference(table([[1.0, 2.0, 3.0, 4.0]])))
        self.assertIsNotNone(W.reference_details(table([[1.0, 2.0, 3.0, 4.0]])).reason)
        self.assertIsNone(W.reference_details(np.zeros((0, 5))))
        self.assertIsNone(W.compute_reference(np.zeros((0, 5))))

    def test_two_to_five_rows_follow_the_b1_quirks(self):
        two = W.reference_details(table([[1.0, 2.0, 3.0, 4.0], [3.0, 6.0, 9.0, 8.0]]))
        self.assertEqual((two.ref_n, two.fallback_kind, two.reason), (2, "whole_array", None))       # n//3 == 0 이면 전체 배열 통계
        np.testing.assert_allclose(two.means, [2.0, 4.0, 6.0, 6.0])
        np.testing.assert_allclose(two.stds, [1.0, 2.0, 3.0, 2.0])
        same = W.reference_details(table([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]]))
        self.assertIsNotNone(same.reason)                                                           # 대체 후에도 표준편차 0
        self.assertIsNone(W.compute_reference(table([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]])))
        values = [[float(i), float(i * 2), float(i * 3), float(i * 4)] for i in range(5)]
        five = W.reference_details(table(values))
        self.assertEqual((five.ref_n, five.fallback, five.fallback_kind), (1, 4, "whole_std"))     # 마지막 1개의 표준편차 0 -> 전체 표준편차로 대체
        np.testing.assert_allclose(five.means, [4.0, 8.0, 12.0, 16.0])
        np.testing.assert_allclose(five.stds, np.array(values).std(axis=0))

    def test_rows_need_five_columns(self):
        with self.assertRaises(ValueError):
            W.as_table(np.zeros((3, 4)))


class VideoWindowsTest(unittest.TestCase):
    def test_normalization_and_reference_video_split(self):
        base = np.arange(1, 181, dtype=np.float64).reshape(45, 4)
        rows = table(base * np.array([1.0, 2.0, 3.0, 4.0]))
        means, stds = W.compute_reference(rows)
        windows, ranges = W.video_windows(rows, means, stds, is_reference=True)             # 45 행 중 앞 45 - 15 = 30 행만 사용
        self.assertEqual((windows.shape, ranges), ((1, 30, 4), [(0, 29)]))
        expected = (rows[:30, 1:] - means) / stds
        np.testing.assert_allclose(windows[0], expected, rtol=0, atol=1e-12)
        windows_all, ranges_all = W.video_windows(rows, means, stds, is_reference=False)    # 다른 영상은 45 행 전부 -> (45-30)//2 + 1 = 8
        self.assertEqual((windows_all.shape[0], ranges_all[-1]), (8, (14, 43)))
        windows_full, _ = W.video_windows(rows, means, stds, is_reference=True, hold_out_reference_tail=False)
        self.assertEqual(windows_full.shape[0], 8)                                          # B1 의 앞 2/3 규칙을 끄면 전부 사용
        self.assertEqual(W.rows_used(45, True), 30)
        self.assertEqual(W.rows_used(45, False), 45)

    def test_short_video_waits_unless_padding_is_requested(self):
        rows = table(np.random.RandomState(0).randn(12, 4))
        means, stds = W.compute_reference(rows)
        self.assertEqual(W.video_windows(rows, means, stds)[0].shape[0], 0)
        padded, ranges = W.video_windows(rows, means, stds, pad_short=True)
        self.assertEqual((padded.shape, ranges), ((1, 30, 4), [(0, 11)]))
        np.testing.assert_array_equal(padded[0, :18], np.zeros((18, 4)))


if __name__ == "__main__":
    unittest.main()
