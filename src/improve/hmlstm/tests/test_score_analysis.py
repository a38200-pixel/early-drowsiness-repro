"""score_analysis 시험: 합성 배열만 쓴다(TensorFlow 불필요). 체크포인트가 필요한 시험은 HMLSTM_CKPT_ROOT 가 없으면 건너뛴다."""
import os
import unittest

import numpy as np

from improve.hmlstm import score_analysis as SA


def record(label, scores, padded=None, person="p", video="v", fold=1):
    scores = np.asarray(scores, dtype=np.float64)
    padded = np.zeros(len(scores), dtype=bool) if padded is None else np.asarray(padded, dtype=bool)
    blinks = 30 + 2 * np.arange(len(scores))
    return SA.VideoRecord(video, person, fold, label, scores, padded, blinks)


class AlarmRuleTest(unittest.TestCase):
    def test_run_must_be_consecutive(self):
        self.assertIsNone(SA.run_alarm_window([7, 1, 7, 1, 7], 6.68, 2))

    def test_alarm_window_is_the_last_of_the_run(self):
        self.assertEqual(SA.run_alarm_window([1, 7, 7, 7, 1], 6.68, 3), 4)

    def test_boundary_value_counts(self):
        self.assertEqual(SA.run_alarm_window([6.68, 6.68], 6.68, 2), 2)
        self.assertIsNone(SA.run_alarm_window([6.679, 6.68], 6.68, 2))

    def test_run_broken_at_the_end(self):
        self.assertIsNone(SA.run_alarm_window([1, 7, 7, 1], 6.68, 3))
        self.assertIsNone(SA.run_alarm_window([1, 1, 7, 7], 6.68, 3))

    def test_single_window_and_empty(self):
        self.assertEqual(SA.run_alarm_window([9], 9.0, 1), 1)
        self.assertIsNone(SA.run_alarm_window([], 5.0, 1))

    def test_alarm_stats_counts_and_blink_conversion(self):
        records = [record(10, [1, 7, 7, 7]), record(10, [1, 1, 1, 1]), record(0, [7, 7, 7, 1])]
        stats = SA.alarm_stats(records, 10, 6.68, 3)
        self.assertEqual((stats["alarmed"], stats["videos"]), (1, 2))
        self.assertEqual(stats["median_window"], 4.0)
        self.assertEqual(stats["median_blinks"], 36.0)          # 30 + 2 * (4 - 1)
        self.assertEqual(SA.alarm_stats(records, 0, 6.68, 3)["alarmed"], 1)


class SummaryTest(unittest.TestCase):
    def test_video_max_mean_and_fraction(self):
        item = record(5, [2, 4, 6, 8])
        self.assertEqual(SA.video_value(item, "max"), 8.0)
        self.assertEqual(SA.video_value(item, "mean"), 5.0)
        self.assertEqual(SA.fraction_at_least(item.scores, 6.0), 0.5)
        self.assertIsNone(SA.fraction_at_least([], 6.0))

    def test_describe_percentiles(self):
        stat = SA.describe(np.arange(101, dtype=np.float64))
        self.assertEqual(stat["n"], 101)
        self.assertAlmostEqual(stat["mean"], 50.0)
        self.assertEqual([stat["pct"][p] for p in SA.PERCENTILES], [5.0, 25.0, 50.0, 75.0, 95.0])
        self.assertIsNone(SA.describe([])["mean"])

    def test_band_fractions(self):
        fractions = SA.band_fractions([0.0, 3.33, 3.34, 6.67, 6.68, 10.0])
        for value in fractions:
            self.assertAlmostEqual(value, 2 / 6.0)
        self.assertIsNone(SA.band_fractions([]))


class AucTest(unittest.TestCase):
    def test_perfect_separation(self):
        self.assertEqual(SA.auc([5, 6, 7], [1, 2, 3]), 1.0)
        self.assertEqual(SA.auc([1, 2, 3], [5, 6, 7]), 0.0)

    def test_near_identical_is_about_half(self):
        rng = np.random.RandomState(3)
        self.assertAlmostEqual(SA.auc(rng.randn(400), rng.randn(400)), 0.5, delta=0.06)

    def test_ties_count_half(self):
        self.assertEqual(SA.auc([1, 1], [1, 1]), 0.5)
        self.assertEqual(SA.auc([2, 1], [1]), 0.75)

    def test_empty_side(self):
        self.assertIsNone(SA.auc([], [1.0]))


class BootstrapTest(unittest.TestCase):
    def rows(self):
        rng = np.random.RandomState(5)
        rows = []
        for person in range(12):
            rows.append(("a%d" % person, 10, 5.0 + rng.randn()))
            rows.append(("a%d" % person, 0, 3.0 + rng.randn()))
        return rows

    def test_reproducible_with_fixed_seed(self):
        first = SA.person_bootstrap_auc(self.rows(), 10, 0, draws=200, seed=0)
        second = SA.person_bootstrap_auc(self.rows(), 10, 0, draws=200, seed=0)
        self.assertEqual(first, second)
        self.assertNotEqual(first[1:3], SA.person_bootstrap_auc(self.rows(), 10, 0, draws=200, seed=1)[1:3])

    def test_interval_contains_point_and_is_ordered(self):
        point, low, high, valid = SA.person_bootstrap_auc(self.rows(), 10, 0, draws=300, seed=0)
        self.assertLessEqual(low, point)
        self.assertLessEqual(point, high)
        self.assertEqual(valid, 300)

    def test_perfect_separation_has_degenerate_interval(self):
        rows = [("p%d" % i, 10, 9.0) for i in range(5)] + [("p%d" % i, 0, 1.0) for i in range(5)]
        self.assertEqual(SA.person_bootstrap_auc(rows, 10, 0, draws=50, seed=0)[:3], (1.0, 1.0, 1.0))

    def test_one_person_draws_stay_together(self):
        # 한 사람이 두 클래스를 모두 가지면 함께 뽑힌다: 사람이 1명이면 모든 추출이 원자료와 같다.
        rows = [("only", 10, 2.0), ("only", 0, 1.0)]
        self.assertEqual(SA.person_bootstrap_auc(rows, 10, 0, draws=20, seed=0)[:3], (1.0, 1.0, 1.0))

    def test_no_rows(self):
        self.assertEqual(SA.person_bootstrap_auc([], 10, 0), (None, None, None, 0))


class EditionTest(unittest.TestCase):
    def test_excluding_padded_removes_only_padded_windows(self):
        item = record(10, [1, 2, 3, 4], padded=[True, False, False, True])
        kept = SA.make_edition([item], include_padded=False)[0]
        self.assertEqual(list(kept.scores), [2.0, 3.0])
        self.assertEqual(list(kept.blinks), [32, 34])
        self.assertEqual(list(kept.padded), [False, False])
        self.assertEqual(list(SA.make_edition([item], include_padded=True)[0].scores), [1.0, 2.0, 3.0, 4.0])

    def test_video_with_only_padded_windows_is_dropped(self):
        only = record(0, [4.0], padded=[True])
        self.assertEqual(SA.make_edition([only, record(0, [1.0])], include_padded=False)[0].scores.tolist(), [1.0])
        self.assertEqual(len(SA.make_edition([only], include_padded=False)), 0)

    def test_padded_flags_use_used_blink_count(self):
        padded, blinks = SA.padded_flags([(0, 11), (0, 28), (0, 29), (2, 31)])
        self.assertEqual(padded.tolist(), [True, True, False, False])
        self.assertEqual(blinks.tolist(), [12, 29, 30, 32])


class TableTest(unittest.TestCase):
    def records(self):
        return [record(0, [1, 2, 3], person="a"), record(5, [4, 5, 6], person="b"), record(10, [7, 8, 9], person="c")]

    def test_tables_render_with_two_decimals_and_no_forbidden_words(self):
        records = self.records()
        text = "\n".join(SA.table_t1(records) + SA.table_t2(records) + SA.table_t3(records) + SA.table_t4(records, draws=20)
                          + SA.table_t5({1: records}) + SA.LIMITS)
        self.assertIn("100.00%", text)
        for word in ("성능", "정확도", "개선 확인", "검출 성능", "저자 모델 대비 개선"):
            self.assertNotIn(word, text)

    def test_t3_false_alarm_and_alarm_shares(self):
        lines = SA.table_t3(self.records())
        row = [line for line in lines if line.startswith("6.68   3")][0]
        self.assertIn("0.00% (0/1)", row)
        self.assertIn("100.00% (1/1)", row)

    def test_input_summary_reports_match_words(self):
        by_fold = {1: self.records()}
        text = "\n".join(SA.input_summary(by_fold, [], [], 0.0, [1e-7], 1.0))
        self.assertIn("일치", text)
        self.assertNotIn("불일치", text)
        text = "\n".join(SA.input_summary(by_fold, None, [], 0.0, [1e-3], 1.0))
        self.assertIn("불일치", text)


class CheckpointTest(unittest.TestCase):
    def test_scorer_output_feeds_records_when_a_checkpoint_root_is_given(self):
        root = os.environ.get("HMLSTM_CKPT_ROOT")
        if not root:
            self.skipTest("환경변수 HMLSTM_CKPT_ROOT(<CKPT_DIR>)가 없습니다")
        try:
            import tensorflow  # noqa: F401
        except ImportError as error:
            self.skipTest("tensorflow 를 가져올 수 없습니다: %s" % error)
        from improve.hmlstm import scoring as SC
        from improve.hmlstm import stage2 as S2
        means, stds = S2.load_stats()[3]
        with SC.FoldScorer(3, SC.checkpoint_prefix(root, 1, 3), means, stds, SC.load_network()) as scorer:
            scores = scorer.score(np.random.RandomState(1).randn(6, 30, 4))
        item = record(10, scores)
        self.assertTrue(0.0 <= SA.video_value(item, "max") <= 10.0)


if __name__ == "__main__":
    unittest.main()
