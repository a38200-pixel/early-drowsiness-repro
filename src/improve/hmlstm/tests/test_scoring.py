"""scoring 시험: 가짜 Training.py 텍스트와 가짜 점수 목록을 쓴다. 실제 third_party 정의부와 체크포인트를 쓰는 시험은 조건이 없으면 건너뛴다."""
import contextlib
import io
import os
import sys
import unittest

import numpy as np

from improve.hmlstm import scoring as SC

FAKE_TRAINING = "\n".join([
    "import sys",
    "SEEN_FIRST_PATH = sys.path[0]",
    "def Network(*args, **kwargs):",
    "    return 'fake network'",
    "print('definitions executed')",
    "load=False  # 경계",
    "raise RuntimeError('training loop must not run')",
    "",
])


class DefinitionsTest(unittest.TestCase):
    def test_exec_stops_before_the_load_line(self):
        namespace = SC.load_definitions(FAKE_TRAINING)
        self.assertEqual(namespace["Network"](), "fake network")
        self.assertNotIn("load", namespace)                                       # load= 줄과 그 뒤는 실행되지 않았다

    def test_print_in_the_definitions_is_swallowed(self):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            SC.load_definitions(FAKE_TRAINING)
        self.assertEqual(buffer.getvalue(), "")

    def test_missing_boundary_marker_raises(self):
        with self.assertRaises(SC.ScoringError):
            SC.load_definitions("def Network():\n    return 1\n")

    def test_executable_statements_in_the_definitions_raise(self):
        for body in ("for i in range(3):\n    pass\n", "Train(1, 2)\n", "while False:\n    pass\n", "with open('x') as f:\n    pass\n"):
            text = "def Network():\n    return 1\n" + body + "load=False\n"
            with self.assertRaises(SC.ScoringError, msg=body):
                SC.load_definitions(text)

    def test_docstring_and_print_are_allowed(self):
        text = '"""docstring"""\nimport os\nVALUE = 3\nprint("x")\ndef Network():\n    return VALUE\nload=False\n'
        self.assertEqual(SC.load_definitions(text)["Network"](), 3)

    def test_sys_path_and_bytecode_flag_are_restored(self):
        before_path, before_flag = list(sys.path), sys.dont_write_bytecode
        namespace = SC.load_definitions(FAKE_TRAINING, upstream_dir="fake_upstream_dir")
        self.assertEqual(namespace["SEEN_FIRST_PATH"], "fake_upstream_dir")        # exec 동안에는 맨 앞에 들어 있었다
        self.assertEqual(sys.path, before_path)
        self.assertEqual(sys.dont_write_bytecode, before_flag)

    def test_sys_path_is_restored_when_the_definitions_fail(self):
        before_path, before_flag = list(sys.path), sys.dont_write_bytecode
        with self.assertRaises(ZeroDivisionError):
            SC.load_definitions("VALUE = 1 / 0\nload=False\n", upstream_dir="fake_upstream_dir")
        self.assertEqual(sys.path, before_path)
        self.assertEqual(sys.dont_write_bytecode, before_flag)


class AggregateTest(unittest.TestCase):
    def test_mean_min_max_per_window_and_per_fold_values(self):
        result = SC.aggregate({3: [5.0, 0.0], 1: [1.0, 2.0], 2: [3.0, 4.0]})
        np.testing.assert_allclose(result.mean, [3.0, 2.0])
        np.testing.assert_allclose(result.min, [1.0, 0.0])
        np.testing.assert_allclose(result.max, [5.0, 4.0])
        self.assertEqual(sorted(result.per_fold), [1, 2, 3])
        np.testing.assert_allclose(result.per_fold[3], [5.0, 0.0])

    def test_empty_input_and_unequal_lengths_are_rejected(self):
        with self.assertRaises(SC.ScoringError):
            SC.aggregate({})
        with self.assertRaises(ValueError):
            SC.aggregate({1: [1.0, 2.0], 2: [1.0]})

    def test_no_windows_gives_empty_aggregates(self):
        result = SC.aggregate({1: np.zeros(0), 2: np.zeros(0)})
        self.assertEqual((len(result.mean), len(result.min), len(result.max)), (0, 0, 0))

    def test_video_mean_clips_to_zero_to_ten(self):
        self.assertEqual(SC.video_mean([-1.0, 5.0, 20.0]), 5.0)
        self.assertEqual(SC.video_mean(np.array([2.0, 4.0], dtype=np.float32)), 3.0)
        self.assertIsNone(SC.video_mean([]))

    def test_checkpoint_prefix_follows_the_b1_layout(self):
        self.assertEqual(SC.checkpoint_prefix("root", 1, 3), os.path.join("root", "b1_seed1", "seed1", "fold3", "my_model3"))


class RealDefinitionsTest(unittest.TestCase):
    def test_third_party_definitions_expose_network(self):
        if not os.path.isfile(SC.TRAINING_PATH):
            self.skipTest("third_party Training.py 가 없습니다")
        try:
            import tensorflow  # noqa: F401
        except ImportError as error:
            self.skipTest("tensorflow 를 가져올 수 없습니다: %s" % error)
        before_path = list(sys.path)
        self.assertTrue(callable(SC.load_network()))
        self.assertEqual(sys.path, before_path)
        with open(SC.TRAINING_PATH, "r", encoding="utf-8") as handle:
            lines = handle.read().split("\n")
        self.assertEqual(sum(1 for line in lines if line.startswith(SC.BOUNDARY_PREFIX)), 1)     # 경계 줄이 하나


class RealCheckpointTest(unittest.TestCase):
    def test_fold_scorer_output_range_when_a_checkpoint_root_is_given(self):
        root = os.environ.get("HMLSTM_CKPT_ROOT")
        if not root:
            self.skipTest("환경변수 HMLSTM_CKPT_ROOT(<CKPT_DIR>)가 없습니다")
        try:
            import tensorflow  # noqa: F401
        except ImportError as error:
            self.skipTest("tensorflow 를 가져올 수 없습니다: %s" % error)
        from improve.hmlstm import stage2 as S2
        means, stds = S2.load_stats()[3]
        with SC.FoldScorer(3, SC.checkpoint_prefix(root, 1, 3), means, stds, SC.load_network()) as scorer:
            windows = np.random.RandomState(1).randn(4, 30, 4)
            scores = scorer.score(windows)
            self.assertEqual(scores.shape, (4,))
            self.assertTrue(np.all(scores > 0) and np.all(scores < 10))
            self.assertEqual(scorer.score(np.zeros((0, 30, 4))).shape, (0,))


if __name__ == "__main__":
    unittest.main()
