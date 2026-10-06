# 유효성 검사(I-09) 시험(합성 샘플). 기대값은 문서 규칙에서 먼저 계산했다.
import unittest

from improve.rules import validity
from improve.rules.tests import synth
from improve.rules.tests.synth import seg


class MedianTest(unittest.TestCase):
    def test_odd_and_even(self):
        self.assertEqual(validity.median([0.3, 0.1, 0.2]), 0.2)
        self.assertAlmostEqual(validity.median([0.1, 0.2, 0.3, 0.4]), 0.25, places=12)

    def test_empty(self):
        with self.assertRaises(ValueError):
            validity.median([])


class CheckReferenceTest(unittest.TestCase):
    def test_normal_when_median_above_threshold(self):
        # 600프레임(span 599/30 = 19.97초), EAR 0.30 일정 -> 중앙값 0.30 > 0.21 이므로 정상
        info = validity.check_reference(synth.seq([seg(600, 0.30)]), 0.21, 10.0)
        self.assertEqual(info["status"], validity.STATUS_OK)
        self.assertAlmostEqual(info["median"], 0.30, places=12)

    def test_undetectable_when_median_at_or_below_threshold(self):
        info = validity.check_reference(synth.seq([seg(600, 0.20)]), 0.21, 10.0)
        self.assertEqual(info["status"], validity.STATUS_UNDETECTABLE)
        # 중앙값이 임계값과 정확히 같아도(<=) 탐지 불가능
        info = validity.check_reference(synth.seq([seg(600, 0.21)]), 0.21, 10.0)
        self.assertEqual(info["status"], validity.STATUS_UNDETECTABLE)

    def test_only_detected_frames_are_used(self):
        # detected 100프레임 EAR 0.30, tracked 200프레임 EAR 0.05, 얼굴 없음 250프레임. detected 만 쓰면 중앙값 0.30(정상).
        # 모든 프레임을 쓰면 중앙값이 0.05 라 탐지 불가능이 될 것이다.
        spec = [seg(100, 0.30), seg(200, 0.05, "tracked"), synth.miss(250)]
        info = validity.check_reference(synth.seq(spec), 0.21, 5.0)
        self.assertEqual(info["status"], validity.STATUS_OK)
        self.assertEqual(info["n_detected_frames"], 100)
        self.assertEqual(info["n_frames"], 550)
        self.assertAlmostEqual(info["median"], 0.30, places=12)

    def test_short_reference_is_not_judged(self):
        # span = 99/30 = 3.3초 < 최소 길이 10초
        info = validity.check_reference(synth.seq([seg(100, 0.05)]), 0.21, 10.0)
        self.assertEqual(info["status"], validity.STATUS_SHORT)
        self.assertIsNone(info["median"])

    def test_no_detected_frames(self):
        info = validity.check_reference(synth.seq([seg(600, 0.05, "tracked")]), 0.21, 10.0)
        self.assertEqual(info["status"], validity.STATUS_SHORT)
        info = validity.check_reference([], 0.21, 10.0)
        self.assertEqual(info["status"], validity.STATUS_SHORT)

    def test_min_duration_has_no_default(self):
        with self.assertRaises(TypeError):
            validity.check_reference(synth.seq([seg(600, 0.30)]), 0.21)

    def test_min_duration_must_be_positive(self):
        for bad in (0, -1, float("nan"), None):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    validity.check_reference(synth.seq([seg(600, 0.30)]), 0.21, bad)


class ReliabilityTest(unittest.TestCase):
    def test_mapping(self):
        ok = {"status": validity.STATUS_OK}
        bad = {"status": validity.STATUS_UNDETECTABLE}
        short = {"status": validity.STATUS_SHORT}
        self.assertEqual(validity.reliability_for(ok), "정상")
        self.assertEqual(validity.reliability_for(bad), "신뢰 불가")
        self.assertEqual(validity.reliability_for(short), "미확정")


if __name__ == "__main__":
    unittest.main()
