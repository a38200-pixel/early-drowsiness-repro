# C2 상태기계 시험(합성 샘플). 기대값은 문서 규칙(RULE_SPEC 5절, I-04~I-08, I-W-11 확정)에서 먼저 계산한 것이다.
# 프레임 번호 f 의 타임스탬프는 f / fps 이다. 닫힘 EAR 0.10, 열림 EAR 0.30, 임계값 0.21(이하이면 닫힘).
import unittest

from improve.rules import episode_machine as EM
from improve.rules.tests import synth
from improve.rules.tests.synth import CLOSED, OPEN, miss, seg


class MachineTestCase(unittest.TestCase):
    def run_spec(self, spec, fps=synth.FPS, **kwargs):
        return EM.run_samples(synth.seq(spec, fps), **kwargs)

    def alarm_times(self, result):
        return [alarm.alarm_time_s for alarm in result.alarms]

    def assertTimes(self, result, expected):
        got = self.alarm_times(result)
        self.assertEqual(len(got), len(expected), "알람 시각 %r, 기대 %r" % (got, expected))
        for value, wanted in zip(got, expected):
            self.assertAlmostEqual(value, wanted, places=6)

    def statuses(self, result):
        return [episode.status for episode in result.episodes]


class BasicTest(MachineTestCase):
    def test_s01_open_eyes_no_alarm(self):
        # 열림 10초: 구간도 알람도 없다.
        result = self.run_spec([seg(300, OPEN)])
        self.assertEqual((result.alarms, result.episodes, result.unjudgeable), ([], [], []))

    def test_s02_closed_ten_seconds_one_alarm(self):
        # 닫힘 프레임 0..299. 구간 시작 0. 길이가 3.0 에 닿는 첫 샘플은 프레임 90(3.0초). 구간당 1회.
        result = self.run_spec([seg(300, CLOSED)])
        self.assertTimes(result, [3.0])
        self.assertAlmostEqual(result.alarms[0].duration_at_alarm_s, 3.0, places=6)
        self.assertEqual(self.statuses(result), [EM.STATUS_KEPT_AT_END])
        episode = result.episodes[0]
        self.assertAlmostEqual(episode.start_s, 0.0)
        self.assertAlmostEqual(episode.end_s, synth.at(299), places=6)    # 영상 끝: 확인된 마지막 닫힘 시각

    def test_s03_just_before_three_seconds(self):
        # 닫힘 프레임 0..88(89프레임), 열림 89부터. 구간 끝 = 열린 구간의 시작 = 89/30 = 2.9667 < 3.0 이므로 알람 없음.
        result = self.run_spec([seg(89, CLOSED), seg(40, OPEN)])
        self.assertEqual(result.alarms, [])
        self.assertEqual(self.statuses(result), [EM.STATUS_RELEASED])
        self.assertAlmostEqual(result.episodes[0].end_s, synth.at(89), places=6)
        self.assertAlmostEqual(result.episodes[0].length_s, synth.at(89), places=6)

    def test_s04_exactly_three_seconds_included(self):
        # 닫힘 프레임 0..89(90프레임), 열림은 프레임 90(3.0초). 구간 끝 = 3.0 이므로 길이 3.0 은 "이상"에 포함되어 알람 1회(3.0초).
        result = self.run_spec([seg(90, CLOSED), seg(40, OPEN)])
        self.assertTimes(result, [3.0])
        self.assertEqual(self.statuses(result), [EM.STATUS_RELEASED])
        self.assertAlmostEqual(result.episodes[0].end_s, 3.0, places=6)
        self.assertTrue(result.episodes[0].alarmed)

    def test_s04b_just_after_three_seconds(self):
        # 닫힘 91프레임: 프레임 90(3.0초)이 닫힘이므로 그 샘플에서 알람. 구간 끝 91/30. 알람은 1회뿐이다.
        result = self.run_spec([seg(91, CLOSED), seg(40, OPEN)])
        self.assertTimes(result, [3.0])
        self.assertAlmostEqual(result.episodes[0].end_s, synth.at(91), places=6)

    def test_s05_threshold_boundary_is_closed(self):
        # EAR 가 임계값과 정확히 같으면(<=) 닫힘. S-02 와 같은 결과.
        result = self.run_spec([seg(300, 0.21)])
        self.assertTimes(result, [3.0])
        # 임계값보다 조금이라도 크면 열림.
        result = self.run_spec([seg(300, 0.2100001)])
        self.assertEqual(result.alarms, [])


class ReleaseTest(MachineTestCase):
    def test_s06_short_gap_is_included(self):
        # 닫힘 0..59, 열림 60..67(8프레임), 닫힘 68..127. 프레임 68 에서 열림 연속 = 68/30 - 60/30 = 0.2667 < 0.3 이므로 해제 안 됨.
        # 하나의 구간이 이어지고 길이는 시작(0)부터 재므로 프레임 90(3.0초)에서 알람 1회.
        result = self.run_spec([seg(60, CLOSED), seg(8, OPEN), seg(60, CLOSED)])
        self.assertTimes(result, [3.0])
        self.assertEqual(len(result.episodes), 1)
        self.assertEqual(self.statuses(result), [EM.STATUS_KEPT_AT_END])

    def test_s07_release_at_point_three(self):
        # 닫힘 0..59, 열림 60..68(9프레임), 닫힘 69..128. 프레임 69 에서 열림 연속 = 69/30 - 60/30 = 0.3 이상이므로 해제.
        # 첫 구간: 시작 0, 끝 = 열린 구간의 시작 60/30 = 2.0, 길이 2.0(알람 없음). 둘째 구간: 시작 69/30, 영상 끝에서 3.0초 미만이라 폐기.
        result = self.run_spec([seg(60, CLOSED), seg(9, OPEN), seg(60, CLOSED)])
        self.assertEqual(result.alarms, [])
        self.assertEqual(self.statuses(result), [EM.STATUS_RELEASED, EM.STATUS_DISCARDED_AT_END])
        first = result.episodes[0]
        self.assertAlmostEqual(first.start_s, 0.0)
        self.assertAlmostEqual(first.end_s, 2.0, places=6)
        self.assertAlmostEqual(first.length_s, 2.0, places=6)
        self.assertAlmostEqual(result.episodes[1].start_s, synth.at(69), places=6)

    def test_release_boundary_independent_of_fps(self):
        # 60 fps 에서 열림 17프레임 = 0.2833초(해제 안 됨), 18프레임 = 0.3초(해제).
        bridged = self.run_spec([seg(120, CLOSED), seg(17, OPEN), seg(120, CLOSED)], fps=60.0)
        self.assertEqual(len(bridged.episodes), 1)
        released = self.run_spec([seg(120, CLOSED), seg(18, OPEN), seg(120, CLOSED)], fps=60.0)
        self.assertEqual(len(released.episodes), 2)
        self.assertEqual(released.episodes[0].status, EM.STATUS_RELEASED)

    def test_one_alarm_per_episode(self):
        # 닫힘 0..119(4초), 열림 120..129(10프레임 = 0.333초), 닫힘 130..259.
        # 프레임 130 에서 열림 연속 130/30 - 120/30 = 0.333 >= 0.3 이므로 해제. 첫 구간: 알람 3.0. 둘째 구간 시작 130/30 = 4.3333,
        # 알람은 시작 + 3.0 = 7.3333(프레임 220). 알람은 구간마다 1회, 모두 2회.
        result = self.run_spec([seg(120, CLOSED), seg(10, OPEN), seg(130, CLOSED)])
        self.assertTimes(result, [3.0, synth.at(130) + 3.0])
        self.assertEqual([alarm.episode_id for alarm in result.alarms], [1, 2])
        self.assertAlmostEqual(result.episodes[0].end_s, 4.0, places=6)


class MissingFaceTest(MachineTestCase):
    def test_s08_short_missing_keeps_state(self):
        # 닫힘 0..29, 얼굴 없음 30..56(27프레임), 닫힘 57..176. 프레임 57 에서 공백 = 57/30 - 30/30 = 0.9 < 1.0 이므로 상태 유지.
        # 구간 시간은 0 부터 이어지므로 프레임 90(3.0초)에서 알람 1회. 판정 불가 구간 없음.
        result = self.run_spec([seg(30, CLOSED), miss(27), seg(120, CLOSED)])
        self.assertTimes(result, [3.0])
        self.assertEqual(result.unjudgeable, [])
        self.assertEqual(len(result.episodes), 1)

    def test_s09_one_second_missing_is_unjudgeable(self):
        # 닫힘 0..29, 얼굴 없음 30..59(30프레임), 닫힘 60..179. 프레임 60 에서 공백 = 60/30 - 30/30 = 1.0 이상이므로 판정 불가
        # 구간 [1.0, 2.0] 길이 1.0. 첫 구간(시작 0)은 폐기. 새 구간은 프레임 60(2.0초)에서 시작해 알람은 2.0 + 3.0 = 5.0(프레임 150).
        # 구간이 이어졌다면 알람은 3.0 이었을 것이다.
        result = self.run_spec([seg(30, CLOSED), miss(30), seg(120, CLOSED)])
        self.assertEqual(len(result.unjudgeable), 1)
        item = result.unjudgeable[0]
        self.assertAlmostEqual(item.start_s, 1.0, places=6)
        self.assertAlmostEqual(item.end_s, 2.0, places=6)
        self.assertAlmostEqual(item.length_s, 1.0, places=6)
        self.assertEqual(self.statuses(result), [EM.STATUS_DISCARDED_UNJUDGEABLE, EM.STATUS_KEPT_AT_END])
        self.assertIsNone(result.episodes[0].end_s)
        self.assertAlmostEqual(result.episodes[1].start_s, 2.0, places=6)
        self.assertTimes(result, [5.0])

    def test_unjudgeable_boundary_at_60fps(self):
        # 60 fps 에서 얼굴 없음 59프레임 = 0.9833초(유지), 60프레임 = 1.0초(판정 불가).
        keep = self.run_spec([seg(30, CLOSED), miss(59), seg(10, CLOSED)], fps=60.0)
        self.assertEqual(keep.unjudgeable, [])
        cut = self.run_spec([seg(30, CLOSED), miss(60), seg(10, CLOSED)], fps=60.0)
        self.assertEqual(len(cut.unjudgeable), 1)

    def test_long_missing_run_detected_on_missing_samples(self):
        # 얼굴 없음이 60프레임(2초) 이어지면 얼굴 없음 샘플 자체에서 1초에 닿는다. 닫힘 0..29, 없음 30..89, 닫힘 90..
        # 판정 불가 [1.0, 3.0] 길이 2.0.
        result = self.run_spec([seg(30, CLOSED), miss(60), seg(30, CLOSED)])
        self.assertEqual(len(result.unjudgeable), 1)
        self.assertAlmostEqual(result.unjudgeable[0].start_s, 1.0, places=6)
        self.assertAlmostEqual(result.unjudgeable[0].end_s, 3.0, places=6)
        self.assertAlmostEqual(result.unjudgeable[0].length_s, 2.0, places=6)

    def test_s10_alarm_survives_unjudgeable(self):
        # 닫힘 0..99(프레임 90 에서 알람 3.0), 얼굴 없음 100..129, 열림 130. 프레임 130 의 공백 = 130/30 - 100/30 = 1.0 -> 판정 불가
        # [3.3333, 4.3333]. 구간은 폐기되지만 이미 울린 알람은 유지(알람 1개, 구간 상태 discarded_unjudgeable, alarmed True).
        result = self.run_spec([seg(100, CLOSED), miss(30), seg(30, OPEN)])
        self.assertTimes(result, [3.0])
        self.assertEqual(self.statuses(result), [EM.STATUS_DISCARDED_UNJUDGEABLE])
        self.assertTrue(result.episodes[0].alarmed)
        self.assertEqual(len(result.unjudgeable), 1)
        self.assertAlmostEqual(result.unjudgeable[0].start_s, synth.at(100), places=6)
        self.assertAlmostEqual(result.unjudgeable[0].length_s, 1.0, places=6)

    def test_missing_until_end_is_unjudgeable_to_last_timestamp(self):
        # 얼굴 없음이 입력 끝까지 1초 이상 이어지면 판정 불가 구간의 끝은 마지막 타임스탬프이다.
        result = self.run_spec([seg(30, OPEN), miss(60)])
        self.assertEqual(len(result.unjudgeable), 1)
        self.assertAlmostEqual(result.unjudgeable[0].start_s, 1.0, places=6)
        self.assertAlmostEqual(result.unjudgeable[0].end_s, synth.at(89), places=6)

    def test_trailing_short_missing_is_not_unjudgeable(self):
        result = self.run_spec([seg(30, OPEN), miss(10)])
        self.assertEqual(result.unjudgeable, [])


class TimestampTest(MachineTestCase):
    def test_s11_irregular_timestamps_use_time_not_frame_count(self):
        # 닫힘 샘플 시각 0.0 1.0 2.0 2.9 3.0 3.5: 시작 0 에서 길이 3.0 에 닿는 첫 샘플은 3.0(여섯 샘플 중 다섯째). 프레임 수로 세면 다르다.
        samples = [(t, CLOSED, "detected") for t in (0.0, 1.0, 2.0, 2.9, 3.0, 3.5)]
        result = EM.run_samples(samples)
        self.assertTimes(result, [3.0])
        # 샘플 3개뿐이어도 시각이 5.0 이면 길이 5.0 >= 3.0 이므로 5.0 에서 알람.
        result = EM.run_samples([(0.0, CLOSED, "detected"), (0.5, CLOSED, "detected"), (5.0, CLOSED, "detected")])
        self.assertTimes(result, [5.0])

    def test_alarm_time_independent_of_fps(self):
        for fps in (15.0, 30.0, 60.0):
            with self.subTest(fps=fps):
                result = EM.run_samples(synth.seq([seg(int(fps * 10), CLOSED)], fps))
                self.assertTimes(result, [3.0])

    def test_timestamps_must_not_go_backwards(self):
        machine = EM.EpisodeMachine()
        machine.feed(1.0, CLOSED, "detected")
        with self.assertRaises(ValueError):
            machine.feed(0.5, CLOSED, "detected")

    def test_input_validation(self):
        machine = EM.EpisodeMachine()
        with self.assertRaises(ValueError):
            machine.feed(0.0, CLOSED, "camera")
        with self.assertRaises(ValueError):
            machine.feed(float("nan"), CLOSED, "detected")
        with self.assertRaises(ValueError):
            machine.feed(0.0, float("inf"), "detected")
        machine.finish()
        with self.assertRaises(RuntimeError):
            machine.feed(0.0, CLOSED, "detected")


class EndOfVideoTest(MachineTestCase):
    def test_s12a_short_episode_discarded_at_end(self):
        # 닫힘 60프레임(마지막 닫힘 59/30 = 1.9667) 뒤 영상 끝: 3.0 미만이므로 폐기, 알람 없음.
        result = self.run_spec([seg(60, CLOSED)])
        self.assertEqual(result.alarms, [])
        self.assertEqual(self.statuses(result), [EM.STATUS_DISCARDED_AT_END])

    def test_s12b_long_episode_kept_and_alarm_kept(self):
        # 닫힘 120프레임: 프레임 90 에서 알람 3.0. 영상 끝에서 구간 유지, 구간 끝 = 마지막 닫힘 119/30.
        result = self.run_spec([seg(120, CLOSED)])
        self.assertTimes(result, [3.0])
        self.assertEqual(self.statuses(result), [EM.STATUS_KEPT_AT_END])
        self.assertAlmostEqual(result.episodes[0].end_s, synth.at(119), places=6)

    def test_video_ends_during_open_gap(self):
        # 닫힘 120프레임 뒤 열림 5프레임에서 끝: 열린 구간이 해제 전이므로 구간 끝 = 열린 구간의 시작 120/30 = 4.0, 알람 유지.
        result = self.run_spec([seg(120, CLOSED), seg(5, OPEN)])
        self.assertTimes(result, [3.0])
        self.assertEqual(self.statuses(result), [EM.STATUS_KEPT_AT_END])
        self.assertAlmostEqual(result.episodes[0].end_s, 4.0, places=6)


class ThresholdAndSourceTest(MachineTestCase):
    def test_s13_threshold_changes_alarm_count(self):
        # EAR 0.18 일정 10초: 임계값 0.15 에서는 열림(알람 0), 0.21 과 0.25 에서는 닫힘(알람 1).
        spec = [seg(300, 0.18)]
        self.assertEqual(len(self.run_spec(spec, threshold=0.15).alarms), 0)
        self.assertEqual(len(self.run_spec(spec, threshold=0.21).alarms), 1)
        self.assertEqual(len(self.run_spec(spec, threshold=0.25).alarms), 1)

    def test_tracked_not_valid_by_default(self):
        # 기본 유효 출처는 detected 하나(제안). tracked 로만 이어진 닫힘은 얼굴 없음이라 1초 뒤 판정 불가가 되고 알람이 없다.
        spec = [seg(300, CLOSED, "tracked")]
        result = self.run_spec(spec)
        self.assertEqual(result.alarms, [])
        self.assertEqual(len(result.unjudgeable), 1)
        self.assertAlmostEqual(result.unjudgeable[0].start_s, 0.0)
        self.assertAlmostEqual(result.unjudgeable[0].end_s, synth.at(299), places=6)

    def test_tracked_valid_when_chosen(self):
        result = self.run_spec([seg(300, CLOSED, "tracked")], valid_sources=("detected", "tracked"))
        self.assertTimes(result, [3.0])
        self.assertEqual(result.unjudgeable, [])

    def test_none_source_or_missing_ear_is_missing(self):
        result = self.run_spec([seg(30, CLOSED), (60, CLOSED, "none"), seg(30, CLOSED)])
        self.assertEqual(len(result.unjudgeable), 1)
        result = self.run_spec([seg(30, CLOSED), (60, None, "detected"), seg(30, CLOSED)])
        self.assertEqual(len(result.unjudgeable), 1)

    def test_alarm_fields(self):
        result = self.run_spec([seg(120, CLOSED)], threshold=0.21, reliability="신뢰 불가")
        alarm = result.alarms[0]
        self.assertEqual(alarm.reliability, "신뢰 불가")
        self.assertEqual(alarm.threshold, 0.21)
        self.assertEqual(alarm.alarm_after_s, 3.0)
        self.assertEqual(alarm.episode_id, 1)
        self.assertAlmostEqual(alarm.episode_start_s, 0.0)

    def test_invalid_settings(self):
        for kwargs in ({"threshold": 0}, {"threshold": -0.1}, {"alarm_after_s": 0}, {"release_open_s": float("nan")},
                       {"valid_sources": ("none",)}, {"valid_sources": ()}, {"valid_sources": ("video",)}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    EM.EpisodeMachine(**kwargs)


if __name__ == "__main__":
    unittest.main()
