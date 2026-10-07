# 구간 길이 정의(I-17, RULE_SPEC.md 13절) 시험. 코드는 바꾸지 않고 현재 동작이 정의와 같은지만 고정한다. 합성 샘플만 쓴다.
# 정의(프레임 길이 방식): 길이 = 그 구간을 확인하는 샘플의 시각 - 구간 시작 샘플의 시각. 닫힘 N프레임 뒤 다음 샘플이 오면(닫힘이든 열림이든)
# 길이는 N/fps 이다. 감김은 길이 >= 3.0초에서 알람, 열림은 길이 >= 0.3초에서 해제이다. 기대값은 아래 식에서 먼저 계산했다.
#   fps 30: 3.0초 = 90프레임, 0.3초 = 9프레임.
#   fps 23.976: 3.0초 = 71.928프레임이므로 72프레임부터 알람(72/23.976 = 3.003초, 71프레임은 2.961초),
#               0.3초 = 7.193프레임이므로 8프레임부터 해제(8/23.976 = 0.334초, 7프레임은 0.292초).
import unittest

from improve.rules import episode_machine as EM
from improve.rules.tests import synth
from improve.rules.tests.synth import CLOSED, OPEN, seg

FPS30 = 30.0
FPS24 = 23.976


class LengthDefinitionTestCase(unittest.TestCase):
    def run_spec(self, spec, fps):
        return EM.run_samples(synth.seq(spec, fps))

    def alarm_times(self, result):
        return [alarm.alarm_time_s for alarm in result.alarms]


class Fps30Test(LengthDefinitionTestCase):
    def test_90_closed_then_closed_sample_alarms(self):
        # 닫힘 90프레임(0~89) 뒤 다음 샘플(프레임 90)이 닫힘: 길이 90/30 = 3.0 -> 알람 시각 3.0
        result = self.run_spec([seg(91, CLOSED)], FPS30)
        self.assertEqual(len(result.alarms), 1)
        self.assertAlmostEqual(result.alarms[0].alarm_time_s, 3.0, places=9)
        self.assertAlmostEqual(result.alarms[0].duration_at_alarm_s, 3.0, places=9)

    def test_90_closed_then_open_sample_alarms(self):
        # 닫힘 90프레임 뒤 다음 샘플이 열림: 구간 끝 = 열림 시작 시각 3.0 -> 길이 3.0 -> 알람 시각 3.0
        result = self.run_spec([seg(90, CLOSED), seg(1, OPEN)], FPS30)
        self.assertEqual(len(result.alarms), 1)
        self.assertAlmostEqual(result.alarms[0].alarm_time_s, 3.0, places=9)
        self.assertAlmostEqual(result.alarms[0].duration_at_alarm_s, 3.0, places=9)

    def test_89_closed_then_open_sample_does_not_alarm(self):
        # 닫힘 89프레임 뒤 열림: 길이 89/30 = 2.9667 < 3.0 -> 알람 없음
        result = self.run_spec([seg(89, CLOSED), seg(1, OPEN)], FPS30)
        self.assertEqual(result.alarms, [])

    def test_89_closed_then_closed_sample_without_next_does_not_alarm(self):
        # 닫힘 샘플 90개(0~89)에서 끝: 확인하는 마지막 샘플은 프레임 89 라 길이 89/30 = 2.9667 -> 알람 없음
        result = self.run_spec([seg(90, CLOSED)], FPS30)
        self.assertEqual(result.alarms, [])

    def test_open_nine_frames_then_closed_releases(self):
        # 닫힘 60프레임(0~59), 열림 9프레임(60~68), 닫힘 1프레임(69): 열림 길이 69/30 - 60/30 = 0.3 -> 해제. 첫 구간 끝 = 2.0
        result = self.run_spec([seg(60, CLOSED), seg(9, OPEN), seg(1, CLOSED)], FPS30)
        self.assertEqual(len(result.episodes), 2)
        first = result.episodes[0]
        self.assertEqual(first.status, EM.STATUS_RELEASED)
        self.assertAlmostEqual(first.end_s, 2.0, places=9)
        self.assertAlmostEqual(first.length_s, 2.0, places=9)

    def test_open_eight_frames_then_closed_keeps_episode(self):
        # 열림 8프레임: 68/30 - 60/30 = 0.2667 < 0.3 -> 같은 구간 유지(해제 없음). 입력이 끝나는 시점에 3.0초 미만이라 폐기 상태.
        result = self.run_spec([seg(60, CLOSED), seg(8, OPEN), seg(1, CLOSED)], FPS30)
        self.assertEqual(len(result.episodes), 1)
        self.assertEqual(result.episodes[0].status, EM.STATUS_DISCARDED_AT_END)
        self.assertAlmostEqual(result.episodes[0].start_s, 0.0, places=9)


class Fps23976Test(LengthDefinitionTestCase):
    def test_72_closed_then_open_alarms(self):
        # 닫힘 72프레임 뒤 열림: 길이 72/23.976 = 3.003 >= 3.0 -> 알람 시각 72/23.976
        result = self.run_spec([seg(72, CLOSED), seg(1, OPEN)], FPS24)
        self.assertEqual(len(result.alarms), 1)
        self.assertAlmostEqual(result.alarms[0].alarm_time_s, 72 / FPS24, places=9)
        self.assertAlmostEqual(result.alarms[0].duration_at_alarm_s, 72 / FPS24, places=9)

    def test_71_closed_then_open_does_not_alarm(self):
        # 닫힘 71프레임 뒤 열림: 71/23.976 = 2.961 < 3.0 -> 알람 없음
        result = self.run_spec([seg(71, CLOSED), seg(1, OPEN)], FPS24)
        self.assertEqual(result.alarms, [])

    def test_72_closed_then_closed_sample_alarms(self):
        # 닫힘 72프레임 뒤 다음 샘플(프레임 72)이 닫힘: 72/23.976 = 3.003 -> 알람 시각 72/23.976
        result = self.run_spec([seg(73, CLOSED)], FPS24)
        self.assertEqual(len(result.alarms), 1)
        self.assertAlmostEqual(result.alarms[0].alarm_time_s, 72 / FPS24, places=9)

    def test_71_closed_then_closed_sample_does_not_alarm(self):
        # 닫힘 71프레임 뒤 다음 샘플이 닫힘(닫힘 샘플 72개, 마지막 프레임 71): 71/23.976 = 2.961 -> 알람 없음
        result = self.run_spec([seg(72, CLOSED)], FPS24)
        self.assertEqual(result.alarms, [])

    def test_open_eight_frames_then_closed_releases(self):
        # 닫힘 60프레임, 열림 8프레임, 닫힘 1프레임: 열림 길이 8/23.976 = 0.334 >= 0.3 -> 해제. 첫 구간 끝 = 60/23.976
        result = self.run_spec([seg(60, CLOSED), seg(8, OPEN), seg(1, CLOSED)], FPS24)
        self.assertEqual(len(result.episodes), 2)
        first = result.episodes[0]
        self.assertEqual(first.status, EM.STATUS_RELEASED)
        self.assertAlmostEqual(first.end_s, 60 / FPS24, places=9)

    def test_open_seven_frames_then_closed_keeps_episode(self):
        # 열림 7프레임: 7/23.976 = 0.292 < 0.3 -> 같은 구간 유지
        result = self.run_spec([seg(60, CLOSED), seg(7, OPEN), seg(1, CLOSED)], FPS24)
        self.assertEqual(len(result.episodes), 1)
        self.assertNotEqual(result.episodes[0].status, EM.STATUS_RELEASED)


if __name__ == "__main__":
    unittest.main()
