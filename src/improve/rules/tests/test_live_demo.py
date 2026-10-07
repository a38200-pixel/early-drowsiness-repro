# 실시간 데모의 규칙 처리부(live_demo.LiveRules)가 샘플을 프레임 단위로 흘려 넣어도 일괄 재생(episode_machine.run_samples,
# replay.py 의 재생)과 같은 알람을 내는지 한 번 확인한다(웹캠, 화면, cv2 를 쓰지 않는다). 별도의 검증 항목은 더하지 않았다.
# 입력: 기본은 합성 샘플(불규칙한 프레임 간격). 환경 변수가 있으면 실제 덤프로 확인한다.
#   LIVE_DEMO_EAR_DUMP=<ear_dump.csv 경로>, LIVE_DEMO_REPLAY_DIR=<replay.py 출력 폴더(alarms_thr_<임계값>.csv)>
#   두 변수를 모두 주면 스트리밍 알람 시각을 기존 재생 산출물의 alarm_time_s(소수 6자리)와도 대조한다.
import contextlib
import csv
import io
import os
import struct
import unittest
from unittest import mock

from improve.rules import episode_machine as EM
from improve.rules import live_demo
from improve.rules import replay


def synthetic_samples():
    """결정적 합성 샘플(불규칙 간격 0.030~0.050초). 닫힘 4초, 0.2초 열림이 낀 닫힘 3.5초, 얼굴 없음 1.5초,
    후보 임계값 사이(EAR 0.165: 0.17만 닫힘)의 닫힘 4초, 출처 tracked 구간을 넣는다."""
    samples = []
    t = 0.0
    index = 0
    while t < 28.0:
        if 5.0 <= t < 9.0:
            sample = (0.10, "detected")
        elif 12.0 <= t < 15.5:
            sample = (0.30, "detected") if 13.0 <= t < 13.2 else (0.10, "detected")
        elif 17.0 <= t < 18.5:
            sample = (None, "none")
        elif 19.0 <= t < 23.0:
            sample = (0.165, "detected")
        elif 25.0 <= t < 25.5:
            sample = (0.10, "tracked")
        else:
            sample = (0.30, "detected")
        samples.append((t, sample[0], sample[1]))
        index += 1
        t += 0.030 + 0.005 * ((index * 7) % 5)
    return samples


def load_samples():
    path = os.environ.get("LIVE_DEMO_EAR_DUMP")
    if path:
        return replay.read_ear_csv(path), "실제 덤프"
    return synthetic_samples(), "합성"


def read_alarm_times(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return [row["alarm_time_s"] for row in csv.DictReader(handle)]


class StreamingMatchesBatchTest(unittest.TestCase):
    def test_streaming_alarms_equal_batch_replay(self):
        samples, kind = load_samples()
        rules = live_demo.LiveRules()
        streamed = dict((threshold, []) for threshold in rules.thresholds)
        for timestamp_s, ear_mean, source in samples:                   # 프레임이 들어오는 대로 하나씩
            for threshold, alarm in rules.update(timestamp_s, ear_mean, source):
                streamed[threshold].append(alarm)
        total = 0
        for threshold in rules.thresholds:
            batch = EM.run_samples(samples, threshold=threshold, valid_sources=live_demo.VALID_SOURCES)
            self.assertEqual(streamed[threshold], batch.alarms, "%s 샘플, 임계값 %.2f" % (kind, threshold))
            self.assertEqual(rules.alarm_count(threshold), len(batch.alarms))
            total += len(batch.alarms)
        self.assertGreater(total, 0)
        replay_dir = os.environ.get("LIVE_DEMO_REPLAY_DIR")
        if replay_dir and os.environ.get("LIVE_DEMO_EAR_DUMP"):
            compared = 0
            for threshold in rules.thresholds:                              # 재생 산출물이 있는 임계값만 대조한다
                path = os.path.join(replay_dir, "alarms_thr_%s.csv" % replay.threshold_label(threshold))
                if not os.path.exists(path):
                    continue
                compared += 1
                got = ["%.6f" % alarm.alarm_time_s for alarm in streamed[threshold]]
                self.assertEqual(got, read_alarm_times(path), "재생 산출물과 알람 시각 불일치, 임계값 %.2f" % threshold)
            self.assertGreater(compared, 0, "대조할 재생 산출물이 없습니다.")


class StickyDisplayTest(unittest.TestCase):
    """알람 표시는 R(reset) 전에는 꺼지지 않고, reset 후 꺼지며, 알람 횟수는 reset 으로 지워지지 않는다(표시 전용 상태)."""

    def run_synthetic(self, latch):
        rules = live_demo.LiveRules()
        last_ts = 0.0
        for timestamp_s, ear_mean, source in synthetic_samples():
            last_ts = timestamp_s
            for threshold, _alarm in rules.update(timestamp_s, ear_mean, source):
                latch.trigger(threshold, timestamp_s)
        return rules, last_ts

    def test_alarm_display_stays_until_reset_and_counts_are_kept(self):
        self.assertIsNone(live_demo.DISPLAY_LATCH_S)                     # 기본은 자동 꺼짐 없음
        latch = live_demo.DisplayLatch()
        rules, last_ts = self.run_synthetic(latch)
        counts = dict((threshold, rules.alarm_count(threshold)) for threshold in rules.thresholds)
        self.assertTrue(all(count > 0 for count in counts.values()))
        for threshold in rules.thresholds:
            for later in (last_ts, last_ts + 3.0, last_ts + 600.0, last_ts + 1e6):
                self.assertTrue(latch.active(threshold, later), "reset 전에 꺼짐: %.2f, +%.1f" % (threshold, later - last_ts))
        latch.reset()
        for threshold in rules.thresholds:
            self.assertFalse(latch.active(threshold, last_ts + 1.0))
        self.assertEqual(dict((threshold, rules.alarm_count(threshold)) for threshold in rules.thresholds), counts)
        latch.trigger(live_demo.MAIN_THRESHOLD, last_ts + 2.0)           # reset 뒤 새 알람은 다시 켜진다
        self.assertTrue(latch.active(live_demo.MAIN_THRESHOLD, last_ts + 50.0))

    def test_auto_off_is_optional_and_display_only(self):
        latch = live_demo.DisplayLatch(duration_s=3.0)
        latch.trigger(0.21, 10.0)
        self.assertTrue(latch.active(0.21, 12.9))
        self.assertFalse(latch.active(0.21, 13.0))
        self.assertFalse(latch.active(0.15, 10.0))                       # 알람이 없던 임계값은 켜지지 않는다


class DisplaySettingsDoNotChangeAlarmsTest(unittest.TestCase):
    def test_display_and_sound_settings_do_not_change_alarm_times(self):
        samples = synthetic_samples()
        reference = None
        for latch_s in (None, 3.0, 0.1):
            for enabled, beep_all, muted in ((False, False, False), (True, False, False), (True, True, True)):
                rules = live_demo.LiveRules()
                latch = live_demo.DisplayLatch(duration_s=latch_s)
                calls = []
                beeper = live_demo.Beeper(enabled=enabled, beep_all=beep_all, muted=muted, backend=lambda: calls.append(1))
                for timestamp_s, ear_mean, source in samples:
                    new_alarms = rules.update(timestamp_s, ear_mean, source)
                    for threshold, _alarm in new_alarms:
                        latch.trigger(threshold, timestamp_s)
                    beeper.on_new_alarms(new_alarms)
                    latch.active(live_demo.MAIN_THRESHOLD, timestamp_s)  # 표시와 깜빡임 계산을 섞어 돌려도 규칙은 같아야 한다
                    live_demo.blink_is_red(timestamp_s)
                alarms = dict((threshold, list(rules.machines[threshold].alarms)) for threshold in rules.thresholds)
                if reference is None:
                    reference = alarms
                self.assertEqual(alarms, reference, "표시/소리 설정 (%s, %s, %s, %s)" % (latch_s, enabled, beep_all, muted))
        for threshold in live_demo.THRESHOLDS:
            batch = EM.run_samples(samples, threshold=threshold, valid_sources=live_demo.VALID_SOURCES)
            self.assertEqual(reference[threshold], batch.alarms)
        # 소리 설정: 기본은 꺼짐이고, --beep 없이 --beep-all 만 주면 소리 없음, --beep 은 주 임계값(목록의 첫 값)만, 함께 주면 모두
        main, second, third = live_demo.THRESHOLDS[:3]
        alarm = reference[main][0]
        calls = []
        self.assertFalse(live_demo.Beeper(backend=lambda: calls.append(1)).on_new_alarms([(main, alarm)]))
        self.assertFalse(live_demo.Beeper(beep_all=True, backend=lambda: calls.append(1)).on_new_alarms([(main, alarm)]))
        self.assertEqual(calls, [])
        beeper = live_demo.Beeper(enabled=True, backend=lambda: calls.append(1))
        self.assertFalse(beeper.on_new_alarms([(second, alarm)]))
        self.assertTrue(beeper.on_new_alarms([(main, alarm)]))
        beeper_all = live_demo.Beeper(enabled=True, beep_all=True, backend=lambda: calls.append(1))
        self.assertTrue(beeper_all.on_new_alarms([(second, alarm), (third, alarm)]))
        self.assertEqual(calls, [1, 1])                                   # 같은 갱신의 여러 알람에도 한 번만 울린다
        beeper_other = live_demo.Beeper(enabled=True, backend=lambda: calls.append(1), primary=second)    # 주 임계값은 목록의 첫 값에서 온다
        self.assertFalse(beeper_other.on_new_alarms([(main, alarm)]))
        self.assertTrue(beeper_other.on_new_alarms([(second, alarm)]))
        self.assertEqual(calls, [1, 1, 1])
        self.assertEqual(live_demo.Beeper().label(), "sound: off")
        self.assertEqual(live_demo.Beeper(enabled=True).label(), "sound: on")
        self.assertEqual(live_demo.Beeper(enabled=True, muted=True).label(), "sound: off")


class FakeWinsound(object):
    """가짜 winsound. 실제 winsound 처럼 SND_ASYNC 와 SND_MEMORY 가 함께 오면 RuntimeError 를 낸다. 소리는 재생하지 않는다."""
    SND_ASYNC = 1
    SND_NODEFAULT = 2
    SND_MEMORY = 4
    SND_LOOP = 8
    SND_NOSTOP = 16
    SND_PURGE = 64

    def __init__(self, error=None, on_call=None):
        self.calls = []
        self.error = error
        self.on_call = on_call

    def PlaySound(self, sound, flags):
        if flags & self.SND_ASYNC and flags & self.SND_MEMORY:
            raise RuntimeError("Cannot play asynchronously from memory")
        if self.error is not None:
            raise self.error
        self.calls.append((sound, flags))
        if self.on_call is not None:
            self.on_call(len(self.calls))


def sample_alarm():
    return EM.run_samples(synthetic_samples(), threshold=live_demo.MAIN_THRESHOLD, valid_sources=live_demo.VALID_SOURCES).alarms[0]


def sign_changes(values):
    return sum(1 for a, b in zip(values, values[1:]) if (a > 0) != (b > 0))


class SirenPieceTest(unittest.TestCase):
    """(a) 조각 바이트가 유효한 WAV(RIFF/WAVE, PCM, 16비트 모노, 44100 Hz)이고 0.25초이며, 전체 사이렌은 조각 12개(3.0초)와 같다."""

    def check_wav(self, wav, seconds):
        self.assertEqual(wav[0:4], b"RIFF")
        self.assertEqual(struct.unpack("<I", wav[4:8])[0], len(wav) - 8)
        self.assertEqual(wav[8:12], b"WAVE")
        self.assertEqual(wav[12:16], b"fmt ")
        fmt_size, audio_format, channels, rate, byte_rate, block_align, bits = struct.unpack("<IHHIIHH", wav[16:36])
        self.assertEqual((fmt_size, audio_format, channels, rate, byte_rate, block_align, bits),
                         (16, 1, 1, 44100, 88200, 2, 16))                  # PCM, 모노, 44100 Hz, 16비트
        self.assertEqual(wav[36:40], b"data")
        data_size = struct.unpack("<I", wav[40:44])[0]
        self.assertEqual(len(wav), 44 + data_size)
        self.assertEqual(data_size / float(2 * 44100), seconds)
        return struct.unpack("<%dh" % (data_size // 2), wav[44:])

    def test_pieces_are_valid_quarter_second_wavs_and_full_siren_is_twelve_pieces(self):
        pieces = live_demo.siren_pieces()
        self.assertIs(pieces, live_demo.siren_pieces())                    # 한 번 만들어 캐시한다
        self.assertEqual(len(pieces), 2)
        for piece, frequency in zip(pieces, (880, 1320)):
            samples = self.check_wav(piece, 0.25)                          # 조각 하나는 0.25초
            peak = max(abs(value) for value in samples)
            self.assertLessEqual(peak, 32767)
            self.assertGreater(peak, int(32767 * 0.9))                     # 최대 음량 쪽, 클리핑은 아님
            self.assertAlmostEqual(sign_changes(samples), 2 * frequency * 0.25, delta=3)
        self.assertEqual(live_demo.SIREN_PIECE_COUNT, 12)
        full = live_demo.siren_wav()
        self.assertIs(full, live_demo.siren_wav())
        self.check_wav(full, 3.0)                                          # 전체 길이 3.0초
        self.assertEqual(full[44:], (pieces[0][44:] + pieces[1][44:]) * 6)   # 조각 12개를 번갈아 이어 붙인 값


class SirenPlaybackTest(unittest.TestCase):
    def test_playsound_never_combines_async_and_memory(self):
        """(b) PlaySound 의 flags 에 SND_ASYNC 와 SND_MEMORY 가 함께 들어가지 않는다(함께 오면 가짜가 RuntimeError 를 낸다)."""
        alarm = sample_alarm()
        fake = FakeWinsound()
        with mock.patch.object(live_demo, "_winsound_module", return_value=fake):
            beeper = live_demo.Beeper(enabled=True)
            self.assertTrue(beeper.on_new_alarms([(live_demo.THRESHOLDS[0], alarm), (live_demo.THRESHOLDS[2], alarm)]))   # 같은 갱신의 두 알람
            beeper.player.join(10.0)
        self.assertIsNone(beeper.player.error)
        self.assertEqual(len(fake.calls), live_demo.SIREN_PIECE_COUNT)       # 스레드 하나가 조각 12개를 재생한다
        low, high = live_demo.siren_pieces()
        self.assertEqual([sound for sound, _flags in fake.calls], [low, high] * 6)
        for _sound, flags in fake.calls:
            self.assertTrue(flags & fake.SND_MEMORY)
            self.assertFalse(flags & fake.SND_ASYNC)
            self.assertFalse(flags & fake.SND_LOOP)
            self.assertFalse(flags & fake.SND_PURGE)
        self.assertEqual(beeper.label(), "sound: on")
        # 정지 신호: 첫 조각이 끝난 직후 정지하면 다음 조각은 재생되지 않는다(조각 사이마다 확인)
        holder = {}
        stopper_fake = FakeWinsound(on_call=lambda number: holder["beeper"].stop() if number == 1 else None)
        with mock.patch.object(live_demo, "_winsound_module", return_value=stopper_fake):
            holder["beeper"] = live_demo.Beeper(enabled=True)
            holder["beeper"].on_new_alarms([(live_demo.MAIN_THRESHOLD, alarm)])
            holder["beeper"].player.join(10.0)
        self.assertEqual(len(stopper_fake.calls), 1)

    def test_playback_error_prints_one_line_and_does_not_change_alarms(self):
        """(c) PlaySound 가 예외를 던지면 콘솔에 한 줄만 나오고(프로세스당 한 번) 규칙 알람 시각은 그대로이다."""
        samples = synthetic_samples()
        fake = FakeWinsound(error=OSError("no sound device"))
        buffer = io.StringIO()
        with mock.patch.object(live_demo, "_winsound_module", return_value=fake), contextlib.redirect_stdout(buffer):
            rules = live_demo.LiveRules()
            beeper = live_demo.Beeper(enabled=True, beep_all=True)
            for timestamp_s, ear_mean, source in samples:
                beeper.on_new_alarms(rules.update(timestamp_s, ear_mean, source))
                beeper.player.join(10.0)
        self.assertEqual(buffer.getvalue().splitlines(), ["sound error: OSError: no sound device"])
        self.assertEqual(beeper.label(), "sound: error")
        for threshold in rules.thresholds:
            batch = EM.run_samples(samples, threshold=threshold, valid_sources=live_demo.VALID_SOURCES)
            self.assertEqual(list(rules.machines[threshold].alarms), batch.alarms)


class ThresholdListTest(unittest.TestCase):
    def test_default_thresholds_and_first_value_is_primary(self):
        """(a) --thresholds 의 기본값은 0.15,0.16,0.17 이고 목록의 첫 값이 주 임계값이다(0.21, 0.25 도 다시 쓸 수 있다)."""
        self.assertEqual(live_demo.THRESHOLDS, (0.15, 0.16, 0.17))
        arguments = live_demo.parse_args(["--predictor", "unused"])
        self.assertEqual(arguments.thresholds, "0.15,0.16,0.17")
        parsed = live_demo.parse_thresholds(arguments.thresholds)
        self.assertEqual(parsed, (0.15, 0.16, 0.17))
        self.assertEqual(live_demo.LiveRules(parsed).primary, 0.15)
        self.assertEqual(live_demo.LiveRules(live_demo.parse_thresholds("0.21,0.25")).primary, 0.21)
        self.assertEqual(live_demo.LiveRules(live_demo.parse_thresholds("0.17,0.15")).primary, 0.17)
        self.assertEqual(live_demo.MAIN_THRESHOLD, live_demo.THRESHOLDS[0])
        self.assertEqual(live_demo.threshold_label(0.16), "0.16")
        self.assertEqual(live_demo.threshold_label(0.155), "0.155")

    def test_threshold_list_format_errors(self):
        """(b) 형식 오류(빈 값, 실수 아님, 범위 밖, 개수 초과, 중복)는 한 줄 메시지를 내고, main 은 한 줄을 출력하고 2 로 끝난다."""
        for text in ("", "   ", "abc", "0.15,,0.16", "0.15,", ",0.15", "0", "1", "1.2", "-0.1", "nan", "inf",
                     "0.1,0.2,0.3,0.4,0.5", "0.15,0.15", "0.15, 0.150"):
            with self.subTest(text=text):
                with self.assertRaises(ValueError) as caught:
                    live_demo.parse_thresholds(text)
                message = str(caught.exception)
                self.assertTrue(message.startswith("--thresholds"))
                self.assertNotIn("\n", message)
        self.assertEqual(live_demo.parse_thresholds("0.16"), (0.16,))
        self.assertEqual(live_demo.parse_thresholds(" 0.15 , 0.17 "), (0.15, 0.17))
        self.assertEqual(len(live_demo.parse_thresholds("0.1,0.2,0.3,0.4")), 4)
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = live_demo.main(["--predictor", "unused", "--thresholds", "0.15,0.15"])
        self.assertEqual(code, 2)
        lines = buffer.getvalue().splitlines()
        self.assertEqual(len(lines), 1)
        self.assertTrue(lines[0].startswith("--thresholds"))

    def test_threshold_list_does_not_change_alarm_times_and_matches_batch(self):
        """(c) 임계값 목록을 바꿔도(순서, 개수, 값) 같은 입력에서 임계값별 알람이 같고 일괄 처리(같은 상태기계)와 일치한다.
        EAR now / min 3s 표시 값(RecentMinimum)을 함께 계산해도 규칙은 그대로이다. 모두 메모리 안에서만 하고 파일을 만들지 않는다."""
        samples = synthetic_samples()
        reference = {}
        for thresholds in ((0.16, 0.17), (0.17, 0.16), (0.15, 0.16, 0.17), (0.16,), (0.21, 0.25), (0.15, 0.16, 0.17, 0.21)):
            rules = live_demo.LiveRules(thresholds)
            recent = live_demo.RecentMinimum()
            observed = []
            for timestamp_s, ear_mean, source in samples:
                if ear_mean is not None and source == "detected":
                    recent.add(timestamp_s, ear_mean)
                    observed.append((timestamp_s, ear_mean))
                    self.assertEqual(recent.minimum(timestamp_s), min(v for t, v in observed if timestamp_s - t <= 3.0))
                rules.update(timestamp_s, ear_mean, source)
            self.assertEqual(rules.primary, thresholds[0])
            for threshold in thresholds:
                alarms = list(rules.machines[threshold].alarms)
                batch = EM.run_samples(samples, threshold=threshold, valid_sources=live_demo.VALID_SOURCES)
                self.assertEqual(alarms, batch.alarms, "목록 %s, 임계값 %.2f" % (thresholds, threshold))
                self.assertEqual(reference.setdefault(threshold, alarms), alarms)
        window = live_demo.RecentMinimum()
        self.assertIsNone(window.minimum(10.0))
        window.add(0.0, 0.3)
        self.assertEqual(window.minimum(3.0), 0.3)
        self.assertIsNone(window.minimum(3.1))                              # 3.0초가 지나면 창에서 빠진다


if __name__ == "__main__":
    unittest.main()
