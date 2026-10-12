# webcam_hook_speed 시험: 합성 시각과 가짜 스트림만 쓴다(웹캠, 창, 모델 파일, TensorFlow 없음). 원본 소스는 텍스트로 읽기만 한다.
import unittest

from improve.hmlstm import original_hook as OH
from improve.hmlstm import webcam_hook_speed as W


class SummarizeTest(unittest.TestCase):
    def test_processing_statistics_and_budget_ratio(self):
        ticks = [0.0, 0.1, 0.2, 0.3, 0.4]
        reads = [0.01, 0.04, 0.05, 0.03, 0.02]                 # k번째 값은 k번째 tick 으로 끝나는 반복의 읽기 시간
        summary = W.summarize(ticks, [], reads)
        self.assertAlmostEqual(summary["fps"], 10.0)
        self.assertAlmostEqual(summary["read_median_ms"], 35.0)           # 첫 반복 제외: 40, 50, 30, 20
        self.assertAlmostEqual(summary["process_median_ms"], 65.0)        # 간격 100 ms 에서 읽기를 뺀 60, 50, 70, 80
        self.assertAlmostEqual(summary["process_p95_ms"], 78.5)
        self.assertAlmostEqual(summary["process_max_ms"], 80.0)
        self.assertAlmostEqual(summary["budget_ms"], 100.0)
        self.assertAlmostEqual(summary["process_median_ratio"], 0.65)
        self.assertAlmostEqual(summary["process_p95_ratio"], 0.785)
        self.assertFalse(summary["preview"])

    def test_without_reads_or_with_too_few_reads_the_new_values_are_none(self):
        ticks = [0.0, 0.1, 0.2]
        for reads in (None, [0.01, 0.02]):
            summary = W.summarize(ticks, [], reads)
            self.assertIsNone(summary["process_median_ms"])
            self.assertIsNone(summary["budget_ms"])
            self.assertAlmostEqual(summary["gap_median_ms"], 100.0)       # 기존 값은 그대로
        self.assertIsNone(W.summarize([0.0], [], [0.01])["process_median_ms"])
        self.assertEqual(W.summarize([], [], [])["frames"], 0)

    def test_processing_time_never_goes_below_zero(self):
        summary = W.summarize([0.0, 0.1, 0.2], [], [0.0, 0.5, 0.5])        # 읽기 시간이 간격보다 길면 0 으로 둔다
        self.assertEqual(summary["process_max_ms"], 0.0)

    def test_existing_values_are_unchanged(self):
        summary = W.summarize([0.0, 0.1, 0.2, 0.3], [0.05, 0.25])
        self.assertEqual((summary["frames"], summary["rows"]), (4, 2))
        self.assertAlmostEqual(summary["last_row_s"], 0.25)
        self.assertAlmostEqual(summary["span_s"], 0.3)


class FormatTest(unittest.TestCase):
    def test_lines_keep_the_old_order_and_add_the_new_ones_before_the_note(self):
        summary = W.summarize([0.0, 0.1, 0.2, 0.3, 0.4], [0.3], [0.01, 0.04, 0.05, 0.03, 0.02])
        lines = W.format_summary(summary, "finished")
        prefixes = ["status:", "frames:", "mean fps:", "frame gap median", "blink rows confirmed:", "last row at", "measured span"]
        for line, prefix in zip(lines[:7], prefixes):
            self.assertTrue(line.startswith(prefix), line)
        self.assertEqual(lines[-1], W.NOTE)
        text = "\n".join(lines)
        self.assertIn("read wait median (ms): 35.0", text)
        self.assertIn("processing median / p95 / max (ms): 65.0 / 78.5 / 80.0", text)
        self.assertIn("frame budget (ms): 100.0 ; processing / budget (median, p95): 0.65 / 0.79", text)
        self.assertEqual(lines[-2], "preview: off")

    def test_preview_flag_and_missing_values(self):
        summary = W.summarize([0.0], [])
        summary["preview"] = True
        lines = W.format_summary(summary, "interrupted")
        self.assertEqual(lines[-2], "preview: on")
        self.assertIn("processing median / p95 / max (ms): - / - / -", "\n".join(lines))
        self.assertTrue(lines[0].endswith("interrupted"))


class SourceTest(unittest.TestCase):
    def test_extra_replacements_pass_the_count_check_and_keep_the_line_count(self):
        original = OH.read_original_source()
        source = W.build_source(2, "PRED.dat", "SVM.sav", original)
        self.assertEqual(len(source.split("\n")), len(original.split("\n")))
        self.assertEqual(source.count("_hook_read(stream)"), 1)
        self.assertNotIn(W.READ_TARGET, source)
        self.assertEqual(source.count("top = tk.Tk(); top.withdraw()"), 1)
        self.assertEqual(source.count("path = 2"), 1)

    def test_no_conflict_with_the_original_hook_replacements(self):
        original = OH.read_original_source()
        base = OH.apply_replacements(original, OH.build_replacements("", "PRED.dat", "SVM.sav", "none", True))
        mine = W.build_source(0, "PRED.dat", "SVM.sav", original)
        base_changed = set(OH.changed_line_numbers(original, base))
        mine_changed = set(OH.changed_line_numbers(original, mine))
        self.assertTrue(base_changed.issubset(mine_changed))
        self.assertEqual(len(mine_changed - base_changed), 3)              # 읽기 줄, Tk 줄, 영상 경로 줄(웹캠 번호)만 더 바뀐다
        lines = mine.split("\n")
        extra = sorted(mine_changed - base_changed)
        self.assertEqual(sum("_hook_read" in lines[number - 1] for number in extra), 1)
        self.assertEqual(sum("withdraw" in lines[number - 1] for number in extra), 1)
        self.assertEqual(sum(lines[number - 1].startswith("path = 0") for number in extra), 1)
        for label, old, _new, _expected in OH.build_replacements("", "PRED.dat", "SVM.sav", "none", True):
            self.assertFalse(old in W.READ_TARGET or old in W.TK_TARGET or W.READ_TARGET in old or W.TK_TARGET in old, label)
        self.assertEqual(original.count(W.READ_TARGET), 1)
        self.assertEqual(original.count(W.TK_TARGET), 1)

    def test_a_changed_original_is_refused(self):
        original = OH.read_original_source().replace(W.READ_TARGET, "(grabbed, frame) = stream.read(  )")
        with self.assertRaises(OH.ReplacementError):
            W.build_source(0, "PRED.dat", "SVM.sav", original)


class MeterTest(unittest.TestCase):
    def _clock(self, step=0.01):
        state = {"now": 0.0}

        def clock():
            state["now"] += step
            return state["now"]
        return clock

    def test_read_wrapper_records_time_and_the_last_frame(self):
        class Stream(object):
            results = [(True, "frame-a"), (False, None)]

            def read(self):
                return self.results.pop(0)

        stream = Stream()
        meter = W.Meter(10.0, clock=self._clock())
        self.assertEqual(meter.read(stream), (True, "frame-a"))
        self.assertEqual(meter.last_frame, "frame-a")
        self.assertEqual(meter.read(stream), (False, None))
        self.assertIsNone(meter.last_frame)
        self.assertEqual(len(meter.reads), 2)
        self.assertAlmostEqual(meter.reads[0], 0.01)

    def test_preview_is_called_only_when_a_frame_exists(self):
        shown = []
        meter = W.Meter(10.0, clock=self._clock(), preview=shown.append)
        meter.tick()
        self.assertEqual(shown, [])
        meter.last_frame = "frame-b"
        meter.tick()
        self.assertEqual(shown, ["frame-b"])

    def test_tick_stops_after_the_duration(self):
        meter = W.Meter(0.05, clock=self._clock())
        codes = [meter.tick() for _ in range(8)]
        self.assertEqual(codes[0], OH.NO_KEY)
        self.assertEqual(codes[-1], 0)

    def test_measurement_over_a_fake_source_gives_read_and_processing_times(self):
        source = "\n".join([
            "class S(object):",
            "    def read(self):",
            "        return (True, 'f')",
            "stream = S()",
            "while True:",
            "    (grabbed, frame) = _hook_read(stream)",
            "    if _hook_progress_tick() != 255:",
            "        break",
            ""])
        meter = W.Meter(0.5, clock=self._clock())
        status = W.run_measurement(source, meter)
        self.assertEqual(status, "finished")
        summary = W.summarize(meter.ticks, meter.row_times, meter.reads)
        self.assertGreater(summary["frames"], 3)
        self.assertAlmostEqual(summary["read_median_ms"], 10.0)            # 시계가 호출마다 10 ms 가므로 읽기 한 번 = 10 ms
        self.assertAlmostEqual(summary["process_median_ms"], 20.0)         # 반복 한 번은 시계 호출 3번(읽기 2번, tick 1번) = 30 ms
        self.assertAlmostEqual(summary["process_median_ratio"], 20.0 / summary["budget_ms"])


class MailboxTest(unittest.TestCase):
    def test_put_overwrites_an_unread_item_and_counts_the_discarded_one(self):
        box = W.Mailbox()
        self.assertIsNone(box.take())
        box.put("a")
        box.put("b")
        box.put("c")
        self.assertEqual((box.put_count, box.discarded), (3, 2))
        self.assertEqual(box.take(), "c")                                  # 최신 한 장만 남는다
        self.assertIsNone(box.take())                                      # 가져가면 빈다
        box.put("d")
        self.assertEqual((box.put_count, box.discarded), (4, 2))           # 가져간 뒤의 put 은 버리는 것이 없다
        self.assertTrue(box.pending())
        box.take()
        self.assertFalse(box.pending())

    def test_it_holds_one_item_not_a_queue(self):
        box = W.Mailbox()
        for number in range(100):
            box.put(number)
        self.assertEqual(box.take(), 99)
        self.assertIsNone(box.take())


class DrawThreadTest(unittest.TestCase):
    def _run(self, hz=30.0, puts=60, pause=0.004):
        import threading
        import time
        calls = {"draw_threads": set(), "close_threads": set(), "frames": []}

        def factory():
            def draw(frame):
                calls["draw_threads"].add(threading.get_ident())
                calls["frames"].append(frame)

            def close():
                calls["close_threads"].add(threading.get_ident())
            return draw, close

        box = W.Mailbox()
        thread = W.DrawThread(box, hz, factory)
        thread.start()
        for number in range(puts):
            box.put(number)
            time.sleep(pause)
        deadline = time.time() + 2.0
        while thread.draws < 1 and time.time() < deadline:
            time.sleep(0.01)
        thread.stop()
        return box, thread, calls

    def test_drawing_and_closing_happen_only_in_the_draw_thread(self):
        import threading
        box, thread, calls = self._run()
        self.assertFalse(thread.is_alive())
        self.assertEqual(len(calls["draw_threads"]), 1)
        self.assertNotIn(threading.get_ident(), calls["draw_threads"])
        self.assertEqual(calls["close_threads"], calls["draw_threads"])
        self.assertIsNone(thread.error)

    def test_every_frame_is_drawn_or_overwritten_or_still_pending(self):
        box, thread, calls = self._run()
        self.assertGreaterEqual(thread.draws, 1)
        self.assertGreater(box.discarded, 0)                               # 넣는 속도(약 200/s)가 그리는 주기(30 Hz)보다 빠르다
        self.assertEqual(thread.draws, len(calls["frames"]))
        self.assertEqual(len(thread.draw_times), thread.draws)
        self.assertEqual(box.put_count, thread.draws + box.discarded + (1 if box.pending() else 0))
        self.assertEqual(calls["frames"], sorted(calls["frames"]))         # 최신 프레임만 오래된 것 뒤에 온다

    def test_an_error_in_the_factory_is_kept_and_the_thread_ends(self):
        def broken():
            raise RuntimeError("no font")
        thread = W.DrawThread(W.Mailbox(), 15.0, broken)
        thread.start()
        thread.join(2.0)
        self.assertFalse(thread.is_alive())
        self.assertIn("RuntimeError: no font", thread.error)
        self.assertEqual(thread.draws, 0)

    def test_stop_before_any_frame_ends_cleanly(self):
        thread = W.DrawThread(W.Mailbox(), 1.0, lambda: (lambda frame: None, lambda: None))
        thread.start()
        thread.stop()
        self.assertFalse(thread.is_alive())
        self.assertEqual(thread.draws, 0)


class PreviewModeSummaryTest(unittest.TestCase):
    def test_mode_priority(self):
        self.assertEqual(W.preview_mode(False, False), "off")
        self.assertEqual(W.preview_mode(True, False), "inline")
        self.assertEqual(W.preview_mode(False, True), "thread")
        self.assertEqual(W.preview_mode(True, True), "thread")             # --preview-thread 가 우선

    def test_summary_carries_mode_draws_discarded_and_draw_time(self):
        ticks = [0.0, 0.1, 0.2]
        summary = W.summarize(ticks, [], [0.01, 0.02, 0.03], "thread", 40, 90, [0.010, 0.020, 0.030])
        self.assertEqual((summary["preview_mode"], summary["preview_draws"], summary["preview_discarded"]), ("thread", 40, 90))
        self.assertTrue(summary["preview"])
        self.assertAlmostEqual(summary["draw_median_ms"], 20.0)
        self.assertAlmostEqual(summary["draw_p95_ms"], 29.0)
        self.assertAlmostEqual(summary["process_median_ms"], 75.0)         # 후킹 루프 처리 시간 계산은 기존 그대로
        off = W.summarize(ticks, [])
        self.assertEqual((off["preview_mode"], off["preview"], off["preview_draws"]), ("off", False, None))

    def test_summary_lines_for_each_mode_and_the_old_lines_stay(self):
        summary = W.summarize([0.0, 0.1, 0.2], [], [0.01, 0.02, 0.03], "thread", 40, 90, [0.010, 0.020, 0.030])
        lines = W.format_summary(summary, "finished")
        self.assertIn("preview mode: thread ; draws: 40 ; frames overwritten in mailbox: 90 ; draw time median / p95 (ms): 20.0 / 29.0", lines)
        self.assertEqual(lines[-2], "preview: on")
        self.assertEqual(lines[-1], W.NOTE)
        off_lines = W.format_summary(W.summarize([0.0, 0.1], []), "finished")
        self.assertIn("preview mode: off ; draws: - ; frames overwritten in mailbox: - ; draw time median / p95 (ms): - / -", off_lines)
        self.assertEqual(off_lines[-2], "preview: off")
        inline = W.format_summary(W.summarize([0.0, 0.1], [], None, "inline", 12), "finished")
        self.assertIn("preview mode: inline ; draws: 12 ; frames overwritten in mailbox: - ; draw time median / p95 (ms): - / -", inline)


class ThreadOptionTest(unittest.TestCase):
    def test_hz_range(self):
        for text, expected in (("1", 1.0), ("15", 15.0), ("30", 30.0), ("7.5", 7.5)):
            self.assertEqual(W.hz_value(text), expected)
        import argparse
        for text in ("0.5", "31", "0", "-3", "abc"):
            with self.assertRaises(argparse.ArgumentTypeError):
                W.hz_value(text)

    def test_parse_args_for_the_thread_options(self):
        import contextlib
        import io
        base = ["--predictor", "p.dat"]
        arguments = W.parse_args(base)
        self.assertEqual((arguments.preview, arguments.preview_thread, arguments.preview_hz), (False, False, 15.0))
        arguments = W.parse_args(base + ["--preview", "--preview-thread", "--preview-hz", "20"])
        self.assertEqual((arguments.preview, arguments.preview_thread, arguments.preview_hz), (True, True, 20.0))
        with contextlib.redirect_stderr(io.StringIO()):
            for bad in ("0", "31", "x"):
                with self.assertRaises(SystemExit):
                    W.parse_args(base + ["--preview-hz", bad])

    def test_fake_screen_uses_the_replay_viewer_composition(self):
        try:
            from improve.rules import replay_viewer as RV
        except ImportError as error:
            self.skipTest("replay_viewer 를 가져올 수 없어 건너뜁니다: %s" % error)
        for name in ("compose", "GaugeStrip", "TimeStrip", "ReplayEngine", "make_painter", "DEFAULT_VIDEO_HEIGHT", "DISPLAY_WIDTH", "GAUGE"):
            self.assertTrue(hasattr(RV, name), name)
        section = W.fake_gauge_section()
        before = RV.GAUGE.display_state(section, W.FAKE_GAUGE_START_FRAME - 1)
        after = RV.GAUGE.display_state(section, W.FAKE_GAUGE_START_FRAME + 2)
        self.assertEqual(before["stage"], RV.GAUGE.STAGE_COLLECTING)
        self.assertEqual(after["stage"], RV.GAUGE.STAGE_VALUE)
        self.assertEqual((after["mean"], after["min"], after["max"]), (5.0, 4.0, 6.0))
        self.assertGreater(after["elapsed_sec"], 0.0)

    def test_default_source_keeps_the_original_cleanup_and_thread_mode_neutralizes_only_that_call(self):
        original = OH.read_original_source()
        plain = W.build_source(0, "PRED.dat", "SVM.sav", original)
        threaded = W.build_source(0, "PRED.dat", "SVM.sav", original, thread_mode=True)
        self.assertEqual(original.count(W.DESTROY_TARGET), 1)
        self.assertEqual(plain.count(W.DESTROY_TARGET), 1)                 # 인라인과 off 모드는 원본 정리 줄을 그대로 둔다
        self.assertEqual(threaded.count(W.DESTROY_TARGET), 0)
        base = set(OH.changed_line_numbers(original, plain))
        more = set(OH.changed_line_numbers(original, threaded))
        self.assertEqual(len(base), 12)
        self.assertEqual(len(more - base), 1)
        self.assertEqual(len(threaded.split("\n")), len(original.split("\n")))
        line = threaded.split("\n")[sorted(more - base)[0] - 1]
        self.assertEqual(line.strip(), "None")


class MailboxTimeoutTest(unittest.TestCase):
    def test_take_with_a_timeout_returns_none_after_waiting_only_that_long(self):
        import time
        box = W.Mailbox()
        started = time.time()
        self.assertIsNone(box.take(0.05))
        elapsed = time.time() - started
        self.assertGreaterEqual(elapsed, 0.03)
        self.assertLess(elapsed, 1.0)
        started = time.time()
        self.assertIsNone(box.take())                                      # 기본은 기다리지 않는다
        self.assertLess(time.time() - started, 0.05)

    def test_take_returns_as_soon_as_an_item_arrives(self):
        import threading
        import time
        box = W.Mailbox()
        threading.Timer(0.05, box.put, args=("x",)).start()
        started = time.time()
        self.assertEqual(box.take(2.0), "x")
        self.assertLess(time.time() - started, 1.0)


class StopTest(unittest.TestCase):
    def test_a_draw_thread_that_never_returns_does_not_block_stop(self):
        import threading
        import time
        release = threading.Event()
        started = threading.Event()

        def factory():
            def draw(frame):
                started.set()
                release.wait(30)                                           # 끝나지 않는 그리기(창 호출이 멈춘 상황)
            return draw, lambda: None
        box = W.Mailbox()
        thread = W.DrawThread(box, 30.0, factory)
        thread.start()
        box.put("frame")
        self.assertTrue(started.wait(2.0))
        begun = time.time()
        stopped = thread.stop(timeout=0.2)
        self.assertFalse(stopped)
        self.assertLess(time.time() - begun, 1.0)                          # 기다리지 않고 돌아온다
        self.assertTrue(thread.daemon)
        release.set()
        thread.join(2.0)

    def test_the_thread_ends_within_one_period_after_the_stop_event(self):
        import time
        thread = W.DrawThread(W.Mailbox(), 10.0, lambda: (lambda frame: None, lambda: None))
        thread.start()
        time.sleep(0.05)
        begun = time.time()
        self.assertTrue(thread.stop(timeout=2.0))
        self.assertLess(time.time() - begun, 0.5)

    def test_close_runs_in_the_draw_thread_after_the_stop(self):
        import threading
        seen = {}

        def factory():
            return (lambda frame: None), (lambda: seen.setdefault("close", threading.get_ident()))
        thread = W.DrawThread(W.Mailbox(), 30.0, factory)
        thread.start()
        thread.stop()
        thread.join(2.0)
        self.assertIn("close", seen)
        self.assertNotEqual(seen["close"], threading.get_ident())


class WatchdogTest(unittest.TestCase):
    def test_due_only_after_the_duration_plus_grace_from_the_first_frame(self):
        self.assertFalse(W.watchdog_due(64.9, 0.0, 4.0, 60.0))             # 첫 프레임 4.0 + 60 + 5 = 69.0 이전
        self.assertFalse(W.watchdog_due(68.9, 0.0, 4.0, 60.0))
        self.assertTrue(W.watchdog_due(69.0, 0.0, 4.0, 60.0))
        self.assertTrue(W.watchdog_due(100.0, 0.0, 4.0, 60.0))

    def test_before_the_first_frame_the_startup_allowance_applies(self):
        self.assertFalse(W.watchdog_due(59.9, 0.0, None, 60.0))            # 모델 로드와 카메라 열기 시간
        self.assertTrue(W.watchdog_due(60.0, 0.0, None, 60.0))
        self.assertFalse(W.watchdog_due(20.0, 0.0, None, 5.0))             # 짧은 측정 시간이어도 시작 허용 시간은 그대로

    def test_check_fires_once_and_not_after_cancel(self):
        fired = []
        state = {"due": False}
        watchdog = W.Watchdog(lambda: state["due"], lambda: fired.append(1))
        self.assertFalse(watchdog.check())
        state["due"] = True
        self.assertTrue(watchdog.check())
        self.assertFalse(watchdog.check())
        self.assertEqual(fired, [1])
        other = W.Watchdog(lambda: True, lambda: fired.append(2))
        other.cancel()
        self.assertFalse(other.check())
        self.assertEqual(fired, [1])

    def test_the_polling_thread_fires_when_due_and_stays_quiet_when_cancelled(self):
        import threading
        import time
        fired = threading.Event()
        watchdog = W.Watchdog(lambda: True, fired.set, poll_s=0.01)
        watchdog.start()
        self.assertTrue(fired.wait(2.0))
        quiet = threading.Event()
        cancelled = W.Watchdog(lambda: True, quiet.set, poll_s=0.01)
        cancelled.cancel()
        cancelled.start()
        time.sleep(0.1)
        self.assertFalse(quiet.is_set())

    def test_the_watchdog_summary_matches_the_normal_summary_plus_one_line(self):
        meter = W.Meter(60.0)
        meter.ticks.extend([0.0, 0.1, 0.2, 0.3])
        meter.reads.extend([0.01, 0.02, 0.02, 0.03])
        meter.row_times.append(0.25)
        normal, forced = [], []
        first = W.SummaryReporter(meter, "off", out=normal.append, flush=lambda: None)
        second = W.SummaryReporter(meter, "off", out=forced.append, flush=lambda: None)
        self.assertIsNotNone(first.emit("finished"))
        self.assertEqual(second.fire_watchdog(), 1)                        # 시간 안에 끝나지 않았으므로 종료 코드 1
        self.assertEqual(forced[0], "status: watchdog")
        extra = [line for line in forced[1:] if line.startswith("watchdog exit")]
        self.assertEqual(len(extra), 1)
        self.assertEqual(forced[-1], W.NOTE)
        self.assertEqual([line for line in forced[1:] if not line.startswith("watchdog exit")], normal[1:])

    def test_the_summary_is_printed_once_and_a_late_watchdog_only_adds_a_line(self):
        meter = W.Meter(60.0)
        meter.ticks.extend([0.0, 0.1])
        meter.reads.extend([0.01, 0.01])
        lines = []
        reporter = W.SummaryReporter(meter, "off", out=lines.append, flush=lambda: None)
        reporter.emit("finished")
        count = len(lines)
        self.assertIsNone(reporter.emit("finished"))
        self.assertEqual(len(lines), count)
        self.assertEqual(reporter.exit_code, 0)
        self.assertEqual(reporter.fire_watchdog(), 0)                      # 이미 낸 요약의 종료 코드를 유지
        self.assertEqual(len(lines), count + 1)
        self.assertTrue(lines[-1].startswith("watchdog exit"))


class MainOrderTest(unittest.TestCase):
    def _run_main(self, extra_args):
        import contextlib
        import io
        import os
        import tempfile
        from unittest import mock
        events = []
        watchdogs = []
        builds = []

        class Recorder(io.StringIO):
            def write(self, text):
                if text.startswith("status:"):
                    events.append("summary")
                return super(Recorder, self).write(text)

        class SpyWatchdog(W.Watchdog):
            def __init__(self, *args, **kwargs):
                super(SpyWatchdog, self).__init__(*args, **kwargs)
                watchdogs.append(self)

        def fake_run(source, meter):
            meter.ticks.extend([0.0, 0.1, 0.2])
            meter.reads.extend([0.01, 0.01, 0.01])
            return "finished"

        original_stop = W.DrawThread.stop

        def spy_stop(thread, timeout=W.STOP_JOIN_TIMEOUT_S):
            events.append("stop")
            return original_stop(thread, timeout)

        def fake_build(camera, predictor, svm, original=None, thread_mode=False):
            builds.append(thread_mode)
            return ""

        def factory():
            return (lambda frame: None), (lambda: None)

        with tempfile.NamedTemporaryFile(suffix=".dat") as predictor:
            argv = ["0", "1", "--predictor", predictor.name] + extra_args
            out = Recorder()
            with mock.patch.object(W, "run_measurement", fake_run), mock.patch.object(W, "find_svm", lambda: "svm"), \
                    mock.patch.object(W, "build_source", fake_build), mock.patch.object(W, "make_fake_screen_drawer", factory), \
                    mock.patch.object(W.DrawThread, "stop", spy_stop), mock.patch.object(W, "Watchdog", SpyWatchdog), \
                    mock.patch.object(W, "make_preview", lambda: (lambda frame: None)), contextlib.redirect_stdout(out):
                code = W.main(argv)
        return code, events, watchdogs, builds, out.getvalue()

    def test_thread_mode_prints_the_summary_before_stopping_the_thread(self):
        code, events, watchdogs, builds, text = self._run_main(["--preview-thread"])
        self.assertEqual(code, 0)
        self.assertEqual(builds, [True])
        self.assertLess(events.index("summary"), events.index("stop"))
        self.assertIn("preview mode: thread", text)
        self.assertEqual(len(watchdogs), 1)
        self.assertTrue(watchdogs[0].cancel_event.is_set())                # 정상 종료에서는 취소된다
        self.assertFalse(watchdogs[0].fired)

    def test_off_mode_keeps_the_original_cleanup_and_the_exit_code(self):
        code, events, watchdogs, builds, text = self._run_main([])
        self.assertEqual(code, 0)
        self.assertEqual(builds, [False])
        self.assertEqual(events, ["summary"])                              # 스레드가 없다
        self.assertIn("preview: off", text)
        self.assertTrue(watchdogs[0].cancel_event.is_set())


class PreviewAndArgumentTest(unittest.TestCase):
    def test_preview_size(self):
        self.assertEqual(W.preview_size(640, 480), (450, 338))
        self.assertEqual(W.preview_size(300, 200), (300, 200))
        self.assertEqual(W.preview_size(450, 450), (450, 450))
        self.assertEqual(W.preview_size(1080, 1920), (450, 800))
        self.assertEqual(W.preview_size(10000, 1), (450, 1))

    def test_parse_args(self):
        base = ["--predictor", "p.dat"]
        arguments = W.parse_args(base)
        self.assertEqual((arguments.camera, arguments.seconds, arguments.preview), (0, 60.0, False))
        arguments = W.parse_args(["1", "30"] + base + ["--preview"])
        self.assertEqual((arguments.camera, arguments.seconds, arguments.preview), (1, 30.0, True))

    def test_no_image_saving_calls(self):
        with open(W.__file__, encoding="utf-8") as handle:
            source = handle.read()
        for token in ("imwrite", "imsave", "VideoWriter", "ImageGrab", "save_frame", "savetxt"):
            self.assertNotIn(token, source)


if __name__ == "__main__":
    unittest.main()
