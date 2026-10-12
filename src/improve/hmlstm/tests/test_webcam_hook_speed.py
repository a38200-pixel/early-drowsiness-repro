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
