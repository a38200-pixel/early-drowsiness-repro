"""original_hook 시험: 실제 영상 없이 가짜 소스와 가짜 open 으로 확인한다. third_party 원본은 읽기만 한다."""
import contextlib
import io
import unittest

import numpy as np

from improve.hmlstm import original_hook as OH

ROW = [1.0, 0.1057, 0.0822, 20.0, 0.0056]


def fake_source(output=1, path=1, predictor=1, svm=1, draw=2, imshow=2, waitkey=2, resize=1):
    """치환 대상이 정해진 횟수만큼 나오는 가짜 소스. 줄 수는 항상 같은 구조다."""
    parts = [OH.OUTPUT_TARGET] * output + [OH.PATH_TARGET] * path + ["x = " + OH.PREDICTOR_TARGET] * predictor
    parts += ["y = " + OH.SVM_TARGET] * svm + [OH.DRAW_TARGET] * draw + [OH.IMSHOW_TARGET] * imshow
    parts += ["k = " + OH.WAITKEY_TARGET] * waitkey + ["    " + OH.RESIZE_TARGET] * resize
    return "\n".join(parts) + "\n"


class OpenOverrideTest(unittest.TestCase):
    def test_only_the_output_name_is_intercepted_and_other_files_are_delegated(self):
        calls = []

        def fake_open(file, mode="r", *args, **kwargs):
            calls.append((file, mode))
            return "real-handle"

        sink = OH.MemoryBlinkSink()
        override = OH.make_open_override(sink, real_open=fake_open)
        self.assertIs(override(OH.OUTPUT_SENTINEL, "ab"), sink)
        self.assertEqual(calls, [])
        self.assertEqual(override("model.sav", "rb"), "real-handle")               # SVM pickle 읽기와 같은 호출 형태
        self.assertEqual(override("other.txt", "ab"), "real-handle")               # 다른 이름의 append 도 위임한다
        self.assertEqual(calls, [("model.sav", "rb"), ("other.txt", "ab")])
        with self.assertRaises(ValueError):
            override(OH.OUTPUT_SENTINEL, "rb")                                      # 출력 이름을 읽기로 열 수는 없다

    def test_real_open_still_reads_a_real_file(self):
        sink = OH.MemoryBlinkSink()
        override = OH.make_open_override(sink)
        with override(OH.ORIGINAL_PATH, "rb") as handle:                            # 원본 소스를 읽기 전용으로 연다(pickle 읽기와 같은 'rb')
            head = handle.read(16)
        self.assertEqual(len(head), 16)
        self.assertEqual(sink.rows, [])


class SinkTest(unittest.TestCase):
    def test_bytes_and_str_writes_both_become_rows(self):
        sink = OH.MemoryBlinkSink()
        with sink as handle:
            handle.write(b"\n")
            handle.write(b"1.0000 0.1057 0.0822 20.0000 0.0056 ")
        with sink as handle:
            handle.write("\n")
            handle.write("2.0000 0.1957 0.0792 12.0000 0.0123 ")
        self.assertEqual(len(sink.rows), 2)
        np.testing.assert_allclose(sink.rows[0], ROW)
        np.testing.assert_allclose(sink.rows[1], [2.0, 0.1957, 0.0792, 12.0, 0.0123])
        with self.assertRaises(TypeError):
            sink.write(1.5)

    def test_numpy_savetxt_with_the_original_arguments(self):
        sink = OH.MemoryBlinkSink()
        with sink as handle:
            handle.write(b"\n")
            np.savetxt(handle, ROW, delimiter=", ", newline=" ", fmt="%.4f")        # 원본 464행과 같은 호출 형태
        self.assertEqual(sink.raw, b"\n1.0000 0.1057 0.0822 20.0000 0.0056 ")
        self.assertEqual(len(sink.rows), 1)
        np.testing.assert_allclose(sink.rows[0], ROW)

    def test_repeated_appends_accumulate_in_order(self):
        seen = []
        sink = OH.MemoryBlinkSink(on_row=lambda row: seen.append(float(row[0])))
        for index in range(4):
            with sink as handle:
                handle.write(b"\n")
                np.savetxt(handle, [index + 1, 0.5, 0.25, 12, 0.0056], delimiter=", ", newline=" ", fmt="%.4f")
        self.assertEqual([float(row[0]) for row in sink.rows], [1.0, 2.0, 3.0, 4.0])
        self.assertEqual(seen, [1.0, 2.0, 3.0, 4.0])
        self.assertEqual(sink.raw.count(b"\n"), 4)
        self.assertTrue(sink.raw.startswith(b"\n1.0000 0.5000"))

    def test_a_row_without_a_closing_with_block_is_finished_by_flush(self):
        sink = OH.MemoryBlinkSink()
        sink.write(b"\n")
        sink.write("3.0000 0.5000 0.2500 12.0000 0.0056 ")
        self.assertEqual(len(sink.rows), 0)
        sink.flush_pending()
        self.assertEqual(len(sink.rows), 1)


class ReplacementTest(unittest.TestCase):
    def replacements(self, **kwargs):
        return OH.build_replacements("VIDEO", "PREDICTOR", "SVM", **kwargs)

    def test_each_target_matches_exactly_the_expected_number_of_times(self):
        source = fake_source()
        replacements = self.replacements(rotate="ccw90")
        OH.check_counts(source, replacements)
        modified = OH.apply_replacements(source, replacements)
        self.assertEqual(len(modified.split("\n")), len(source.split("\n")))
        self.assertIn("output_file = %r" % OH.OUTPUT_SENTINEL, modified)
        self.assertIn("cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)", modified)
        self.assertNotIn(OH.WAITKEY_TARGET, modified)

    def test_zero_matches_and_too_many_matches_both_fail(self):
        for kwargs, label in ((dict(svm=0), "svm path"), (dict(output=2), "output file"), (dict(draw=1), "plot_frame.draw()"),
                              (dict(imshow=3), "cv2.imshow"), (dict(path=0), "input video")):
            with self.assertRaises(OH.ReplacementError) as caught:
                OH.check_counts(fake_source(**kwargs), self.replacements())
            self.assertIn(label, str(caught.exception))

    def test_waitkey_is_255_without_tick_and_a_call_with_tick(self):
        plain = OH.apply_replacements(fake_source(), self.replacements())
        ticked = OH.apply_replacements(fake_source(), self.replacements(tick=True))
        self.assertIn("k = 255", plain)
        self.assertNotIn(OH.TICK_NAME, plain)
        self.assertIn("k = %s()" % OH.TICK_NAME, ticked)

    def test_unknown_rotate_option_is_rejected(self):
        with self.assertRaises(ValueError):
            self.replacements(rotate="45")

    def test_real_original_source_matches_and_only_the_known_lines_change(self):
        source = OH.read_original_source()                                          # third_party 원본을 읽기만 한다
        for tick in (False, True):
            replacements = self.replacements(tick=tick)
            OH.check_counts(source, replacements)
            changed = OH.changed_line_numbers(source, OH.apply_replacements(source, replacements))
            self.assertEqual(changed, [313, 315, 474, 476, 480, 556, 558, 562, 579, 580])
        rotated = OH.changed_line_numbers(source, OH.apply_replacements(source, self.replacements(rotate="cw90")))
        self.assertEqual(rotated, [313, 315, 359, 474, 476, 480, 556, 558, 562, 579, 580])


class RunSourceTest(unittest.TestCase):
    def test_fake_source_writes_rows_through_open_override(self):
        source = "\n".join(["import numpy as np",
                            "output_file = %r" % OH.OUTPUT_SENTINEL,
                            "for i in range(3):",
                            "    with open(output_file, 'ab') as f_handle:",
                            "        f_handle.write(b'\\n')",
                            "        np.savetxt(f_handle, [i + 1, 0.5, 0.25, 12, 0.0056], delimiter=', ', newline=' ', fmt='%.4f')",
                            "print('noisy')", ""])
        sink = OH.MemoryBlinkSink()
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            OH.run_source(source, sink, quiet=True)
        self.assertEqual([float(row[0]) for row in sink.rows], [1.0, 2.0, 3.0])
        self.assertEqual(buffer.getvalue(), "")                                     # 원본의 print 는 무음
        sink = OH.MemoryBlinkSink()
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            OH.run_source(source, sink, quiet=False)
        self.assertIn("noisy", buffer.getvalue())

    def test_progress_tick_prints_every_n_frames_and_max_frames_stops_the_loop(self):
        source = ("for i in range(10):\n"
                  "    key = %s()\n"
                  "    if key != 255:\n"
                  "        break\n") % OH.TICK_NAME
        sink = OH.MemoryBlinkSink()
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            frames = OH.run_source(source, sink, progress_every=2, total_frames=10, max_frames=4)
        self.assertEqual(frames, 4)
        self.assertEqual(buffer.getvalue().strip().split("\n"), ["frames=2/10 rows=0", "frames=4/10 rows=0"])
        sink = OH.MemoryBlinkSink()
        self.assertEqual(OH.run_source(source, sink), 10)                            # 제한이 없으면 끝까지 돈다

    def test_extract_rows_checks_files_before_anything_runs(self):
        with self.assertRaises(FileNotFoundError):
            OH.extract_rows("missing.mp4", "missing.sav", "missing.dat")


if __name__ == "__main__":
    unittest.main()
