# ear_hook_compare 시험: 순수 함수와 가짜 소스만 쓴다(웹캠, 영상, 모델 파일, dlib 없음). 원본 소스는 텍스트와 ast 로 읽기만 하고 실행하지 않는다.
import ast
import contextlib
import io
import json
import os
import tempfile
import unittest

from improve.hmlstm import ear_hook_compare as EC
from improve.hmlstm import original_hook as OH
from improve.hmlstm import webcam_hook_speed as WH


def frames_from(ears):
    """EAR 목록(None 이면 얼굴 없음)을 replay_precompute 프레임 목록으로 만든다."""
    polygon = [[1, 1]] * 6
    return [{"face": True, "ear": value, "left": polygon, "right": polygon} if value is not None else {"face": False, "ear": None, "left": None, "right": None}
            for value in ears]


def samples_from(ears, tracked=()):
    """EAR 열에서 후킹 표본을 만든다. tracked 에 든 프레임 번호는 얼굴 없음 + 추적 EAR 로 한다. number_of_frames 는 얼굴 검출 프레임의 누적 개수이다."""
    samples, faces = [], 0
    for index, value in enumerate(ears):
        if index in tracked:
            samples.append(EC.Sample(False, 0.3, "tracked", faces))
        elif value is None:
            samples.append(EC.Sample(False, None, "none", faces))
        else:
            faces += 1
            samples.append(EC.Sample(True, value, "detected", faces))
    return samples


def closed_run_ears():
    return [0.30] * 20 + [0.10] * 120 + [None] * 3 + [0.30] * 20 + [0.12] * 110 + [0.30] * 10


class SourceTest(unittest.TestCase):
    def test_both_callback_replacements_pass_the_count_check_and_keep_the_line_count(self):
        original = OH.read_original_source()
        source = EC.build_source("video.mp4", "PRED.dat", "SVM.sav", "none", original)
        self.assertEqual(len(source.split("\n")), len(original.split("\n")))
        self.assertEqual(original.count(EC.DETECTED_TARGET), 1)
        self.assertEqual(original.count(EC.TRACKED_TARGET), 1)
        self.assertEqual(source.count("_hook_ear(ear, 'detected', number_of_frames)"), 1)
        self.assertEqual(source.count("_hook_ear(ear, 'tracked', number_of_frames)"), 1)
        lines = source.split("\n")
        detected_line = [line for line in lines if "'detected'" in line and "_hook_ear" in line][0]
        tracked_line = [line for line in lines if "'tracked'" in line and "_hook_ear" in line][0]
        self.assertEqual(len(detected_line) - len(detected_line.lstrip(" ")), 12)
        self.assertEqual(len(tracked_line) - len(tracked_line.lstrip(" ")), 16)
        self.assertTrue(detected_line.strip().startswith("ear = (leftEAR + rightEAR) / 2.0;"))

    def test_the_two_new_targets_do_not_overlap_each_other_or_the_existing_replacements(self):
        self.assertNotIn(EC.DETECTED_TARGET, EC.TRACKED_TARGET)
        self.assertNotIn(EC.TRACKED_TARGET, EC.DETECTED_TARGET)
        for label, old, _new, _expected in OH.build_replacements("video.mp4", "PRED.dat", "SVM.sav", "none", True):
            for target in (EC.DETECTED_TARGET, EC.TRACKED_TARGET, WH.TK_TARGET):
                self.assertFalse(old in target or target in old, label)

    def test_only_the_expected_lines_change(self):
        original = OH.read_original_source()
        base = OH.apply_replacements(original, OH.build_replacements("video.mp4", "PRED.dat", "SVM.sav", "none", True))
        mine = EC.build_source("video.mp4", "PRED.dat", "SVM.sav", "none", original)
        extra = sorted(set(OH.changed_line_numbers(original, mine)) - set(OH.changed_line_numbers(original, base)))
        self.assertIn(421, extra)
        self.assertIn(506, extra)
        self.assertEqual(len(extra), 3)                                    # Tk 숨김 줄, 421행, 506행 (영상 경로 줄은 문자열 경로라 기본 치환과 같다)

    def test_a_changed_original_is_refused(self):
        original = OH.read_original_source()
        for broken in (original.replace(EC.DETECTED_TARGET, "\n            ear = (leftEAR+rightEAR) / 2.0"), original.replace(EC.TRACKED_TARGET, "\n    x")):
            with self.assertRaises(OH.ReplacementError):
                EC.build_source("video.mp4", "PRED.dat", "SVM.sav", "none", broken)

    def test_rotation_option_is_passed_to_the_original_hook(self):
        source = EC.build_source("video.mp4", "PRED.dat", "SVM.sav", "ccw90")
        self.assertIn("cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)", source)

    def test_number_of_frames_and_ear_live_in_the_same_function_scope(self):
        original = OH.read_original_source()
        tree = ast.parse(original)
        found = {"ear": [], "number_of_frames": []}

        def visit(node, stack):
            if isinstance(node, ast.FunctionDef):
                stack = stack + [node.name]
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in found:
                        found[target.id].append((node.lineno, tuple(stack)))
            for child in ast.iter_child_nodes(node):
                visit(child, stack)
        visit(tree, [])
        ear_lines = [(line, stack) for line, stack in found["ear"] if line in (421, 506)]
        self.assertEqual([line for line, _stack in ear_lines], [421, 506])
        for line, stack in ear_lines:
            self.assertEqual(stack, ("blink_detector",), line)            # 중첩 함수 안이 아니라 blink_detector 의 본문이다
        counter_lines = dict(found["number_of_frames"])
        self.assertEqual(counter_lines[350], ("blink_detector",))          # 0 으로 만드는 곳
        self.assertEqual(counter_lines[380], ("blink_detector",))          # 1 올리는 곳(얼굴 검출 분기)
        self.assertLess(350, 421)
        self.assertLess(380, 421)


class EarTapTest(unittest.TestCase):
    def test_each_tick_makes_one_sample_including_frames_without_a_callback(self):
        tap = EC.EarTap(progress_every=0)
        tap.ear(0.31, "detected", 1)
        self.assertEqual(tap.tick(), OH.NO_KEY)
        self.assertEqual(tap.tick(), OH.NO_KEY)                            # 콜백이 없던 반복
        tap.ear(0.22, "tracked", 1)
        tap.tick()
        tap.ear(0.20, "detected", 2)
        tap.tick()
        self.assertEqual(tap.samples, [EC.Sample(True, 0.31, "detected", 1), EC.Sample(False, None, "none", 1),
                                       EC.Sample(False, 0.22, "tracked", 1), EC.Sample(True, 0.20, "detected", 2)])

    def test_callback_errors_are_swallowed_and_counted(self):
        tap = EC.EarTap(progress_every=0)
        tap.ear("not a number", "detected", 1)
        tap.ear(0.3, "detected", "x")
        tap.ear(None, "detected", 1)
        self.assertEqual(tap.errors, 3)
        self.assertEqual(tap.tick(), OH.NO_KEY)                            # 오류 뒤에도 tick 은 정상이다
        self.assertEqual(tap.samples, [EC.Sample(False, None, "none", 0)])

    def test_progress_is_reported_every_n_frames(self):
        lines = []
        tap = EC.EarTap(progress_every=3, out=lines.append)
        for _ in range(7):
            tap.tick()
        self.assertEqual(lines, ["frames=3", "frames=6"])

    def test_a_fake_source_run_collects_samples_and_never_raises_from_the_callback(self):
        source = "\n".join(["for number in range(5):",
                           "    if number == 1:",
                           "        _hook_ear('bad', 'detected', 1)",
                           "    elif number != 3:",
                           "        _hook_ear(0.3, 'detected', number)",
                           "    _hook_progress_tick()", ""])
        tap = EC.EarTap(progress_every=0)
        status, rows = EC.run_source(source, tap)
        self.assertEqual((status, rows), ("finished", []))
        self.assertEqual(len(tap.samples), 5)
        self.assertEqual(tap.errors, 1)
        self.assertEqual([sample.face for sample in tap.samples], [True, False, True, False, True])

    def test_a_source_error_is_reported_in_the_status(self):
        status, rows = EC.run_source("raise ValueError('boom')\n", EC.EarTap(progress_every=0))
        self.assertTrue(status.startswith("error: ValueError"))


class CompareColumnsTest(unittest.TestCase):
    def test_identical_columns(self):
        ears = closed_run_ears()
        stats = EC.compare_columns(samples_from(ears), frames_from(ears))
        self.assertTrue(stats["frames_equal"])
        self.assertEqual((stats["face_same_ratio"], stats["face_mismatch_frames"], stats["faces_mismatch_frames"]), (1.0, 0, 0))
        self.assertEqual((stats["ear_max"], stats["ear_mean"], stats["ear_over_1e6"], stats["ear_over_1e9"]), (0.0, 0.0, 0, 0))
        self.assertEqual(stats["hook_faces_final"], stats["rules_faces_final"])

    def test_ear_differences_are_counted_by_threshold(self):
        ears = [0.3] * 10
        samples = samples_from(ears)
        samples[2] = samples[2]._replace(ear=0.3 + 5e-10)
        samples[4] = samples[4]._replace(ear=0.3 + 5e-7)
        samples[6] = samples[6]._replace(ear=0.3 + 2e-3)
        stats = EC.compare_columns(samples, frames_from(ears))
        self.assertAlmostEqual(stats["ear_max"], 2e-3)
        self.assertEqual((stats["ear_over_1e9"], stats["ear_over_1e6"]), (2, 1))
        self.assertEqual(stats["ear_pairs"], 10)

    def test_face_flag_and_counter_mismatches_report_the_first_frame(self):
        ears = [0.3, 0.3, None, 0.3, 0.3, None, 0.3]
        samples = samples_from(ears)
        samples[3] = EC.Sample(False, None, "none", samples[2].faces)          # 얼굴 있음을 없음으로 본 프레임
        samples[5] = samples[5]._replace(faces=samples[5].faces + 2)           # 누적 개수만 다른 프레임
        stats = EC.compare_columns(samples, frames_from(ears))
        self.assertEqual((stats["face_mismatch_frames"], stats["first_face_mismatch"]), (1, 3))
        self.assertEqual(stats["first_faces_mismatch"], 3)                    # 3번 프레임부터 누적 개수가 어긋난다
        self.assertGreaterEqual(stats["faces_mismatch_frames"], 2)
        self.assertAlmostEqual(stats["face_same_ratio"], 6 / 7.0)

    def test_columns_of_different_lengths_compare_the_common_prefix(self):
        ears = [0.3] * 12
        stats = EC.compare_columns(samples_from(ears[:10]), frames_from(ears))
        self.assertFalse(stats["frames_equal"])
        self.assertEqual((stats["hook_frames"], stats["rules_frames"], stats["compared"]), (10, 12, 10))
        self.assertEqual(stats["face_same_ratio"], 1.0)
        empty = EC.compare_columns([], frames_from(ears))
        self.assertEqual((empty["compared"], empty["face_same_ratio"], empty["ear_max"]), (0, None, None))
        self.assertFalse(empty["frames_equal"])

    def test_tracked_frames_are_counted_and_do_not_count_as_a_face(self):
        ears = [0.3, None, None, 0.3]
        stats = EC.compare_columns(samples_from(ears, tracked=(1,)), frames_from(ears))
        self.assertEqual(stats["tracked_frames"], 1)
        self.assertEqual(stats["face_mismatch_frames"], 0)                    # 추적 프레임은 얼굴 없음으로 같다


class AlarmComparisonTest(unittest.TestCase):
    def test_the_same_columns_give_the_same_alarms(self):
        ears = closed_run_ears()
        fps = 30.0
        hook = EC.alarm_times(EC.hook_series(samples_from(ears)), fps)
        rules = EC.alarm_times(EC.rules_series(frames_from(ears)), fps)
        self.assertGreaterEqual(len(rules), 1)
        report = EC.compare_alarms(hook, rules)
        self.assertEqual((report["same_count"], report["same"], report["max_time_diff"]), (True, True, 0.0))

    def test_a_tracked_frame_counts_as_missing_exactly_like_no_face(self):
        ears = [0.3] * 5 + [None] * 4 + [0.3] * 5
        with_tracking = EC.alarm_times(EC.hook_series(samples_from(ears, tracked=(5, 6, 7, 8))), 30.0)
        without = EC.alarm_times(EC.hook_series(samples_from(ears)), 30.0)
        self.assertEqual(with_tracking, without)

    def test_a_column_that_differs_around_the_threshold_changes_the_alarms(self):
        ears = [0.30] * 20 + [0.1499] * 100 + [0.30] * 20
        shifted = [0.30] * 20 + [0.1501] * 100 + [0.30] * 20
        hook = EC.alarm_times(EC.hook_series(samples_from(shifted)), 30.0)
        rules = EC.alarm_times(EC.rules_series(frames_from(ears)), 30.0)
        report = EC.compare_alarms(hook, rules)
        self.assertEqual((report["hook_alarms"], report["rules_alarms"]), (0, 1))
        self.assertFalse(report["same"])
        self.assertIsNone(report["max_time_diff"])

    def test_compare_alarms_with_different_times(self):
        report = EC.compare_alarms([3.0, 9.0], [3.0, 9.5])
        self.assertTrue(report["same_count"])
        self.assertFalse(report["same"])
        self.assertAlmostEqual(report["max_time_diff"], 0.5)


class ReportTest(unittest.TestCase):
    def _lines(self, ears, change=None, rows_report=None):
        samples = samples_from(ears)
        if change:
            samples = change(samples)
        frames = frames_from(ears)
        columns = EC.compare_columns(samples, frames)
        alarms = EC.compare_alarms(EC.alarm_times(EC.hook_series(samples), 30.0), EC.alarm_times(EC.rules_series(frames), 30.0))
        return EC.format_report("finished", columns, alarms, 0.15, rows_report, None, 0), columns, alarms

    def test_identical_columns_give_a_match_and_the_closing_line(self):
        lines, columns, alarms = self._lines(closed_run_ears())
        self.assertEqual(EC.verdict(columns, alarms), "일치")
        self.assertIn("verdict: 일치", lines)
        self.assertEqual(lines[-1], EC.NOTE)
        self.assertTrue(lines[0].startswith("status: finished"))
        self.assertTrue(any(line.startswith("LiveRules alarms at threshold 0.15") for line in lines))

    def test_any_difference_gives_a_mismatch(self):
        ears = closed_run_ears()
        lines, columns, alarms = self._lines(ears, lambda samples: samples[:-1])
        self.assertIn("verdict: 불일치", lines)
        lines, columns, alarms = self._lines(ears, lambda samples: [s._replace(ear=s.ear + 1e-6) if s.face else s for s in samples])
        self.assertIn("verdict: 불일치", lines)

    def test_rows_line_is_added_only_when_a_rows_report_exists(self):
        rows_report = {"hook_rows": 5, "reference_rows": 5, "verdict": "일치", "rounded_mismatch_rows": 0, "max_abs_diff": [0.0, 0.0, 0.0, 0.0, 0.0]}
        lines, _c, _a = self._lines(closed_run_ears(), rows_report=rows_report)
        self.assertTrue(any(line.startswith("blink rows: hook 5 / file 5 ; verdict: 일치") for line in lines))
        lines, _c, _a = self._lines(closed_run_ears())
        self.assertFalse(any(line.startswith("blink rows") for line in lines))

    def test_empty_columns_do_not_crash_the_report(self):
        columns = EC.compare_columns([], [])
        alarms = EC.compare_alarms([], [])
        lines = EC.format_report("interrupted", columns, alarms, 0.15)
        self.assertEqual(lines[-1], EC.NOTE)
        self.assertIn("status: interrupted", lines)


class InputTest(unittest.TestCase):
    def _document(self, angle=270):
        frames = frames_from([0.3, None])
        return {"schema_version": 1, "kind": "replay_rules", "video_name": "a.mp4", "fps": 30.0, "frames_read": 2,
                "rotation": {"requested": "auto", "metadata_angle": angle, "applied_angle": angle, "double_rotation_guard": False},
                "frame_size": [10, 10], "preprocess": {"width": 450, "gamma": 1.5, "upsample": 0}, "frames": frames}

    def test_rotation_follows_the_rules_document_unless_given(self):
        document = self._document(270)
        self.assertEqual(EC.resolve_rotation("auto", document), ("ccw90", None))
        self.assertEqual(EC.resolve_rotation("ccw90", document), ("ccw90", None))
        option, note = EC.resolve_rotation("none", document)
        self.assertEqual(option, "none")
        self.assertIn("differs", note)
        self.assertEqual(EC.resolve_rotation("auto", self._document(0))[0], "none")

    def test_rules_document_loading(self):
        with tempfile.TemporaryDirectory() as folder:
            good = os.path.join(folder, "rules.json")
            with open(good, "w", encoding="utf-8") as handle:
                json.dump(self._document(), handle)
            self.assertEqual(EC.load_rules_document(good)["fps"], 30.0)
            bad = os.path.join(folder, "bad.json")
            with open(bad, "w", encoding="utf-8") as handle:
                handle.write("not json")
            with self.assertRaises(EC.CompareError):
                EC.load_rules_document(bad)
            broken = self._document()
            broken["frames"][0]["ear"] = None
            wrong = os.path.join(folder, "wrong.json")
            with open(wrong, "w", encoding="utf-8") as handle:
                json.dump(broken, handle)
            with self.assertRaises(EC.CompareError):
                EC.load_rules_document(wrong)

    def test_parse_args_and_threshold_check(self):
        arguments = EC.parse_args(["--video", "v.mp4", "--predictor", "p.dat", "--rules-json", "r.json"])
        self.assertEqual((arguments.rotate, arguments.threshold, arguments.rows_json), ("auto", 0.15, None))
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                EC.parse_args(["--video", "v.mp4"])
            with self.assertRaises(SystemExit):
                EC.parse_args(["--video", "v.mp4", "--predictor", "p", "--rules-json", "r", "--rotate", "45"])
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(EC.main(["--video", "v.mp4", "--predictor", "p.dat", "--rules-json", "r.json", "--threshold", "1.5"]), 2)

    def test_missing_files_give_a_clear_error_and_code_two(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = EC.main(["--video", "no_such_video.mp4", "--predictor", "no_such.dat", "--rules-json", "no_such.json"])
        self.assertEqual(code, 2)
        self.assertIn("ear_hook_compare:", out.getvalue())

    def test_no_image_saving_and_no_tensorflow(self):
        with open(EC.__file__, encoding="utf-8") as handle:
            source = handle.read()
        for token in ("imwrite", "imsave", "VideoWriter", "ImageGrab", "save_frame", "import tensorflow"):
            self.assertNotIn(token, source)


if __name__ == "__main__":
    unittest.main()
