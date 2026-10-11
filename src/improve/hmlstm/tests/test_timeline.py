"""timeline 시험: 합성 데이터와 임시 폴더만 쓴다(영상, cv2, TensorFlow 불필요). 저장소 안 경로 시험은 저장소 위치를 original_hook.REPO 에서 얻는다."""
import contextlib
import copy
import io
import json
import os
import tempfile
import unittest
from unittest import mock

from improve.hmlstm import original_hook as OH
from improve.hmlstm import timeline as TL

ROWS = [[1.0, 0.1057, 0.0822, 20.0, 0.0056], [2.0, 0.1957, 0.0792, 12.0, 0.0123], [3.0, 0.2, 0.05, 9.0, 0.01]]


def good_document():
    return TL.build_rows_document("clip.mp4", 29.998351920013185, 100, "ccw90", ROWS, [10, 10, 60])


class DocumentRoundTripTest(unittest.TestCase):
    def test_write_then_read_gives_the_same_values(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "nested", "deeper", "rows.json")             # 상위 폴더가 없으면 만든다
            document = good_document()
            TL.write_rows_document(path, document)
            self.assertEqual(TL.read_rows_document(path), document)

    def test_values_are_kept_without_rounding_and_the_document_has_numbers_and_a_bare_file_name_only(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "rows.json")
            rows = [[1.0, 0.123456789012345, 0.1, 20.0, 0.0056]]
            TL.write_rows_document(path, TL.build_rows_document("clip.mp4", 30.0, 10, "none", rows, [3]))
            with open(path, "r", encoding="utf-8") as handle:
                text = handle.read()
            self.assertEqual(json.loads(text)["rows"], rows)
            self.assertNotIn(os.sep, json.loads(text)["video_name"])
            self.assertEqual(set(json.loads(text)), set(TL.REQUIRED_KEYS))

    def test_empty_rows_are_a_valid_document(self):
        TL.validate_document(TL.build_rows_document("clip.mp4", 30.0, 5, "none", [], []))


class SchemaErrorTest(unittest.TestCase):
    def check(self, mutate, expected_text):
        document = copy.deepcopy(good_document())
        mutate(document)
        with self.assertRaises(TL.TimelineError) as caught:
            TL.validate_document(document)
        self.assertIn(expected_text, str(caught.exception))

    def test_missing_key(self):
        self.check(lambda d: d.pop("rows"), "필수 키가 없습니다: rows")

    def test_wrong_schema_version(self):
        self.check(lambda d: d.update(schema_version=2), "schema_version")

    def test_video_name_with_a_path(self):
        self.check(lambda d: d.update(video_name="folder/clip.mp4"), "video_name")
        self.check(lambda d: d.update(video_name="folder\\clip.mp4"), "video_name")

    def test_bad_fps_and_frames(self):
        self.check(lambda d: d.update(fps=0), "fps")
        self.check(lambda d: d.update(frames_read=-1), "frames_read")

    def test_unknown_rotation(self):
        self.check(lambda d: d.update(rotation="45"), "rotation")

    def test_wrong_columns(self):
        self.check(lambda d: d.update(columns=["a", "b", "c", "d", "e"]), "columns")

    def test_row_with_the_wrong_number_of_columns(self):
        self.check(lambda d: d["rows"][1].pop(), "rows[1] 의 열 수")

    def test_non_numeric_value(self):
        self.check(lambda d: d["rows"][2].__setitem__(3, "x"), "rows[2][3]")

    def test_length_mismatch_between_rows_and_frame_indices(self):
        self.check(lambda d: d["frame_indices"].pop(), "length mismatch")

    def test_decreasing_frame_indices(self):
        self.check(lambda d: d.update(frame_indices=[10, 60, 59]), "decreasing")

    def test_frame_index_out_of_range(self):
        self.check(lambda d: d.update(frame_indices=[10, 10, 100]), "out of range")
        self.check(lambda d: d.update(frame_indices=[-1, 10, 60]), "out of range")

    def test_frame_index_that_is_not_an_integer(self):
        self.check(lambda d: d["frame_indices"].__setitem__(0, 1.5), "frame_indices[0]")

    def test_not_json_and_not_an_object(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "bad.json")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("{not json")
            with self.assertRaises(TL.TimelineError) as caught:
                TL.read_rows_document(path)
            self.assertIn("JSON", str(caught.exception))
        with self.assertRaises(TL.TimelineError):
            TL.validate_document([1, 2])


class OutputPathTest(unittest.TestCase):
    def test_paths_inside_the_repository_are_detected(self):
        self.assertTrue(TL.is_inside(os.path.join(OH.REPO, "out", "rows.json"), OH.REPO))
        self.assertTrue(TL.is_inside(OH.REPO, OH.REPO))
        self.assertFalse(TL.is_inside(OH.REPO + "_other", OH.REPO))                  # 이름만 앞부분이 같은 형제 폴더는 안이 아니다
        with tempfile.TemporaryDirectory() as folder:
            self.assertFalse(TL.is_inside(os.path.join(folder, "rows.json"), OH.REPO))

    def test_writing_inside_the_repository_is_refused_and_nothing_is_created(self):
        target = os.path.join(OH.REPO, "timeline_test_output_should_not_exist", "rows.json")
        with self.assertRaises(TL.TimelineError) as caught:
            TL.write_rows_document(target, good_document())
        self.assertIn("저장소 폴더 안", str(caught.exception))
        self.assertFalse(os.path.exists(os.path.dirname(target)))

    def test_existing_file_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "rows.json")
            TL.write_rows_document(path, good_document())
            with open(path, "rb") as handle:
                before = handle.read()
            with self.assertRaises(TL.TimelineError) as caught:
                TL.write_rows_document(path, TL.build_rows_document("other.mp4", 30.0, 5, "none", [], []))
            self.assertIn("덮어쓰지 않습니다", str(caught.exception))
            with open(path, "rb") as handle:
                self.assertEqual(handle.read(), before)

    def test_invalid_documents_are_not_written(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "rows.json")
            bad = good_document()
            bad["frame_indices"] = [10, 9, 60]
            with self.assertRaises(TL.TimelineError):
                TL.write_rows_document(path, bad)
            self.assertFalse(os.path.exists(path))


class RotationTest(unittest.TestCase):
    def test_metadata_angles_map_to_the_hook_options(self):
        self.assertEqual([TL.angle_to_option(a) for a in (0, 90, 180, 270)], ["none", "cw90", "180", "ccw90"])
        with self.assertRaises(TL.TimelineError):
            TL.angle_to_option(45)

    def test_explicit_option_is_passed_through_and_unknown_is_rejected(self):
        self.assertEqual(TL.resolve_rotation("ccw90", "unused.mp4"), "ccw90")
        with self.assertRaises(TL.TimelineError):
            TL.resolve_rotation("45", "unused.mp4")

    def test_every_hook_rotation_option_is_a_rotation_in_documents(self):
        for option in OH.ROTATE_OPTIONS:
            TL.validate_document(TL.build_rows_document("clip.mp4", 30.0, 1, option, [], []))


class ArgumentTest(unittest.TestCase):
    BASE = ["extract", "--video", "v.mp4", "--svm", "m.sav", "--predictor", "p.dat", "--out", "o.json"]

    def test_defaults(self):
        options = TL.parse_args(self.BASE)
        self.assertEqual(options.command, "extract")
        self.assertEqual((options.rotate, options.progress_every, options.max_frames), ("none", 500, 0))

    def test_options_are_read(self):
        options = TL.parse_args(self.BASE + ["--rotate", "auto", "--progress-every", "0", "--max-frames", "300"])
        self.assertEqual((options.rotate, options.progress_every, options.max_frames), ("auto", 0, 300))
        self.assertEqual(TL.parse_args(self.BASE + ["--rotate", "ccw90"]).rotate, "ccw90")

    def test_missing_required_arguments_and_unknown_choices_exit(self):
        for argv in (["extract"], self.BASE + ["--rotate", "45"], []):
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                TL.parse_args(argv)

    def test_there_is_no_overwrite_option(self):
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            TL.parse_args(self.BASE + ["--force"])


class MainSafetyTest(unittest.TestCase):
    def run_main(self, argv):
        buffer = io.StringIO()
        with mock.patch.object(OH, "extract_rows") as extract, contextlib.redirect_stdout(buffer):
            code = TL.main(argv)
        return code, buffer.getvalue(), extract

    def test_output_inside_the_repository_is_refused_before_any_extraction(self):
        inside = os.path.join(OH.REPO, "should_not_be_written.json")
        code, text, extract = self.run_main(["extract", "--video", "v.mp4", "--svm", "m.sav", "--predictor", "p.dat", "--out", inside])
        self.assertEqual(code, 2)
        self.assertIn("저장소 폴더 안", text)
        extract.assert_not_called()

    def test_existing_output_is_refused_before_any_extraction(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "rows.json")
            with open(path, "w") as handle:
                handle.write("{}")
            code, text, extract = self.run_main(["extract", "--video", "v.mp4", "--svm", "m.sav", "--predictor", "p.dat", "--out", path])
        self.assertEqual(code, 2)
        self.assertIn("덮어쓰지 않습니다", text)
        extract.assert_not_called()

    def test_missing_input_files_are_reported_with_exit_code_two(self):
        with tempfile.TemporaryDirectory() as folder:
            code, text, extract = self.run_main(["extract", "--video", "v.mp4", "--svm", "m.sav", "--predictor", "p.dat",
                                                 "--out", os.path.join(folder, "rows.json")])
        self.assertEqual(code, 2)
        self.assertIn("--video", text)
        extract.assert_not_called()


if __name__ == "__main__":
    unittest.main()
