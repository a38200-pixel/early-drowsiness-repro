"""timeline 시험: 합성 데이터와 임시 폴더만 쓴다(영상, cv2, TensorFlow 불필요). 저장소 안 경로 시험은 저장소 위치를 original_hook.REPO 에서 얻는다."""
import contextlib
import copy
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

from improve.hmlstm import gauge as G
from improve.hmlstm import original_hook as OH
from improve.hmlstm import timeline as TL
from improve.hmlstm import windows as W

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


SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))


def fold_scores(count, offset=0.0):
    """폴드 5개의 점수 배열(창 count 개, 폴드마다 조금씩 다른 값)."""
    return dict((fold, np.linspace(2.0, 6.0, count) + 0.1 * fold + offset) for fold in range(1, 6))


def make_section_document(rows_count=40, fps=30.0):
    """행 rows_count 개(프레임 10, 20, ...)에서 창을 만든 시간선 구간(합성 점수)."""
    indices = [10 * (i + 1) for i in range(rows_count)]
    ranges = W.window_ranges(rows_count, False)
    return TL.build_timeline_section("clip.mp4", fps, 10 * rows_count + 50, indices, ranges, fold_scores(len(ranges)))


def make_timeline_document():
    base, target = make_section_document(60), make_section_document(40)
    return TL.build_timeline_document(1, base, target, TL.build_summary(base, reference_blinks=20, sample_blinks=40), TL.build_summary(target))


class WindowMappingTest(unittest.TestCase):
    def test_end_row_and_frame_index_follow_window_ranges(self):
        for rows_count in (30, 31, 40, 61):
            indices = [7 * (i + 1) for i in range(rows_count)]
            ranges = W.window_ranges(rows_count, False)
            section = TL.build_timeline_section("clip.mp4", 25.0, 7 * rows_count + 1, indices, ranges, fold_scores(len(ranges)))
            self.assertEqual(len(section["windows"]), (rows_count - 30) // 2 + 1)
            for position, window in enumerate(section["windows"]):
                self.assertEqual(window["k"], position + 1)
                self.assertEqual(window["end_row"], ranges[position][1])
                self.assertEqual(window["end_row"], 29 + 2 * position)
                self.assertEqual(window["frame_index"], indices[window["end_row"]])
                self.assertAlmostEqual(window["time_sec"], window["frame_index"] / 25.0)

    def test_fewer_than_30_rows_give_no_windows_and_an_empty_summary(self):
        section = TL.build_timeline_section("clip.mp4", 30.0, 400, [10 * (i + 1) for i in range(20)], W.window_ranges(20, False), fold_scores(0))
        self.assertEqual(section["windows"], [])
        summary = TL.build_summary(section)
        self.assertEqual((summary["window_count"], summary["fold_means"], summary["window_mean_min"]), (0, None, None))

    def test_mean_min_max_are_taken_over_the_five_folds_without_rounding(self):
        section = make_section_document(32)
        scores = fold_scores(2)
        for position, window in enumerate(section["windows"]):
            values = [scores[fold][position] for fold in range(1, 6)]
            self.assertEqual(window["folds"], [float(v) for v in values])
            self.assertAlmostEqual(window["mean"], float(np.mean(values)), places=12)
            self.assertEqual((window["min"], window["max"]), (float(min(values)), float(max(values))))

    def test_summary_fold_means_are_the_means_over_windows(self):
        section = make_section_document(40)
        summary = TL.build_summary(section, reference_blinks=3)
        self.assertEqual(summary["window_count"], 6)
        self.assertEqual(summary["reference_blinks"], 3)
        for fold in range(5):
            self.assertAlmostEqual(summary["fold_means"][fold], float(np.mean([w["folds"][fold] for w in section["windows"]])), places=12)
        self.assertEqual(summary["window_mean_min"], min(w["mean"] for w in section["windows"]))

    def test_the_gauge_sees_the_window_at_the_frame_the_row_was_confirmed(self):
        section = make_section_document(40)
        self.assertEqual(G.display_state(section, 299)["stage"], "collecting")
        self.assertEqual(G.display_state(section, 300)["window"], 1)


class TimelineDocumentTest(unittest.TestCase):
    def check(self, mutate, expected_text):
        document = copy.deepcopy(make_timeline_document())
        mutate(document)
        with self.assertRaises(TL.TimelineError) as caught:
            TL.validate_timeline_document(document)
        self.assertIn(expected_text, str(caught.exception))

    def test_valid_document_passes_and_round_trips(self):
        document = make_timeline_document()
        TL.validate_timeline_document(document)
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "a", "timeline.json")
            TL.write_json_document(path, document, TL.validate_timeline_document)
            self.assertEqual(TL.read_timeline_document(path), json.loads(json.dumps(document)))
            with self.assertRaises(TL.TimelineError):
                TL.write_json_document(path, document, TL.validate_timeline_document)             # 덮어쓰지 않는다
        with self.assertRaises(TL.TimelineError):
            TL.write_json_document(os.path.join(OH.REPO, "x_should_not_exist", "t.json"), document, TL.validate_timeline_document)

    def test_a_document_with_an_empty_window_list_is_valid(self):
        base = make_section_document(20)
        document = TL.build_timeline_document(1, base, base, TL.build_summary(base), TL.build_summary(base))
        TL.validate_timeline_document(document)

    def test_schema_errors(self):
        self.check(lambda d: d.pop("summary"), "필수 키가 없습니다: summary")
        self.check(lambda d: d.update(kind="rows"), "kind")
        self.check(lambda d: d.update(schema_version=3), "schema_version")
        self.check(lambda d: d.update(folds=[1, 2, 3]), "folds")
        self.check(lambda d: d["baseline"].pop("windows"), "baseline: 필수 키가 없습니다: windows")
        self.check(lambda d: d["target"].update(video_name="dir/clip.mp4"), "video_name")
        self.check(lambda d: d["target"]["windows"][2].update(k=9), "k 가 3 여야")
        self.check(lambda d: d["target"]["windows"][1].update(end_row=100), "end_row")
        self.check(lambda d: d["target"]["windows"][1].update(frame_index=1), "blink_frame_indices")
        self.check(lambda d: d["target"]["windows"][1].update(time_sec=99.0), "time_sec")
        self.check(lambda d: d["target"]["windows"][0]["folds"].pop(), "folds 는 숫자 5개")
        self.check(lambda d: d["target"]["windows"][0].update(mean=99.0), "min <= mean <= max")
        self.check(lambda d: d["baseline"]["blink_frame_indices"].reverse(), "blink_frame_indices 검사 실패")
        self.check(lambda d: d["summary"]["target"].update(window_count=1), "window_count")
        self.check(lambda d: d["summary"]["baseline"].update(fold_means=[1.0]), "fold_means")


class ScoreArgumentAndSafetyTest(unittest.TestCase):
    BASE = ["score", "--baseline", "b.json", "--target", "t.json", "--ckpt-root", "ck", "--out", "o.json"]

    def test_arguments(self):
        options = TL.parse_args(self.BASE)
        self.assertEqual((options.command, options.seed, options.stats), ("score", 1, None))
        self.assertEqual(TL.parse_args(self.BASE + ["--seed", "2"]).seed, 2)
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            TL.parse_args(["score", "--baseline", "b.json"])

    def run_score(self, baseline_rows, target_rows, out=None):
        with tempfile.TemporaryDirectory() as folder:
            paths = {}
            for name, rows in (("baseline", baseline_rows), ("target", target_rows)):
                paths[name] = os.path.join(folder, name + ".json")
                indices = [10 * (i + 1) for i in range(len(rows))]
                TL.write_rows_document(paths[name], TL.build_rows_document(name + ".mp4", 30.0, 10 * len(rows) + 10, "none", rows, indices))
            argv = ["score", "--baseline", paths["baseline"], "--target", paths["target"], "--ckpt-root", "unused",
                    "--out", out or os.path.join(folder, "timeline.json")]
            buffer = io.StringIO()
            with mock.patch.object(TL.SC, "EnsembleScorer") as scorer, contextlib.redirect_stdout(buffer):
                code = TL.main(argv)
            return code, buffer.getvalue(), scorer

    def test_a_degenerate_baseline_is_reported_before_any_scoring(self):
        constant = [[float(i + 1), 0.1, 0.2, 10.0, 0.01] for i in range(40)]
        code, text, scorer = self.run_score(constant, constant)
        self.assertEqual(code, 2)
        self.assertIn("기준 통계", text)
        scorer.assert_not_called()

    def test_output_inside_the_repository_is_refused_before_scoring(self):
        rows = [[float(i + 1), 0.1 + 0.001 * i, 0.2, 10.0 + i % 3, 0.01] for i in range(40)]
        code, text, scorer = self.run_score(rows, rows, out=os.path.join(OH.REPO, "x_should_not_exist.json"))
        self.assertEqual(code, 2)
        self.assertIn("저장소 폴더 안", text)
        scorer.assert_not_called()

    def test_a_missing_input_file_exits_with_code_two(self):
        buffer = io.StringIO()
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(buffer):
            code = TL.main(["score", "--baseline", os.path.join(folder, "no.json"), "--target", os.path.join(folder, "no2.json"),
                            "--ckpt-root", "ck", "--out", os.path.join(folder, "o.json")])
        self.assertEqual(code, 2)

    def test_importing_the_module_does_not_load_cv2_dlib_or_tensorflow(self):
        code = ("import sys; import improve.hmlstm.timeline; "
                "bad = [name for name in ('tensorflow', 'cv2', 'dlib') if name in sys.modules]; sys.exit(1 if bad else 0)")
        env = dict(os.environ, PYTHONPATH=SRC_ROOT, PYTHONDONTWRITEBYTECODE="1")
        self.assertEqual(subprocess.call([sys.executable, "-c", code], env=env), 0)


class ScoreWithCheckpointsTest(unittest.TestCase):
    def test_score_on_synthetic_rows_when_a_checkpoint_root_is_given(self):
        root = os.environ.get("HMLSTM_CKPT_ROOT")
        if not root:
            self.skipTest("환경변수 HMLSTM_CKPT_ROOT(<CKPT_DIR>)가 없습니다")
        try:
            import tensorflow  # noqa: F401
        except ImportError as error:
            self.skipTest("tensorflow 를 가져올 수 없습니다: %s" % error)
        rng = np.random.RandomState(0)

        def rows(count):
            return [[float(i + 1), float(0.1 + 0.02 * rng.rand()), float(0.1 + 0.05 * rng.rand()), float(8 + 6 * rng.rand()),
                     float(0.005 + 0.01 * rng.rand())] for i in range(count)]
        baseline, target = rows(100), rows(70)
        with tempfile.TemporaryDirectory() as folder:
            paths = {}
            for name, data in (("baseline", baseline), ("target", target)):
                paths[name] = os.path.join(folder, name + ".json")
                TL.write_rows_document(paths[name], TL.build_rows_document(name + ".mp4", 30.0, 10 * len(data) + 10, "none", data,
                                                                           [10 * (i + 1) for i in range(len(data))]))
            out = os.path.join(folder, "timeline.json")
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                code = TL.main(["score", "--baseline", paths["baseline"], "--target", paths["target"], "--ckpt-root", root, "--out", out])
            self.assertEqual(code, 0, buffer.getvalue())
            document = TL.read_timeline_document(out)
        self.assertEqual(len(document["baseline"]["windows"]), (100 - 100 // 3 - 30) // 2 + 1)         # 샘플 구간 앞 n-n//3 행
        self.assertEqual(len(document["target"]["windows"]), (70 - 30) // 2 + 1)
        self.assertEqual(document["summary"]["baseline"]["reference_blinks"], 100 // 3)
        for window in document["target"]["windows"]:
            self.assertTrue(0.0 <= window["min"] <= window["mean"] <= window["max"] <= 10.0)


if __name__ == "__main__":
    unittest.main()
