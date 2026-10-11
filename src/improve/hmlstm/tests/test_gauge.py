"""gauge 시험: 합성 시간선 구간만 쓴다(TensorFlow, cv2, PIL 불필요). 블링크는 프레임 10, 20, ... 400 에서 40개 확정되고 창은 6개다."""
import ast
import os
import subprocess
import sys
import unittest

from improve.hmlstm import gauge as G

SOURCE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "gauge.py")
SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))


def make_section(windows=True):
    indices = [10 * (i + 1) for i in range(40)]
    entries = []
    if windows:
        for k in range(1, 7):
            end = 29 + 2 * (k - 1)                                  # window_ranges(40) 의 끝 행 번호
            mean = 4.0 + 0.1 * k
            entries.append({"k": k, "end_row": end, "frame_index": indices[end], "time_sec": indices[end] / 30.0,
                            "mean": mean, "min": mean - 1.0, "max": mean + 1.0, "folds": [mean] * 5})
    return {"video_name": "clip.mp4", "fps": 30.0, "frames_read": 450, "blink_frame_indices": indices, "windows": entries}


class StateBoundaryTest(unittest.TestCase):
    def test_before_the_first_blink(self):
        state = G.display_state(make_section(), 0)
        self.assertEqual((state["stage"], state["blinks"], state["window"]), ("collecting", 0, None))
        self.assertIsNone(state["elapsed_sec"])

    def test_29_blinks_are_still_collecting_and_30_give_the_first_window(self):
        section = make_section()
        for frame, blinks in ((289, 28), (290, 29), (299, 29)):
            state = G.display_state(section, frame)
            self.assertEqual((state["stage"], state["blinks"]), ("collecting", blinks))
        state = G.display_state(section, 300)                          # 30번째 블링크 = 첫 창이 확정되는 프레임
        self.assertEqual((state["stage"], state["blinks"], state["window"]), ("value", 30, 1))
        self.assertEqual((state["updated_frame"], state["elapsed_sec"]), (300, 0.0))

    def test_a_new_window_replaces_the_old_one_exactly_when_it_is_confirmed(self):
        section = make_section()
        self.assertEqual(G.display_state(section, 319)["window"], 1)
        self.assertEqual(G.display_state(section, 320)["window"], 2)
        self.assertEqual(G.display_state(section, 399)["window"], 5)
        self.assertEqual(G.display_state(section, 400)["window"], 6)

    def test_elapsed_time_is_frames_since_the_update_divided_by_fps(self):
        state = G.display_state(make_section(), 319)
        self.assertAlmostEqual(state["elapsed_sec"], 19 / 30.0)
        self.assertEqual(state["updated_frame"], 300)

    def test_frames_after_the_video_end_keep_the_last_window_and_the_elapsed_time_grows(self):
        state = G.display_state(make_section(), 1000)
        self.assertEqual((state["stage"], state["blinks"], state["window"]), ("value", 40, 6))
        self.assertAlmostEqual(state["elapsed_sec"], 600 / 30.0)

    def test_value_fields_come_from_the_latest_window(self):
        state = G.display_state(make_section(), 340)
        self.assertEqual(state["window"], 3)
        self.assertAlmostEqual(state["mean"], 4.3)
        self.assertAlmostEqual(state["min"], 3.3)
        self.assertAlmostEqual(state["max"], 5.3)

    def test_the_caller_can_report_a_missing_baseline(self):
        state = G.display_state(make_section(), 1000, baseline_missing=True)
        self.assertEqual(state["stage"], "no_baseline")
        self.assertIsNone(state["mean"])

    def test_enough_blinks_but_no_window_stays_collecting(self):
        state = G.display_state(make_section(windows=False), 1000)
        self.assertEqual((state["stage"], state["blinks"]), ("collecting", 40))
        self.assertIn("수집 중 30/30 블링크", G.format_gauge_text(state))


class TextTest(unittest.TestCase):
    def test_collecting_text(self):
        text = G.format_gauge_text(G.display_state(make_section(), 290))
        self.assertEqual(text.split("\n"), [G.LABEL, "수집 중 29/30 블링크"])

    def test_value_text_has_one_decimal_with_rounding(self):
        state = {"stage": "value", "blinks": 40, "needed": 30, "window": 12, "mean": 6.64, "min": 4.76, "max": 8.04,
                 "updated_frame": 100, "elapsed_sec": 3.54}
        self.assertEqual(G.format_value(state), "6.6 [4.8-8.0]")
        self.assertEqual(G.format_gauge_text(state).split("\n"), [G.LABEL, "6.6 [4.8-8.0]", "창 12 · 갱신 후 3.5초"])

    def test_label_is_the_decided_wording_and_always_first(self):
        self.assertEqual(G.LABEL, "HM-LSTM score (0~10), 참고용, 알람 아님")
        for frame in (0, 300, 1000):
            self.assertEqual(G.format_gauge_lines(G.display_state(make_section(), frame))[0], G.LABEL)
        self.assertEqual(G.format_gauge_lines(G.display_state(make_section(), 0, baseline_missing=True)), [G.LABEL, "기준 없음"])

    def test_no_probability_wording_anywhere(self):
        for frame in (0, 300, 1000):
            self.assertNotIn("확률", G.format_gauge_text(G.display_state(make_section(), frame)))

    def test_scale_constants_have_no_interval_names(self):
        self.assertEqual((G.SCALE_MIN, G.SCALE_MAX, G.BOUNDARIES), (0.0, 10.0, (3.34, 6.68)))


class ImportsTest(unittest.TestCase):
    FORBIDDEN = {"tensorflow", "cv2", "PIL", "dlib"}

    def test_the_source_does_not_import_heavy_libraries(self):
        with open(SOURCE_PATH, "r", encoding="utf-8") as handle:
            tree = ast.parse(handle.read())
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        self.assertEqual(imported & self.FORBIDDEN, set())

    def test_importing_the_module_loads_none_of_them(self):
        code = ("import sys; import improve.hmlstm.gauge; "
                "bad = [name for name in ('tensorflow', 'cv2', 'PIL', 'dlib') if name in sys.modules]; "
                "sys.exit(1 if bad else 0)")
        env = dict(os.environ, PYTHONPATH=SRC_ROOT, PYTHONDONTWRITEBYTECODE="1")
        self.assertEqual(subprocess.call([sys.executable, "-c", code], env=env), 0)


if __name__ == "__main__":
    unittest.main()
