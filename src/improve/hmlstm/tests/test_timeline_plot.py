"""timeline_plot 시험: 합성 시간선과 임시 폴더만 쓴다(영상, TensorFlow, cv2 불필요). 그림을 그리는 시험은 끝에서 끝까지 도는 스모크 하나뿐이다."""
import contextlib
import copy
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest

from improve.hmlstm import timeline as TL
from improve.hmlstm import timeline_plot as TP
from improve.hmlstm import windows as W

SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
EVENTS = [{"id": "B1", "group": "closure", "start_s": 10.0, "end_s": 12.0},
          {"id": "D1", "group": "down", "start_s": 20.0, "end_s": 22.0},
          {"id": "E1", "group": "frown", "start_s": 30.0, "end_s": 31.0}]


def window(k, mean, time_sec=None, end_row=None):
    end = 29 + 2 * (k - 1) if end_row is None else end_row
    return {"k": k, "end_row": end, "frame_index": 0, "time_sec": float(k) if time_sec is None else time_sec,
            "mean": mean, "min": mean - 1, "max": mean + 1, "folds": [mean] * 5}


def make_section(count_rows, frames_read, fps=30.0):
    indices = [10 * (i + 1) for i in range(count_rows)]
    ranges = W.window_ranges(count_rows, False)
    scores = dict((fold, [2.0 + 0.1 * fold + 0.2 * j for j in range(len(ranges))]) for fold in range(1, 6))
    return TL.build_timeline_section("clip.mp4", fps, frames_read, indices, ranges, scores), indices


def make_rows(count, dur=None):
    return [[float(i + 1), 0.1, 0.05 + 0.001 * i, (dur[i] if dur else 8.0 + (i % 5)), 0.01 + 0.0001 * i] for i in range(count)]


class OverlapTest(unittest.TestCase):
    def test_closed_interval_boundaries(self):
        self.assertTrue(TP.overlaps(0, 10, 10, 20))                     # 끝점만 닿아도 겹침
        self.assertTrue(TP.overlaps(10, 20, 0, 10))
        self.assertFalse(TP.overlaps(0, 9.999, 10, 20))
        self.assertTrue(TP.overlaps(0, 100, 40, 50))                    # 포함
        self.assertTrue(TP.overlaps(5, 5, 5, 5))

    def test_counts_per_kind(self):
        span = {"start_s": 9.0, "end_s": 21.0}
        self.assertEqual(TP.overlap_counts(span, EVENTS), {"closure": 1, "down": 1, "frown": 0})
        self.assertEqual(TP.overlap_counts({"start_s": 40.0, "end_s": 50.0}, EVENTS), {"closure": 0, "down": 0, "frown": 0})
        self.assertEqual(TP.overlap_counts({"start_s": 12.0, "end_s": 20.0}, EVENTS), {"closure": 1, "down": 1, "frown": 0})

    def test_window_span_uses_the_first_and_last_row_of_the_window(self):
        indices = [10 * (i + 1) for i in range(40)]
        spans = TP.window_spans([window(1, 3.0), window(6, 3.0, end_row=39)], indices, 10.0)
        self.assertEqual((spans[0]["start_row"], spans[0]["end_row"]), (0, 29))
        self.assertEqual((spans[0]["start_s"], spans[0]["end_s"]), (1.0, 30.0))
        self.assertEqual((spans[1]["start_row"], spans[1]["end_row"]), (10, 39))
        self.assertEqual((spans[1]["start_s"], spans[1]["end_s"]), (11.0, 40.0))


class StatisticsTest(unittest.TestCase):
    def test_group_statistics(self):
        windows = [window(1, 2.0), window(2, 4.0), window(3, 6.0), window(4, 8.0)]
        counts = [{"closure": 1, "down": 0, "frown": 0}, {"closure": 2, "down": 0, "frown": 1},
                  {"closure": 0, "down": 1, "frown": 0}, {"closure": 0, "down": 0, "frown": 0}]
        stats = dict((row[0], row[1:]) for row in TP.group_stats(windows, counts))
        self.assertEqual(stats["overlaps an eye-closing event"], (2, 3.0, 2.0, 4.0))
        self.assertEqual(stats["no eye-closing event overlap"], (2, 7.0, 6.0, 8.0))
        self.assertEqual(stats["overlaps looking-down or frowning"], (2, 5.0, 4.0, 6.0))
        self.assertEqual(stats["no event overlap at all"], (1, 8.0, 8.0, 8.0))

    def test_empty_group_has_no_statistics(self):
        stats = dict((row[0], row[1:]) for row in TP.group_stats([window(1, 2.0)], [{"closure": 0, "down": 0, "frown": 0}]))
        self.assertEqual(stats["overlaps an eye-closing event"], (0, None, None, None))

    def test_adjacent_changes_ranking(self):
        means = [5.0, 3.0, 4.0, 1.0, 1.5, 6.5, 6.0]
        windows = [window(i + 1, m) for i, m in enumerate(means)]
        down, up = TP.adjacent_changes(windows)
        self.assertEqual([(k, round(d, 3)) for k, _t, d in down], [(4, -3.0), (2, -2.0), (7, -0.5)])
        self.assertEqual([(k, round(d, 3)) for k, _t, d in up], [(6, 5.0), (3, 1.0), (5, 0.5)])

    def test_ties_go_to_the_smaller_window_number_and_zero_change_is_neither(self):
        windows = [window(1, 5.0), window(2, 4.0), window(3, 4.0), window(4, 3.0)]
        down, up = TP.adjacent_changes(windows)
        self.assertEqual([k for k, _t, _d in down], [2, 4])
        self.assertEqual(up, [])

    def test_describe(self):
        self.assertEqual(TP.describe([1.0, 3.0]), (2, 2.0, 1.0, 3.0))
        self.assertEqual(TP.describe([]), (0, None, None, None))


class DurRowsTest(unittest.TestCase):
    def test_sorted_by_dur_with_ties_to_the_smaller_row_and_event_membership(self):
        dur = [8.0, 30.0, 8.0, 30.0, 12.0, 9.0]
        rows = make_rows(6, dur)
        indices = [60, 150, 300, 450, 600, 900]               # fps 30 -> 2, 5, 10, 15, 20, 30 초
        top = TP.top_dur_rows(rows, indices, 30.0, EVENTS, 4)
        self.assertEqual([r["row"] for r in top], [1, 3, 4, 5])
        self.assertEqual([r["event"] for r in top], ["-", "-", "D1", "E1"])             # 행 5 는 30초 = E1 의 시작 경계
        self.assertEqual(top[0]["frame"], 150)
        self.assertAlmostEqual(top[0]["second"], 5.0)
        self.assertEqual(TP.event_at(10.0, EVENTS), "B1")                 # 경계 포함
        self.assertEqual(TP.event_at(12.0, EVENTS), "B1")
        self.assertEqual(TP.event_at(12.001, EVENTS), "-")

    def test_row_limit_restricts_the_candidates(self):
        rows = make_rows(6, [8.0, 9.0, 10.0, 99.0, 98.0, 97.0])
        top = TP.top_dur_rows(rows, [10 * (i + 1) for i in range(6)], 30.0, [], 2, row_limit=3)
        self.assertEqual([r["row"] for r in top], [2, 1])

    def test_amp_and_vel_come_from_the_same_row(self):
        rows = make_rows(3, [1.0, 5.0, 2.0])
        top = TP.top_dur_rows(rows, [30, 60, 90], 30.0, [], 1)
        self.assertEqual((top[0]["amp"], top[0]["vel"]), (rows[1][2], rows[1][4]))


class VerificationTest(unittest.TestCase):
    def setUp(self):
        self.section, indices = make_section(60, 700)
        self.rows_document = TL.build_rows_document("clip.mp4", 30.0, 700, "none", make_rows(60), indices)

    def results(self, section, rows_document):
        return dict((name, ok) for name, ok, _detail in TP.verify_section(section, rows_document))

    def test_a_consistent_section_passes_every_check(self):
        self.assertTrue(all(self.results(self.section, self.rows_document).values()))
        self.assertEqual(len(TP.verify_section(self.section, self.rows_document)), 5)
        self.assertEqual(len(TP.verify_section(self.section, None)), 2)

    def test_decreasing_time_is_detected(self):
        section = copy.deepcopy(self.section)
        section["windows"][3]["time_sec"] = 0.1
        self.assertFalse(self.results(section, None)["time_sec is non-decreasing"])

    def test_a_time_beyond_the_video_length_is_detected(self):
        section = copy.deepcopy(self.section)
        section["windows"][-1]["time_sec"] = 700 / 30.0 + 1.0
        self.assertFalse(self.results(section, None)["last time_sec within video length"])

    def test_end_row_and_frame_index_mismatch_is_detected(self):
        document = copy.deepcopy(self.rows_document)
        document["frame_indices"][self.section["windows"][2]["end_row"]] += 1
        results = self.results(self.section, document)
        self.assertFalse(results["window end_row/frame_index match the rows file"])
        self.assertFalse(results["blink_frame_indices equal the rows file"])

    def test_metadata_mismatch_is_detected(self):
        document = copy.deepcopy(self.rows_document)
        document["fps"] = 25.0
        self.assertFalse(self.results(self.section, document)["fps and frames_read equal the rows file"])


class TruthTest(unittest.TestCase):
    def test_kinds_are_grouped(self):
        document = {"events": [{"id": "B1", "interval": "B", "kind": "눈 감기", "start_s": 31, "end_s": 37},
                               {"id": "D1", "interval": "D", "kind": "동작", "start_s": 72, "end_s": 77},
                               {"id": "E1", "interval": "E", "kind": "동작", "start_s": 90, "end_s": 96}]}
        self.assertEqual([(e["id"], e["group"]) for e in TP.parse_truth(document)], [("B1", "closure"), ("D1", "down"), ("E1", "frown")])

    def test_unknown_kind_and_missing_events_are_errors(self):
        with self.assertRaises(TL.TimelineError):
            TP.parse_truth({"events": [{"id": "X1", "interval": "X", "kind": "동작", "start_s": 1, "end_s": 2}]})
        with self.assertRaises(TL.TimelineError):
            TP.parse_truth({})


class TableFormatTest(unittest.TestCase):
    def test_table_alignment_and_two_decimals(self):
        lines = TP.format_table(["k", "mean"], [(1, TP.f2(3.14159)), (12, TP.f2(None))])
        self.assertEqual(lines, ["k   mean", " 1  3.14", "12     -"])

    def test_change_table_has_a_sign(self):
        self.assertEqual(TP.change_table([(4, 10.0, -3.0)])[1].split(), ["3", "->", "4", "10.00", "-3.00"])


class EndToEndSmokeTest(unittest.TestCase):
    def build_inputs(self, folder):
        target, target_indices = make_section(60, 700)
        baseline, baseline_indices = make_section(50, 600)
        document = TL.build_timeline_document(1, baseline, target, TL.build_summary(baseline, reference_blinks=16, sample_blinks=34),
                                              TL.build_summary(target))
        paths = {"timeline": os.path.join(folder, "timeline.json"), "target": os.path.join(folder, "target_rows.json"),
                 "baseline": os.path.join(folder, "baseline_rows.json"), "truth": os.path.join(folder, "truth.json")}
        TL.write_json_document(paths["timeline"], document, TL.validate_timeline_document)
        TL.write_rows_document(paths["target"], TL.build_rows_document("clip.mp4", 30.0, 700, "none", make_rows(60), target_indices))
        TL.write_rows_document(paths["baseline"], TL.build_rows_document("clip.mp4", 30.0, 600, "none", make_rows(50), baseline_indices))
        with open(paths["truth"], "w", encoding="utf-8") as handle:
            json.dump({"events": [{"id": "B1", "interval": "B", "kind": "눈 감기", "start_s": 10, "end_s": 12},
                                  {"id": "D1", "interval": "D", "kind": "동작", "start_s": 14, "end_s": 16},
                                  {"id": "E1", "interval": "E", "kind": "동작", "start_s": 18, "end_s": 20}]}, handle)
        return paths

    def test_figures_are_written_once_and_never_overwritten(self):
        try:
            import matplotlib  # noqa: F401
        except ImportError as error:
            self.skipTest("matplotlib 를 가져올 수 없습니다: %s" % error)
        with tempfile.TemporaryDirectory() as folder:
            paths = self.build_inputs(folder)
            out, base_out = os.path.join(folder, "figs", "target.png"), os.path.join(folder, "figs", "baseline.png")
            argv = ["--timeline", paths["timeline"], "--target-rows", paths["target"], "--truth", paths["truth"], "--out-png", out,
                    "--out-baseline-png", base_out, "--baseline-rows", paths["baseline"]]
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                code = TP.main(argv)
            text = buffer.getvalue()
            self.assertEqual(code, 0, text)
            for path in (out, base_out):
                self.assertGreater(os.path.getsize(path), 0)
            for heading in ("== a. target windows ==", "== b. ", "== c. ", "== d. top 8 rows by dur ==", "== e. baseline video", "== f. verification =="):
                self.assertIn(heading, text)
            self.assertNotIn("불일치", text)
            with open(out, "rb") as handle:
                before = handle.read()
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                again = TP.main(argv)
            self.assertEqual(again, 2)
            self.assertIn("덮어쓰지 않습니다", buffer.getvalue())
            with open(out, "rb") as handle:
                self.assertEqual(handle.read(), before)


class FigureValuesTest(unittest.TestCase):
    def test_first_window_start_and_end_seconds(self):
        indices = [10 * (i + 1) for i in range(40)]                    # fps 10 -> 행 i 는 (i+1) 초에 확정
        windows = [window(1, 3.0, time_sec=30.0), window(2, 4.0, time_sec=32.0)]
        spans = TP.window_spans(windows, indices, 10.0)
        info = TP.first_window_info(spans, windows)
        self.assertEqual((info["start_s"], info["end_s"], info["time_sec"], info["mean"]), (1.0, 30.0, 30.0, 3.0))

    def test_window_length_range_over_all_windows(self):
        indices = [0, 5, 7] + [10 + i for i in range(40)]
        windows = [window(1, 1.0, end_row=29), window(2, 1.0, end_row=31), window(3, 1.0, end_row=39)]
        spans = TP.window_spans(windows, indices, 1.0)
        lengths = [s["end_s"] - s["start_s"] for s in spans]
        self.assertEqual(TP.window_length_range(spans), (min(lengths), max(lengths)))
        self.assertEqual(TP.window_length_range(spans[:1]), (lengths[0], lengths[0]))

    def test_reference_boundary_is_the_confirmation_time_of_the_last_sample_row(self):
        indices = [10 * (i + 1) for i in range(20)]
        self.assertEqual(TP.reference_boundary_second(indices, 12, 10.0), 12.0)            # 12번째 행 = 번호 11 = 프레임 120
        self.assertEqual(TP.reference_boundary_second(indices, 1, 10.0), 1.0)

    def test_bar_duration_text_and_grouping(self):
        self.assertEqual(TP.duration_text(31, 37), "6 s")
        self.assertEqual(TP.duration_text(65, 67.5), "2.5 s")
        bars = TP.event_bars(EVENTS)
        self.assertEqual(bars["closure"], [(10.0, 12.0, "2 s")])
        self.assertEqual(bars["down"], [(20.0, 22.0, "2 s")])
        self.assertEqual(bars["frown"], [(30.0, 31.0, "1 s")])
        self.assertTrue(all(len(item) == 3 for group in bars.values() for item in group))        # 사건 코드는 담지 않는다

    def test_long_blinks_use_30_frames_inclusive_and_29_is_excluded(self):
        rows = make_rows(4, [29.0, 30.0, 12.0, 214.0])
        self.assertEqual(TP.long_blink_seconds(rows, [30, 60, 90, 120], 30.0), [2.0, 4.0])

    def test_title_rounds_the_length_and_has_no_file_name(self):
        self.assertEqual(TP.video_title("script", 5733, 29.99843022343153), "Script test video (191 s)")
        self.assertEqual(TP.video_title("normal", 9101, 29.998351920013185), "Normal everyday video (303 s)")
        self.assertEqual(TP.video_title("script", 1005, 10.0), "Script test video (101 s)")           # 100.5 는 올림

    def test_how_to_read_has_three_lines_and_the_script_one_has_a_fourth(self):
        self.assertEqual(len(TP.how_to_read_lines((18.2, 69.4), False)), 3)
        lines = TP.how_to_read_lines((18.2, 69.4), True)
        self.assertEqual(len(lines), 4)
        self.assertIn("(18-69 s long here)", lines[0])
        self.assertIn("personal reference", lines[3])
        self.assertTrue(TP.how_to_read_text((18.2, 69.4), True).startswith(TP.HOW_TITLE))

    def test_figure_wording_has_no_internal_terms_event_codes_or_class_names(self):
        texts = [TP.SUBTITLE, TP.LEGEND_LINE, TP.LEGEND_BAND, TP.LEGEND_DOTS, TP.END_HIGH, TP.END_LOW, TP.NO_SCORE, TP.REFERENCE_TEXT,
                 TP.LONG_BLINK_LABEL, TP.EVENT_NOTE, TP.how_to_read_text((10, 20), True)] + [label for _group, label in TP.EVENT_ROWS]
        joined = " ".join(texts)
        self.assertNotIn("fold", joined.lower())
        for code in ("B1", "B2", "B3", "C1", "C2", "D1", "D2", "E1", "E2"):
            self.assertNotIn(code, joined)
        self.assertIn("reference only, not an alarm", TP.SUBTITLE)
        self.assertNotIn("확률", joined)


class NearestClosureTest(unittest.TestCase):
    CLOSURES = [{"id": "B1", "group": "closure", "start_s": 31.0, "end_s": 37.0}, {"id": "B2", "group": "closure", "start_s": 39.0, "end_s": 45.0},
                {"id": "D1", "group": "down", "start_s": 72.0, "end_s": 77.0}]

    def test_nearest_end_and_signed_difference(self):
        self.assertEqual(TP.nearest_closure(37.74, self.CLOSURES)[0], "B1")
        self.assertAlmostEqual(TP.nearest_closure(37.74, self.CLOSURES)[1], 0.74)
        self.assertEqual(TP.nearest_closure(44.0, self.CLOSURES)[0], "B2")
        self.assertAlmostEqual(TP.nearest_closure(44.0, self.CLOSURES)[1], -1.0)                 # 사건 끝보다 이르면 음수
        self.assertEqual(TP.nearest_closure(100.0, self.CLOSURES)[0], "B2")                      # 눈 감기가 아닌 사건은 후보가 아니다

    def test_tie_goes_to_the_earlier_event_and_no_closures_gives_none(self):
        self.assertEqual(TP.nearest_closure(41.0, self.CLOSURES)[0], "B1")                        # 끝 37 과 45 에서 같은 거리(4와 4)
        self.assertIsNone(TP.nearest_closure(5.0, [e for e in self.CLOSURES if e["group"] != "closure"]))

    def test_dur_table_has_the_two_new_columns_with_signed_seconds(self):
        rows = make_rows(3, [8.0, 214.0, 9.0])
        top = TP.top_dur_rows(rows, [30, 1130, 3000], 30.0, self.CLOSURES, 1)
        self.assertEqual((top[0]["nearest"], round(top[0]["delta_s"], 2)), ("B1", 0.67))
        lines = TP.dur_table(top)
        self.assertIn("nearest closure event", lines[0])
        self.assertIn("second minus event end", lines[0])
        self.assertEqual(lines[1].split()[-2:], ["B1", "+0.67"])
        self.assertEqual(TP.dur_table(top, with_event=False)[0].split(), ["row", "frame", "second", "dur", "amp", "vel"])

    def test_rows_without_closure_events_show_a_dash(self):
        top = TP.top_dur_rows(make_rows(2, [5.0, 6.0]), [30, 60], 30.0, [], 1)
        self.assertEqual(TP.dur_table(top)[1].split()[-3:], ["-", "-", "-"])


class ImportWeightTest(unittest.TestCase):
    def test_importing_the_module_loads_no_heavy_library(self):
        code = ("import sys; import improve.hmlstm.timeline_plot; "
                "bad = [name for name in ('tensorflow', 'cv2', 'dlib', 'matplotlib') if name in sys.modules]; sys.exit(1 if bad else 0)")
        env = dict(os.environ, PYTHONPATH=SRC_ROOT, PYTHONDONTWRITEBYTECODE="1")
        self.assertEqual(subprocess.call([sys.executable, "-c", code], env=env), 0)

    def test_arguments(self):
        options = TP.parse_args(["--timeline", "t.json", "--target-rows", "r.json", "--truth", "e.json", "--out-png", "o.png"])
        self.assertIsNone(options.out_baseline_png)
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            TP.parse_args(["--timeline", "t.json"])


if __name__ == "__main__":
    unittest.main()
