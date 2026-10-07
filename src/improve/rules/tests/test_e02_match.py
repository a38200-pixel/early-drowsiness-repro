# E-02 일치 계산(e02_match.py) 시험. 합성 알람과 합성 EAR CSV, 문서 텍스트(PREREG_E-02.md, DATA_REGISTRY.md, e02_truth.json)만 읽는다.
# 영상과 실제 데이터 파일을 열지 않는다. 기대값은 PREREG_E-02.md 규칙에서 먼저 계산했고 코드 결과에 맞추지 않았다.
#   창 = 눈 감기 사건마다 [시작 + 2.0, 끝 + 1.0](경계 포함): B1 33~38, B2 41~46, B3 49~55, C1 61~62, C2 67~69(초).
#   분류: 길이 3.5초 이상 양성, 2.5초 이하 음성, 그 사이 경계. 양성 사건은 B1, B2, B3, 음성 C1, 경계 C2.
#   구간(초): A 0~30, B 31~54, C 59~68, D 72~87, E 90~107, F 108~끝. 구간 사이 공백 30~31, 54~59, 68~72, 87~90, 107~108.
import contextlib
import copy
import io
import json
import os
import re
import tempfile
import unittest

from improve.rules import e02_match as M
from improve.rules import replay

PACKAGE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPO_ROOT = os.path.abspath(os.path.join(PACKAGE_DIR, "..", "..", ".."))
DOCS = os.path.join(REPO_ROOT, "docs", "improvement")
TRUTH_PATH = os.path.join(DOCS, "e02_truth.json")
PREREG = os.path.join(DOCS, "PREREG_E-02.md")
REGISTRY = os.path.join(DOCS, "DATA_REGISTRY.md")


def cells(line):
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def table_lines(path):
    with open(path, encoding="utf-8") as handle:
        return [line.rstrip("\n") for line in handle if line.startswith("|")]


def alarm(time_s, episode_id):
    return {"episode_id": episode_id, "alarm_time_s": time_s}


def compute(times, truth=None):
    truth = truth or M.load_truth(TRUTH_PATH)
    return M.compute_combination([alarm(t, i + 1) for i, t in enumerate(times)], {}, [], truth)


def event(result, event_id):
    return [e for e in result["events"] if e["id"] == event_id][0]


class TruthAgreementTest(unittest.TestCase):
    """정답 JSON 이 PREREG_E-02.md 표와 DATA_REGISTRY.md 표의 값과 같은지 본문에서 읽어 확인한다."""

    @classmethod
    def setUpClass(cls):
        for path in (TRUTH_PATH, PREREG, REGISTRY):
            if not os.path.exists(path):
                raise unittest.SkipTest("문서 파일이 없어 대조 시험을 건너뜁니다: %s" % os.path.basename(path))
        cls.truth = M.load_truth(TRUTH_PATH)
        cls.events = dict((e["id"], e) for e in cls.truth["events"])
        cls.intervals = dict((i["id"], i) for i in cls.truth["intervals"])
        with open(PREREG, encoding="utf-8") as handle:
            cls.prereg_text = handle.read()

    def test_prereg_event_table_equals_truth(self):
        found = []
        for line in table_lines(PREREG):
            row = cells(line)
            if len(row) == 8 and re.match(r"^[A-F]\d$", row[0]):
                found.append(row[0])
                event_truth = self.events[row[0]]
                self.assertEqual(row[1], event_truth["interval"])
                self.assertTrue(row[2].startswith(event_truth["kind"]), row[2])
                start_text, start_s = row[3].split(" / ")
                end_text, end_s = row[4].split(" / ")
                self.assertEqual((start_text, int(start_s)), (event_truth["start_text"], event_truth["start_s"]))
                self.assertEqual((end_text, int(end_s)), (event_truth["end_text"], event_truth["end_s"]))
                self.assertEqual(int(row[5]), event_truth["length_s"])
                self.assertEqual(row[6], event_truth["class"])
        self.assertEqual(sorted(found), sorted(self.events))

    def test_prereg_windows_equal_computed_windows(self):
        wins = dict((w["id"], w) for w in M.closure_windows(self.truth))
        for line in table_lines(PREREG):
            row = cells(line)
            if len(row) == 8 and re.match(r"^[A-F]\d$", row[0]):
                if row[7] == "없음":
                    self.assertNotIn(row[0], wins)
                else:
                    self.assertEqual(row[7], "%g~%g" % (wins[row[0]]["ws"], wins[row[0]]["we"]))

    def test_prereg_interval_table_equals_truth(self):
        found = []
        for line in table_lines(PREREG):
            row = cells(line)
            if len(row) == 4 and re.match(r"^[A-F]$", row[0]):
                found.append(row[0])
                interval = self.intervals[row[0]]
                self.assertEqual(row[1], interval["label"])
                self.assertEqual(int(row[2]), interval["start_s"])
                if interval["end_s"] is None:
                    self.assertEqual(row[3], "영상 끝")
                else:
                    self.assertEqual(int(row[3]), interval["end_s"])
        self.assertEqual(sorted(found), sorted(self.intervals))

    def test_registry_event_times_equal_truth_text(self):
        found = []
        for line in table_lines(REGISTRY):
            row = cells(line)
            match = re.match(r"^([B-E])\S* (?:감김|동작) (\d)번$", row[0]) if len(row) == 3 else None
            if match:
                event_id = match.group(1) + match.group(2)
                found.append(event_id)
                self.assertEqual((row[1], row[2]), (self.events[event_id]["start_text"], self.events[event_id]["end_text"]))
        self.assertEqual(sorted(found), sorted(self.events))

    def test_registry_interval_table_equals_truth_text(self):
        found = []
        for line in table_lines(REGISTRY):
            row = cells(line)
            if len(row) == 4 and re.match(r"^[A-F]$", row[0]):
                found.append(row[0])
                interval = self.intervals[row[0]]
                self.assertEqual(row[1], interval["label"].replace("정상", "정상") if row[0] == "A" else interval["label"])
                self.assertEqual(row[2], interval["start_text"])
                if row[0] == "F":
                    self.assertTrue(row[3].startswith("영상 끝"))
                else:
                    self.assertEqual(row[3], interval["end_text"])
        self.assertEqual(sorted(found), sorted(self.intervals))

    def test_constants_agree_with_prereg_text(self):
        self.assertIn("[시작 + 2.0초, 끝 + 1.0초]", self.prereg_text)
        self.assertEqual((M.WINDOW_START_OFFSET_S, M.WINDOW_END_OFFSET_S), (2.0, 1.0))
        self.assertIn("길이 3.5초 이상 = 양성, 2.5초 이하 = 음성", self.prereg_text)
        self.assertEqual((M.POSITIVE_MIN_LENGTH_S, M.NEGATIVE_MAX_LENGTH_S), (3.5, 2.5))
        self.assertEqual(self.truth["classification"]["positive_min_length_s"], 3.5)
        self.assertEqual(self.truth["classification"]["negative_max_length_s"], 2.5)
        self.assertIn("0.15 / 0.21 / 0.25", self.prereg_text)
        self.assertEqual(M.THRESHOLDS, (0.15, 0.21, 0.25))
        self.assertEqual(M.MAIN_COMBINATION, (0.21, "detected"))
        for text in (M.INTERP_ALL, M.INTERP_ONE_SHORT, M.INTERP_TWO_SHORT):
            self.assertIn("| %s |" % text, self.prereg_text)
        self.assertIn("| 전부 일치(3/3) |", self.prereg_text)
        self.assertIn("| 1건 모자람(2/3) |", self.prereg_text)
        self.assertIn("| 2건 이상 모자람(1/3 이하) |", self.prereg_text)

    def test_truth_counts(self):
        classes = [e["class"] for e in self.truth["events"] if e["kind"] == "눈 감기"]
        self.assertEqual((classes.count("양성"), classes.count("음성"), classes.count("경계")), (3, 1, 1))


class ClassificationTest(unittest.TestCase):
    def test_boundaries(self):
        cases = [(7.0, "양성"), (3.5, "양성"), (3.4999, "경계"), (3.0, "경계"), (2.5001, "경계"), (2.5, "음성"), (2.0, "음성")]
        for length, expected in cases:
            with self.subTest(length=length):
                self.assertEqual(M.classify_length(length), expected)

    def test_validate_truth_rejects_inconsistent_values(self):
        truth = copy.deepcopy(M.load_truth(TRUTH_PATH))
        truth["events"][0]["length_s"] = 5            # 끝 - 시작 = 6
        with self.assertRaises(ValueError):
            M.validate_truth(truth)
        truth = copy.deepcopy(M.load_truth(TRUTH_PATH))
        truth["events"][3]["class"] = "양성"           # C1 길이 2 는 음성
        with self.assertRaises(ValueError):
            M.validate_truth(truth)


class WindowTest(unittest.TestCase):
    def test_window_values(self):
        wins = dict((w["id"], (w["ws"], w["we"])) for w in M.closure_windows(M.load_truth(TRUTH_PATH)))
        self.assertEqual(wins, {"B1": (33, 38), "B2": (41, 46), "B3": (49, 55), "C1": (61, 62), "C2": (67, 69)})

    def test_edges_of_first_window(self):
        self.assertTrue(event(compute([33.0]), "B1")["matched"])       # 창 시작은 창 안
        self.assertFalse(event(compute([32.999]), "B1")["matched"])    # 시작 직전은 창 밖
        self.assertTrue(event(compute([38.0]), "B1")["matched"])       # 창 끝은 창 안
        self.assertFalse(event(compute([38.001]), "B1")["matched"])    # 끝 직후는 창 밖
        self.assertEqual(compute([32.999])["extra_alarms_by_segment"]["B"]["count"], 1)
        self.assertEqual(compute([38.001])["extra_alarms_by_segment"]["B"]["count"], 1)

    def test_edges_of_other_windows(self):
        self.assertTrue(event(compute([41.0]), "B2")["matched"])
        self.assertFalse(event(compute([40.999]), "B2")["matched"])
        self.assertTrue(event(compute([55.0]), "B3")["matched"])
        self.assertFalse(event(compute([55.001]), "B3")["matched"])

    def test_excess_in_window(self):
        # B1 창 안 알람 2개 -> 일치 1, 창 안 초과 1
        result = compute([34.0, 36.0])
        self.assertTrue(event(result, "B1")["matched"])
        self.assertEqual(event(result, "B1")["excess_in_window"], 1)
        self.assertEqual((result["matched_positives"], result["excess_in_window"], result["extra_alarms_total"]), (1, 1, 0))

    def test_negative_and_borderline_window_alarms_are_separate(self):
        result = compute([61.5, 68.0])
        self.assertEqual(result["negative_window_alarms"], 1)        # C1 창 61~62
        self.assertEqual(result["borderline_window_alarms"], 1)      # C2 창 67~69
        self.assertEqual(result["matched_positives"], 0)
        self.assertEqual(result["extra_alarms_total"], 0)
        self.assertIsNone(event(result, "C1")["matched"])
        self.assertIsNone(event(result, "C2")["matched"])
        # C1 창 끝 직후는 창 밖이고 구간 C 의 추가 알람
        result = compute([62.001])
        self.assertEqual((result["negative_window_alarms"], result["extra_alarms_by_segment"]["C"]["count"]), (0, 1))

    def test_overlapping_windows_assign_to_earlier_window(self):
        truth = {"video": "synthetic", "intervals": [{"id": "A", "label": "정상", "start_s": 0, "end_s": 100}],
                 "events": [{"id": "X1", "interval": "A", "kind": "눈 감기", "start_s": 10, "end_s": 14, "length_s": 4, "class": "양성"},
                            {"id": "X2", "interval": "A", "kind": "눈 감기", "start_s": 13, "end_s": 20, "length_s": 7, "class": "양성"}]}
        # 창: X1 12~15, X2 15~21. 15.0 은 두 창에 모두 속하고 시작이 이른 X1 에만 귀속한다.
        result = compute([15.0], truth)
        self.assertEqual(result["window_overlaps"], [["X1", "X2"]])
        self.assertEqual(result["window_overlap_status"], "미확정")
        self.assertTrue(event(result, "X1")["matched"])
        self.assertFalse(event(result, "X2")["matched"])
        self.assertEqual(result["matched_positives"], 1)
        # 실제 정답은 창이 겹치지 않는다
        self.assertEqual(compute([])["window_overlap_status"], "없음")


class SegmentAttributionTest(unittest.TestCase):
    def test_intervals_and_gaps(self):
        times = [10.0, 30.5, 39.0, 56.0, 70.0, 80.0, 88.0, 95.0, 107.5, 120.0]
        result = compute(times)
        by = result["extra_alarms_by_segment"]
        self.assertEqual(result["extra_alarms_total"], 10)
        for label in ("A", "B", "D", "E", "F"):
            self.assertEqual(by[label]["count"], 1, label)
        self.assertEqual(by["C"]["count"], 0)
        for gap in ("공백(30~31초)", "공백(54~59초)", "공백(68~72초)", "공백(87~90초)", "공백(107~108초)"):
            self.assertEqual(by[gap]["count"], 1, gap)
        self.assertEqual(by["B"]["times"], [39.0])
        self.assertEqual(by["A"]["times_text"], ["0:10.0"])

    def test_interval_boundaries_are_inclusive(self):
        result = compute([0.0, 30.0, 31.0, 108.0])
        by = result["extra_alarms_by_segment"]
        self.assertEqual((by["A"]["count"], by["B"]["count"], by["F"]["count"]), (2, 1, 1))

    def test_c_and_d_end_follow_registry_values(self):
        # 구간 표(레지스트리): C 59~68, D 72~87. 68.0 은 C, 87.0 은 D 에 속하고 68.5 와 87.5 는 공백이다.
        # 68.0 은 C2 창(67~69)의 안이므로 창 귀속이 구간 귀속보다 먼저이다 -> 구간 표 확인에는 창 밖 시각만 쓴다.
        result = compute([87.0, 87.5, 68.5, 72.0])
        by = result["extra_alarms_by_segment"]
        self.assertEqual(by["D"]["times"], [72.0, 87.0])
        self.assertEqual(by["공백(87~90초)"]["times"], [87.5])
        # 68.5 는 C2 창(67~69) 안이라 추가 알람이 아니다
        self.assertEqual(result["borderline_window_alarms"], 1)
        self.assertNotIn("공백(68~72초)", [k for k, v in by.items() if v["count"] > 0])
        self.assertEqual(by["C"]["count"], 0)

    def test_extra_alarm_carries_episode_info(self):
        truth = M.load_truth(TRUTH_PATH)
        episodes = {5: {"length_s": 4.5, "status": "released"}}
        result = M.compute_combination([alarm(10.0, 5)], episodes, [], truth)
        self.assertEqual((result["extra_alarms"][0]["episode_length_s"], result["extra_alarms"][0]["episode_status"]), (4.5, "released"))


class InterpretationTest(unittest.TestCase):
    def test_function(self):
        self.assertEqual(M.interpretation(3, 3), "이 영상에서 일치")
        self.assertEqual(M.interpretation(2, 3), "부분 일치, 원인 분석 필요(미확정)")
        self.assertEqual(M.interpretation(1, 3), "불일치, 규칙 정의 재검토")
        self.assertEqual(M.interpretation(0, 3), "불일치, 규칙 정의 재검토")
        for bad in ((4, 3), (-1, 3), (True, 3), (1.5, 3)):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    M.interpretation(*bad)

    def test_through_compute(self):
        # B1 34, B2 42, B3 50 은 각각 창 안 -> 일치 3/3
        self.assertEqual(compute([34.0, 42.0, 50.0])["interpretation"], "이 영상에서 일치")
        self.assertEqual(compute([34.0, 42.0])["interpretation"], "부분 일치, 원인 분석 필요(미확정)")
        self.assertEqual(compute([34.0])["interpretation"], "불일치, 규칙 정의 재검토")
        self.assertEqual(compute([])["interpretation"], "불일치, 규칙 정의 재검토")
        # 추가 알람과 음성·경계 창 알람은 해석 구간에 들어가지 않는다
        self.assertEqual(compute([34.0, 42.0, 50.0, 10.0, 61.5, 68.0])["interpretation"], "이 영상에서 일치")

    def test_unjudgeable_pass_through(self):
        truth = M.load_truth(TRUTH_PATH)
        result = M.compute_combination([], {}, [{"length_s": 1.0}, {"length_s": 2.5}], truth)
        self.assertEqual(dict(result["unjudgeable"]), {"n": 2, "total_length_s": 3.5, "longest_s": 2.5})
        empty = M.compute_combination([], {}, [], truth)
        self.assertEqual(dict(empty["unjudgeable"]), {"n": 0, "total_length_s": 0, "longest_s": 0.0})


class EndToEndTest(unittest.TestCase):
    """합성 EAR CSV -> replay.py -> e02_match.py. 영상 2 합성: 감김 B1 31~37, B2 39~45, B3 47~54, C1 59~61, C2 65~68(초)만 닫힘."""

    FPS = 30.0
    CLOSURES = ((31, 37), (39, 45), (47, 54), (59, 61), (65, 68))

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = self.temp.name
        self.video2 = self.write_dump("v2.csv", 3600, self.CLOSURES)       # 120초
        self.video1 = self.write_dump("v1.csv", 9300, ())                  # 310초, 모두 열림(EAR 0.30)

    def write_dump(self, name, frames, closures):
        path = os.path.join(self.root, name)
        with open(path, "w", encoding="utf-8", newline="") as handle:
            handle.write("timestamp_s,ear_left,ear_right,source\n")
            for index in range(frames):
                t = index / self.FPS
                ear = 0.10 if any(start <= t < end for start, end in closures) else 0.30
                handle.write("%r,%r,%r,detected\n" % (t, ear, ear))
        return path

    def quiet(self, function, args):
        with contextlib.redirect_stdout(io.StringIO()):
            return function(args)

    def replay_all(self):
        dirs = {}
        for tag, dump in (("v2", self.video2), ("v1", self.video1)):
            for name, sources in (("detected", ["detected"]), ("detected+tracked", ["detected", "tracked"])):
                out = os.path.join(self.root, "%s_%s" % (tag, name.replace("+", "_")))
                self.quiet(replay.main, ["--input", dump, "--out-dir", out, "--thresholds", "0.15", "0.21", "0.25",
                                         "--valid-sources"] + sources)
                dirs[(tag, name)] = out
        return dirs

    def match_args(self, dirs, out):
        return ["--truth", TRUTH_PATH,
                "--video2-replay", "detected=%s" % dirs[("v2", "detected")],
                "--video2-replay", "detected+tracked=%s" % dirs[("v2", "detected+tracked")],
                "--video1-replay", "detected=%s" % dirs[("v1", "detected")],
                "--video1-replay", "detected+tracked=%s" % dirs[("v1", "detected+tracked")],
                "--video1-dump", self.video1, "--min-reference-s", "300", "--out-dir", out]

    def run_match(self, dirs, name):
        out = os.path.join(self.root, name)
        self.quiet(M.main, self.match_args(dirs, out))
        with open(os.path.join(out, "e02_match.json"), encoding="utf-8") as handle:
            return out, json.load(handle)

    def test_results(self):
        dirs = self.replay_all()
        _out, payload = self.run_match(dirs, "out")
        video2 = payload["result"]["video2"]
        self.assertEqual(len(video2), 6)
        for item in video2:
            # B1, B2, B3 는 길이 6, 6, 7초라 시작 + 3.0초(34, 42, 50초)에 알람 -> 각 창 안. C1(2초)은 알람 없음.
            # C2(3초)는 열림 샘플(68.0초)에서 길이 3.0 -> 알람 시각 68.0, 창 67~69 안 -> 경계 창 알람 1.
            self.assertEqual((item["matched_positives"], item["positives"]), (3, 3))
            self.assertEqual((item["excess_in_window"], item["negative_window_alarms"], item["borderline_window_alarms"],
                              item["extra_alarms_total"]), (0, 0, 1, 0))
            self.assertEqual(item["interpretation"], "이 영상에서 일치")
            by_id = dict((e["id"], e) for e in item["events"])
            self.assertAlmostEqual(by_id["B1"]["alarm_times_in_window"][0], 34.0, places=5)
            self.assertAlmostEqual(by_id["B2"]["alarm_times_in_window"][0], 42.0, places=5)
            self.assertAlmostEqual(by_id["B3"]["alarm_times_in_window"][0], 50.0, places=5)
            self.assertAlmostEqual(by_id["C2"]["alarm_times_in_window"][0], 68.0, places=5)
            self.assertEqual(by_id["C1"]["alarm_times_in_window"], [])
        self.assertEqual([(i["threshold"], i["valid_sources"]) for i in video2],
                         [(t, n) for n in ("detected", "detected+tracked") for t in (0.15, 0.21, 0.25)])
        self.assertEqual([i["is_main"] for i in video2], [False, True, False, False, False, False])
        video1 = payload["result"]["video1"]
        self.assertEqual(len(video1), 6)
        for item in video1:
            self.assertEqual(item["alarm_count"], 0)
            self.assertAlmostEqual(item["ear_median"], 0.30, places=9)
            self.assertEqual(item["i09_status"], "정상")

    def test_i09_undetectable_for_low_ear_video1(self):
        self.video1 = self.write_dump("v1_low.csv", 9300, ((0, 400),))     # 모두 닫힘(EAR 0.10), 중앙값 0.10 <= 0.21
        dirs = self.replay_all()
        _out, payload = self.run_match(dirs, "out")
        for item in payload["result"]["video1"]:
            self.assertEqual(item["i09_status"], "탐지 불가능")
            self.assertAlmostEqual(item["ear_median"], 0.10, places=9)

    def test_deterministic_and_outputs(self):
        dirs = self.replay_all()
        out_a, _ = self.run_match(dirs, "out_a")
        out_b, _ = self.run_match(dirs, "out_b")
        for name in ("e02_match.json", "e02_match.txt", "e02_video2_summary.csv"):
            with open(os.path.join(out_a, name), "rb") as first, open(os.path.join(out_b, name), "rb") as second:
                self.assertEqual(first.read(), second.read(), name)
        with open(os.path.join(out_a, "e02_video2_summary.csv"), encoding="utf-8") as handle:
            lines = handle.read().splitlines()
        self.assertEqual(len(lines), 7)
        self.assertTrue(lines[0].startswith("threshold,valid_sources,matched_positives,positives"))
        with open(os.path.join(out_a, "e02_match.txt"), encoding="utf-8") as handle:
            text = handle.read()
        for word in ("성능", "정확도", "개선 확인", "검출 성능"):
            self.assertNotIn(word, text)

    def test_required_arguments_and_no_overwrite(self):
        dirs = self.replay_all()
        for missing in ("--out-dir", "--min-reference-s", "--truth"):
            args = self.match_args(dirs, os.path.join(self.root, "x"))
            index = args.index(missing)
            del args[index:index + 2]
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    M.main(args)
        self.run_match(dirs, "out")
        with self.assertRaises(FileExistsError):
            self.run_match(dirs, "out")

    def test_source_has_no_image_libraries(self):
        with open(os.path.join(PACKAGE_DIR, "e02_match.py"), encoding="utf-8") as handle:
            source = handle.read()
        for token in ("cv2", "PIL", "imwrite"):
            self.assertNotIn(token, source)


if __name__ == "__main__":
    unittest.main()
