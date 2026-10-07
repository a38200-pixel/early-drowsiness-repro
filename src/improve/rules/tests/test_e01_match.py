# E-01 일치 계산(e01_match.py) 시험. 합성 알람과 합성 EAR CSV만 쓰고, 영상과 실제 데이터 파일을 열지 않는다.
# 기대값은 PREREG_E-01.md 규칙(구간 창 = 구간 시작 +2.0초 ~ 끝 +1.0초, 구간별 일치 건수 = min(창 안 알람 수, 사건 수), 해석 구간)에서
# 먼저 계산했고 코드 결과에 맞추지 않았다. 창(초): 구간 1 = 14~20, 구간 2 = 47~66, 구간 3 = 71~83. 구간 시각(초): 12~19, 45~65, 69~82.
import contextlib
import io
import json
import os
import re
import tempfile
import unittest

from improve.rules import e01_match as M
from improve.rules import replay
from improve.rules.tests import synth
from improve.rules.tests.synth import CLOSED, OPEN, seg

PACKAGE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPO_ROOT = os.path.abspath(os.path.join(PACKAGE_DIR, "..", "..", ".."))
PREREG = os.path.join(REPO_ROOT, "docs", "improvement", "PREREG_E-01.md")


def alarm(time_s, episode_id=1):
    return {"episode_id": episode_id, "alarm_time_s": time_s}


def compute(times):
    return M.compute_combination([alarm(t, i + 1) for i, t in enumerate(times)], {}, [])


class ConstantsAgreeWithPreregTest(unittest.TestCase):
    """상수가 PREREG_E-01.md 본문의 숫자와 같은지 본문에서 읽어 확인한다."""

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(PREREG):
            raise unittest.SkipTest("PREREG_E-01.md 가 없어 상수 대조를 건너뜁니다.")
        with open(PREREG, encoding="utf-8") as handle:
            cls.text = handle.read()

    def test_interval_table(self):
        pattern = re.compile(r"^\| (\d) \| (\d+):(\d\d)~(\d+):(\d\d) \| (\d) \| (\d+):(\d\d)\.(\d)~(\d+):(\d\d)\.(\d) \((\d+)초~(\d+)초\) \|$",
                             re.MULTILINE)
        rows = pattern.findall(self.text)
        self.assertEqual(len(rows), 3)
        for row, interval, window in zip(rows, M.INTERVALS, M.windows()):
            index, sm, ss, em, es, events, wsm, wss, wst, wem, wes, wet, ws_sec, we_sec = row
            start = int(sm) * 60 + int(ss)
            end = int(em) * 60 + int(es)
            self.assertEqual((int(index), float(start), float(end), int(events)), interval)
            self.assertAlmostEqual(int(wsm) * 60 + int(wss) + int(wst) / 10.0, window[1])
            self.assertAlmostEqual(int(wem) * 60 + int(wes) + int(wet) / 10.0, window[2])
            self.assertEqual((float(ws_sec), float(we_sec)), (window[1], window[2]))

    def test_offsets_events_and_tolerance(self):
        self.assertIn("시작 +2.0초 ~ 끝 +1.0초", self.text)
        self.assertEqual((M.WINDOW_START_OFFSET_S, M.WINDOW_END_OFFSET_S), (2.0, 1.0))
        self.assertEqual(M.total_events(), 6)
        self.assertIn("사건 수의 합은 6이다", self.text)

    def test_thresholds_and_sources(self):
        self.assertIn("0.15, 0.21, 0.25", self.text)
        self.assertEqual(M.THRESHOLDS, (0.15, 0.21, 0.25))
        self.assertEqual(M.SOURCE_SET_NAMES, ("detected", "detected+tracked"))
        self.assertEqual(M.MAIN_COMBINATION, (0.21, "detected"))

    def test_interpretation_table(self):
        for low, high, text in M.INTERPRETATION:
            self.assertIn("| %d~%d건 | %s |" % (low, high, text), self.text)


class InterpretationTest(unittest.TestCase):
    def test_boundaries(self):
        expected = {6: "이 영상에서 대부분 일치", 5: "이 영상에서 대부분 일치",
                    4: "부분 일치, 원인 분석 필요(미확정)", 3: "부분 일치, 원인 분석 필요(미확정)",
                    2: "불일치, 규칙 정의 재검토", 1: "불일치, 규칙 정의 재검토", 0: "불일치, 규칙 정의 재검토"}
        for count, text in expected.items():
            with self.subTest(count=count):
                self.assertEqual(M.interpretation(count), text)

    def test_invalid_counts(self):
        for bad in (-1, 7, True, 1.5, None):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    M.interpretation(bad)


class WindowEdgeTest(unittest.TestCase):
    def test_windows_are_disjoint_and_values(self):
        self.assertEqual([(w[1], w[2]) for w in M.windows()], [(14.0, 20.0), (47.0, 66.0), (71.0, 83.0)])

    def test_interval_one_edges(self):
        self.assertEqual(compute([14.0])["alarms_in_window"], [1, 0, 0])        # 창 시작은 창 안(이상)
        self.assertEqual(compute([13.999])["alarms_in_window"], [0, 0, 0])      # 시작 직전은 창 밖
        self.assertEqual(compute([20.0])["alarms_in_window"], [1, 0, 0])        # 창 끝은 창 안(이하)
        self.assertEqual(compute([20.001])["alarms_in_window"], [0, 0, 0])      # 끝 직후는 창 밖

    def test_interval_two_and_three_edges(self):
        self.assertEqual(compute([46.999])["alarms_in_window"], [0, 0, 0])
        self.assertEqual(compute([47.0])["alarms_in_window"], [0, 1, 0])
        self.assertEqual(compute([66.0])["alarms_in_window"], [0, 1, 0])
        self.assertEqual(compute([66.001])["alarms_in_window"], [0, 0, 0])
        self.assertEqual(compute([70.999])["alarms_in_window"], [0, 0, 0])
        self.assertEqual(compute([71.0])["alarms_in_window"], [0, 0, 1])
        self.assertEqual(compute([83.0])["alarms_in_window"], [0, 0, 1])
        self.assertEqual(compute([83.001])["alarms_in_window"], [0, 0, 0])

    def test_extra_alarm_description(self):
        # 13.999초: 구간 1 시각(12~19) 안이지만 창(14~20) 밖. 가장 가까운 창은 1이고 거리 0.001.
        extra = compute([13.999])["extra_alarms"][0]
        self.assertEqual(extra["relation"], "구간 안(창 밖)")
        self.assertEqual(extra["nearest_interval"], 1)
        self.assertAlmostEqual(extra["distance_to_window_s"], 0.001, places=6)
        # 20.001초: 구간 1 시각(12~19) 밖. 거리 0.001.
        extra = compute([20.001])["extra_alarms"][0]
        self.assertEqual(extra["relation"], "구간 밖")
        self.assertAlmostEqual(extra["distance_to_window_s"], 0.001, places=6)
        # 100초: 가장 가까운 창은 구간 3(71~83), 거리 17.
        extra = compute([100.0])["extra_alarms"][0]
        self.assertEqual(extra["nearest_interval"], 3)
        self.assertAlmostEqual(extra["distance_to_window_s"], 17.0, places=6)
        self.assertEqual(extra["alarm_time"], "1:40.0")

    def test_extra_alarm_joins_episode_info(self):
        episodes = {7: {"start_s": 100.0, "end_s": 110.0, "length_s": 10.0, "status": "released"}}
        result = M.compute_combination([alarm(103.0, 7)], episodes, [])
        extra = result["extra_alarms"][0]
        self.assertEqual((extra["episode_start_s"], extra["episode_length_s"], extra["episode_status"]), (100.0, 10.0, "released"))
        self.assertEqual(extra["alarm_time"], "1:43.0")


class MatchCountTest(unittest.TestCase):
    def test_full_match(self):
        # 구간 1: 15 / 구간 2: 48, 60 / 구간 3: 72, 75, 80 -> 일치 [1, 2, 3] = 6
        result = compute([15.0, 48.0, 60.0, 72.0, 75.0, 80.0])
        self.assertEqual((result["matches"], result["total_match"]), ([1, 2, 3], 6))
        self.assertEqual(result["excess_in_window"], [0, 0, 0])
        self.assertEqual(result["extra_alarms"], [])
        self.assertEqual(result["interpretation"], "이 영상에서 대부분 일치")

    def test_interval_three_with_two_alarms(self):
        # 구간 3 사건 3건, 알람 2개 -> min(2, 3) = 2. 전체 2건 -> 불일치 구간.
        result = compute([72.0, 75.0])
        self.assertEqual((result["matches"], result["total_match"]), ([0, 0, 2], 2))
        self.assertEqual(result["interpretation"], "불일치, 규칙 정의 재검토")

    def test_excess_in_window(self):
        # 구간 1 사건 1건, 창 안 알람 2개 -> 일치 1, 창 안 초과 1. 초과는 추가 알람(창 밖)이 아니다.
        result = compute([15.0, 16.0])
        self.assertEqual((result["alarms_in_window"], result["matches"], result["excess_in_window"]), ([2, 0, 0], [1, 0, 0], [1, 0, 0]))
        self.assertEqual(result["total_match"], 1)
        self.assertEqual(result["extra_alarms"], [])

    def test_totals_at_interpretation_boundaries(self):
        cases = [([15.0, 48.0, 60.0, 72.0, 75.0], 5, "이 영상에서 대부분 일치"),
                 ([15.0, 48.0, 72.0, 75.0], 4, "부분 일치, 원인 분석 필요(미확정)"),
                 ([48.0, 60.0, 72.0], 3, "부분 일치, 원인 분석 필요(미확정)"),
                 ([15.0, 72.0], 2, "불일치, 규칙 정의 재검토")]
        for times, total, text in cases:
            with self.subTest(total=total):
                result = compute(times)
                self.assertEqual(result["total_match"], total)
                self.assertEqual(result["interpretation"], text)

    def test_no_alarms(self):
        result = compute([])
        self.assertEqual((result["total_match"], result["matches"]), (0, [0, 0, 0]))
        self.assertEqual(result["interpretation"], "불일치, 규칙 정의 재검토")

    def test_unjudgeable_aggregates(self):
        result = M.compute_combination([], {}, [{"length_s": 1.0}, {"length_s": 2.5}])
        self.assertEqual(result["unjudgeable"], {"n": 2, "total_length_s": 3.5, "longest_s": 2.5})
        self.assertEqual(M.compute_combination([], {}, [])["unjudgeable"], {"n": 0, "total_length_s": 0, "longest_s": 0.0})

    def test_format_time(self):
        self.assertEqual(M.format_time(103.0), "1:43.0")
        self.assertEqual(M.format_time(74.658), "1:14.7")
        self.assertEqual(M.format_time(0.04), "0:00.0")


class EndToEndTest(unittest.TestCase):
    """합성 EAR CSV -> replay.py(두 유효 출처) -> e01_match.py. 임시 폴더에만 쓴다."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = self.temp.name
        # 15.0초(프레임 450)부터 닫힘 10초(300프레임). 구간 시작 0 이므로 알람은 15.0 + 3.0 = 18.0초(구간 1 창 14~20 안).
        samples = synth.seq([seg(450, OPEN), seg(300, CLOSED)])
        self.csv = os.path.join(self.root, "ear.csv")
        with open(self.csv, "w", encoding="utf-8", newline="") as handle:
            handle.write("timestamp_s,ear_left,ear_right,source\n")
            for timestamp, ear, source in samples:
                handle.write("%r,%r,%r,%s\n" % (timestamp, ear, ear, source))

    def run_quiet(self, function, args):
        with contextlib.redirect_stdout(io.StringIO()):
            return function(args)

    def replay_both(self, suffix=""):
        detected = os.path.join(self.root, "detected" + suffix)
        both = os.path.join(self.root, "both" + suffix)
        self.run_quiet(replay.main, ["--input", self.csv, "--out-dir", detected, "--thresholds", "0.15", "0.21", "0.25",
                                     "--valid-sources", "detected"])
        self.run_quiet(replay.main, ["--input", self.csv, "--out-dir", both, "--thresholds", "0.15", "0.21", "0.25",
                                     "--valid-sources", "detected", "tracked"])
        return detected, both

    def match(self, detected, both, name):
        out = os.path.join(self.root, name)
        self.run_quiet(M.main, ["--replay", "detected=%s" % detected, "--replay", "detected+tracked=%s" % both, "--out-dir", out])
        with open(os.path.join(out, "e01_match.json"), encoding="utf-8") as handle:
            return out, json.load(handle)

    def test_six_combinations(self):
        detected, both = self.replay_both()
        _out, payload = self.match(detected, both, "out")
        combos = payload["combinations"]
        self.assertEqual(len(combos), 6)
        self.assertEqual([(c["threshold"], c["valid_sources"]) for c in combos],
                         [(t, n) for n in ("detected", "detected+tracked") for t in (0.15, 0.21, 0.25)])
        for combo in combos:
            self.assertEqual(combo["matches"], [1, 0, 0])
            self.assertEqual(combo["total_match"], 1)
            self.assertEqual(combo["extra_alarms"], [])
            self.assertAlmostEqual(combo["alarm_times_in_window"][0][0], 18.0, places=5)
        self.assertEqual([c["is_main"] for c in combos], [False, True, False, False, False, False])

    def test_deterministic(self):
        detected, both = self.replay_both()
        out_a, _ = self.match(detected, both, "out_a")
        out_b, _ = self.match(detected, both, "out_b")
        for name in ("e01_match.json", "e01_match.txt"):
            with open(os.path.join(out_a, name), encoding="utf-8") as first, open(os.path.join(out_b, name), encoding="utf-8") as second:
                self.assertEqual(first.read(), second.read())

    def test_text_has_no_forbidden_words(self):
        detected, both = self.replay_both()
        out, _ = self.match(detected, both, "out")
        with open(os.path.join(out, "e01_match.txt"), encoding="utf-8") as handle:
            text = handle.read()
        for word in ("성능", "정확도", "개선 확인", "검출 성능"):
            self.assertNotIn(word, text)

    def test_out_dir_required_and_existing_not_overwritten(self):
        detected, both = self.replay_both()
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                M.main(["--replay", "detected=%s" % detected, "--replay", "detected+tracked=%s" % both])
        self.match(detected, both, "out")
        with self.assertRaises(FileExistsError):
            self.match(detected, both, "out")

    def test_bad_replay_names(self):
        detected, both = self.replay_both()
        out = os.path.join(self.root, "bad")
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                M.main(["--replay", "camera=%s" % detected, "--out-dir", out])
        with self.assertRaises(ValueError):
            M.compute_all({"detected": detected})        # detected+tracked 폴더가 없다


if __name__ == "__main__":
    unittest.main()
