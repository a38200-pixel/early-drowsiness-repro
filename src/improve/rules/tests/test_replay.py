# C2 오프라인 재생 시험(합성 CSV, 임시 폴더). 기대값은 문서 규칙에서 먼저 계산했다.
import contextlib
import csv
import inspect
import io
import json
import os
import tempfile
import unittest

from improve.rules import replay
from improve.rules.tests import synth
from improve.rules.tests.synth import CLOSED, OPEN, miss, seg


def write_csv(path, samples, header=("timestamp_s", "ear_left", "ear_right", "source")):
    """합성 샘플 (timestamp_s, ear, source) 을 CSV 로 쓴다. ear 가 None 이면 두 EAR 칸을 비운다."""
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for timestamp, ear, source in samples:
            writer.writerow(["%r" % timestamp, "" if ear is None else "%r" % ear, "" if ear is None else "%r" % ear, source])


def run_main(args):
    with contextlib.redirect_stdout(io.StringIO()):
        return replay.main(args)


class ReplayTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = self.temp.name

    def path(self, name):
        return os.path.join(self.root, name)

    def load_summary(self, out_dir):
        with open(os.path.join(out_dir, "summary.json"), encoding="utf-8") as handle:
            return json.load(handle)

    def test_thresholds_change_alarm_counts(self):
        # EAR 0.18 일정 10초: 0.15 에서는 열림(알람 0), 0.21 과 0.25 에서는 닫힘(알람 1).
        write_csv(self.path("ear.csv"), synth.seq([seg(300, 0.18)]))
        out_dir = self.path("out")
        self.assertEqual(run_main(["--input", self.path("ear.csv"), "--out-dir", out_dir]), 0)
        summary = self.load_summary(out_dir)
        self.assertEqual([item["threshold"] for item in summary], [0.15, 0.21, 0.25])
        self.assertEqual([item["n_alarms"] for item in summary], [0, 1, 1])
        for label in ("0.15", "0.21", "0.25"):
            for kind in ("alarms", "episodes", "unjudgeable"):
                self.assertTrue(os.path.exists(os.path.join(out_dir, "%s_thr_%s.csv" % (kind, label))))
        with open(os.path.join(out_dir, "unjudgeable_thr_0.21.csv"), encoding="utf-8") as handle:
            self.assertEqual(handle.read().strip(), "start_s,end_s,length_s")      # 판정 불가가 없어도 파일은 쓴다
        with open(os.path.join(out_dir, "alarms_thr_0.21.csv"), encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(float(rows[0]["alarm_time_s"]), 3.0, places=5)

    def test_unjudgeable_intervals_are_reported(self):
        # 닫힘 0..29, 얼굴 없음 30..74(45프레임 = 1.5초), 닫힘 75..: 판정 불가 [1.0, 2.5] 길이 1.5, 임계값과 무관.
        write_csv(self.path("ear.csv"), synth.seq([seg(30, CLOSED), miss(45), seg(60, CLOSED)]))
        out_dir = self.path("out")
        run_main(["--input", self.path("ear.csv"), "--out-dir", out_dir])
        for item in self.load_summary(out_dir):
            self.assertEqual(item["unjudgeable"]["n"], 1)
            self.assertAlmostEqual(item["unjudgeable"]["total_length_s"], 1.5, places=5)
        with open(os.path.join(out_dir, "unjudgeable_thr_0.21.csv"), encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(float(rows[0]["start_s"]), 1.0, places=5)
        self.assertAlmostEqual(float(rows[0]["end_s"]), 2.5, places=5)
        with open(os.path.join(out_dir, "summary.txt"), encoding="utf-8") as handle:
            self.assertIn("판정 불가 구간: 1개", handle.read())

    def test_valid_sources_option(self):
        # tracked 로 이어진 닫힘 10초: 기본(detected 만 유효)에서는 알람 0, --valid-sources detected tracked 에서는 알람 1(임계값 0.21).
        write_csv(self.path("ear.csv"), synth.seq([seg(300, CLOSED, "tracked")]))
        default_dir = self.path("out_default")
        run_main(["--input", self.path("ear.csv"), "--out-dir", default_dir, "--thresholds", "0.21"])
        self.assertEqual(self.load_summary(default_dir)[0]["n_alarms"], 0)
        chosen_dir = self.path("out_chosen")
        run_main(["--input", self.path("ear.csv"), "--out-dir", chosen_dir, "--thresholds", "0.21",
                  "--valid-sources", "detected", "tracked"])
        self.assertEqual(self.load_summary(chosen_dir)[0]["n_alarms"], 1)

    def test_out_dir_is_required(self):
        write_csv(self.path("ear.csv"), synth.seq([seg(30, OPEN)]))
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                replay.main(["--input", self.path("ear.csv")])
        self.assertIs(inspect.signature(replay.write_outputs).parameters["out_dir"].default, inspect.Parameter.empty)

    def test_existing_outputs_are_not_overwritten(self):
        write_csv(self.path("ear.csv"), synth.seq([seg(30, OPEN)]))
        out_dir = self.path("out")
        run_main(["--input", self.path("ear.csv"), "--out-dir", out_dir])
        with self.assertRaises(FileExistsError):
            run_main(["--input", self.path("ear.csv"), "--out-dir", out_dir])

    def test_csv_reading(self):
        write_csv(self.path("ear.csv"), [(0.0, 0.2, "detected"), (0.1, None, "detected"), (0.2, 0.3, "none")])
        samples = replay.read_ear_csv(self.path("ear.csv"))
        self.assertEqual(samples[0], (0.0, 0.2, "detected"))
        self.assertEqual(samples[1], (0.1, None, "detected"))
        self.assertEqual(samples[2], (0.2, None, "none"))       # source none 이면 값이 있어도 EAR 없음

    def test_left_and_right_are_averaged(self):
        with open(self.path("ear.csv"), "w", encoding="utf-8", newline="") as handle:
            handle.write("timestamp_s,ear_left,ear_right,source\n0.0,0.10,0.30,detected\n0.1,0.10,,detected\n")
        samples = replay.read_ear_csv(self.path("ear.csv"))
        self.assertAlmostEqual(samples[0][1], 0.20, places=12)
        self.assertIsNone(samples[1][1])                         # 한쪽이 비면 EAR 없음

    def test_bad_csv(self):
        with open(self.path("bad1.csv"), "w", encoding="utf-8", newline="") as handle:
            handle.write("timestamp_s,ear_left,source\n0.0,0.1,detected\n")
        with self.assertRaises(ValueError):
            replay.read_ear_csv(self.path("bad1.csv"))
        with open(self.path("bad2.csv"), "w", encoding="utf-8", newline="") as handle:
            handle.write("timestamp_s,ear_left,ear_right,source\n0.0,0.1,0.1,camera\n")
        with self.assertRaises(ValueError):
            replay.read_ear_csv(self.path("bad2.csv"))
        with open(self.path("bad3.csv"), "w", encoding="utf-8", newline="") as handle:
            handle.write("timestamp_s,ear_left,ear_right,source\nabc,0.1,0.1,detected\n")
        with self.assertRaises(ValueError):
            replay.read_ear_csv(self.path("bad3.csv"))

    def test_length_distribution_in_summary(self):
        # 닫힘 60프레임(2.0초) 뒤 열림 20프레임(0.667초)에서 해제, 다시 닫힘 120프레임에서 끝(알람 유지).
        # 첫 구간: 시작 0, 끝 = 열린 구간의 시작 2.0 -> 길이 2.0(해제됨). 둘째 구간: 시작 80/30, 마지막 닫힘 199/30 -> 길이 119/30 = 3.9667
        # (영상 끝에서 알람이 울렸으므로 유지). 길이 목록은 [2.0, 3.9667].
        write_csv(self.path("ear.csv"), synth.seq([seg(60, CLOSED), seg(20, OPEN), seg(120, CLOSED)]))
        out_dir = self.path("out")
        run_main(["--input", self.path("ear.csv"), "--out-dir", out_dir, "--thresholds", "0.21"])
        item = self.load_summary(out_dir)[0]
        stats = item["episode_length_distribution"]["stats"]
        self.assertEqual(stats["n"], 2)
        self.assertAlmostEqual(stats["min"], 2.0, places=5)
        self.assertAlmostEqual(stats["max"], 119.0 / 30.0, places=5)
        self.assertEqual(item["n_alarms"], 1)


if __name__ == "__main__":
    unittest.main()
