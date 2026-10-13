# live_score_recheck 시험: 합성 JSON 과 가짜 채점기(score_service.FakeScorer)만 쓴다. 실제 TensorFlow 채점, 체크포인트, 웹캠은 쓰지 않는다.
import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest

from improve.hmlstm import live_combined as LC
from improve.hmlstm import live_score_recheck as RC
from improve.hmlstm import live_session as LS
from improve.hmlstm import original_hook as OH
from improve.hmlstm import score_service as SS
from improve.hmlstm.tests.test_live_combined import iterate, make_hook
from improve.hmlstm.tests.test_live_session import synthetic_rows

CONFIG = {"reference_s": 60.0, "extension_s": 120.0, "min_reference_rows": 30}


def base_rows(reference_count=40, post_count=36, gap_after_reference=False):
    """기준 구간(60초 설정)에 행 reference_count 개, 이후 post_count 개. 프레임 번호는 t * 30 이다."""
    rows, faces = synthetic_rows(reference_count + post_count)
    entries = []
    for index in range(reference_count + post_count):
        if index < reference_count:
            t = 1.0 + 1.45 * index
        else:
            t = 70.0 + (index - reference_count) if gap_after_reference else 61.0 + (index - reference_count)
        entries.append({"row": rows[index], "frame_index": int(round(t * 30)), "t": t, "face_frames": faces[index]})
    return entries


def make_document(entries=None, config=CONFIG, scorer=None):
    """라이브가 남겼을 법한 JSON: 행을 다시 넣어 만든 창을 가짜 채점기로 점수화한 값을 scores 로 기록한다(기록값 = 재계산값인 기준 문서)."""
    entries = base_rows() if entries is None else entries
    document = {"schema_version": RC.SCHEMA_VERSION, "kind": RC.KIND, "status": "finished", "fps_assumed": 30.0, "frames": 3000, "rows": entries,
                "session_phase": None, "reference": None, "scores": [], "config": dict(config)}
    session, jobs, _source = RC.replay_session(document)
    RC.score_jobs(session, jobs, scorer or SS.FakeScorer())
    document["session_phase"] = session.phase
    document["scores"] = [dict(item) for item in session.section["windows"]]
    if session.reference is not None:
        document["reference"] = {"rows": len(session.ref_rows), "end_time": session.reference_end_time,
                                 "means": [float(v) for v in session.reference.means], "stds": [float(v) for v in session.reference.stds]}
    return document


def recheck(document, tolerance=RC.DEFAULT_TOLERANCE):
    return RC.run(copy.deepcopy(document), SS.FakeScorer(), tolerance)


class ValidationTest(unittest.TestCase):
    def test_missing_fields_give_the_exact_field_name(self):
        document = make_document()
        prefix = "입력 JSON 에 필요한 필드가 없거나 형식이 맞지 않습니다: "
        cases = []

        def broken(change):
            copy_ = copy.deepcopy(document)
            change(copy_)
            return copy_
        cases.append((broken(lambda d: d["rows"][2].pop("t")), "rows[2].t"))
        cases.append((broken(lambda d: d["rows"][0].pop("face_frames")), "rows[0].face_frames"))
        cases.append((broken(lambda d: d["rows"][1].__setitem__("row", [1.0, 2.0])), "rows[1].row"))
        cases.append((broken(lambda d: d["scores"][0].pop("mean")), "scores[0].mean"))
        cases.append((broken(lambda d: d.pop("fps_assumed")), "fps_assumed"))
        cases.append((broken(lambda d: d.pop("scores")), "scores"))
        cases.append((broken(lambda d: d.__setitem__("reference", {"rows": 3, "means": [0.0]})), "reference.stds"))
        for bad, field in cases:
            with self.assertRaises(RC.RecheckError, msg=field) as caught:
                RC.validate_document(bad)
            self.assertEqual(str(caught.exception), prefix + field)
        with self.assertRaises(RC.RecheckError) as caught:
            RC.validate_document(broken(lambda d: d.__setitem__("kind", "other")))
        self.assertEqual(str(caught.exception), prefix + "kind (live_combined_run 이어야 함)")
        self.assertIs(RC.validate_document(document), document)

    def test_files_that_are_missing_not_json_or_not_finite_are_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(RC.RecheckError):
                RC.load_document(os.path.join(folder, "none.json"))
            bad = os.path.join(folder, "bad.json")
            with open(bad, "w", encoding="utf-8") as handle:
                handle.write("{not json")
            with self.assertRaises(RC.RecheckError):
                RC.load_document(bad)
            nan = os.path.join(folder, "nan.json")
            with open(nan, "w", encoding="utf-8") as handle:
                handle.write('{"kind": "live_combined_run", "x": NaN}')
            with self.assertRaises(RC.RecheckError) as caught:
                RC.load_document(nan)
            self.assertIn("NaN", str(caught.exception))

    def test_the_kind_and_schema_follow_live_combined_and_its_document_is_accepted(self):
        self.assertEqual((RC.KIND, RC.SCHEMA_VERSION), (LC.KIND, LC.SCHEMA_VERSION))
        hook, clock, stream = make_hook()
        rows, faces = synthetic_rows(3)
        for index in range(3):
            iterate(hook, stream, ear=0.3, source="detected", faces=faces[index], row=rows[index])
        reporter = LC.LiveReporter(hook, None, None, None, out=lambda line: None, flush=lambda: None)
        summary = reporter.emit("finished")
        document = json.loads(json.dumps(LC.build_run_document(hook, summary, None, "finished")))
        RC.validate_document(document)
        result, report = RC.run(document, SS.FakeScorer(), RC.DEFAULT_TOLERANCE)
        self.assertEqual(result["verdict"], RC.UNDECIDED)                    # 창이 없으면 비교할 것이 없다
        self.assertIn("settings: 기본값(JSON 에 기록 없음) ; fps assumed 30", report)


class ReplayTest(unittest.TestCase):
    def test_the_replay_reproduces_phases_and_windows(self):
        session, jobs, source = RC.replay_session(copy.deepcopy(make_document()))
        self.assertEqual(source, "JSON 의 config")
        self.assertEqual((len(session.ref_rows), len(session.post_rows)), (40, 36))
        self.assertEqual([job.k for job in jobs], [1, 2, 3, 4])              # 기준 이후 36행: 30번째 행에서 첫 창, 이후 2행마다

    def test_without_config_the_live_session_defaults_are_used_and_said(self):
        document = make_document()
        del document["config"]
        settings, source = RC.session_settings(document)
        self.assertEqual((settings["reference_s"], settings["extension_s"], settings["min_reference_rows"]), (LS.REFERENCE_S, LS.EXTENSION_S, LS.MIN_REFERENCE_ROWS))
        self.assertIn("기본값", source)

    def test_the_first_row_after_the_reference_time_is_a_post_row_and_the_end_time_is_that_rows_time(self):
        document = make_document(base_rows(gap_after_reference=True))
        session, _jobs, _source = RC.replay_session(copy.deepcopy(document))
        self.assertEqual(len(session.ref_rows), 40)                          # 60초 이후의 첫 행(70초)은 add_row 안의 advance 로 기준 이후 행이 된다
        self.assertEqual(session.reference_end_time, document["rows"][40]["t"])
        document["reference"]["end_time"] = 60.0                             # 라이브는 종료를 일으킨 틱의 시각을 기록한다
        result, report = recheck(document)
        self.assertEqual(result["verdict"], RC.MATCH)                        # 종료 시각의 차이는 판정에 쓰지 않는다
        self.assertAlmostEqual(result["reference"]["end_time_diff"], 10.0)

    def test_a_calibration_short_json_gives_no_windows(self):
        entries = base_rows(reference_count=10, post_count=0)
        entries.append(dict(copy.deepcopy(entries[-1]), frame_index=entries[-1]["frame_index"] + 3000, t=entries[-1]["t"] + 110.0))
        document = make_document(entries)
        self.assertEqual(document["session_phase"], LS.PHASE_SHORT)
        result, report = recheck(document)
        self.assertEqual((result["verdict"], result["count_recorded"], result["count_recomputed"]), (RC.UNDECIDED, 0, 0))
        self.assertIn("calibration_short", "\n".join(report))

    def test_rows_the_session_refuses_are_reported(self):
        document = make_document()
        document["rows"][5]["frame_index"] = 0
        with self.assertRaises(RC.RecheckError) as caught:
            RC.replay_session(document)
        self.assertIn("frame_index", str(caught.exception))


class CompareTest(unittest.TestCase):
    def test_an_untouched_document_matches_with_zero_differences(self):
        result, report = recheck(make_document())
        self.assertEqual(result["verdict"], RC.MATCH)
        self.assertEqual((result["count_recorded"], result["count_recomputed"], result["frame_mismatches"], result["exceeded"]), (4, 4, 0, 0))
        self.assertEqual(result["max_diff"], {"mean": 0.0, "min": 0.0, "max": 0.0})
        self.assertTrue(result["reference"]["rows_match"] and result["reference"]["stats_match"])
        self.assertEqual(report[-2], "verdict: 일치")
        self.assertEqual(report[-1], "이 값은 같은 행의 재채점 비교이며 B1 오프라인 경로와의 비교가 아님")

    def test_a_different_window_count_is_a_mismatch(self):
        document = make_document()
        del document["scores"][-1]
        result, report = recheck(document)
        self.assertEqual((result["verdict"], result["count_recorded"], result["count_recomputed"], result["counts_match"]), (RC.MISMATCH, 3, 4, False))
        self.assertIn("windows: recorded 3 ; recomputed 4 ; 불일치", report)

    def test_frame_numbers_are_compared_per_window(self):
        document = make_document()
        document["scores"][1]["frame_index"] += 1
        document["scores"][2]["frame_index"] += 5
        result, _report = recheck(document)
        self.assertEqual((result["verdict"], result["frame_mismatches"], result["first_frame_mismatch"]), (RC.MISMATCH, 2, 2))
        document = make_document()
        document["scores"][0]["end_row"] += 2
        result, _report = recheck(document)
        self.assertEqual((result["verdict"], result["row_mismatches"], result["first_row_mismatch"]), (RC.MISMATCH, 1, 1))

    def test_the_tolerance_boundary(self):
        document = make_document()
        original = document["scores"][1]["mean"]
        document["scores"][1]["mean"] = original + 0.25
        diff = abs(document["scores"][1]["mean"] - original)
        at_limit, _r = recheck(document, tolerance=diff)
        self.assertEqual((at_limit["verdict"], at_limit["exceeded"]), (RC.MATCH, 0))        # 허용값과 같은 차이는 통과
        over, report = recheck(document, tolerance=diff * 0.999)
        self.assertEqual((over["verdict"], over["exceeded"], over["first_exceeded"]), (RC.MISMATCH, 1, 2))
        self.assertAlmostEqual(over["max_diff"]["mean"], diff)
        self.assertEqual(over["max_diff"]["min"], 0.0)
        self.assertTrue(any(line.startswith("over tolerance") and "windows 1 ; first window 2" in line for line in report))
        small = make_document()
        small["scores"][0]["max"] += 5e-5
        self.assertEqual(recheck(small)[0]["verdict"], RC.MATCH)             # 기본 허용값 1e-4 보다 작은 차이
        small["scores"][0]["max"] += 1e-4
        self.assertEqual(recheck(small)[0]["verdict"], RC.MISMATCH)

    def test_a_changed_score_of_one_window_is_a_mismatch_and_the_last_window_is_shown(self):
        document = make_document()
        document["scores"][3]["mean"] += 1.5
        result, report = recheck(document)
        self.assertEqual((result["verdict"], result["exceeded"], result["first_exceeded"]), (RC.MISMATCH, 1, 4))
        recorded = [line for line in report if line.startswith("last window recorded")][0]
        recomputed = [line for line in report if line.startswith("last window recomputed")][0]
        self.assertIn("창 4", recorded)
        self.assertNotEqual(recorded.split(": ")[1], recomputed.split(": ")[1])

    def test_reference_rows_and_stats_are_compared_when_recorded(self):
        document = make_document()
        document["reference"]["rows"] += 1
        result, report = recheck(document)
        self.assertFalse(result["reference"]["rows_match"])
        self.assertTrue(any(line.startswith("reference rows") and line.endswith("불일치") for line in report))
        document = make_document()
        document["reference"]["means"][0] += 0.01
        result, _report = recheck(document)
        self.assertFalse(result["reference"]["stats_match"])
        self.assertAlmostEqual(result["reference"]["means_diff"], 0.01)
        document = make_document()
        document["reference"] = None
        result, report = recheck(document)
        self.assertIsNone(result["reference"])
        self.assertIn("reference: JSON 에 기준 정보가 없어 비교하지 않음", report)

    def test_time_sec_is_not_used(self):
        document = make_document()
        for item in document["scores"]:
            item["time_sec"] = 99999.0                                       # 쓰지 않는 값
        self.assertEqual(recheck(document)[0]["verdict"], RC.MATCH)


class MainTest(unittest.TestCase):
    def _write(self, folder, document):
        path = os.path.join(folder, "live.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
        return path

    def _main(self, argv, factory=None):
        lines = []
        code = RC.main(argv, scorer_factory=factory, out=lines.append)
        return code, lines

    def test_exit_codes_and_output(self):
        events = []

        class Scorer(SS.FakeScorer):
            def close(self):
                events.append("closed")

        def factory(root):
            events.append(root)
            return Scorer()
        with tempfile.TemporaryDirectory() as folder:
            path = self._write(folder, make_document())
            code, lines = self._main(["--live-json", path, "--ckpt-root", "<CKPT_DIR>"], factory)
            self.assertEqual((code, lines[-2], lines[-1]), (0, "verdict: 일치", RC.NOTE))
            self.assertEqual(events, ["<CKPT_DIR>", "closed"])
            changed = make_document()
            changed["scores"][0]["min"] -= 1.0
            code, lines = self._main(["--live-json", self._write(folder, changed), "--ckpt-root", "x", "--tolerance", "1e-4"], lambda root: SS.FakeScorer())
            self.assertEqual((code, lines[-2]), (1, "verdict: 불일치"))
            code, lines = self._main(["--live-json", os.path.join(folder, "none.json"), "--ckpt-root", "x"], lambda root: self.fail("채점기를 만들면 안 된다"))
            self.assertEqual(code, 2)
            self.assertTrue(lines[0].startswith("live_score_recheck: "))
            code, lines = self._main(["--live-json", path, "--ckpt-root", "x", "--tolerance", "-1"], lambda root: self.fail("채점기를 만들면 안 된다"))
            self.assertEqual(code, 2)

    def test_no_scorer_is_built_when_there_are_no_windows(self):
        entries = base_rows(reference_count=10, post_count=0)
        with tempfile.TemporaryDirectory() as folder:
            path = self._write(folder, make_document(entries))
            code, lines = self._main(["--live-json", path, "--ckpt-root", "x"], lambda root: self.fail("창이 없으면 채점기를 만들지 않는다"))
            self.assertEqual((code, lines[-2]), (1, "verdict: 미확정"))

    def test_required_arguments(self):
        import contextlib
        import io
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                RC.parse_args(["--live-json", "a.json"])
            with self.assertRaises(SystemExit):
                RC.parse_args(["--ckpt-root", "x"])
        self.assertEqual(RC.parse_args(["--live-json", "a.json", "--ckpt-root", "x"]).tolerance, 1e-4)


class ImportTest(unittest.TestCase):
    def test_importing_the_module_loads_neither_tensorflow_cv2_nor_dlib(self):
        code = "import sys; import improve.hmlstm.live_score_recheck; print('tensorflow' in sys.modules, 'cv2' in sys.modules, 'dlib' in sys.modules)"
        env = dict(os.environ, PYTHONPATH=os.path.join(OH.REPO, "src"))
        done = subprocess.run([sys.executable, "-c", code], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, env=env, timeout=120)
        self.assertEqual(done.stdout.strip(), "False False False", done.stderr)

    def test_the_module_writes_no_files_and_saves_no_images(self):
        with open(RC.__file__, encoding="utf-8") as handle:
            source = handle.read()
        for token in ("imwrite", "VideoWriter", "write_json_document", "open(path, \"w\""):
            self.assertNotIn(token, source)


if __name__ == "__main__":
    unittest.main()
