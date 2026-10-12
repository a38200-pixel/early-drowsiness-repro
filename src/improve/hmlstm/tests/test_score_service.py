# score_service 시험: 가짜 점수기와 가짜 자식 프로세스(--fake)만 쓴다(TensorFlow, 체크포인트 없음).
import io
import json
import os
import subprocess
import sys
import unittest

import numpy as np

from improve.hmlstm import live_session as LS
from improve.hmlstm import original_hook as OH
from improve.hmlstm import score_service as SS


def one_window(value=0.5):
    return np.full((30, 4), value).tolist()


def run_serve(lines, scorer=None):
    stdin = io.StringIO("\n".join(lines) + ("\n" if lines else ""))
    stdout, stderr = io.StringIO(), io.StringIO()
    handled = SS.serve(stdin, stdout, stderr, scorer if scorer is not None else SS.FakeScorer())
    return handled, [json.loads(line) for line in stdout.getvalue().splitlines()], stdout.getvalue(), stderr.getvalue()


class ParseRequestTest(unittest.TestCase):
    def test_a_valid_request(self):
        seq, windows = SS.parse_request(json.dumps({"seq": 7, "windows": [one_window(), one_window(1.0)]}))
        self.assertEqual(seq, 7)
        self.assertEqual(windows.shape, (2, 30, 4))

    def test_bad_requests_raise_protocol_error_with_the_seq_when_known(self):
        for line, has_seq in (("not json", False), ("[1, 2]", False), ('{"seq": 1}', False), ('{"windows": []}', False),
                              (json.dumps({"seq": "a", "windows": [[[1, 2]]]}), True), (json.dumps({"seq": 2, "windows": []}), True),
                              (json.dumps({"seq": 3, "windows": [one_window()[:29]]}), True), ('{"seq": 4, "windows": [[[NaN, 0, 0, 0]]]}', True),
                              (json.dumps({"seq": 5, "windows": [[["x"] * 4] * 30]}), True)):
            with self.assertRaises(SS.ProtocolError, msg=line) as caught:
                SS.parse_request(line)
            if has_seq:
                self.assertTrue(hasattr(caught.exception, "seq"), line)

    def test_non_finite_values_are_refused(self):
        windows = one_window()
        windows[0][0] = float("inf")
        with self.assertRaises(SS.ProtocolError):
            SS.parse_request(json.dumps({"seq": 1, "windows": [windows]}, allow_nan=True))


class ServeTest(unittest.TestCase):
    def test_round_trip_with_the_fake_scorer(self):
        handled, responses, raw, _err = run_serve([json.dumps({"seq": 1, "windows": [one_window(1.0)]}),
                                                   json.dumps({"seq": "two", "windows": [one_window(0.0), one_window(2.0)]})])
        self.assertEqual(handled, 2)
        self.assertEqual([r["seq"] for r in responses], [1, "two"])
        first = responses[0]
        self.assertEqual(set(first), {"seq", "mean", "min", "max", "folds"})
        self.assertAlmostEqual(first["mean"][0], 1.3)                       # 가짜 점수기: 창 평균 1.0 + 0.1 * 폴드 번호(1에서 5)의 평균
        self.assertAlmostEqual(first["min"][0], 1.1)
        self.assertAlmostEqual(first["max"][0], 1.5)
        self.assertEqual(len(first["folds"]), 1)
        self.assertEqual(len(first["folds"][0]), 5)
        self.assertEqual(first["folds"][0], sorted(first["folds"][0]))      # 폴드 번호 순
        second = responses[1]
        self.assertEqual((len(second["mean"]), len(second["folds"])), (2, 2))
        self.assertTrue(all(line.endswith("}") for line in raw.splitlines()))

    def test_error_lines_keep_the_service_running(self):
        handled, responses, _raw, _err = run_serve(["not json", json.dumps({"seq": 9, "windows": [[[1, 2]]]}), "",
                                                    json.dumps({"seq": 10, "windows": [one_window()]})])
        self.assertEqual(handled, 3)                                        # 빈 줄은 건너뛴다
        self.assertEqual(responses[0], {"seq": None, "error": responses[0]["error"]})
        self.assertEqual(responses[1]["seq"], 9)
        self.assertIn("error", responses[1])
        self.assertEqual(responses[2]["seq"], 10)
        self.assertNotIn("error", responses[2])

    def test_a_scorer_exception_becomes_an_error_line_with_the_seq(self):
        class Broken(object):
            def score(self, windows):
                raise RuntimeError("model exploded")
        handled, responses, _raw, err = run_serve([json.dumps({"seq": 3, "windows": [one_window()]}), json.dumps({"seq": 4, "windows": [one_window()]})], Broken())
        self.assertEqual([r["seq"] for r in responses], [3, 4])
        self.assertIn("RuntimeError: model exploded", responses[0]["error"])
        self.assertIn("model exploded", err)                                # 오류 문장은 표준 에러에도 남는다

    def test_stdout_carries_only_protocol_lines_even_when_the_scorer_prints(self):
        class Noisy(SS.FakeScorer):
            def score(self, windows):
                print("loading noise from the scorer")
                sys.stdout.write("more noise\n")
                return SS.FakeScorer.score(self, windows)
        handled, responses, raw, err = run_serve([json.dumps({"seq": 1, "windows": [one_window()]})], Noisy())
        self.assertEqual(len(raw.splitlines()), 1)
        self.assertNotIn("noise", raw)
        self.assertIn("loading noise from the scorer", err)
        self.assertEqual(responses[0]["seq"], 1)

    def test_eof_ends_the_loop(self):
        handled, responses, raw, err = run_serve([])
        self.assertEqual((handled, responses, raw), (0, [], ""))


class MainTest(unittest.TestCase):
    def test_ready_line_then_responses_and_exit_code_zero(self):
        stdin = io.StringIO(json.dumps({"seq": 1, "windows": [one_window()]}) + "\n")
        stdout, stderr = io.StringIO(), io.StringIO()
        code = SS.main(["--fake"], stdin, stdout, stderr)
        lines = [json.loads(line) for line in stdout.getvalue().splitlines()]
        self.assertEqual(code, 0)
        self.assertEqual(lines[0], {"ready": True, "folds": [1, 2, 3, 4, 5]})
        self.assertEqual(lines[1]["seq"], 1)
        self.assertEqual(len(lines), 2)

    def test_missing_checkpoint_root_is_an_error_line(self):
        stdout = io.StringIO()
        code = SS.main([], io.StringIO(""), stdout, io.StringIO())
        self.assertEqual(code, 2)
        self.assertIn("--ckpt-root", json.loads(stdout.getvalue())["error"])

    def test_a_scorer_that_cannot_be_built_ends_with_an_error_line_and_code_one(self):
        from unittest import mock
        stdout, stderr = io.StringIO(), io.StringIO()
        with mock.patch.object(SS.SC, "EnsembleScorer", side_effect=FileNotFoundError("체크포인트를 찾을 수 없습니다: fold1")):      # 실제 복원 없이 실패를 흉내 낸다
            code = SS.main(["--ckpt-root", os.path.join(OH.REPO, "no_such_checkpoint_root")], io.StringIO(""), stdout, stderr)
        self.assertEqual(code, 1)
        self.assertIn("체크포인트를 찾을 수 없습니다", stderr.getvalue())
        lines = [json.loads(line) for line in stdout.getvalue().splitlines()]
        self.assertEqual(len(lines), 1)
        self.assertIn("error", lines[0])

    def test_parse_args(self):
        arguments = SS.parse_args(["--ckpt-root", "x", "--seed", "2", "--stats", "s.json"])
        self.assertEqual((arguments.ckpt_root, arguments.seed, arguments.stats, arguments.fake), ("x", 2, "s.json", False))
        self.assertEqual(SS.parse_args(["--fake"]).seed, 1)


class ChildProcessTest(unittest.TestCase):
    def test_fake_child_round_trip_and_clean_stdout(self):
        env = dict(os.environ, PYTHONPATH=os.path.join(OH.REPO, "src"), PYTHONIOENCODING="utf-8")
        requests = "\n".join([json.dumps({"seq": 1, "windows": [one_window(0.0)]}), "garbage", json.dumps({"seq": 2, "windows": [one_window(1.0)]})]) + "\n"
        done = subprocess.run([sys.executable, "-m", "improve.hmlstm.score_service", "--fake"], input=requests, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              universal_newlines=True, env=env, timeout=120)
        self.assertEqual(done.returncode, 0, done.stderr)                   # 표준 입력이 닫히면 정상 종료
        lines = [json.loads(line) for line in done.stdout.splitlines()]     # 모든 줄이 JSON 이다(프로토콜 줄만 있다)
        self.assertEqual(lines[0], {"ready": True, "folds": [1, 2, 3, 4, 5]})
        self.assertEqual([line.get("seq") for line in lines[1:]], [1, None, 2])
        self.assertIn("error", lines[2])
        self.assertAlmostEqual(lines[3]["mean"][0], 1.3)


class SessionWithServiceTest(unittest.TestCase):
    def test_jobs_go_through_the_service_and_come_back_into_the_gauge(self):
        from improve.hmlstm.tests.test_live_session import ready_session
        session, rows, faces, jobs = ready_session(reference_count=40, post_count=34)
        self.assertEqual(len(jobs), 3)
        handled, responses, _raw, _err = run_serve([json.dumps(job.message()) for job in jobs])
        self.assertEqual(handled, 3)
        for response in responses:
            session.apply_response(response)
        self.assertEqual(session.pending(), [])
        self.assertEqual(session.phase, LS.PHASE_SCORING)
        state = session.gauge_state(jobs[-1].frame_index)
        self.assertEqual(state["window"], 3)
        self.assertEqual(state["stage"], "value")
        self.assertLessEqual(state["min"], state["mean"])
        self.assertLessEqual(state["mean"], state["max"])
        expected = float(np.mean(jobs[-1].window) + 0.1 * 3)                 # 가짜 점수기의 평균
        self.assertAlmostEqual(state["mean"], expected)


if __name__ == "__main__":
    unittest.main()
