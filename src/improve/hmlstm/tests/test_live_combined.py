# live_combined 시험: 가짜 시계, 가짜 스트림, 가짜 자식 프로세스(--fake, python -c)만 쓴다. 웹캠, 실제 창, TensorFlow 는 쓰지 않는다(cv2 의 창 함수는 시험에서 바꿔 끼운다).
import ast
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

import numpy as np

from improve.hmlstm import gauge as GAUGE
from improve.hmlstm import live_combined as LC
from improve.hmlstm import live_session as LS
from improve.hmlstm import original_hook as OH
from improve.hmlstm import webcam_hook_speed as WH
from improve.hmlstm.tests.test_live_session import ready_session, synthetic_rows


class Clock(object):
    def __init__(self, start=100.0):
        self.now = start

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class FakeStream(object):
    """read 한 번에 dt 초가 지나는 가짜 스트림(원본의 stream.read 자리)."""

    def __init__(self, clock, dt=0.1):
        self.clock, self.dt = clock, dt
        self.reads = 0

    def read(self):
        self.reads += 1
        self.clock.advance(self.dt)
        return True, np.zeros((4, 4, 3), dtype=np.uint8)


class PlainStream(object):
    def read(self):
        return True, np.zeros((2, 2, 3), dtype=np.uint8)


def iterate(hook, stream, ear=None, source="detected", faces=0, row=None, left=None, right=None):
    """원본 루프의 한 반복과 같은 호출 순서: 읽기 -> (EAR 콜백) -> (행 기록) -> tick. tick 의 반환값을 돌려준다."""
    hook.read(stream)
    if ear is not None:
        hook.ear(ear, source, faces, left, right)
    if row is not None:
        hook.on_row(row)
    return hook.tick()


def make_hook(**kwargs):
    clock = Clock()
    hook = LC.LiveHook(clock=clock, **kwargs)
    return hook, clock, FakeStream(clock)


def wait_for(condition, seconds=15.0):
    deadline = time.time() + seconds
    while time.time() < deadline:
        if condition():
            return True
        time.sleep(0.02)
    return condition()


class SourceTest(unittest.TestCase):
    def test_all_replacements_pass_the_count_check_and_only_the_expected_lines_change(self):
        original = OH.read_original_source()
        source = LC.build_source(0, "PRED.dat", "SVM.sav", "none", original)
        base = OH.apply_replacements(original, OH.build_replacements("", "PRED.dat", "SVM.sav", "none", True))
        extra = sorted(set(OH.changed_line_numbers(original, source)) - set(OH.changed_line_numbers(original, base)))
        self.assertEqual(len(source.split("\n")), len(original.split("\n")))
        self.assertIn(421, extra)
        self.assertIn(506, extra)
        self.assertIn(352, extra)                                          # 읽기 래퍼
        self.assertIn(334, extra)                                          # Tk 숨김
        self.assertIn(570, extra)                                          # destroyAllWindows 무력화
        self.assertEqual(source.count("path = 0"), 1)

    def test_callbacks_pass_eye_points_and_the_frame_counter(self):
        source = LC.build_source(0, "PRED.dat", "SVM.sav", "none")
        self.assertEqual(source.count("_hook_ear(ear, 'detected', number_of_frames, leftEye, rightEye)"), 1)
        self.assertEqual(source.count("_hook_ear(ear, 'tracked', number_of_frames, p1, p2)"), 1)
        self.assertEqual(source.count("_hook_read(stream)"), 1)
        self.assertEqual(source.count("top = tk.Tk(); top.withdraw()"), 1)
        self.assertNotIn("cv2.destroyAllWindows()", source)                # 창은 그리기 스레드가 닫는다
        self.assertNotIn("cv2.imshow", source)

    def test_rotation_option_reaches_the_original_hook(self):
        self.assertIn("cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)", LC.build_source(0, "PRED.dat", "SVM.sav", "cw90"))

    def test_a_changed_original_is_refused(self):
        original = OH.read_original_source()
        for old in (LC.EC.DETECTED_TARGET, LC.EC.TRACKED_TARGET, LC.WH.READ_TARGET, LC.WH.DESTROY_TARGET, LC.WH.TK_TARGET):
            with self.assertRaises(OH.ReplacementError, msg=old):
                LC.build_source(0, "PRED.dat", "SVM.sav", "none", original.replace(old, "pass  # changed"))

    def test_row_timing_and_eye_variables_follow_the_source_order(self):
        tree = ast.parse(OH.read_original_source())
        assigns = {}
        calls = []

        def visit(node, stack):
            if isinstance(node, ast.FunctionDef):
                stack = stack + [node.name]
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        assigns.setdefault(target.id, []).append((node.lineno, tuple(stack)))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "savetxt":
                calls.append((node.lineno, tuple(stack)))
            for child in ast.iter_child_nodes(node):
                visit(child, stack)
        visit(tree, [])
        function = ("blink_detector",)
        self.assertIn((380, function), assigns["number_of_frames"])        # 얼굴 검출 분기에서 증가
        self.assertIn((421, function), assigns["ear"])
        self.assertIn((415, function), assigns["leftEye"])                 # 421행 앞에서 배정되어 읽을 수 있다
        self.assertIn((416, function), assigns["rightEye"])
        self.assertIn((455, function), assigns["BLINK_FRAME_FREQ"])
        self.assertTrue(any(line < 506 and stack == function for line, stack in assigns["p1"]))      # 추적 분기: 506행 앞에서 p1, p2 가 정해진다
        self.assertTrue(any(line < 506 and stack == function for line, stack in assigns["p2"]))
        self.assertIn((506, function), assigns["ear"])
        self.assertTrue(any(line == 464 for line, _stack in calls))        # 얼굴 분기의 행 기록
        self.assertLess(380, 421)
        self.assertLess(421, 455)
        self.assertLess(455, 464)


class HookTest(unittest.TestCase):
    def test_read_returns_false_after_the_stop_flag_without_touching_the_stream(self):
        hook, clock, stream = make_hook()
        self.assertEqual(hook.read(stream)[0], True)
        hook.stop_event.set()
        self.assertEqual(hook.read(stream), (False, None))
        self.assertEqual(stream.reads, 1)

    def test_the_first_frame_is_time_zero(self):
        hook, clock, stream = make_hook()
        for expected in (0.0, 0.1, 0.2):
            hook.read(stream)
            self.assertAlmostEqual(hook.current_t, expected)

    def test_a_failed_read_clears_the_frame_and_keeps_the_time(self):
        hook, clock, stream = make_hook()
        hook.read(stream)

        class Dead(object):
            def read(self):
                return False, None
        self.assertEqual(hook.read(Dead()), (False, None))
        self.assertIsNone(hook.last_frame)

    def test_rules_are_updated_exactly_once_per_iteration(self):
        hook, clock, stream = make_hook()
        calls = []
        original = hook.rules.update
        hook.rules.update = lambda t, ear, source: (calls.append((round(t, 6), ear, source)), original(t, ear, source))[1]
        iterate(hook, stream, ear=0.30, source="detected", faces=1)
        iterate(hook, stream)                                              # 콜백이 없던 반복
        iterate(hook, stream, ear=0.22, source="tracked", faces=1)
        iterate(hook, stream, ear=0.28, source="detected", faces=2)
        self.assertEqual(len(calls), 4)
        self.assertEqual(calls, [(0.0, 0.30, "detected"), (0.1, None, "none"), (0.2, 0.22, "tracked"), (0.3, 0.28, "detected")])
        self.assertEqual(hook.rule_updates, 4)

    def test_a_tracked_frame_is_handled_like_a_missing_face_by_default(self):
        tracked, _c, stream_a = make_hook()
        missing, _c2, stream_b = make_hook()
        for _ in range(6):
            iterate(tracked, stream_a, ear=0.10, source="tracked", faces=1)
            iterate(missing, stream_b)
        self.assertEqual(tracked.rules.face_status(), missing.rules.face_status())
        self.assertEqual(tracked.rules.closed_length_s(0.15), 0.0)         # 추적 EAR 로는 닫힘 구간이 시작되지 않는다
        self.assertEqual(tracked.rules.alarm_count(0.15), 0)

    def test_a_closed_run_gives_an_alarm_and_the_latch_and_reset(self):
        hook, clock, stream = make_hook()
        for number in range(40):
            iterate(hook, stream, ear=0.10, source="detected", faces=number + 1)
        self.assertEqual(hook.rules.alarm_count(0.15), 1)
        self.assertTrue(hook.latch.active(0.15, hook.current_t))
        rules_snapshot, latch_snapshot = LC.snapshot_rules(hook.rules, hook.latch, hook.current_t)
        self.assertEqual(rules_snapshot.alarm_count(0.15), 1)
        self.assertTrue(latch_snapshot.active(0.15, 0))
        self.assertGreater(rules_snapshot.closed_length_s(0.15), 3.0)
        self.assertEqual(rules_snapshot.primary, 0.15)
        hook.reset_request.set()
        iterate(hook, stream, ear=0.10, source="detected", faces=41)
        self.assertFalse(hook.latch.active(0.15, hook.current_t))
        self.assertEqual(hook.rules.alarm_count(0.15), 1)                  # 표시만 끄고 횟수는 그대로

    def test_session_time_advances_on_every_tick(self):
        hook, clock, stream = make_hook()
        seen = []
        original = hook.session.advance
        hook.session.advance = lambda t: (seen.append(round(t, 6)), original(t))[1]
        for _ in range(5):
            iterate(hook, stream)
        self.assertEqual(seen, [0.0, 0.1, 0.2, 0.3, 0.4])

    def test_tick_returns_a_stop_code_on_the_stop_flag_and_after_the_duration(self):
        hook, clock, stream = make_hook(duration_s=0.35)
        codes = [iterate(hook, stream) for _ in range(5)]
        self.assertEqual(codes[:4], [OH.NO_KEY] * 4)
        self.assertEqual(codes[4], 0)
        self.assertTrue(hook.stop_event.is_set())
        other, _c, stream2 = make_hook()
        other.stop_event.set()
        self.assertEqual(iterate(other, stream2), 0)

    def test_callback_errors_are_swallowed_and_counted(self):
        hook, clock, stream = make_hook()
        hook.ear("bad", "detected", 1, None, None)
        hook.ear(0.3, "detected", "x", None, None)
        hook.on_row([1.0, 2.0])
        self.assertEqual(hook.errors, 3)
        self.assertEqual(iterate(hook, stream), OH.NO_KEY)                 # 오류 뒤에도 루프는 계속된다

    def test_rows_use_the_tick_count_the_row_time_and_the_latest_frame_counter(self):
        hook, clock, stream = make_hook()
        rows, faces = synthetic_rows(5)
        iterate(hook, stream, ear=0.3, source="detected", faces=faces[0], row=rows[0])
        iterate(hook, stream)
        iterate(hook, stream, ear=0.3, source="detected", faces=faces[1], row=rows[1])
        iterate(hook, stream, row=rows[2])                                  # 얼굴 없음 분기에서 나온 행: number_of_frames 는 직전 값
        self.assertEqual([r["frame_index"] for r in hook.rows], [0, 2, 3])
        self.assertEqual([r["face_frames"] for r in hook.rows], [faces[0], faces[1], faces[1]])
        self.assertEqual([round(r["t"], 6) for r in hook.rows], [0.0, 0.2, 0.3])
        self.assertEqual(len(hook.row_times), 3)

    def test_self_check_passes_for_consistent_rows_and_counts_violations(self):
        hook, clock, stream = make_hook()
        rows, faces = synthetic_rows(6)
        for index in range(6):
            row = list(rows[index])
            if index in (2, 4):
                row[1] += 0.01                                              # 일부러 어긋나게 한 Freq
            iterate(hook, stream, ear=0.3, source="detected", faces=faces[index], row=row)
        self.assertEqual((hook.check.checked, hook.check.violations, hook.check.first_violation), (6, 2, 2))
        clean, _c, stream2 = make_hook()
        for index in range(6):
            iterate(clean, stream2, ear=0.3, source="detected", faces=faces[index], row=rows[index])
        self.assertEqual((clean.check.checked, clean.check.violations, clean.check.first_violation), (6, 0, None))

    def test_a_zero_frame_counter_is_a_violation_and_the_session_recovers_it_from_the_row(self):
        hook, clock, stream = make_hook()
        rows, _faces = synthetic_rows(3)
        iterate(hook, stream, row=rows[0])
        self.assertEqual((hook.check.checked, hook.check.violations), (1, 1))
        self.assertEqual(len(hook.session.ref_rows), 1)

    def test_rows_go_to_the_session_and_jobs_to_the_submit_function(self):
        submitted = []
        hook, clock, stream = make_hook(submit=submitted.append)
        rows, faces = synthetic_rows(80)
        for index in range(40):
            clock.advance(6.0)
            iterate(hook, stream, ear=0.3, source="detected", faces=faces[index], row=rows[index])
        clock.advance(80.0)
        iterate(hook, stream, ear=0.3, source="detected", faces=faces[40], row=rows[40])
        self.assertEqual(hook.session.phase, LS.PHASE_COLLECTING)         # 300초가 지난 첫 행부터 기준 이후
        for index in range(41, 70):
            iterate(hook, stream, ear=0.3, source="detected", faces=faces[index], row=rows[index])
        self.assertEqual(len(submitted), 1)                                 # 기준 이후 30번째 행에서 첫 창
        self.assertEqual(submitted[0]["seq"], 1)
        self.assertEqual(np.array(submitted[0]["windows"]).shape, (1, 30, 4))
        self.assertEqual(hook.check.violations, 0)

    def test_posts_are_limited_to_the_preview_rate_and_carry_a_frame_copy(self):
        mailbox = WH.Mailbox()
        hook, clock, stream = make_hook(mailbox=mailbox, preview_hz=3.0)
        for _ in range(31):
            iterate(hook, stream, ear=0.3, source="detected", faces=1, left=np.zeros((6, 2), dtype=np.int32), right=np.ones((6, 2), dtype=np.int32))
        self.assertEqual(mailbox.put_count, 8)                              # 3.0초 동안 3 Hz: 4프레임마다
        post = mailbox.take()
        self.assertIsNot(post.frame, hook.last_frame)                       # 게시할 때 복사한 프레임
        self.assertEqual(post.left.shape, (6, 2))
        self.assertEqual(post.source, "detected")
        self.assertAlmostEqual(post.ear, 0.3)

    def test_the_mailbox_keeps_only_the_latest_post(self):
        mailbox = WH.Mailbox()
        hook, clock, stream = make_hook(mailbox=mailbox, preview_hz=30.0)
        for _ in range(10):
            iterate(hook, stream, ear=0.3, source="detected", faces=1)
        self.assertGreaterEqual(mailbox.discarded, 1)
        self.assertEqual(mailbox.take().frame_index, 9)

    def test_gauge_info_is_read_under_the_lock(self):
        hook, clock, stream = make_hook()
        iterate(hook, stream)
        info = hook.live_info(0, 0.0)
        self.assertEqual(info["phase"], LS.PHASE_REFERENCE)
        self.assertEqual(info["score_state"], LC.SCORE_DISABLED)
        self.assertFalse(hook.session_lock.locked())


class GaugeViewTest(unittest.TestCase):
    def _info(self, phase=LS.PHASE_REFERENCE, score_state=LC.SCORE_READY, elapsed=0.0, reference_rows=0, windows_scored=0, pending=(), status="기준 수집 중 0초", state=None):
        return {"phase": phase, "score_state": score_state, "elapsed": elapsed, "reference_rows": reference_rows, "windows_scored": windows_scored,
                "pending": list(pending), "status_text": status, "gauge_state": state}

    def test_reference_phase_shows_progress_row_count_and_the_caution(self):
        view = LC.gauge_view(self._info(elapsed=150.0, reference_rows=24, status="기준 수집 중 150초"))
        self.assertEqual(view[0], "text")
        self.assertEqual(view[1][0], GAUGE.LABEL)
        self.assertEqual(view[1][1], "기준 수집 중 150초 / 300초, 행 24개")
        self.assertEqual(view[1][2], LC.REFERENCE_NOTE)
        self.assertAlmostEqual(view[2], 0.5)

    def test_calibration_short(self):
        view = LC.gauge_view(self._info(phase=LS.PHASE_SHORT, status="보정 부족"))
        self.assertEqual(view, ("text", [GAUGE.LABEL, "보정 부족"], None))

    def test_loading_disabled_pending_and_stopped(self):
        base = dict(phase=LS.PHASE_COLLECTING, status="수집 중 12/30")
        self.assertEqual(LC.gauge_view(self._info(score_state=LC.SCORE_LOADING, **base))[1][1], "점수 프로세스 로딩 중")
        self.assertEqual(LC.gauge_view(self._info(score_state=LC.SCORE_DISABLED, **base))[1][1], "점수 없음(--no-score)")
        waiting = LC.gauge_view(self._info(score_state=LC.SCORE_READY, pending=[1], phase=LS.PHASE_COLLECTING, status="수집 중 30/30"))
        self.assertEqual(waiting[1][1], "점수 계산 대기 중")
        stopped = LC.gauge_view(self._info(score_state=LC.SCORE_STOPPED, **base))
        self.assertEqual(stopped, ("text", [GAUGE.LABEL, "점수 프로세스 중단"], None))
        self.assertEqual(LC.gauge_view(self._info(score_state=LC.SCORE_STOPPED))[1][1], "점수 프로세스 중단")       # 기준 구간 중에도

    def test_collecting_and_scoring_use_the_gauge_module_text_as_is(self):
        session, rows, faces, jobs = ready_session(reference_count=40, post_count=34)
        state = session.gauge_state(session.post_frame_indices[10])
        view = LC.gauge_view(self._info(phase=LS.PHASE_COLLECTING, windows_scored=0, state=state, status=session.status_text()))
        self.assertEqual(view[0], "gauge")
        self.assertEqual(GAUGE.format_gauge_text(view[1]).split("\n")[0], GAUGE.LABEL)
        session.apply_score(jobs[0].seq, 5.0, 4.0, 6.0, [4.0, 5.0, 5.0, 5.0, 6.0])
        scored = session.gauge_state(jobs[0].frame_index)
        view = LC.gauge_view(self._info(phase=LS.PHASE_SCORING, windows_scored=1, state=scored, status="점수 있음"))
        self.assertEqual(view[0], "gauge")
        self.assertEqual(GAUGE.format_gauge_text(view[1]).split("\n")[1], "5.0 [4.0-6.0]")


class ScoreClientTest(unittest.TestCase):
    def _fake_command(self):
        return [sys.executable, "-m", "improve.hmlstm.score_service", "--fake"]

    def test_round_trip_with_the_fake_service_reaches_the_gauge(self):
        session, rows, faces, jobs = ready_session(reference_count=40, post_count=34)
        lock = threading.Lock()

        def on_response(message):
            with lock:
                session.apply_response(message)
        client = LC.ScoreClient(self._fake_command(), on_response, env=LC.child_environment())
        self.assertEqual(client.state(), LC.SCORE_LOADING)
        client.start()
        try:
            self.assertTrue(wait_for(lambda: client.state() == LC.SCORE_READY))
            self.assertEqual(client.folds, [1, 2, 3, 4, 5])
            for job in jobs:
                self.assertTrue(client.submit(job.message()))
            self.assertTrue(wait_for(lambda: client.received == len(jobs)))
            with lock:
                state = session.gauge_state(jobs[-1].frame_index)
            self.assertEqual((state["stage"], state["window"]), (GAUGE.STAGE_VALUE, 3))
            self.assertEqual(session.phase, LS.PHASE_SCORING)
            self.assertEqual(client.sent, len(jobs))
        finally:
            code = client.close()
        self.assertEqual(code, 0)                                           # 표준 입력이 닫히면 자식이 스스로 끝난다

    def test_a_child_that_dies_is_reported_and_submit_stops_queueing(self):
        client = LC.ScoreClient([sys.executable, "-c", "pass"], lambda message: None, env=LC.child_environment())
        client.start()
        try:
            self.assertTrue(wait_for(lambda: client.state() == LC.SCORE_STOPPED))
            self.assertFalse(client.submit({"seq": 1, "windows": []}))
        finally:
            client.close()

    def test_the_hook_loop_keeps_running_after_the_score_process_dies(self):
        client = LC.ScoreClient([sys.executable, "-c", "pass"], lambda message: None, env=LC.child_environment())
        hook, clock, stream = make_hook(submit=client.submit, score_state=client.state)
        client.start()
        try:
            self.assertTrue(wait_for(lambda: client.state() == LC.SCORE_STOPPED))
            for _ in range(10):
                self.assertEqual(iterate(hook, stream, ear=0.3, source="detected", faces=1), OH.NO_KEY)
            self.assertEqual(hook.rule_updates, 10)
            self.assertEqual(LC.gauge_view(hook.live_info(0, 0.0)), ("text", [GAUGE.LABEL, "점수 프로세스 중단"], None))
        finally:
            client.close()

    def test_submit_never_blocks_even_if_the_child_does_not_read(self):
        client = LC.ScoreClient([sys.executable, "-c", "import time; time.sleep(60)"], lambda message: None, env=LC.child_environment())
        client.start()
        big = {"seq": 1, "windows": [np.zeros((30, 4)).tolist()] * 20}
        begun = time.time()
        try:
            for _ in range(300):
                client.submit(big)
            self.assertLess(time.time() - begun, 2.0)                       # 파이프가 차도 호출은 큐에 넣고 바로 돌아온다
        finally:
            started = time.time()
            client.close(timeout=1.0)
            self.assertLess(time.time() - started, 10.0)
        self.assertIsNotNone(client.exit_code)                              # 읽지 않던 자식도 정리된다

    def test_stderr_of_the_child_is_drained_so_the_child_never_blocks(self):
        script = ("import sys; sys.stderr.write('x' * 3000000); sys.stderr.flush(); "
                  "print('{\"ready\": true, \"folds\": [1]}'); sys.stdout.flush(); sys.stdin.read()")
        client = LC.ScoreClient([sys.executable, "-c", script], lambda message: None, env=LC.child_environment())
        client.start()
        try:
            self.assertTrue(wait_for(client.ready.is_set))
        finally:
            self.assertEqual(client.close(), 0)

    def test_error_lines_and_garbage_are_counted_not_applied(self):
        script = ("import sys; print('garbage'); print('{\"seq\": 1, \"error\": \"boom\"}'); print('{\"ready\": true, \"folds\": [1]}'); sys.stdout.flush(); sys.stdin.read()")
        applied = []
        client = LC.ScoreClient([sys.executable, "-c", script], applied.append, env=LC.child_environment())
        client.start()
        try:
            self.assertTrue(wait_for(client.ready.is_set))
            self.assertEqual((client.error_lines, client.last_error, applied), (2, "boom", []))
        finally:
            client.close()

    def test_the_child_is_started_with_below_normal_priority_on_windows(self):
        captured = {}

        def popen(command, **kwargs):
            captured.update(kwargs)
            raise RuntimeError("stop here")
        client = LC.ScoreClient(["x"], lambda message: None, popen=popen)
        with self.assertRaises(RuntimeError):
            client.start()
        if os.name == "nt":
            self.assertEqual(captured["creationflags"], getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", LC.BELOW_NORMAL_PRIORITY_CLASS))
        else:
            self.assertNotIn("creationflags", captured)
        self.assertEqual(captured["stderr"], subprocess.PIPE)
        self.assertEqual(LC.BELOW_NORMAL_PRIORITY_CLASS, 0x4000)

    def test_command_and_environment(self):
        command = LC.score_command("<CKPT_DIR>")
        self.assertEqual(command[1:], ["-m", "improve.hmlstm.score_service", "--ckpt-root", "<CKPT_DIR>"])
        self.assertIn(os.path.join(OH.REPO, "src"), LC.child_environment()["PYTHONPATH"].split(os.pathsep))


class DrawingTest(unittest.TestCase):
    def _post(self, hook, ear=0.3, left=True):
        rules_snapshot, latch_snapshot = LC.snapshot_rules(hook.rules, hook.latch, 1.0)
        points = (np.array([[100, 100], [140, 100], [150, 110], [140, 120], [100, 120], [95, 110]]) if left else None)
        return LC.Post(np.zeros((480, 640, 3), dtype=np.uint8), points, points, ear, "detected", 0.2, rules_snapshot, latch_snapshot, 7, 1.0)

    @contextlib.contextmanager
    def _factory_run(self, hook, keys, visible=(1,), duration=None, rotate="none"):
        """cv2 의 창 함수를 바꿔 끼운 채로 (draw, 보인 화면 목록, painter 에 간 문구 목록)을 준다. with 블록 안에서만 유효하다."""
        cv2 = __import__("cv2")
        shown, recorded = [], []
        key_iter, visible_iter = iter(keys), iter(visible)
        painter = lambda strip, items: recorded.append([item[0] for item in items])
        with mock.patch.object(cv2, "imshow", lambda name, image: shown.append((name, image.shape))), \
                mock.patch.object(cv2, "waitKey", lambda delay: next(key_iter, 255)), \
                mock.patch.object(cv2, "getWindowProperty", lambda name, prop: next(visible_iter, 1)), \
                mock.patch("improve.rules.replay_viewer.make_painter", lambda c, n: painter):
            draw, close = LC.make_live_drawer(hook, rotate, duration)()
            self.assertIs(close, cv2.destroyAllWindows)
            yield draw, shown, recorded

    def _need_cv2(self):
        try:
            import cv2  # noqa: F401
        except ImportError:
            self.skipTest("cv2 를 가져올 수 없어 건너뜁니다")

    def test_the_screen_is_rules_plus_time_row_plus_gauge_strip(self):
        self._need_cv2()
        from improve.rules import replay_viewer as RV
        hook, clock, stream = make_hook()
        with self._factory_run(hook, keys=[255, 255], duration=60.0) as (draw, shown, recorded):
            draw(self._post(hook))
        self.assertEqual(shown, [(LC.WINDOW_NAME, (110 + 28 + RV.DEFAULT_VIDEO_HEIGHT + RV.TIME_H + RV.GAUGE_H, RV.DISPLAY_WIDTH, 3))])
        self.assertEqual(recorded[0][0], "00:01.0 / 01:00.0")                # 시간 줄: 경과 시간 / 전체
        self.assertIn("frame 7", recorded[0])
        texts = [text for items in recorded for text in items]
        self.assertIn(GAUGE.LABEL, texts)
        self.assertIn(LC.REFERENCE_NOTE, texts)                            # 기준 구간 동안의 주의 문구
        self.assertTrue(any(text.startswith("기준 수집 중") and "/ 300초" in text for text in texts))

    def test_keys_set_the_flags(self):
        self._need_cv2()
        for key, check in ((ord("q"), lambda h: h.stop_event.is_set()), (27, lambda h: h.stop_event.is_set()), (ord("r"), lambda h: h.reset_request.is_set()),
                           (ord("m"), lambda h: h.mute_request.is_set())):
            hook, clock, stream = make_hook()
            with self._factory_run(hook, keys=[key]) as (draw, _shown, _recorded):
                draw(self._post(hook))
            self.assertTrue(check(hook), key)
        quiet, _c, _s = make_hook()
        with self._factory_run(quiet, keys=[255]) as (draw, _shown, _recorded):
            draw(self._post(quiet))
        self.assertFalse(quiet.stop_event.is_set())

    def test_closing_the_window_stops_the_loop_only_after_it_was_seen_visible(self):
        self._need_cv2()
        hook, clock, stream = make_hook()
        with self._factory_run(hook, keys=[255, 255, 255], visible=(-1, 1, 0)) as (draw, _shown, _recorded):
            draw(self._post(hook))
            self.assertFalse(hook.stop_event.is_set())                      # 아직 보인 적이 없으면 닫힘으로 보지 않는다
            draw(self._post(hook))
            self.assertFalse(hook.stop_event.is_set())
            draw(self._post(hook))
            self.assertTrue(hook.stop_event.is_set())

    def test_rotation_and_missing_eyes_are_drawn_without_error(self):
        self._need_cv2()
        hook, clock, stream = make_hook()
        with self._factory_run(hook, keys=[255, 255], rotate="ccw90") as (draw, shown, _recorded):
            draw(self._post(hook))
            draw(self._post(hook, ear=None, left=False))
        self.assertEqual(len(shown), 2)

    def test_the_font_error_sets_the_stop_flag_and_is_raised(self):
        self._need_cv2()
        hook, clock, stream = make_hook()

        def no_font(c, n):
            raise ValueError("no korean font")
        with mock.patch("improve.rules.replay_viewer.make_painter", no_font):
            with self.assertRaises(ValueError):
                LC.make_live_drawer(hook, "none", None)()
        self.assertTrue(hook.stop_event.is_set())

    def test_the_draw_thread_keeps_an_error_from_its_factory(self):
        def factory():
            raise ValueError("no font")
        thread = WH.DrawThread(WH.Mailbox(), 30.0, factory)
        thread.start()
        thread.join(2.0)
        self.assertIn("no font", thread.error)

    def test_the_draw_thread_draws_only_new_posts_at_the_poll_rate(self):
        mailbox = WH.Mailbox()
        drawn = []
        thread = WH.DrawThread(mailbox, 30.0, lambda: (drawn.append, lambda: None))
        thread.start()
        try:
            for number in range(5):
                mailbox.put(number)
                time.sleep(0.12)
        finally:
            thread.stop()
        self.assertEqual(drawn, [0, 1, 2, 3, 4])                           # 새 게시마다 한 번, 같은 것을 다시 그리지 않는다

    def test_text_strips_have_the_fixed_size_and_the_painter_gets_the_phase_lines(self):
        self._need_cv2()
        import cv2
        from improve.rules import replay_viewer as RV
        recorded = []
        strip = LC.draw_text_strip(cv2, np, 640, ["a", "b", "c"], 0.5, lambda image, items: recorded.append(items), RV)
        self.assertEqual(strip.shape, (RV.GAUGE_H, 640, 3))
        self.assertEqual([item[0] for item in recorded[0]], ["a", "b", "c"])
        time_strip = LC.draw_live_time_strip(cv2, np, 640, 61.25, 1837, None, lambda image, items: recorded.append(items), RV)
        self.assertEqual(time_strip.shape, (RV.TIME_H, 640, 3))
        self.assertEqual([item[0] for item in recorded[1]], ["01:01.3", "frame 1837"])


class ReporterAndMainTest(unittest.TestCase):
    def _reporter(self, hook, client=None):
        lines = []
        return LC.LiveReporter(hook, None, None, client, out=lines.append, flush=lambda: None), lines

    def _run_some(self, hook, stream, count=20):
        for _ in range(count):
            iterate(hook, stream, ear=0.3, source="detected", faces=1)

    def test_the_summary_has_all_the_lines_and_the_fixed_closing_sentence(self):
        hook, clock, stream = make_hook()
        self._run_some(hook, stream)
        reporter, lines = self._reporter(hook)
        self.assertIsNotNone(reporter.emit("finished"))
        text = "\n".join(lines)
        for fragment in ("status: finished", "frames: 20", "mean fps:", "frame gap median", "processing median", "frame budget", "preview mode:", "blink rows confirmed: 0",
                         "live session: phase reference", "score process: disabled", "self check", "rule alarms: 0.15: 0", "swallowed exceptions: 0"):
            self.assertIn(fragment, text)
        self.assertEqual(lines[-1], LC.NOTE)
        self.assertEqual(reporter.exit_code, 0)

    def test_the_summary_is_printed_once_and_the_watchdog_adds_a_line(self):
        hook, clock, stream = make_hook()
        self._run_some(hook, stream, 5)
        reporter, lines = self._reporter(hook)
        reporter.emit("finished")
        count = len(lines)
        self.assertIsNone(reporter.emit("finished"))
        self.assertEqual(reporter.fire_watchdog(), 0)
        self.assertEqual(len(lines), count + 1)
        other, other_lines = self._reporter(hook)
        self.assertEqual(other.fire_watchdog(), 1)
        self.assertTrue(any(line.startswith("watchdog exit") for line in other_lines))
        self.assertEqual(other_lines[-1], LC.NOTE)

    def test_no_frames_gives_exit_code_one(self):
        hook, clock, stream = make_hook()
        reporter, _lines = self._reporter(hook)
        reporter.emit("finished")
        self.assertEqual(reporter.exit_code, 1)

    def test_the_run_document_has_numbers_only(self):
        hook, clock, stream = make_hook()
        rows, faces = synthetic_rows(3)
        for index in range(3):
            iterate(hook, stream, ear=0.10, source="detected", faces=faces[index], row=rows[index])
        reporter, _lines = self._reporter(hook)
        summary = reporter.emit("finished")
        document = LC.build_run_document(hook, summary, None, "finished")
        LC.validate_run_document(document)
        self.assertEqual(len(document["rows"]), 3)
        self.assertIsNone(document["score_process"])
        self.assertEqual(document["self_check"]["checked"], 3)
        json.dumps(document, allow_nan=False)                               # 유한한 숫자와 문자열, 목록뿐이다
        self.assertNotIn("frames_data", document)
        with self.assertRaises(LC.LiveError):
            LC.validate_run_document({"kind": "other"})

    def _document_parts(self):
        hook, clock, stream = make_hook()
        self._run_some(hook, stream, 3)
        reporter, _lines = self._reporter(hook)
        return hook, reporter.emit("finished"), None, "finished"

    def test_out_json_refuses_inside_the_repository_and_existing_files(self):
        document = LC.build_run_document(*self._document_parts())
        inside = os.path.join(OH.REPO, "never_written_run.json")
        with self.assertRaises(LC.TL.TimelineError):
            LC.write_run_json(inside, document)
        self.assertFalse(os.path.exists(inside))
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "run.json")
            LC.write_run_json(path, document)
            with self.assertRaises(LC.TL.TimelineError):
                LC.write_run_json(path, document)
            with open(path, encoding="utf-8") as handle:
                self.assertEqual(json.load(handle)["kind"], LC.KIND)

    def _main(self, argv, run_source=None, client_class=None):
        events = []
        built = []

        def fake_build(camera, predictor, svm, rotate="none", original=None):
            built.append((camera, rotate))
            return ""

        def fake_run(source, hook):
            if run_source is not None:
                return run_source(source, hook)
            for _ in range(10):
                hook.read(PlainStream())
                hook.tick()
            return "finished"

        class Recorder(io.StringIO):
            def write(self, text):
                if text.startswith("status:"):
                    events.append("summary")
                return super(Recorder, self).write(text)

        original_stop = WH.DrawThread.stop

        def spy_stop(thread, timeout=2.0):
            events.append("drawer.stop")
            return original_stop(thread, timeout)

        with tempfile.NamedTemporaryFile(suffix=".dat") as predictor:
            argv = list(argv) + ["--predictor", predictor.name]
            out = Recorder()
            with contextlib.ExitStack() as stack:
                stack.enter_context(mock.patch.object(LC, "build_source", fake_build))
                stack.enter_context(mock.patch.object(LC, "run_source", fake_run))
                stack.enter_context(mock.patch.object(WH, "find_svm", lambda: "svm"))
                stack.enter_context(mock.patch.object(WH.DrawThread, "stop", spy_stop))
                stack.enter_context(mock.patch.object(LC, "make_live_drawer", lambda hook, rotate, duration: (lambda: ((lambda post: None), (lambda: None)))))
                if client_class is not None:
                    stack.enter_context(mock.patch.object(LC, "ScoreClient", client_class))
                stack.enter_context(contextlib.redirect_stdout(out))
                try:
                    code = LC.main(argv)
                except BaseException as error:
                    code = error
        return code, events, built, out.getvalue()

    def test_no_score_runs_without_a_child_and_says_so(self):
        created = []

        class Spy(object):
            def __init__(self, *args, **kwargs):
                created.append(1)
        code, events, built, text = self._main(["2", "--no-score", "--rotate", "cw90"], client_class=Spy)
        self.assertEqual(code, 0)
        self.assertEqual(created, [])
        self.assertEqual(built, [(2, "cw90")])
        self.assertIn("score process: disabled", text)

    def test_the_summary_comes_before_the_cleanup_even_when_the_run_raises(self):
        closed = []

        class FakeClient(object):
            def __init__(self, command, on_response, **kwargs):
                self.command = command
                self.ready = threading.Event()
                self.folds = None
                self.exit_code = None
                self.sent = self.received = self.error_lines = 0

            def start(self):
                return self

            def submit(self, message):
                return True

            def state(self):
                return LC.SCORE_LOADING

            def close(self, timeout=2.0):
                closed.append("client.close")
                return 0

        def boom(source, hook):
            raise RuntimeError("unexpected")
        code, events, _built, _text = self._main(["--ckpt-root", "<CKPT_DIR>"], run_source=boom, client_class=FakeClient)
        self.assertIsInstance(code, RuntimeError)
        self.assertLess(events.index("summary"), events.index("drawer.stop"))
        self.assertEqual(closed, ["client.close"])

    def test_out_json_is_off_by_default_and_written_only_when_asked(self):
        with tempfile.TemporaryDirectory() as folder:
            code, _events, _built, _text = self._main(["--no-score"])
            self.assertEqual(code, 0)
            self.assertEqual(os.listdir(folder), [])
            path = os.path.join(folder, "run.json")
            code, _events, _built, text = self._main(["--no-score", "--out-json", path])
            self.assertEqual(code, 0)
            with open(path, encoding="utf-8") as handle:
                self.assertEqual(json.load(handle)["status"], "finished")
            code, _events, _built, text = self._main(["--no-score", "--out-json", path])       # 이미 있으면 거부
            self.assertEqual(code, 2)
            self.assertIn("live_combined:", text)

    def test_argument_errors(self):
        code, _e, _b, text = self._main([])
        self.assertEqual(code, 2)
        self.assertIn("--ckpt-root", text)
        code, _e, _b, text = self._main(["--no-score", "--duration", "0"])
        self.assertEqual(code, 2)
        code, _e, _b, text = self._main(["1", "--camera", "2", "--no-score"])
        self.assertEqual(code, 2)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(LC.main(["--predictor", "no_such.dat", "--no-score"]), 2)
        self.assertIn("랜드마크 모델", out.getvalue())

    def test_parse_args_defaults(self):
        arguments = LC.parse_args(["--predictor", "p.dat", "--no-score"])
        self.assertEqual((arguments.preview_hz, arguments.rotate, arguments.duration, arguments.out_json, arguments.beep), (15.0, "none", None, None, False))
        self.assertEqual(LC.resolve_camera(arguments), 0)
        self.assertEqual(LC.resolve_camera(LC.parse_args(["3", "--predictor", "p"])), 3)
        self.assertEqual(LC.resolve_camera(LC.parse_args(["--camera", "4", "--predictor", "p"])), 4)
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                LC.parse_args(["--predictor", "p", "--preview-hz", "31"])
            with self.assertRaises(SystemExit):
                LC.parse_args(["--predictor", "p", "--rotate", "auto"])


class SourceStatsTest(unittest.TestCase):
    def _run(self, plan):
        """plan: (EAR 또는 None, 출처) 목록. 0.1초 간격으로 한 반복씩 돌린다."""
        hook, clock, stream = make_hook()
        for ear, source in plan:
            iterate(hook, stream, ear=ear, source=source, faces=1)
        return hook

    def test_source_ratios_percentiles_and_below_threshold_counts(self):
        plan = [(0.30, "detected"), (0.10, "detected"), (0.20, "detected"), (0.12, "tracked"), (None, "none"), (0.40, "detected"), (0.14, "tracked"), (None, "none")]
        data = self._run(plan).stats.summary()
        self.assertEqual(data["counts"], {"detected": 4, "tracked": 2, "none": 2})
        self.assertAlmostEqual(data["ratios"]["detected"], 0.5)
        self.assertAlmostEqual(data["ratios"]["none"], 0.25)
        ear = data["detected_ear"]
        self.assertAlmostEqual(ear["min"], 0.10)
        self.assertAlmostEqual(ear["median"], 0.25)                         # 0.10, 0.20, 0.30, 0.40 의 중앙값
        self.assertAlmostEqual(ear["p5"], float(np.percentile([0.30, 0.10, 0.20, 0.40], 5)))
        self.assertAlmostEqual(ear["p95"], float(np.percentile([0.30, 0.10, 0.20, 0.40], 95)))
        self.assertEqual(data["at_or_below_threshold"], {"detected": 1, "tracked": 2})
        self.assertEqual(data["threshold"], 0.15)

    def test_longest_runs_for_detected_only_and_combined(self):
        plan = [(0.30, "detected"), (0.10, "detected"), (0.10, "detected"), (0.10, "detected"),      # detected 만: 0.2초
                (0.10, "tracked"), (0.10, "tracked"), (0.30, "detected"),                            # 합치면 앞 구간이 tracked 로 이어져 0.4초
                (0.10, "detected"), (None, "none"), (0.10, "detected")]
        data = self._run(plan).stats.summary()
        self.assertAlmostEqual(data["longest_run_s"]["detected"], 0.2)
        self.assertAlmostEqual(data["longest_run_s"]["combined"], 0.4)
        self.assertEqual(self._run([(0.30, "detected")]).stats.summary()["longest_run_s"], {"detected": 0.0, "combined": 0.0})

    def test_no_frames_prints_dashes_and_the_observation_sentence(self):
        lines = SourceStatsTest._lines_of(LC.SourceStats())
        self.assertEqual(lines[1], "detected EAR: min - ; p5 - ; median - ; p95 -")
        self.assertTrue(lines[0].startswith("sources: detected 0 (-)"))
        self.assertEqual(lines[-1], "이 값은 관찰값이며 알람 판정이 아님")

    @staticmethod
    def _lines_of(stats):
        return stats.lines()

    def test_summary_json_and_order_before_the_fixed_sentence(self):
        hook = self._run([(0.10, "detected"), (0.30, "tracked"), (None, "none")])
        lines = []
        reporter = LC.LiveReporter(hook, None, None, None, out=lines.append, flush=lambda: None)
        summary = reporter.emit("finished")
        self.assertLess(lines.index("swallowed exceptions: 0"), [i for i, line in enumerate(lines) if line.startswith("sources:")][0])
        self.assertEqual(lines[-2], "이 값은 관찰값이며 알람 판정이 아님")
        self.assertEqual(lines[-1], LC.NOTE)
        document = LC.build_run_document(hook, summary, None, "finished")
        LC.validate_run_document(document)
        self.assertEqual(document["source_stats"]["counts"]["tracked"], 1)
        self.assertEqual(len(hook.stats.detected_ears), 1)                  # 프레임이 아니라 부동소수점 하나만 저장


class ExecutionTest(unittest.TestCase):
    def test_a_fake_source_run_wires_the_namespace_and_collects_everything(self):
        source = "\n".join([
            "class _F(object):",
            "    def copy(self):",
            "        return self",
            "class S(object):",
            "    def read(self):",
            "        return (True, _F())",
            "stream = S()",
            "for number in range(6):",
            "    (grabbed, frame) = _hook_read(stream)",
            "    if number % 2 == 0:",
            "        _hook_ear(0.3, 'detected', number // 2 + 1, None, None)",
            "    if _hook_progress_tick() != 255:",
            "        break", ""])
        hook = LC.LiveHook(clock=Clock())
        status = LC.run_source(source, hook)
        self.assertEqual(status, "finished")
        self.assertEqual((len(hook.ticks), hook.rule_updates, len(hook.reads)), (6, 6, 6))
        self.assertTrue(hook.stop_event.is_set())                           # 실행이 끝나면 종료 플래그가 선다

    def test_the_stop_flag_ends_an_original_style_loop(self):
        source = "\n".join([
            "class S(object):",
            "    def read(self):",
            "        return (True, None)",
            "stream = S()",
            "while True:",
            "    (grabbed, frame) = _hook_read(stream)",
            "    if not grabbed:",
            "        break",
            "    _hook_progress_tick()", ""])
        hook = LC.LiveHook(clock=Clock())
        original_read = hook.read
        counter = {"reads": 0}

        def read_then_stop(stream):
            counter["reads"] += 1
            if counter["reads"] == 3:
                hook.stop_event.set()                                       # 세 번째 읽기 전에 종료 플래그가 선다(창에서 q 를 누른 것과 같다)
            return original_read(stream)
        hook.read = read_then_stop
        self.assertEqual(LC.run_source(source, hook), "finished")
        self.assertEqual(len(hook.ticks), 2)                                # 두 반복 뒤 (False, None) 으로 루프가 끝난다

    def test_importing_the_module_loads_neither_cv2_dlib_nor_tensorflow(self):
        code = "import sys; import improve.hmlstm.live_combined; print('tensorflow' in sys.modules, 'cv2' in sys.modules, 'dlib' in sys.modules)"
        env = dict(os.environ, PYTHONPATH=os.path.join(OH.REPO, "src"))
        done = subprocess.run([sys.executable, "-c", code], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, env=env, timeout=120)
        self.assertEqual(done.stdout.strip(), "False False False", done.stderr)

    def test_no_image_saving_calls_in_the_module(self):
        with open(LC.__file__, encoding="utf-8") as handle:
            source = handle.read()
        for token in ("imwrite", "imsave", "VideoWriter", "ImageGrab", "save_frame", "import tensorflow"):
            self.assertNotIn(token, source)


if __name__ == "__main__":
    unittest.main()
