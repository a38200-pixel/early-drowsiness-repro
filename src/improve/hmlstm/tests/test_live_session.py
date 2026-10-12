# live_session 시험: 합성 행만 쓴다(웹캠, cv2, TensorFlow 없음). 실데이터 일치 시험은 환경변수 TIMELINE_DIR 가 있고 파일이 있을 때만 돈다.
import json
import os
import subprocess
import sys
import unittest

import numpy as np

from improve.hmlstm import gauge as GAUGE
from improve.hmlstm import live_session as LS
from improve.hmlstm import original_hook as OH
from improve.hmlstm import timeline as TL
from improve.hmlstm import windows as W


def synthetic_rows(count, seed=0, first_faces=100):
    """TOTAL, 얼굴 검출 프레임 수, Freq*100(소수 4자리)이 서로 맞는 합성 행과 행마다의 얼굴 검출 프레임 수."""
    rng = np.random.RandomState(seed)
    rows, faces_list = [], []
    faces = first_faces
    for number in range(1, count + 1):
        faces += int(rng.randint(60, 200))
        rows.append([float(number), LS.freq_text(number, faces), 0.05 + 0.1 * rng.rand(), 5 + 10 * rng.rand(), 0.1 + 0.3 * rng.rand()])
        faces_list.append(faces)
    return rows, faces_list


def ready_session(reference_count=40, start=1000.0, post_count=0, seed=0):
    """기준 구간에 행 reference_count 개를 넣고 300초를 넘겨 기준을 끝낸 세션(face_frames 를 정확히 줌). 기준 이후 행 post_count 개도 넣는다."""
    rows, faces = synthetic_rows(reference_count + post_count + 5, seed)
    session = LS.LiveSession(start)
    for index in range(reference_count):
        session.add_row(rows[index], index * 100, start + 5.0 + 6.0 * index, faces[index])
    session.advance(start + 300.0)
    jobs = []
    for index in range(reference_count, reference_count + post_count):
        jobs += session.add_row(rows[index], index * 100, start + 301.0 + index, faces[index])
    return session, rows, faces, jobs


class ReferenceBoundaryTest(unittest.TestCase):
    def test_a_row_just_before_300_seconds_is_reference_and_a_row_at_300_is_not(self):
        rows, faces = synthetic_rows(60)
        session = LS.LiveSession(1000.0)
        for index in range(40):
            session.add_row(rows[index], index * 100, 1000.0 + 7.0 * index, faces[index])
        self.assertEqual(session.phase, LS.PHASE_REFERENCE)
        session.add_row(rows[40], 4000, 1299.99, faces[40])
        self.assertEqual((session.phase, len(session.ref_rows)), (LS.PHASE_REFERENCE, 41))
        session.add_row(rows[41], 4100, 1300.0, faces[41])
        self.assertEqual(session.phase, LS.PHASE_COLLECTING)
        self.assertEqual((len(session.ref_rows), len(session.post_rows)), (41, 1))
        self.assertEqual(session.reference_end_time, 1300.0)
        self.assertEqual(session.base_total, 41.0)
        self.assertEqual(session.base_faces, faces[40])

    def test_time_alone_ends_the_reference_when_enough_rows_exist(self):
        session = LS.LiveSession(0.0)
        rows, faces = synthetic_rows(40)
        for index in range(35):
            session.add_row(rows[index], index, 5.0 + 8.0 * index, faces[index])
        session.advance(299.9)
        self.assertEqual(session.phase, LS.PHASE_REFERENCE)
        self.assertEqual(session.status_text(), "기준 수집 중 299초")
        session.advance(300.0)
        self.assertEqual(session.phase, LS.PHASE_COLLECTING)
        self.assertEqual(session.status_text(), "수집 중 0/30")
        with self.assertRaises(LS.SessionError):
            session.advance(299.0)

    def test_extension_ends_as_soon_as_the_thirtieth_reference_row_arrives(self):
        rows, faces = synthetic_rows(60)
        session = LS.LiveSession(0.0)
        for index in range(20):
            session.add_row(rows[index], index, 10.0 + 14.0 * index, faces[index])
        session.advance(300.0)
        self.assertEqual(session.phase, LS.PHASE_REFERENCE)                  # 20개뿐이라 연장
        for index in range(20, 29):
            session.add_row(rows[index], index, 310.0 + 10.0 * (index - 20), faces[index])
        self.assertEqual((session.phase, len(session.ref_rows)), (LS.PHASE_REFERENCE, 29))
        session.add_row(rows[29], 29, 420.0, faces[29])
        self.assertEqual((session.phase, len(session.ref_rows), session.reference_end_time), (LS.PHASE_COLLECTING, 30, 420.0))
        session.add_row(rows[30], 30, 430.0, faces[30])
        self.assertEqual((len(session.ref_rows), len(session.post_rows)), (30, 1))

    def test_not_enough_rows_by_600_seconds_gives_calibration_short(self):
        rows, faces = synthetic_rows(60)
        session = LS.LiveSession(0.0)
        for index in range(25):
            session.add_row(rows[index], index, 10.0 + 23.0 * index, faces[index])
        session.add_row(rows[25], 25, 599.9, faces[25])
        self.assertEqual(session.phase, LS.PHASE_REFERENCE)                  # 600초 직전에는 아직 기준 행으로 받는다
        self.assertEqual(session.add_row(rows[26], 26, 600.0, faces[26]), [])
        self.assertEqual(session.phase, LS.PHASE_SHORT)
        self.assertEqual(len(session.ref_rows), 26)                          # 600초의 행은 버려진다
        self.assertEqual(session.status_text(), "보정 부족")
        self.assertIn("26 reference rows", session.reason)
        self.assertEqual(session.add_row(rows[27], 27, 610.0, faces[27]), [])
        self.assertEqual(len(session.post_rows), 0)
        self.assertEqual(session.gauge_state(0)["stage"], GAUGE.STAGE_NO_BASELINE)

    def test_a_degenerate_reference_is_calibration_short(self):
        session = LS.LiveSession(0.0)
        for index in range(35):
            total = float(index + 1)
            session.add_row([total, 1.0, 0.1, 10.0, 0.2], index, 5.0 + 8.0 * index, int(total * 100))        # 모든 특징이 같은 값
        session.advance(300.0)
        self.assertEqual(session.phase, LS.PHASE_SHORT)
        self.assertIn("reference unusable", session.reason)
        self.assertIsNone(session.reference)

    def test_the_reference_statistics_use_the_last_third_of_the_reference_rows(self):
        session, rows, faces, _jobs = ready_session(reference_count=40)
        expected = W.reference_details(np.array(rows[:40]))
        self.assertEqual(session.reference.ref_n, 40 // 3)
        self.assertTrue(np.array_equal(session.reference.means, expected.means))
        self.assertTrue(np.array_equal(session.reference.stds, expected.stds))


class PostRowTest(unittest.TestCase):
    def test_total_and_freq_are_recomputed_from_the_end_of_the_reference(self):
        session, rows, faces, _ = ready_session(reference_count=40, post_count=3)
        base_total, base_faces = rows[39][0], faces[39]
        for offset in range(3):
            derived = session.post_rows[offset]
            index = 40 + offset
            self.assertEqual(derived[0], rows[index][0] - base_total)
            self.assertAlmostEqual(derived[1], 100.0 * (rows[index][0] - base_total) / (faces[index] - base_faces), places=12)
            self.assertEqual(list(derived[2:]), rows[index][2:])

    def test_without_face_frames_the_count_is_recovered_from_the_row(self):
        rows, faces = synthetic_rows(60)
        session = LS.LiveSession(0.0)
        for index in range(40):
            session.add_row(rows[index], index, 5.0 + 7.0 * index)
        session.advance(300.0)
        session.add_row(rows[40], 40, 301.0)
        self.assertEqual(session.base_faces, faces[39])                      # F_L 도 행에서 복원한 값
        exact = 100.0 * 1.0 / (faces[40] - faces[39])
        self.assertAlmostEqual(session.post_rows[0][1], exact, delta=0.05 * exact)

    def test_a_batch_of_rows_that_straddles_the_boundary_never_gives_a_negative_total(self):
        session, rows, faces, _ = ready_session(reference_count=40)
        session.add_row(list(rows[39]), 5000, 1400.0, faces[39])
        self.assertEqual(session.post_rows[0][0], 0.0)
        self.assertEqual(session.post_rows[0][1], 0.0)

    def test_an_unrecoverable_count_gives_zero_frequency_and_is_counted(self):
        session, rows, faces, _ = ready_session(reference_count=40)
        session.add_row([50.0, 0.0, 0.1, 10.0, 0.2], 5000, 1400.0)
        self.assertEqual(session.unrecoverable_rows, 1)
        self.assertEqual(session.post_rows[0][1], 0.0)


class WindowTest(unittest.TestCase):
    def test_first_window_at_the_thirtieth_post_row_then_every_second_row(self):
        rows, faces = synthetic_rows(120)
        session = LS.LiveSession(0.0)
        for index in range(40):
            session.add_row(rows[index], index * 100, 5.0 + 7.0 * index, faces[index])
        session.advance(300.0)
        produced = [len(session.add_row(rows[index], index * 100, 301.0 + index, faces[index])) for index in range(40, 100)]
        self.assertTrue(all(count == 0 for count in produced[:29]))
        indexes = [position for position, count in enumerate(produced) if count]
        self.assertEqual(indexes[:4], [29, 31, 33, 35])
        self.assertEqual(indexes, list(range(29, 60, 2)))
        self.assertEqual(sorted(session.jobs), list(range(1, len(indexes) + 1)))

    def test_window_values_equal_the_batch_computation_on_the_post_rows(self):
        session, rows, faces, jobs = ready_session(reference_count=40, post_count=45, seed=3)
        self.assertEqual(len(jobs), (45 - 30) // 2 + 1)
        batch, ranges = W.video_windows(np.array(session.post_rows), session.reference.means, session.reference.stds, is_reference=False)
        self.assertEqual(batch.shape[0], len(jobs))
        for number, job in enumerate(jobs):
            self.assertEqual((job.k, job.seq, job.end_row), (number + 1, number + 1, ranges[number][1]))
            self.assertTrue(np.allclose(job.window, batch[number], rtol=0, atol=1e-12))
            self.assertEqual(job.frame_index, session.post_frame_indices[job.end_row])

    def test_a_job_message_is_json_with_one_window(self):
        session, rows, faces, jobs = ready_session(reference_count=40, post_count=30)
        message = json.loads(json.dumps(jobs[0].message()))
        self.assertEqual(message["seq"], 1)
        self.assertEqual(np.array(message["windows"]).shape, (1, 30, 4))

    def test_no_window_below_thirty_post_rows_and_none_during_the_reference(self):
        session, rows, faces, jobs = ready_session(reference_count=40, post_count=29)
        self.assertEqual(jobs, [])
        reference_only = LS.LiveSession(0.0)
        rows2, faces2 = synthetic_rows(40)
        for index in range(40):
            self.assertEqual(reference_only.add_row(rows2[index], index, 5.0 + 7.0 * index, faces2[index]), [])


class StatusAndScoreTest(unittest.TestCase):
    def test_status_text_transitions(self):
        rows, faces = synthetic_rows(100)
        session = LS.LiveSession(0.0)
        self.assertEqual(session.status_text(), "기준 수집 중 0초")
        for index in range(40):
            session.add_row(rows[index], index * 100, 5.0 + 7.0 * index, faces[index])
        self.assertEqual(session.status_text(), "기준 수집 중 278초")
        session.advance(300.0)
        self.assertEqual(session.status_text(), "수집 중 0/30")
        jobs = []
        for index in range(40, 70):
            jobs += session.add_row(rows[index], index * 100, 301.0 + index, faces[index])
        self.assertEqual(session.status_text(), "수집 중 30/30")
        self.assertEqual(len(jobs), 1)
        session.apply_score(jobs[0].seq, 5.0, 4.0, 6.0, [4.0, 4.5, 5.0, 5.5, 6.0])
        self.assertEqual(session.phase, LS.PHASE_SCORING)
        self.assertEqual(session.status_text(), "점수 있음")
        self.assertEqual(session.describe()["windows_scored"], 1)

    def test_scores_go_into_the_gauge_section_and_the_boundary_frame(self):
        session, rows, faces, jobs = ready_session(reference_count=40, post_count=34)
        self.assertEqual(len(jobs), 3)
        first = jobs[0]
        before = session.gauge_state(first.frame_index)
        self.assertEqual((before["stage"], before["blinks"]), (GAUGE.STAGE_COLLECTING, 30))          # 점수 응답 전에는 값이 없다
        session.apply_score(first.seq, 5.0, 4.0, 6.0, [4.0, 4.5, 5.0, 5.5, 6.0])
        at = session.gauge_state(first.frame_index)
        self.assertEqual((at["stage"], at["mean"], at["min"], at["max"], at["window"]), (GAUGE.STAGE_VALUE, 5.0, 4.0, 6.0, 1))
        self.assertEqual(session.gauge_state(first.frame_index - 1)["stage"], GAUGE.STAGE_COLLECTING)
        self.assertEqual(session.gauge_state(first.frame_index - 1)["blinks"], 29)

    def test_out_of_order_scores_are_kept_in_window_order_and_unknown_seq_is_refused(self):
        session, rows, faces, jobs = ready_session(reference_count=40, post_count=34)
        session.apply_score(jobs[1].seq, 6.0, 5.0, 7.0)
        session.apply_score(jobs[0].seq, 5.0, 4.0, 6.0)
        self.assertEqual([entry["k"] for entry in session.section["windows"]], [1, 2])
        self.assertEqual(session.pending(), [jobs[2].seq])
        with self.assertRaises(LS.SessionError):
            session.apply_score(999, 1.0, 1.0, 1.0)
        with self.assertRaises(LS.SessionError):
            session.apply_score(jobs[0].seq, 1.0, 1.0, 1.0)                  # 같은 순번을 두 번 반영할 수 없다

    def test_apply_response_accepts_the_service_shape_and_refuses_error_lines(self):
        session, rows, faces, jobs = ready_session(reference_count=40, post_count=30)
        entry = session.apply_response({"seq": jobs[0].seq, "mean": [5.0], "min": [4.0], "max": [6.0], "folds": [[4.0, 4.5, 5.0, 5.5, 6.0]]})
        self.assertEqual((entry["mean"], entry["folds"]), (5.0, [4.0, 4.5, 5.0, 5.5, 6.0]))
        with self.assertRaises(LS.SessionError):
            session.apply_response({"seq": 1, "error": "ValueError: bad"})

    def test_bad_input_is_refused(self):
        session = LS.LiveSession(0.0)
        with self.assertRaises(LS.SessionError):
            session.add_row([1.0, 2.0, 3.0], 0, 1.0)
        with self.assertRaises(LS.SessionError):
            session.add_row([1.0, float("nan"), 3.0, 4.0, 5.0], 0, 1.0)
        session.add_row([1.0, 1.0, 0.1, 10.0, 0.2], 10, 5.0, 100)
        with self.assertRaises(LS.SessionError):
            session.add_row([2.0, 1.0, 0.1, 10.0, 0.2], 9, 6.0, 200)        # 프레임 번호 역행
        with self.assertRaises(LS.SessionError):
            session.add_row([2.0, 1.0, 0.1, 10.0, 0.2], 11, 4.0, 200)       # 시각 역행
        with self.assertRaises(LS.SessionError):
            LS.LiveSession(0.0, fps=0)


class FaceFrameRecoveryTest(unittest.TestCase):
    def test_the_rounding_to_four_decimals_matches_the_original_format(self):
        self.assertEqual(LS.freq_text(5, 3000), float("%.4f" % (5 / 3000.0 * 100.0)))
        self.assertEqual(LS.freq_text(1, 3), 33.3333)

    def test_recovery_is_exact_when_the_row_determines_the_count(self):
        rng = np.random.RandomState(0)
        exact = trials = 0
        for _ in range(3000):
            faces = int(rng.randint(300, 3000))
            total = int(rng.randint(1, 400))
            trials += 1
            exact += LS.recover_face_frames(total, LS.freq_text(total, faces)) == faces
        self.assertGreater(exact / float(trials), 0.99)                    # 측정: 얼굴 검출 프레임이 3000 미만이면 99.7%

    def test_the_true_count_is_always_among_the_candidates_in_the_session_regime(self):
        rng = np.random.RandomState(1)
        for _ in range(2000):
            faces = int(rng.randint(3000, 12000))
            total = int(rng.randint(40, 300))
            self.assertIn(faces, LS.candidate_face_frames(total, LS.freq_text(total, faces)))

    def test_rounding_error_of_the_frequency_after_the_reset_is_small(self):
        rng = np.random.RandomState(2)
        errors = []
        for _ in range(3000):
            base_faces, base_total = int(rng.randint(8000, 10000)), int(rng.randint(60, 120))
            post_faces, post_total = int(rng.randint(500, 6000)), int(rng.randint(1, 100))
            got_total = LS.recover_face_frames(base_total + post_total, LS.freq_text(base_total + post_total, base_faces + post_faces))
            got_base = LS.recover_face_frames(base_total, LS.freq_text(base_total, base_faces))
            true = 100.0 * post_total / post_faces
            errors.append(abs(100.0 * post_total / max(got_total - got_base, 1) - true) / true)
        errors = np.array(errors)
        self.assertEqual(float(np.median(errors)), 0.0)
        self.assertLess(float(np.percentile(errors, 95)), 1e-3)            # 측정: 95 백분위 3.4e-4, 최댓값 3.6e-3

    def test_unrecoverable_rows_give_none(self):
        self.assertIsNone(LS.recover_face_frames(0.0, 1.0))
        self.assertIsNone(LS.recover_face_frames(5.0, 0.0))
        self.assertEqual(LS.candidate_face_frames(5.0, -1.0), [])


class NoHeavyImportTest(unittest.TestCase):
    def test_importing_live_session_and_score_service_loads_neither_cv2_nor_tensorflow(self):
        code = ("import sys; import improve.hmlstm.live_session, improve.hmlstm.score_service; "
                "print('tensorflow' in sys.modules, 'cv2' in sys.modules, 'dlib' in sys.modules)")
        env = dict(os.environ, PYTHONPATH=os.path.join(OH.REPO, "src"))
        done = subprocess.run([sys.executable, "-c", code], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, env=env, timeout=120)
        self.assertEqual(done.stdout.strip(), "False False False", done.stderr)


class RealDataTest(unittest.TestCase):
    """평소영상과 대본영상의 행을 이어 붙인 한 세션이 timeline score 의 입력(창)과 같은지 본다. 환경변수 TIMELINE_DIR(timeline.json 과 두 영상의 행 JSON 이 있는 폴더)가 필요하다."""

    def _load(self):
        folder = os.environ.get("TIMELINE_DIR")
        if not folder or not os.path.isfile(os.path.join(folder, "timeline.json")):
            self.skipTest("TIMELINE_DIR 에 timeline.json 이 없어 건너뜁니다")
        with open(os.path.join(folder, "timeline.json"), encoding="utf-8") as handle:
            timeline = json.load(handle)
        names = [os.path.join(folder, os.path.splitext(timeline[key]["video_name"])[0] + "_rows.json") for key in ("baseline", "target")]
        if not all(os.path.isfile(name) for name in names):
            self.skipTest("행 JSON 이 없어 건너뜁니다")
        return timeline, TL.read_rows_document(names[0]), TL.read_rows_document(names[1])

    def _run(self, base, target, exact_faces):
        base_rows, target_rows = [list(r) for r in base["rows"]], [list(r) for r in target["rows"]]
        session = LS.LiveSession(0.0, fps=30.0)
        jobs = []
        count = len(base_rows)
        for index, row in enumerate(base_rows):
            jobs += session.add_row(row, int(base["frame_indices"][index]), 1.0 + 298.0 * index / count)
        base_faces = LS.recover_face_frames(base_rows[-1][0], base_rows[-1][1])
        base_total = base_rows[-1][0]
        recovered = []
        for index, row in enumerate(target_rows):
            faces = LS.recover_face_frames(row[0], row[1])
            recovered.append(faces)
            total = base_total + row[0]
            combined = [total, LS.freq_text(total, base_faces + faces), row[2], row[3], row[4]]       # 카운터 오프셋을 더한 이어 붙인 열
            jobs += session.add_row(combined, base["frames_read"] + int(target["frame_indices"][index]), 301.0 + 0.5 * index,
                                    base_faces + faces if exact_faces else None)
        return session, jobs, recovered

    def test_windows_match_timeline_score_inputs(self):
        timeline, base, target = self._load()
        base_rows, target_rows = np.array(base["rows"]), np.array(target["rows"])
        details = W.reference_details(base_rows)
        expected, ranges = W.video_windows(target_rows, details.means, details.stds, is_reference=False)
        for exact in (True, False):
            session, jobs, recovered = self._run(base, target, exact)
            self.assertEqual(session.phase, LS.PHASE_COLLECTING)
            self.assertEqual(len(session.ref_rows), len(base_rows))
            self.assertTrue(np.array_equal(session.reference.means, details.means))          # 기준 통계는 완전히 같다
            self.assertTrue(np.array_equal(session.reference.stds, details.stds))
            got = np.array([job.window for job in jobs])
            self.assertEqual(got.shape, expected.shape)
            diff = np.abs(got - expected)
            self.assertLessEqual(float(diff[:, :, 1:].max()), 1e-12)                          # Amp, Dur, Vel 은 같다
            self.assertLessEqual(float(diff[:, :, 0].max()), 5e-3)                            # Freq 는 행의 소수 4자리 반올림 때문에 어긋난다(측정 최대 3.0e-3)
            self.assertEqual([job.frame_index - base["frames_read"] for job in jobs], [w["frame_index"] for w in timeline["target"]["windows"]])
        session, jobs, recovered = self._run(base, target, True)
        own = np.array(target_rows, dtype=np.float64)
        own[:, 1] = [100.0 * row[0] / faces for row, faces in zip(target_rows, recovered)]    # 복원한 프레임 수로 다시 계산한 Freq
        same, _ranges = W.video_windows(own, details.means, details.stds, is_reference=False)
        self.assertLessEqual(float(np.abs(np.array([job.window for job in jobs]) - same).max()), 1e-9)       # 이어 붙임과 사후 계산의 산술은 같다


if __name__ == "__main__":
    unittest.main()
