# live_session.py
# 역할: 라이브 게이지의 세션 상태기계(순수 numpy, cv2와 TensorFlow를 가져오지 않는다). 후킹 추출이 내는 깜빡임 행을 시각 순서로 받아
#       기준 구간 수집 -> 기준 통계 -> 기준 이후 행의 Freq 사후 계산 -> 창 생성 -> 점수 반영 -> 게이지 상태까지를 맡는다.
# 결정: I-23(기준 구간 300초), I-26 (a)(d), I-28 (6), I-W-28 (가)(프레임 수 기록과 Freq 사후 계산, 경계는 행 기록 시각 기준, 행 확정 지연 보정 없음),
#       I-W-29(기준 구간이 졸음 없는 상태라는 보장이 없다는 한계 명시, 코드로 막지 않음), I-27(평균과 최솟값, 최댓값 표시), I-33(점수는 참고용).
# 입력: add_row(row, frame_index, t, face_frames=None). row 는 [TOTAL, Freq*100, Amp, Dur, Vel] 5개 값, frame_index 는 행이 확정된 반복의 번호,
#       t 는 행이 기록된 단조 시각(초, 생성자의 start_time 과 같은 시계). face_frames 는 원본의 number_of_frames(얼굴이 검출된 프레임 수)이며
#       훅이 따로 기록해 주면 그대로 쓰고, 없으면 행의 TOTAL 과 Freq*100 으로 복원한다.
# 기준 구간: 세션 시작부터 reference_s(300초) 동안(경과 시간이 reference_s 미만인 행)이다. 이 구간에는 점수를 내지 않는다.
#       reference_s 시점에 기준 행이 min_reference_rows(30)개 미만이면 extension_s(600초)까지 연장하고, 연장 중 30개가 되는 즉시 그 행에서 끝낸다.
#       extension_s 에도 30개가 안 되면 "보정 부족"으로 점수를 내지 않는다. 기준 통계는 기준 행의 마지막 n//3행(windows.reference_details, 퇴행 판정 포함)이다.
# Freq 사후 계산(I-26 (b), I-W-28 (가)): 원본은 행에 TOTAL_BLINKS(깜빡임 누적)와 BLINK_FRAME_FREQ*100 = TOTAL_BLINKS / number_of_frames * 100 을 쓴다
#       (blink_video.py 380, 455, 462~464행, 소수 4자리 %.4f). number_of_frames 는 얼굴이 검출된 프레임만 센다. 그래서 행에서 F = TOTAL * 100 / Freq100 으로
#       얼굴 검출 프레임 수를 복원한다. 카운터를 기준 구간의 마지막 행에서 0으로 되돌린 것과 같게, 기준 이후 행은
#           TOTAL' = TOTAL - TOTAL_L,  F' = F - F_L,  Freq100' = 100 * TOTAL' / F'      (L 은 기준 구간의 마지막 행)
#       로 바꾼다. 리셋 시점은 "기준 구간의 마지막 행이 기록된 시점"이고, 그 시점과 300초 경계 사이의 얼굴 검출 프레임은 F' 에 들어간다(행 기록 시각 기준 귀속).
#       F 복원은 소수 4자리 반올림 때문에 정수 후보가 여럿일 수 있어(recover_face_frames) 오차가 생길 수 있으며 크기는 시험에서 잰다.
# 창: 기준 이후 행만으로 길이 30, 이동 2(windows 의 함수). 30번째 기준 이후 행에서 첫 창, 이후 행 2개마다 창 하나. 새 창은 WindowJob 으로 돌려주고 점수 서비스로 보낸다.
# 점수 반영: apply_score 로 응답(순번, 평균, 최솟값, 최댓값, 폴드별 값)을 받아 gauge.display_state 에 넘길 시간선 구간(section)을 점진적으로 쌓는다.
"""라이브 세션 상태기계(Python 3.6 호환, numpy 만 사용)."""
import math

import numpy as np

from . import gauge as GAUGE
from . import windows as W

PHASE_REFERENCE = "reference"
PHASE_COLLECTING = "collecting"
PHASE_SCORING = "scoring"
PHASE_SHORT = "calibration_short"

REFERENCE_S = 300.0
EXTENSION_S = 600.0
MIN_REFERENCE_ROWS = 30
FACE_SEARCH_RADIUS = 3                 # 복원한 얼굴 검출 프레임 수 주변에서 정확히 맞는 정수를 찾는 범위
ROW_LENGTH = 5


class SessionError(ValueError):
    """입력이 틀렸을 때(행 길이, 시각 역행 등)."""


def freq_text(total, faces):
    """원본이 행에 쓰는 Freq*100 값(소수 4자리 문자열을 다시 읽은 값). total / faces * 100, '%.4f'."""
    return float("%.4f" % (total / float(faces) * 100.0))


def candidate_face_frames(total, freq100):
    """행의 TOTAL 과 Freq*100(소수 4자리)에 정확히 맞는 얼굴 검출 프레임 수 후보(정수 목록, 추정값에 가까운 순). 없으면 빈 목록."""
    if not (total > 0 and freq100 > 0) or not math.isfinite(total) or not math.isfinite(freq100):
        return []
    estimate = total * 100.0 / freq100
    center = int(round(estimate))
    found = []
    for faces in range(max(1, center - FACE_SEARCH_RADIUS), center + FACE_SEARCH_RADIUS + 1):
        if freq_text(total, faces) == float("%.4f" % freq100):
            found.append(faces)
    found.sort(key=lambda faces: abs(faces - estimate))
    return found


def recover_face_frames(total, freq100):
    """행에서 얼굴 검출 프레임 수 F 를 복원한다. 정확히 맞는 정수 후보 중 추정값 TOTAL*100/Freq100 에 가장 가까운 것, 후보가 없으면 추정값을 반올림한 정수. 복원 불가면 None."""
    if not (total > 0 and freq100 > 0) or not math.isfinite(total) or not math.isfinite(freq100):
        return None
    found = candidate_face_frames(total, freq100)
    if found:
        return found[0]
    return max(1, int(round(total * 100.0 / freq100)))


class WindowJob(object):
    """점수 서비스로 보낼 창 하나. seq 는 요청 순번, k 는 1부터 센 창 번호, end_row 는 기준 이후 행 중 창의 마지막 행 번호(0부터), frame_index 는 그 행의 프레임 번호."""

    def __init__(self, seq, k, end_row, frame_index, window):
        self.seq, self.k, self.end_row, self.frame_index = seq, k, end_row, frame_index
        self.window = window                       # (30, 4) float64, 1단계 정규화 완료

    def message(self):
        """표준 입출력 JSON 한 줄로 보낼 dict: {"seq", "windows": [창 하나]}."""
        return {"seq": self.seq, "windows": [self.window.tolist()]}


class LiveSession(object):
    def __init__(self, start_time, fps=30.0, reference_s=REFERENCE_S, extension_s=EXTENSION_S, min_reference_rows=MIN_REFERENCE_ROWS):
        if not fps > 0:
            raise SessionError("fps 는 0 보다 커야 합니다: %r" % (fps,))
        self.start_time = float(start_time)
        self.fps = float(fps)
        self.reference_s = float(reference_s)
        self.extension_s = float(extension_s)
        self.min_reference_rows = int(min_reference_rows)
        self.phase = PHASE_REFERENCE
        self.last_time = self.start_time
        self.last_frame_index = -1
        self.ref_rows = []
        self.ref_faces = []
        self.reference = None                      # windows.Reference (기준 통계, 기준 구간이 끝난 뒤)
        self.reference_end_time = None
        self.reason = None                         # 보정 부족의 이유(문자열)
        self.base_total = None
        self.base_faces = None
        self.post_rows = []
        self.post_frame_indices = []
        self.unrecoverable_rows = 0
        self.jobs = {}
        self.next_seq = 1
        self.section = {"fps": self.fps, "blink_frame_indices": [], "windows": []}

    # ----- 시간 -----
    def elapsed(self, now):
        return now - self.start_time

    def advance(self, now):
        """행이 없어도 시간만으로 일어나는 전이를 처리한다: 기준 구간이 끝났는지, 연장 한도를 넘었는지."""
        if now < self.last_time:
            raise SessionError("시각이 거꾸로 갑니다: %r 다음에 %r" % (self.last_time, now))
        self.last_time = now
        if self.phase != PHASE_REFERENCE:
            return
        elapsed = self.elapsed(now)
        if elapsed >= self.reference_s and len(self.ref_rows) >= self.min_reference_rows:
            self._finalize_reference(now)
        elif elapsed >= self.extension_s:
            self._fail("only %d reference rows after %.0f s" % (len(self.ref_rows), self.extension_s))

    def _fail(self, reason):
        self.phase = PHASE_SHORT
        self.reason = reason

    def _finalize_reference(self, now):
        details = W.reference_details(np.array(self.ref_rows))
        if details is None or details.reason is not None:
            self._fail("reference unusable: %s" % ("no rows" if details is None else details.reason))
            return
        faces = self.ref_faces[-1]
        if faces is None:
            self._fail("face frame count of the last reference row cannot be recovered")
            return
        self.reference = details
        self.reference_end_time = now
        self.base_total = float(self.ref_rows[-1][0])
        self.base_faces = int(faces)
        self.phase = PHASE_COLLECTING

    # ----- 행 -----
    def add_row(self, row, frame_index, t, face_frames=None):
        """행 하나를 넣는다. 새 창이 생기면 [WindowJob] 을, 아니면 빈 목록을 돌려준다."""
        values = [float(v) for v in row]
        if len(values) != ROW_LENGTH or any(not math.isfinite(v) for v in values):
            raise SessionError("행은 유한한 수 %d개여야 합니다: %r" % (ROW_LENGTH, row))
        if isinstance(frame_index, bool) or not isinstance(frame_index, int) or frame_index < 0:
            raise SessionError("frame_index 는 0 이상의 정수여야 합니다: %r" % (frame_index,))
        if frame_index < self.last_frame_index:
            raise SessionError("frame_index 가 거꾸로 갑니다: %r 다음에 %r" % (self.last_frame_index, frame_index))
        self.advance(t)
        self.last_frame_index = frame_index
        if self.phase == PHASE_REFERENCE:
            self.ref_rows.append(values)
            self.ref_faces.append(int(face_frames) if face_frames is not None else recover_face_frames(values[0], values[1]))
            if self.elapsed(t) >= self.reference_s and len(self.ref_rows) >= self.min_reference_rows:
                self._finalize_reference(t)          # 연장 중 30개가 되면 그 행이 기준 구간의 마지막 행이다
            return []
        if self.phase == PHASE_SHORT:
            return []
        return self._add_post_row(values, frame_index, face_frames)

    def _add_post_row(self, values, frame_index, face_frames):
        faces = int(face_frames) if face_frames is not None else recover_face_frames(values[0], values[1])
        total = max(values[0] - self.base_total, 0.0)
        freq = 0.0
        if faces is None:
            self.unrecoverable_rows += 1
        elif faces - self.base_faces > 0:
            freq = 100.0 * total / (faces - self.base_faces)
        derived = np.array([total, freq, values[2], values[3], values[4]], dtype=np.float64)
        self.post_rows.append(derived)
        self.post_frame_indices.append(frame_index)
        self.section["blink_frame_indices"].append(frame_index)
        count = len(self.post_rows)
        if count < W.WINDOW_SIZE or (count - W.WINDOW_SIZE) % W.STRIDE != 0:
            return []
        means, stds = self.reference.means, self.reference.stds
        normalized = W.normalize_table(np.array(self.post_rows[-W.WINDOW_SIZE:]), means, stds)
        window = W.unroll_in_time(normalized, False)[0]
        job = WindowJob(self.next_seq, (count - W.WINDOW_SIZE) // W.STRIDE + 1, count - 1, frame_index, window)
        self.next_seq += 1
        self.jobs[job.seq] = job
        return [job]

    # ----- 점수 -----
    def apply_score(self, seq, mean, minimum, maximum, folds=None):
        """점수 응답 하나를 시간선 구간에 쌓는다. 모르는 순번이면 SessionError."""
        job = self.jobs.pop(seq, None)
        if job is None:
            raise SessionError("모르는 순번의 점수입니다: %r" % (seq,))
        entry = {"k": job.k, "end_row": job.end_row, "frame_index": job.frame_index, "time_sec": job.frame_index / self.fps,
                 "mean": float(mean), "min": float(minimum), "max": float(maximum), "folds": [float(v) for v in folds] if folds is not None else []}
        windows = self.section["windows"]
        windows.append(entry)
        windows.sort(key=lambda item: item["k"])
        if self.phase == PHASE_COLLECTING:
            self.phase = PHASE_SCORING
        return entry

    def apply_response(self, message):
        """점수 서비스 응답 dict({"seq", "mean", "min", "max", "folds"}; 값은 창 하나짜리 목록이거나 숫자)를 쌓는다. 오류 응답이면 SessionError."""
        if "error" in message:
            raise SessionError("점수 서비스 오류(seq %r): %s" % (message.get("seq"), message["error"]))

        def first(value):
            return value[0] if isinstance(value, (list, tuple)) else value

        folds = message.get("folds")
        if isinstance(folds, (list, tuple)) and folds and isinstance(folds[0], (list, tuple)):
            folds = folds[0]
        return self.apply_score(message["seq"], first(message["mean"]), first(message["min"]), first(message["max"]), folds)

    # ----- 읽기 -----
    def status_text(self, now=None):
        """단계 문자열: 기준 수집 중 n초, 수집 중 k/30, 점수 있음, 보정 부족."""
        if self.phase == PHASE_REFERENCE:
            elapsed = self.elapsed(self.last_time if now is None else now)
            return "기준 수집 중 %d초" % max(0, int(elapsed))
        if self.phase == PHASE_COLLECTING:
            return "수집 중 %d/%d" % (min(len(self.post_rows), W.WINDOW_SIZE), W.WINDOW_SIZE)
        if self.phase == PHASE_SCORING:
            return "점수 있음"
        return "보정 부족"

    def gauge_state(self, frame_index):
        """frame_index 프레임에서의 게이지 상태(gauge.display_state). 보정 부족이면 기준 없음 단계이다."""
        return GAUGE.display_state(self.section, frame_index, baseline_missing=(self.phase == PHASE_SHORT))

    def pending(self):
        """아직 점수 응답을 받지 못한 요청 순번 목록."""
        return sorted(self.jobs)

    def describe(self):
        return {"phase": self.phase, "text": self.status_text(), "reference_rows": len(self.ref_rows), "post_rows": len(self.post_rows),
                "windows_scored": len(self.section["windows"]), "pending": self.pending(), "reason": self.reason,
                "unrecoverable_rows": self.unrecoverable_rows}
