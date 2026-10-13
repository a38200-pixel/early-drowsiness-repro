# live_combined.py
# 역할: 라이브 웹캠 통합 프로그램. 원본 blink_video.py 를 웹캠 입력으로 후킹한 경로 한 번으로 (1) 규칙 알람(눈 감김 EAR 규칙, LiveRules)과 (2) HM-LSTM 점수 게이지(LiveSession + 점수 자식 프로세스)를
#       한 창에 같이 구동하고 그린다. 설계 방침 (a)(b)(c)(d)(결정 2026-10-12, I-W-37)를 따른다. 프레임, 이미지, 영상은 저장하지 않는다.
# 스레드: (1) 주 스레드 = 원본 후킹 루프(그리기 없음). tick 마다 규칙 갱신, 세션 시간 진행, 최신 상태를 preview_hz 로만 우편함에 넣는다. (2) 그리기 스레드 = 창 만들기, 그리기, waitKey, 닫기를 모두 맡는다.
#       (3) 점수 프로세스 쓰기 스레드(큐에서 꺼내 JSON 한 줄씩 쓴다, 주 스레드는 큐에 넣기만 하므로 막히지 않는다). (4) 점수 프로세스 읽기 스레드(JSON 줄을 읽어 세션에 반영). 점수 프로세스의 표준 에러는 읽어서 버린다.
# 점수 프로세스: sys.executable -m improve.hmlstm.score_service 로 띄우는 자식 프로세스(TensorFlow 는 자식만 쓴다). 이 모듈은 TensorFlow, cv2, dlib 를 가져올 때 읽지 않는다(cv2 는 그리기 스레드 안에서 처음 불린다).
# 후킹 치환: webcam_hook_speed 와 같은 읽기 래퍼, Tk 숨김, 루프 뒤 destroyAllWindows 무력화(창은 그리기 스레드가 닫는다) + ear_hook_compare 의 두 대상(421행 detected, 506행 tracked)에
#       눈 좌표를 더한 콜백 문장. 421행은 leftEye, rightEye(415, 416행, 450폭 좌표), 506행은 p1, p2(현재 프레임에서 추적한 점)를 넘긴다. 모든 치환은 원본에서 정확히 한 번 일치해야 한다.
# 행 시점: 같은 반복 안에서 380행(number_of_frames 증가) -> 421행(콜백) -> 455행(Freq 계산) -> 462에서 464행(행 기록) -> tick 순서라서, 행이 나갈 때 "가장 최근 콜백이 준 number_of_frames" 가 행의 Freq 계산에 쓰인 값이다.
#       (얼굴 없음 분기에서는 number_of_frames 가 변하지 않는다.) frame_index 는 그 시점의 tick 횟수, t 는 그 반복의 읽기 직후 시각이다.
# 자기 점검: 행마다 |100 * TOTAL / face_frames - Freq100| <= 1e-4 인지 검사해 위반 개수와 첫 위반 행을 센다. 위반이 있어도 계속 돈다.
# 스레드 안전: LiveSession 은 내부 잠금이 없고 section 목록을 append 와 sort 로 바꾸므로 세션을 만지는 모든 호출을 잠금 하나(session_lock)로 감싼다. LiveRules, DisplayLatch, RecentMinimum 은 주 스레드만 만지고
#       그리기 스레드에는 필요한 값만 복사한 스냅샷을 넘긴다.
# 사용: PYTHONPATH=src python -m improve.hmlstm.live_combined [카메라 번호] --predictor <PREDICTOR> --ckpt-root <CKPT_DIR> [--duration 60] [--rotate none] [--preview-hz 15] [--no-score] [--beep] [--out-json <OUT_DIR>/run.json]
# 키(그리기 창): q 또는 Esc 종료, r 알람 표시 끄기, m 소리 켜고 끄기. 창을 닫거나 Ctrl+C 를 눌러도 종료한다.
"""라이브 웹캠 통합 프로그램(Python 3.6 호환)."""
import argparse
import collections
import json
import os
import queue
import subprocess
import sys
import threading
import time

import numpy as np

from . import ear_hook_compare as EC
from . import gauge as GAUGE
from . import live_session as LS
from . import original_hook as OH
from . import timeline as TL
from . import webcam_hook_speed as WH
from ..rules import live_demo as LD
from ..rules import video_rotation

LIVE_FPS = 30.0                               # 게이지의 갱신 후 경과 시간 계산에 쓰는 가정 fps(웹캠 측정 평균 30.00)
WINDOW_NAME = "live_combined"
REFERENCE_NOTE = "기준 구간(처음 300초)에는 졸음이 없는 상태여야 함"
NOTE = "이 값은 라이브 실행 요약이며 점수나 알람의 타당성 검증이 아님"
ROTATE_CHOICES = ("none", "cw90", "180", "ccw90")
ANGLE_BY_OPTION = {"none": 0, "cw90": 90, "180": 180, "ccw90": 270}
DETECTED_REPLACEMENT = EC.DETECTED_TARGET + "; _hook_ear(ear, 'detected', number_of_frames, leftEye, rightEye)"
TRACKED_REPLACEMENT = EC.TRACKED_TARGET + "; _hook_ear(ear, 'tracked', number_of_frames, p1, p2)"
SELF_CHECK_TOLERANCE = 1e-4
BELOW_NORMAL_PRIORITY_CLASS = 0x00004000      # Win32 값. 파이썬 3.6 의 subprocess 에는 이 이름이 없다(3.7 에서 추가)
STOP_KEYS = (ord("q"), 27)
SCHEMA_VERSION = 1
KIND = "live_combined_run"
SCORE_DISABLED, SCORE_LOADING, SCORE_READY, SCORE_STOPPED = "disabled", "loading", "ready", "stopped"

Post = collections.namedtuple("Post", ["frame", "left", "right", "ear", "source", "ear_min", "rules", "latch", "frame_index", "t"])


class LiveError(ValueError):
    """입력 인자나 파일의 오류(메시지는 한 줄로 사용자에게 보인다)."""


# ----- 치환 -----
def build_source(camera, predictor, svm, rotate="none", original=None):
    """웹캠 후킹용 원본 소스: original_hook 치환(영상 경로 자리에 정수 카메라 번호) + 읽기 래퍼 + Tk 숨김 + destroyAllWindows 무력화 + EAR 콜백 2개. 개수가 기대와 다르면 ReplacementError."""
    original = OH.read_original_source() if original is None else original
    replacements = []
    for label, old, new, expected in OH.build_replacements("", predictor, svm, rotate, True):
        if label == "input video":
            new = "path = %d" % int(camera)
        replacements.append((label, old, new, expected))
    replacements.append(("read timing wrapper", WH.READ_TARGET, WH.READ_REPLACEMENT, 1))
    replacements.append(("hide empty Tk window", WH.TK_TARGET, WH.TK_REPLACEMENT, 1))
    replacements.append(("skip destroyAllWindows in the main thread", WH.DESTROY_TARGET, WH.DESTROY_REPLACEMENT, 1))
    replacements.append(("ear callback (face detected)", EC.DETECTED_TARGET, DETECTED_REPLACEMENT, 1))
    replacements.append(("ear callback (optical flow tracked)", EC.TRACKED_TARGET, TRACKED_REPLACEMENT, 1))
    OH.check_counts(original, replacements)
    return OH.apply_replacements(original, replacements)


# ----- 규칙 스냅샷(그리기 스레드에 넘기는 읽기 전용 값) -----
class RulesSnapshot(object):
    """draw_ui 가 LiveRules 에서 읽는 값만 담은 복사본(같은 메서드 이름)."""

    def __init__(self, thresholds, primary, data, face_status):
        self.thresholds, self.primary, self._data, self._face_status = tuple(thresholds), primary, data, face_status

    def last_alarm_time(self, threshold):
        return self._data[threshold]["last"]

    def alarm_count(self, threshold):
        return self._data[threshold]["count"]

    def bar_fraction(self, threshold):
        return self._data[threshold]["fraction"]

    def closed_length_s(self, threshold):
        return self._data[threshold]["closed"]

    def face_status(self):
        return self._face_status


class LatchSnapshot(object):
    def __init__(self, data):
        self._data = data

    def active(self, key, now):
        return self._data[key]["latched"]


def snapshot_rules(rules, latch, now):
    """LiveRules 와 DisplayLatch 의 현재 값을 복사한다(주 스레드에서만 부른다). 반환: (RulesSnapshot, LatchSnapshot)."""
    data = {}
    for threshold in rules.thresholds:
        data[threshold] = {"last": rules.last_alarm_time(threshold), "count": rules.alarm_count(threshold), "fraction": rules.bar_fraction(threshold),
                           "closed": rules.closed_length_s(threshold), "latched": latch.active(threshold, now)}
    return RulesSnapshot(rules.thresholds, rules.primary, data, rules.face_status()), LatchSnapshot(data)


class FreqCheck(object):
    """자기 점검: 행의 Freq*100 이 100 * TOTAL / face_frames 와 tolerance 이내로 같은지 센다."""

    def __init__(self, tolerance=SELF_CHECK_TOLERANCE):
        self.tolerance = tolerance
        self.checked = 0
        self.violations = 0
        self.first_violation = None

    def check(self, row_number, row, faces):
        self.checked += 1
        ok = faces is not None and faces > 0 and abs(100.0 * row[0] / faces - row[1]) <= self.tolerance
        if not ok:
            self.violations += 1
            if self.first_violation is None:
                self.first_violation = row_number
        return ok


# ----- 출처별 관찰값(진단 전용) -----
class SourceStats(object):
    """프레임(반복)마다 출처와 EAR 를 세는 관찰용 집계. 판정에 쓰지 않는다. 저장은 카운터와 detected EAR 의 부동소수점 목록뿐이다."""
    SOURCES = ("detected", "tracked", "none")

    def __init__(self):
        self.counts = dict((name, 0) for name in self.SOURCES)
        self.detected_ears = []
        self.below = {"detected": 0, "tracked": 0}
        self.longest = {"detected": 0.0, "combined": 0.0}      # 임계값 이하가 이어진 가장 긴 구간(초): detected 만 / detected+tracked
        self._start = {"detected": None, "combined": None}
        self._threshold = None

    def add(self, t, ear, source, threshold):
        self._threshold = threshold
        name = source if source in self.counts else "none"
        self.counts[name] += 1
        low = ear is not None and name != "none" and ear <= threshold
        if name == "detected" and ear is not None:
            self.detected_ears.append(float(ear))
        if low:
            self.below[name] += 1
        for key, member in (("detected", name == "detected"), ("combined", name in ("detected", "tracked"))):
            if low and member:
                if self._start[key] is None:
                    self._start[key] = t
                self.longest[key] = max(self.longest[key], t - self._start[key])
            else:
                self._start[key] = None

    def summary(self):
        total = sum(self.counts.values())
        ears = np.array(self.detected_ears, dtype=float)
        percentile = (lambda q: float(np.percentile(ears, q))) if ears.size else (lambda q: None)
        return {"threshold": self._threshold, "frames": total, "counts": dict(self.counts),
                "ratios": dict((name, (self.counts[name] / float(total) if total else None)) for name in self.SOURCES),
                "detected_ear": {"min": float(ears.min()) if ears.size else None, "p5": percentile(5), "median": percentile(50), "p95": percentile(95)},
                "at_or_below_threshold": dict(self.below), "longest_run_s": dict(self.longest)}

    def lines(self):
        data = self.summary()
        number = lambda value: "-" if value is None else "%.4f" % value
        percent = lambda value: "-" if value is None else "%.1f%%" % (100.0 * value)
        counts, ratios, ear = data["counts"], data["ratios"], data["detected_ear"]
        threshold = "-" if data["threshold"] is None else "%g" % data["threshold"]
        return ["sources: " + " ; ".join("%s %d (%s)" % (name, counts[name], percent(ratios[name])) for name in self.SOURCES),
                "detected EAR: min %s ; p5 %s ; median %s ; p95 %s" % (number(ear["min"]), number(ear["p5"]), number(ear["median"]), number(ear["p95"])),
                "frames at or below %s: detected %d ; tracked %d" % (threshold, data["at_or_below_threshold"]["detected"], data["at_or_below_threshold"]["tracked"]),
                "longest run at or below %s: detected only %.2f s ; detected+tracked %.2f s" % (threshold, data["longest_run_s"]["detected"], data["longest_run_s"]["combined"]),
                "이 값은 관찰값이며 알람 판정이 아님"]


# ----- 후킹 루프 쪽 상태 -----
class LiveHook(object):
    """원본 후킹 루프가 부르는 함수 모음: read(읽기 래퍼), ear(EAR 콜백), on_row(깜빡임 행), tick(반복 끝). 모든 예외는 삼키고 errors 만 센다. 시각 함수는 시험에서 바꿀 수 있다."""

    def __init__(self, thresholds=LD.THRESHOLDS, duration_s=None, preview_hz=15.0, clock=time.perf_counter, session=None, submit=None, mailbox=None, beeper=None,
                 score_state=None):
        self.clock = clock
        self.rules = LD.LiveRules(thresholds)
        self.stats = SourceStats()
        self.latch = LD.DisplayLatch()
        self.recent = LD.RecentMinimum()
        self.session = session if session is not None else LS.LiveSession(0.0, fps=LIVE_FPS)
        self.session_lock = threading.Lock()
        self.submit = submit                      # callable(message) 또는 None. 막히면 안 된다(큐에 넣기만 한다).
        self.mailbox = mailbox
        self.beeper = beeper
        self.score_state = score_state if score_state is not None else (lambda: SCORE_DISABLED)
        self.duration_s = duration_s
        self.post_interval = 1.0 / preview_hz
        self.stop_event = threading.Event()
        self.reset_request = threading.Event()
        self.mute_request = threading.Event()
        self.t0 = None
        self.current_t = 0.0
        self.last_frame = None
        self.pending = None
        self.last_faces = 0
        self.last_post = None
        self.reads = []
        self.ticks = []
        self.row_times = []
        self.rows = []
        self.check = FreqCheck()
        self.errors = 0
        self.rule_updates = 0
        self.last_source = "none"

    # --- 읽기 래퍼 ---
    def read(self, stream):
        """원본의 `(grabbed, frame) = stream.read()` 자리. 종료 플래그가 서면 (False, None)을 돌려 원본 루프를 끝낸다. 첫 프레임 도착 시각이 시간 0이다."""
        if self.stop_event.is_set():
            return (False, None)
        started = self.clock()
        result = stream.read()
        now = self.clock()
        self.reads.append(now - started)
        if result[0]:
            if self.t0 is None:
                self.t0 = now
            self.current_t = now - self.t0
            self.last_frame = result[1]
        else:
            self.last_frame = None
        return result

    # --- 콜백 ---
    def ear(self, value, source, faces, left, right):
        try:
            self.pending = (float(value), str(source), int(faces), left, right)
            self.last_faces = int(faces)
        except Exception:
            self.errors += 1

    def on_row(self, row):
        try:
            values = [float(v) for v in row]
            index = len(self.ticks)               # 지금까지 끝난 tick 횟수 = 이 반복의 번호
            t = self.current_t
            faces = self.last_faces
            self.row_times.append(self.clock())
            self.rows.append({"row": values, "frame_index": index, "t": t, "face_frames": faces})
            self.check.check(len(self.rows) - 1, values, faces)
            with self.session_lock:
                jobs = self.session.add_row(values, index, t, faces if faces > 0 else None)
            if self.submit is not None:
                for job in jobs:
                    self.submit(job.message())
        except Exception:
            self.errors += 1

    # --- 반복 끝 ---
    def tick(self):
        now = self.clock()
        index = len(self.ticks)
        self.ticks.append(now)
        try:
            t = self.current_t
            pending, self.pending = self.pending, None
            if pending is None:
                ear_value, source, left, right = None, "none", None, None
                new_alarms = self.rules.update(t, None, "none")
            else:
                ear_value, source, _faces, left, right = pending
                new_alarms = self.rules.update(t, ear_value, source)
                if source == "detected":
                    self.recent.add(t, ear_value)
            self.rule_updates += 1
            self.stats.add(t, ear_value, source, self.rules.primary)
            self.last_source = source
            for threshold, _alarm in new_alarms:
                self.latch.trigger(threshold, t)
            if new_alarms and self.beeper is not None:
                self.beeper.on_new_alarms(new_alarms)
            if self.reset_request.is_set():
                self.reset_request.clear()
                self.latch.reset()
                if self.beeper is not None:
                    self.beeper.stop()
            if self.mute_request.is_set():
                self.mute_request.clear()
                if self.beeper is not None:
                    self.beeper.toggle_mute()
            with self.session_lock:
                self.session.advance(t)
            shown = ear_value if source == "detected" else None
            ear_min = self.recent.minimum(t) if shown is not None else None
            self._post(now, index, t, shown, source, ear_min, left, right)
        except Exception:
            self.errors += 1
        if self.stop_event.is_set():
            return 0
        if self.duration_s and self.current_t >= self.duration_s:
            self.stop_event.set()
            return 0
        return OH.NO_KEY

    def _post(self, now, index, t, ear_value, source, ear_min, left, right):
        if self.mailbox is None or self.last_frame is None:
            return
        if self.last_post is not None and now - self.last_post < self.post_interval:
            return
        self.last_post = now
        rules_snapshot, latch_snapshot = snapshot_rules(self.rules, self.latch, t)
        self.mailbox.put(Post(self.last_frame.copy(), None if left is None else np.array(left), None if right is None else np.array(right), ear_value, source,
                              ear_min, rules_snapshot, latch_snapshot, index, t))

    # --- 그리기 스레드가 읽는 값(잠금 안에서 짧게) ---
    def live_info(self, frame_index, t):
        with self.session_lock:
            description = self.session.describe()
            info = {"gauge_state": self.session.gauge_state(frame_index), "status_text": self.session.status_text(t), "phase": description["phase"],
                    "reference_rows": description["reference_rows"], "post_rows": description["post_rows"], "windows_scored": description["windows_scored"],
                    "pending": list(description["pending"]), "reason": description["reason"]}
        info["score_state"] = self.score_state()
        info["elapsed"] = t
        return info


# ----- 게이지 줄에 무엇을 보일지(순수 함수) -----
def gauge_view(info):
    """게이지 줄의 내용을 정한다. 반환: ("gauge", 게이지 상태) 이면 gauge.py 의 문구를 그대로 쓰고, ("text", 줄 목록, 진행 비율 또는 None) 이면 단계 문구를 쓴다."""
    label = GAUGE.LABEL
    phase, score_state = info["phase"], info["score_state"]
    if score_state == SCORE_STOPPED:
        return ("text", [label, "점수 프로세스 중단"], None)
    if phase == LS.PHASE_REFERENCE:
        fraction = min(max(info["elapsed"] / LS.REFERENCE_S, 0.0), 1.0)
        return ("text", [label, "%s / %d초, 행 %d개" % (info["status_text"], int(LS.REFERENCE_S), info["reference_rows"]), REFERENCE_NOTE], fraction)
    if phase == LS.PHASE_SHORT:
        return ("text", [label, info["status_text"]], None)
    if info["windows_scored"] == 0:
        if score_state == SCORE_LOADING:
            return ("text", [label, "점수 프로세스 로딩 중", info["status_text"]], None)
        if score_state == SCORE_DISABLED:
            return ("text", [label, "점수 없음(--no-score)", info["status_text"]], None)
        if info["pending"]:
            return ("text", [label, "점수 계산 대기 중", info["status_text"]], None)
    return ("gauge", info["gauge_state"])


# ----- 점수 자식 프로세스 -----
_STOP = object()


class ScoreClient(object):
    """점수 서비스 자식 프로세스: 쓰기 큐 + 쓰기 스레드 + 읽기 스레드 + 표준 에러를 버리는 스레드. submit 은 큐에 넣기만 하므로 호출한 쪽이 막히지 않는다."""

    def __init__(self, command, on_response, popen=subprocess.Popen, env=None):
        self.command, self.on_response, self.popen, self.env = list(command), on_response, popen, env
        self.queue = queue.Queue()
        self.proc = None
        self.ready = threading.Event()
        self.dead = threading.Event()
        self.closing = threading.Event()
        self.folds = None
        self.sent = 0
        self.received = 0
        self.error_lines = 0
        self.last_error = None
        self.exit_code = None
        self._threads = []

    def start(self):
        kwargs = dict(stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, encoding="utf-8", env=self.env)
        flags = getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 0)
        if flags:
            kwargs["creationflags"] = flags
        self.proc = self.popen(self.command, **kwargs)
        for name, target in (("score-write", self._write_loop), ("score-read", self._read_loop), ("score-stderr", self._drain_stderr)):
            thread = threading.Thread(target=target, name=name)
            thread.daemon = True
            thread.start()
            self._threads.append(thread)
        return self

    def submit(self, message):
        """작업 하나를 큐에 넣는다(막히지 않음). 점수 프로세스가 죽었거나 닫는 중이면 False."""
        if self.dead.is_set() or self.closing.is_set():
            return False
        self.queue.put(message)
        return True

    def state(self):
        if self.dead.is_set():
            return SCORE_STOPPED
        return SCORE_READY if self.ready.is_set() else SCORE_LOADING

    def _write_loop(self):
        while True:
            item = self.queue.get()
            if item is _STOP:
                return
            if self.dead.is_set():
                continue
            try:
                self.proc.stdin.write(json.dumps(item, separators=(",", ":")) + "\n")
                self.proc.stdin.flush()
                self.sent += 1
            except (OSError, ValueError):
                if not self.closing.is_set():
                    self.dead.set()

    def _read_loop(self):
        try:
            for line in self.proc.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    message = json.loads(line)
                except ValueError:
                    self.error_lines += 1
                    continue
                if not isinstance(message, dict):
                    self.error_lines += 1
                elif "ready" in message:
                    self.folds = message.get("folds")
                    self.ready.set()
                elif "error" in message:
                    self.error_lines += 1
                    self.last_error = message.get("error")
                else:
                    self.received += 1
                    try:
                        self.on_response(message)
                    except Exception:
                        self.error_lines += 1
        except (OSError, ValueError):
            pass
        if not self.closing.is_set():
            self.dead.set()

    def _drain_stderr(self):
        try:
            for _line in self.proc.stderr:
                pass                                          # 텍스트는 저장하지 않고 버린다(파이프가 차서 자식이 막히지 않게)
        except (OSError, ValueError):
            pass

    def close(self, timeout=2.0):
        """쓰기 큐를 닫고 표준 입력을 닫아(EOF) 자식이 스스로 끝나게 한 뒤 timeout 초 안에 안 끝나면 terminate, 그래도 안 끝나면 kill. 쓰기가 막혀 있으면 먼저 terminate."""
        self.closing.set()
        self.queue.put(_STOP)
        proc = self.proc
        if proc is None:
            return None
        writer = self._threads[0] if self._threads else None
        if writer is not None:
            writer.join(0.5)
        if writer is not None and writer.is_alive():
            self._terminate(proc)
        else:
            try:
                proc.stdin.close()
            except (OSError, ValueError):
                pass
            try:
                proc.wait(timeout)
            except subprocess.TimeoutExpired:
                self._terminate(proc)
        if proc.poll() is None:
            try:
                proc.kill()
                proc.wait(1.0)
            except Exception:
                pass
        self.exit_code = proc.poll()
        return self.exit_code

    @staticmethod
    def _terminate(proc):
        try:
            proc.terminate()
            proc.wait(1.0)
        except Exception:
            pass


def score_command(ckpt_root, python=None):
    return [python or sys.executable, "-m", "improve.hmlstm.score_service", "--ckpt-root", ckpt_root]


def child_environment():
    """자식 프로세스 환경: 저장소의 src 를 PYTHONPATH 앞에 둔다(자식이 improve 패키지를 찾게)."""
    src = os.path.join(OH.REPO, "src")
    env = dict(os.environ)
    env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


# ----- 요약과 JSON -----
class LiveReporter(object):
    """요약을 만들어 한 번만 출력한다(main 의 finally 와 감시 타이머가 같은 함수를 쓴다). 출력 뒤 바로 flush 한다."""

    def __init__(self, hook, drawer=None, mailbox=None, client=None, out=print, flush=None):
        self.hook, self.drawer, self.mailbox, self.client = hook, drawer, mailbox, client
        self.out = out
        self.flush = flush if flush is not None else sys.stdout.flush
        self.exit_code = None
        self.summary_dict = None
        self._lock = threading.Lock()

    def summary(self):
        hook, drawer = self.hook, self.drawer
        return WH.summarize(list(hook.ticks), list(hook.row_times), list(hook.reads), "thread", drawer.draws if drawer else None,
                            self.mailbox.discarded if self.mailbox else None, list(drawer.draw_times) if drawer else None)

    def extra_lines(self):
        hook = self.hook
        with hook.session_lock:
            description = hook.session.describe()
            windows = list(hook.session.section["windows"])
            sent_jobs = hook.session.next_seq - 1
        last = windows[-1] if windows else None
        client = self.client
        lines = ["live session: phase %s ; reference rows %d ; windows sent %d ; responses received %d ; last score %s" % (
            description["phase"], description["reference_rows"], sent_jobs, len(windows),
            "-" if last is None else "%.2f [%.2f-%.2f] (window %d)" % (last["mean"], last["min"], last["max"], last["k"]))]
        if client is None:
            lines.append("score process: disabled (--no-score)")
        else:
            lines.append("score process: state %s ; ready %s ; folds %s ; exit code %s ; lines written %d ; lines received %d ; error lines %d" % (
                client.state(), "yes" if client.ready.is_set() else "no", client.folds, client.exit_code if client.exit_code is not None else "-",
                client.sent, client.received, client.error_lines))
        check = hook.check
        lines.append("self check (100*TOTAL/face_frames vs Freq100, tolerance %g): rows checked %d ; violations %d ; first violating row %s" % (
            check.tolerance, check.checked, check.violations, "-" if check.first_violation is None else check.first_violation))
        rules = hook.rules
        lines.append("rule alarms: %s ; rule updates %d" % (", ".join("%g: %d" % (threshold, rules.alarm_count(threshold)) for threshold in rules.thresholds), hook.rule_updates))
        lines.append("swallowed exceptions: %d" % hook.errors)
        lines.extend(hook.stats.lines())
        return lines

    def emit(self, status, note=None):
        """요약을 출력하고 요약 dict 를 돌려준다. 이미 출력했으면 None. note 는 고정 문장 바로 앞에 한 줄로 들어간다."""
        with self._lock:
            if self.exit_code is not None:
                return None
            summary = self.summary()
            text = status
            if self.drawer is not None and self.drawer.error:
                text = "%s ; preview thread error: %s" % (status, self.drawer.error)
            lines = WH.format_summary(summary, text)[:-1] + self.extra_lines()
            if note:
                lines.append(note)
            lines.append(NOTE)
            ok = status in ("finished", "interrupted") and summary["frames"] > 0 and not (self.drawer is not None and self.drawer.error)
            self.exit_code = 0 if ok else 1
            self.summary_dict = summary
        for line in lines:
            self.out(line)
        self.flush()
        return summary

    def fire_watchdog(self):
        if self.emit("watchdog", "watchdog exit: the run did not end in time ; summary from the values collected so far") is None:
            self.out("watchdog exit: cleanup after the summary did not finish in time")
            self.flush()
        return self.exit_code


def build_run_document(hook, summary, client, status):
    """--out-json 에 쓸 숫자 문서: 행, 점수, 알람과 에피소드, 처리시간 요약, 자기 점검, 점수 프로세스 상태. 프레임과 이미지는 없다."""
    with hook.session_lock:
        windows = [dict(entry) for entry in hook.session.section["windows"]]
        reference = hook.session.reference
        reference_doc = None if reference is None else {"rows": len(hook.session.ref_rows), "end_time": hook.session.reference_end_time,
                                                         "means": [float(v) for v in reference.means], "stds": [float(v) for v in reference.stds]}
        phase = hook.session.phase
    alarms, episodes = [], []
    for threshold, machine in hook.rules.machines.items():
        alarms.extend(dict(alarm._asdict(), threshold=threshold) for alarm in machine.alarms)
        episodes.extend(dict(episode._asdict(), threshold=threshold) for episode in machine.episodes)
    timing = dict((key, value) for key, value in summary.items() if isinstance(value, (int, float)) and not isinstance(value, bool))
    return {"schema_version": SCHEMA_VERSION, "kind": KIND, "status": status, "fps_assumed": LIVE_FPS, "frames": len(hook.ticks), "rows": list(hook.rows),
            "session_phase": phase, "reference": reference_doc, "scores": windows, "alarms": alarms, "episodes": episodes, "timing": timing,
            "self_check": {"checked": hook.check.checked, "violations": hook.check.violations, "first_violation": hook.check.first_violation},
            "score_process": None if client is None else {"state": client.state(), "ready": client.ready.is_set(), "exit_code": client.exit_code,
                                                          "lines_written": client.sent, "lines_received": client.received, "error_lines": client.error_lines},
            "swallowed_exceptions": hook.errors, "source_stats": hook.stats.summary()}


def validate_run_document(document):
    if not isinstance(document, dict) or document.get("kind") != KIND or document.get("schema_version") != SCHEMA_VERSION:
        raise LiveError("지원하지 않는 문서입니다")
    json.dumps(document, allow_nan=False)                      # 숫자는 유한해야 한다(NaN, 무한대 거부)
    return document


# ----- 그리기(그리기 스레드 안에서만 불린다) -----
def draw_text_strip(cv2, np_module, width, lines, fraction, painter, replay_viewer):
    """게이지 자리에 단계 문구를 그린다(점수가 아직 없을 때). 배경과 색은 replay_viewer 의 게이지 띠와 같다."""
    RV = replay_viewer
    strip = np_module.zeros((RV.GAUGE_H, width, 3), dtype=np_module.uint8)
    strip[:] = RV.GAUGE_BG
    x0, x1, y0, y1 = 10, width - 10, RV.GAUGE_H - 22, RV.GAUGE_H - 8
    cv2.rectangle(strip, (x0, y0), (x1, y1), RV.GAUGE_BAR, -1)
    if fraction is not None and fraction > 0:
        cv2.rectangle(strip, (x0, y0), (x0 + int((x1 - x0) * fraction), y1), RV.GAUGE_RANGE, -1)
    items = [(lines[0], (10, 3), 15, RV.TEXT_DIM)]
    if len(lines) > 1:
        items.append((lines[1], (10, 24), 24 if len(lines[1]) <= 24 else 18, RV.TEXT_MAIN))
    if len(lines) > 2:
        items.append((lines[2], (10, 56), 15, RV.TEXT_DIM))
    painter(strip, items)
    return strip


def draw_live_time_strip(cv2, np_module, width, t, frame_index, duration_s, painter, replay_viewer):
    """시간 줄: 왼쪽 경과 시간(mm:ss.s, 측정 시간이 있으면 " / 전체"), 오른쪽 프레임 번호, 측정 시간이 있으면 진행 막대."""
    RV = replay_viewer
    strip = np_module.zeros((RV.TIME_H, width, 3), dtype=np_module.uint8)
    strip[:] = RV.TIME_BG
    if duration_s:
        left = "%s / %s" % (RV.format_time(t), RV.format_time(duration_s))
        cv2.rectangle(strip, (0, RV.TIME_H - 4), (width - 1, RV.TIME_H - 1), RV.GAUGE_BAR, -1)
        filled = int(round((width - 1) * min(max(t / float(duration_s), 0.0), 1.0)))
        if filled > 0:
            cv2.rectangle(strip, (0, RV.TIME_H - 4), (filled, RV.TIME_H - 1), RV.GAUGE_RANGE, -1)
    else:
        left = RV.format_time(t)
    right = "frame %d" % frame_index
    painter(strip, [(left, (8, 2), 14, RV.TEXT_MAIN), (right, (max(8, width - 10 - int(len(right) * 7.8)), 2), 14, RV.TEXT_DIM)])
    return strip


def make_live_drawer(hook, rotate, duration_s):
    """그리기 스레드가 시작할 때 불러 (draw, close) 를 만드는 factory 를 돌려준다. 글꼴이 없으면 종료 플래그를 세우고 오류를 다시 던진다."""
    angle = ANGLE_BY_OPTION[rotate]

    def factory():
        try:
            import cv2
            from ..rules import replay_viewer as RV
            painter = RV.make_painter(cv2, np)
        except Exception:
            hook.stop_event.set()
            raise
        gauge_cache = {"key": None, "strip": None}
        state = {"seen_visible": False}

        def gauge_strip(info):
            view = gauge_view(info)
            if view[0] == "gauge":
                key = ("gauge", GAUGE.format_gauge_text(view[1]), view[1]["stage"], view[1]["mean"], view[1]["min"], view[1]["max"], view[1]["blinks"])
            else:
                key = ("text", tuple(view[1]), None if view[2] is None else int(view[2] * (RV.DISPLAY_WIDTH - 20)))
            if key != gauge_cache["key"]:
                gauge_cache["strip"] = (RV.draw_gauge_strip(cv2, np, RV.DISPLAY_WIDTH, view[1], painter) if view[0] == "gauge"
                                        else draw_text_strip(cv2, np, RV.DISPLAY_WIDTH, view[1], view[2], painter, RV))
                gauge_cache["key"] = key
            return gauge_cache["strip"]

        def draw(post):
            frame = video_rotation.apply_rotation(post.frame, angle) if angle else post.frame
            cell, scales, offsets = RV.letterbox(cv2, np, frame, RV.DISPLAY_WIDTH, RV.DEFAULT_VIDEO_HEIGHT)
            factor = frame.shape[1] / float(LD.PREPROCESS_WIDTH)
            polygons = [RV.transform_polygon(np, np.asarray(points, dtype=np.float64) * factor, scales, offsets) for points in (post.left, post.right) if points is not None]
            top = LD.draw_ui(cv2, np, cell, polygons, post.rules, post.latch, post.ear, post.ear_min, "", post.t, None, None, True)
            info = hook.live_info(post.frame_index, post.t)
            image = np.vstack([top, draw_live_time_strip(cv2, np, RV.DISPLAY_WIDTH, post.t, post.frame_index, duration_s, painter, RV), gauge_strip(info)])
            cv2.imshow(WINDOW_NAME, image)
            key = cv2.waitKey(1) & 0xFF
            if key in STOP_KEYS:
                hook.stop_event.set()
            elif key in (ord("r"), ord("R")):
                hook.reset_request.set()
            elif key == ord("m"):
                hook.mute_request.set()
            visible = cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE)
            if visible >= 1:
                state["seen_visible"] = True
            elif state["seen_visible"]:
                hook.stop_event.set()                          # 창을 닫았다

        return draw, cv2.destroyAllWindows

    return factory


# ----- 실행 -----
def _silent_print(*_args, **_kwargs):
    return None


def run_source(source, hook):
    """치환된 소스를 메모리에서 실행한다. 반환: 상태 문자열. Ctrl+C 와 예외도 상태로 돌려주고 종료 플래그를 세운다."""
    sink = OH.MemoryBlinkSink(on_row=hook.on_row)
    namespace = {"__name__": "blink_video_inmemory", "open": OH.make_open_override(sink), OH.TICK_NAME: hook.tick, "_hook_read": hook.read, "_hook_ear": hook.ear,
                 "print": _silent_print}
    status = "finished"
    try:
        exec(compile(source, "blink_video.py[live-combined]", "exec"), namespace)
    except KeyboardInterrupt:
        status = "interrupted"
    except Exception as error:
        status = "error: %s: %s" % (type(error).__name__, error)
    finally:
        hook.stop_event.set()
        try:
            sink.flush_pending()
        except Exception:
            pass
        OH.close_gui_windows()
    return status


def write_run_json(path, document):
    """숫자 JSON 을 새 파일로 쓴다(저장소 안과 이미 있는 파일은 TimelineError)."""
    TL.write_json_document(path, document, validate_run_document)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="라이브 웹캠 통합 프로그램: 원본 후킹 경로 한 번으로 규칙 알람과 HM-LSTM 점수 게이지를 한 창에 구동한다(프레임은 저장하지 않음).")
    parser.add_argument("camera", nargs="?", type=int, default=None, help="웹캠 번호(기본 0)")
    parser.add_argument("--camera", dest="camera_option", type=int, default=None, help="웹캠 번호(위치 인자 대신)")
    parser.add_argument("--predictor", required=True, help="dlib 68점 랜드마크 모델(.dat) 경로")
    parser.add_argument("--ckpt-root", dest="ckpt_root", default=None, help="학습 출력 루트(<CKPT_DIR>). --no-score 가 아니면 필요")
    parser.add_argument("--duration", type=float, default=None, help="측정 시간(초). 없으면 q 로 종료")
    parser.add_argument("--rotate", choices=list(ROTATE_CHOICES), default="none", help="회전 옵션(기본 none)")
    parser.add_argument("--preview-hz", dest="preview_hz", type=WH.hz_value, default=WH.DEFAULT_PREVIEW_HZ, help="그리기 주기(초당 횟수, 기본 %g)" % WH.DEFAULT_PREVIEW_HZ)
    parser.add_argument("--no-score", dest="no_score", action="store_true", help="점수 프로세스 없이 규칙 알람 표시만 구동")
    parser.add_argument("--beep", action="store_true", help="알람 순간에 경보음(주 임계값만)")
    parser.add_argument("--out-json", dest="out_json", default=None, help="(선택) 종료 시 숫자 JSON 을 쓸 새 경로(저장소 밖, 덮어쓰지 않음)")
    return parser.parse_args(argv)


def resolve_camera(arguments):
    if arguments.camera is not None and arguments.camera_option is not None and arguments.camera != arguments.camera_option:
        raise LiveError("카메라 번호가 위치 인자와 --camera 에서 서로 다릅니다")
    for value in (arguments.camera, arguments.camera_option):
        if value is not None:
            return value
    return 0


def main(argv=None):
    arguments = parse_args(argv)
    try:
        camera = resolve_camera(arguments)
        if arguments.duration is not None and arguments.duration <= 0:
            raise LiveError("측정 시간은 0보다 커야 합니다.")
        if not arguments.no_score and not arguments.ckpt_root:
            raise LiveError("--ckpt-root 가 필요합니다(점수 없이 돌리려면 --no-score)")
        if not os.path.isfile(arguments.predictor):
            raise LiveError("랜드마크 모델 파일을 찾을 수 없습니다: --predictor 경로를 확인하세요.")
        if arguments.out_json:
            TL.check_output_path(arguments.out_json)
        source = build_source(camera, arguments.predictor, WH.find_svm(), arguments.rotate)
    except (LiveError, TL.TimelineError, OH.ReplacementError, FileNotFoundError) as error:
        print("live_combined: %s" % error)
        return 2
    mailbox = WH.Mailbox()
    beeper = LD.Beeper(enabled=True, beep_all=False, primary=LD.MAIN_THRESHOLD) if arguments.beep else None
    client = None
    hook = LiveHook(duration_s=arguments.duration, preview_hz=arguments.preview_hz, mailbox=mailbox, beeper=beeper)
    if not arguments.no_score:
        def on_response(message):
            with hook.session_lock:
                hook.session.apply_response(message)
        client = ScoreClient(score_command(arguments.ckpt_root), on_response, env=child_environment())
        hook.submit = client.submit
        hook.score_state = client.state
    drawer = WH.DrawThread(mailbox, min(WH.MAX_PREVIEW_HZ, 2.0 * arguments.preview_hz), make_live_drawer(hook, arguments.rotate, arguments.duration))
    reporter = LiveReporter(hook, drawer, mailbox, client)
    started = hook.clock()
    watchdog = None
    if arguments.duration:
        watchdog = WH.Watchdog(lambda: WH.watchdog_due(hook.clock(), started, hook.ticks[0] if hook.ticks else None, arguments.duration),
                               lambda: os._exit(reporter.fire_watchdog()))
        watchdog.start()
    print("라이브 시작: 카메라 %d%s (q 또는 Esc 로 종료, Ctrl+C 로 중단해도 요약을 출력합니다)." % (camera, ", %g초" % arguments.duration if arguments.duration else ""))
    sys.stdout.flush()
    status = "error: the run did not complete"
    try:
        if client is not None:
            client.start()
        drawer.start()
        status = run_source(source, hook)
    finally:
        hook.stop_event.set()
        summary = reporter.emit(status)                       # 요약을 가장 먼저 낸다(아래 정리가 멈춰도 요약은 이미 나와 있다)
        if arguments.out_json and summary is not None:
            try:
                write_run_json(arguments.out_json, build_run_document(hook, summary, client, status))
            except Exception as error:
                print("out-json: %s" % error)
                sys.stdout.flush()
        drawer.stop()
        if client is not None:
            client.close()
        if beeper is not None:
            beeper.stop()
    if watchdog is not None:
        watchdog.cancel()
    return reporter.exit_code if reporter.exit_code is not None else 1


if __name__ == "__main__":
    sys.exit(main())
