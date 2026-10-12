# webcam_hook_speed.py
# 역할: 원본 깜빡임 추출(third_party 의 blink_video.py)을 웹캠 입력으로 후킹해 돌렸을 때의 처리 속도를 사용자가 직접 재는 도구(결정 I-28 (a)안의
#       웹캠 후킹은 실행한 이력이 없다). 점수 계산(TensorFlow)은 하지 않고 cv2 와 dlib 만 쓴다. 실시간 게이지 구현 여부(I-W-37)를 정하는 데 쓴다.
# 방법: original_hook.py 의 치환 방식을 그대로 쓴다. 원본 소스를 텍스트로 읽어 메모리에서 문자열만 바꾼 뒤 exec 한다. 이 도구가 따로 바꾸는 것은
#       영상 경로 치환 한 곳뿐이다: `path = ''` 를 `path = <웹캠 번호>` (정수)로 바꾼다(original_hook.build_replacements 는 문자열 경로만 만든다).
#       나머지 치환(출력 파일 이름, predictor, SVM, plot_frame.draw() 2곳, cv2.imshow 2곳, waitKey 2곳)은 original_hook 의 것과 같다.
#       waitKey 자리에는 진행 호출(tick)이 들어가 반복마다 한 번 불린다. 이 도구의 tick 은 호출 시각을 메모리에 쌓고, 측정 시간이 지나면 0 을 돌려
#       원본 루프를 끝낸다(원본은 waitKey 값이 255 가 아니면 루프를 끝낸다). 깜빡임 행은 original_hook.MemoryBlinkSink 의 on_row 콜백으로 받고 시각만 쌓는다.
# 창: 기본으로는 띄우지 않는다. original_hook 이 cv2.imshow 와 매 프레임 plot_frame.draw() 를 None 으로 바꾸고, 원본이 만드는 빈 Tk 창(blink_video.py 334행
#     `top = tk.Tk()`, 335에서 343행 Frame 과 그림 캔버스)은 이 도구가 그 줄을 `top = tk.Tk(); top.withdraw()` 로 바꿔 숨긴다(줄 수는 그대로, 치환 개수 1).
#     이유: 화면 그리기 시간을 빼고 추출 경로만 재기 위해서이다. 키 입력으로 멈출 수 없고, 측정 시간이 지나거나 Ctrl+C 를 누르면 끝난다(중단돼도 요약을 출력한다).
#     --preview 를 주면 읽은 프레임을 폭 450 이하로 줄여 창에 보여 준다(위치 확인용). 이때 그리기 시간이 처리 시간에 들어가며 요약에 preview on 이라고 적는다.
# --preview-thread: 후킹 루프(메인 스레드)는 tick 에서 최신 프레임을 한 칸 사서함에 넣기만 하고(덮어쓰기, 버려진 수를 셈), 별도 스레드가 --preview-hz 마다 가져와
#     replay_viewer 의 합성 함수로 가짜 게이지 화면(영상 칸 640x480, 시간 줄, 게이지 띠)을 그린다. cv2 창 호출은 그 스레드에서만 한다.
# 처리 여유: 원본의 프레임 읽기 줄(352행)을 시간 재는 래퍼로 바꿔 반복마다 읽기 대기 시간을 쌓고, 인접 tick 간격에서 뺀 값을 처리 시간으로 요약한다.
# 입력: 카메라 번호(기본 0), 측정 시간(초, 기본 60), --predictor(dlib 68점 랜드마크 모델 경로). SVM 은 third_party 의 .sav 한 개를 자동으로 찾는다.
# 출력: 콘솔 요약만(처리 프레임 수, 평균 fps, 프레임 간격의 중앙값, 95 백분위, 최댓값, 확정된 행 수, 마지막 행 시각). 파일, 프레임, 이미지, 영상을 저장하지 않는다.
# 한계: 프레임 간격은 원본 루프가 한 반복을 마칠 때마다 불리는 tick 사이의 시간이라 카메라 대기 시간을 포함한다. 카메라 해상도는 원본이 정하지 않으므로
#       카메라 기본값을 따른다. 읽기에 실패하면(grabbed 가 False) 원본이 루프를 끝낸다.
# 사용: PYTHONPATH=src python -m improve.hmlstm.webcam_hook_speed [카메라 번호] [측정 시간(초)] --predictor <PREDICTOR>
"""웹캠 후킹 속도 측정(Python 3.6 호환)."""
import argparse
import glob
import os
import sys
import threading
import time

import numpy as np

from . import original_hook as OH

DEFAULT_SECONDS = 60.0
NOTE = "이 값은 원본 후킹 경로의 웹캠 속도 측정이며 점수 계산은 포함하지 않음"


PREVIEW_WINDOW = "webcam_hook_speed_preview"
PREVIEW_MAX_WIDTH = 450
READ_TARGET = "(grabbed, frame) = stream.read()"           # blink_video.py 352행: 프레임 읽기. 시간을 재는 래퍼로 바꾼다
READ_REPLACEMENT = "(grabbed, frame) = _hook_read(stream)"
TK_TARGET = "top = tk.Tk()"                                # blink_video.py 334행: 빈 Tk 창의 출처(335에서 343행의 Frame, 그림 캔버스가 이어진다)
TK_REPLACEMENT = "top = tk.Tk(); top.withdraw()"           # 창을 숨긴다. 그림 그리기(343행)는 숨긴 창에서도 된다


class Meter(object):
    """tick 호출 시각, 행 확정 시각, 프레임 읽기 시간을 메모리에 쌓는다. 시각 함수는 시험에서 바꿀 수 있다.

    reads: 반복마다 프레임 읽기 호출에 걸린 시간(초). k번째 값은 k번째 tick 으로 끝나는 반복의 것이다.
    preview: 함수 또는 None. 있으면 tick 마다 마지막으로 읽은 프레임을 넘긴다(그리는 시간은 다음 반복 간격에 들어간다).
    """

    def __init__(self, duration_s, clock=time.perf_counter, preview=None):
        self.duration_s = duration_s
        self.clock = clock
        self.preview = preview
        self.ticks = []
        self.row_times = []
        self.reads = []
        self.last_frame = None

    def read(self, stream):
        """stream.read() 를 부르고 걸린 시간을 쌓는다. 원본의 `(grabbed, frame) = stream.read()` 자리에서 불린다."""
        started = self.clock()
        result = stream.read()
        self.reads.append(self.clock() - started)
        self.last_frame = result[1] if result[0] else None
        return result

    def tick(self):
        now = self.clock()
        self.ticks.append(now)
        if self.preview is not None and self.last_frame is not None:
            self.preview(self.last_frame)
        if now - self.ticks[0] >= self.duration_s:
            return 0                                   # 255 가 아니므로 원본 루프가 끝난다
        return OH.NO_KEY

    def on_row(self, _row):
        self.row_times.append(self.clock())


def preview_size(width, height, max_width=PREVIEW_MAX_WIDTH):
    """미리보기 크기: 폭이 max_width 이하이면 그대로, 크면 폭 max_width 로 종횡비를 지켜 줄인다. 순수 함수."""
    if width <= max_width:
        return width, height
    return max_width, max(1, int(round(height * max_width / float(width))))


def make_preview():
    """읽은 프레임을 폭 450 이하로 줄여 고정 이름의 창에 보여 주는 함수를 만든다. 파일로 저장하지 않는다."""
    import cv2

    def show(frame):
        height, width = frame.shape[:2]
        new_width, new_height = preview_size(width, height)
        image = frame if (new_width, new_height) == (width, height) else cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        cv2.imshow(PREVIEW_WINDOW, image)
        cv2.waitKey(1)

    return show


DEFAULT_PREVIEW_HZ = 15.0
MIN_PREVIEW_HZ = 1.0
MAX_PREVIEW_HZ = 30.0
FAKE_GAUGE_START_FRAME = 29               # 가짜 게이지의 첫 창이 확정된 프레임 번호(블링크 30개째)
FAKE_FRAME_COUNT = 100000                 # 가짜 시간 줄의 전체 프레임 수(막대 길이를 정하는 값일 뿐이다)
STOP_JOIN_TIMEOUT_S = 2.0                 # 그리기 스레드를 정지시킬 때 기다리는 최대 시간
WATCHDOG_GRACE_S = 5.0                    # 첫 프레임 이후 측정 시간 + 이 시간이 지나도 끝나지 않으면 감시 타이머가 요약을 내고 끝낸다
WATCHDOG_STARTUP_S = 60.0                 # 첫 프레임이 오기 전(모델 로드, 카메라 열기)에 허용하는 시간
WATCHDOG_POLL_S = 0.5
DESTROY_TARGET = "cv2.destroyAllWindows()"                 # blink_video.py 570행: 루프 뒤 정리. 메인 스레드에서 불린다
DESTROY_REPLACEMENT = "None"


class Mailbox(object):
    """한 칸 사서함(큐가 아니다). put 은 아직 가져가지 않은 이전 항목을 덮어쓰고 버려진 횟수를 센다. take 는 최신 항목을 가져가며 비운다."""

    def __init__(self):
        self._condition = threading.Condition()
        self._item = None
        self.put_count = 0
        self.discarded = 0

    def put(self, item):
        with self._condition:
            if self._item is not None:
                self.discarded += 1
            self._item = item
            self.put_count += 1
            self._condition.notify()

    def take(self, timeout=0.0):
        """최신 항목을 가져가며 비운다. 항목이 없고 timeout(초)이 0보다 크면 그 시간까지만 기다리고(영원히 기다리지 않는다) 없으면 None 을 돌려준다."""
        with self._condition:
            if self._item is None and timeout and timeout > 0:
                self._condition.wait(timeout)
            item, self._item = self._item, None
            return item

    def pending(self):
        with self._condition:
            return self._item is not None


def preview_mode(preview, preview_thread):
    """미리보기 방식: --preview-thread 가 --preview 보다 우선한다. "off", "inline", "thread" 중 하나."""
    if preview_thread:
        return "thread"
    return "inline" if preview else "off"


def hz_value(text):
    """--preview-hz 값 검사: MIN_PREVIEW_HZ 이상 MAX_PREVIEW_HZ 이하의 수."""
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError("수여야 합니다: %r" % text)
    if not MIN_PREVIEW_HZ <= value <= MAX_PREVIEW_HZ:
        raise argparse.ArgumentTypeError("%g 이상 %g 이하여야 합니다: %g" % (MIN_PREVIEW_HZ, MAX_PREVIEW_HZ, value))
    return value


class DrawThread(threading.Thread):
    """사서함의 최신 프레임을 hz 마다 가져와 그리는 스레드. highgui 호출(창 만들기, 그리기, 닫기)은 이 스레드에서만 한다.

    drawer_factory() 는 (draw, close) 를 돌려주며 이 스레드 안에서 불린다. 늦으면 따라잡지 않고 다음 주기로 넘어간다.
    draws: 그린 횟수, draw_times: 한 번 그리는 데 걸린 시간(초), error: 스레드 안에서 난 오류 문장(없으면 None).
    """

    def __init__(self, mailbox, hz, drawer_factory, clock=time.perf_counter):
        super(DrawThread, self).__init__(name="preview-draw")
        self.daemon = True
        self.mailbox = mailbox
        self.period = 1.0 / hz
        self.drawer_factory = drawer_factory
        self.clock = clock
        self.stop_event = threading.Event()
        self.draws = 0
        self.draw_times = []
        self.error = None
        self.thread_ids = set()

    def run(self):
        close = None
        try:
            draw, close = self.drawer_factory()
            self.thread_ids.add(threading.get_ident())
            next_time = self.clock()
            while not self.stop_event.is_set():
                frame = self.mailbox.take()
                if frame is not None:
                    started = self.clock()
                    draw(frame)
                    self.draw_times.append(self.clock() - started)
                    self.draws += 1
                next_time += self.period
                wait = next_time - self.clock()
                if wait < 0:
                    next_time = self.clock()
                    wait = 0.0
                self.stop_event.wait(wait)
        except Exception as error:
            self.error = "%s: %s" % (type(error).__name__, error)
        finally:
            if close is not None:
                try:
                    close()
                except Exception:
                    pass

    def stop(self, timeout=STOP_JOIN_TIMEOUT_S):
        """정지 신호를 보내고 timeout 초까지만 기다린다. 그래도 끝나지 않으면 기다리지 않고 돌아온다(스레드는 데몬). 끝났으면 True."""
        self.stop_event.set()
        if self.is_alive():
            self.join(timeout)
        return not self.is_alive()


def fake_gauge_section():
    """가짜 게이지 값의 시간선 구간: 블링크 30개가 확정되고 첫 창(점수 5.0 [4.0-6.0])이 FAKE_GAUGE_START_FRAME 프레임에 확정된 상태. 순수 함수."""
    return {"fps": 30.0, "blink_frame_indices": list(range(FAKE_GAUGE_START_FRAME + 1)),
            "windows": [{"k": 1, "frame_index": FAKE_GAUGE_START_FRAME, "mean": 5.0, "min": 4.0, "max": 6.0}]}


def make_fake_screen_drawer():
    """실제 게이지 화면과 비슷한 비용으로 그리는 (draw, close) 를 만든다. replay_viewer 의 합성 함수를 그대로 쓴다.

    그리는 것: 영상 칸 640 x 480(종횡비 유지, 검정 여백), 규칙 화면(draw_ui clean), 시간 줄, 게이지 띠(글자 줄과 막대). 값은 가짜이다(얼굴 없음, 규칙 알람 없음,
    점수 5.0 [4.0-6.0]). 시간 줄과 게이지의 갱신 경과 시간은 그릴 때마다 바뀌어 replay_viewer 가 재생 중 다시 그리는 만큼 한글 그리기 비용이 든다.
    이 함수는 DrawThread 안에서 불러야 한다(cv2 창 호출이 그 스레드에서만 일어나게).
    """
    import cv2
    from improve.rules import replay_viewer as RV

    document = {"fps": 30.0, "frames": [{"face": False, "ear": None, "left": None, "right": None}]}
    engine = RV.ReplayEngine(document, (0.15,))
    state = engine.advance_to(0)
    painter = RV.make_painter(cv2, np)
    gauge = RV.GaugeStrip(cv2, np, RV.DISPLAY_WIDTH, painter)
    time_line = RV.TimeStrip(cv2, np, RV.DISPLAY_WIDTH, painter)
    section = fake_gauge_section()
    counter = [0]

    def draw(frame):
        counter[0] += 1
        gauge_state = RV.GAUGE.display_state(section, FAKE_GAUGE_START_FRAME + counter[0] * 2)
        strip = gauge.render(gauge_state)
        row = time_line.render(counter[0], FAKE_FRAME_COUNT, 30.0, False)
        cv2.imshow(PREVIEW_WINDOW, RV.compose(cv2, np, frame, 0, engine, state, strip, RV.DEFAULT_VIDEO_HEIGHT, row))
        cv2.waitKey(1)

    return draw, cv2.destroyAllWindows


def summarize(ticks, row_times, reads=None, mode="off", draws=None, discarded=None, draw_times=None):
    """tick 시각 목록과 행 시각 목록에서 요약 값(dict)을 만든다. 순수 함수. 값을 낼 수 없는 항목은 None.

    frames: tick 횟수. fps: (frames - 1) / (마지막 tick - 첫 tick). gap_*_ms: 인접 tick 간격(ms)의 중앙값, 95 백분위, 최댓값.
    rows: 확정된 행 수. last_row_s: 마지막 행 시각 - 첫 tick 시각(초, 행이 없으면 None).
    reads 가 있으면(반복마다 프레임 읽기 시간, 초): k번째 반복의 처리 시간 = (k-1번째 tick 에서 k번째 tick 까지의 간격) - (k번째 읽기 시간), k = 1 부터.
    process_*_ms 는 그 처리 시간의 중앙값, 95 백분위, 최댓값, read_median_ms 는 읽기 시간(첫 반복 제외)의 중앙값, budget_ms 는 프레임 예산 1000 / 평균 fps,
    process_median_ratio 와 process_p95_ratio 는 처리 시간을 예산으로 나눈 값이다.
    mode 는 미리보기 방식("off", "inline", "thread")이며 preview 는 mode 가 off 가 아닐 때 True 이다. draws 는 그린 횟수, discarded 는 사서함에서 덮어써 버려진 프레임 수,
    draw_times(초)가 있으면 한 번 그리는 시간의 중앙값과 95 백분위가 draw_median_ms, draw_p95_ms 에 들어간다.
    """
    frames = len(ticks)
    summary = {"frames": frames, "fps": None, "gap_median_ms": None, "gap_p95_ms": None, "gap_max_ms": None,
               "rows": len(row_times), "last_row_s": None, "span_s": None,
               "process_median_ms": None, "process_p95_ms": None, "process_max_ms": None, "read_median_ms": None,
               "budget_ms": None, "process_median_ratio": None, "process_p95_ratio": None, "preview": mode != "off",
               "preview_mode": mode, "preview_draws": draws, "preview_discarded": discarded, "draw_median_ms": None, "draw_p95_ms": None}
    if draw_times:
        summary["draw_median_ms"] = float(np.median(draw_times)) * 1000.0
        summary["draw_p95_ms"] = float(np.percentile(draw_times, 95)) * 1000.0
    if frames >= 2:
        gaps = np.diff(np.asarray(ticks, dtype=np.float64)) * 1000.0
        span = ticks[-1] - ticks[0]
        summary["span_s"] = span
        summary["fps"] = (frames - 1) / span if span > 0 else None
        summary["gap_median_ms"] = float(np.median(gaps))
        summary["gap_p95_ms"] = float(np.percentile(gaps, 95))
        summary["gap_max_ms"] = float(np.max(gaps))
        if reads is not None and len(reads) >= frames:
            read_ms = np.asarray(reads[1:frames], dtype=np.float64) * 1000.0
            process = np.maximum(gaps - read_ms, 0.0)
            summary["process_median_ms"] = float(np.median(process))
            summary["process_p95_ms"] = float(np.percentile(process, 95))
            summary["process_max_ms"] = float(np.max(process))
            summary["read_median_ms"] = float(np.median(read_ms))
            if summary["fps"]:
                summary["budget_ms"] = 1000.0 / summary["fps"]
                summary["process_median_ratio"] = summary["process_median_ms"] / summary["budget_ms"]
                summary["process_p95_ratio"] = summary["process_p95_ms"] / summary["budget_ms"]
    if row_times and ticks:
        summary["last_row_s"] = row_times[-1] - ticks[0]
    return summary


def format_summary(summary, status="finished"):
    """콘솔에 찍을 요약 줄 목록(마지막 줄은 고정 문구)."""
    def number(value, pattern):
        return "-" if value is None else pattern % value

    return [
        "status: %s" % status,
        "frames: %d" % summary["frames"],
        "mean fps: %s" % number(summary["fps"], "%.2f"),
        "frame gap median / p95 / max (ms): %s / %s / %s" % (number(summary["gap_median_ms"], "%.1f"), number(summary["gap_p95_ms"], "%.1f"),
                                                            number(summary["gap_max_ms"], "%.1f")),
        "blink rows confirmed: %d" % summary["rows"],
        "last row at (s after first frame): %s" % number(summary["last_row_s"], "%.1f"),
        "measured span (s): %s" % number(summary["span_s"], "%.1f"),
        "read wait median (ms): %s" % number(summary.get("read_median_ms"), "%.1f"),
        "processing median / p95 / max (ms): %s / %s / %s" % (number(summary.get("process_median_ms"), "%.1f"), number(summary.get("process_p95_ms"), "%.1f"),
                                                             number(summary.get("process_max_ms"), "%.1f")),
        "frame budget (ms): %s ; processing / budget (median, p95): %s / %s" % (number(summary.get("budget_ms"), "%.1f"),
                                                                                number(summary.get("process_median_ratio"), "%.2f"),
                                                                                number(summary.get("process_p95_ratio"), "%.2f")),
        "preview mode: %s ; draws: %s ; frames overwritten in mailbox: %s ; draw time median / p95 (ms): %s / %s" % (
            summary.get("preview_mode", "off"), "-" if summary.get("preview_draws") is None else "%d" % summary["preview_draws"],
            "-" if summary.get("preview_discarded") is None else "%d" % summary["preview_discarded"],
            number(summary.get("draw_median_ms"), "%.1f"), number(summary.get("draw_p95_ms"), "%.1f")),
        "preview: %s" % ("on" if summary.get("preview") else "off"),
        NOTE,
    ]


def find_svm():
    """third_party 의 SVM 파일(.sav)을 찾는다. 정확히 한 개여야 한다."""
    found = sorted(glob.glob(os.path.join(os.path.dirname(OH.ORIGINAL_PATH), "*.sav")))
    if len(found) != 1:
        raise FileNotFoundError("third_party 에서 SVM(.sav) 파일 하나를 찾지 못했습니다: %d개" % len(found))
    return found[0]


def build_source(camera, predictor, svm, original=None, thread_mode=False):
    """원본 소스를 웹캠 입력용으로 치환한 문자열. 영상 경로 자리에 정수 카메라 번호가 들어간다. 치환 개수가 기대와 다르면 ReplacementError.

    thread_mode 이면 원본의 루프 뒤 정리 줄 `cv2.destroyAllWindows()`(blink_video.py 570행)를 무동작으로 바꾼다. 이 줄은 메인 스레드에서 실행되므로
    다른 스레드가 만든 창을 닫으려다 멈출 수 있다. 창은 그것을 만든 그리기 스레드가 닫는다.
    """
    original = OH.read_original_source() if original is None else original
    replacements = []
    for label, old, new, expected in OH.build_replacements("", predictor, svm, "none", True):
        if label == "input video":
            new = "path = %d" % int(camera)
        replacements.append((label, old, new, expected))
    replacements.append(("read timing wrapper", READ_TARGET, READ_REPLACEMENT, 1))
    replacements.append(("hide empty Tk window", TK_TARGET, TK_REPLACEMENT, 1))
    if thread_mode:
        replacements.append(("skip destroyAllWindows in the main thread", DESTROY_TARGET, DESTROY_REPLACEMENT, 1))
    OH.check_counts(original, replacements)
    return OH.apply_replacements(original, replacements)


def _silent_print(*_args, **_kwargs):
    return None


def run_measurement(source, meter):
    """치환된 소스를 메모리에서 실행하고 상태 문자열을 돌려준다. 중단이나 예외가 나도 쌓인 값은 meter 에 남는다."""
    sink = OH.MemoryBlinkSink(on_row=meter.on_row)
    namespace = {"__name__": "blink_video_inmemory", "open": OH.make_open_override(sink), OH.TICK_NAME: meter.tick, "_hook_read": meter.read, "print": _silent_print}
    status = "finished"
    try:
        exec(compile(source, "blink_video.py[webcam-hook-speed]", "exec"), namespace)
    except KeyboardInterrupt:
        status = "interrupted"
    except Exception as error:
        status = "error: %s: %s" % (type(error).__name__, error)
    finally:
        try:
            sink.flush_pending()
        except Exception:
            pass
        OH.close_gui_windows()
    return status


def watchdog_due(now, started, first_tick, duration_s, grace_s=WATCHDOG_GRACE_S, startup_s=WATCHDOG_STARTUP_S):
    """감시 타이머가 발동할 때인지. 순수 함수.

    첫 프레임(first_tick)이 이미 왔으면 첫 프레임 이후 duration_s + grace_s 가 지났을 때, 아직 안 왔으면 시작 이후 startup_s 가 지났을 때 True 이다.
    측정 시간은 첫 프레임부터 재므로 모델 로드와 카메라 열기 시간은 기준에 넣지 않는다.
    """
    if first_tick is not None:
        return now - first_tick >= duration_s + grace_s
    return now - started >= startup_s


class Watchdog(object):
    """데몬 스레드가 poll_s 마다 is_due() 를 보고, 참이면 on_fire() 를 한 번 부른 뒤 끝난다. cancel() 하면 발동하지 않는다. 모든 대기에 시간 제한이 있다."""

    def __init__(self, is_due, on_fire, poll_s=WATCHDOG_POLL_S):
        self.is_due = is_due
        self.on_fire = on_fire
        self.poll_s = poll_s
        self.cancel_event = threading.Event()
        self.fired = False
        self._thread = None

    def check(self):
        """한 번 점검한다: 취소되지 않았고 발동할 때이며 아직 발동하지 않았으면 on_fire 를 부르고 True. 시험에서 직접 부를 수 있다."""
        if self.cancel_event.is_set() or self.fired or not self.is_due():
            return False
        self.fired = True
        self.on_fire()
        return True

    def _run(self):
        while not self.cancel_event.wait(self.poll_s):
            if self.check():
                return

    def start(self):
        self._thread = threading.Thread(target=self._run, name="watchdog")
        self._thread.daemon = True
        self._thread.start()

    def cancel(self):
        self.cancel_event.set()


class SummaryReporter(object):
    """요약을 만들어 한 번만 출력한다(main 과 감시 타이머가 같은 함수를 쓴다). 출력 뒤 바로 flush 한다."""

    def __init__(self, meter, mode, mailbox=None, drawer=None, inline_draws=None, out=print, flush=None):
        self.meter, self.mode, self.mailbox, self.drawer = meter, mode, mailbox, drawer
        self.inline_draws = inline_draws if inline_draws is not None else [0]
        self.out = out
        self.flush = flush if flush is not None else sys.stdout.flush
        self.exit_code = None
        self._lock = threading.Lock()

    def summary(self):
        meter = self.meter
        if self.drawer is not None:
            return summarize(list(meter.ticks), list(meter.row_times), list(meter.reads), self.mode, self.drawer.draws, self.mailbox.discarded,
                             list(self.drawer.draw_times))
        return summarize(list(meter.ticks), list(meter.row_times), list(meter.reads), self.mode, self.inline_draws[0] if self.mode == "inline" else None)

    def emit(self, status, note=None):
        """요약을 출력하고 요약 dict 를 돌려준다. 이미 출력했으면 아무것도 하지 않고 None. note 는 마지막 고정 문구 바로 앞에 한 줄로 들어간다."""
        with self._lock:
            if self.exit_code is not None:
                return None
            summary = self.summary()
            text = status
            if self.drawer is not None and self.drawer.error:
                text = "%s ; preview thread error: %s" % (status, self.drawer.error)
            lines = format_summary(summary, text)
            if note:
                lines.insert(len(lines) - 1, note)
            self.exit_code = 0 if status in ("finished", "interrupted") and summary["frames"] > 0 else 1
        for line in lines:
            self.out(line)
        self.flush()
        return summary

    def fire_watchdog(self):
        """감시 타이머가 발동했을 때: 아직 요약이 없으면 지금까지 모은 값으로 요약을 내고, 이미 냈으면 정리가 끝나지 않았다는 한 줄만 낸다. 종료 코드를 돌려준다."""
        if self.emit("watchdog", "watchdog exit: the run did not end in time ; summary from the values collected so far") is None:
            self.out("watchdog exit: cleanup after the summary did not finish in time")
            self.flush()
        return self.exit_code


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="원본 깜빡임 추출을 웹캠 입력으로 후킹했을 때의 처리 속도를 잰다(콘솔 요약만, 아무것도 저장하지 않음).")
    parser.add_argument("camera", nargs="?", type=int, default=0, help="웹캠 번호(기본 0)")
    parser.add_argument("seconds", nargs="?", type=float, default=DEFAULT_SECONDS, help="측정 시간(초, 기본 %g)" % DEFAULT_SECONDS)
    parser.add_argument("--predictor", required=True, help="dlib 68점 랜드마크 모델(.dat) 경로")
    parser.add_argument("--preview", action="store_true",
                        help="읽은 프레임을 폭 450 이하로 줄여 창에 보여 준다(기본 꺼짐). 그리는 시간이 처리 시간에 들어가므로 요약에 preview on/off 를 적는다")
    parser.add_argument("--preview-thread", dest="preview_thread", action="store_true",
                        help="별도 스레드에서 실제 게이지 화면과 비슷한 가짜 화면(영상 칸 640x480, 시간 줄, 게이지 띠)을 그린다(기본 꺼짐, --preview 와 함께 쓰면 이것이 우선). "
                             "후킹 루프는 최신 프레임 한 장을 사서함에 넣기만 한다")
    parser.add_argument("--preview-hz", dest="preview_hz", type=hz_value, default=DEFAULT_PREVIEW_HZ,
                        help="--preview-thread 의 그리기 주기(초당 횟수, %g에서 %g, 기본 %g)" % (MIN_PREVIEW_HZ, MAX_PREVIEW_HZ, DEFAULT_PREVIEW_HZ))
    return parser.parse_args(argv)


def main(argv=None):
    arguments = parse_args(argv)
    if arguments.seconds <= 0:
        print("측정 시간은 0보다 커야 합니다.")
        return 2
    if not os.path.isfile(arguments.predictor):
        print("랜드마크 모델 파일을 찾을 수 없습니다: --predictor 경로를 확인하세요.")
        return 1
    mode = preview_mode(arguments.preview, arguments.preview_thread)
    source = build_source(arguments.camera, arguments.predictor, find_svm(), thread_mode=(mode == "thread"))
    print("측정 시작: 카메라 %d, %g초(Ctrl+C 로 중단해도 요약을 출력합니다). 미리보기 옵션이 없으면 창은 뜨지 않습니다." % (arguments.camera, arguments.seconds))
    sys.stdout.flush()
    mailbox = drawer = None
    inline_draws = [0]
    if mode == "thread":
        mailbox = Mailbox()
        drawer = DrawThread(mailbox, arguments.preview_hz, make_fake_screen_drawer)
        drawer.start()                                     # highgui 호출은 이 스레드에서만 한다
        preview = mailbox.put                              # 후킹 루프는 최신 프레임을 사서함에 넣기만 한다
    elif mode == "inline":
        show = make_preview()

        def preview(frame):
            inline_draws[0] += 1
            show(frame)
    else:
        preview = None
    meter = Meter(arguments.seconds, preview=preview)
    reporter = SummaryReporter(meter, mode, mailbox, drawer, inline_draws)
    started = meter.clock()
    watchdog = Watchdog(lambda: watchdog_due(meter.clock(), started, meter.ticks[0] if meter.ticks else None, arguments.seconds),
                        lambda: os._exit(reporter.fire_watchdog()))
    watchdog.start()
    status = "error: the measurement did not complete"
    try:
        status = run_measurement(source, meter)
    finally:
        reporter.emit(status)                              # 요약을 먼저 낸다(이 아래의 정리가 멈춰도 요약은 이미 나와 있다)
        if drawer is not None:
            drawer.stop()                                  # 정지 신호 + 제한 시간 join. 창은 그리기 스레드가 닫는다
        elif mode == "inline":
            import cv2
            cv2.destroyAllWindows()                        # 인라인 창은 이 스레드가 만들었다
    watchdog.cancel()
    return reporter.exit_code if reporter.exit_code is not None else 1


if __name__ == "__main__":
    sys.exit(main())
