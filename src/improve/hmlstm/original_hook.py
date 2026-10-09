# original_hook.py
# 역할: 저자 원본 blink_video.py(third_party)를 수정하지 않고, 텍스트로 읽어 메모리에서 문자열 치환만 한 뒤 실행하고,
#       원본이 파일에 쓰는 깜빡임 행을 파일 대신 메모리로 받는다(결정 I-28 (a)안의 1단계). 영상 파일 입력만 다루며
#       웹캠 연결과 HM-LSTM 점수 계산은 다루지 않는다. 원본 코드를 이 파일로 복사해 넣지 않는다(치환 대상 문자열만 가진다).
# 방법: exec 네임스페이스(globals)에 open 대체 함수를 넣는다. 출력 파일 이름(OUTPUT_SENTINEL)을 append 로 여는 호출만 메모리
#       기록기(MemoryBlinkSink)를 돌려주고, 나머지 호출(특히 315행의 SVM pickle 읽기)은 실제 open 에 넘긴다.
#       원본이 쓰는 형식은 f_handle.write(b'\n') 뒤 np.savetxt(f_handle, 5개 값, delimiter=', ', newline=' ', fmt='%.4f') 이며
#       numpy 1.14.6 의 savetxt 는 값을 하나씩 문자열로 write 한다. 기록기는 bytes 와 str 을 모두 받고 줄바꿈 단위로 행을 만든다.
# B1 추출(src/repro/reproduction_blink_headless.py)과 같은 치환: 출력 경로, 영상 경로, predictor 경로, plot_frame.draw() 2곳,
#       cv2.imshow 2곳, cv2.waitKey(1) 2곳, 회전이면 resize 1곳. 이 파일에서 다른 점은 SVM 경로 문자열 치환, 진행 표시용 waitKey 치환
#       (진행 표시나 부분 실행을 켰을 때만), print 무음 처리(네임스페이스), 작업 폴더를 바꾸지 않는 것이다.
# 사용: 같은 패키지의 verify_blinks.py 와 이후 라이브 도구가 `from improve.hmlstm import original_hook` 로 가져다 쓴다.
"""blink_video.py 메모리 치환 실행과 깜빡임 행 수신(Python 3.6 호환).

extract_rows(video, svm, predictor, rotate="none", ...) 가 ExtractionResult 를 돌려준다.
파일은 어떤 것도 쓰지 않는다. 읽는 파일은 third_party 의 원본 소스, SVM, predictor, 영상뿐이다.
"""
import builtins
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
ORIGINAL_PATH = os.path.join(REPO, "third_party", "ghoddoosian", "blink_video.py")

OUTPUT_SENTINEL = "<memory:blink_rows>"     # 원본의 출력 파일 자리에 넣는 이름. 실제 경로가 아니며 이 이름의 append 열기만 메모리로 간다
TICK_NAME = "_hook_progress_tick"           # 진행 표시용 waitKey 대체 호출 이름(네임스페이스에 넣는다)
NO_KEY = 255                                # waitKey 가 키 없음일 때 돌려주는 값. 이 값이 아니면 원본 루프가 끝난다(483, 565행)

DRAW_INDENT = " " * 16                      # 매 프레임 draw 호출의 들여쓰기(Tk 생성부의 draw 는 4칸이라 구분된다)
OUTPUT_TARGET = "output_file = 'alert.txt'"
PATH_TARGET = "path = ''"
PREDICTOR_TARGET = "'shape_predictor_68_face_landmarks.dat'"
SVM_TARGET = "'Trained_SVM_C=1000_gamma=0.1_for 7kNegSample.sav'"
DRAW_TARGET = DRAW_INDENT + "plot_frame.draw()"
IMSHOW_TARGET = 'cv2.imshow("Frame", frameMinus7)'
WAITKEY_TARGET = "cv2.waitKey(1)"
RESIZE_TARGET = "frame = imutils.resize(frame, width=450)"

ROTATE_OPTIONS = ("none", "cw90", "ccw90", "180")
_ROTATE_CONSTANTS = {"cw90": "ROTATE_90_CLOCKWISE", "ccw90": "ROTATE_90_COUNTERCLOCKWISE", "180": "ROTATE_180"}


class ReplacementError(ValueError):
    """치환 대상의 일치 횟수가 기대와 다르거나 치환 뒤 줄 수가 달라졌을 때."""


def rotation_expression(rotate):
    """프레임 변수 frame 에 적용할 회전 표현식 문자열. none 이면 "frame"."""
    if rotate not in ROTATE_OPTIONS:
        raise ValueError("알 수 없는 회전 옵션: %s" % rotate)
    if rotate == "none":
        return "frame"
    return "cv2.rotate(frame, cv2.%s)" % _ROTATE_CONSTANTS[rotate]


def build_replacements(video, predictor, svm, rotate="none", tick=False):
    """메모리에서 바꿀 문자열 목록을 만든다. 각 항목은 (설명, 원문, 새 문장, 기대 일치 횟수).

    입력: video, predictor, svm - 파일 경로 문자열(치환 문장에만 들어간다), rotate - 회전 옵션,
          tick - True 이면 waitKey 자리에 진행 표시 호출(값 255 를 돌려줌)을 넣고, False 이면 B1 과 같이 255 를 넣는다
    반환: 튜플 list. 줄을 지우지 않고 표현식만 바꾼다.
    """
    replacements = [
        ("output file", OUTPUT_TARGET, "output_file = %r" % OUTPUT_SENTINEL, 1),
        ("input video", PATH_TARGET, "path = %r" % video, 1),
        ("predictor path", PREDICTOR_TARGET, repr(predictor), 1),
        ("svm path", SVM_TARGET, repr(svm), 1),
        ("plot_frame.draw() per frame", DRAW_TARGET, DRAW_INDENT + "None", 2),
        ("cv2.imshow", IMSHOW_TARGET, "None", 2),
        ("cv2.waitKey(1)", WAITKEY_TARGET, ("%s()" % TICK_NAME) if tick else str(NO_KEY), 2),
    ]
    if rotate != "none":
        replacements.append(("rotate before resize", RESIZE_TARGET, "frame = imutils.resize(%s, width=450)" % rotation_expression(rotate), 1))
    return replacements


def count_matches(source, replacements):
    """각 치환 대상이 소스에 몇 번 나오는지 센다. 반환: (설명, 기대, 실제) 튜플 list."""
    return [(label, expected, source.count(old)) for label, old, _new, expected in replacements]


def check_counts(source, replacements):
    """모든 치환 대상이 기대 횟수와 정확히 같을 때만 통과한다. 하나라도 다르면 ReplacementError."""
    problems = ["[%s] 기대 %d회, 실제 %d회" % (label, expected, actual)
                for label, expected, actual in count_matches(source, replacements) if expected != actual]
    if problems:
        raise ReplacementError("치환 대상 개수가 달라 중단합니다: " + "; ".join(problems))


def apply_replacements(source, replacements):
    """check_counts 를 통과한 소스에 치환을 적용한다. 줄 수가 달라지면 ReplacementError."""
    modified = source
    for _label, old, new, _expected in replacements:
        modified = modified.replace(old, new)
    if len(modified.split("\n")) != len(source.split("\n")):
        raise ReplacementError("치환 후 줄 수가 달라졌습니다(줄을 지우면 안 됩니다)")
    return modified


def changed_line_numbers(original, modified):
    """치환 전후에서 내용이 달라진 줄 번호(1부터)."""
    return [index + 1 for index, (before, after) in enumerate(zip(original.split("\n"), modified.split("\n"))) if before != after]


def read_original_source(path=None):
    """third_party 의 blink_video.py 를 텍스트로 읽는다(B1 과 같은 읽기 방식, 파일은 수정하지 않는다)."""
    with open(path or ORIGINAL_PATH, "r", encoding="utf-8") as handle:
        return handle.read()


class MemoryBlinkSink(object):
    """원본이 append 로 여는 출력 파일을 대신하는 메모리 기록기. bytes 와 str 쓰기를 모두 받는다.

    행은 줄바꿈 단위로 만든다. 줄바꿈 뒤의 값들은 with 블록이 끝날 때(__exit__, close) 한 행으로 확정된다.
    rows: float 배열 목록, raw: 받은 원문 바이트 전체, on_row: 행이 확정될 때마다 부르는 함수(선택)
    """

    def __init__(self, on_row=None):
        self.rows = []
        self.on_row = on_row
        self._raw = bytearray()
        self._pending = ""

    @property
    def raw(self):
        return bytes(self._raw)

    def write(self, data):
        if isinstance(data, (bytes, bytearray)):
            self._raw.extend(data)
            text = bytes(data).decode("latin-1")
        elif isinstance(data, str):
            self._raw.extend(data.encode("utf-8"))
            text = data
        else:
            raise TypeError("bytes 또는 str 만 쓸 수 있습니다: %s" % type(data).__name__)
        self._pending += text
        if "\n" in self._pending:
            parts = self._pending.split("\n")
            self._pending = parts[-1]
            for line in parts[:-1]:
                self._emit(line)
        return len(data)

    def _emit(self, line):
        if line.strip() == "":
            return
        try:
            row = np.array([float(token) for token in line.split()], dtype=float)
        except ValueError:
            raise ValueError("깜빡임 행을 읽을 수 없습니다: %r" % line)
        self.rows.append(row)
        if self.on_row is not None:
            self.on_row(row)

    def flush_pending(self):
        pending, self._pending = self._pending, ""
        self._emit(pending)

    def flush(self):
        return None

    def close(self):
        self.flush_pending()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.flush_pending()
        return False


def make_open_override(sink, sentinel=OUTPUT_SENTINEL, real_open=None):
    """exec 네임스페이스에 넣을 open 대체 함수를 만든다.

    sentinel 을 append 모드로 여는 호출만 sink 를 돌려주고, 다른 모든 호출(SVM pickle 읽기 포함)은 real_open(기본 내장 open)에 그대로 넘긴다.
    sentinel 을 다른 모드로 열면 ValueError 이다(원본은 append 만 쓴다).
    """
    delegate = real_open if real_open is not None else builtins.open

    def open_override(file, mode="r", *args, **kwargs):
        if file == sentinel:
            if "a" in mode:
                return sink
            raise ValueError("메모리 출력은 append 모드만 지원합니다: mode=%r" % mode)
        return delegate(file, mode, *args, **kwargs)

    return open_override


def _silent_print(*args, **kwargs):
    return None


def run_source(source, sink, quiet=True, progress_every=0, total_frames=None, max_frames=0):
    """치환된 소스를 메모리에서 컴파일해 실행한다(원본 하단의 main 부분이 그대로 실행된다).

    입력: source - apply_replacements 결과, sink - MemoryBlinkSink, quiet - True 이면 원본의 print 를 무음으로(네임스페이스만 바꾼다),
          progress_every - N프레임마다 한 줄 출력(0 이면 안 함), total_frames - 진행 표시용 전체 프레임 수(선택),
          max_frames - 0 이 아니면 그 프레임 수에서 루프를 끝낸다(부분 실행, 합격 판정과 무관)
    반환: 진행 호출 횟수로 센 처리 프레임 수(진행 표시 치환을 쓰지 않았으면 0)
    """
    counter = {"frames": 0}
    real_print = builtins.print

    def tick():
        counter["frames"] += 1
        frames = counter["frames"]
        if progress_every and frames % progress_every == 0:
            total_text = "/%d" % total_frames if total_frames else ""
            real_print("frames=%d%s rows=%d" % (frames, total_text, len(sink.rows)))
            sys.stdout.flush()
        if max_frames and frames >= max_frames:
            return 0
        return NO_KEY

    namespace = {"__name__": "blink_video_inmemory", "open": make_open_override(sink), TICK_NAME: tick}
    if quiet:
        namespace["print"] = _silent_print
    exec(compile(source, "blink_video.py[hook-in-memory]", "exec"), namespace)
    sink.flush_pending()
    return counter["frames"]


def close_gui_windows():
    """원본이 만든 Tk 창과 matplotlib 그림을 실행이 끝난 뒤 닫는다(실패해도 무시한다)."""
    tkinter = sys.modules.get("tkinter")
    root = getattr(tkinter, "_default_root", None) if tkinter is not None else None
    if root is not None:
        try:
            root.destroy()
        except Exception:
            pass
    pyplot = sys.modules.get("matplotlib.pyplot")
    if pyplot is not None:
        try:
            pyplot.close("all")
        except Exception:
            pass


class ExtractionResult(object):
    """extract_rows 의 결과. rows: float 배열 목록, raw: 원문 바이트, frames: 진행 호출로 센 프레임 수, elapsed_s, changed_lines: 치환된 줄 번호."""

    def __init__(self, rows, raw, frames, elapsed_s, changed_lines):
        self.rows = rows
        self.raw = raw
        self.frames = frames
        self.elapsed_s = elapsed_s
        self.changed_lines = changed_lines


def extract_rows(video, svm, predictor, rotate="none", progress_every=0, max_frames=0, quiet=True, source=None, total_frames=None, on_row=None):
    """영상 1개에 원본을 메모리 치환으로 실행하고 깜빡임 행을 돌려준다. 디스크에 아무 파일도 쓰지 않는다.

    입력: video, svm, predictor - 존재하는 파일 경로, rotate - ROTATE_OPTIONS 중 하나, source - 원본 소스 문자열(없으면 third_party 에서 읽음)
    반환: ExtractionResult. 치환 개수가 다르면 영상을 열기 전에 ReplacementError
    """
    for label, path in (("video", video), ("svm", svm), ("predictor", predictor)):
        if not os.path.isfile(path):
            raise FileNotFoundError("%s 파일을 찾을 수 없습니다: %s" % (label, path))
    original = read_original_source() if source is None else source
    tick = bool(progress_every or max_frames)
    replacements = build_replacements(video, predictor, svm, rotate, tick)
    check_counts(original, replacements)
    modified = apply_replacements(original, replacements)
    changed = changed_line_numbers(original, modified)
    sink = MemoryBlinkSink(on_row=on_row)
    started = time.time()
    try:
        frames = run_source(modified, sink, quiet, progress_every, total_frames, max_frames)
    finally:
        close_gui_windows()
    return ExtractionResult(sink.rows, sink.raw, frames, time.time() - started, changed)
