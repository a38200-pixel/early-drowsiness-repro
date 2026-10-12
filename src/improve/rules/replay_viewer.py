# replay_viewer.py
# 역할: 녹화 영상에 규칙 알람 화면과 HM-LSTM 점수 게이지를 겹쳐 영상 fps로 재생한다(결정 I-W-33 대안안: 규칙과 HM-LSTM을 선계산하고 검출 없이 재생).
#       영상은 디코딩해 보여 주기만 한다(얼굴 검출, 점수 계산 없음). 아무것도 저장하지 않는다.
# 입력: --video(영상), --rules-json(replay_precompute.py 출력: 프레임별 EAR 와 눈 윤곽), --timeline-json(timeline.py score 출력의 시간선),
#       --rotate(0, 90, 180, 270, auto; 기본 auto = 선계산 문서에 기록된 적용 각도), --thresholds(쉼표 목록, 기본 0.15).
# 규칙: 시간은 프레임 번호 / fps 이다. 선계산한 EAR 열을 그 시간으로 LiveRules 에 프레임 순서대로 넣어 알람 상태를 다시 계산한다(결정적).
#       화면을 건너뛰어도 건너뛴 프레임의 EAR 는 모두 넣는다. 규칙 화면은 live_demo.draw_ui(clean=True)를 그대로 쓴다(draw_ui 는 수정하지 않았다).
# 게이지: 그 화면 아래에 띠를 이어 붙인다. 문구는 gauge.format_gauge_text(display_state(...))의 출력을 그대로 그린다(결정 I-33 확정 문구).
#       눈금선 위치만 gauge.BOUNDARIES 로 흐리게 긋고 구간 이름은 쓰지 않는다(I-W-35 미결). 알람과 구분하려고 차가운 색을 쓰고 깜빡임, 테두리, 소리와 연결하지 않는다.
#       현재 프레임 번호 이하에서 확정된 마지막 창의 점수를 보여 준다(행 확정은 블링크 끝보다 8에서 15프레임 늦다, I-W-28 미검증).
#       한글은 cv2.putText 로 그릴 수 없어 Pillow 와 Windows 한글 글꼴(FONT_NAMES)로 그린다. 글꼴을 찾지 못하면 시작할 때 오류로 끝낸다.
# 화면: 위에서 아래로 규칙 화면(임계값 줄, 얼굴 상태 띠, 영상 칸), 시간 줄(높이 24), 게이지 띠(높이 96). 영상 칸은 폭 640 x 높이 --video-height(기본 480)로 고정이며
#       프레임은 종횡비를 지켜 칸 중앙에 맞추고 남는 곳은 검정이다(세로 영상은 좌우 검정). 기본 전체 높이 110 + 28 + 480 + 24 + 96 = 738.
#       시간 줄: 왼쪽 "mm:ss.s / mm:ss.s"(현재 / 전체, 시간 = 프레임 번호 / fps), 오른쪽 "frame N / M"(N 은 0부터 센 현재 프레임 번호, M 은 마지막 프레임 번호),
#       아래 진행 막대, 일시정지 중에는 "일시정지" 표시.
# 재생: 영상 fps 에 맞춘 시계 기준으로 진행하고, 느리면 화면 프레임을 건너뛴다. 키: q 종료, 스페이스 일시정지와 계속, r 처음부터.
# 사용: PYTHONPATH=src python -m improve.rules.replay_viewer --video <VIDEO> --rules-json <TIMELINE_DIR>/rules.json --timeline-json <TIMELINE_DIR>/timeline.json [--rotate auto]
"""규칙 알람과 점수 게이지 재생 뷰어(Python 3.6 호환)."""
import argparse
import collections
import os
import sys
import time

from improve.hmlstm import gauge as GAUGE
from improve.hmlstm import timeline as TL

from . import live_demo as LD
from . import replay_precompute as RP
from . import video_rotation

DISPLAY_WIDTH = 640                   # 영상 칸의 폭(화면 폭)
DEFAULT_VIDEO_HEIGHT = 480            # 영상 칸의 높이 기본값(--video-height). 영상은 종횡비를 지켜 이 칸 안에 맞추고 나머지는 검정
MIN_VIDEO_HEIGHT = 120
MAX_VIDEO_HEIGHT = 1400
TIME_H = 24                           # 영상 칸과 게이지 띠 사이의 시간 줄 높이
GAUGE_H = 96
WINDOW_NAME = "replay_viewer"
FONT_NAMES = ("malgun.ttf", "malgunbd.ttf", "gulim.ttc")
ROTATE_CHOICES = RP.ROTATE_CHOICES
TIME_BG = (64, 42, 24)
PAUSED_TEXT = "일시정지"
GAUGE_BG = (80, 52, 30)              # BGR. 알람 줄의 회색, 빨강, 주황과 구분되는 차가운 색(어두운 파랑)
GAUGE_BAR = (120, 92, 64)
GAUGE_TICK = (170, 140, 110)
GAUGE_RANGE = (235, 200, 120)
GAUGE_MEAN = (255, 255, 210)
TEXT_MAIN = (255, 255, 255)
TEXT_DIM = (215, 195, 175)
KEY_QUIT = ord("q")
KEY_SPACE = 32
KEY_RESTART = (ord("r"), ord("R"))
MAX_WAIT_MS = 30

EngineState = collections.namedtuple("EngineState", ["index", "time_s", "ear", "ear_min"])
Step = collections.namedtuple("Step", ["kind", "index", "frame", "state", "skipped"])


class ViewerError(ValueError):
    """입력 인자나 환경의 오류(메시지는 한 줄로 사용자에게 보인다)."""


# ----- 규칙 재생 -----
class ReplayEngine(object):
    """선계산한 프레임별 EAR 열을 프레임 순서대로 LiveRules 에 넣는다. cv2 를 쓰지 않는다."""

    def __init__(self, rules_document, thresholds=LD.THRESHOLDS):
        self.fps = float(rules_document["fps"])
        self.frames = rules_document["frames"]
        self.thresholds = tuple(thresholds)
        self.reset()

    def reset(self):
        self.rules = LD.LiveRules(self.thresholds)
        self.latch = LD.DisplayLatch()
        self.recent = LD.RecentMinimum()
        self.cursor = 0                       # 다음에 넣을 프레임 번호
        self.last = None

    def advance_to(self, index):
        """프레임 cursor 부터 index 까지의 EAR 를 모두 넣고 index 프레임의 표시 값을 돌려준다. 이미 지난 프레임으로는 돌아갈 수 없다."""
        if index < self.cursor - 1 or index >= len(self.frames):
            raise ValueError("프레임 번호가 범위를 벗어났습니다: %r (다음 %d, 전체 %d)" % (index, self.cursor, len(self.frames)))
        if index == self.cursor - 1 and self.last is not None:
            return self.last
        state = None
        for position in range(self.cursor, index + 1):
            timestamp = position / self.fps
            frame = self.frames[position]
            if frame["face"]:
                self.recent.add(timestamp, frame["ear"])
                new_alarms = self.rules.update(timestamp, frame["ear"], "detected")
                ear_value = frame["ear"]
                ear_min = self.recent.minimum(timestamp)
            else:
                new_alarms = self.rules.update(timestamp, None, "none")
                ear_value = None
                ear_min = None
            for threshold, _alarm in new_alarms:
                self.latch.trigger(threshold, timestamp)
            state = EngineState(position, timestamp, ear_value, ear_min)
        self.cursor = index + 1
        self.last = state
        return state


# ----- 재생 시계와 재생기 -----
class PlaybackClock(object):
    """영상 시간(초)을 시계 시각에서 구한다. 일시정지 동안은 멈춘다."""

    def __init__(self, fps):
        self.fps = float(fps)
        self.start(0.0)

    def start(self, now):
        self._elapsed = 0.0
        self._running_since = now

    @property
    def paused(self):
        return self._running_since is None

    def elapsed(self, now):
        return self._elapsed + (0.0 if self._running_since is None else now - self._running_since)

    def toggle_pause(self, now):
        if self._running_since is None:
            self._running_since = now
        else:
            self._elapsed += now - self._running_since
            self._running_since = None

    def frame_index(self, now):
        return int(self.elapsed(now) * self.fps)

    def seconds_until(self, index, now):
        return index / self.fps - self.elapsed(now)


class Player(object):
    """프레임 소스, 규칙 재생, 시계를 묶는다. 화면을 그리지 않으며 소스는 read, grab, restart 를 가진 객체이면 된다."""

    def __init__(self, source, engine, clock, frame_count):
        self.source = source
        self.engine = engine
        self.clock = clock
        self.frame_count = frame_count
        self._reset_position()

    def _reset_position(self):
        self.decoded = 0                      # 소스가 다음에 읽을 프레임 번호
        self.shown = -1
        self.ended = False

    def start(self, now):
        self.clock.start(now)

    def step(self, now):
        """now 시각에 보여 줄 프레임이 새로 있으면 Step("frame", ...) 을, 없으면 Step("idle", ...) 을, 영상이 끝나면 Step("end", ...) 를 돌려준다."""
        idle = Step("idle", self.shown, None, None, 0)
        if self.clock.paused or self.ended:
            return idle
        target = min(self.clock.frame_index(now), self.frame_count - 1)
        if target <= self.shown:
            return idle
        skipped = target - self.shown - 1
        while self.decoded < target:
            if not self.source.grab():
                self.ended = True
                return Step("end", self.shown, None, None, 0)
            self.decoded += 1
        grabbed, frame = self.source.read()
        self.decoded += 1
        if not grabbed:
            self.ended = True
            return Step("end", self.shown, None, None, 0)
        self.shown = target
        return Step("frame", target, frame, self.engine.advance_to(target), skipped)

    def handle_key(self, key, now):
        """키를 처리하고 동작 이름("quit", "pause", "resume", "restart")을 돌려준다. 해당 없으면 None."""
        if key == KEY_QUIT:
            return "quit"
        if key == KEY_SPACE:
            self.clock.toggle_pause(now)
            return "pause" if self.clock.paused else "resume"
        if key in KEY_RESTART:
            self.source.restart()
            self.engine.reset()
            self._reset_position()
            self.clock.start(now)
            return "restart"
        return None

    def wait_ms(self, now):
        """다음 키 확인까지 기다릴 시간(ms). 멈춰 있거나 끝났으면 최대값, 아니면 다음 프레임까지(1에서 최대값)."""
        if self.clock.paused or self.ended:
            return MAX_WAIT_MS
        remaining = self.clock.seconds_until(self.shown + 1, now)
        return max(1, min(MAX_WAIT_MS, int(remaining * 1000)))


# ----- 게이지 띠 -----
def gauge_state(timeline_document, index):
    """프레임 index 에서의 게이지 상태. 대상 영상 구간의 index 이하에서 확정된 마지막 창을 쓴다."""
    return GAUGE.display_state(timeline_document["target"], index)


def draw_gauge_strip(cv2, np, width, state, painter):
    """게이지 띠 한 장(높이 GAUGE_H)을 그려 돌려준다. 문구는 gauge.format_gauge_text 의 출력 그대로이다.

    painter(strip, items): items 는 (문구, (x, y), 글자 크기, BGR 색)의 목록이며 한글을 그리는 함수이다(시험에서는 기록만 하는 가짜를 쓴다).
    """
    strip = np.zeros((GAUGE_H, width, 3), dtype=np.uint8)
    strip[:] = GAUGE_BG
    x0, x1 = 10, width - 10
    y0, y1 = GAUGE_H - 22, GAUGE_H - 8
    span = float(GAUGE.SCALE_MAX - GAUGE.SCALE_MIN)

    def position(value):
        clamped = min(max(value, GAUGE.SCALE_MIN), GAUGE.SCALE_MAX)
        return x0 + int(round((clamped - GAUGE.SCALE_MIN) / span * (x1 - x0)))

    cv2.rectangle(strip, (x0, y0), (x1, y1), GAUGE_BAR, -1)
    for boundary in GAUGE.BOUNDARIES:
        cv2.line(strip, (position(boundary), y0), (position(boundary), y1), GAUGE_TICK, 1)
    if state["stage"] == GAUGE.STAGE_VALUE:
        cv2.rectangle(strip, (position(state["min"]), y0 + 4), (position(state["max"]), y1 - 4), GAUGE_RANGE, -1)
        cv2.line(strip, (position(state["mean"]), y0 - 2), (position(state["mean"]), y1 + 2), GAUGE_MEAN, 3)
    elif state["stage"] == GAUGE.STAGE_COLLECTING:
        filled = x0 + int((x1 - x0) * min(state["blinks"], state["needed"]) / float(state["needed"]))
        if filled > x0:
            cv2.rectangle(strip, (x0, y0), (filled, y1), GAUGE_RANGE, -1)
    lines = GAUGE.format_gauge_text(state).split("\n")
    items = [(lines[0], (10, 3), 15, TEXT_DIM)]
    if len(lines) > 1:
        items.append((lines[1], (10, 24), 24, TEXT_MAIN))
    if len(lines) > 2:
        items.append((lines[2], (10, 56), 15, TEXT_DIM))
    painter(strip, items)
    return strip


class GaugeStrip(object):
    """상태가 같으면 같은 띠를 다시 쓴다(한글 그리기 비용을 줄이는 캐시)."""

    def __init__(self, cv2, np, width, painter):
        self.cv2, self.np, self.width, self.painter = cv2, np, width, painter
        self._key = None
        self._strip = None

    def render(self, state):
        key = (GAUGE.format_gauge_text(state), state["stage"], state["mean"], state["min"], state["max"], state["blinks"])
        if key != self._key:
            self._strip = draw_gauge_strip(self.cv2, self.np, self.width, state, self.painter)
            self._key = key
        return self._strip


def load_font_getter(font_names=FONT_NAMES):
    """크기별 글꼴을 돌려주는 함수를 만든다. 한글 글꼴을 하나도 열지 못하면 ViewerError."""
    from PIL import ImageFont

    cache = {}
    chosen = None
    for name in font_names:
        try:
            ImageFont.truetype(name, 16)
            chosen = name
            break
        except (OSError, IOError):
            continue
    if chosen is None:
        raise ViewerError("게이지 문구를 그릴 한글 글꼴을 찾지 못했습니다(%s)" % ", ".join(font_names))

    def get(size):
        if size not in cache:
            cache[size] = ImageFont.truetype(chosen, size)
        return cache[size]

    return get


def make_painter(cv2, np):
    """Pillow 로 한글을 그리는 painter. 띠 한 장을 한 번 변환해 모든 문구를 그린다."""
    from PIL import Image, ImageDraw

    get_font = load_font_getter()

    def painter(strip, items):
        image = Image.fromarray(cv2.cvtColor(strip, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(image)
        for text, xy, size, color in items:
            draw.text(xy, text, font=get_font(size), fill=(color[2], color[1], color[0]))
        strip[:] = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)

    return painter


# ----- 영상 칸 맞춤 -----
def fit_geometry(source_width, source_height, cell_width, cell_height):
    """원본 크기를 종횡비를 지켜 칸 안에 맞춘다. 반환: (새 너비, 새 높이, 왼쪽 이동량, 위쪽 이동량). 칸보다 커지지 않고 가운데에 놓인다."""
    scale = min(cell_width / float(source_width), cell_height / float(source_height))
    new_width = max(1, min(cell_width, int(round(source_width * scale))))
    new_height = max(1, min(cell_height, int(round(source_height * scale))))
    return new_width, new_height, (cell_width - new_width) // 2, (cell_height - new_height) // 2


def letterbox(cv2, np, frame, cell_width, cell_height):
    """프레임을 칸(cell_width x cell_height)에 맞춰 검정 바탕 가운데에 놓는다. 반환: (칸 영상, (가로 배율, 세로 배율), (왼쪽 이동량, 위쪽 이동량))."""
    height, width = frame.shape[:2]
    new_width, new_height, offset_x, offset_y = fit_geometry(width, height, cell_width, cell_height)
    cell = np.zeros((cell_height, cell_width, 3), dtype=np.uint8)
    fitted = frame if (new_width, new_height) == (width, height) else cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
    cell[offset_y:offset_y + new_height, offset_x:offset_x + new_width] = fitted
    return cell, (new_width / float(width), new_height / float(height)), (offset_x, offset_y)


def transform_polygon(np, points, scales, offsets):
    """원본 크기 기준 눈 윤곽 좌표를 칸 좌표로 바꾼다(같은 배율과 이동량)."""
    return (np.asarray(points, dtype=np.float64) * np.array(scales, dtype=np.float64) + np.array(offsets, dtype=np.float64)).astype(np.int32)


# ----- 시간 줄 -----
def format_time(seconds):
    """초를 "mm:ss.s"로 쓴다(소수 첫째 자리에서 반올림, 올림이 생기면 분으로 넘어간다: 59.95초 -> 01:00.0). 분은 두 자리 이상이다."""
    tenths = int(max(seconds, 0.0) * 10 + 0.5)
    return "%02d:%04.1f" % (tenths // 600, (tenths % 600) / 10.0)


def progress_fraction(index, frame_count):
    """진행 막대의 채움 비율: 첫 프레임 0, 마지막 프레임 1. 프레임이 하나뿐이면 1."""
    if frame_count <= 1:
        return 1.0
    return min(max(index / float(frame_count - 1), 0.0), 1.0)


def time_texts(index, frame_count, fps):
    """시간 줄의 왼쪽 문구와 오른쪽 문구. 시간은 프레임 번호 / fps 이고 N 은 0부터 센 프레임 번호, M 은 마지막 프레임 번호이다."""
    left = "%s / %s" % (format_time(index / float(fps)), format_time(frame_count / float(fps)))
    return left, "frame %d / %d" % (index, frame_count - 1)


def draw_time_strip(cv2, np, width, index, frame_count, fps, paused, painter):
    """시간 줄 한 장(높이 TIME_H)을 그린다. painter 는 draw_gauge_strip 과 같은 모양의 한글 그리기 함수이다."""
    strip = np.zeros((TIME_H, width, 3), dtype=np.uint8)
    strip[:] = TIME_BG
    bar_top = TIME_H - 4
    cv2.rectangle(strip, (0, bar_top), (width - 1, TIME_H - 1), GAUGE_BAR, -1)
    filled = int(round((width - 1) * progress_fraction(index, frame_count)))
    if filled > 0:
        cv2.rectangle(strip, (0, bar_top), (filled, TIME_H - 1), GAUGE_RANGE, -1)
    left, right = time_texts(index, frame_count, fps)
    items = [(left, (8, 2), 14, TEXT_MAIN), (right, (max(8, width - 10 - int(len(right) * 7.8)), 2), 14, TEXT_DIM)]
    if paused:
        items.append((PAUSED_TEXT, (width // 2 - 30, 1), 15, GAUGE_MEAN))
    painter(strip, items)
    return strip


class TimeStrip(object):
    """상태가 같으면 같은 시간 줄을 다시 쓴다."""

    def __init__(self, cv2, np, width, painter):
        self.cv2, self.np, self.width, self.painter = cv2, np, width, painter
        self._key = None
        self._strip = None

    def render(self, index, frame_count, fps, paused):
        key = (time_texts(index, frame_count, fps), bool(paused), int(round((self.width - 1) * progress_fraction(index, frame_count))))
        if key != self._key:
            self._strip = draw_time_strip(self.cv2, self.np, self.width, index, frame_count, fps, paused, self.painter)
            self._key = key
        return self._strip


# ----- 화면 만들기 -----
def compose(cv2, np, frame, index, engine, state, strip, video_height=DEFAULT_VIDEO_HEIGHT, time_row=None):
    """회전을 마친 프레임으로 한 장의 화면을 만든다: draw_ui(clean=True) 화면(영상 칸 DISPLAY_WIDTH x video_height) 아래에 시간 줄과 게이지 띠를 이어 붙인다."""
    cell, scales, offsets = letterbox(cv2, np, frame, DISPLAY_WIDTH, video_height)
    polygons = []
    record = engine.frames[index]
    if record["face"]:
        for key in ("left", "right"):
            polygons.append(transform_polygon(np, record[key], scales, offsets))
    top = LD.draw_ui(cv2, np, cell, polygons, engine.rules, engine.latch, state.ear, state.ear_min, "", state.time_s, None, None, True)
    if time_row is None:
        time_row = np.zeros((TIME_H, DISPLAY_WIDTH, 3), dtype=np.uint8)
    return np.vstack([top, time_row, strip])


def resolve_angle(rotate, rules_document):
    """--rotate 값에서 적용할 각도를 정한다. auto 는 선계산 문서에 기록된 적용 각도이다."""
    if rotate == "auto":
        return rules_document["rotation"]["applied_angle"]
    return int(rotate)


def check_alignment(rules_document, timeline_document):
    """선계산 문서와 시간선 문서가 같은 영상에서 나왔는지 겉으로 확인한다. 어긋난 점의 설명 목록(비어 있으면 이상 없음)."""
    target = timeline_document["target"]
    notes = []
    if abs(float(rules_document["fps"]) - float(target["fps"])) > 1e-3:
        notes.append("fps 가 다릅니다: 규칙 문서 %.3f, 시간선 %.3f" % (rules_document["fps"], target["fps"]))
    if rules_document["frames_read"] != target["frames_read"]:
        notes.append("읽은 프레임 수가 다릅니다: 규칙 문서 %d, 시간선 %d" % (rules_document["frames_read"], target["frames_read"]))
    if os.path.splitext(rules_document["video_name"])[0] != os.path.splitext(target["video_name"])[0]:
        notes.append("영상 이름이 다릅니다: 규칙 문서 %s, 시간선 %s" % (rules_document["video_name"], target["video_name"]))
    return notes


class CaptureSource(object):
    """cv2.VideoCapture 를 Player 가 쓰는 모양(read, grab, restart)으로 감싼다."""

    def __init__(self, cv2, capture):
        self.cv2, self.capture = cv2, capture

    def read(self):
        return self.capture.read()

    def grab(self):
        return self.capture.grab()

    def restart(self):
        self.capture.set(self.cv2.CAP_PROP_POS_FRAMES, 0)


def load_rules_document(path):
    import json

    with open(path, "r", encoding="utf-8") as handle:
        try:
            document = json.load(handle)
        except ValueError as error:
            raise ViewerError("규칙 JSON 을 읽을 수 없습니다: %s" % error)
    try:
        return RP.validate_document(document)
    except RP.PrecomputeError as error:
        raise ViewerError("규칙 JSON 형식 오류: %s" % error)


def run(video, rules_path, timeline_path, rotate, thresholds, video_height=DEFAULT_VIDEO_HEIGHT, clock=time.perf_counter):
    import cv2
    import numpy as np

    if not os.path.isfile(video):
        raise ViewerError("영상 파일을 찾을 수 없습니다")
    rules_document = load_rules_document(rules_path)
    try:
        timeline_document = TL.read_timeline_document(timeline_path)
    except (TL.TimelineError, IOError, OSError) as error:
        raise ViewerError("시간선 JSON 을 읽을 수 없습니다: %s" % error)
    for note in check_alignment(rules_document, timeline_document):
        print("주의: %s" % note)
    angle = resolve_angle(rotate, rules_document)
    if angle != rules_document["rotation"]["applied_angle"]:
        print("주의: 요청 각도 %d 가 선계산에 쓴 각도 %d 와 다릅니다. 눈 윤곽이 어긋날 수 있습니다." % (angle, rules_document["rotation"]["applied_angle"]))
    painter = make_painter(cv2, np)
    capture = cv2.VideoCapture(video)
    if not capture.isOpened():
        raise ViewerError("영상을 열 수 없습니다")
    frame_count = len(rules_document["frames"])
    video_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if video_frames and video_frames != frame_count:
        print("주의: 영상의 프레임 수(%d)가 규칙 문서(%d)와 다릅니다." % (video_frames, frame_count))
    engine = ReplayEngine(rules_document, thresholds)
    player = Player(CaptureSource(cv2, capture), engine, PlaybackClock(rules_document["fps"]), frame_count)
    gauge = GaugeStrip(cv2, np, DISPLAY_WIDTH, painter)
    time_line = TimeStrip(cv2, np, DISPLAY_WIDTH, painter)
    fps = float(rules_document["fps"])
    last_view = [None]                    # 마지막으로 보인 (회전한 프레임, 번호, 상태). 일시정지 표시를 바꿀 때 다시 그리는 데 쓴다

    def show(view):
        frame_view, index, state = view
        cv2.imshow(WINDOW_NAME, compose(cv2, np, frame_view, index, engine, state, gauge.render(gauge_state(timeline_document, index)), video_height,
                                        time_line.render(index, frame_count, fps, player.clock.paused)))

    expected_size = tuple(rules_document["frame_size"])
    size_checked = False
    player.start(clock())
    try:
        while True:
            now = clock()
            step = player.step(now)
            if step.kind == "frame":
                frame = video_rotation.apply_rotation(step.frame, angle)
                if not size_checked:
                    size_checked = True
                    if (frame.shape[1], frame.shape[0]) != expected_size:
                        print("주의: 회전한 프레임 크기 %dx%d 가 규칙 문서의 %dx%d 와 다릅니다." % (frame.shape[1], frame.shape[0], expected_size[0], expected_size[1]))
                last_view[0] = (frame, step.index, step.state)
                show(last_view[0])
            key = cv2.waitKey(player.wait_ms(clock())) & 0xFF
            action = player.handle_key(key, clock())
            if action == "quit":
                break
            if action in ("pause", "resume") and last_view[0] is not None:
                show(last_view[0])
            elif action == "restart":
                last_view[0] = None
    finally:
        capture.release()
        cv2.destroyAllWindows()
    return 0


def video_height_value(text):
    """--video-height 값 검사: MIN_VIDEO_HEIGHT 이상 MAX_VIDEO_HEIGHT 이하의 정수."""
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError("정수여야 합니다: %r" % text)
    if not MIN_VIDEO_HEIGHT <= value <= MAX_VIDEO_HEIGHT:
        raise argparse.ArgumentTypeError("%d 이상 %d 이하여야 합니다: %d" % (MIN_VIDEO_HEIGHT, MAX_VIDEO_HEIGHT, value))
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="녹화 영상에 규칙 알람과 HM-LSTM 점수 게이지를 겹쳐 재생한다(참고용 점수, 아무것도 저장하지 않음). 키: q 종료, 스페이스 일시정지, r 처음부터.")
    parser.add_argument("--video", required=True, help="영상 파일")
    parser.add_argument("--rules-json", dest="rules_json", required=True, help="replay_precompute.py 가 만든 규칙용 JSON")
    parser.add_argument("--timeline-json", dest="timeline_json", required=True, help="timeline.py score 가 만든 시간선 JSON")
    parser.add_argument("--rotate", choices=list(ROTATE_CHOICES), default="auto", help="회전 각도(기본 auto = 규칙용 JSON 에 기록된 적용 각도)")
    parser.add_argument("--video-height", dest="video_height", type=video_height_value, default=DEFAULT_VIDEO_HEIGHT,
                        help="영상 칸의 높이(폭은 %d 고정, 기본 %d). 영상은 종횡비를 지켜 칸 안에 맞추고 남는 곳은 검정" % (DISPLAY_WIDTH, DEFAULT_VIDEO_HEIGHT))
    parser.add_argument("--thresholds", default=",".join(str(value) for value in LD.THRESHOLDS),
                        help="닫힘 임계값 목록(쉼표로 구분한 1에서 %d개, 첫 값이 주 임계값, 기본 %s)" % (LD.MAX_THRESHOLDS, ",".join(str(v) for v in LD.THRESHOLDS)))
    return parser.parse_args(argv)


def main(argv=None):
    arguments = parse_args(argv)
    try:
        thresholds = LD.parse_thresholds(arguments.thresholds)
        return run(arguments.video, arguments.rules_json, arguments.timeline_json, arguments.rotate, thresholds, arguments.video_height)
    except (ViewerError, ValueError) as error:
        print("replay_viewer: %s" % error)
        return 2


if __name__ == "__main__":
    sys.exit(main())
