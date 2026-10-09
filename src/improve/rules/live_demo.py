# 역할: 웹캠 실시간 데모(C4 일부, 결정 I-20). 사용자가 눈으로 확인하는 도구이며 규칙의 평가, 측정, 검증 도구가 아니다.
#   알람 횟수와 시각은 육안 확인용 표시이고 어떤 결과의 근거로도 쓰지 않는다.
# 규칙: docs/improvement/RULE_SPEC.md v1, 13절, 결정 I-17을 그대로 쓴다. episode_machine.EpisodeMachine 을 임계값마다 하나씩
#   --thresholds 목록(1~4개, 기본 0.15 하나(결정 I-22), 다른 값은 --thresholds 0.15,0.21,0.25 처럼 줄 수 있다)의 모든 값에 대해 동시에 돌린다. 3.0초 알람, 열림 0.3초 해제,
#   얼굴 없음 1초 이상 "판정 불가"는 기존 규칙과 같다. 목록의 첫 값이 주 임계값(가장 큰 줄, 테두리 깜빡임, --beep 의 기본 소리 대상)이다.
#   이 파일의 표시와 소리 설정(DISPLAY_LATCH_S, 깜빡임, --beep)은 규칙 판정(알람 시각, 에피소드, 해제)과 섞이지 않는다.
# 시간: 프레임이 들어온 시각(time.monotonic)을 timestamp 로 쓴다. 프레임 번호를 쓰지 않는다.
# 입력: 웹캠. 얼굴 검출, 랜드마크, EAR 계산은 blink_video_dump.py 와 같은 방식이다(전처리: 너비 450 리사이즈, 회색, 감마 1.5,
#   dlib 정면 검출기 detector(gray, 0)의 첫 얼굴, 같은 predictor). 감마 함수는 blink_video_dump 에서 가져오고, EAR 은 ear_math 를 쓴다.
#   광학 흐름 추적(tracked)은 만들지 않았다: 기본 유효 출처가 detected 하나이므로(I-W-14 제안) 검출이 실패한 프레임은 source none 으로 넣는다.
# 화면: 위에서부터 임계값 목록(주 임계값이 가장 큰 줄) 각각 화면 폭 전체의 줄, 얼굴 없음/판정 불가 노란 띠, 영상(왼쪽 위에 EAR now: 현재
#   프레임의 EAR 평균(detected 일 때만), min 3s: 최근 3.0초의 EAR 최솟값, 얼굴이 없으면 "-"; 둘 다 표시 전용이며 최근 3초분만 메모리에 둔다),
#   키 안내 한 줄. 각 줄에는 임계값, 닫힘 진행 막대(0~3.0초), 알람 횟수(큰 숫자), "last alarm: N.N s ago"(없으면 "-")가 있다.
#   알람이 난 줄은 빨강/주황이 0.25초 간격으로 번갈아 깜빡이며 R 키를 누를 때까지 유지된다(주 임계값은 화면 테두리도 같이).
#   알람 횟수는 R 키로 지워지지 않는다. 글자는 영문이다(화면 글꼴이 한글을 그리지 못한다). 영상 오른쪽 위에 primary: <주 임계값>, 오른쪽 아래에 sound: on/off/error 를 표시한다.
# 표시 유지: R 키를 누를 때까지 유지가 기본이다. DISPLAY_LATCH_S 는 "R 없이 자동으로 꺼지는 시간(초)"이며 기본값 None(자동 꺼짐 없음)이다.
#   이 값과 깜빡임 주기는 표시 전용이며 규칙 상수와 이름과 코드가 분리되어 있다. 규칙 판정에는 쓰지 않는다.
# 소리: 기본 꺼짐. --beep 을 주면 Windows 에서 알람 순간에 경보음(2음 사이렌, 880/1320 Hz 사각파를 0.25초씩 번갈아 총 3.0초)을 낸다.
#   소리는 0.25초 조각 둘(880 Hz 조각, 1320 Hz 조각)을 표준 라이브러리 struct 로 메모리에서 조립해 캐시하고, 알람마다 데몬 스레드가
#   조각을 번갈아 winsound.PlaySound(조각, SND_MEMORY | SND_NODEFAULT)로 동기 재생한다(SND_MEMORY 는 SND_ASYNC 와 함께 쓸 수 없고,
#   SND_ASYNC, SND_LOOP, 정지용 플래그는 쓰지 않는다). 메인 영상 루프를 막지 않는다. 기본은 주 임계값만, --beep 과 함께 --beep-all 을
#   주면 세 임계값 모두(--beep 없이 --beep-all 만 주면 소리 없음). 같은 갱신의 알람은 스레드 하나만 시작한다.
#   정지: R 키(표시 끄기), m 키(음소거로 전환), q 종료, 새 알람이 정지 신호(세대 번호)를 올리면 재생 스레드가 조각 사이에 확인해
#   현재 조각이 끝난 뒤(최대 0.25초) 멈춘다. 새 알람은 이전 재생을 정지시키고 새 재생을 시작한다.
#   재생 중 예외가 나면 그 스레드를 끝내고 프로세스당 한 번만 콘솔에 "sound error: <예외 이름>: <메시지>" 한 줄을 출력하며(파일에 쓰지 않음),
#   화면 표시와 규칙은 그대로이고 구석 표시는 sound: error 가 된다. Windows 가 아니면 소리 없이 동작한다.
# 키: q 종료, r 알람 표시 끄기(소리도 멈춤), m 음소거(소리도 멈춤). 웹캠이 열리지 않으면 이유를 한 줄 출력하고 끝낸다.
# 저장: 영상, 프레임, 이미지, CSV, 로그를 어떤 형태로도 만들지 않는다. 파일을 쓰지 않으며 모든 표시는 화면에만 그린다.
# 처리율 표시: 영상 왼쪽 아래 두 줄(최근 3초의 처리 fps와 최대 프레임 간격, 웹캠 드라이버가 알려 주는 해상도와 fps)은 사용자 육안 확인용이며 따로 검증하지 않았다. 시각 목록은 메모리에만 두고 표시 전용이다.
# 단계 시간 진단: 영상 왼쪽 아래 세 번째 줄(최근 3초의 capture.read() 대기 평균과 최댓값, 읽은 뒤 처리 시간 평균, detector 호출 시간 평균, 읽기 실패 횟수)도 사용자 육안 확인용이며 따로 검증하지 않았다. 메모리 목록뿐이고 표시 전용이다.
# 사용: (src 폴더에서 또는 PYTHONPATH=src) python -m improve.rules.live_demo --predictor <PREDICTOR> [--camera 0] [--beep] [--beep-all]
import argparse
import collections
import os
import struct
import sys
import threading
import time

from . import ear_math
from . import episode_machine as EM

THRESHOLDS = (0.15,)                      # 기본 임계값 0.15 하나(I-22, 사용자 결정). 다른 값은 --thresholds 로 준다(예: 0.15,0.21,0.25).
MAIN_THRESHOLD = THRESHOLDS[0]            # 기본 목록의 첫 값(주 임계값). 실행 중에는 목록의 첫 값(LiveRules.primary)을 쓴다.
MAX_THRESHOLDS = 4
EAR_MIN_WINDOW_S = 3.0                    # "min 3s" 표시 창. 표시 전용이며 규칙 상수와 별개이고 규칙 판정에 쓰지 않는다.
VALID_SOURCES = ("detected",)             # 기본 유효 출처(I-W-14 제안, 미확정)
DISPLAY_LATCH_S = None                    # R 없이 자동으로 꺼지는 시간(초), None 이면 자동 꺼짐 없음. 표시 전용이며 규칙 상수와 별개이고 규칙 판정에 쓰지 않는다.
BLINK_PERIOD_S = 0.25                     # 알람 줄의 빨강/주황 깜빡임 간격(표시 전용)
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
PREPROCESS_WIDTH = 450                    # blink_video_dump.py 와 같은 전처리 너비
PREPROCESS_GAMMA = 1.5                    # blink_video_dump.py 와 같은 감마

SIREN_SAMPLE_RATE = 44100                 # 경보음(표시와 규칙과 무관한 소리 설정)
SIREN_SECONDS = 3.0
SIREN_FREQS_HZ = (880, 1320)              # 0.25초 조각 둘을 번갈아 낸다
SIREN_SEGMENT_S = 0.25
SIREN_AMPLITUDE = 0.95                    # 최대치(32767)의 약 95%: 큰 음량 쪽으로, 클리핑은 피한다
SIREN_PIECE_COUNT = int(round(SIREN_SECONDS / SIREN_SEGMENT_S))     # 조각 12개 = 3.0초

ROW_MAIN_H = 110
ROW_OTHER_H = 62
STRIP_H = 28
HINT_H = 30
HINT_TEXT = "q quit | r reset alarm | m mute"
COLOR_GRAY = (90, 90, 90)
COLOR_GRAY_DARK = (65, 65, 65)
COLOR_DARK = (45, 45, 45)
COLOR_RED = (0, 0, 210)
COLOR_ORANGE = (0, 140, 255)
COLOR_YELLOW = (0, 215, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_BAR = (0, 165, 255)


class LiveRules(object):
    """임계값별 EpisodeMachine 묶음. 프레임 단위로 update 를 부르고 화면에 필요한 값을 읽는다. cv2 를 쓰지 않는다."""

    def __init__(self, thresholds=THRESHOLDS, valid_sources=VALID_SOURCES):
        self.thresholds = tuple(thresholds)
        if not self.thresholds:
            raise ValueError("임계값 목록이 비어 있습니다.")
        self.primary = self.thresholds[0]         # 목록의 첫 값이 주 임계값이다(가장 큰 줄, 테두리 깜빡임, 기본 소리 대상)
        self.machines = collections.OrderedDict(
            (threshold, EM.EpisodeMachine(threshold=threshold, valid_sources=valid_sources)) for threshold in self.thresholds)
        self._seen = dict((threshold, 0) for threshold in self.thresholds)

    def update(self, timestamp_s, ear_mean, source):
        """샘플 하나를 모든 임계값의 상태기계에 넣는다.

        입력: timestamp_s - 프레임이 들어온 시각(초). ear_mean - 양쪽 눈 EAR 평균 또는 None. source - detected / tracked / none
        반환: 이번 샘플에서 새로 난 알람의 list of (임계값, Alarm)
        """
        new_alarms = []
        for threshold, machine in self.machines.items():
            machine.feed(timestamp_s, ear_mean, source)
            count = len(machine.alarms)
            if count > self._seen[threshold]:
                new_alarms.extend((threshold, machine.alarms[index]) for index in range(self._seen[threshold], count))
                self._seen[threshold] = count
        return new_alarms

    def alarm_count(self, threshold):
        return len(self.machines[threshold].alarms)

    def last_alarm_time(self, threshold):
        """가장 최근 알람의 시각(초, update 에 넣은 시각 기준). 알람이 없으면 None."""
        alarms = self.machines[threshold].alarms
        return alarms[-1].alarm_time_s if alarms else None

    def closed_length_s(self, threshold):
        """진행 중인 구간의 길이(초). 규칙이 알람 판정에 재는 끝(제안 2)을 그대로 쓴다. 구간이 없으면 0."""
        machine = self.machines[threshold]
        if machine.state != "closed" or machine.t_start is None:
            return 0.0
        end = machine.open_run_start if machine.open_run_start is not None else machine.last_closed_ts
        if end is None:
            return 0.0
        return max(0.0, end - machine.t_start)

    def bar_fraction(self, threshold):
        """닫힘 진행 막대의 채움 비율(0~1). 가득 찬 값은 규칙의 알람 지속 시간(3.0초)이다."""
        machine = self.machines[threshold]
        return min(1.0, self.closed_length_s(threshold) / machine.alarm_after_s)

    def face_status(self):
        """"ok", "missing"(얼굴 없음이 1초 미만), "unjudgeable"(1초 이상, 판정 불가). 첫 임계값의 상태기계를 본다(모두 같다)."""
        machine = next(iter(self.machines.values()))
        if machine.state == "unjudgeable":
            return "unjudgeable"
        if machine.missing_start is not None:
            return "missing"
        return "ok"


class DisplayLatch(object):
    """알람 표시를 유지한다. 표시 전용이며 규칙 판정에 쓰지 않는다.

    duration_s 가 None(기본, DISPLAY_LATCH_S)이면 reset 을 부를 때까지 유지되고, 숫자이면 그 초가 지나면 자동으로 꺼진다.
    reset 은 표시만 끈다(알람 횟수와 규칙 상태는 건드리지 않는다).
    """

    def __init__(self, duration_s=DISPLAY_LATCH_S):
        self.duration_s = duration_s
        self._started = {}

    def trigger(self, key, now):
        self._started[key] = now

    def active(self, key, now):
        started = self._started.get(key)
        if started is None:
            return False
        if self.duration_s is None:
            return True
        return (now - started) < self.duration_s

    def reset(self):
        self._started.clear()


def blink_is_red(now, period_s=BLINK_PERIOD_S):
    """알람 줄의 깜빡임 위상(표시 전용): period_s 마다 빨강과 주황이 번갈아 나온다. now 는 time.monotonic 값이다."""
    return int(now // period_s) % 2 == 0


def threshold_label(threshold):
    """화면에 쓰는 임계값 표기(소수 둘째 자리, 그것으로 구분되지 않으면 셋째 자리)."""
    return "%.2f" % threshold if round(threshold, 2) == threshold else "%.3f" % threshold


def parse_thresholds(text):
    """쉼표로 구분한 임계값 목록을 튜플로 바꾼다. 1~4개, 각 값은 0 초과 1 미만의 실수, 중복 없음. 오류는 한 줄짜리 ValueError 이다."""
    parts = [part.strip() for part in str(text).split(",")]
    if not str(text).strip() or any(part == "" for part in parts):
        raise ValueError("--thresholds 오류: 쉼표로 구분한 실수를 1~%d개 주세요(빈 값 없음)." % MAX_THRESHOLDS)
    values = []
    for part in parts:
        try:
            value = float(part)
        except ValueError:
            raise ValueError("--thresholds 오류: %r 은(는) 실수가 아닙니다." % part)
        if not (0.0 < value < 1.0):
            raise ValueError("--thresholds 오류: %r 은(는) 0 초과 1 미만이어야 합니다." % part)
        values.append(value)
    if len(values) > MAX_THRESHOLDS:
        raise ValueError("--thresholds 오류: 임계값은 최대 %d개입니다(받은 개수 %d)." % (MAX_THRESHOLDS, len(values)))
    if len(set(values)) != len(values):
        raise ValueError("--thresholds 오류: 같은 값이 중복되어 있습니다.")
    return tuple(values)


class RecentMinimum(object):
    """최근 window_s 초 동안의 값의 최솟값(표시 전용). 최근분만 메모리의 deque 에 두며 파일로 쓰지 않고 규칙 판정에 쓰지 않는다."""

    def __init__(self, window_s=EAR_MIN_WINDOW_S):
        self.window_s = window_s
        self._items = collections.deque()

    def _prune(self, now):
        while self._items and now - self._items[0][0] > self.window_s:
            self._items.popleft()

    def add(self, timestamp_s, value):
        self._items.append((timestamp_s, value))
        self._prune(timestamp_s)

    def minimum(self, now):
        """now 기준 최근 window_s 초(경계 포함) 안의 최솟값. 값이 없으면 None."""
        self._prune(now)
        return min(value for _timestamp, value in self._items) if self._items else None


class FrameRateMeter(object):
    """최근 window_s 초 안의 프레임 도착 시각으로 처리 fps와 최대 프레임 간격(ms)을 구한다(표시 전용). 시각 목록은 메모리의 deque 뿐이며 파일로 쓰지 않는다."""

    def __init__(self, window_s=EAR_MIN_WINDOW_S):
        self.window_s = window_s
        self._times = collections.deque()

    def add(self, t_s):
        """도착 시각을 넣는다. 직전 시각보다 작은 시각(거꾸로 온 표본)은 무시한다. window_s 보다 오래된 시각은 버린다."""
        if self._times and t_s < self._times[-1]:
            return
        self._times.append(t_s)
        while self._times and t_s - self._times[0] > self.window_s:
            self._times.popleft()

    def stats(self):
        """(fps 또는 None, 최대 간격 ms 또는 None). fps = (표본 수 - 1) / (마지막 시각 - 첫 시각), 표본 2개 미만 또는 시간 차 0이면 None."""
        if len(self._times) < 2:
            return None, None
        times = list(self._times)
        gaps = [later - earlier for earlier, later in zip(times, times[1:])]
        span = times[-1] - times[0]
        fps = (len(times) - 1) / span if span > 0 else None
        return fps, max(gaps) * 1000.0


def format_rate_lines(fps, max_gap_ms, window_s, cam_width, cam_height, reported_fps):
    """화면에 그릴 두 줄(영문)을 돌려준다. 값이 None 이면 "--", 보고 fps 가 None 이거나 0 이하이면 "n/a"."""
    fps_text = "--" if fps is None else "%.1f" % fps
    gap_text = "--" if max_gap_ms is None else "%d ms (%g s)" % (int(round(max_gap_ms)), window_s)
    reported = "n/a" if (reported_fps is None or reported_fps <= 0) else "%.1f" % reported_fps
    return ("fps %s  max gap %s" % (fps_text, gap_text), "cam %dx%d  reported fps %s" % (cam_width, cam_height, reported))


class StageTimeMeter(object):
    """최근 window_s 초의 단계 시간(ms)을 모은다(표시 전용). 읽기 대기, 읽은 뒤 처리, detector 호출 시간과 읽기 실패 횟수를 센다. 메모리 목록뿐이며 파일로 쓰지 않는다.

    add 의 t_s 는 표본 시각(초, perf_counter 기준)이며 없으면 지금 perf_counter 를 쓴다. 알람에 쓰는 시각(arrival)과 무관하다.
    """

    def __init__(self, window_s=EAR_MIN_WINDOW_S):
        self.window_s = window_s
        self._items = collections.deque()

    def add(self, read_ms, process_ms, detect_ms, read_failed, t_s=None):
        """표본 하나를 넣는다. 값이 None 이면 그 항목의 평균에서 빠진다(검출을 하지 않은 반복의 detect_ms 등)."""
        if t_s is None:
            t_s = time.perf_counter()
        self._items.append((t_s, read_ms, process_ms, detect_ms, bool(read_failed)))
        while self._items and t_s - self._items[0][0] > self.window_s:
            self._items.popleft()

    def stats(self):
        """(read 평균, process 평균, detect 평균, read 최댓값, 읽기 실패 횟수). 표본이 없으면 모두 None, 값이 없는 항목은 None."""
        if not self._items:
            return None, None, None, None, None

        def mean(index):
            values = [item[index] for item in self._items if item[index] is not None]
            return sum(values) / len(values) if values else None

        reads = [item[1] for item in self._items if item[1] is not None]
        return mean(1), mean(2), mean(3), (max(reads) if reads else None), sum(1 for item in self._items if item[4])


def format_stage_line(read_avg, process_avg, detect_avg, read_max, read_failures):
    """화면에 그릴 한 줄(영문). 값이 None 이면 "--"."""
    def ms(value):
        return "--" if value is None else "%d ms" % int(round(value))

    read_text = "--" if read_avg is None else "%s (max %s)" % (ms(read_avg), ms(read_max))
    bad_text = "--" if read_failures is None else "%d" % read_failures
    return "read %s  proc %s  det %s  bad %s" % (read_text, ms(process_avg), ms(detect_avg), bad_text)


_SIREN_CACHE = {}


def _wav_bytes(pcm, sample_rate=SIREN_SAMPLE_RATE):
    """16비트 모노 PCM 바이트에 WAV 헤더(44바이트)를 붙인다. struct 로 직접 조립하며 파일을 쓰지 않는다."""
    header = (b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVE" + b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, sample_rate,
              sample_rate * 2, 2, 16) + b"data" + struct.pack("<I", len(pcm)))
    return header + pcm


def _square_pcm(frequency_hz, sample_rate, seconds, amplitude):
    """사각파 16비트 PCM. 표본 번호의 정수 계산만 쓰므로 결과는 결정적이다."""
    count = int(round(sample_rate * seconds))
    peak = int(32767 * amplitude)
    half = sample_rate // 2
    samples = [peak if (n * frequency_hz) % sample_rate < half else -peak for n in range(count)]
    return struct.pack("<%dh" % count, *samples)


def build_siren_piece(frequency_hz, sample_rate=SIREN_SAMPLE_RATE, seconds=SIREN_SEGMENT_S, amplitude=SIREN_AMPLITUDE):
    """한 주파수의 사각파 조각(기본 0.25초)을 유효한 WAV 바이트(헤더 포함, 16비트 모노)로 만든다."""
    return _wav_bytes(_square_pcm(frequency_hz, sample_rate, seconds, amplitude), sample_rate)


def siren_pieces():
    """(880 Hz 조각, 1320 Hz 조각)의 WAV 바이트. 한 번 만들어 모듈 안에 캐시한다."""
    if "pieces" not in _SIREN_CACHE:
        _SIREN_CACHE["pieces"] = tuple(build_siren_piece(frequency) for frequency in SIREN_FREQS_HZ)
    return _SIREN_CACHE["pieces"]


def build_siren_wav(sample_rate=SIREN_SAMPLE_RATE, seconds=SIREN_SECONDS, freqs_hz=SIREN_FREQS_HZ,
                    segment_s=SIREN_SEGMENT_S, amplitude=SIREN_AMPLITUDE):
    """조각을 번갈아 이어 붙인 전체 사이렌 WAV 바이트(기본 조각 12개 = 3.0초). 재생에는 쓰지 않고 길이와 내용 확인에 쓴다."""
    pcms = [_square_pcm(frequency, sample_rate, segment_s, amplitude) for frequency in freqs_hz]
    pieces = int(round(seconds / segment_s))
    return _wav_bytes(b"".join(pcms[index % len(pcms)] for index in range(pieces)), sample_rate)


def siren_wav():
    """전체 사이렌 WAV 바이트(캐시)."""
    if "wav" not in _SIREN_CACHE:
        _SIREN_CACHE["wav"] = build_siren_wav()
    return _SIREN_CACHE["wav"]


def _winsound_module():
    """Windows 에서만 winsound 를 돌려준다(그 밖에는 None 이어서 소리 없이 동작한다)."""
    if not sys.platform.startswith("win"):
        return None
    try:
        import winsound
    except ImportError:
        return None
    return winsound


class SirenPlayer(object):
    """사이렌을 데몬 스레드에서 조각 단위로 동기 재생한다. 메인 영상 루프를 막지 않는다.

    start: 이전 재생에 정지 신호(세대 번호)를 올리고 새 스레드를 시작한다. 스레드는 880 Hz 조각과 1320 Hz 조각을 번갈아
    SIREN_PIECE_COUNT 번(3.0초) PlaySound(조각, SND_MEMORY | SND_NODEFAULT)로 재생하며, 조각 사이마다 세대 번호를 확인해
    바뀌었으면 멈춘다(현재 조각이 끝난 뒤, 최대 0.25초). SND_ASYNC, SND_LOOP, 정지용 플래그는 쓰지 않는다.
    stop: 정지 신호만 올린다. 재생 중 예외가 나면 그 스레드를 끝내고 프로세스당 한 번만 콘솔에 한 줄을 출력하며 error 에 기록한다.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._generation = 0
        self._thread = None
        self._error_reported = False
        self.error = None

    def start(self):
        with self._lock:
            self._generation += 1
            generation = self._generation
            previous = self._thread
            thread = threading.Thread(target=self._run, args=(generation, previous), name="siren", daemon=True)
            self._thread = thread
        thread.start()
        return thread

    def stop(self):
        with self._lock:
            self._generation += 1

    def join(self, timeout=None):
        thread = self._thread
        if thread is not None:
            thread.join(timeout)

    def _report(self, error):
        self.error = "%s: %s" % (type(error).__name__, error)
        if not self._error_reported:
            self._error_reported = True
            print("sound error: %s" % self.error)
            sys.stdout.flush()

    def _run(self, generation, previous):
        try:
            if previous is not None:
                previous.join(1.0)                          # 이전 재생의 현재 조각이 끝날 때까지 기다려 소리가 겹치지 않게 한다
            winsound = _winsound_module()
            if winsound is None:
                return
            pieces = siren_pieces()
            flags = winsound.SND_MEMORY | winsound.SND_NODEFAULT
            for index in range(SIREN_PIECE_COUNT):
                if self._generation != generation:
                    return
                winsound.PlaySound(pieces[index % len(pieces)], flags)
        except Exception as error:
            self._report(error)


class Beeper(object):
    """알람 순간의 경보음(2음 사이렌). 기본 꺼짐(enabled=False). 같은 update 에서 여러 알람이 나도 재생 스레드는 하나만 시작한다.

    새 알람이 나면 이전 재생에 정지 신호를 올리고 새로 시작한다. 재생 오류가 한 번이라도 났으면 label 이 sound: error 가 된다.
    backend 와 stopper 는 시험이나 다른 환경에서 재생과 정지 함수를 바꿔 끼우는 용도이다(없으면 SirenPlayer).
    """

    def __init__(self, enabled=False, beep_all=False, muted=False, backend=None, stopper=None, player=None, primary=None):
        self.enabled = enabled
        self.primary = primary if primary is not None else MAIN_THRESHOLD     # 기본 소리 대상(목록의 첫 값)
        self.beep_all = beep_all
        self.muted = muted
        self._backend = backend
        self._stopper = stopper
        self.player = player if player is not None else SirenPlayer()
        self.beeps = 0
        self.stops = 0

    @property
    def sound_on(self):
        return self.enabled and not self.muted

    def wants(self, threshold):
        return self.beep_all or threshold == self.primary

    def toggle_mute(self):
        """음소거를 전환한다. 음소거로 바뀌면 재생 중인 소리에 정지 신호를 올린다."""
        self.muted = not self.muted
        if self.muted:
            self.stop()
        return self.muted

    def stop(self):
        """재생 중인 소리에 정지 신호를 올린다(R 키, m 키, 종료). 소리를 쓰지 않는 설정이면 아무것도 하지 않는다. 실패는 무시한다."""
        if not self.enabled:
            return False
        self.stops += 1
        try:
            if self._stopper is not None:
                self._stopper()
            else:
                self.player.stop()
        except Exception:
            pass
        return True

    def label(self):
        if self.enabled and self.player.error is not None:
            return "sound: error"
        return "sound: on" if self.sound_on else "sound: off"

    def on_new_alarms(self, new_alarms):
        if not self.sound_on or not any(self.wants(threshold) for threshold, _alarm in new_alarms):
            return False
        self.beeps += 1
        try:
            if self._backend is not None:
                self._backend()
            else:
                self.player.start()                           # 새 스레드를 시작하고 바로 돌아온다
        except Exception:
            pass
        return True


def _text_size(cv2, text, scale, thickness):
    (width, height), _baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    return width, height


def _fit_scale(cv2, text, thickness, max_width, start, minimum=0.3):
    """글자가 max_width 안에 들어오도록 글자 크기를 줄여 돌려준다."""
    scale = start
    while scale > minimum and _text_size(cv2, text, scale, thickness)[0] > max_width:
        scale -= 0.05
    return max(scale, minimum)


def _put_centered(cv2, canvas, text, x0, x1, y_center, scale, thickness, color):
    width, height = _text_size(cv2, text, scale, thickness)
    cv2.putText(canvas, text, (x0 + ((x1 - x0) - width) // 2, y_center + height // 2), cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, thickness, cv2.LINE_AA)


def draw_ui(cv2, np, frame, eye_polygons, rules, latch, ear_value, ear_min, sound_label, now, rate_lines=None, stage_line=None):
    """화면 한 장을 그려 돌려준다(저장하지 않는다). 구성: 임계값별 전체 폭 줄(주 임계값이 가장 큼), 얼굴 상태 띠, 영상, 키 안내.

    ear_value 는 현재 프레임의 EAR 평균(detected 일 때만, 없으면 None), ear_min 은 최근 3.0초의 EAR 최솟값(없으면 None)이다. 둘 다 표시 전용이다.
    rate_lines 는 format_rate_lines 의 두 줄(없으면 None)이며 영상 왼쪽 아래에 작게 그린다(표시 전용).
    stage_line 은 format_stage_line 의 한 줄(없으면 None)이며 그 두 줄 바로 위에 그린다(표시 전용).
    """
    height, width = frame.shape[:2]
    order = [t for t in rules.thresholds if t == rules.primary] + [t for t in rules.thresholds if t != rules.primary]
    rows_h = sum(ROW_MAIN_H if t == rules.primary else ROW_OTHER_H for t in order)
    total_h = rows_h + STRIP_H + height + HINT_H
    canvas = np.zeros((total_h, width, 3), dtype=np.uint8)
    canvas[:] = COLOR_DARK
    flash = COLOR_RED if blink_is_red(now) else COLOR_ORANGE

    top = 0
    main_latched = False
    for threshold in order:
        is_main = threshold == rules.primary
        row_h = ROW_MAIN_H if is_main else ROW_OTHER_H
        mid = top + row_h // 2
        latched = latch.active(threshold, now)
        main_latched = main_latched or (is_main and latched)
        canvas[top:top + row_h, :] = flash if latched else (COLOR_GRAY if is_main else COLOR_GRAY_DARK)

        label = "%s ALARM" % threshold_label(threshold) if latched else threshold_label(threshold)
        label_thickness = 4 if is_main else 2
        label_scale = _fit_scale(cv2, label, label_thickness, int(width * 0.38), 2.0 if is_main else 1.0)
        cv2.putText(canvas, label, (14, top + (row_h * 11) // 20), cv2.FONT_HERSHEY_SIMPLEX, label_scale, COLOR_WHITE,
                    label_thickness, cv2.LINE_AA)
        last = rules.last_alarm_time(threshold)
        last_text = "last alarm: -" if last is None else "last alarm: %.1f s ago" % max(0.0, now - last)
        last_scale = _fit_scale(cv2, last_text, 1, int(width * 0.38), 0.55 if is_main else 0.5)
        cv2.putText(canvas, last_text, (14, top + row_h - 8), cv2.FONT_HERSHEY_SIMPLEX, last_scale, COLOR_WHITE, 1, cv2.LINE_AA)

        bar_x0, bar_x1 = int(width * 0.40), int(width * 0.72)
        half = 17 if is_main else 11
        cv2.rectangle(canvas, (bar_x0, mid - half), (bar_x1, mid + half), COLOR_WHITE, 2)
        fraction = rules.bar_fraction(threshold)
        fill_x = bar_x0 + int((bar_x1 - bar_x0) * fraction)
        if fill_x > bar_x0 + 2:
            cv2.rectangle(canvas, (bar_x0 + 2, mid - half + 2), (fill_x, mid + half - 2), COLOR_RED if fraction >= 1.0 else COLOR_BAR, -1)
        cv2.putText(canvas, "%.1f s" % rules.closed_length_s(threshold), (bar_x0 + 6, mid + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    COLOR_WHITE, 1, cv2.LINE_AA)

        count_x0, count_x1 = int(width * 0.74), int(width * 0.88)
        _put_centered(cv2, canvas, str(rules.alarm_count(threshold)), count_x0, count_x1, mid, 2.4 if is_main else 1.4,
                      4 if is_main else 2, COLOR_WHITE)
        cv2.putText(canvas, "alarms", (int(width * 0.885), mid + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_WHITE, 1, cv2.LINE_AA)
        if latched:
            cv2.rectangle(canvas, (2, top + 2), (width - 3, top + row_h - 3), COLOR_WHITE, 3)
        top += row_h

    status = rules.face_status()
    if status == "ok":
        canvas[top:top + STRIP_H, :] = COLOR_DARK
        _put_centered(cv2, canvas, "face ok", 0, width, top + STRIP_H // 2, 0.6, 1, (170, 170, 170))
    else:
        canvas[top:top + STRIP_H, :] = COLOR_YELLOW
        label = "FACE MISSING" if status == "missing" else "UNJUDGEABLE (no face for %.1f s or more)" % EM.UNJUDGEABLE_AFTER_S
        _put_centered(cv2, canvas, label, 0, width, top + STRIP_H // 2, 0.7, 2, COLOR_BLACK)
    top += STRIP_H

    video = frame.copy()
    for polygon in eye_polygons:
        cv2.polylines(video, [polygon], True, (0, 255, 0), 1)
    ear_text = "EAR now: -" if ear_value is None else "EAR now: %.3f" % ear_value
    min_text = "min 3s: -" if (ear_value is None or ear_min is None) else "min 3s: %.3f" % ear_min
    ear_scale = _fit_scale(cv2, "EAR now: 0.000", 2, width - 20, 1.0)
    cv2.putText(video, ear_text, (10, 34), cv2.FONT_HERSHEY_SIMPLEX, ear_scale, COLOR_WHITE, 2, cv2.LINE_AA)
    cv2.putText(video, min_text, (10, 34 + int(36 * ear_scale) + 8), cv2.FONT_HERSHEY_SIMPLEX, ear_scale, COLOR_WHITE, 2, cv2.LINE_AA)
    primary_text = "primary: %s" % threshold_label(rules.primary)         # 지금 주 임계값(목록의 첫 값)
    primary_width, _primary_height = _text_size(cv2, primary_text, 0.6, 2)
    cv2.putText(video, primary_text, (max(10, width - primary_width - 10), 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 2,
                cv2.LINE_AA)
    sound_width, _sound_height = _text_size(cv2, sound_label, 0.6, 2)
    cv2.putText(video, sound_label, (max(10, width - sound_width - 10), height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                COLOR_RED if sound_label == "sound: error" else COLOR_YELLOW, 2, cv2.LINE_AA)
    if rate_lines:
        rate_scale = min(_fit_scale(cv2, line, 1, int(width * 0.6), 0.5) for line in rate_lines)
        for index, line in enumerate(rate_lines):                             # 왼쪽 아래: 왼쪽 위 EAR, 오른쪽 위 primary, 오른쪽 아래 sound 와 겹치지 않는다
            y = height - 12 - 22 * (len(rate_lines) - 1 - index)
            cv2.putText(video, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, rate_scale, COLOR_WHITE, 1, cv2.LINE_AA)
    if stage_line:
        stage_scale = _fit_scale(cv2, stage_line, 1, width - 20, 0.5)          # 맨 아래 줄 높이에만 sound 표시가 있으므로 그 위 세 번째 줄은 폭 전체를 쓸 수 있다
        stage_y = height - 12 - 22 * (len(rate_lines) if rate_lines else 0)
        cv2.putText(video, stage_line, (10, stage_y), cv2.FONT_HERSHEY_SIMPLEX, stage_scale, COLOR_WHITE, 1, cv2.LINE_AA)
    canvas[top:top + height, :] = video
    top += height

    hint_scale = _fit_scale(cv2, HINT_TEXT, 1, width - 20, 0.6)
    _put_centered(cv2, canvas, HINT_TEXT, 0, width, top + HINT_H // 2, hint_scale, 1, COLOR_WHITE)

    if main_latched:
        cv2.rectangle(canvas, (0, 0), (width - 1, total_h - 1), flash, 10)
    return canvas


def run(camera, predictor_path, beep=False, beep_all=False, thresholds=THRESHOLDS):
    """웹캠을 열어 실시간 데모를 실행한다. 종료 코드를 돌려준다."""
    import cv2
    import dlib
    import imutils
    import numpy as np
    from imutils import face_utils
    from .blink_video_dump import adjust_gamma

    if not os.path.isfile(predictor_path):
        print("랜드마크 모델 파일을 찾을 수 없습니다: --predictor 경로를 확인하세요.")
        return 1
    capture = cv2.VideoCapture(camera)
    if not capture.isOpened():
        print("웹캠을 열 수 없습니다(--camera %s): 다른 프로그램이 쓰고 있거나 인덱스가 다를 수 있습니다." % camera)
        return 1
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    # 아래 세 값은 드라이버가 알려 주는 값이며 실제 프레임 도착 간격과 다를 수 있다(실제 간격은 FrameRateMeter 가 도착 시각으로 잰다). 한 번만 읽는다.
    cam_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    reported_fps = capture.get(cv2.CAP_PROP_FPS)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    (l_start, l_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (r_start, r_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    rules = LiveRules(thresholds)
    latch = DisplayLatch()
    beeper = Beeper(enabled=beep, beep_all=beep_all, primary=rules.primary)
    recent = RecentMinimum()                         # 최근 3초 EAR 최솟값(표시 전용, 메모리만, R 키로 지워지지 않는다)
    meter = FrameRateMeter()                         # 최근 3초 처리 fps와 최대 프레임 간격(표시 전용, 메모리만)
    stage_meter = StageTimeMeter()                   # 최근 3초 읽기 대기, 처리, 검출 시간과 읽기 실패 횟수(표시 전용, 메모리만)
    try:
        while True:
            read_start = time.perf_counter()         # 진단 전용 시각. arrival 과 규칙에는 쓰지 않는다
            grabbed, frame = capture.read()
            arrival = time.monotonic()               # 프레임이 들어온 시각이 규칙의 시간이다
            read_end = time.perf_counter()
            if not grabbed:
                stage_meter.add((read_end - read_start) * 1000.0, None, None, True, read_end)
                print("웹캠에서 프레임을 읽지 못해 끝냅니다.")
                break
            meter.add(arrival)                       # arrival 은 읽기만 하고 바꾸지 않는다
            small = imutils.resize(frame, width=PREPROCESS_WIDTH)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            gray = adjust_gamma(gray, gamma=PREPROCESS_GAMMA)
            detect_start = time.perf_counter()
            rects = detector(gray, 0)
            detect_ms = (time.perf_counter() - detect_start) * 1000.0
            ear_value = None
            eye_polygons = []
            if np.size(rects) != 0:
                shape = face_utils.shape_to_np(predictor(gray, rects[0]))
                landmarks = [(int(x), int(y)) for x, y in shape]
                _left, _right, ear_value = ear_math.ear_from_landmarks(landmarks)
                scale = frame.shape[1] / float(small.shape[1])
                for start, end in ((l_start, l_end), (r_start, r_end)):
                    eye_polygons.append((shape[start:end] * scale).astype(np.int32))
                recent.add(arrival, ear_value)
                new_alarms = rules.update(arrival, ear_value, "detected")
            else:
                new_alarms = rules.update(arrival, None, "none")
            for threshold, _alarm in new_alarms:
                latch.trigger(threshold, arrival)
            beeper.on_new_alarms(new_alarms)
            ear_min = recent.minimum(arrival) if ear_value is not None else None
            process_end = time.perf_counter()        # 처리 시간은 읽은 직후부터 여기까지(draw_ui, imshow, waitKey 는 포함하지 않는다)
            stage_meter.add((read_end - read_start) * 1000.0, (process_end - read_end) * 1000.0, detect_ms, False, process_end)
            rate_fps, rate_gap_ms = meter.stats()
            rate_lines = format_rate_lines(rate_fps, rate_gap_ms, meter.window_s, cam_width, cam_height, reported_fps)
            stage_line = format_stage_line(*stage_meter.stats())
            cv2.imshow("live_demo", draw_ui(cv2, np, frame, eye_polygons, rules, latch, ear_value, ear_min, beeper.label(), arrival,
                                            rate_lines, stage_line))
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key in (ord("r"), ord("R")):
                latch.reset()                        # 표시만 끈다(알람 횟수와 규칙 상태는 그대로)
                beeper.stop()                        # 재생 중인 소리에도 정지 신호를 올린다(현재 조각 뒤 멈춤)
            if key == ord("m"):
                beeper.toggle_mute()                 # 음소거로 바뀌면 소리에도 정지 신호를 올린다
    finally:
        beeper.stop()
        capture.release()
        cv2.destroyAllWindows()
    return 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="웹캠 실시간 졸음 규칙 데모(사용자 육안 확인용, 아무것도 저장하지 않음).")
    parser.add_argument("--predictor", required=True, help="dlib 랜드마크 모델(.dat) 경로")
    parser.add_argument("--camera", type=int, default=0, help="웹캠 인덱스(기본 0, 640x480 요청)")
    parser.add_argument("--beep", action="store_true", help="알람 순간에 경보음(2음 사이렌, 3.0초)을 낸다(기본 꺼짐, 기본은 주 임계값만)")
    parser.add_argument("--beep-all", dest="beep_all", action="store_true",
                        help="--beep 과 함께 쓰면 모든 임계값에서 경보음(--beep 없이는 소리 없음)")
    parser.add_argument("--thresholds", default=",".join(str(value) for value in THRESHOLDS),
                        help="닫힘 임계값 목록(쉼표로 구분한 1~%d개, 0 초과 1 미만, 첫 값이 주 임계값, 기본 %s)"
                             % (MAX_THRESHOLDS, ",".join(str(value) for value in THRESHOLDS)))
    return parser.parse_args(argv)


def main(argv=None):
    arguments = parse_args(argv)
    try:
        thresholds = parse_thresholds(arguments.thresholds)
    except ValueError as error:
        print(error)                                  # 한 줄 메시지를 출력하고 끝낸다
        return 2
    return run(arguments.camera, arguments.predictor, arguments.beep, arguments.beep_all, thresholds)


if __name__ == "__main__":
    sys.exit(main())
