# webcam_hook_speed.py
# 역할: 원본 깜빡임 추출(third_party 의 blink_video.py)을 웹캠 입력으로 후킹해 돌렸을 때의 처리 속도를 사용자가 직접 재는 도구(결정 I-28 (a)안의
#       웹캠 후킹은 실행한 이력이 없다). 점수 계산(TensorFlow)은 하지 않고 cv2 와 dlib 만 쓴다. 실시간 게이지 구현 여부(I-W-37)를 정하는 데 쓴다.
# 방법: original_hook.py 의 치환 방식을 그대로 쓴다. 원본 소스를 텍스트로 읽어 메모리에서 문자열만 바꾼 뒤 exec 한다. 이 도구가 따로 바꾸는 것은
#       영상 경로 치환 한 곳뿐이다: `path = ''` 를 `path = <웹캠 번호>` (정수)로 바꾼다(original_hook.build_replacements 는 문자열 경로만 만든다).
#       나머지 치환(출력 파일 이름, predictor, SVM, plot_frame.draw() 2곳, cv2.imshow 2곳, waitKey 2곳)은 original_hook 의 것과 같다.
#       waitKey 자리에는 진행 호출(tick)이 들어가 반복마다 한 번 불린다. 이 도구의 tick 은 호출 시각을 메모리에 쌓고, 측정 시간이 지나면 0 을 돌려
#       원본 루프를 끝낸다(원본은 waitKey 값이 255 가 아니면 루프를 끝낸다). 깜빡임 행은 original_hook.MemoryBlinkSink 의 on_row 콜백으로 받고 시각만 쌓는다.
# 창: 띄우지 않는다(original_hook 이 cv2.imshow 와 plot_frame.draw() 를 None 으로 바꾸므로). 이유: 화면 그리기 시간을 빼고 추출 경로만 재기 위해서이다.
#     따라서 키 입력으로 멈출 수 없고, 측정 시간이 지나거나 Ctrl+C 를 누르면 끝난다(중단돼도 요약을 출력한다).
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
import time

import numpy as np

from . import original_hook as OH

DEFAULT_SECONDS = 60.0
NOTE = "이 값은 원본 후킹 경로의 웹캠 속도 측정이며 점수 계산은 포함하지 않음"


class Meter(object):
    """tick 호출 시각과 행 확정 시각을 메모리에 쌓는다. 시각 함수는 시험에서 바꿀 수 있다."""

    def __init__(self, duration_s, clock=time.perf_counter):
        self.duration_s = duration_s
        self.clock = clock
        self.ticks = []
        self.row_times = []

    def tick(self):
        now = self.clock()
        self.ticks.append(now)
        if now - self.ticks[0] >= self.duration_s:
            return 0                                   # 255 가 아니므로 원본 루프가 끝난다
        return OH.NO_KEY

    def on_row(self, _row):
        self.row_times.append(self.clock())


def summarize(ticks, row_times):
    """tick 시각 목록과 행 시각 목록에서 요약 값(dict)을 만든다. 순수 함수. 값을 낼 수 없는 항목은 None.

    frames: tick 횟수. fps: (frames - 1) / (마지막 tick - 첫 tick). gap_*_ms: 인접 tick 간격(ms)의 중앙값, 95 백분위, 최댓값.
    rows: 확정된 행 수. last_row_s: 마지막 행 시각 - 첫 tick 시각(초, 행이 없으면 None).
    """
    frames = len(ticks)
    summary = {"frames": frames, "fps": None, "gap_median_ms": None, "gap_p95_ms": None, "gap_max_ms": None,
               "rows": len(row_times), "last_row_s": None, "span_s": None}
    if frames >= 2:
        gaps = np.diff(np.asarray(ticks, dtype=np.float64)) * 1000.0
        span = ticks[-1] - ticks[0]
        summary["span_s"] = span
        summary["fps"] = (frames - 1) / span if span > 0 else None
        summary["gap_median_ms"] = float(np.median(gaps))
        summary["gap_p95_ms"] = float(np.percentile(gaps, 95))
        summary["gap_max_ms"] = float(np.max(gaps))
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
        NOTE,
    ]


def find_svm():
    """third_party 의 SVM 파일(.sav)을 찾는다. 정확히 한 개여야 한다."""
    found = sorted(glob.glob(os.path.join(os.path.dirname(OH.ORIGINAL_PATH), "*.sav")))
    if len(found) != 1:
        raise FileNotFoundError("third_party 에서 SVM(.sav) 파일 하나를 찾지 못했습니다: %d개" % len(found))
    return found[0]


def build_source(camera, predictor, svm, original=None):
    """원본 소스를 웹캠 입력용으로 치환한 문자열. 영상 경로 자리에 정수 카메라 번호가 들어간다. 치환 개수가 기대와 다르면 ReplacementError."""
    original = OH.read_original_source() if original is None else original
    replacements = []
    for label, old, new, expected in OH.build_replacements("", predictor, svm, "none", True):
        if label == "input video":
            new = "path = %d" % int(camera)
        replacements.append((label, old, new, expected))
    OH.check_counts(original, replacements)
    return OH.apply_replacements(original, replacements)


def _silent_print(*_args, **_kwargs):
    return None


def run_measurement(source, meter):
    """치환된 소스를 메모리에서 실행하고 상태 문자열을 돌려준다. 중단이나 예외가 나도 쌓인 값은 meter 에 남는다."""
    sink = OH.MemoryBlinkSink(on_row=meter.on_row)
    namespace = {"__name__": "blink_video_inmemory", "open": OH.make_open_override(sink), OH.TICK_NAME: meter.tick, "print": _silent_print}
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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="원본 깜빡임 추출을 웹캠 입력으로 후킹했을 때의 처리 속도를 잰다(콘솔 요약만, 아무것도 저장하지 않음).")
    parser.add_argument("camera", nargs="?", type=int, default=0, help="웹캠 번호(기본 0)")
    parser.add_argument("seconds", nargs="?", type=float, default=DEFAULT_SECONDS, help="측정 시간(초, 기본 %g)" % DEFAULT_SECONDS)
    parser.add_argument("--predictor", required=True, help="dlib 68점 랜드마크 모델(.dat) 경로")
    return parser.parse_args(argv)


def main(argv=None):
    arguments = parse_args(argv)
    if arguments.seconds <= 0:
        print("측정 시간은 0보다 커야 합니다.")
        return 2
    if not os.path.isfile(arguments.predictor):
        print("랜드마크 모델 파일을 찾을 수 없습니다: --predictor 경로를 확인하세요.")
        return 1
    source = build_source(arguments.camera, arguments.predictor, find_svm())
    print("측정 시작: 카메라 %d, %g초(Ctrl+C 로 중단해도 요약을 출력합니다). 창은 뜨지 않습니다." % (arguments.camera, arguments.seconds))
    sys.stdout.flush()
    meter = Meter(arguments.seconds)
    status = run_measurement(source, meter)
    summary = summarize(meter.ticks, meter.row_times)
    for line in format_summary(summary, status):
        print(line)
    return 0 if status in ("finished", "interrupted") and summary["frames"] > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
