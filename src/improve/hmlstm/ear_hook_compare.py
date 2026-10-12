# ear_hook_compare.py
# 역할: 원본 blink_video.py 를 영상 파일 입력으로 후킹해 프레임마다 모은 EAR 열(콜백)과 replay_precompute.py 가 만든 EAR 열을 비교한다. 규칙 알람을 원본 후킹의 EAR로 구동할 수 있는지
#       (읽기 전용 점검 2026-10-12)를 실행으로 확인하는 도구이다. 모델 점수와 TensorFlow는 쓰지 않는다. 프레임과 이미지는 저장하지 않고 콘솔 요약만 출력한다.
# 후킹: original_hook 의 치환(출력, 영상 경로, predictor, SVM, plot_frame.draw() 2곳, imshow 2곳, waitKey 2곳을 tick 으로)에 아래 치환을 더한다. 각각 원본에서 정확히 한 번 일치하고 줄 수는 그대로이다.
#       blink_video.py 421행(얼굴 검출 분기):  ear = (leftEAR + rightEAR) / 2.0  뒤에  ; _hook_ear(ear, 'detected', number_of_frames)
#       blink_video.py 506행(얼굴 없음 분기, 광학 흐름 추적 성공):  같은 문장 뒤에  ; _hook_ear(ear, 'tracked', number_of_frames)
#       두 문장은 들여쓰기(12칸, 16칸)로 구분한다(앞에 줄바꿈을 붙여 각각 한 번만 일치). number_of_frames 는 blink_detector 함수의 지역 변수로 350행에서 0으로 만들고
#       380행(얼굴 검출 분기)에서만 1씩 올리며, 421행과 506행은 같은 함수의 같은 while 루프 안이라 같은 범위에서 읽힌다(시험이 ast 로 확인한다).
#       빈 Tk 창은 webcam_hook_speed 와 같은 치환(top = tk.Tk(); top.withdraw())으로 숨기고 영상 창은 original_hook 이 imshow 를 없애므로 뜨지 않는다.
# 표본: tick(waitKey 자리)이 불릴 때마다 한 반복이 한 표본이다 = (프레임 번호, 얼굴 검출 여부, EAR 또는 없음, 출처 detected/tracked/none, number_of_frames).
#       콜백이 불리지 않은 반복(얼굴도 없고 추적도 실패)은 출처 none, EAR 없음, number_of_frames 는 직전 값이다. 콜백은 예외를 삼키고 건수만 센다.
# 비교: 프레임 수, 얼굴 유무 일치 비율, detected 프레임의 EAR 차이(최댓값, 평균, 1e-6 초과, 1e-9 초과), number_of_frames 와 replay_precompute 의 얼굴 있음 프레임 누적 개수의 일치,
#       두 열을 같은 시각(프레임 번호 / fps)으로 LiveRules 에 넣었을 때의 알람. --rows-json 이 있으면 이 후킹의 깜빡임 행도 그 JSON 의 행과 대조한다.
# 사용: PYTHONPATH=src python -m improve.hmlstm.ear_hook_compare --video <VIDEO> --predictor <PREDICTOR> --rules-json <TIMELINE_DIR>/rules.json [--rows-json <TIMELINE_DIR>/rows.json] [--rotate auto] [--threshold 0.15]
"""원본 후킹 EAR 열과 replay_precompute EAR 열 비교(Python 3.6 호환)."""
import argparse
import collections
import json
import os
import sys

import numpy as np

from . import compare_rows as CR
from . import original_hook as OH
from . import timeline as TL
from . import webcam_hook_speed as WH
from ..rules import live_demo as LD
from ..rules import replay_precompute as RP

DETECTED_TARGET = "\n            ear = (leftEAR + rightEAR) / 2.0"
DETECTED_REPLACEMENT = DETECTED_TARGET + "; _hook_ear(ear, 'detected', number_of_frames)"
TRACKED_TARGET = "\n                ear = (leftEAR + rightEAR) / 2.0"
TRACKED_REPLACEMENT = TRACKED_TARGET + "; _hook_ear(ear, 'tracked', number_of_frames)"
HOOK_NAME = "_hook_ear"
DEFAULT_THRESHOLD = 0.15
PROGRESS_EVERY = 1000
NOTE = "이 값은 EAR 열 비교이며 규칙 알람 검증을 대체하지 않음"
ROTATE_CHOICES = tuple(OH.ROTATE_OPTIONS) + ("auto",)

Sample = collections.namedtuple("Sample", ["face", "ear", "source", "faces"])


class CompareError(ValueError):
    """입력 인자나 파일의 오류(메시지는 한 줄로 사용자에게 보인다)."""


# ----- 치환 -----
def build_source(video, predictor, svm, rotate="none", original=None):
    """원본 소스에 original_hook 의 치환과 이 도구의 치환 3개(Tk 숨김, EAR 콜백 2개)를 적용한 문자열. 개수가 기대와 다르면 ReplacementError."""
    original = OH.read_original_source() if original is None else original
    replacements = list(OH.build_replacements(video, predictor, svm, rotate, True))
    replacements.append(("hide empty Tk window", WH.TK_TARGET, WH.TK_REPLACEMENT, 1))
    replacements.append(("ear callback (face detected)", DETECTED_TARGET, DETECTED_REPLACEMENT, 1))
    replacements.append(("ear callback (optical flow tracked)", TRACKED_TARGET, TRACKED_REPLACEMENT, 1))
    OH.check_counts(original, replacements)
    return OH.apply_replacements(original, replacements)


# ----- 표본 모으기 -----
class EarTap(object):
    """tick 마다 한 표본을 만든다. 콜백(ear)은 마지막 반복의 EAR 를 잠깐 들고 있다가 tick 이 표본으로 바꾼다. 예외는 삼키고 errors 만 센다."""

    def __init__(self, progress_every=PROGRESS_EVERY, out=print):
        self.samples = []
        self.pending = None
        self.last_faces = 0
        self.errors = 0
        self.progress_every = progress_every
        self.out = out

    def ear(self, value, source, faces):
        try:
            self.pending = (float(value), str(source), int(faces))
        except Exception:
            self.errors += 1

    def tick(self):
        try:
            if self.pending is None:
                sample = Sample(False, None, "none", self.last_faces)
            else:
                value, source, faces = self.pending
                self.last_faces = faces
                sample = Sample(source == "detected", value, source, faces)
            self.pending = None
            self.samples.append(sample)
            if self.progress_every and len(self.samples) % self.progress_every == 0:
                self.out("frames=%d" % len(self.samples))
                sys.stdout.flush()
        except Exception:
            self.errors += 1
        return OH.NO_KEY


def _silent_print(*_args, **_kwargs):
    return None


def run_source(source, tap):
    """치환된 소스를 메모리에서 실행한다. 반환: (상태 문자열, 깜빡임 행 목록). 원본이 만드는 창은 실행이 끝난 뒤 닫는다."""
    sink = OH.MemoryBlinkSink()
    namespace = {"__name__": "blink_video_inmemory", "open": OH.make_open_override(sink), OH.TICK_NAME: tap.tick, HOOK_NAME: tap.ear, "print": _silent_print}
    status = "finished"
    try:
        exec(compile(source, "blink_video.py[ear-hook-compare]", "exec"), namespace)
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
    return status, sink.rows


# ----- 비교(순수 함수) -----
def compare_columns(samples, frames):
    """후킹 표본 목록과 replay_precompute 의 프레임 목록(각 항목 {"face", "ear", ...})을 비교한다. 길이가 다르면 앞쪽 공통 구간만 비교하고 frames_equal 이 False 이다."""
    count = min(len(samples), len(frames))
    same_face = tracked = faces_mismatch = 0
    first_face_mismatch = first_faces_mismatch = None
    cumulative = 0
    diffs = []
    for index in range(count):
        sample, frame = samples[index], frames[index]
        if frame["face"]:
            cumulative += 1
        if sample.face == frame["face"]:
            same_face += 1
        elif first_face_mismatch is None:
            first_face_mismatch = index
        if sample.source == "tracked":
            tracked += 1
        if sample.face and frame["face"]:
            diffs.append(abs(sample.ear - frame["ear"]))
        if sample.faces != cumulative:
            faces_mismatch += 1
            if first_faces_mismatch is None:
                first_faces_mismatch = index
    diffs = np.asarray(diffs, dtype=np.float64)
    return {"hook_frames": len(samples), "rules_frames": len(frames), "frames_equal": len(samples) == len(frames), "compared": count,
            "face_same_ratio": (same_face / float(count)) if count else None, "face_mismatch_frames": count - same_face, "first_face_mismatch": first_face_mismatch,
            "tracked_frames": tracked, "ear_pairs": int(diffs.size), "ear_max": float(diffs.max()) if diffs.size else None,
            "ear_mean": float(diffs.mean()) if diffs.size else None, "ear_over_1e6": int((diffs > 1e-6).sum()), "ear_over_1e9": int((diffs > 1e-9).sum()),
            "faces_mismatch_frames": faces_mismatch, "first_faces_mismatch": first_faces_mismatch,
            "hook_faces_final": samples[count - 1].faces if count else None, "rules_faces_final": cumulative if count else None}


def hook_series(samples):
    """표본 목록을 LiveRules 입력 (EAR 또는 None, 출처) 목록으로 바꾼다. 추적(tracked)은 출처를 그대로 넘겨 기본 유효 출처(detected)에서 빠지게 한다."""
    return [(sample.ear, sample.source) if sample.source in ("detected", "tracked") else (None, "none") for sample in samples]


def rules_series(frames):
    """replay_precompute 프레임 목록을 live_demo 와 같은 (EAR 또는 None, 출처) 목록으로 바꾼다."""
    return [(frame["ear"], "detected") if frame["face"] else (None, "none") for frame in frames]


def alarm_times(series, fps, threshold=DEFAULT_THRESHOLD):
    """(EAR, 출처) 열을 프레임 번호 / fps 시각으로 LiveRules 에 넣어 알람 시각 목록을 돌려준다."""
    rules = LD.LiveRules((threshold,))
    for index, (ear, source) in enumerate(series):
        rules.update(index / float(fps), ear, source)
    return [alarm.alarm_time_s for alarm in rules.machines[threshold].alarms]


def compare_alarms(hook_times, rules_times, tolerance=1e-9):
    same_count = len(hook_times) == len(rules_times)
    differences = [abs(a - b) for a, b in zip(hook_times, rules_times)]
    return {"hook_alarms": len(hook_times), "rules_alarms": len(rules_times), "same_count": same_count,
            "max_time_diff": max(differences) if differences else None,
            "same": same_count and all(value <= tolerance for value in differences)}


def verdict(columns, alarms, tolerance=1e-9):
    """일치 조건: 프레임 수가 같고, 얼굴 유무와 number_of_frames 가 모든 프레임에서 같고, EAR 차이가 tolerance 이하이며, 알람이 같다. 아니면 불일치."""
    ok = (columns["frames_equal"] and columns["face_mismatch_frames"] == 0 and columns["faces_mismatch_frames"] == 0
          and (columns["ear_max"] is None or columns["ear_max"] <= tolerance) and alarms["same"])
    return "일치" if ok else "불일치"


def format_report(status, columns, alarms, threshold, rows_report=None, rotate_note=None, errors=0):
    def number(value, pattern):
        return "-" if value is None else pattern % value

    lines = ["status: %s" % status,
             "hook frames / rules frames: %d / %d ; equal: %s" % (columns["hook_frames"], columns["rules_frames"], "yes" if columns["frames_equal"] else "no"),
             "face flag equal ratio: %s ; frames with different face flag: %d (first: %s) ; tracked frames: %d" % (
                 number(columns["face_same_ratio"], "%.6f"), columns["face_mismatch_frames"],
                 "-" if columns["first_face_mismatch"] is None else columns["first_face_mismatch"], columns["tracked_frames"]),
             "EAR diff on %d shared detected frames: max %s ; mean %s ; over 1e-6: %d ; over 1e-9: %d" % (
                 columns["ear_pairs"], number(columns["ear_max"], "%.3e"), number(columns["ear_mean"], "%.3e"), columns["ear_over_1e6"], columns["ear_over_1e9"]),
             "number_of_frames vs cumulative face frames: different frames %d (first: %s) ; final hook %s / rules %s" % (
                 columns["faces_mismatch_frames"], "-" if columns["first_faces_mismatch"] is None else columns["first_faces_mismatch"],
                 "-" if columns["hook_faces_final"] is None else columns["hook_faces_final"], "-" if columns["rules_faces_final"] is None else columns["rules_faces_final"]),
             "LiveRules alarms at threshold %g: hook %d / rules %d ; same count: %s ; max time diff (s): %s ; same: %s" % (
                 threshold, alarms["hook_alarms"], alarms["rules_alarms"], "yes" if alarms["same_count"] else "no", number(alarms["max_time_diff"], "%.3e"),
                 "yes" if alarms["same"] else "no")]
    if rows_report is not None:
        diffs = rows_report["max_abs_diff"]
        lines.append("blink rows: hook %d / file %d ; verdict: %s ; mismatched rows (4 decimals): %d ; max abs diff per column: %s" % (
            rows_report["hook_rows"], rows_report["reference_rows"], rows_report["verdict"], rows_report["rounded_mismatch_rows"],
            "-" if diffs is None else " ".join("%.1e" % value for value in diffs)))
    if rotate_note:
        lines.append(rotate_note)
    lines.append("callback errors swallowed: %d" % errors)
    lines.append("verdict: %s" % verdict(columns, alarms))
    lines.append(NOTE)
    return lines


# ----- 입력과 실행 -----
def load_rules_document(path):
    with open(path, "r", encoding="utf-8") as handle:
        try:
            document = json.load(handle)
        except ValueError as error:
            raise CompareError("규칙 JSON 을 읽을 수 없습니다: %s" % error)
    try:
        return RP.validate_document(document)
    except RP.PrecomputeError as error:
        raise CompareError("규칙 JSON 형식 오류: %s" % error)


def resolve_rotation(rotate, rules_document):
    """--rotate 를 original_hook 회전 옵션으로 정한다. auto 는 규칙 JSON 에 기록된 적용 각도를 따른다. 반환: (옵션, 안내 문장 또는 None)."""
    derived = TL.angle_to_option(rules_document["rotation"]["applied_angle"])
    if rotate == "auto":
        return derived, None
    if rotate != derived:
        return rotate, "warning: --rotate %s differs from the rotation used for the rules JSON (%s)" % (rotate, derived)
    return rotate, None


def run(video, predictor, rules_path, rows_path, rotate, threshold):
    if not os.path.isfile(video):
        raise CompareError("영상 파일을 찾을 수 없습니다")
    if not os.path.isfile(predictor):
        raise CompareError("랜드마크 모델 파일을 찾을 수 없습니다: --predictor 경로를 확인하세요.")
    rules_document = load_rules_document(rules_path)
    rows_document = TL.read_rows_document(rows_path) if rows_path else None
    option, rotate_note = resolve_rotation(rotate, rules_document)
    source = build_source(video, predictor, WH.find_svm(), option)
    print("측정 시작: 영상 전체를 읽습니다(창은 뜨지 않습니다, Ctrl+C 로 중단하면 지금까지의 표본으로 비교합니다).")
    sys.stdout.flush()
    tap = EarTap()
    status, rows = run_source(source, tap)
    frames = rules_document["frames"]
    fps = rules_document["fps"]
    columns = compare_columns(tap.samples, frames)
    alarms = compare_alarms(alarm_times(hook_series(tap.samples), fps, threshold), alarm_times(rules_series(frames), fps, threshold))
    rows_report = CR.compare_rows(rows, [list(row) for row in rows_document["rows"]]) if rows_document is not None else None
    for line in format_report(status, columns, alarms, threshold, rows_report, rotate_note, tap.errors):
        print(line)
    return 0 if status == "finished" else 1


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="원본 후킹이 모은 프레임별 EAR 열과 replay_precompute 의 EAR 열을 비교한다(콘솔 요약만, 아무것도 저장하지 않음).")
    parser.add_argument("--video", required=True, help="영상 파일")
    parser.add_argument("--predictor", required=True, help="dlib 68점 랜드마크 모델(.dat) 경로")
    parser.add_argument("--rules-json", dest="rules_json", required=True, help="replay_precompute.py 가 같은 영상에서 만든 규칙용 JSON")
    parser.add_argument("--rows-json", dest="rows_json", default=None, help="(선택) timeline extract 가 같은 영상에서 만든 행 JSON. 있으면 후킹이 만든 깜빡임 행과 대조")
    parser.add_argument("--rotate", choices=list(ROTATE_CHOICES), default="auto", help="후킹의 회전 옵션(기본 auto = 규칙 JSON 에 기록된 적용 각도)")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="LiveRules 닫힘 임계값(기본 %g)" % DEFAULT_THRESHOLD)
    return parser.parse_args(argv)


def main(argv=None):
    arguments = parse_args(argv)
    if not 0.0 < arguments.threshold < 1.0:
        print("임계값은 0 초과 1 미만이어야 합니다.")
        return 2
    try:
        return run(arguments.video, arguments.predictor, arguments.rules_json, arguments.rows_json, arguments.rotate, arguments.threshold)
    except (CompareError, TL.TimelineError, OH.ReplacementError, FileNotFoundError) as error:
        print("ear_hook_compare: %s" % error)
        return 2


if __name__ == "__main__":
    sys.exit(main())
