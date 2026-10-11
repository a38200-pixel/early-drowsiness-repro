# timeline.py
# 역할: 영상 파일에서 HM-LSTM 게이지용 숫자만 뽑아 JSON 하나로 저장하는 도구(결정 I-W-34 (2): 숫자 행과 프레임 번호만 저장소 밖 출력 폴더에 저장하는
#       I-20의 한정 예외). 프레임, 이미지, 영상 조각은 저장하지 않는다. 지금은 서브명령 extract 하나뿐이고, 점수 계산(score)은 다음 단계에서
#       별도 프로세스로 만든다(cv2/dlib 와 TensorFlow 를 한 프로세스에서 같이 쓴 적이 없어서 미검증이기 때문).
# extract: original_hook.extract_rows(record_frames=True)로 블링크 행과 행별 프레임 번호를 얻고, check_frame_indices 로 형식을 검사한 뒤 저장한다.
#       무거운 라이브러리(cv2, dlib)는 이 서브명령 안에서만 불러온다(원본 실행은 original_hook 안에서 일어난다). 이 파일의 순수 함수는 그것 없이 시험된다.
# 출력 JSON(숫자만): schema_version(1), video_name(파일 이름만), fps, frames_read, rotation(적용한 original_hook 회전 옵션), columns(5개 이름),
#       rows(후크가 낸 값 그대로), frame_indices(행과 같은 길이의 정수, 행이 기록된 반복의 0부터 센 번호).
# 안전장치: --out 이 저장소 폴더 안이면 거부, 이미 있는 파일은 덮어쓰지 않음(덮어쓰기 옵션 없음), 상위 폴더가 없으면 만든다.
#       이 검사는 영상 처리 전에 먼저 하고 쓰는 순간에 다시 한다(독점 생성 'x').
# 사용법: (저장소 폴더에서 conda activate ghoddoosian_tf1, set PYTHONPATH=src)
#           python -m improve.hmlstm.timeline extract --video <영상> --svm <SVM .sav> --predictor <dlib .dat> --out <저장소 밖 .json> [--rotate none|cw90|ccw90|180|auto]
#                                                      [--progress-every 500] [--max-frames N]
# 종료 코드: 0 = 저장함, 1 = 프레임 번호 형식 검사 불일치(저장하지 않음), 2 = 인자나 입력 오류
"""블링크 행과 프레임 번호의 JSON 추출 도구(Python 3.6 호환)."""
import argparse
import json
import os
import sys
import time

from . import original_hook as OH

SCHEMA_VERSION = 1
COLUMNS = ("total_blinks", "freq_x100", "amp", "dur", "vel")        # 후크 행 5열: [TOTAL_BLINKS, Freq*100, Amp, Dur, Vel]
ROTATION_BY_ANGLE = {0: "none", 90: "cw90", 180: "180", 270: "ccw90"}      # 메타데이터 각도(시계 방향) -> original_hook 회전 옵션
ROTATE_CHOICES = tuple(OH.ROTATE_OPTIONS) + ("auto",)
REQUIRED_KEYS = ("schema_version", "video_name", "fps", "frames_read", "rotation", "columns", "rows", "frame_indices")


class TimelineError(ValueError):
    """문서 형식, 경로, 입력 인자의 오류(메시지는 한 줄로 사용자에게 보인다)."""


# ----- 순수 함수(cv2/TensorFlow 불필요) -----
def angle_to_option(angle):
    """메타데이터 각도(0, 90, 180, 270)를 original_hook 회전 옵션으로 바꾼다. 다른 값이면 TimelineError."""
    if angle not in ROTATION_BY_ANGLE:
        raise TimelineError("지원하지 않는 회전 각도입니다: %r" % (angle,))
    return ROTATION_BY_ANGLE[angle]


def build_rows_document(video_name, fps, frames_read, rotation, rows, frame_indices):
    """저장할 문서(dict)를 만든다. 행은 값을 그대로 float 목록으로 옮기고 반올림하지 않는다."""
    return {"schema_version": SCHEMA_VERSION, "video_name": video_name, "fps": float(fps), "frames_read": int(frames_read),
            "rotation": rotation, "columns": list(COLUMNS), "rows": [[float(value) for value in row] for row in rows],
            "frame_indices": [int(value) for value in frame_indices]}


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def validate_document(document):
    """문서 형식을 검사해 문서를 돌려준다. 어긋나면 어디가 어떻게 틀렸는지 구체적으로 적은 TimelineError."""
    if not isinstance(document, dict):
        raise TimelineError("문서가 객체(dict)가 아닙니다")
    for key in REQUIRED_KEYS:
        if key not in document:
            raise TimelineError("필수 키가 없습니다: %s" % key)
    if document["schema_version"] != SCHEMA_VERSION:
        raise TimelineError("지원하지 않는 schema_version 입니다: %r (지원 %d)" % (document["schema_version"], SCHEMA_VERSION))
    name = document["video_name"]
    if not isinstance(name, str) or not name or os.path.basename(name) != name or "/" in name or "\\" in name:
        raise TimelineError("video_name 은 경로 없는 파일 이름이어야 합니다: %r" % (name,))
    if not _is_number(document["fps"]) or document["fps"] <= 0:
        raise TimelineError("fps 는 0 보다 큰 수여야 합니다: %r" % (document["fps"],))
    frames = document["frames_read"]
    if not isinstance(frames, int) or isinstance(frames, bool) or frames < 0:
        raise TimelineError("frames_read 는 0 이상의 정수여야 합니다: %r" % (frames,))
    if document["rotation"] not in OH.ROTATE_OPTIONS:
        raise TimelineError("rotation 은 %s 중 하나여야 합니다: %r" % ("/".join(OH.ROTATE_OPTIONS), document["rotation"]))
    if list(document["columns"]) != list(COLUMNS):
        raise TimelineError("columns 가 기대와 다릅니다: %r (기대 %r)" % (document["columns"], list(COLUMNS)))
    rows, indices = document["rows"], document["frame_indices"]
    if not isinstance(rows, list) or not isinstance(indices, list):
        raise TimelineError("rows 와 frame_indices 는 목록이어야 합니다")
    for position, row in enumerate(rows):
        if not isinstance(row, list) or len(row) != len(COLUMNS):
            raise TimelineError("rows[%d] 의 열 수가 %d 가 아닙니다: %r" % (position, len(COLUMNS), row if not isinstance(row, list) else len(row)))
        for column, value in enumerate(row):
            if not _is_number(value):
                raise TimelineError("rows[%d][%d] 가 숫자가 아닙니다: %r" % (position, column, value))
    for position, value in enumerate(indices):
        if not isinstance(value, int) or isinstance(value, bool):
            raise TimelineError("frame_indices[%d] 가 정수가 아닙니다: %r" % (position, value))
    errors = OH.check_frame_indices(rows, indices, frames)
    if errors:
        raise TimelineError("frame_indices 검사 실패: " + "; ".join(errors))
    return document


def _canonical(path):
    return os.path.normcase(os.path.realpath(os.path.abspath(path)))


def is_inside(path, folder):
    """path 가 folder 안(또는 folder 자체)인지 판정한다. 대소문자와 심볼릭 링크를 정리해 비교하고 다른 드라이브면 False."""
    target, base = _canonical(path), _canonical(folder)
    try:
        return os.path.commonpath([target, base]) == base
    except ValueError:
        return False


def check_output_path(path, repo=None):
    """--out 경로 점검: 저장소 안이면 거부, 이미 파일(또는 폴더)이 있으면 거부. 문제가 없으면 아무 일도 하지 않는다."""
    repo = OH.REPO if repo is None else repo
    if is_inside(path, repo):
        raise TimelineError("출력 경로가 저장소 폴더 안입니다. 저장소 밖 폴더를 지정하세요.")
    if os.path.exists(path):
        raise TimelineError("이미 있는 경로에는 덮어쓰지 않습니다: %s" % os.path.basename(path))


def write_rows_document(path, document, repo=None):
    """문서를 검사한 뒤 새 파일로 쓴다(독점 생성). 저장소 안, 이미 있는 파일은 TimelineError. 상위 폴더가 없으면 만든다."""
    validate_document(document)
    check_output_path(path, repo)
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    try:
        handle = open(path, "x", encoding="utf-8", newline="\n")
    except FileExistsError:
        raise TimelineError("이미 있는 경로에는 덮어쓰지 않습니다: %s" % os.path.basename(path))
    with handle:
        json.dump(document, handle, ensure_ascii=False, separators=(",", ":"))
        handle.write("\n")


def read_rows_document(path):
    """저장된 문서를 읽고 형식을 검사해 돌려준다. 형식이 틀리면 구체적인 TimelineError."""
    with open(path, "r", encoding="utf-8") as handle:
        try:
            document = json.load(handle)
        except ValueError as error:
            raise TimelineError("JSON 을 읽을 수 없습니다: %s" % error)
    return validate_document(document)


# ----- 서브명령 extract -----
def resolve_rotation(request, video):
    """--rotate 값을 original_hook 회전 옵션으로 바꾼다. auto 는 MP4 메타데이터에서 각도를 읽는다(읽지 못하거나 180도면 TimelineError)."""
    if request == "auto":
        from ..rules import video_rotation
        try:
            angle, _metadata = video_rotation.plan_rotation("auto", video)
        except video_rotation.RotationError as error:
            raise TimelineError(str(error))
        return angle_to_option(angle)
    if request not in OH.ROTATE_OPTIONS:
        raise TimelineError("알 수 없는 회전 옵션입니다: %r" % (request,))
    return request


def read_video_meta(video):
    """(fps, 메타데이터 프레임 수). 영상 헤더 값만 읽고 프레임을 저장하지 않는다. cv2 는 여기서만 불러온다."""
    import cv2
    capture = cv2.VideoCapture(video)
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    if fps <= 0:
        raise TimelineError("영상에서 fps 를 읽지 못했습니다")
    return fps, (count if count > 0 else None)


def run_extract(options):
    """extract 를 실행한다. 반환: 종료 코드."""
    try:
        check_output_path(options.out)                                  # 영상 처리 전에 먼저 거른다
        for name in ("video", "svm", "predictor"):
            if not os.path.isfile(getattr(options, name)):
                raise TimelineError("파일을 찾을 수 없습니다: --%s" % name)
        rotation = resolve_rotation(options.rotate, options.video)
        fps, metadata_frames = read_video_meta(options.video)
        print("extract: video=%s rotation=%s fps=%.3f frames(metadata)=%s max_frames=%d" % (
            os.path.basename(options.video), rotation, fps, metadata_frames, options.max_frames))
        started = time.time()
        result = OH.extract_rows(options.video, options.svm, options.predictor, rotation, options.progress_every, options.max_frames, True,
                                 total_frames=metadata_frames, record_frames=True)
        elapsed = time.time() - started
        errors = OH.check_frame_indices(result.rows, result.frame_indices, result.frames)
        if result.frames == 0:
            errors.append("no frames were read")
        partial = " (partial run: max-frames %d)" % options.max_frames if options.max_frames else ""
        print("extract: frames=%d rows=%d check=%s elapsed_s=%.1f(reference only)%s" % (
            result.frames, len(result.rows), "불일치" if errors else "일치", elapsed, partial))
        if errors:
            print("extract: not written; " + "; ".join(errors))
            return 1
        document = build_rows_document(os.path.basename(options.video), fps, result.frames, rotation, result.rows, result.frame_indices)
        write_rows_document(options.out, document)
        print("extract: written %s" % os.path.basename(options.out))
        return 0
    except (TimelineError, OH.ReplacementError, FileNotFoundError, ValueError) as error:
        print("extract: %s" % error)
        return 2


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="영상에서 블링크 행과 행별 프레임 번호를 뽑아 저장소 밖에 JSON 으로 저장한다(숫자만).")
    commands = parser.add_subparsers(dest="command")
    commands.required = True
    extract = commands.add_parser("extract", help="원본 훅으로 블링크 행과 프레임 번호를 추출해 JSON 으로 저장")
    extract.add_argument("--video", required=True, help="영상 파일 경로")
    extract.add_argument("--svm", required=True, help="저자 SVM(.sav) 경로")
    extract.add_argument("--predictor", required=True, help="dlib 68점 랜드마크(.dat) 경로")
    extract.add_argument("--out", required=True, help="출력 JSON 경로(저장소 밖, 이미 있으면 거부)")
    extract.add_argument("--rotate", choices=list(ROTATE_CHOICES), default="none", help="회전 옵션(기본 none, auto 는 MP4 메타데이터에서 읽음)")
    extract.add_argument("--progress-every", type=int, default=500, dest="progress_every", help="N프레임마다 한 줄 출력(기본 500, 0 이면 출력 없음)")
    extract.add_argument("--max-frames", type=int, default=0, dest="max_frames", help="앞 N프레임만 처리(스모크용 부분 실행)")
    return parser.parse_args(argv)


def main(argv=None):
    options = parse_args(argv)
    if options.command == "extract":
        return run_extract(options)
    return 2


if __name__ == "__main__":
    sys.exit(main())
