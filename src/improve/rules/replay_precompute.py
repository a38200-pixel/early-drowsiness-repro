# replay_precompute.py
# 역할: 재생 뷰어(replay_viewer.py, 결정 I-W-33 대안안)가 쓸 규칙용 숫자를 영상 한 편에서 미리 계산한다. 프레임마다 얼굴 상태, 평균 EAR, 두 눈 윤곽 좌표를
#       숫자 JSON 으로 만든다. 이미지, 프레임, 영상은 저장하지 않는다. 규칙(3.0초 알람)은 여기서 계산하지 않고 뷰어가 LiveRules 로 다시 계산한다.
# 전처리: live_demo.run 의 프레임 처리(632에서 646행)와 같다. 상수는 live_demo 에서 가져온다(PREPROCESS_WIDTH, PREPROCESS_GAMMA).
#       live_demo.run 은 웹캠 루프 안에 있어 가져와 쓸 수 없으므로 아래 process_frame 이 그 줄들을 같은 순서로 복제했다:
#         632 imutils.resize(frame, width=PREPROCESS_WIDTH)   633 cv2.cvtColor(small, COLOR_BGR2GRAY)   634 adjust_gamma(gray, gamma=PREPROCESS_GAMMA)
#         636 detector(gray, 0) (업샘플 0)   640 np.size(rects) != 0, 첫 얼굴만   641 shape_to_np(predictor(gray, rects[0]))
#         642에서 643 landmarks, ear_math.ear_from_landmarks   644에서 646 scale = frame.shape[1] / small.shape[1], 눈 윤곽 = shape[start:end] * scale
#       adjust_gamma 는 live_demo 와 같이 blink_video_dump 것을 쓰고, 눈 번호는 face_utils.FACIAL_LANDMARKS_IDXS 의 left_eye, right_eye 이다(live_demo 613, 614행).
# 회전: blink_video_dump 와 같은 규칙(video_rotation.plan_rotation, guard_applied_angle, apply_rotation)을 프레임을 읽은 직후, 리사이즈 앞에서 적용한다.
#       좌표는 회전한 뒤 원래 크기(리사이즈 전) 기준이다. 적용한 각도를 문서에 기록하므로 뷰어가 같은 각도를 쓴다.
# 출력 문서(JSON): schema_version, kind, video_name(파일 이름만), fps, frames_read, rotation{requested, metadata_angle, applied_angle,
#       double_rotation_guard}, frame_size[너비, 높이], preprocess{width, gamma, upsample}, frames[ {face, ear, left, right} ].
#       face 가 False 이면 ear, left, right 는 null 이다. left, right 는 (x, y) 6점이다.
# 출력 위치: 저장소 밖의 새 경로만 허용한다(timeline.write_json_document 와 같은 규칙: 저장소 안과 이미 있는 파일은 거부).
# 사용: PYTHONPATH=src python -m improve.rules.replay_precompute --video <VIDEO> --predictor <PREDICTOR> --rotate auto --out <TIMELINE_DIR>/rules.json
"""재생용 프레임별 EAR 선계산(Python 3.6 호환)."""
import argparse
import os
import sys

from improve.hmlstm import timeline as TL

from . import ear_math
from . import live_demo as LD
from . import video_rotation

SCHEMA_VERSION = 1
KIND = "replay_rules"
ROTATE_CHOICES = ("0", "90", "180", "270", "auto")
UPSAMPLE = 0                                  # detector(gray, 0): live_demo 636행과 같다
EYE_POINTS = 6
PROGRESS_EVERY = 1000


class PrecomputeError(ValueError):
    """문서 형식이나 입력 인자의 오류(메시지는 한 줄로 사용자에게 보인다)."""


# ----- 프레임 한 장 처리(live_demo.run 632에서 646행과 같은 순서) -----
def process_frame(cv2, np, imutils, adjust_gamma, detector, predictor, shape_to_np, eye_indexes, frame):
    """프레임 한 장에서 {"face", "ear", "left", "right"} 를 만든다. 얼굴이 없으면 face 가 False 이고 나머지는 None.

    eye_indexes: ((왼쪽 시작, 끝), (오른쪽 시작, 끝)). 눈 좌표는 frame 의 크기 기준 정수이다.
    """
    small = imutils.resize(frame, width=LD.PREPROCESS_WIDTH)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = adjust_gamma(gray, gamma=LD.PREPROCESS_GAMMA)
    rects = detector(gray, UPSAMPLE)
    if np.size(rects) == 0:
        return {"face": False, "ear": None, "left": None, "right": None}
    shape = shape_to_np(predictor(gray, rects[0]))
    landmarks = [(int(x), int(y)) for x, y in shape]
    _left, _right, ear_value = ear_math.ear_from_landmarks(landmarks)
    scale = frame.shape[1] / float(small.shape[1])
    polygons = []
    for start, end in eye_indexes:
        polygons.append((shape[start:end] * scale).astype(np.int32).tolist())
    return {"face": True, "ear": float(ear_value), "left": polygons[0], "right": polygons[1]}


# ----- 문서 -----
def build_document(video_name, fps, frames, rotation, frame_size):
    return {"schema_version": SCHEMA_VERSION, "kind": KIND, "video_name": video_name, "fps": float(fps), "frames_read": len(frames),
            "rotation": rotation, "frame_size": [int(frame_size[0]), int(frame_size[1])],
            "preprocess": {"width": LD.PREPROCESS_WIDTH, "gamma": LD.PREPROCESS_GAMMA, "upsample": UPSAMPLE}, "frames": frames}


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _check_polygon(where, polygon):
    if not isinstance(polygon, list) or len(polygon) != EYE_POINTS:
        raise PrecomputeError("%s: 눈 윤곽은 점 %d개의 목록이어야 합니다" % (where, EYE_POINTS))
    for point in polygon:
        if not isinstance(point, list) or len(point) != 2 or any(not isinstance(v, int) or isinstance(v, bool) for v in point):
            raise PrecomputeError("%s: 점은 정수 두 개의 목록이어야 합니다: %r" % (where, point))


def validate_document(document):
    """문서 형식을 검사해 돌려준다. 어긋나면 어디가 틀렸는지 적은 PrecomputeError."""
    if not isinstance(document, dict):
        raise PrecomputeError("문서가 객체(dict)가 아닙니다")
    for key in ("schema_version", "kind", "video_name", "fps", "frames_read", "rotation", "frame_size", "preprocess", "frames"):
        if key not in document:
            raise PrecomputeError("필수 키가 없습니다: %s" % key)
    if document["schema_version"] != SCHEMA_VERSION or document["kind"] != KIND:
        raise PrecomputeError("지원하지 않는 문서입니다: schema_version %r, kind %r" % (document["schema_version"], document["kind"]))
    name = document["video_name"]
    if not isinstance(name, str) or not name or os.path.basename(name) != name or "/" in name or "\\" in name:
        raise PrecomputeError("video_name 은 경로 없는 파일 이름이어야 합니다: %r" % (name,))
    if not _is_number(document["fps"]) or document["fps"] <= 0:
        raise PrecomputeError("fps 는 0 보다 큰 수여야 합니다: %r" % (document["fps"],))
    frames = document["frames"]
    if not isinstance(frames, list) or document["frames_read"] != len(frames):
        raise PrecomputeError("frames_read(%r)가 frames 의 길이(%s)와 다릅니다" % (document["frames_read"], len(frames) if isinstance(frames, list) else "?"))
    size = document["frame_size"]
    if not isinstance(size, list) or len(size) != 2 or any(not isinstance(v, int) or isinstance(v, bool) or v <= 0 for v in size):
        raise PrecomputeError("frame_size 는 양의 정수 두 개의 목록이어야 합니다: %r" % (size,))
    rotation = document["rotation"]
    if not isinstance(rotation, dict) or rotation.get("applied_angle") not in video_rotation.VALID_ANGLES:
        raise PrecomputeError("rotation.applied_angle 은 0, 90, 180, 270 중 하나여야 합니다")
    for position, frame in enumerate(frames):
        where = "frames[%d]" % position
        if not isinstance(frame, dict) or not isinstance(frame.get("face"), bool):
            raise PrecomputeError("%s: face 는 true 또는 false 여야 합니다" % where)
        if frame["face"]:
            if not _is_number(frame.get("ear")):
                raise PrecomputeError("%s: ear 는 숫자여야 합니다" % where)
            _check_polygon(where + ".left", frame.get("left"))
            _check_polygon(where + ".right", frame.get("right"))
        elif frame.get("ear") is not None or frame.get("left") is not None or frame.get("right") is not None:
            raise PrecomputeError("%s: 얼굴이 없는 프레임의 ear, left, right 는 null 이어야 합니다" % where)
    return document


# ----- 실행 -----
def run(video, predictor_path, rotate, out_path, progress=PROGRESS_EVERY):
    """영상을 읽어 문서를 만들어 새 파일로 쓴다. 종료 코드를 돌려준다."""
    import cv2
    import dlib
    import imutils
    import numpy as np
    from imutils import face_utils
    from .blink_video_dump import adjust_gamma

    TL.check_output_path(out_path)
    if not os.path.isfile(video):
        raise PrecomputeError("영상 파일을 찾을 수 없습니다")
    if not os.path.isfile(predictor_path):
        raise PrecomputeError("랜드마크 모델 파일을 찾을 수 없습니다: --predictor 경로를 확인하세요.")
    try:
        intended, metadata = video_rotation.plan_rotation(rotate, video)
    except video_rotation.RotationError as error:
        raise PrecomputeError(str(error))
    capture = cv2.VideoCapture(video)
    if not capture.isOpened():
        raise PrecomputeError("영상을 열 수 없습니다")
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not fps > 0:
        raise PrecomputeError("영상 fps 를 읽지 못했습니다(CAP_PROP_FPS 가 0 이하이거나 없음).")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    eye_indexes = (face_utils.FACIAL_LANDMARKS_IDXS["left_eye"], face_utils.FACIAL_LANDMARKS_IDXS["right_eye"])
    applied = None
    guard = False
    frame_size = None
    frames = []
    try:
        while True:
            grabbed, frame = capture.read()
            if not grabbed:
                break
            if applied is None:
                applied, guard = video_rotation.guard_applied_angle(rotate, intended, frame.shape, capture.get(cv2.CAP_PROP_FRAME_WIDTH),
                                                                    capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame = video_rotation.apply_rotation(frame, applied)
            if frame_size is None:
                frame_size = (frame.shape[1], frame.shape[0])
            frames.append(process_frame(cv2, np, imutils, adjust_gamma, detector, predictor, face_utils.shape_to_np, eye_indexes, frame))
            if progress and len(frames) % progress == 0:
                print("frames=%d" % len(frames))
                sys.stdout.flush()
    finally:
        capture.release()
    if not frames:
        raise PrecomputeError("읽은 프레임이 없습니다")
    rotation = {"requested": str(rotate), "metadata_angle": metadata, "applied_angle": applied, "double_rotation_guard": bool(guard)}
    document = build_document(os.path.basename(video), fps, frames, rotation, frame_size)
    TL.write_json_document(out_path, document, validate_document)
    print("rows written: frames=%d fps=%.3f applied_angle=%d faces=%d" % (len(frames), fps, applied, sum(1 for f in frames if f["face"])))
    return 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="재생 뷰어용 프레임별 얼굴 상태와 EAR 를 숫자 JSON 으로 선계산한다(이미지는 저장하지 않음).")
    parser.add_argument("--video", required=True, help="영상 파일")
    parser.add_argument("--predictor", required=True, help="dlib 68점 랜드마크 모델(.dat) 경로")
    parser.add_argument("--rotate", choices=list(ROTATE_CHOICES), default="0", help="회전 각도(시계 방향), auto 는 MP4 메타데이터에서 읽음(기본 0)")
    parser.add_argument("--out", required=True, help="출력 JSON 경로(저장소 밖의 새 경로, 덮어쓰지 않음)")
    return parser.parse_args(argv)


def main(argv=None):
    arguments = parse_args(argv)
    try:
        return run(arguments.video, arguments.predictor, arguments.rotate, arguments.out)
    except (PrecomputeError, TL.TimelineError) as error:
        print("replay_precompute: %s" % error)
        return 2


if __name__ == "__main__":
    sys.exit(main())
