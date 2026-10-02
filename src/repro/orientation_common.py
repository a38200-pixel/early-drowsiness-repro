# orientation_common.py
# 역할: 회전 방향 이름과 회전 함수의 정의를 한 곳에 둔다. 조사(survey_orientation.py), 무화면 실행
#       (reproduction_blink_headless.py 의 --rotate), 일괄 추출(extract_blinks_batch.py)이 같은 정의를 써서
#       방향의 의미가 어긋나지 않게 한다.
# 사용법: 같은 폴더의 스크립트에서 `from orientation_common import ...` 로 가져다 쓴다. 단독 실행하지 않는다.
"""회전 방향 정의(Python 3.6 호환).

방향 이름: "0"(회전 없음), "cw90"(시계 방향 90도), "180", "ccw90"(반시계 방향 90도).
--rotate 옵션 이름: "none"(= "0"), "cw90", "ccw90", "180".
방향은 "프레임을 이 방향으로 돌린 뒤 얼굴을 찾는다/blink 를 추출한다"는 뜻이다.
영상의 회전 태그를 읽은 값이 아니다.
"""

SURVEY_ORIENTATIONS = ("0", "cw90", "180", "ccw90")
ROTATE_OPTIONS = ("none", "cw90", "ccw90", "180")
_CANONICAL = {"0": "0", "none": "0", "cw90": "cw90", "180": "180", "ccw90": "ccw90"}
_CONSTANT_NAMES = {"cw90": "ROTATE_90_CLOCKWISE", "180": "ROTATE_180", "ccw90": "ROTATE_90_COUNTERCLOCKWISE"}


def normalize_orientation(name):
    """방향 이름이나 --rotate 옵션 이름을 조사에서 쓰는 표준 이름("0", "cw90", "180", "ccw90")으로 바꾼다.

    입력: name - "0", "none", "cw90", "ccw90", "180" 중 하나
    반환: 표준 이름 문자열. 알 수 없는 이름이면 None
    """
    return _CANONICAL.get(name)


def to_rotate_option(orientation):
    """표준 방향 이름을 --rotate 옵션 이름으로 바꾼다("0"은 "none").

    입력: orientation - 방향 이름(옵션 이름도 허용)
    반환: --rotate 옵션 이름. 알 수 없는 이름이면 ValueError
    """
    canonical = normalize_orientation(orientation)
    if canonical is None:
        raise ValueError("알 수 없는 방향 이름: %s" % orientation)
    return "none" if canonical == "0" else canonical


def rotation_expression(option):
    """blink_video.py 의 프레임 변수 `frame` 에 적용할 회전 표현식 문자열을 만든다.

    입력: option - --rotate 옵션 이름(방향 이름도 허용)
    반환: 회전 없음이면 "frame", 아니면 "cv2.rotate(frame, cv2.ROTATE_...)" 문자열
    """
    canonical = normalize_orientation(option)
    if canonical is None:
        raise ValueError("알 수 없는 방향 이름: %s" % option)
    if canonical == "0":
        return "frame"
    return "cv2.rotate(frame, cv2.%s)" % _CONSTANT_NAMES[canonical]


def rotate_frame(frame, orientation):
    """프레임을 해당 방향으로 돌려 돌려준다. 회전 없음이면 원본 프레임을 그대로 돌려준다.

    입력: frame - BGR 프레임 배열, orientation - 방향 이름(옵션 이름도 허용)
    반환: 회전된 프레임. 알 수 없는 이름이면 ValueError
    """
    import cv2   # 표현식 문자열만 필요한 호출자가 cv2 없이도 이 모듈을 가져올 수 있게 늦게 가져온다
    canonical = normalize_orientation(orientation)
    if canonical is None:
        raise ValueError("알 수 없는 방향 이름: %s" % orientation)
    if canonical == "0":
        return frame
    return cv2.rotate(frame, getattr(cv2, _CONSTANT_NAMES[canonical]))
