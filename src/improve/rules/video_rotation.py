# 영상 회전 처리(결정 I-19, PREREG_E-02.md 10번 조건 3). 표준 라이브러리 struct 와 numpy 만 쓴다(영상·이미지 라이브러리를 가져오지 않는다).
# read_rotation: MP4 박스를 seek 로 걸어 올라가 비디오 트랙의 tkhd 변환 행렬에서 각도를 읽는다(프레임 디코딩 없음, 파일 전체를 읽지 않음).
# apply_rotation: 프레임을 시계 방향으로 angle 도 회전한다.
#
# 각도 정의: 행렬은 저장 프레임의 점 (x, y)를 (a*x + c*y, b*x + d*y)로 옮긴다(y 는 아래 방향).
# 표시를 위해 저장 프레임에 시계 방향으로 적용할 회전 각도:
#   (a, b, c, d) = (1, 0, 0, 1) -> 0, (0, 1, -1, 0) -> 90, (-1, 0, 0, -1) -> 180, (0, -1, 1, 0) -> 270
# 반사(행렬식이 음수)이거나 축에 맞지 않는 행렬은 "unsupported" 이고 angle 은 None 이다.
# 방향은 코드만으로 확정되지 않으며, 실제 영상에서는 얼굴 검출 일치 확인(IMPROVEMENT_LOG.md)이 판정 근거이다.
import os
import struct

import numpy as np

STATUS_OK = "ok"
STATUS_UNSUPPORTED = "unsupported"
STATUS_NO_VIDEO_TRACK = "no_video_track"

VALID_ANGLES = (0, 90, 180, 270)
MATRIX_TOLERANCE = 0.01          # 16.16 고정소수점 행렬 성분을 0, 1, -1 과 비교하는 허용 오차
_ANGLE_BY_MATRIX = {(1, 0, 0, 1): 0, (0, 1, -1, 0): 90, (-1, 0, 0, -1): 180, (0, -1, 1, 0): 270}


class _Malformed(Exception):
    """박스 크기가 헤더보다 작거나 상위 박스(또는 파일) 밖으로 나간다."""


def _result(status, angle=None, matrix=None, width=None, height=None):
    return {"angle": angle, "matrix": matrix, "tkhd_width": width, "tkhd_height": height, "status": status}


def _read_exact(handle, count):
    data = handle.read(count)
    if len(data) != count:
        raise _Malformed("파일이 잘렸습니다.")
    return data


def _iter_boxes(handle, start, end):
    """[start, end) 안의 박스를 차례로 낸다: (type, payload_start, box_end). 박스 안으로 들어가지 않고 건너뛴다."""
    position = start
    while position < end:
        if end - position < 8:
            raise _Malformed("박스 헤더가 잘렸습니다.")
        handle.seek(position)
        size, box_type = struct.unpack(">I4s", _read_exact(handle, 8))
        header = 8
        if size == 1:
            if end - position < 16:
                raise _Malformed("largesize 헤더가 잘렸습니다.")
            size = struct.unpack(">Q", _read_exact(handle, 8))[0]
            header = 16
        elif size == 0:
            size = end - position
        if size < header or position + size > end:
            raise _Malformed("박스 크기가 범위를 벗어납니다.")
        yield box_type, position + header, position + size
        position += size


def _handler_type(handle, trak_payload, trak_end):
    """trak 안의 mdia/hdlr 의 handler_type 을 반환한다(없으면 None)."""
    for box_type, payload, box_end in _iter_boxes(handle, trak_payload, trak_end):
        if box_type != b"mdia":
            continue
        for inner_type, inner_payload, inner_end in _iter_boxes(handle, payload, box_end):
            if inner_type == b"hdlr":
                if inner_end - inner_payload < 12:
                    raise _Malformed("hdlr 가 너무 짧습니다.")
                handle.seek(inner_payload + 8)
                return _read_exact(handle, 4)
    return None


def _read_tkhd(handle, trak_payload, trak_end):
    """trak 안의 tkhd 에서 (a, b, c, d, width, height)를 반환한다(없으면 None). 행렬은 16.16 이므로 실수로 바꾼다."""
    for box_type, payload, box_end in _iter_boxes(handle, trak_payload, trak_end):
        if box_type != b"tkhd":
            continue
        handle.seek(payload)
        version = _read_exact(handle, 1)[0]
        if version == 0:
            matrix_offset = 40
        elif version == 1:
            matrix_offset = 52
        else:
            raise _Malformed("지원하지 않는 tkhd 버전입니다.")
        if box_end - payload < matrix_offset + 36 + 8:
            raise _Malformed("tkhd 가 너무 짧습니다.")
        handle.seek(payload + matrix_offset)
        values = struct.unpack(">9i", _read_exact(handle, 36))      # a b u c d v x y w
        width, height = struct.unpack(">II", _read_exact(handle, 8))
        a, b, c, d = values[0] / 65536.0, values[1] / 65536.0, values[3] / 65536.0, values[4] / 65536.0
        return a, b, c, d, width / 65536.0, height / 65536.0
    return None


def _angle_from_matrix(a, b, c, d):
    """표의 네 행렬 중 허용 오차 안에서 같은 것의 각도. 반사나 축에 맞지 않는 행렬이면 None."""
    for key, angle in _ANGLE_BY_MATRIX.items():
        if all(abs(value - expected) <= MATRIX_TOLERANCE for value, expected in zip((a, b, c, d), key)):
            return angle
    return None


def read_rotation(path):
    """MP4 의 비디오 트랙(handler_type 'vide') tkhd 에서 회전 정보를 읽는다. 파일 바이트만 읽는다.

    반환: {"angle": 0/90/180/270 또는 None, "matrix": (a, b, c, d) 또는 None, "tkhd_width", "tkhd_height",
           "status": "ok" / "unsupported" / "no_video_track"}
    unsupported: 반사나 축에 맞지 않는 행렬, 박스 구조 손상(크기 오류, 잘림), 지원하지 않는 tkhd 버전.
    no_video_track: moov 가 없거나 비디오 트랙이 없음.
    """
    try:
        file_size = os.path.getsize(path)
        with open(path, "rb") as handle:
            for box_type, payload, box_end in _iter_boxes(handle, 0, file_size):
                if box_type != b"moov":
                    continue
                for trak_type, trak_payload, trak_end in _iter_boxes(handle, payload, box_end):
                    if trak_type != b"trak" or _handler_type(handle, trak_payload, trak_end) != b"vide":
                        continue
                    tkhd = _read_tkhd(handle, trak_payload, trak_end)
                    if tkhd is None:
                        return _result(STATUS_UNSUPPORTED)
                    a, b, c, d, width, height = tkhd
                    angle = _angle_from_matrix(a, b, c, d)
                    if angle is None:
                        return _result(STATUS_UNSUPPORTED, None, (a, b, c, d), width, height)
                    return _result(STATUS_OK, angle, (a, b, c, d), width, height)
                return _result(STATUS_NO_VIDEO_TRACK)
        return _result(STATUS_NO_VIDEO_TRACK)
    except (_Malformed, struct.error):
        return _result(STATUS_UNSUPPORTED)


class RotationError(Exception):
    """회전 요청을 처리할 수 없다(추출을 중단하고 사용자에게 보고해야 한다)."""


def plan_rotation(requested, path):
    """요청값("0", "90", "180", "270", "auto")에서 (의도한 각도, 메타데이터 각도)를 정한다.

    auto 는 파일 메타데이터를 읽는다. status 가 ok 가 아니거나 angle 이 None 이면, 또는 angle 이 180 이면 RotationError
    (180 도는 프레임 모양으로 이중 회전을 판별할 수 없다). 숫자 요청은 파일을 읽지 않으며 메타데이터 각도는 None 이다.
    """
    requested = str(requested)
    if requested == "auto":
        info = read_rotation(path)
        if info["status"] != STATUS_OK or info["angle"] is None:
            raise RotationError("회전 각도를 메타데이터에서 읽지 못했습니다(status=%s). 추출을 중단합니다." % info["status"])
        if info["angle"] == 180:
            raise RotationError("메타데이터 각도가 180도입니다. 프레임 모양으로 이중 회전을 판별할 수 없어 중단합니다.")
        return info["angle"], info["angle"]
    if requested in ("0", "90", "180", "270"):
        return int(requested), None
    raise RotationError("알 수 없는 회전 요청값입니다: %r" % requested)


def guard_applied_angle(requested, intended, frame_shape, cap_width, cap_height):
    """이중 회전 보호. 반환: (적용할 각도, 보호 작동 여부).

    요청이 auto 이고 각도가 90/270 일 때, 첫 프레임이 이미 (높이, 너비) = (CAP 너비, CAP 높이)로 뒤집힌 모양이면
    OpenCV 가 자동 회전한 것으로 보고 적용 각도를 0 으로 한다. 그 밖에는 의도한 각도를 그대로 쓴다.
    """
    if str(requested) != "auto" or intended not in (90, 270):
        return intended, False
    height, width = int(frame_shape[0]), int(frame_shape[1])
    if cap_width > 0 and cap_height > 0 and int(round(cap_width)) != int(round(cap_height)):
        if height == int(round(cap_width)) and width == int(round(cap_height)):
            return 0, True
    return intended, False


def _shape_text(shape):
    return "x".join(str(int(value)) for value in shape)


def rotation_line(requested, metadata_angle, applied_angle, shape_before, shape_after, guard_active):
    """rotation.txt 에 쓸 한 줄(줄바꿈 포함)."""
    return "requested=%s metadata_angle=%s applied_angle=%d shape_before=%s shape_after=%s double_rotation_guard=%s\n" % (
        requested, "not_read" if metadata_angle is None else metadata_angle, applied_angle,
        _shape_text(shape_before), _shape_text(shape_after), "yes" if guard_active else "no")


def apply_rotation(frame, angle):
    """프레임을 시계 방향으로 angle 도 회전한다. angle 0 이면 입력 객체를 그대로 반환한다(복사 없음).

    90/270 이면 앞 두 축(높이, 너비)이 서로 바뀐다. 결과는 연속 배열이다. 0/90/180/270 이외의 값은 ValueError.
    """
    if isinstance(angle, bool) or angle not in VALID_ANGLES:
        raise ValueError("angle 은 0, 90, 180, 270 중 하나여야 합니다: %r" % (angle,))
    angle = int(angle)
    if angle == 0:
        return frame
    return np.ascontiguousarray(np.rot90(frame, k=-(angle // 90)))
