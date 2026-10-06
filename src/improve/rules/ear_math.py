# 역할: 저자 blink_video.py 의 EAR(눈 가로세로 비율) 계산을 순수 함수로 다시 쓴다(C1 순수 함수부).
#       저자 파일은 import 하지도 수정하지도 않는다. 근거 줄 번호는 third_party/ghoddoosian/blink_video.py 이다.
# 입력: 눈 랜드마크 6개((x, y) 좌표 목록) 또는 68점 랜드마크 전체.
# 출력: EAR 값(float). 파일을 읽거나 쓰지 않는다.
# 주의: EAR 은 사람 눈의 감김 비율이 아니다(docs/improvement/RULE_SPEC.md 3절).
import math

# 눈 랜드마크 구간(저자 코드 318-319줄이 imutils 의 face_utils.FACIAL_LANDMARKS_IDXS 에서 읽는 값).
# 값은 이 환경의 imutils 0.5.4 에서 읽은 것이다(왼쪽 눈 42~47, 오른쪽 눈 36~41, 끝은 포함하지 않음).
LEFT_EYE_INDEX = (42, 48)
RIGHT_EYE_INDEX = (36, 42)

SMALL_WIDTH = 0.1        # 저자 84줄: C < 0.1 이면 수치 문제 대비로 EAR 을 고정한다
SMALL_WIDTH_EAR = 0.3    # 저자 85줄
EAR_CAP = 0.45           # 저자 89-90줄: EAR > 0.45 이면 0.45 로 제한한다


def _distance(first, second):
    """두 점 사이의 유클리드 거리. 저자 코드의 scipy dist.euclidean 과 같은 값을 낸다.

    입력: first, second - (x, y) 좌표
    반환: float
    """
    dx = float(first[0]) - float(second[0])
    dy = float(first[1]) - float(second[1])
    return math.sqrt(dx * dx + dy * dy)


def eye_aspect_ratio(eye):
    """눈 하나의 EAR 을 저자 74-92줄과 같은 순서로 계산한다.

    입력: eye - 랜드마크 6개 ((x, y) 목록). eye[0]~eye[5] 는 저자 코드의 인덱스와 같은 의미이다.
    반환: float. 수식 (A + B) / (2 * C) (A=eye[1]~eye[5], B=eye[2]~eye[4], C=eye[0]~eye[3], 저자 77-78, 82, 88줄),
          C < 0.1 이면 0.3(84-85줄), 결과가 0.45 보다 크면 0.45(89-90줄).
    """
    if len(eye) != 6:
        raise ValueError("눈 랜드마크는 6개여야 합니다(받은 개수 %d)." % len(eye))
    vertical_a = _distance(eye[1], eye[5])
    vertical_b = _distance(eye[2], eye[4])
    horizontal = _distance(eye[0], eye[3])
    if horizontal < SMALL_WIDTH:
        ear = SMALL_WIDTH_EAR
    else:
        ear = (vertical_a + vertical_b) / (2.0 * horizontal)
    if ear > EAR_CAP:
        ear = EAR_CAP
    return ear


def mean_ear(left_ear, right_ear):
    """양쪽 눈 EAR 의 평균(저자 421줄).

    입력: left_ear, right_ear - 눈별 EAR
    반환: float
    """
    return (left_ear + right_ear) / 2.0


def eyes_from_landmarks(landmarks):
    """68점 랜드마크에서 왼쪽 눈과 오른쪽 눈 6점씩을 잘라 낸다(저자 415-416줄).

    입력: landmarks - 68개 이상의 (x, y) 목록
    반환: (왼쪽 눈 6점, 오른쪽 눈 6점)
    """
    if len(landmarks) < LEFT_EYE_INDEX[1]:
        raise ValueError("랜드마크가 %d개 이상 있어야 합니다(받은 개수 %d)." % (LEFT_EYE_INDEX[1], len(landmarks)))
    left_eye = landmarks[LEFT_EYE_INDEX[0]:LEFT_EYE_INDEX[1]]
    right_eye = landmarks[RIGHT_EYE_INDEX[0]:RIGHT_EYE_INDEX[1]]
    return left_eye, right_eye


def ear_from_landmarks(landmarks):
    """68점 랜드마크에서 (왼쪽 EAR, 오른쪽 EAR, 평균 EAR)을 계산한다.

    입력: landmarks - 68개 이상의 (x, y) 목록
    반환: (float, float, float)
    """
    left_eye, right_eye = eyes_from_landmarks(landmarks)
    left_ear = eye_aspect_ratio(left_eye)
    right_ear = eye_aspect_ratio(right_eye)
    return left_ear, right_ear, mean_ear(left_ear, right_ear)
