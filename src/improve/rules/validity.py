# 역할: 유효성 검사(I-09). 알림 기준 구간(클립 1)의 EAR 중앙값이 임계값 이하이면 "탐지 불가능"을 돌려준다.
#       근거는 docs/improvement/RULE_SPEC.md 7절이다. 이 판단이 맞는지는 검증하지 않았다(미검증).
# 입력: 기준 구간의 (timestamp_s, ear_mean 또는 None, source) 시퀀스. 얼굴이 검출된 프레임(source 가 detected 이고
#       ear_mean 이 있는 프레임)의 EAR 만 쓴다.
# 출력: dict(status, median, n_detected_frames, n_frames, span_s). 파일을 읽거나 쓰지 않는다.
# 기준 구간의 최소 길이(min_duration_s)는 기본값이 없다. 호출자가 지정해야 한다(RULE_SPEC 의 제안 5분은 미확정, I-W-10).
import math

STATUS_OK = "정상"
STATUS_UNDETECTABLE = "탐지 불가능"
STATUS_SHORT = "기준 구간 부족"

# 알람의 reliability 필드 값은 제안이다(I-W-12, 미확정).
RELIABILITY_OK = "정상"
RELIABILITY_UNRELIABLE = "신뢰 불가"
RELIABILITY_UNKNOWN = "미확정"


def median(values):
    """값 목록의 중앙값. 짝수 개이면 가운데 두 값의 평균이다.

    입력: values - 비어 있지 않은 수 목록
    반환: float
    """
    ordered = sorted(values)
    if not ordered:
        raise ValueError("중앙값을 구할 값이 없습니다.")
    middle = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return float(ordered[middle])
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def check_reference(samples, threshold, min_duration_s):
    """기준 구간의 얼굴 검출 프레임 EAR 중앙값이 임계값 이하인지 본다.

    입력: samples - (timestamp_s, ear_mean 또는 None, source) 시퀀스. threshold - 닫힘 임계값(이하이면 탐지 불가능).
          min_duration_s - 기준 구간의 최소 길이(초). 기본값 없음.
    반환: dict. status 는 "정상", "탐지 불가능", "기준 구간 부족" 중 하나이다. 기준 구간이 min_duration_s 보다 짧거나
          검출된 프레임이 없으면 "기준 구간 부족"이다(이 경우 median 은 None).
    """
    if isinstance(min_duration_s, bool) or not isinstance(min_duration_s, (int, float)) or not math.isfinite(min_duration_s) \
            or min_duration_s <= 0:
        raise ValueError("min_duration_s 는 0 보다 큰 유한한 수여야 합니다(기본값 없음).")
    samples = list(samples)
    n_frames = len(samples)
    span = (samples[-1][0] - samples[0][0]) if samples else 0.0
    detected = [ear for _ts, ear, source in samples if source == "detected" and ear is not None]
    info = {"status": STATUS_SHORT, "median": None, "n_detected_frames": len(detected), "n_frames": n_frames,
            "span_s": span, "threshold": threshold, "min_duration_s": min_duration_s}
    if not detected or span < min_duration_s:
        return info
    info["median"] = median(detected)
    info["status"] = STATUS_UNDETECTABLE if info["median"] <= threshold else STATUS_OK
    return info


def reliability_for(check):
    """유효성 검사 결과로 알람의 reliability 값을 정한다(제안, I-W-12).

    입력: check - check_reference 의 반환 dict
    반환: "정상" / "신뢰 불가" / "미확정"
    """
    if check["status"] == STATUS_UNDETECTABLE:
        return RELIABILITY_UNRELIABLE
    if check["status"] == STATUS_OK:
        return RELIABILITY_OK
    return RELIABILITY_UNKNOWN
