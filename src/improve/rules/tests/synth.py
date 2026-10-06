# 합성 샘플 생성 도우미(시험 전용). 프레임 번호 ÷ fps 로 타임스탬프를 만든다.
# 실제 영상, 웹캠, 실제 데이터 파일을 읽지 않는다.
FPS = 30.0
CLOSED = 0.10   # 임계값 0.21 보다 작은 값(닫힘)
OPEN = 0.30     # 임계값 0.21 보다 큰 값(열림)


def seg(count, ear, source="detected"):
    """같은 값이 count 프레임 이어지는 구간 하나."""
    return (count, ear, source)


def miss(count):
    """얼굴 없음(EAR 없음, source none)이 count 프레임 이어지는 구간."""
    return (count, None, "none")


def seq(spec, fps=FPS):
    """구간 목록(spec)을 (timestamp_s, ear, source) 샘플 목록으로 펼친다. 타임스탬프 = 프레임 번호 / fps."""
    samples = []
    index = 0
    for count, ear, source in spec:
        for _ in range(count):
            samples.append((index / fps, ear, source))
            index += 1
    return samples


def at(frame, fps=FPS):
    """프레임 번호의 타임스탬프."""
    return frame / fps
