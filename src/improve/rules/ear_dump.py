# 역할: C1b 영상 추출기(blink_video_dump.py)가 프레임별 EAR 덤프 CSV 를 쓰는 순수 함수와 작은 클래스.
# 형식: 열 timestamp_s, ear_left, ear_right, source. 프레임마다 정확히 한 행이다. docs/improvement/RULE_SPEC.md 7.1절의 source 열
#       (detected / tracked / none)과 replay.py 의 read_ear_csv 가 읽는 형식과 같다.
#   - timestamp_s = 프레임 번호 / fps (소수 6자리). 프레임 번호는 읽은 모든 프레임 기준이다(0부터).
#   - source 가 none 이면 ear 두 칸은 빈 문자열이다. 그 밖에는 소수 6자리이다.
# 파일 경로는 호출자가 인자로 준다(저장소 안의 기본값 없음). 영상, 프레임, 이미지를 읽거나 저장하지 않는다.
import csv
import math

DUMP_COLUMNS = ("timestamp_s", "ear_left", "ear_right", "source")
SOURCES = ("detected", "tracked", "none")


def _check_number(name, value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("%s 는 유한한 수여야 합니다: %r" % (name, value))


def format_row(frame_index, fps, ear_left, ear_right, source):
    """덤프 CSV 한 행의 칸 4개(문자열)를 만든다.

    입력: frame_index - 읽은 프레임 번호(0 이상 정수). fps - 영상 fps(0 보다 커야 함). ear_left, ear_right - 눈별 EAR
          (source 가 none 이면 무시). source - detected / tracked / none
    반환: (timestamp_s, ear_left, ear_right, source) 문자열 튜플. 시각과 EAR 은 소수 6자리, none 의 EAR 칸은 빈 문자열
    """
    if source not in SOURCES:
        raise ValueError("source 는 %s 중 하나여야 합니다: %r" % (", ".join(SOURCES), source))
    _check_number("fps", fps)
    if fps <= 0:
        raise ValueError("fps 는 0 보다 커야 합니다: %r" % (fps,))
    if isinstance(frame_index, bool) or not isinstance(frame_index, int) or frame_index < 0:
        raise ValueError("frame_index 는 0 이상의 정수여야 합니다: %r" % (frame_index,))
    timestamp = "%.6f" % (frame_index / float(fps))
    if source == "none":
        return (timestamp, "", "", source)
    if ear_left is None or ear_right is None:
        raise ValueError("source 가 %s 이면 ear_left 와 ear_right 가 있어야 합니다." % source)
    _check_number("ear_left", ear_left)
    _check_number("ear_right", ear_right)
    return (timestamp, "%.6f" % ear_left, "%.6f" % ear_right, source)


class DumpWriter(object):
    """덤프 CSV 를 쓰는 컨텍스트 매니저. 헤더 한 줄을 쓰고 행을 이어 쓰며 닫을 때 flush 한다."""

    def __init__(self, out_path):
        """출력 파일 경로를 받는다(필수). 이미 있는 파일은 덮어쓰지 않고 FileExistsError 를 낸다.

        입력: out_path - 덤프 CSV 경로
        """
        if not out_path:
            raise ValueError("출력 경로가 필요합니다(기본값 없음).")
        self.out_path = out_path
        self._handle = None
        self._writer = None
        self.rows_written = 0

    def __enter__(self):
        self._handle = open(self.out_path, "x", encoding="utf-8", newline="")
        self._writer = csv.writer(self._handle)
        self._writer.writerow(DUMP_COLUMNS)
        return self

    def write_row(self, frame_index, fps, ear_left, ear_right, source):
        """행 하나를 쓴다. 인자 규칙은 format_row 와 같다."""
        if self._writer is None:
            raise RuntimeError("DumpWriter 는 with 블록 안에서 써야 합니다.")
        self._writer.writerow(format_row(frame_index, fps, ear_left, ear_right, source))
        self.rows_written += 1

    def close(self):
        """flush 하고 파일을 닫는다(여러 번 불러도 된다)."""
        if self._handle is not None:
            self._handle.flush()
            self._handle.close()
            self._handle = None
            self._writer = None

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False
