# reproduction_blink_headless.py
# 역할: blink_video.py 를 수정하지 않고, 텍스트로 읽어 메모리에서만 GUI 호출(draw, imshow, waitKey)을 무력화한 뒤
#       영상 1개에 실행한다. 실행이 끝나면 소요 시간과 출력 파일의 blink 행 수를 출력한다.
#       --compare 서브모드로 두 출력 파일을 줄 단위로 비교할 수도 있다(영상은 읽지 않는다).
# 사용법: conda 환경 ghoddoosian_tf1 을 활성화하고 저장소 루트에서 실행한다.
#           python src/repro/reproduction_blink_headless.py <영상경로> [출력txt] [--rotate none|cw90|ccw90|180]
#           python src/repro/reproduction_blink_headless.py <영상경로> --dry-run [--rotate ...]  (치환만 확인, 영상은 열지 않음)
#           python src/repro/reproduction_blink_headless.py --compare <a.txt> <b.txt>
# --rotate: 원본 프레임을 먼저 회전한 뒤 같은 resize(width=450)를 한다(조사 스크립트와 같은 순서). 방향 이름과 회전 함수는
#           orientation_common.py 의 정의를 조사 스크립트와 함께 쓴다. none 이면 이전 버전과 같은 소스를 실행한다.
# 환경변수: DLIB_LANDMARK_PATH (영상 실행 시 필수), RLDD_DATA_DIR (영상경로가 상대경로일 때 필수)
"""blink_video.py 무화면 실행 래퍼와 출력 파일 비교 도구(Python 3.6 호환).

사용법:  python reproduction_blink_headless.py <영상경로> [출력txt] [--rotate none|cw90|ccw90|180] [--dry-run]
         python reproduction_blink_headless.py --compare <a.txt> <b.txt>
영상경로가 상대경로이면 RLDD_DATA_DIR 아래에서 찾는다. 출력txt를 생략하면
experiments/00_reproduction/blink_test/<영상이름>_blinks.txt 에 쓴다(기존 래퍼와 같은 규칙).
이미 있는 출력 파일에는 이어 쓰지 않고 종료한다.
원본이 열 때 쓰는 Tk 창과 figure 생성부는 그대로 두고, 아래 세 가지 표현식만 메모리에서 바꾼다.
  매 프레임 plot_frame.draw() -> None,  cv2.imshow(...) -> None,  cv2.waitKey(1) -> 255
waitKey 가 255 로 고정되므로 키 입력으로 중간에 멈출 수 없고 영상 끝까지 처리한다.
--rotate 가 none 이 아니면 `frame = imutils.resize(frame, width=450)` 한 곳(정확히 1회 일치해야 하며 아니면 중단)을
"원본 프레임을 회전한 뒤 같은 resize" 로 바꾼다. 이 방향은 회전 태그를 적용하는 디코더가 주는 프레임과 같을 것으로 추측하며,
영상의 회전 태그는 읽지 않는다. --dry-run 은 치환 일치 횟수와 치환된 줄 번호만 출력하고 영상은 열지 않고 끝낸다.
"""
import argparse
import os
import sys
import time

from orientation_common import ROTATE_OPTIONS, rotation_expression

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.join(REPO, "third_party", "ghoddoosian")
DRAW_INDENT = " " * 16   # 매 프레임 draw 호출의 들여쓰기. Tk 생성부의 draw 는 4칸이라 이 문자열과 구분된다
RESIZE_TARGET = "frame = imutils.resize(frame, width=450)"   # blink_video.py 의 프레임 리사이즈(회전을 끼워 넣을 한 곳)


def get_dlib_path():
    """dlib 랜드마크 모델(.dat) 경로를 환경변수에서 읽고 파일이 있는지 확인한다.

    반환: 파일 경로 문자열. 설정이 없거나 파일이 없으면 안내 메시지와 함께 종료한다
    """
    dlib_path = os.environ.get("DLIB_LANDMARK_PATH")
    if not dlib_path or not os.path.isfile(dlib_path):
        sys.exit("환경변수 DLIB_LANDMARK_PATH 를 shape_predictor_68_face_landmarks.dat 로 설정하세요 (data/README.md 참고).")
    return dlib_path


def resolve_video_path(video_argument):
    """영상 경로를 절대경로로 바꾼다. 상대경로이면 RLDD_DATA_DIR 아래에서 찾는다.

    입력: video_argument - 명령줄로 받은 영상 경로
    반환: 존재하는 영상 파일의 절대경로. 없으면 메시지와 함께 종료한다
    """
    if not os.path.isabs(video_argument):
        data_dir = os.environ.get("RLDD_DATA_DIR")
        if not data_dir:
            sys.exit("환경변수 RLDD_DATA_DIR 를 UTA-RLDD 영상 폴더로 설정하세요 (또는 영상의 절대경로를 넘기세요).")
        video_argument = os.path.join(data_dir, video_argument)
    video = os.path.abspath(video_argument)
    if not os.path.isfile(video):
        sys.exit("영상을 찾을 수 없습니다: %s" % video)
    return video


def resolve_output_path(video, output_argument):
    """출력 txt 경로를 정한다. 작업 폴더를 바꾸기 전에 절대경로로 만들어 third_party 안으로 새지 않게 한다.

    입력: video - 영상 절대경로, output_argument - 명령줄의 출력 경로 또는 None
    반환: 절대경로. 이미 파일이 있으면 이어 쓰지 않고 종료한다(폴더는 아직 만들지 않는다)
    """
    if output_argument is not None:
        output = os.path.abspath(output_argument)
    else:
        file_name = os.path.splitext(os.path.basename(video))[0] + "_blinks.txt"
        output = os.path.join(REPO, "experiments", "00_reproduction", "blink_test", file_name)
    if os.path.exists(output):
        sys.exit("이미 있는 파일에는 이어 쓰지 않습니다: %s" % output)
    return output


def build_replacements(output, video, dlib_path, rotate="none"):
    """메모리에서 바꿀 문자열 목록을 만든다. 각 항목은 (설명, 원문, 새 문장, 기대 일치 횟수).

    입력: output - 출력 txt 경로, video - 영상 경로, dlib_path - .dat 경로, rotate - --rotate 옵션 이름
    반환: 튜플 list. 줄을 지우지 않고 표현식만 바꾼다. rotate 가 none 이면 이전 버전과 같은 6개 항목이고,
          그 외에는 프레임 회전 후 resize 하는 치환 1개(정확히 1회 일치해야 함)가 더해진다
    """
    replacements = [
        ("출력 파일 경로", "output_file = 'alert.txt'", "output_file = %r" % output, 1),
        ("입력 영상 경로", "path = ''", "path = %r" % video, 1),
        ("dlib .dat 경로", "'shape_predictor_68_face_landmarks.dat'", repr(dlib_path), 1),
        ("매 프레임 plot_frame.draw()", DRAW_INDENT + "plot_frame.draw()", DRAW_INDENT + "None", 2),
        ("cv2.imshow", 'cv2.imshow("Frame", frameMinus7)', "None", 2),
        ("cv2.waitKey(1)", "cv2.waitKey(1)", "255", 2),
    ]
    if rotate != "none":
        rotated = "frame = imutils.resize(%s, width=450)" % rotation_expression(rotate)
        replacements.append(("원본 프레임 회전 후 resize", RESIZE_TARGET, rotated, 1))
    return replacements


def check_counts(source, replacements):
    """원본에서 모든 치환 대상의 일치 횟수를 먼저 확인한다. 하나라도 다르면 영상을 처리하기 전에 중단한다.

    입력: source - blink_video.py 원문, replacements - build_replacements 결과
    개수가 기대와 다르면 어느 항목이 몇 번 나왔는지 알리고 종료한다(단언을 완화하지 않는다)
    """
    problems = []
    for label, old, _new, expected in replacements:
        actual = source.count(old)
        print("치환 대상 [%s]: 기대 %d회, 실제 %d회" % (label, expected, actual))
        if actual != expected:
            problems.append("[%s] 기대 %d회, 실제 %d회" % (label, expected, actual))
    if problems:
        sys.exit("치환 대상 개수가 달라 중단합니다: " + "; ".join(problems))


def apply_replacements(source, replacements):
    """check_counts 를 통과한 뒤 치환을 적용한다.

    입력: source - blink_video.py 원문, replacements - build_replacements 결과
    반환: 치환된 소스 문자열(줄 수는 그대로)
    """
    for _label, old, new, _expected in replacements:
        source = source.replace(old, new)
    return source


def changed_line_numbers(original, modified):
    """치환 전후에서 내용이 달라진 줄 번호를 모은다. 줄 수가 달라졌으면 줄이 지워진 것이므로 종료한다.

    입력: original - 원문, modified - 치환된 소스
    반환: 1부터 시작하는 줄 번호 list
    """
    original_lines = original.split("\n")
    modified_lines = modified.split("\n")
    if len(original_lines) != len(modified_lines):
        sys.exit("치환 후 줄 수가 달라졌습니다(줄을 지우면 안 됩니다): %d -> %d" % (len(original_lines), len(modified_lines)))
    numbers = []
    for index in range(len(original_lines)):
        if original_lines[index] != modified_lines[index]:
            numbers.append(index + 1)
    return numbers


def read_original_source():
    """third_party 의 blink_video.py 를 텍스트로 읽는다(파일은 수정하지 않는다).

    반환: 소스 문자열
    """
    with open(os.path.join(ROOT, "blink_video.py"), "r", encoding="utf-8") as file:
        return file.read()


def count_blink_rows(output):
    """출력 파일의 blink 행 수를 센다. 첫 빈 줄은 제외하고 내용이 있는 줄만 센다.

    입력: output - 출력 txt 경로
    반환: 행 수(파일이 없으면 0)
    """
    if not os.path.exists(output):
        return 0
    with open(output, "r", encoding="utf-8", errors="replace") as file:
        lines = file.read().split("\n")
    if lines and lines[0].strip() == "":
        lines = lines[1:]
    rows = 0
    for line in lines:
        if line.strip() != "":
            rows += 1
    return rows


def run_source(source):
    """치환된 소스를 메모리에서 컴파일해 실행한다(원본 하단의 main 부분이 그대로 실행된다).

    입력: source - apply_replacements 결과
    """
    exec(compile(source, "blink_video.py[headless-in-memory]", "exec"), {"__name__": "blink_video_inmemory"})


def prepare_runtime():
    """원본이 상대경로로 .sav 를 열고 같은 폴더의 모듈을 쓰므로 작업 폴더와 모듈 경로를 맞춘다.

    바이트코드 파일(__pycache__)은 만들지 않는다.
    """
    os.chdir(ROOT)
    sys.path.insert(0, ROOT)
    sys.dont_write_bytecode = True


def parse_video_arguments(arguments):
    """영상 실행 모드의 인자를 해석한다.

    입력: arguments - sys.argv[1:] (영상경로, 선택적 출력경로, --rotate, --dry-run)
    반환: argparse.Namespace (video, output, rotate, dry_run)
    """
    parser = argparse.ArgumentParser(
        prog="reproduction_blink_headless.py",
        usage="%(prog)s <영상경로> [출력txt] [--rotate none|cw90|ccw90|180] [--dry-run]")
    parser.add_argument("video", nargs="?", default=None)
    parser.add_argument("output", nargs="?", default=None)
    parser.add_argument("--rotate", choices=list(ROTATE_OPTIONS), default="none")
    parser.add_argument("--dry-run", action="store_true", dest="dry_run")
    return parser.parse_args(arguments)


def run_dry_run(rotate):
    """--dry-run: 치환 일치 횟수와 치환된 줄 번호만 출력하고 끝낸다. 영상은 열지 않고 경로·환경변수도 검사하지 않는다.

    입력: rotate - --rotate 옵션 이름
    """
    original = read_original_source()
    replacements = build_replacements("OUTPUT.txt", "VIDEO", "DLIB.dat", rotate)
    print("회전: %s (--dry-run: 영상을 열지 않음)" % rotate)
    check_counts(original, replacements)
    modified = apply_replacements(original, replacements)
    print("치환된 줄 번호: %s" % changed_line_numbers(original, modified))


def run_video_mode(arguments):
    """영상 1개를 무화면으로 실행하고 소요 시간과 blink 행 수를 출력한다. --dry-run 이면 치환만 확인하고 끝낸다.

    입력: arguments - 영상경로, 선택적 출력경로, --rotate, --dry-run
    """
    options = parse_video_arguments(arguments)
    if options.dry_run:
        run_dry_run(options.rotate)
        return
    if options.video is None:
        sys.exit(__doc__)
    dlib_path = get_dlib_path()
    video = resolve_video_path(options.video)
    output = resolve_output_path(video, options.output)
    original = read_original_source()
    replacements = build_replacements(output, video, dlib_path, options.rotate)
    check_counts(original, replacements)
    modified = apply_replacements(original, replacements)
    print("치환된 줄 번호: %s" % changed_line_numbers(original, modified))
    os.makedirs(os.path.dirname(output), exist_ok=True)
    prepare_runtime()
    print("회전: %s\n영상: %s\n출력: %s" % (options.rotate, video, output))
    started = time.time()
    try:
        run_source(modified)
    finally:
        print("소요 시간: %.1f초" % (time.time() - started))
        print("blink 행 수(첫 빈 줄 제외): %d" % count_blink_rows(output))


def read_text_lines(path):
    """파일을 줄바꿈 변환 없이 읽어 '\\n' 으로 나눈 줄 목록을 돌려준다.

    입력: path - 텍스트 파일 경로
    반환: 줄 문자열 list (마지막 줄바꿈 유무도 비교에 드러나도록 그대로 나눈다)
    """
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as file:
        return file.read().split("\n")


def find_first_difference(lines_a, lines_b):
    """두 줄 목록을 순서대로 비교해 처음 달라지는 곳을 찾는다.

    입력: lines_a, lines_b - 줄 문자열 list
    반환: 같으면 None, 다르면 (줄 번호(1부터), a의 줄 또는 None, b의 줄 또는 None)
    """
    longest = max(len(lines_a), len(lines_b))
    for index in range(longest):
        line_a = lines_a[index] if index < len(lines_a) else None
        line_b = lines_b[index] if index < len(lines_b) else None
        if line_a != line_b:
            return index + 1, line_a, line_b
    return None


def run_compare(path_a, path_b):
    """두 출력 파일을 줄 단위로 비교해 결과를 출력한다. 영상은 읽지 않는다.

    입력: path_a, path_b - 비교할 txt 경로
    반환: 종료 코드(같으면 0, 다르면 1)
    """
    for path in (path_a, path_b):
        if not os.path.isfile(path):
            sys.exit("파일을 찾을 수 없습니다: %s" % path)
    lines_a = read_text_lines(path_a)
    lines_b = read_text_lines(path_b)
    difference = find_first_difference(lines_a, lines_b)
    print("A 줄 수 %d, B 줄 수 %d" % (len(lines_a), len(lines_b)))
    if difference is None:
        print("두 파일이 완전히 같습니다.")
        return 0
    number, line_a, line_b = difference
    print("처음 다른 줄: %d번째 줄" % number)
    print("  A: %s" % ("(줄 없음)" if line_a is None else repr(line_a)))
    print("  B: %s" % ("(줄 없음)" if line_b is None else repr(line_b)))
    return 1


def main():
    """명령줄 인자에 따라 --compare 서브모드 또는 영상 실행 모드로 나눈다."""
    arguments = sys.argv[1:]
    if arguments and arguments[0] == "--compare":
        if len(arguments) != 3:
            sys.exit(__doc__)
        sys.exit(run_compare(arguments[1], arguments[2]))
    run_video_mode(arguments)


if __name__ == "__main__":
    main()
