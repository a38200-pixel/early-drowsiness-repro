# verify_blinks.py
# 역할: 저자 원본 blink_video.py 를 메모리에서 치환 실행해 영상 1개의 깜빡임 행을 받고, B1 이 파일로 남긴 blinks 파일과 대조한다
#       (결정 I-28 1단계 합격 기준: 행 수 일치, 열 수 일치, 모든 값 5e-5 이내). 영상 파일 입력만 다루며 디스크에 아무 파일도 쓰지 않는다.
#       프레임이나 이미지를 저장하지 않고 웹캠을 열지 않는다.
# 사용법: (저장소 폴더에서 conda activate ghoddoosian_tf1, set PYTHONPATH=src)
#           python -m improve.hmlstm.verify_blinks --video <영상> --svm <SVM .sav> --predictor <68 landmarks .dat> --reference <B1 blinks txt> [--rotate none]
#           python -m improve.hmlstm.verify_blinks --dry-run     (치환 일치 횟수와 치환된 줄 번호만 출력, 영상과 모델 파일은 필요 없음)
#         --max-frames N : 앞 N프레임만 처리하고 끝낸다(부분 실행). 기준의 앞쪽 행과만 비교하며 합격 판정이 아니다.
#         --progress-every N : N프레임마다 한 줄 출력(기본 500, 0 이면 끝날 때까지 출력 없음).
#         --show-original-output : 원본이 찍는 print 를 그대로 출력한다(기본은 끈다).
# 종료 코드: 0 = 일치(전체 실행이면 합격), 1 = 불일치, 2 = 입력 오류
# 참고: 원본은 Tk 창과 matplotlib 그림을 실제로 만든다(I-W-26). 실행 중 작은 빈 창이 보일 수 있고 끝나면 닫는다.
"""원본 메모리 치환 실행과 B1 blinks 대조 CLI(Python 3.6 호환)."""
import argparse
import os
import sys

from . import compare_rows as CR
from . import original_hook as OH


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="저자 원본을 메모리 치환으로 실행해 깜빡임 행을 B1 blinks 파일과 대조한다(파일을 쓰지 않음).")
    parser.add_argument("--video", help="영상 파일 경로")
    parser.add_argument("--svm", help="저자 SVM(.sav) 경로")
    parser.add_argument("--predictor", help="dlib 68점 랜드마크(.dat) 경로")
    parser.add_argument("--reference", help="B1 blinks 파일 경로")
    parser.add_argument("--rotate", choices=list(OH.ROTATE_OPTIONS), default="none", help="B1 추출에 쓴 회전 옵션(기본 none)")
    parser.add_argument("--progress-every", type=int, default=500, dest="progress_every", help="N프레임마다 한 줄 출력(0 이면 출력 없음)")
    parser.add_argument("--max-frames", type=int, default=0, dest="max_frames", help="앞 N프레임만 처리(부분 실행, 합격 판정 아님)")
    parser.add_argument("--show-original-output", action="store_true", dest="show_original_output", help="원본의 print 출력을 끄지 않는다")
    parser.add_argument("--dry-run", action="store_true", dest="dry_run", help="치환 일치 횟수와 치환된 줄 번호만 출력")
    parser.add_argument("--context", type=int, default=3, help="첫 불일치 앞뒤로 보여 줄 행 수(기본 3)")
    return parser.parse_args(argv)


def run_dry_run(rotate):
    """치환 일치 횟수와 치환된 줄 번호만 출력한다. 영상, 모델 파일은 열지 않는다. 반환: 종료 코드."""
    original = OH.read_original_source()
    replacements = OH.build_replacements("VIDEO", "PREDICTOR", "SVM", rotate, tick=True)
    for label, expected, actual in OH.count_matches(original, replacements):
        print("replacement [%s]: expected %d, actual %d" % (label, expected, actual))
    try:
        OH.check_counts(original, replacements)
        modified = OH.apply_replacements(original, replacements)
    except OH.ReplacementError as error:
        print(error)
        return 2
    print("changed lines: %s" % OH.changed_line_numbers(original, modified))
    return 0


def total_frame_count(video):
    """진행 표시용 전체 프레임 수(메타데이터 값, 실패하면 None)."""
    try:
        import cv2
        capture = cv2.VideoCapture(video)
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()
        return count if count > 0 else None
    except Exception:
        return None


def main(argv=None):
    options = parse_args(argv)
    if options.dry_run:
        return run_dry_run(options.rotate)
    missing = [name for name in ("video", "svm", "predictor", "reference") if not getattr(options, name)]
    if missing:
        print("필수 인자가 없습니다: " + ", ".join("--" + name for name in missing))
        return 2
    for name in ("video", "svm", "predictor", "reference"):
        if not os.path.isfile(getattr(options, name)):
            print("파일을 찾을 수 없습니다: --%s" % name)
            return 2
    try:
        reference_rows = CR.read_blink_file(options.reference)
        total = total_frame_count(options.video)
        print("video=%s frames(metadata)=%s rotate=%s max_frames=%d" % (os.path.basename(options.video), total, options.rotate, options.max_frames))
        result = OH.extract_rows(options.video, options.svm, options.predictor, options.rotate, options.progress_every, options.max_frames,
                                 not options.show_original_output, total_frames=total)
    except (OH.ReplacementError, ValueError) as error:
        print(error)
        return 2
    print("done: elapsed_s=%.1f counted_frames=%d rows=%d changed_lines=%s" % (result.elapsed_s, result.frames, len(result.rows), result.changed_lines))
    report = CR.compare_rows(result.rows, reference_rows, allow_prefix=bool(options.max_frames))
    for line in CR.format_report(report, result.rows, reference_rows, options.context):
        print(line)
    if options.max_frames:
        print("note: partial run, compared against the first rows of the reference only; this is not the pass/fail judgment")
        return 0 if report["verdict"] == "일치" else 1
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
