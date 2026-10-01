# train_fold_seeds.py
# 역할: 지정한 fold(1~5)를 지정한 시드로 재학습하고 epoch마다 지표를 CSV로 기록한다. train_fold1_seeds.py의 일반화 버전.
# 사용법: python src\repro\train_fold_seeds.py <fold 1~5> <시드> [epoch수=80] [출력폴더명=fold<fold>_seed<시드>]
#         출력: experiments\02_fold_seeds\<출력폴더명>\ (metrics.csv, train.log, run_meta.json, my_model<fold>.*)
# metrics.csv 열 (00_reproduction\fold1_training\training_history.csv와 앞 3개 이름이 같음, 한 행 = epoch 하나):
#   epoch            - 1부터 시작하는 epoch 번호
#   train_loss_sum   - 그 epoch 안 학습 배치 손실의 합
#   train_moving_acc - 그 epoch 안 학습 배치 정확도의 이동 평균
#   BSA_loss         - 테스트 세트 전체 손실 (기존 파일의 test_bsa_loss)
#   BSA_acc          - 테스트 세트 blink sequence 정확도 (기존 test_bsa_acc)
#   VA               - 테스트 영상 단위 투표 정확도 (기존 va)
#   BSRE             - blink sequence 회귀 오차 (기존 bsre)
#   VRE              - 영상 단위 회귀 오차 (기존 vre)
"""fold 하나 지정 재학습 스크립트(train_fold1_seeds.py를 일반화한 것).

원본 third_party/ghoddoosian/Training.py는 디스크에서 수정하지 않는다. 텍스트로 읽어 메모리에서만
아래 치환을 한 뒤 exec한다(기존 reproduction_train_fold1.py와 같은 방식).
원본은 반복 변수 i(0부터)로 입력 파일(Fold<i+1>)과 start_indices(if i==0, if i==1 ...)를 고르므로,
반복 범위를 range(fold-1, fold) 하나로 좁히면 입력과 start_indices가 모두 해당 fold에 맞게 선택된다.

  Train() 안 (정의부)
    * tf.reset_default_graph() 바로 다음 줄에 tf.set_random_seed(<시드>) 추가
      -> Train()이 매번 그래프를 새로 만들므로, 그 전에 건 그래프 시드는 버려진다. 새 그래프에 걸어야 적용된다.
    * variable_path='./' -> 출력 폴더 (체크포인트 저장 위치)
    * y_v[idx]=... 다음 줄에 _on_epoch_end(...) 호출 추가 (epoch 지표 기록용, 계산에는 영향 없음)
  fold 반복부
    * for i in range(5) -> range(fold-1, fold) (지정 fold만),  ii=i -> ii=i+1 (my_model<fold> 이름으로 저장)
    * num_epochs=80 -> 명령행 epoch 수,  open('./...') 6곳 -> 출력 폴더

하이퍼파라미터(80 epoch, batch 64, lr 0.000053, 4x32, step 30), 정규화, 모델, 손실은 그대로다.
체크포인트는 원본과 같이 학습이 끝난 뒤 한 번만 저장된다. 보고값은 마지막 epoch 값이다.
"""
import sys

sys.dont_write_bytecode = True  # third_party(원본 모듈 import)에 __pycache__가 생기지 않게 한다

import csv
import datetime
import json
import os
import platform
import random
import time
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import tensorflow as tf

import common

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
UPSTREAM_DIR = os.path.join(REPO_ROOT, "third_party", "ghoddoosian")
OUTPUT_PARENT = os.path.join(REPO_ROOT, "experiments", "02_fold_seeds")
DEFAULT_EPOCHS = 80
INPUT_FILE_PATTERNS = ["Blinks_30_Fold%d.npy", "Labels_30_Fold%d.npy",
                       "BlinksTest_30_Fold%d.npy", "LabelsTest_30_Fold%d.npy"]
USAGE = "사용법: python src\\repro\\train_fold_seeds.py <fold 1~5> <시드> [epoch수=80] [출력폴더명=fold<fold>_seed<시드>]"
# 기존 training_history.csv와 같은 순서, 같은 의미다(각 열 의미는 파일 상단 주석 참고).
METRIC_COLUMNS = ["epoch", "train_loss_sum", "train_moving_acc", "BSA_loss", "BSA_acc", "VA", "BSRE", "VRE"]

_log_file = None  # log()가 함께 기록할 train.log 파일 핸들


def parse_arguments(argv):
    """명령행 인자를 읽는다.

    입력: argv - sys.argv
    반환: (fold, seed, num_epochs, output_name) 튜플. 인자가 잘못되면 사용법을 출력하고 종료한다.
    """
    if len(argv) < 3 or len(argv) > 5:
        sys.exit(USAGE)
    try:
        fold = int(argv[1])
        seed = int(argv[2])
        num_epochs = int(argv[3]) if len(argv) > 3 else DEFAULT_EPOCHS
    except ValueError:
        sys.exit("fold, 시드, epoch 수는 정수여야 합니다: %s\n%s" % (argv[1:], USAGE))
    if fold < 1 or fold > 5:
        sys.exit("fold는 1~5여야 합니다: %d" % fold)
    if num_epochs < 1:
        sys.exit("epoch 수는 1 이상이어야 합니다: %d" % num_epochs)
    output_name = argv[4] if len(argv) > 4 else "fold%d_seed%d" % (fold, seed)
    return fold, seed, num_epochs, output_name


def make_output_dir(output_name):
    """출력 폴더를 만든다. 이미 있으면 덮어쓰지 않고 종료한다.

    입력: output_name - experiments\\02_fold_seeds 아래에 만들 폴더 이름
    반환: 만든 폴더의 절대경로
    """
    output_dir = os.path.join(OUTPUT_PARENT, output_name)
    if os.path.exists(output_dir):
        sys.exit("출력 폴더가 이미 있습니다(덮어쓰지 않음): %s" % output_dir)
    os.makedirs(output_dir)
    return output_dir


def start_log(output_dir):
    """train.log를 열어 log()가 쓸 수 있게 한다.

    입력: output_dir - 출력 폴더 경로
    반환: 없음
    """
    global _log_file
    _log_file = open(os.path.join(output_dir, "train.log"), "w", encoding="utf-8")


def log(message):
    """메시지를 화면에 출력하고 train.log에도 같은 내용을 기록한다.

    입력: message - 출력할 문자열
    반환: 없음
    """
    print(message)
    sys.stdout.flush()
    if _log_file is not None:
        _log_file.write(message + "\n")
        _log_file.flush()


def upstream_print(*values, **options):
    """원본 Training.py의 print를 대신해 화면과 train.log에 함께 남긴다.

    입력: print와 같은 인자(sep, end 등 옵션은 무시하고 공백으로 이어 붙인다)
    반환: 없음
    """
    texts = []
    for value in values:
        texts.append(str(value))
    log(" ".join(texts))


def set_python_and_numpy_seeds(seed):
    """파이썬 random과 numpy의 전역 시드를 설정한다.

    입력: seed - 정수 시드
    반환: 없음. TensorFlow 그래프 시드는 Train() 안에서 메모리 치환으로 설정된다(모듈 설명 참고).
    """
    random.seed(seed)
    np.random.seed(seed)


def replace_once(text, old, new):
    """text 안에서 old가 정확히 한 번 나오는지 확인하고 new로 바꾼다.

    입력: text - 원본 문자열, old - 찾을 문자열, new - 바꿀 문자열
    반환: 치환된 문자열. old가 0번 또는 2번 이상 나오면 AssertionError(원본이 예상과 다르다는 뜻)
    """
    count = text.count(old)
    assert count == 1, "원본에서 %r가 %d번 나옵니다(1번이어야 함)" % (old, count)
    return text.replace(old, new)


def patch_definitions(head, seed, output_prefix):
    """Training.py 정의부(Train 함수 포함)에 시드 설정, 저장 경로, epoch 기록 호출을 넣는다.

    입력: head - fold 반복부 이전의 원본 코드, seed - 정수 시드, output_prefix - '/'로 끝나는 출력 폴더 경로
    반환: 치환된 코드 문자열
    """
    head = replace_once(head, "    tf.reset_default_graph()\n",
                        "    tf.reset_default_graph()\n    tf.set_random_seed(%d)\n" % seed)
    head = replace_once(head, "    variable_path='./'\n", "    variable_path=%r\n" % output_prefix)
    head = replace_once(head, "            y_v[idx]=accuracy_per_videoV\n",
                        "            y_v[idx]=accuracy_per_videoV\n"
                        "            _on_epoch_end(idx, y[idx], yy[idx], y_test[idx], yy_test[idx], y_v[idx])\n")
    return head


def patch_fold_loop(tail, fold, num_epochs, output_prefix):
    """Training.py fold 반복부를 지정 fold 하나만, 지정 epoch 수로, 출력 폴더에 저장하도록 바꾼다.

    입력: tail - 'load=False'부터 끝까지의 원본 코드, fold - 1~5, num_epochs - epoch 수, output_prefix - 출력 폴더 경로
    반환: 치환된 코드 문자열
    """
    tail = replace_once(tail, "for i in range(5):", "for i in range(%d, %d):" % (fold - 1, fold))
    tail = replace_once(tail, "ii=i    #", "ii=i+1    #")
    tail = replace_once(tail, "num_epochs=80", "num_epochs=%d" % num_epochs)
    assert tail.count("open('./") == 6, "원본의 np.save(open('./...')) 개수가 예상(6)과 다릅니다"
    tail = tail.replace("open('./", "open(%r + '" % output_prefix)
    return tail


def build_training_code(fold, seed, num_epochs, output_dir):
    """원본 Training.py를 읽어 정의부와 반복부로 나누고 각각 메모리에서 치환한다.

    입력: fold - 1~5, seed - 정수 시드, num_epochs - epoch 수, output_dir - 출력 폴더 경로
    반환: (정의부 코드, 반복부 코드) 튜플
    """
    with open(os.path.join(UPSTREAM_DIR, "Training.py"), "r") as file:
        source = file.read()
    lines = source.split("\n")
    loop_start = 0
    while not lines[loop_start].startswith("load=False"):
        loop_start += 1
    head = "\n".join(lines[:loop_start]) + "\n"
    tail = "\n".join(lines[loop_start:]) + "\n"
    output_prefix = output_dir.replace("\\", "/") + "/"
    return patch_definitions(head, seed, output_prefix), patch_fold_loop(tail, fold, num_epochs, output_prefix)


def write_metrics_header(metrics_path):
    """metrics.csv를 만들고 열 이름 한 줄을 쓴다.

    입력: metrics_path - metrics.csv 경로
    반환: 없음
    """
    with open(metrics_path, "w", newline="") as file:
        csv.writer(file).writerow(METRIC_COLUMNS)


def append_metrics_row(metrics_path, values):
    """metrics.csv에 한 줄을 추가하고 바로 디스크에 반영(flush)한다.

    입력: metrics_path - metrics.csv 경로, values - METRIC_COLUMNS 순서의 값 list
    반환: 없음
    """
    with open(metrics_path, "a", newline="") as file:
        row = [values[0]]
        for value in values[1:]:
            row.append("%.6f" % value)
        csv.writer(file).writerow(row)
        file.flush()
        os.fsync(file.fileno())


def make_value_catcher(original_function, store, key):
    """원본 지표 함수를 감싸 반환값을 store[key]에 남기는 함수를 만든다.

    입력: original_function - 원본 함수(regression_per_blink 등), store - 값을 담을 dict, key - 저장할 키
    반환: 원본과 같은 인자/반환값을 갖는 함수
    """
    def catcher(*args, **kwargs):
        """원본 지표 함수를 그대로 호출하고 반환값을 기록한다.

        입력: 원본 함수와 같은 인자
        반환: 원본 함수의 반환값(그대로)
        """
        value = original_function(*args, **kwargs)
        store[key] = float(value)
        return value
    return catcher


def make_epoch_recorder(metrics_path, regression_values, history):
    """Train() 안에서 epoch마다 불릴 기록 함수를 만든다.

    입력: metrics_path - metrics.csv 경로, regression_values - BSRE/VRE가 담기는 dict,
          history - 기록한 행을 모아 둘 list
    반환: _on_epoch_end(idx, train_loss_sum, train_moving_acc, bsa_loss, bsa_acc, va) 함수
    """
    def on_epoch_end(idx, train_loss_sum, train_moving_acc, bsa_loss, bsa_acc, va):
        """epoch 하나의 지표를 metrics.csv에 한 줄로 추가한다.

        입력: idx - 0부터 시작하는 epoch 번호, 나머지 - Train()의 y, yy, y_test, yy_test, y_v 해당 칸 값
        반환: 없음
        """
        values = [idx + 1, float(train_loss_sum), float(train_moving_acc), float(bsa_loss), float(bsa_acc), float(va),
                  regression_values["BSRE"], regression_values["VRE"]]
        append_metrics_row(metrics_path, values)
        history.append(values)
    return on_epoch_end


def prepare_namespace(head, metrics_path, history):
    """정의부를 실행해 exec용 이름공간을 만들고, 지표 기록 함수와 print 대체 함수를 연결한다.

    입력: head - 치환된 정의부 코드, metrics_path - metrics.csv 경로, history - 기록 행을 모을 list
    반환: 반복부 exec에 쓸 dict 이름공간
    """
    namespace = {"__name__": "training_inmemory", "print": upstream_print}
    exec(compile(head, "Training.py[정의부]", "exec"), namespace)
    regression_values = {"BSRE": float("nan"), "VRE": float("nan")}
    namespace["regression_per_blink"] = make_value_catcher(namespace["regression_per_blink"], regression_values, "BSRE")
    namespace["regression_per_video"] = make_value_catcher(namespace["regression_per_video"], regression_values, "VRE")
    namespace["_on_epoch_end"] = make_epoch_recorder(metrics_path, regression_values, history)
    return namespace


def input_file_hashes(fold):
    """지정 fold의 입력 .npy 4개의 SHA-256을 계산한다.

    입력: fold - 1~5
    반환: {파일 이름: SHA-256} dict
    """
    hashes = {}
    for pattern in INPUT_FILE_PATTERNS:
        name = pattern % fold
        hashes[name] = common.sha256_of_file(os.path.join(UPSTREAM_DIR, name))
    return hashes


def now_text():
    """현재 시각을 초 단위 ISO 8601 문자열로 돌려준다.

    입력: 없음
    반환: 예) '2026-01-01T09:00:00'
    """
    return datetime.datetime.now().isoformat(timespec="seconds")


def build_run_meta(fold, seed, num_epochs):
    """run_meta.json에 들어갈 실행 정보(시작 시점 기준)를 만든다.

    입력: fold - 1~5, seed - 정수 시드, num_epochs - epoch 수
    반환: dict (종료 시각, 소요 초, 최종 지표는 None으로 두고 학습 후 채운다)
    """
    git_info = common.get_git_info(REPO_ROOT)
    return {
        "fold": fold,
        "seed": seed,
        "num_epochs": num_epochs,
        "git_commit": git_info["commit"],
        "git_dirty": git_info["dirty"],
        "python_version": platform.python_version(),
        "tensorflow_version": tf.__version__,
        "numpy_version": np.__version__,
        "started_at": now_text(),
        "finished_at": None,
        "elapsed_seconds": None,
        "status": "running",
        "input_sha256": input_file_hashes(fold),
        "final_epoch_metrics": None,
    }


def write_run_meta(output_dir, run_meta):
    """run_meta.json을 쓴다(같은 실행 안에서 시작 시 한 번, 종료 시 한 번 갱신).

    입력: output_dir - 출력 폴더 경로, run_meta - 기록할 dict
    반환: 없음
    """
    with open(os.path.join(output_dir, "run_meta.json"), "w", encoding="utf-8") as file:
        json.dump(run_meta, file, ensure_ascii=False, indent=2)


def main():
    """지정 fold를 시드 하나로 학습하고 지표, 로그, 실행 정보를 출력 폴더에 남긴다.

    입력: 없음(명령행 인자 사용)
    반환: 없음. 무결성 검사나 학습이 실패하면 예외로 종료된다(0이 아닌 종료 코드).
    """
    fold, seed, num_epochs, output_name = parse_arguments(sys.argv)
    common.check_manifest(REPO_ROOT)
    output_dir = make_output_dir(output_name)
    start_log(output_dir)
    log("시작: fold %d, 시드 %d, epoch %d, 출력 폴더 %s" % (fold, seed, num_epochs, output_dir))
    log("third_party 무결성 검사 통과(시작 시)")

    run_meta = build_run_meta(fold, seed, num_epochs)
    write_run_meta(output_dir, run_meta)
    metrics_path = os.path.join(output_dir, "metrics.csv")
    write_metrics_header(metrics_path)

    os.chdir(UPSTREAM_DIR)  # 원본은 .npy를 상대경로로 읽는다(쓰기는 모두 출력 폴더로 돌려 놓음)
    sys.path.insert(0, UPSTREAM_DIR)
    set_python_and_numpy_seeds(seed)
    head, tail = build_training_code(fold, seed, num_epochs, output_dir)
    history = []
    namespace = prepare_namespace(head, metrics_path, history)
    started = time.time()
    exec(compile(tail, "Training.py[fold 반복부]", "exec"), namespace)
    elapsed = time.time() - started

    common.check_manifest(REPO_ROOT)
    log("third_party 무결성 검사 통과(종료 시)")
    finish_run(output_dir, run_meta, history, elapsed)


def finish_run(output_dir, run_meta, history, elapsed):
    """마지막 epoch 지표를 출력하고 run_meta.json을 완료 상태로 갱신한다.

    입력: output_dir - 출력 폴더, run_meta - 시작 시 만든 dict, history - epoch별 기록 행, elapsed - 학습 소요 초
    반환: 없음
    """
    last_row = history[-1]
    final_metrics = dict(zip(METRIC_COLUMNS, last_row))
    run_meta["finished_at"] = now_text()
    run_meta["elapsed_seconds"] = round(elapsed, 1)
    run_meta["status"] = "completed"
    run_meta["final_epoch_metrics"] = final_metrics
    write_run_meta(output_dir, run_meta)
    log("마지막 epoch %d: BSA loss %.6f | BSA acc %.6f | VA %.6f | BSRE %.6f | VRE %.6f"
        % (last_row[0], last_row[3], last_row[4], last_row[5], last_row[6], last_row[7]))
    log("완료: 학습 소요 %.1f분" % (elapsed / 60.0))


if __name__ == "__main__":
    main()
