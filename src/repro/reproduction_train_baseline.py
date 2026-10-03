# reproduction_train_baseline.py
# 역할: 우리가 만든 시퀀스 산출물(SEQ_OUT_DIR 의 npy 와 start_indices JSON)로 저자 Training.py 와 같은 학습·평가를 한다.
#       Training.py 는 텍스트로 읽어 메모리에서만 치환해 exec 한다(third_party 는 디스크에서 수정하지 않는다).
# 사용법: 저장소 루트에서 conda 환경 ghoddoosian_tf1 의 python 으로 실행한다.
#           python src/repro/reproduction_train_baseline.py --fold 1 --seed 1 --dry-run   (치환 결과와 줄 번호만 출력, TF 를 import 하지 않음)
#           python src/repro/reproduction_train_baseline.py --fold all --seed 1           (80 epoch 학습, 저자 하이퍼파라미터 그대로)
#           python src/repro/reproduction_train_baseline.py --fold 1 --seed 1 --smoke --epochs 2   (스모크 시험, 결과 집계는 무시)
#         옵션: --fold {1..5,all}, --seed N(필수), --data-dir, --start-indices-dir, --smoke, --epochs N(--smoke 전용),
#               --run-name NAME, --dry-run
# 환경변수: TRAIN_OUT_DIR (출력 루트, 저장소 밖, 기본값 없음. 저장소 안이거나 my_model1~5 폴더이면 중단. --dry-run 에서는 불필요)
#           SEQ_OUT_DIR (--data-dir, --start-indices-dir 를 주지 않을 때 npy, metadata 폴더를 찾는 기본 위치)
# 출력: TRAIN_OUT_DIR/<실행이름>/fold<k>/ 아래 체크포인트(my_model<k>.*), x/loss/accuracy*.npy, metrics.csv(epoch별 지표),
#       final_metrics.json(마지막 epoch 의 영상 단위 VA/VRE 등), run_config.json, train.log. --smoke 이면 SMOKE 표식 파일도 쓴다.
#       이미 있는 fold 폴더가 있으면 덮어쓰지 않고 중단한다.
"""저자 Training.py 와 같은 학습을 우리 산출물로 수행하는 러너(Python 3.6 호환, TF 1.8).

저자 코드와 같은 점: 모델, 손실(Δ=1.253, L2 0.1), 최적화(Adam, lr 0.000053), 80 epoch, batch 64, 2차 정규화, 지표 함수
  (영상 다수결 VA, VRE, BSRE, 임계값 3.34)는 원본 소스를 그대로 실행한다. --smoke 가 없으면 epoch 도 치환하지 않는다.
의도한 차이: 데이터 경로와 start_indices 를 우리 산출물로 바꾼다. 출력 위치를 바꾼다. 난수 시드를 고정한다(원본은 시드가 없다).
  체크포인트 이름을 my_model<fold> 로 맞춘다(원본의 ii=i 를 ii=i+1 로). --smoke(짧은 epoch)는 스모크 시험 전용이다.
"""
import sys

sys.dont_write_bytecode = True  # third_party 에 __pycache__ 가 생기지 않게 한다

import argparse
import csv
import datetime
import hashlib
import json
import os
import platform
import random
import re
import time
import warnings

warnings.filterwarnings("ignore")
import numpy as np

import common

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
UPSTREAM_DIR = os.path.join(REPO, "third_party", "ghoddoosian")
TRAINING_PATH = os.path.join(UPSTREAM_DIR, "Training.py")
INPUT_PATTERNS = ("Blinks_30_Fold%d.npy", "Labels_30_Fold%d.npy", "BlinksTest_30_Fold%d.npy", "LabelsTest_30_Fold%d.npy")
METRIC_COLUMNS = ["epoch", "train_loss_sum", "train_moving_acc", "BSA_loss", "BSA_acc", "VA", "BSRE", "VRE"]
AUTHOR_EPOCHS = 80
SMOKE_DEFAULT_EPOCHS = 2
SMOKE_MARKER = "SMOKE"
# 저자 하이퍼파라미터 문자열(원본에서 각각 정확히 1번 나와야 한다)
HYPERPARAMETER_TOKENS = ("batch_size=64,num_epochs=80,", "hstate_size=[32,32,32,32],num_layers=4,step_size=30,drop_out_p=1.0,",
                         "lr=0.000053,th=1.253,", "0.1 * (tf.nn.l2_loss")

_log_file = None  # log() 가 함께 기록할 train.log 파일 핸들


def parse_arguments():
    """명령줄 옵션을 해석한다.

    반환: argparse.Namespace (fold, seed, data_dir, start_dir, smoke, epochs, run_name, dry_run)
    """
    parser = argparse.ArgumentParser(description="저자 Training.py 와 같은 학습을 우리 산출물로 수행(메모리 치환)")
    parser.add_argument("--fold", default="all", help="1~5 또는 all(기본)")
    parser.add_argument("--seed", type=int, required=True, help="난수 시드(원본에는 시드가 없어 재현용으로 고정)")
    parser.add_argument("--data-dir", dest="data_dir", default=None, help="npy 폴더(기본 SEQ_OUT_DIR/npy)")
    parser.add_argument("--start-indices-dir", dest="start_dir", default=None, help="start_indices JSON 폴더(기본 SEQ_OUT_DIR/metadata)")
    parser.add_argument("--smoke", action="store_true", help="스모크 시험(fold 하나, epoch 수 지정 가능, SMOKE 표식 파일을 씀)")
    parser.add_argument("--epochs", type=int, default=None, help="--smoke 에서만 쓰는 epoch 수(기본 %d)" % SMOKE_DEFAULT_EPOCHS)
    parser.add_argument("--run-name", dest="run_name", default=None, help="실행 이름(기본 seed<시드> 또는 smoke_seed<시드>)")
    parser.add_argument("--dry-run", action="store_true", dest="dry_run", help="치환 결과와 줄 번호만 출력(TF 를 import 하지 않음)")
    return parser.parse_args()


def parse_folds_and_epochs(arguments):
    """--fold, --smoke, --epochs 의 조합을 검사하고 (fold 목록, 치환할 epoch 수 또는 None)을 돌려준다.

    입력: arguments - parse_arguments 결과
    반환: (fold list, epochs). epochs 는 --smoke 일 때만 정수이고, 아니면 None(저자 값 그대로)
    """
    if arguments.fold == "all":
        folds = [1, 2, 3, 4, 5]
    elif arguments.fold in ("1", "2", "3", "4", "5"):
        folds = [int(arguments.fold)]
    else:
        sys.exit("--fold 는 1~5 또는 all 이어야 합니다: %s" % arguments.fold)
    if arguments.epochs is not None and not arguments.smoke:
        sys.exit("--epochs 는 --smoke 와 함께만 쓸 수 있습니다(스모크가 아닌 실행은 저자의 80 epoch 그대로).")
    if arguments.smoke and len(folds) != 1:
        sys.exit("--smoke 는 fold 하나만 지정할 수 있습니다.")
    epochs = (arguments.epochs or SMOKE_DEFAULT_EPOCHS) if arguments.smoke else None
    if epochs is not None and epochs < 1:
        sys.exit("--epochs 는 1 이상이어야 합니다.")
    return folds, epochs


def is_inside_dir(path, base):
    """path 가 base 와 같거나 그 안에 있는지 본다(Windows 대소문자 무시).

    입력: path, base - 경로
    반환: bool
    """
    path = os.path.normcase(os.path.realpath(path))
    base = os.path.normcase(os.path.realpath(base))
    try:
        return os.path.commonpath([path, base]) == base
    except ValueError:
        return False


def is_my_model_location(path):
    """경로의 어느 구성 요소가 my_model1~5 이름이거나, 그 폴더에 my_model1~5 체크포인트 파일이 있는지 본다.

    입력: path - 경로
    반환: bool
    """
    for part in re.split(r"[\\/]+", path):
        if re.match(r"^my_model[1-5]$", part, re.IGNORECASE):
            return True
    if os.path.isdir(path):
        return any(re.match(r"^my_model[1-5]\.", name) for name in os.listdir(path))
    return False


def require_train_dir():
    """TRAIN_OUT_DIR 를 읽고 검사한다. 설정이 없거나, 저장소 안이거나, my_model1~5 위치이면 중단한다(기본값 없음).

    반환: 출력 루트의 실제 경로
    """
    raw_value = os.environ.get("TRAIN_OUT_DIR")
    if not raw_value:
        sys.exit("환경변수 TRAIN_OUT_DIR 를 출력 루트로 설정하세요(저장소 밖, 기본값 없음).")
    path = os.path.realpath(raw_value)
    if is_inside_dir(path, REPO):
        sys.exit("TRAIN_OUT_DIR 이 저장소 폴더 안입니다. 저장소 밖 폴더를 지정하세요.")
    if is_my_model_location(path):
        sys.exit("TRAIN_OUT_DIR 이 my_model1~5 체크포인트 폴더이거나 그 안입니다. 다른 폴더를 지정하세요.")
    return path


def resolve_input_dirs(arguments):
    """npy 폴더와 start_indices 폴더를 정한다(옵션이 없으면 SEQ_OUT_DIR 아래 기본 위치).

    입력: arguments - parse_arguments 결과
    반환: (data_dir, start_dir) 실제 경로 튜플
    """
    seq_raw = os.environ.get("SEQ_OUT_DIR")
    data_dir, start_dir = arguments.data_dir, arguments.start_dir
    if (data_dir is None or start_dir is None) and not seq_raw:
        sys.exit("--data-dir 와 --start-indices-dir 를 주거나 환경변수 SEQ_OUT_DIR 를 설정하세요.")
    if data_dir is None:
        data_dir = os.path.join(seq_raw, "npy")
    if start_dir is None:
        start_dir = os.path.join(seq_raw, "metadata")
    return os.path.realpath(data_dir), os.path.realpath(start_dir)


def read_training_source():
    """원본 Training.py 를 텍스트로 읽는다(디스크는 수정하지 않는다).

    반환: 소스 문자열(줄바꿈은 \\n 으로 통일)
    """
    with open(TRAINING_PATH, "r", encoding="utf-8", errors="replace") as file:
        return file.read()


def check_hyperparameters(source):
    """저자 하이퍼파라미터 문자열이 원본에서 정확히 1번씩 나오는지 확인한다.

    입력: source - 원본 Training.py 소스
    반환: 없음. 하나라도 다르면 RuntimeError
    """
    problems = ["%r %d번" % (token, source.count(token)) for token in HYPERPARAMETER_TOKENS if source.count(token) != 1]
    if problems:
        raise RuntimeError("원본의 하이퍼파라미터 문자열이 예상과 다릅니다(각 1번이어야 함): " + ", ".join(problems))


def find_line_numbers(source, old):
    """source 안에서 old 가 시작하는 줄 번호(1부터)를 모두 찾는다.

    입력: source - 소스 문자열, old - 찾을 문자열
    반환: 줄 번호 list
    """
    lines, start = [], 0
    while True:
        position = source.find(old, start)
        if position < 0:
            return lines
        lines.append(source.count("\n", 0, position) + 1)
        start = position + len(old)


def make_substitution(name, old, new, expected):
    """치환 항목 하나를 dict 로 만든다.

    입력: name - 이름, old - 찾을 문자열, new - 바꿀 문자열, expected - 원본에서 나와야 하는 개수
    반환: dict(name, old, new, expected)
    """
    return {"name": name, "old": old, "new": new, "expected": expected}


def build_substitutions(plan):
    """치환 항목 목록을 만든다(줄 번호는 원본 Training.py 기준).

    입력: plan - dict(fold, seed, epochs, out_prefix, data_prefix, starts). epochs 가 None 이면 epoch 는 치환하지 않는다
    반환: 치환 항목 dict list
    """
    hook = "            _on_epoch_end(idx, y[idx], yy[idx], y_test[idx], yy_test[idx], y_v[idx])\n"
    reset = "    tf.reset_default_graph()\n"
    fold, prefix, data = plan["fold"], plan["out_prefix"], plan["data_prefix"]
    items = [make_substitution("seed(304)", reset, reset + "    tf.set_random_seed(%d)\n" % plan["seed"], 1),
             make_substitution("save_path(316)", "    variable_path='./'\n", "    variable_path=%r\n" % prefix, 1),
             make_substitution("epoch_hook(390)", "            y_v[idx]=accuracy_per_videoV\n",
                               "            y_v[idx]=accuracy_per_videoV\n" + hook, 1),
             make_substitution("fold_loop(408)", "for i in range(5):", "for i in range(%d, %d):" % (fold - 1, fold), 1),
             make_substitution("model_name(411)", "ii=i    #", "ii=i+1    #", 1)]
    loads = ((414, "Blinks_30_Fold", "Blinks_30_Fold"), (415, "Labels_30_Fold", "Labels_30_Fold"),
             (416, "BlinksTest_30_Fold", "BlinksTest_30_Fold"), (417, "./LabelsTest_30_Fold", "LabelsTest_30_Fold"))
    for number, old_name, new_name in loads:
        items.append(make_substitution("load_%s(%d)" % (new_name, number), "np.load('%s" % old_name,
                                       "np.load(%r + '%s" % (data, new_name), 1))
    items.append(make_substitution("start_indices(502)", "    start_indices=np.asarray(start_indices)\n",
                                   "    start_indices=%r\n    start_indices=np.asarray(start_indices)\n" % list(plan["starts"]), 1))
    items.append(make_substitution("save_files(510-515)", "open('./", "open(%r + '" % prefix, 6))
    if plan["epochs"] is not None:
        items.append(make_substitution("smoke_epochs(504)", "num_epochs=80", "num_epochs=%d" % plan["epochs"], 1))
    return items


def apply_substitutions(source, substitutions):
    """치환 항목마다 원본에서 나오는 개수를 검증하고 메모리에서 치환한다. 하나라도 기대와 다르면 RuntimeError.

    입력: source - 원본 소스, substitutions - build_substitutions 결과
    반환: (치환된 소스, 보고 list(dict(name, expected, found, lines)))
    """
    report, mismatches = [], []
    for item in substitutions:
        found = source.count(item["old"])
        report.append({"name": item["name"], "expected": item["expected"], "found": found,
                       "lines": find_line_numbers(source, item["old"])})
        if found != item["expected"]:
            mismatches.append("%s: %d번(기대 %d번)" % (item["name"], found, item["expected"]))
    if mismatches:
        raise RuntimeError("치환 대상 개수가 원본과 다릅니다: " + "; ".join(mismatches))
    text = source
    for item in substitutions:
        text = text.replace(item["old"], item["new"])
    return text, report


def split_source(text):
    """치환된 소스를 정의부(load=False 줄 앞)와 fold 반복부로 나눈다.

    입력: text - 치환된 소스
    반환: (head, tail)
    """
    lines = text.split("\n")
    starts = [index for index, line in enumerate(lines) if line.startswith("load=False")]
    if len(starts) != 1:
        raise RuntimeError("원본에서 'load=False' 줄이 %d개입니다(1개여야 함)" % len(starts))
    return "\n".join(lines[:starts[0]]) + "\n", "\n".join(lines[starts[0]:]) + "\n"


def read_start_indices(start_dir, fold):
    """start_indices_fold<k>.json 을 읽어 정수 list 로 돌려준다.

    입력: start_dir - JSON 폴더, fold - fold 번호
    반환: 정수 list
    """
    path = os.path.join(start_dir, "start_indices_fold%d.json" % fold)
    if not os.path.isfile(path):
        raise RuntimeError("start_indices 파일이 없습니다: start_indices_fold%d.json" % fold)
    with open(path, "r", encoding="utf-8") as file:
        starts = json.load(file)
    if not isinstance(starts, list) or not starts or not all(isinstance(v, int) for v in starts):
        raise RuntimeError("start_indices_fold%d.json 은 정수 list 여야 합니다." % fold)
    return starts


def check_start_indices(starts, test_labels, fold):
    """start_indices 가 저자 형식(0 시작, 증가, 마지막 영상은 끝까지)이고 구간마다 라벨이 하나인지 검사한다.

    입력: starts - 정수 list, test_labels - (N,1) 테스트 라벨, fold - fold 번호
    반환: 없음. 어긋나면 RuntimeError
    """
    total = len(test_labels)
    ends = list(starts[1:]) + [total]
    lengths = [end - start for start, end in zip(starts, ends)]
    if starts[0] != 0 or min(lengths) < 1 or starts[-1] >= total or sum(lengths) != total:
        raise RuntimeError("Fold%d start_indices 가 테스트 시퀀스 수 %d 와 맞지 않습니다." % (fold, total))
    for start, end in zip(starts, ends):
        segment = np.asarray(test_labels[start:end]).reshape(-1)
        if not np.all(segment == segment[0]) or segment[0] not in (0, 5, 10):
            raise RuntimeError("Fold%d 영상 구간 %d~%d 의 라벨이 하나가 아닙니다." % (fold, start, end))


def validate_fold_inputs(data_dir, start_dir, fold):
    """한 fold 의 입력 npy 4개와 start_indices 를 읽어 형상을 검사한다(읽기 전용 mmap).

    입력: data_dir - npy 폴더, start_dir - JSON 폴더, fold - fold 번호
    반환: dict(names, n_train, n_test, starts)
    """
    names, arrays = [], []
    for pattern in INPUT_PATTERNS:
        names.append(pattern % fold)
        path = os.path.join(data_dir, names[-1])
        if not os.path.isfile(path):
            raise RuntimeError("입력 파일이 없습니다: %s" % names[-1])
        arrays.append(np.load(path, mmap_mode="r"))
    train_x, train_y, test_x, test_y = arrays
    for x_array, y_array in ((train_x, train_y), (test_x, test_y)):
        if x_array.ndim != 3 or x_array.shape[1:] != (30, 4) or y_array.shape != (len(x_array), 1):
            raise RuntimeError("Fold%d 입력 형상이 (N,30,4)/(N,1) 이 아닙니다." % fold)
    starts = read_start_indices(start_dir, fold)
    check_start_indices(starts, test_y, fold)
    return {"names": names, "n_train": len(train_x), "n_test": len(test_x), "starts": starts}


def make_plan(fold, seed, epochs, info, out_prefix, data_prefix):
    """치환 계획 dict 를 만든다.

    입력: fold, seed, epochs(None 이면 저자 값), info - validate_fold_inputs 결과, out_prefix, data_prefix - '/'로 끝나는 경로
    반환: build_substitutions 에 넘길 dict
    """
    return {"fold": fold, "seed": seed, "epochs": epochs, "out_prefix": out_prefix, "data_prefix": data_prefix,
            "starts": info["starts"]}


def prepare_patched(source, plan):
    """치환 계획으로 소스를 치환하고 정의부와 반복부로 나눈다.

    입력: source - 원본 소스, plan - make_plan 결과
    반환: (head, tail, 보고 list, 치환된 전체 소스)
    """
    text, report = apply_substitutions(source, build_substitutions(plan))
    head, tail = split_source(text)
    return head, tail, report, text


def print_dry_run(fold, info, report, text, plan):
    """--dry-run 결과(치환 개수, 줄 번호, start_indices 사용 여부, 컴파일 결과)를 출력한다.

    입력: fold, info, report, text - 치환된 소스, plan - 치환 계획
    반환: 없음
    """
    print("== Fold%d: 입력 train %d, test %d 시퀀스, start_indices %d개(합 %d) ==" % (
        fold, info["n_train"], info["n_test"], len(info["starts"]), info["n_test"]))
    for item in report:
        print("  %-24s 기대 %d, 원본에서 %d번, 줄 %s" % (item["name"], item["expected"], item["found"], item["lines"]))
    print("  치환된 소스에 우리 start_indices 값이 쓰임: %s" % (("start_indices=%r\n" % list(plan["starts"])) in text))
    head, tail = split_source(text)
    compile(head, "Training.py[정의부]", "exec")
    compile(tail, "Training.py[fold 반복부]", "exec")
    print("  치환된 소스 컴파일: 통과")


def dry_run(source, folds, seed, epochs, data_dir, start_dir):
    """--dry-run: 모든 fold 의 치환 결과를 출력만 한다(TF import 없음, 파일 쓰기 없음).

    입력: source, folds, seed, epochs, data_dir, start_dir
    반환: 없음
    """
    before = common.sha256_of_file(TRAINING_PATH)
    for fold in folds:
        info = validate_fold_inputs(data_dir, start_dir, fold)
        plan = make_plan(fold, seed, epochs, info, "<출력 폴더>/", "<데이터 폴더>/")
        head, tail, report, text = prepare_patched(source, plan)
        print_dry_run(fold, info, report, text, plan)
    after = common.sha256_of_file(TRAINING_PATH)
    print("third_party/Training.py 해시 변화 없음: %s" % (before == after))
    print("--dry-run: 아무 파일도 쓰지 않았고 TensorFlow 도 import 하지 않았습니다.")


def log(message):
    """메시지를 화면에 출력하고 train.log 에도 같은 내용을 기록한다.

    입력: message - 문자열
    반환: 없음
    """
    print(message)
    sys.stdout.flush()
    if _log_file is not None:
        _log_file.write(message + "\n")
        _log_file.flush()


def upstream_print(*values, **options):
    """원본 Training.py 의 print 를 대신해 화면과 train.log 에 함께 남긴다.

    입력: print 와 같은 인자(옵션은 무시하고 공백으로 이어 붙인다)
    반환: 없음
    """
    log(" ".join(str(value) for value in values))


def append_metrics_row(metrics_path, values):
    """metrics.csv 에 한 줄을 추가하고 바로 디스크에 반영한다.

    입력: metrics_path - 경로, values - METRIC_COLUMNS 순서 값 list
    반환: 없음
    """
    with open(metrics_path, "a", newline="") as file:
        csv.writer(file).writerow([values[0]] + ["%.6f" % value for value in values[1:]])
        file.flush()
        os.fsync(file.fileno())


def make_value_catcher(original_function, store, key):
    """원본 지표 함수를 감싸 반환값을 store[key] 에 남기는 함수를 만든다(원본과 같은 인자·반환값).

    입력: original_function - 원본 함수, store - dict, key - 저장 키
    반환: 감싼 함수
    """
    def catcher(*args, **kwargs):
        """원본 함수를 그대로 호출하고 반환값을 기록한다.

        입력: 원본 함수와 같은 인자
        반환: 원본 함수의 반환값
        """
        value = original_function(*args, **kwargs)
        store[key] = float(value)
        return value
    return catcher


def make_epoch_recorder(metrics_path, regression_values, history):
    """Train() 안에서 epoch 마다 불릴 기록 함수를 만든다.

    입력: metrics_path - metrics.csv 경로, regression_values - BSRE/VRE 가 담기는 dict, history - 기록 행 list
    반환: _on_epoch_end 함수
    """
    def on_epoch_end(idx, train_loss_sum, train_moving_acc, bsa_loss, bsa_acc, va):
        """epoch 하나의 지표를 metrics.csv 에 추가한다.

        입력: idx - 0부터 시작하는 epoch 번호, 나머지 - Train() 의 y, yy, y_test, yy_test, y_v 칸 값
        반환: 없음
        """
        values = [idx + 1, float(train_loss_sum), float(train_moving_acc), float(bsa_loss), float(bsa_acc), float(va),
                  regression_values["BSRE"], regression_values["VRE"]]
        append_metrics_row(metrics_path, values)
        history.append(values)
    return on_epoch_end


def prepare_namespace(head, metrics_path, history):
    """정의부를 실행해 exec 용 이름공간을 만들고 지표 기록 함수와 print 대체 함수를 연결한다.

    입력: head - 치환된 정의부, metrics_path, history
    반환: dict 이름공간
    """
    namespace = {"__name__": "training_inmemory", "print": upstream_print}
    exec(compile(head, "Training.py[정의부]", "exec"), namespace)
    values = {"BSRE": float("nan"), "VRE": float("nan")}
    namespace["regression_per_blink"] = make_value_catcher(namespace["regression_per_blink"], values, "BSRE")
    namespace["regression_per_video"] = make_value_catcher(namespace["regression_per_video"], values, "VRE")
    namespace["_on_epoch_end"] = make_epoch_recorder(metrics_path, values, history)
    return namespace


def now_text():
    """현재 시각을 초 단위 ISO 8601 문자열로 돌려준다.

    반환: 문자열
    """
    return datetime.datetime.now().isoformat(timespec="seconds")


def write_json(path, data):
    """JSON 을 쓴다(같은 실행 안에서 시작 시와 종료 시 갱신하므로 덮어쓰기 모드).

    입력: path, data
    반환: 없음
    """
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def build_run_config(context, fold, info, report, patched_text, data_dir):
    """run_config.json 에 들어갈 구성 요약(시작 시점 기준)을 만든다. 경로는 쓰지 않고 파일 이름만 쓴다.

    입력: context - 실행 설정 dict, fold, info, report, patched_text - 치환된 소스, data_dir - npy 폴더
    반환: dict
    """
    import tensorflow as tf  # 실제 실행에서만 import 한다(--dry-run 에서는 호출되지 않음)
    git_info = common.get_git_info(REPO)
    return {"run_name": context["run_name"], "fold": fold, "seed": context["seed"], "smoke": context["smoke"],
            "num_epochs": context["epochs"] or AUTHOR_EPOCHS, "hyperparameter_tokens": list(HYPERPARAMETER_TOKENS),
            "input_sha256": dict((name, common.sha256_of_file(os.path.join(data_dir, name))) for name in info["names"]),
            "start_indices_file": "start_indices_fold%d.json" % fold, "start_indices_count": len(info["starts"]),
            "n_train_sequences": info["n_train"], "n_test_sequences": info["n_test"], "substitutions": report,
            "training_py_sha256": common.sha256_of_file(TRAINING_PATH),
            "patched_source_sha256": hashlib.sha256(patched_text.encode("utf-8")).hexdigest(),
            "git_commit": git_info["commit"], "git_dirty": git_info["dirty"], "python_version": platform.python_version(),
            "tensorflow_version": tf.__version__, "numpy_version": np.__version__, "started_at": now_text(),
            "finished_at": None, "elapsed_seconds": None, "status": "running", "final_epoch_metrics": None}


def prepare_fold_dir(train_dir, run_name, fold, smoke):
    """fold 출력 폴더를 만든다(이미 있으면 덮어쓰지 않고 중단). 스모크이면 SMOKE 표식 파일을 쓴다.

    입력: train_dir - 출력 루트, run_name, fold, smoke - 스모크 여부
    반환: 만든 폴더 경로
    """
    fold_dir = os.path.join(train_dir, run_name, "fold%d" % fold)
    if os.path.exists(fold_dir):
        sys.exit("출력 폴더가 이미 있습니다(덮어쓰지 않음): %s/fold%d" % (run_name, fold))
    os.makedirs(fold_dir)
    if smoke:
        with open(os.path.join(fold_dir, SMOKE_MARKER), "w", encoding="utf-8") as file:
            file.write("스모크 시험 실행입니다. 결과 집계는 이 폴더를 무시해야 합니다.\n")
    return fold_dir


def is_smoke_dir(fold_dir):
    """fold 출력 폴더가 스모크 시험 결과인지 본다(결과 집계에서 건너뛸 때 쓴다).

    입력: fold_dir - fold 출력 폴더
    반환: bool
    """
    return os.path.exists(os.path.join(fold_dir, SMOKE_MARKER))


def finish_fold(config, history, elapsed, fold_dir):
    """마지막 epoch 지표를 출력하고 run_config.json 과 final_metrics.json 을 완료 상태로 쓴다.

    입력: config - run_config dict, history - epoch별 기록 행, elapsed - 학습 소요 초, fold_dir - 출력 폴더
    반환: 없음
    """
    last = dict(zip(METRIC_COLUMNS, history[-1]))
    config.update({"finished_at": now_text(), "elapsed_seconds": round(elapsed, 1), "status": "completed",
                   "final_epoch_metrics": last})
    write_json(os.path.join(fold_dir, "run_config.json"), config)
    write_json(os.path.join(fold_dir, "final_metrics.json"), last)
    log("마지막 epoch %d: BSA loss %.6f | BSA acc %.6f | VA %.6f | BSRE %.6f | VRE %.6f" % (
        last["epoch"], last["BSA_loss"], last["BSA_acc"], last["VA"], last["BSRE"], last["VRE"]))
    log("완료: 학습 소요 %.1f분" % (elapsed / 60.0))


def execute_fold(context, head, tail, config, fold_dir):
    """치환된 소스를 실행해 학습하고 마지막 epoch 지표로 구성 파일과 final_metrics.json 을 마무리한다.

    입력: context, head, tail - 치환된 코드, config - run_config dict, fold_dir - 출력 폴더
    반환: 없음
    """
    global _log_file
    _log_file = open(os.path.join(fold_dir, "train.log"), "w", encoding="utf-8")
    metrics_path = os.path.join(fold_dir, "metrics.csv")
    with open(metrics_path, "w", newline="") as file:
        csv.writer(file).writerow(METRIC_COLUMNS)
    write_json(os.path.join(fold_dir, "run_config.json"), config)
    random.seed(context["seed"])
    np.random.seed(context["seed"])
    history = []
    namespace = prepare_namespace(head, metrics_path, history)
    started = time.time()
    exec(compile(tail, "Training.py[fold 반복부]", "exec"), namespace)
    common.check_manifest(REPO)
    finish_fold(config, history, time.time() - started, fold_dir)


def train_one_fold(context, fold):
    """한 fold 를 검증, 치환, 학습한다.

    입력: context - 실행 설정 dict, fold - fold 번호
    반환: 없음
    """
    info = validate_fold_inputs(context["data_dir"], context["start_dir"], fold)
    fold_dir = prepare_fold_dir(context["train_dir"], context["run_name"], fold, context["smoke"])
    out_prefix = fold_dir.replace("\\", "/") + "/"
    data_prefix = context["data_dir"].replace("\\", "/") + "/"
    plan = make_plan(fold, context["seed"], context["epochs"], info, out_prefix, data_prefix)
    head, tail, report, text = prepare_patched(context["source"], plan)
    config = build_run_config(context, fold, info, report, text, context["data_dir"])
    sys.path.insert(0, UPSTREAM_DIR)  # 원본의 'import HMSLSTM_Main' 용
    execute_fold(context, head, tail, config, fold_dir)


def check_fold_dirs_free(train_dir, run_name, folds):
    """실행할 fold 출력 폴더가 하나라도 이미 있으면 아무것도 하지 않고 중단한다.

    입력: train_dir, run_name, folds
    반환: 없음
    """
    for fold in folds:
        if os.path.exists(os.path.join(train_dir, run_name, "fold%d" % fold)):
            sys.exit("출력 폴더가 이미 있습니다(덮어쓰지 않음): %s/fold%d" % (run_name, fold))


def main():
    """전체 흐름: 인자 검사, 입력 검사, (--dry-run 이면 치환 결과만 출력) 출력 폴더 검사, fold 별 학습."""
    arguments = parse_arguments()
    folds, epochs = parse_folds_and_epochs(arguments)
    data_dir, start_dir = resolve_input_dirs(arguments)
    source = read_training_source()
    check_hyperparameters(source)
    common.check_manifest(REPO)
    if arguments.dry_run:
        dry_run(source, folds, arguments.seed, epochs, data_dir, start_dir)
        return
    train_dir = require_train_dir()
    run_name = arguments.run_name or ("smoke_seed%d" if arguments.smoke else "seed%d") % arguments.seed
    check_fold_dirs_free(train_dir, run_name, folds)
    context = {"source": source, "seed": arguments.seed, "epochs": epochs, "smoke": arguments.smoke,
               "data_dir": data_dir, "start_dir": start_dir, "train_dir": train_dir, "run_name": run_name}
    for fold in folds:
        train_one_fold(context, fold)


if __name__ == "__main__":
    main()
