# diagnose_video_level_baseline.py
# 역할: B1 기준선(LSTM 계열 모델)과 비교해 보기 위한 영상 단위 진단. 영상마다 정규화된 blink 4특징(freq x100, amplitude,
#       duration, velocity)의 평균과 표준편차 8개를 만들고, 단순 분류기 2개(LogisticRegression, RandomForest)와 대조군
#       (다수 클래스, train 라벨을 섞은 대조)를 B1 의 공식 fold 로 학습·평가한다.
#       이 진단은 재현이 아니라 한계 분석이다(논문과 저자 코드가 하지 않은 실험). 설정은 하나만 쓰고 성능을 보고 바꾸지 않는다.
# 사용법: 저장소 루트에서 conda 환경 ghoddoosian_tf1 의 python 으로 실행한다(sklearn 0.19.2).
#           python src/repro/diagnose_video_level_baseline.py --out-dir <집계 폴더>
#         옵션: --data-dir (기본 SEQ_OUT_DIR/npy), --metadata-dir (기본 SEQ_OUT_DIR/metadata)
# 입력: SEQ_OUT_DIR 의 npy 와 메타데이터(읽기만 한다). 경로는 인자 또는 SEQ_OUT_DIR 로만 받는다.
# 출력(집계 폴더, 저장소 밖, 입력 폴더 밖): video_features.csv, video_predictions.csv, fold_va.csv, class_accuracy.csv,
#       confusion_<분류기>.csv, diagnostic_summary.txt. 이미 있는 출력 파일은 덮어쓰지 않고 중단한다.
"""영상 단위 진단 기준선 도구(Python 3.6 호환, sklearn 0.19.2, TensorFlow 를 쓰지 않음).

영상의 blink 행 복원: 시퀀스 1개 영상은 앞쪽 0 패딩을 뺀 마지막 n_blinks_used 행, 시퀀스 N>=2 영상은 첫 윈도우 30행과
이후 각 윈도우의 마지막 2행(길이 28+2N). 홀수 n_blinks_used 인 N>=2 영상은 마지막 blink 1개가 어떤 윈도우에도 들어가지 않아
복원되지 않는다. 영상은 각자 자기 fold 의 test 배열에 연속으로 한 번씩 있으므로 특징은 test 배열에서 만들고, train 배열은
메타데이터 정합성 점검에만 쓴다.
"""
import sys

sys.dont_write_bytecode = True

import argparse
import csv
import os
import warnings

warnings.filterwarnings("ignore")
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import common
import summarize_training as ST

REPO = ST.REPO
FOLDS = ST.FOLDS
CLASS_POOL = (0, 5, 10)
FEATURE_COLUMNS = ["mean_freq", "mean_amp", "mean_dur", "mean_vel", "std_freq", "std_amp", "std_dur", "std_vel"]
CLASSIFIERS = ("logreg", "rf", "majority")
CONTROLS = ("logreg", "rf")
SHUFFLE_REPEATS = 5
SHUFFLE_SEED = 0
CONTROL_LIMIT = 0.40
B1_REFERENCE = {"VA": 0.586, "class_accuracy": (0.731, 0.327, 0.698)}  # docs/REPRODUCTION_LOG.md 에 적힌 B1 값(시드 평균 5-fold VA, 마지막 epoch 클래스별 정확도)


def parse_arguments():
    """명령줄 옵션을 해석한다.

    반환: argparse.Namespace (out_dir, data_dir, meta_dir)
    """
    parser = argparse.ArgumentParser(description="영상 단위 진단 기준선(재현이 아닌 한계 분석)")
    parser.add_argument("--out-dir", dest="out_dir", required=True, help="집계 폴더(저장소 밖, 입력 폴더 밖)")
    parser.add_argument("--data-dir", dest="data_dir", default=None, help="npy 폴더(기본 SEQ_OUT_DIR/npy)")
    parser.add_argument("--metadata-dir", dest="meta_dir", default=None, help="메타데이터 폴더(기본 SEQ_OUT_DIR/metadata)")
    return parser.parse_args()


def validate_paths(arguments):
    """입력 폴더와 집계 폴더를 검사한다(집계 폴더는 저장소 밖이고 입력 폴더와 겹치지 않아야 한다).

    입력: arguments - parse_arguments 결과
    반환: (npy 폴더, 메타데이터 폴더, 집계 폴더) 실제 경로
    """
    seq_raw = os.environ.get("SEQ_OUT_DIR")
    if (arguments.data_dir is None or arguments.meta_dir is None) and not seq_raw:
        sys.exit("--data-dir 와 --metadata-dir 를 주거나 환경변수 SEQ_OUT_DIR 를 설정하세요.")
    data_dir = os.path.realpath(arguments.data_dir or os.path.join(seq_raw, "npy"))
    meta_dir = os.path.realpath(arguments.meta_dir or os.path.join(seq_raw, "metadata"))
    out_dir = os.path.realpath(arguments.out_dir)
    if ST.is_inside(out_dir, REPO) or any(ST.is_inside(out_dir, d) or ST.is_inside(d, out_dir) for d in (data_dir, meta_dir)):
        sys.exit("집계 폴더는 저장소 밖이어야 하고 입력 폴더와 겹치면 안 됩니다.")
    return data_dir, meta_dir, out_dir


def read_csv_rows(path):
    """CSV 를 dict list 로 읽는다(BOM 허용).

    입력: path - CSV 경로
    반환: dict list
    """
    with open(path, "r", newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def snapshot_hashes(folders):
    """입력 폴더 안 모든 파일의 SHA-256 을 모은다(실행 전후 불변 확인용).

    입력: folders - 폴더 list
    반환: {폴더 이름/상대 경로: 해시}
    """
    hashes = {}
    for folder in folders:
        for current, _subfolders, names in os.walk(folder):
            for name in sorted(names):
                path = os.path.join(current, name)
                key = "%s/%s" % (os.path.basename(folder), os.path.relpath(path, folder).replace("\\", "/"))
                hashes[key] = common.sha256_of_file(path)
    return hashes


def group_metadata(rows):
    """메타데이터 행을 영상별로(첫 등장 순서) 묶는다.

    입력: rows - 테스트 메타데이터 dict list
    반환: (영상 식별자 list, {영상 식별자: 행 list})
    """
    order, groups = [], {}
    for row in rows:
        if row["video_id"] not in groups:
            groups[row["video_id"]] = []
            order.append(row["video_id"])
        groups[row["video_id"]].append(row)
    return order, groups


def reconstruct_blinks(sequences, n_used):
    """한 영상의 시퀀스에서 blink 행을 복원한다(정규화된 4특징).

    입력: sequences - (N, 30, 4) 배열, n_used - 메타데이터의 n_blinks_used
    반환: (n, 4) 배열. 행 수나 패딩이 메타데이터와 맞지 않으면 RuntimeError
    """
    sequences = np.asarray(sequences)
    if len(sequences) == 1:
        if n_used < 30 and np.any(sequences[0][:30 - n_used] != 0):
            raise RuntimeError("패딩 행이 0 이 아닙니다.")
        return sequences[0][-n_used:]
    rows = np.concatenate([sequences[0], sequences[1:, -2:, :].reshape(-1, 4)], axis=0)
    if len(rows) not in (n_used, n_used - 1):
        raise RuntimeError("복원한 blink 행 %d개가 n_blinks_used %d 와 맞지 않습니다." % (len(rows), n_used))
    return rows


def build_fold_videos(fold, x_test, y_test, rows):
    """한 fold 의 테스트 배열과 메타데이터에서 영상별 blink 행을 복원한다(메타데이터 정합성을 검증).

    입력: fold, x_test, y_test - 테스트 배열, rows - 테스트 메타데이터 dict list
    반환: (영상 dict list, {(영상 식별자, seq_in_video): seq_index})
    """
    order, groups = group_metadata(rows)
    videos, lookup = [], {}
    for name in order:
        items = groups[name]
        indices = [int(item["seq_index"]) for item in items]
        positions = [int(item["seq_in_video"]) for item in items]
        if indices != list(range(indices[0], indices[0] + len(items))) or positions != list(range(len(items))):
            raise RuntimeError("Fold%d %s: 시퀀스 순서가 연속이 아닙니다." % (fold, name))
        label, n_used = int(items[0]["class_label"]), int(items[0]["n_blinks_used"])
        if int(y_test[indices[0]][0]) != label or set(int(i["class_label"]) for i in items) != set([label]):
            raise RuntimeError("Fold%d %s: 클래스가 테스트 라벨과 다릅니다." % (fold, name))
        blink_rows = reconstruct_blinks(x_test[indices[0]:indices[-1] + 1], n_used)
        videos.append({"video_id": name, "fold": fold, "label": label, "n_seq": len(items), "rows": blink_rows})
        for position, index in zip(positions, indices):
            lookup[(name, position)] = index
    return videos, lookup


def load_all(data_dir, meta_dir):
    """모든 fold 의 영상을 복원한다(test 배열 기준).

    입력: data_dir, meta_dir
    반환: (영상 list, {fold: 테스트 배열 mmap}, {fold: lookup})
    """
    videos, tests, lookups = [], {}, {}
    for fold in FOLDS:
        tests[fold] = np.load(os.path.join(data_dir, "BlinksTest_30_Fold%d.npy" % fold), mmap_mode="r")
        y_test = np.load(os.path.join(data_dir, "LabelsTest_30_Fold%d.npy" % fold), mmap_mode="r")
        rows = read_csv_rows(os.path.join(meta_dir, "sequence_metadata_fold%d_test.csv" % fold))
        fold_videos, lookups[fold] = build_fold_videos(fold, tests[fold], y_test, rows)
        videos.extend(fold_videos)
    return videos, tests, lookups


def check_train_alignment(data_dir, meta_dir, videos, tests, lookups):
    """train 메타데이터 행이 가리키는 시퀀스가 test 배열의 같은 (영상, seq_in_video) 시퀀스와 같은지, train 영상 집합이 맞는지 점검한다.

    입력: data_dir, meta_dir, videos, tests - fold별 test 배열, lookups - fold별 (영상, seq_in_video) -> 행 번호
    반환: (문제 list, 대조한 시퀀스 수)
    """
    home = dict((video["video_id"], video["fold"]) for video in videos)
    problems, checked = [], 0
    for fold in FOLDS:
        x_train = np.load(os.path.join(data_dir, "Blinks_30_Fold%d.npy" % fold), mmap_mode="r")
        rows = read_csv_rows(os.path.join(meta_dir, "sequence_metadata_fold%d_train.csv" % fold))
        if set(row["video_id"] for row in rows) != set(name for name, home_fold in home.items() if home_fold != fold):
            problems.append("Fold%d train 영상 집합이 다른 fold 영상 전체와 다름" % fold)
        for row in rows:
            name, position = row["video_id"], int(row["seq_in_video"])
            source = tests[home[name]][lookups[home[name]][(name, position)]]
            checked += 1
            if not np.array_equal(x_train[int(row["seq_index"])], source):
                problems.append("Fold%d train 행 %s 가 test 시퀀스와 다름" % (fold, row["seq_index"]))
    return problems, checked


def video_features(blink_rows):
    """영상의 blink 행에서 4특징의 평균과 표준편차(모집단, ddof=0) 8개를 만든다.

    입력: blink_rows - (n, 4) 배열
    반환: 길이 8 list (평균 4개, 표준편차 4개)
    """
    return [float(v) for v in blink_rows.mean(axis=0)] + [float(v) for v in blink_rows.std(axis=0)]


def predict_logreg(train_x, train_y, test_x):
    """train 영상으로만 fit 한 StandardScaler 와 기본 LogisticRegression 으로 예측한다(sklearn 0.19.2 기본값: C=1.0, ovr, liblinear).

    입력: train_x, train_y, test_x
    반환: 예측 클래스 배열
    """
    scaler = StandardScaler().fit(train_x)
    model = LogisticRegression().fit(scaler.transform(train_x), train_y)
    return model.predict(scaler.transform(test_x))


def predict_forest(train_x, train_y, test_x):
    """RandomForestClassifier(n_estimators=200, random_state=0)로 예측한다(특징 표준화 없음).

    입력: train_x, train_y, test_x
    반환: 예측 클래스 배열
    """
    model = RandomForestClassifier(n_estimators=200, random_state=0).fit(train_x, train_y)
    return model.predict(test_x)


def predict_majority(train_y, count):
    """train 영상에서 가장 많은 클래스(동률이면 작은 클래스 번호)를 count 개 영상에 대해 예측한다.

    입력: train_y - train 라벨, count - test 영상 수
    반환: 예측 클래스 배열
    """
    best = max(CLASS_POOL, key=lambda label: (int(np.sum(train_y == label)), -label))
    return np.full(count, best, dtype=train_y.dtype)


def make_permutations(fold_of):
    """fold 1~5 x 반복 5회의 train 라벨 순열을 RandomState(0)으로 순서대로 한 번 만든다(두 분류기가 같은 순열을 쓴다).

    입력: fold_of - 영상별 fold 배열
    반환: {fold: 순열 list}
    """
    rng = np.random.RandomState(SHUFFLE_SEED)
    permutations = {}
    for fold in FOLDS:
        n_train = int(np.sum(fold_of != fold))
        permutations[fold] = [rng.permutation(n_train) for _ in range(SHUFFLE_REPEATS)]
    return permutations


def fold_predictions(data, fold, permutations):
    """한 fold 의 분류기 예측과 라벨을 섞은 대조의 예측을 만든다.

    입력: data - dict(x, y, fold_of), fold, permutations - make_permutations 결과
    반환: (test 인덱스, {분류기: 예측}, {분류기: 반복별 예측 list})
    """
    test = np.where(data["fold_of"] == fold)[0]
    train = np.where(data["fold_of"] != fold)[0]
    x_train, y_train, x_test = data["x"][train], data["y"][train], data["x"][test]
    predictions = {"logreg": predict_logreg(x_train, y_train, x_test), "rf": predict_forest(x_train, y_train, x_test),
                   "majority": predict_majority(y_train, len(test))}
    controls = {"logreg": [], "rf": []}
    for permutation in permutations[fold]:
        controls["logreg"].append(predict_logreg(x_train, y_train[permutation], x_test))
        controls["rf"].append(predict_forest(x_train, y_train[permutation], x_test))
    return test, predictions, controls


def confusion_of(true, pred):
    """3x3 혼동행렬(행=정답 0/5/10, 열=예측 0/5/10)을 만든다.

    입력: true, pred - 클래스 배열
    반환: 3x3 정수 list
    """
    matrix = [[0, 0, 0] for _ in CLASS_POOL]
    for actual, guess in zip(true, pred):
        matrix[CLASS_POOL.index(int(actual))][CLASS_POOL.index(int(guess))] += 1
    return matrix


def record(result, fold, true, prediction_list):
    """한 fold 의 예측(반복이 있으면 여러 개)으로 VA(반복 평균), 맞은 영상 수(반복 평균), 혼동행렬(반복 합산)을 기록한다.

    입력: result - 결과 dict, fold, true - 정답, prediction_list - 예측 배열 list
    반환: 없음
    """
    correct = [int(np.sum(true == pred)) for pred in prediction_list]
    result["fold_va"][fold] = float(np.mean(correct)) / len(true)
    result["fold_correct"][fold] = float(np.mean(correct))
    result["n_test"][fold] = len(true)
    for pred in prediction_list:
        result["matrix"] = ST.add_matrices(result["matrix"], confusion_of(true, pred))


def evaluate_all(data, permutations):
    """모든 fold 를 평가한다(분류기 3개와 라벨을 섞은 대조 2개).

    입력: data, permutations
    반환: (결과 dict {이름: dict(fold_va, fold_correct, n_test, matrix)}, 영상별 예측 {인덱스: {분류기: 예측}})
    """
    names = list(CLASSIFIERS) + [name + "_shuffled" for name in CONTROLS]
    results = dict((name, {"fold_va": {}, "fold_correct": {}, "n_test": {}, "matrix": [[0, 0, 0] for _ in CLASS_POOL]}) for name in names)
    per_video = {}
    for fold in FOLDS:
        test, predictions, controls = fold_predictions(data, fold, permutations)
        true = data["y"][test]
        for name in CLASSIFIERS:
            record(results[name], fold, true, [predictions[name]])
        for name in CONTROLS:
            record(results[name + "_shuffled"], fold, true, controls[name])
        for position, index in enumerate(test):
            per_video[int(index)] = dict((name, int(predictions[name][position])) for name in CLASSIFIERS)
    return results, per_video


def mean_va(result):
    """fold VA 의 가중 없는 평균(B1 집계와 같은 방식)을 구한다.

    입력: result - 결과 dict
    반환: float
    """
    return float(np.mean([result["fold_va"][fold] for fold in FOLDS]))


def write_csv(path, columns, rows):
    """CSV 를 새 파일로만 쓴다(이미 있으면 예외).

    입력: path, columns, rows
    반환: 없음
    """
    with open(path, "x", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def feature_rows(videos, features):
    """video_features.csv 행을 만든다.

    입력: videos, features - (영상 수, 8) 배열
    반환: dict list
    """
    rows = []
    for video, values in zip(videos, features):
        row = {"video_id": video["video_id"], "fold": video["fold"], "class": video["label"], "n_seq": video["n_seq"],
               "n_blink_rows": len(video["rows"])}
        row.update(dict((name, "%.6f" % value) for name, value in zip(FEATURE_COLUMNS, values)))
        rows.append(row)
    return rows


def prediction_rows(videos, per_video):
    """video_predictions.csv 행(B1 의 video_predictions_seed<s>.csv 와 같은 video_id)을 만든다.

    입력: videos, per_video - {인덱스: {분류기: 예측}}
    반환: dict list
    """
    return [{"fold": video["fold"], "video_id": video["video_id"], "true_class": video["label"],
             "pred_logreg": per_video[index]["logreg"], "pred_rf": per_video[index]["rf"],
             "pred_majority": per_video[index]["majority"]} for index, video in enumerate(videos)]


def result_rows(results):
    """fold_va.csv 와 class_accuracy.csv 행을 만든다.

    입력: results - evaluate_all 결과
    반환: (fold_va 행 list, class_accuracy 행 list)
    """
    va_rows, class_rows = [], []
    for name, result in results.items():
        for fold in FOLDS:
            va_rows.append({"classifier": name, "fold": fold, "n_videos": result["n_test"][fold],
                            "correct": "%.1f" % result["fold_correct"][fold], "VA": "%.6f" % result["fold_va"][fold]})
        va_rows.append({"classifier": name, "fold": "mean", "n_videos": sum(result["n_test"].values()), "correct": "", "VA": "%.6f" % mean_va(result)})
        accuracy = ST.class_accuracy(result["matrix"])
        for index, label in enumerate(CLASS_POOL):
            class_rows.append({"classifier": name, "class": label, "n_videos": sum(result["matrix"][index]),
                               "correct": result["matrix"][index][index], "accuracy": "%.6f" % accuracy[index]})
    return va_rows, class_rows


def summary_lines(results, alignment, hash_ok, n_videos):
    """diagnostic_summary.txt 줄(성격, 정합성 점검, fold별 VA, 혼동행렬, 대조 점검, B1 인용)을 만든다.

    입력: results, alignment - (문제 list, 대조 수), hash_ok - 입력 해시 불변 여부, n_videos - 영상 수
    반환: 문자열 list
    """
    lines = ["== 영상 단위 진단 기준선 (재현이 아니라 한계 분석, 설정 하나만 사용) ==",
             "정합성: train 메타데이터 행 %d개 대조, 문제 %s | 입력 파일 해시 불변: %s | 영상 %d개" % (
                 alignment[1], alignment[0] or "없음", hash_ok, n_videos),
             "분류기: logreg=StandardScaler(train 영상으로만 fit)+LogisticRegression()(sklearn 0.19.2 기본값 C=1.0, ovr, liblinear), "
             "rf=RandomForestClassifier(n_estimators=200, random_state=0), majority=train 최빈 클래스", "",
             "fold별 VA(영상 단위; shuffled 는 라벨을 섞은 대조 5회 평균)", "| 분류기 | Fold1 | Fold2 | Fold3 | Fold4 | Fold5 | 5-fold 평균 |",
             "|---|---|---|---|---|---|---|"]
    for name, result in results.items():
        lines.append("| %s | %s | %.3f |" % (name, " | ".join("%.3f" % result["fold_va"][fold] for fold in FOLDS), mean_va(result)))
    lines.append("")
    for name, result in results.items():
        lines.extend(ST.format_confusion("%s(5-fold 합산)" % name, result["matrix"]))
    controls = dict((name, mean_va(results[name + "_shuffled"])) for name in CONTROLS)
    failed = [name for name, value in controls.items() if value > CONTROL_LIMIT]
    verdict = "누출 가능성, 중단하고 보고" if failed else "통과"
    lines += ["", "대조 점검(라벨을 섞은 5-fold 평균 VA <= %.2f): %s -> %s" % (
        CONTROL_LIMIT, ", ".join("%s %.3f" % item for item in controls.items()), verdict),
        "B1 인용(docs/REPRODUCTION_LOG.md): 시드 평균 5-fold VA %.3f, 마지막 epoch 클래스별 정확도 0 %.3f, 5 %.3f, 10 %.3f" % (
            (B1_REFERENCE["VA"],) + B1_REFERENCE["class_accuracy"])]
    return lines


def output_names():
    """쓸 출력 파일 이름 목록을 만든다.

    반환: 파일 이름 list
    """
    names = ["video_features.csv", "video_predictions.csv", "fold_va.csv", "class_accuracy.csv", "diagnostic_summary.txt"]
    return names + ["confusion_%s.csv" % name for name in list(CLASSIFIERS) + [c + "_shuffled" for c in CONTROLS]]


def confusion_csv_rows(matrix):
    """혼동행렬 CSV 행(정답 클래스마다 한 줄)을 만든다.

    입력: matrix - 3x3 list
    반환: dict list
    """
    accuracy = ST.class_accuracy(matrix)
    return [{"true_label": CLASS_POOL[i], "pred_0": matrix[i][0], "pred_5": matrix[i][1], "pred_10": matrix[i][2],
             "n_videos": sum(matrix[i]), "class_accuracy": "%.6f" % accuracy[i]} for i in range(3)]


def write_outputs(out_dir, videos, features, results, per_video, lines):
    """집계 폴더에 모든 출력을 새로 쓴다.

    입력: out_dir, videos, features, results, per_video, lines
    반환: 없음
    """
    os.makedirs(out_dir, exist_ok=True)
    feature_columns = ["video_id", "fold", "class", "n_seq", "n_blink_rows"] + FEATURE_COLUMNS
    write_csv(os.path.join(out_dir, "video_features.csv"), feature_columns, feature_rows(videos, features))
    prediction_columns = ["fold", "video_id", "true_class", "pred_logreg", "pred_rf", "pred_majority"]
    write_csv(os.path.join(out_dir, "video_predictions.csv"), prediction_columns, prediction_rows(videos, per_video))
    va_rows, class_rows = result_rows(results)
    write_csv(os.path.join(out_dir, "fold_va.csv"), ["classifier", "fold", "n_videos", "correct", "VA"], va_rows)
    write_csv(os.path.join(out_dir, "class_accuracy.csv"), ["classifier", "class", "n_videos", "correct", "accuracy"], class_rows)
    confusion_columns = ["true_label", "pred_0", "pred_5", "pred_10", "n_videos", "class_accuracy"]
    for name, result in results.items():
        write_csv(os.path.join(out_dir, "confusion_%s.csv" % name), confusion_columns, confusion_csv_rows(result["matrix"]))
    with open(os.path.join(out_dir, "diagnostic_summary.txt"), "x", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")


def main():
    """전체 흐름: 입력 검사, 영상 복원, train 정합성 점검, 평가, 입력 해시 대조, 집계 폴더에 저장."""
    arguments = parse_arguments()
    data_dir, meta_dir, out_dir = validate_paths(arguments)
    existing = [name for name in output_names() if os.path.exists(os.path.join(out_dir, name))]
    if existing:
        sys.exit("이미 있는 출력 파일이 있어 덮어쓰지 않고 중단합니다: %s" % existing)
    before = snapshot_hashes([data_dir, meta_dir])
    videos, tests, lookups = load_all(data_dir, meta_dir)
    alignment = check_train_alignment(data_dir, meta_dir, videos, tests, lookups)
    if alignment[0]:
        sys.exit("train 메타데이터 정합성 점검 실패: %s" % alignment[0][:5])
    features = np.array([video_features(video["rows"]) for video in videos])
    data = {"x": features, "y": np.array([video["label"] for video in videos]), "fold_of": np.array([video["fold"] for video in videos])}
    results, per_video = evaluate_all(data, make_permutations(data["fold_of"]))
    hash_ok = snapshot_hashes([data_dir, meta_dir]) == before
    lines = summary_lines(results, alignment, hash_ok, len(videos))
    print("\n".join(lines))
    write_outputs(out_dir, videos, features, results, per_video, lines)
    print("\n집계 폴더에 썼습니다.")


if __name__ == "__main__":
    main()
