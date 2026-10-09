# scoring.py
# 역할: B1 체크포인트(폴드 k 를 테스트로 둔 모델)로 HM-LSTM 점수를 계산한다(결정 I-27, I-30 D1~D3). 입력은 1단계 정규화가 끝난 창 [N, 30, 4]
#       (windows.py 의 결과)이고, 2단계 표준화(stage2.py 의 통계)를 적용한 뒤 phase_train=False, keep_p=1.0 으로 추론한다. 출력은 10*sigmoid 값 [N].
# Network 정의: third_party 의 Training.py 텍스트에서 첫 `load=` 줄 앞까지(정의부)만 메모리에서 exec 해서 얻는다. 저자 코드를 복사하지 않고
#       third_party 는 읽기만 한다. 정의부의 최상위 문장은 import, 함수/클래스 정의, 대입, docstring, print 호출만 허용한다(그 밖의 문장, 곧
#       학습/평가 루프나 Train() 호출이 있으면 ScoringError). exec 동안 sys.path 에 third_party 폴더를 넣었다가 되돌린다(HMSLSTM_Main 모듈 때문).
#       exec 동안 바이트코드 파일을 만들지 않고, 정의부가 찍는 print(TF 버전)는 버린다.
# 체크포인트: <CKPT_DIR>/b1_seed{s}/seed{s}/fold{k}/my_model{k}. tf.train.Saver.restore 만 한다(체크포인트 폴더에 쓰지 않는다).
#       폴드마다 tf.Graph 와 tf.Session 을 따로 만들고 close() 로 닫는다.
# 사용: from improve.hmlstm import scoring; with scoring.EnsembleScorer(ckpt_root) as scorer: result = scorer.score(windows)
"""HM-LSTM 점수 계산(Python 3.6, TF 1.8 호환). TensorFlow 는 점수기를 만들 때 처음 가져온다."""
import ast
import collections
import contextlib
import io
import os
import sys
import time

import numpy as np

from . import stage2 as S2

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
UPSTREAM_DIR = os.path.join(REPO, "third_party", "ghoddoosian")
TRAINING_PATH = os.path.join(UPSTREAM_DIR, "Training.py")
BOUNDARY_PREFIX = "load="           # 정의부와 학습 루프의 경계 줄(Training.py 407행)
# Training.py 503~506행의 호출 인자 중 Network 에 쓰이는 값
NETWORK_ARGS = dict(Pre_fc1_size=32, Post_fc1_size_per_layer=16, embb_size=16, embb_size2=16, Post_fc2_size=8,
                    hstate_size=[32, 32, 32, 32], num_layers=4, feature_size=4, step_size=30, output_size=1)
CLIP_LOW, CLIP_HIGH = 0.0, 10.0     # 영상 평균을 구하기 전의 클리핑(evaluate_baseline_checkpoints.py 의 방식)

EnsembleResult = collections.namedtuple("EnsembleResult", ["per_fold", "mean", "min", "max"])


class ScoringError(RuntimeError):
    """정의부 경계나 내용이 기대와 다르거나 체크포인트를 쓸 수 없을 때."""


def split_definitions(text):
    """Training.py 텍스트에서 첫 `load=` 줄 앞까지(정의부)를 돌려준다. 경계 줄이 없거나 정의부에 실행 문장이 있으면 ScoringError."""
    lines = text.split("\n")
    cuts = [index for index, line in enumerate(lines) if line.startswith(BOUNDARY_PREFIX)]
    if not cuts:
        raise ScoringError("정의부 경계 줄('%s'로 시작하는 줄)을 찾지 못했습니다" % BOUNDARY_PREFIX)
    head = "\n".join(lines[:cuts[0]]) + "\n"
    check_definitions_only(head)
    return head


def check_definitions_only(head):
    """정의부의 최상위 문장이 정의, import, 대입, docstring, print 호출뿐인지 정적으로 확인한다. 아니면 ScoringError."""
    tree = ast.parse(head)
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef, ast.Assign)):
            continue
        if isinstance(node, ast.Expr):
            value = node.value
            if isinstance(value, ast.Str):
                continue
            if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == "print":
                continue
        raise ScoringError("정의부에서 실행되는 문장이 있습니다(줄 %d, %s)" % (node.lineno, type(node).__name__))


@contextlib.contextmanager
def upstream_on_path(upstream_dir):
    """upstream_dir 를 sys.path 맨 앞에 넣고 바이트코드 쓰기를 끄고, 끝나면 둘 다 원래대로 되돌린다(upstream_dir 가 None 이면 아무 것도 안 함)."""
    saved_path = list(sys.path)
    saved_flag = sys.dont_write_bytecode
    if upstream_dir is not None:
        sys.path.insert(0, upstream_dir)
        sys.dont_write_bytecode = True
    try:
        yield
    finally:
        sys.dont_write_bytecode = saved_flag
        sys.path[:] = saved_path


def load_definitions(text, upstream_dir=None, label="Training.py[definitions]"):
    """정의부만 메모리에서 exec 하고 이름공간 dict 를 돌려준다. 정의부가 찍는 출력은 버린다."""
    head = split_definitions(text)
    namespace = {"__name__": "training_definitions"}
    with upstream_on_path(upstream_dir), contextlib.redirect_stdout(io.StringIO()):
        exec(compile(head, label, "exec"), namespace)
    return namespace


def load_network(training_path=None, upstream_dir=None):
    """third_party 의 Training.py 에서 Network 함수를 얻는다(파일은 읽기만 한다)."""
    path = training_path or TRAINING_PATH
    with open(path, "r", encoding="utf-8") as handle:
        text = handle.read()
    namespace = load_definitions(text, upstream_dir if upstream_dir is not None else os.path.dirname(os.path.abspath(path)))
    network = namespace.get("Network")
    if not callable(network):
        raise ScoringError("정의부에 Network 함수가 없습니다")
    return network


def checkpoint_prefix(ckpt_root, seed, fold):
    """체크포인트 접두어: <ckpt_root>/b1_seed{s}/seed{s}/fold{k}/my_model{k}."""
    return os.path.join(ckpt_root, "b1_seed%d" % seed, "seed%d" % seed, "fold%d" % fold, "my_model%d" % fold)


class FoldScorer(object):
    """폴드 하나의 체크포인트를 복원해 창에 점수를 매긴다. restore_ms 는 복원 시간, last_ms 는 마지막 score 호출의 추론 시간이다."""

    def __init__(self, fold, prefix, means, stds, network):
        import tensorflow as tf
        if not os.path.isfile(prefix + ".index"):
            raise FileNotFoundError("체크포인트를 찾을 수 없습니다: fold%d" % fold)
        self.fold = fold
        self._means, self._stds = np.asarray(means, dtype=np.float64), np.asarray(stds, dtype=np.float64)
        self._session = None
        self.last_ms = None
        graph = tf.Graph()
        with graph.as_default():
            self._input = tf.placeholder(tf.float32, shape=(None, None, 4), name="bacth_in")      # 원문 철자(Training.py 306행)
            self._keep = tf.placeholder(tf.float32)
            self._training = tf.placeholder(tf.bool, name="phase_train")
            self._output, _end, _concat = network(input=self._input, keep_p=self._keep, training=self._training, **NETWORK_ARGS)
            saver = tf.train.Saver()
            self._session = tf.Session(graph=graph)
            started = time.perf_counter()
            saver.restore(self._session, prefix)
            self.restore_ms = (time.perf_counter() - started) * 1000.0

    def _run(self, batch):
        return self._session.run(self._output, {self._input: batch, self._keep: 1.0, self._training: False})

    def score(self, windows):
        """창 [N, 30, 4] (1단계 정규화 완료)에 2단계 표준화를 적용해 점수 [N] float32 를 돌려준다. 창이 없으면 빈 배열."""
        array = np.asarray(windows)
        if array.ndim != 3 or array.shape[1:] != (30, 4):
            raise ValueError("창 모양이 (N, 30, 4) 가 아닙니다: %s" % (array.shape,))
        if array.shape[0] == 0:
            return np.zeros((0,), dtype=np.float32)
        batch = S2.standardize(array, self._means, self._stds).astype(np.float32)
        started = time.perf_counter()
        output = self._run(batch)
        self.last_ms = (time.perf_counter() - started) * 1000.0
        return output.reshape(-1)

    def benchmark(self, repeats=100):
        """창 1개(합성 입력)를 repeats 번 추론해 (중앙값 ms, 최댓값 ms)를 돌려준다. 앞의 3번은 예열이라 뺀다."""
        sample = np.random.RandomState(0).randn(1, 30, 4).astype(np.float32)
        for _ in range(3):
            self._run(sample)
        times = []
        for _ in range(repeats):
            started = time.perf_counter()
            self._run(sample)
            times.append((time.perf_counter() - started) * 1000.0)
        return float(np.median(times)), float(max(times))

    def close(self):
        if self._session is not None:
            self._session.close()
            self._session = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False


def aggregate(per_fold):
    """{폴드: 점수 배열 [N]} 에서 폴드 평균, 최솟값, 최댓값(창마다)을 구한다. 비었거나 길이가 다르면 오류."""
    if not per_fold:
        raise ScoringError("점수가 없습니다")
    ordered = [np.asarray(per_fold[fold], dtype=np.float64) for fold in sorted(per_fold)]
    if len(set(len(scores) for scores in ordered)) != 1:
        raise ValueError("폴드마다 점수 길이가 다릅니다")
    stacked = np.vstack(ordered)
    return EnsembleResult(per_fold=dict((fold, np.asarray(per_fold[fold])) for fold in sorted(per_fold)),
                          mean=stacked.mean(axis=0), min=stacked.min(axis=0), max=stacked.max(axis=0))


def video_mean(scores):
    """영상 하나의 점수를 0~10 으로 클리핑한 뒤 평균(evaluate_baseline_checkpoints.py video_results 와 같은 방식). 점수가 없으면 None."""
    scores = np.asarray(scores)
    if scores.size == 0:
        return None
    return float(np.mean(np.clip(scores, CLIP_LOW, CLIP_HIGH)))


class EnsembleScorer(object):
    """폴드 1~5 의 체크포인트를 모두 복원해 같은 창에 점수를 매기고 평균, 최솟값, 최댓값, 폴드별 값을 돌려준다(I-27 표시용)."""

    def __init__(self, ckpt_root, seed=1, folds=S2.FOLDS, stats=None, training_path=None, upstream_dir=None):
        stats = stats if stats is not None else S2.load_stats()
        network = load_network(training_path, upstream_dir)
        self.scorers = collections.OrderedDict()
        try:
            for fold in folds:
                means, stds = stats[fold]
                self.scorers[fold] = FoldScorer(fold, checkpoint_prefix(ckpt_root, seed, fold), means, stds, network)
        except Exception:
            self.close()
            raise

    def score(self, windows):
        return aggregate(dict((fold, scorer.score(windows)) for fold, scorer in self.scorers.items()))

    def close(self):
        for scorer in self.scorers.values():
            scorer.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False
