# score_service.py
# 역할: 라이브 게이지의 점수 프로세스(결정 설계 방침 (a): 2프로세스, 표준 입출력으로 JSON 한 줄씩). 추출과 화면 프로세스가 보낸 창을 5개 폴드 체크포인트로 점수화해 돌려준다.
#       TensorFlow 는 main 안에서 EnsembleScorer 를 만들 때 처음 불리며(scoring.py 가 TensorFlow 를 FoldScorer 안에서 가져온다) 이 파일의 맨 위에서는 가져오지 않는다.
# 프로토콜(표준 입력 한 줄 = 요청 하나, 표준 출력 한 줄 = 응답 하나, 순서대로):
#   요청  {"seq": 정수 또는 문자열, "windows": [[[4개 수] x 30] x N]}   N >= 1, 각 창은 1단계 정규화를 마친 (30, 4)
#   응답  {"seq": ..., "mean": [N개], "min": [N개], "max": [N개], "folds": [[폴드 수만큼] x N]}      폴드 순서는 폴드 번호 순
#   오류  {"seq": ... 또는 null, "error": "문장"}                       한 줄이 틀려도 서비스는 계속 돈다
#   준비  {"ready": true, "folds": 폴드 번호 목록}                        체크포인트 복원이 끝나면 한 번 쓴다(main 에서만)
# 표준 입력이 닫히면 종료한다(종료 코드 0). 표준 출력에는 프로토콜 줄만 쓴다: 점수 계산 중 print 와 TensorFlow 의 출력은 표준 에러로 돌린다. 로그도 표준 에러로 쓴다.
# 사용: PYTHONPATH=src python -m improve.hmlstm.score_service --ckpt-root <CKPT_DIR> [--seed 1] [--stats <stage2_stats.json>]
#       시험용: --fake 는 TensorFlow 없이 창의 평균을 점수로 쓰는 가짜 점수기를 쓴다(프로토콜 시험 전용).
"""점수 서비스(Python 3.6 호환)."""
import argparse
import contextlib
import json
import sys

import numpy as np

from . import scoring as SC
from . import stage2 as S2

WINDOW_SHAPE = (30, 4)


class ProtocolError(ValueError):
    """요청 한 줄이 프로토콜에 맞지 않을 때(응답은 오류 줄로 나간다)."""


class FakeScorer(object):
    """TensorFlow 없이 쓰는 시험용 점수기: 폴드 5개의 점수를 창 값의 평균에 폴드 번호를 더한 값으로 한다."""

    folds = tuple(S2.FOLDS)

    def score(self, windows):
        base = np.asarray(windows, dtype=np.float64).reshape(len(windows), -1).mean(axis=1)
        per_fold = dict((fold, base + 0.1 * fold) for fold in self.folds)
        return SC.aggregate(per_fold)

    def close(self):
        return None


def parse_request(line):
    """요청 한 줄을 (seq, windows 배열)로 바꾼다. 형식이 틀리면 ProtocolError(seq 를 알 수 있으면 e.seq 에 담는다)."""
    try:
        message = json.loads(line)
    except ValueError as error:
        raise ProtocolError("JSON 을 읽을 수 없습니다: %s" % error)
    if not isinstance(message, dict) or "seq" not in message or "windows" not in message:
        raise ProtocolError("seq 와 windows 가 필요합니다")
    seq = message["seq"]
    try:
        windows = np.asarray(message["windows"], dtype=np.float64)
    except (ValueError, TypeError) as error:
        failure = ProtocolError("windows 를 수 배열로 읽을 수 없습니다: %s" % error)
        failure.seq = seq
        raise failure
    if windows.ndim != 3 or windows.shape[0] < 1 or windows.shape[1:] != WINDOW_SHAPE:
        failure = ProtocolError("windows 모양이 (N, 30, 4) 이고 N >= 1 이어야 합니다: %s" % (windows.shape,))
        failure.seq = seq
        raise failure
    if not np.all(np.isfinite(windows)):
        failure = ProtocolError("windows 에 유한하지 않은 값이 있습니다")
        failure.seq = seq
        raise failure
    return seq, windows


def build_response(seq, result):
    """EnsembleResult 를 응답 dict 로 바꾼다: 평균, 최솟값, 최댓값은 창마다, folds 는 창마다 폴드 번호 순의 값."""
    folds = sorted(result.per_fold)
    count = len(result.mean)
    return {"seq": seq, "mean": [float(v) for v in result.mean], "min": [float(v) for v in result.min], "max": [float(v) for v in result.max],
            "folds": [[float(np.asarray(result.per_fold[fold])[index]) for fold in folds] for index in range(count)]}


def write_line(stream, message):
    stream.write(json.dumps(message, separators=(",", ":")) + "\n")
    stream.flush()


def serve(stdin, stdout, stderr, scorer):
    """요청 줄을 EOF 까지 읽어 응답 줄을 쓴다. 처리한 요청 수를 돌려준다. 점수 계산 중 표준 출력으로 가는 출력은 stderr 로 돌린다."""
    handled = 0
    for raw in stdin:
        line = raw.strip()
        if not line:
            continue
        handled += 1
        seq = None
        try:
            seq, windows = parse_request(line)
            with contextlib.redirect_stdout(stderr):
                result = scorer.score(windows)
            response = build_response(seq, result)
        except ProtocolError as error:
            response = {"seq": getattr(error, "seq", seq), "error": str(error)}
        except Exception as error:
            stderr.write("score error: %s: %s\n" % (type(error).__name__, error))
            response = {"seq": seq, "error": "%s: %s" % (type(error).__name__, error)}
        write_line(stdout, response)
    return handled


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="라이브 게이지 점수 서비스: 표준 입력의 JSON 한 줄(창)을 5개 폴드 체크포인트로 점수화해 표준 출력에 JSON 한 줄로 쓴다.")
    parser.add_argument("--ckpt-root", dest="ckpt_root", help="학습 출력 루트(<CKPT_DIR>, b1_seed{s}/seed{s}/fold{k} 아래 체크포인트)")
    parser.add_argument("--seed", type=int, default=1, help="체크포인트 시드(기본 1)")
    parser.add_argument("--stats", default=None, help="2단계 표준화 통계 JSON(기본: 저장소의 stage2_stats.json)")
    parser.add_argument("--fake", action="store_true", help="시험용 가짜 점수기를 쓴다(TensorFlow 없음)")
    return parser.parse_args(argv)


def main(argv=None, stdin=None, stdout=None, stderr=None):
    arguments = parse_args(argv)
    stdin = stdin if stdin is not None else sys.stdin
    protocol_out = stdout if stdout is not None else sys.stdout
    stderr = stderr if stderr is not None else sys.stderr
    if not arguments.fake and not arguments.ckpt_root:
        write_line(protocol_out, {"seq": None, "error": "--ckpt-root 가 필요합니다"})
        return 2
    scorer = None
    try:
        with contextlib.redirect_stdout(stderr):                      # 로드 중의 출력은 표준 에러로
            scorer = FakeScorer() if arguments.fake else SC.EnsembleScorer(arguments.ckpt_root, arguments.seed, stats=S2.load_stats(arguments.stats))
        write_line(protocol_out, {"ready": True, "folds": list(S2.FOLDS)})
        serve(stdin, protocol_out, stderr, scorer)
        return 0
    except Exception as error:
        stderr.write("score_service: %s: %s\n" % (type(error).__name__, error))
        write_line(protocol_out, {"seq": None, "error": "%s: %s" % (type(error).__name__, error)})
        return 1
    finally:
        if scorer is not None:
            scorer.close()


if __name__ == "__main__":
    sys.exit(main())
