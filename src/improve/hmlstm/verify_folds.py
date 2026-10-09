# verify_folds.py
# 역할: 2단계 2a 의 5개 폴드 대조(결정 I-30 D5). 폴드 1~5 마다 한 명씩 골라, 그 사람의 B1 blinks 파일에서 창을 만들어
#       (a) sequence_metadata_fold{k}_test.csv 의 seq_index 구간에 해당하는 BlinksTest_30_Fold{k}.npy 슬라이스와 비교하고(허용 오차 1e-9),
#       (b) 그 폴드의 시드 1 체크포인트로 점수를 계산해 0~10 클리핑 영상 평균을 video_predictions_seed{s}.csv 의 regression_mean 과 비교한다(허용 오차 1e-5).
#       영상은 처리하지 않는다. 읽기만 하고 아무 것도 쓰지 않는다. 경로는 모두 인자로 받는다.
# 사람 고르기: 알림 영상(클래스 0)이 있고 B1 평가 배열에 시퀀스가 있는 영상이 2개 이상인 사람 중 그 폴드 테스트 창 수가 가장 적은 사람
#       (같으면 피험자 id 문자열이 작은 쪽). variant_deferred 로 제외된 영상(10_1 등)은 메타데이터에 없으므로 자동으로 빠진다.
#       --subjects 3=32 처럼 폴드별로 사람을 지정할 수 있다. 창은 B1 과 같게 pad_short=True 로 만든다(B1 은 30개 미만 영상도 0 패딩 창을 만든다).
# 참고 출력: --ensemble-fold 의 사람 영상으로 5폴드 앙상블 평균과 그 폴드 단독 점수의 영상 평균 차이(대조 기준 아님). 0 이면 생략.
# 사용법: (저장소 폴더에서 conda activate ghoddoosian_tf1, set PYTHONPATH=src)
#           python -m improve.hmlstm.verify_folds --blink-dir <OUT_DIR>/batch --seq-dir <SEQ_DIR> --ckpt-root <CKPT_DIR> --eval-dir <CKPT_DIR>/b1_eval
# 종료 코드: 0 = 모든 폴드 일치, 1 = 불일치, 2 = 입력 오류
"""폴드 1~5 입력 수준(창)과 출력 수준(영상 평균 점수) 대조(Python 3.6, TF 1.8)."""
import argparse
import collections
import csv
import os
import sys

import numpy as np

from . import compare_rows as CR
from . import scoring as SC
from . import stage2 as S2
from . import windows as W

WINDOW_TOL = 1e-9
SCORE_TOL = 1e-5


def read_metadata(seq_dir, fold):
    """sequence_metadata_fold{k}_test.csv 의 행(dict) list."""
    path = os.path.join(seq_dir, "metadata", "sequence_metadata_fold%d_test.csv" % fold)
    with open(path, "r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def subjects_of(rows):
    """메타데이터 행을 {피험자: {영상 id: {class_label, seq(seq_index list)}}} 로 묶는다(등장 순서 유지)."""
    result = collections.OrderedDict()
    for row in rows:
        videos = result.setdefault(row["subject_id"], collections.OrderedDict())
        entry = videos.setdefault(row["video_id"], {"class_label": int(row["class_label"]), "seq": []})
        entry["seq"].append(int(row["seq_index"]))
    return result


def choose_subject(subjects, forced=None):
    """대조할 사람을 고른다. forced 가 있으면 그 사람(없으면 ValueError). 조건에 맞는 사람이 없으면 None."""
    if forced is not None:
        if forced not in subjects:
            raise ValueError("피험자 %s 는 이 폴드 테스트 메타데이터에 없습니다" % forced)
        return forced
    eligible = []
    for subject, videos in subjects.items():
        if any(info["class_label"] == 0 for info in videos.values()) and len(videos) >= 2:
            eligible.append((sum(len(info["seq"]) for info in videos.values()), subject))
    return min(eligible)[1] if eligible else None


def read_predictions(eval_dir, seed):
    """video_predictions_seed{s}.csv 를 {(폴드, 영상 id): regression_mean} 으로 읽는다."""
    path = os.path.join(eval_dir, "video_predictions_seed%d.csv" % seed)
    with open(path, "r", newline="", encoding="utf-8-sig") as handle:
        return dict(((int(row["fold"]), row["video_id"]), float(row["regression_mean"])) for row in csv.DictReader(handle))


def read_table(blink_dir, video_id):
    """영상의 B1 blinks 파일을 (n, 5) 배열로 읽는다. 파일이 없으면 None."""
    path = os.path.join(blink_dir, *video_id.split("/")) + ".txt"
    if not os.path.isfile(path):
        return None
    rows = CR.read_blink_file(path)
    return np.vstack(rows) if rows else np.zeros((0, 5))


def context_lines(ours, stored, window, step, feature, context):
    """첫 불일치 위치의 같은 창, 같은 특징에서 시간 방향 앞뒤 context 칸의 값을 재계산/저장 두 줄로 만든다."""
    low, high = max(0, step - context), min(ours.shape[1], step + context + 1)
    return ["    window=%d feature=%s step=%d (앞뒤 %d칸):" % (window, W.FEATURE_NAMES[feature], step, context),
            "      recomputed: " + " ".join("%.9f" % v for v in ours[window, low:high, feature]),
            "      stored    : " + " ".join("%.9f" % v for v in stored[window, low:high, feature])]


def check_fold(fold, subject, videos, scorer, ctx):
    """폴드 하나를 대조한다. 반환 dict: fold, subject, videos, windows, window_diff, score_diff, problems(list), details(list), windows_by_video."""
    out = {"fold": fold, "subject": subject, "videos": len(videos), "windows": 0, "window_diff": 0.0, "score_diff": 0.0,
           "problems": [], "details": [], "windows_by_video": collections.OrderedDict(), "video_means": {}}
    tables = dict((video_id, read_table(ctx["blink_dir"], video_id)) for video_id in videos)
    for video_id, table in tables.items():
        if table is None:
            out["problems"].append("B1 blinks 파일이 없습니다: %s" % video_id)
    alert_ids = [video_id for video_id, info in videos.items() if info["class_label"] == 0]
    if out["problems"] or len(alert_ids) != 1:
        out["problems"].append("알림 영상이 하나가 아닙니다" if len(alert_ids) != 1 else "입력 파일 문제")
        return out
    reference = W.compute_reference(tables[alert_ids[0]])
    if reference is None:
        out["problems"].append("알림 영상의 기준 통계를 쓸 수 없습니다(퇴행)")
        return out
    means, stds = reference
    test = ctx["test_arrays"][fold]
    for video_id, info in videos.items():
        windows, _ranges = W.video_windows(tables[video_id], means, stds, is_reference=info["class_label"] == 0, pad_short=True)
        out["windows_by_video"][video_id] = windows
        stored = np.asarray(test[info["seq"]])
        out["windows"] += len(windows)
        if len(windows) != len(stored):
            out["problems"].append("창 수 불일치: %s (재계산 %d, 저장 %d)" % (video_id, len(windows), len(stored)))
            continue
        diff = np.abs(windows - stored)
        top = float(diff.max()) if diff.size else 0.0
        out["window_diff"] = max(out["window_diff"], top)
        if top > ctx["window_tol"]:
            window, step, feature = [int(v) for v in np.argwhere(diff > ctx["window_tol"])[0]]
            out["problems"].append("창 값 불일치: %s 최대 차이 %.3e" % (video_id, top))
            out["details"].append("  첫 불일치 영상 %s" % video_id)
            out["details"].extend(context_lines(windows, stored, window, step, feature, ctx["context"]))
    for video_id, windows in out["windows_by_video"].items():
        scores = scorer.score(windows)
        mean = SC.video_mean(scores)
        key = (fold, video_id)
        if mean is None or key not in ctx["predictions"]:
            out["problems"].append("점수 비교 불가: %s (창 %d, 저장된 regression_mean %s)" % (video_id, len(windows), key in ctx["predictions"]))
            continue
        out["video_means"][video_id] = mean
        gap = abs(mean - ctx["predictions"][key])
        out["score_diff"] = max(out["score_diff"], gap)
        if gap > ctx["score_tol"]:
            out["problems"].append("점수 영상 평균 불일치: %s 계산 %.6f 저장 %.6f 차이 %.3e" % (video_id, mean, ctx["predictions"][key], gap))
            out["details"].append("  %s 앞쪽 창 점수: %s" % (video_id, [round(float(v), 6) for v in scores[:5]]))
    return out


def parse_subjects(text):
    """'3=32,1=5' 형식을 {폴드: 피험자 id 문자열} 로 바꾼다."""
    result = {}
    for part in [p for p in (text or "").split(",") if p.strip()]:
        fold, subject = part.split("=")
        result[int(fold)] = subject.strip()
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="폴드 1~5 입력 수준(창)과 출력 수준(영상 평균 점수) 대조(읽기 전용)")
    parser.add_argument("--blink-dir", dest="blink_dir", required=True, help="B1 blinks 폴더(<OUT_DIR>/batch)")
    parser.add_argument("--seq-dir", dest="seq_dir", required=True, help="시퀀스 산출물 루트(<SEQ_DIR>, npy/ 와 metadata/ 가 있음)")
    parser.add_argument("--ckpt-root", dest="ckpt_root", required=True, help="학습 출력 루트(<CKPT_DIR>)")
    parser.add_argument("--eval-dir", dest="eval_dir", required=True, help="video_predictions_seed{s}.csv 가 있는 폴더")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--stats", default=None, help="2단계 통계 JSON(기본: 패키지 안의 stage2_stats.json)")
    parser.add_argument("--subjects", default="", help="폴드별 피험자 지정, 예: 3=32")
    parser.add_argument("--ensemble-fold", dest="ensemble_fold", type=int, default=3, help="앙상블 참고 출력을 낼 폴드(0 이면 생략)")
    parser.add_argument("--bench-repeats", dest="bench_repeats", type=int, default=100)
    parser.add_argument("--context", type=int, default=3)
    return parser.parse_args(argv)


def print_table(results):
    print("%-5s %-8s %-7s %-7s %-22s %-22s %s" % ("fold", "subject", "videos", "windows", "max window diff", "max score-mean diff", "verdict"))
    for result in results:
        print("%-5d %-8s %-7d %-7d %-22s %-22s %s" % (result["fold"], result["subject"], result["videos"], result["windows"],
                                                      "%.3e" % result["window_diff"], "%.3e" % result["score_diff"],
                                                      "불일치" if result["problems"] else "일치"))


def main(argv=None):
    options = parse_args(argv)
    for path in (options.blink_dir, options.seq_dir, options.ckpt_root, options.eval_dir):
        if not os.path.isdir(path):
            print("폴더를 찾을 수 없습니다: %s" % os.path.basename(path))
            return 2
    try:
        forced = parse_subjects(options.subjects)
        stats = S2.load_stats(options.stats)
        predictions = read_predictions(options.eval_dir, options.seed)
        chosen = {}
        for fold in S2.FOLDS:
            subjects = subjects_of(read_metadata(options.seq_dir, fold))
            subject = choose_subject(subjects, forced.get(fold))
            if subject is None:
                print("fold %d: 조건에 맞는 사람이 없습니다" % fold)
                return 2
            chosen[fold] = (subject, subjects[subject])
        ctx = {"blink_dir": options.blink_dir, "seq_dir": options.seq_dir, "predictions": predictions, "window_tol": WINDOW_TOL,
               "score_tol": SCORE_TOL, "context": options.context,
               "test_arrays": dict((fold, np.load(os.path.join(options.seq_dir, "npy", "BlinksTest_30_Fold%d.npy" % fold), mmap_mode="r")) for fold in S2.FOLDS)}
        ensemble = SC.EnsembleScorer(options.ckpt_root, options.seed, stats=stats)
    except (OSError, ValueError, SC.ScoringError) as error:
        print("%s: %s" % (type(error).__name__, error))
        return 2
    try:
        results = [check_fold(fold, chosen[fold][0], chosen[fold][1], ensemble.scorers[fold], ctx) for fold in S2.FOLDS]
        print_table(results)
        for result in results:
            for problem in result["problems"]:
                print("fold %d 문제: %s" % (result["fold"], problem))
            for line in result["details"]:
                print(line)
        print("선택한 사람과 영상:")
        for result in results:
            print("  fold %d subject %s: %s" % (result["fold"], result["subject"], ", ".join("%s(창 %d)" % (v.split("/", 1)[1], len(w)) for v, w in result["windows_by_video"].items())))
        print("시간(ms): fold별 복원 / 창 1개 추론 중앙값(최댓값, %d회)" % options.bench_repeats)
        restore_total, infer_total = 0.0, 0.0
        for fold in S2.FOLDS:
            median, maximum = ensemble.scorers[fold].benchmark(options.bench_repeats)
            restore_total += ensemble.scorers[fold].restore_ms
            infer_total += median
            print("  fold %d: %.0f / %.2f (%.2f)" % (fold, ensemble.scorers[fold].restore_ms, median, maximum))
        print("  합계: 복원 %.0f ms, 5개 순차 추론 중앙값 합 %.2f ms" % (restore_total, infer_total))
        target = next((r for r in results if r["fold"] == options.ensemble_fold), None)
        if target is not None and target["windows_by_video"]:
            print("참고(대조 기준 아님): fold %d 사람 %s 의 영상 평균, 단독 대 5폴드 앙상블" % (target["fold"], target["subject"]))
            for video_id, windows in target["windows_by_video"].items():
                result = ensemble.score(windows)
                single = SC.video_mean(result.per_fold[target["fold"]])
                together = SC.video_mean(result.mean)
                means = [SC.video_mean(scores) for scores in result.per_fold.values()]
                print("  %s: fold %d 단독 %.6f, 앙상블 평균 %.6f, 차이 %.6f, 폴드별 영상 평균 최솟값 %.6f 최댓값 %.6f" % (
                    video_id, target["fold"], single, together, abs(single - together), min(means), max(means)))
    finally:
        ensemble.close()
    return 1 if any(result["problems"] for result in results) else 0


if __name__ == "__main__":
    sys.exit(main())
