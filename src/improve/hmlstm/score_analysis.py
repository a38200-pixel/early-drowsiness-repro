# score_analysis.py
# 역할: 오프라인 점수 분석(결정 I-27 재검토와 포트폴리오 서술의 근거 자료, DECISIONS.md I-W-30). B1 시드 1의 폴드 k 체크포인트로 그 폴드의 검증용(test) 영상만
#       채점해 클래스(0=깨어 있음, 5=낮은 졸음, 10=졸음)별 창 점수 분포, 영상 단위 요약, 점수 기반 알람 영상 비율, 영상 단위 구분 지표(AUC)를 표준출력으로 낸다.
#       앙상블은 쓰지 않는다: 폴드 k 의 검증 영상은 폴드 k 체크포인트(시드 1)로만 채점한다. 라이브 파이프라인과의 일치 여부는 이 분석의 범위가 아니다.
# 입력: verify_folds.py 와 같다. 폴드별로 B1 blinks 파일에서 창을 만들고(windows.video_windows, pad_short=True: B1 과 같이 30개 미만 영상도 0 패딩 창 1개),
#       저장된 BlinksTest_30_Fold{k}.npy 의 해당 구간과 대조한 뒤(허용 1e-9), scoring.FoldScorer 로 채점하고 영상 점수 평균을 video_predictions_seed{s}.csv 와 대조한다(허용 1e-5).
#       variant_deferred 등 B1 평가 배열에 없는 영상은 입력 메타데이터에 없으므로 자동으로 빠진다. 기준 구간을 만들 수 없는 사람도 B1 배열에 없으며 exclusions.csv 로 보고한다.
# 출력: 표준출력만. 파일을 쓰지 않는다. 프레임이나 이미지를 만들지 않는다.
# 용어: 창 단위 라벨은 영상 라벨의 복사이다. 패딩 창은 영상의 blink 수가 30 미만이어서 앞쪽을 0 으로 채운 창이다(blink 수 30 은 패딩이 없다).
# 알람 규칙(시뮬레이션): 점수 >= T 가 연속 m 개 창이면 그 영상은 알람이다. "첫 알람 창"은 그 연속이 완성되는 창(연속의 m 번째 창)의 1부터 센 번호이다.
#       blink 수 환산은 그 창이 사용한 blink 수(끝 blink 번호 + 1)이며 패딩 없는 창에서는 30 + 2 × (창 번호 - 1)과 같다. 창에는 시간이 없으므로 시간 환산은 하지 않는다.
# 사용법: (저장소 폴더에서 conda activate ghoddoosian_tf1, set PYTHONPATH=src)
#           python -m improve.hmlstm.score_analysis --blink-dir <OUT_DIR>/batch --seq-dir <SEQ_DIR> --ckpt-root <CKPT_DIR> --eval-dir <CKPT_DIR>/b1_eval
"""오프라인 점수 분석(Python 3.6, 순수 계산부는 numpy 만 쓴다. TensorFlow 는 채점할 때만 가져온다)."""
import argparse
import collections
import csv
import os
import sys
import time

import numpy as np

from . import scoring as SC
from . import stage2 as S2
from . import verify_folds as VF
from . import windows as W

THRESHOLDS = (5.0, 6.0, 6.68, 7.0, 8.0, 9.0)
RUN_LENGTHS = (1, 2, 3, 5)
PERCENTILES = (5, 25, 50, 75, 95)
CLASSES = (0, 5, 10)
PAIRS = ((10, 0), (5, 0), (10, 5))                 # (양성 클래스, 음성 클래스)
BAND_EDGES = (3.34, 6.68)
WINDOW_TOLERANCE = 1e-9
MEAN_TOLERANCE = 1e-5
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 0

VideoRecord = collections.namedtuple("VideoRecord", ["video_id", "person", "fold", "label", "scores", "padded", "blinks"])


# ----- 순수 계산(TensorFlow 불필요) -----
def run_alarm_window(scores, threshold, run_length):
    """점수 >= threshold 가 연속 run_length 개인 곳이 처음 완성되는 창의 번호(1부터). 없으면 None. 경계값(= threshold)은 포함한다."""
    streak = 0
    for index, value in enumerate(scores):
        if value >= threshold:
            streak += 1
            if streak >= run_length:
                return index + 1
        else:
            streak = 0
    return None


def describe(values):
    """개수, 평균, 표준편차(ddof=0), 백분위(5, 25, 50, 75, 95; numpy 선형 보간). 값이 없으면 None 으로 채운다."""
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"n": 0, "mean": None, "std": None, "pct": dict((p, None) for p in PERCENTILES)}
    return {"n": int(array.size), "mean": float(array.mean()), "std": float(array.std()),
            "pct": dict((p, float(np.percentile(array, p))) for p in PERCENTILES)}


def band_fractions(scores):
    """점수 구간 0~3.34, 3.34~6.68, 6.68~10 의 비율(라벨 경계와 같은 `// 3.34` 규칙: 3.34 는 둘째, 6.68 은 셋째 구간). 값이 없으면 None."""
    array = np.asarray(scores, dtype=np.float64)
    if array.size == 0:
        return None
    index = np.clip((array // BAND_EDGES[0]).astype(int), 0, 2)
    return [float(np.mean(index == band)) for band in range(3)]


def fraction_at_least(values, threshold):
    """values 중 threshold 이상인 비율. 값이 없으면 None."""
    array = np.asarray(values, dtype=np.float64)
    return None if array.size == 0 else float(np.mean(array >= threshold))


def median_or_none(values):
    values = [v for v in values if v is not None]
    return None if len(values) == 0 else float(np.median(np.asarray(values, dtype=np.float64)))


def drop_padded(record):
    """패딩 창만 걸러낸 기록을 돌려준다(점수, 패딩 표시, blink 수의 같은 위치를 함께 거른다)."""
    keep = ~np.asarray(record.padded, dtype=bool)
    return record._replace(scores=np.asarray(record.scores)[keep], padded=np.asarray(record.padded)[keep], blinks=np.asarray(record.blinks)[keep])


def make_edition(records, include_padded):
    """패딩 창 포함 판(그대로) 또는 제외 판(패딩 창을 걸러냄). 창이 하나도 남지 않은 영상은 뺀다."""
    chosen = records if include_padded else [drop_padded(record) for record in records]
    return [record for record in chosen if len(record.scores) > 0]


def auc(positive, negative):
    """양성 값이 음성 값보다 클 확률(동점은 0.5)에 해당하는 AUC(Mann-Whitney). 한쪽이 비면 None."""
    pos, neg = np.asarray(positive, dtype=np.float64), np.asarray(negative, dtype=np.float64)
    if pos.size == 0 or neg.size == 0:
        return None
    greater = float(np.sum(pos[:, None] > neg[None, :]))
    ties = float(np.sum(pos[:, None] == neg[None, :]))
    return (greater + 0.5 * ties) / (pos.size * neg.size)


def person_bootstrap_auc(rows, positive_label, negative_label, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED):
    """사람 단위 부트스트랩 AUC. rows 는 (사람, 라벨, 값) 목록이고 사람을 복원추출하며 한 사람의 영상은 한 덩어리로 뽑는다.

    반환: (원자료 AUC, 95% 구간 하한, 상한(백분위법 2.5, 97.5), 구간 계산에 쓴 유효 추출 횟수). 한쪽 클래스가 비는 추출은 건너뛴다.
    """
    groups = collections.OrderedDict()
    for person, label, value in rows:
        entry = groups.setdefault(person, ([], []))
        if label == positive_label:
            entry[0].append(value)
        elif label == negative_label:
            entry[1].append(value)
    people = list(groups)
    if not people:
        return None, None, None, 0
    pos_arrays = [np.asarray(groups[p][0], dtype=np.float64) for p in people]
    neg_arrays = [np.asarray(groups[p][1], dtype=np.float64) for p in people]
    point = auc(np.concatenate(pos_arrays), np.concatenate(neg_arrays))
    rng = np.random.RandomState(seed)
    estimates = []
    for _ in range(draws):
        chosen = rng.randint(0, len(people), size=len(people))
        value = auc(np.concatenate([pos_arrays[i] for i in chosen]), np.concatenate([neg_arrays[i] for i in chosen]))
        if value is not None:
            estimates.append(value)
    if not estimates:
        return point, None, None, 0
    low, high = np.percentile(estimates, [2.5, 97.5])
    return point, float(low), float(high), len(estimates)


def video_value(record, statistic):
    """영상 단위 값: 'max' 는 창 점수 최댓값, 'mean' 은 창 점수 평균."""
    return float(np.max(record.scores)) if statistic == "max" else float(np.mean(record.scores))


def alarm_stats(records, label, threshold, run_length):
    """클래스 하나의 알람 영상 수, 영상 수, 알람 영상의 첫 알람 창 번호 중앙값, 그 창이 사용한 blink 수 중앙값."""
    members = [r for r in records if r.label == label]
    windows, blinks = [], []
    for record in members:
        hit = run_alarm_window(record.scores, threshold, run_length)
        if hit is not None:
            windows.append(hit)
            blinks.append(int(record.blinks[hit - 1]))
    return {"alarmed": len(windows), "videos": len(members), "median_window": median_or_none(windows), "median_blinks": median_or_none(blinks)}


# ----- 표 만들기 -----
def num(value, digits=2):
    return "-" if value is None else "%.*f" % (digits, value)


def pct(value):
    return "-" if value is None else "%.2f%%" % (100.0 * value)


def stat_cells(stat):
    return [num(stat["mean"]), num(stat["std"])] + [num(stat["pct"][p]) for p in PERCENTILES]


def table_t1(records):
    lines = ["%-6s %7s %8s | %7s %7s %7s %7s %7s %7s %7s | %9s %9s %9s" % (
        "class", "videos", "windows", "mean", "std", "p5", "p25", "p50", "p75", "p95", "0~3.34", "3.34~6.68", "6.68~10")]
    for label in CLASSES:
        members = [r for r in records if r.label == label]
        pooled = np.concatenate([r.scores for r in members]) if members else np.zeros(0)
        stat, bands = describe(pooled), band_fractions(pooled)
        lines.append("%-6d %7d %8d | %s | %s" % (label, len(members), pooled.size, " ".join("%7s" % c for c in stat_cells(stat)),
                                                 " ".join("%9s" % pct(b) for b in (bands or [None, None, None]))))
    return lines


def table_t2(records):
    lines = ["video max score distribution by class", "%-6s %7s | %7s %7s %7s %7s %7s %7s %7s" % ("class", "videos", "mean", "std", "p5", "p25", "p50", "p75", "p95")]
    maxima = {}
    for label in CLASSES:
        values = [video_value(r, "max") for r in records if r.label == label]
        maxima[label] = values
        stat = describe(values)
        lines.append("%-6d %7d | %s" % (label, stat["n"], " ".join("%7s" % c for c in stat_cells(stat))))
    lines.append("share of videos whose max >= T")
    lines.append("%-6s " % "T" + " ".join("%18s" % ("class %d" % c) for c in CLASSES))
    for threshold in THRESHOLDS:
        cells = []
        for label in CLASSES:
            values = maxima[label]
            share = fraction_at_least(values, threshold)
            cells.append("%18s" % ("%s (%d/%d)" % (pct(share), sum(1 for v in values if v >= threshold), len(values))))
        lines.append("%-6s " % num(threshold) + " ".join(cells))
    lines.append("class medians: video mean score; share of windows >= T")
    lines.append("%-6s %10s " % ("class", "video mean") + " ".join("%9s" % ("T=" + num(t)) for t in THRESHOLDS))
    for label in CLASSES:
        members = [r for r in records if r.label == label]
        means = [video_value(r, "mean") for r in members]
        cells = ["%9s" % pct(median_or_none([fraction_at_least(r.scores, t) for r in members])) for t in THRESHOLDS]
        lines.append("%-6d %10s %s" % (label, num(median_or_none(means)), " ".join(cells)))
    return lines


def table_t3(records, run_lengths=RUN_LENGTHS):
    lines = ["alarm videos: share of videos with >= m consecutive windows with score >= T (class 0 = false-alarm video share, class 10 = alarm video share)",
             "%-6s %-3s " % ("T", "m") + " ".join("%18s" % ("class %d" % c) for c in CLASSES) + " | first alarm window / blinks used (median over alarmed videos)"]
    for threshold in THRESHOLDS:
        for run_length in run_lengths:
            cells, firsts = [], []
            for label in CLASSES:
                stats = alarm_stats(records, label, threshold, run_length)
                share = stats["alarmed"] / float(stats["videos"]) if stats["videos"] else None
                cells.append("%18s" % ("%s (%d/%d)" % (pct(share), stats["alarmed"], stats["videos"])))
                firsts.append("c%d:%s/%s" % (label, num(stats["median_window"], 1), num(stats["median_blinks"], 1)))
            lines.append("%-6s %-3d %s | %s" % (num(threshold), run_length, " ".join(cells), "  ".join(firsts)))
    return lines


def table_t4(records, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED):
    lines = ["video-level AUC (positive vs negative class); person bootstrap 95%% interval (%d draws, seed %d, percentile method)" % (draws, seed),
             "%-12s %-6s %8s %20s %7s %s" % ("pair", "stat", "AUC", "95% interval", "draws", "videos(pos/neg)")]
    for statistic in ("max", "mean"):
        for positive, negative in PAIRS:
            rows = [(r.person, r.label, video_value(r, statistic)) for r in records if r.label in (positive, negative)]
            point, low, high, valid = person_bootstrap_auc(rows, positive, negative, draws, seed)
            counts = (sum(1 for r in rows if r[1] == positive), sum(1 for r in rows if r[1] == negative))
            lines.append("%-12s %-6s %8s %20s %7d %d/%d" % ("%d vs %d" % (positive, negative), statistic, num(point),
                                                          "[%s, %s]" % (num(low), num(high)), valid, counts[0], counts[1]))
    return lines


def table_t3_short(records):
    return table_t3(records, run_lengths=(1, 3))


def table_t5(by_fold):
    lines = []
    for fold in sorted(by_fold):
        records = by_fold[fold]
        lines.append("--- fold %d" % fold)
        lines.append("videos: " + ", ".join("class %d: n=%d videos" % (c, sum(1 for r in records if r.label == c)) for c in CLASSES))
        lines.extend(table_t1(records))
        lines.extend(table_t2(records))
        lines.extend(table_t3_short(records))
    return lines


# ----- 채점 호출부 -----
def read_exclusions(seq_dir):
    path = os.path.join(seq_dir, "metadata", "exclusions.csv")
    if not os.path.isfile(path):
        return None
    with open(path, "r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def padded_flags(ranges):
    """창마다 패딩 여부(사용한 blink 수가 30 미만)와 사용한 blink 수(끝 blink 번호 + 1)."""
    return (np.array([(end - start + 1) < W.WINDOW_SIZE for start, end in ranges], dtype=bool),
            np.array([end + 1 for _start, end in ranges], dtype=np.int64))


def collect_fold(fold, subjects, blink_dir, test_array, predictions, scorer):
    """폴드 하나의 검증 영상을 다시 만든 창으로 채점해 VideoRecord 목록을 만든다.

    반환: (기록 목록, 문제 목록, 창 값 최대 차이, 영상 평균 대조 차이 목록). 기준 구간을 만들 수 없거나 입력 파일이 없는 사람은 문제로 남기고 뺀다.
    """
    records, problems, mean_gaps, window_gap = [], [], [], 0.0
    for person, videos in subjects.items():
        tables = dict((video_id, VF.read_table(blink_dir, video_id)) for video_id in videos)
        if any(table is None for table in tables.values()):
            problems.append("fold %d person %s: B1 blinks file missing, person left out" % (fold, person))
            continue
        alert = [video_id for video_id, info in videos.items() if info["class_label"] == 0]
        reference = W.compute_reference(tables[alert[0]]) if len(alert) == 1 else None
        if reference is None:
            problems.append("fold %d person %s: no usable baseline from the alert video, person left out" % (fold, person))
            continue
        means, stds = reference
        for video_id, info in videos.items():
            windows, ranges = W.video_windows(tables[video_id], means, stds, is_reference=info["class_label"] == 0, pad_short=True)
            stored = np.asarray(test_array[info["seq"]])
            if np.shape(windows) != np.shape(stored):
                problems.append("fold %d video %s: window array shape differs (%s vs %s), video left out" % (fold, video_id, np.shape(windows), np.shape(stored)))
                continue
            if len(windows):
                window_gap = max(window_gap, float(np.abs(windows - stored).max()))
            raw = np.clip(np.asarray(scorer.score(windows), dtype=np.float64), SC.CLIP_LOW, SC.CLIP_HIGH)
            mean = SC.video_mean(raw)
            if (fold, video_id) in predictions and mean is not None:
                mean_gaps.append(abs(float(mean) - predictions[(fold, video_id)]))
            padded, blinks = padded_flags(ranges)
            records.append(VideoRecord(video_id, person, fold, info["class_label"], raw, padded, blinks))
    return records, problems, window_gap, mean_gaps


def input_summary(by_fold, exclusions, problems, window_gap, mean_gaps, elapsed):
    lines = ["== input summary =="]
    for fold in sorted(by_fold):
        records = by_fold[fold]
        per_class = [sum(1 for r in records if r.label == c) for c in CLASSES]
        lines.append("fold %d: persons=%d videos=%d (class 0: %d, 5: %d, 10: %d) windows=%d padded windows=%d" % (
            fold, len(set(r.person for r in records)), len(records), per_class[0], per_class[1], per_class[2],
            sum(len(r.scores) for r in records), sum(int(np.sum(r.padded)) for r in records)))
    total = [r for fold in by_fold for r in by_fold[fold]]
    lines.append("total: persons=%d videos=%d windows=%d padded windows=%d" % (
        len(set((r.fold, r.person) for r in total)), len(total), sum(len(r.scores) for r in total), sum(int(np.sum(r.padded)) for r in total)))
    if exclusions is None:
        lines.append("excluded by the B1 pipeline: exclusions.csv not found")
    else:
        reasons = collections.Counter((row["reason"], row["level"]) for row in exclusions)
        lines.append("excluded by the B1 pipeline (not in the evaluation arrays; all folds): " + ", ".join(
            "%s(%s)=%d" % (key[0], key[1], value) for key, value in sorted(reasons.items())))
        subject_rows = [row for row in exclusions if row["level"] == "subject"]
        lines.append("persons without a usable baseline (person level): %d persons, %d video rows" % (
            len(set(row["subject_id"] for row in subject_rows)), len(subject_rows)))
    lines.extend(["left out by this tool: %s" % p for p in problems] or ["left out by this tool: none"])
    gap = max(mean_gaps) if mean_gaps else None
    lines.append("window values vs stored test arrays: max abs diff %.3e -> %s (tolerance %.0e)" % (
        window_gap, "일치" if window_gap <= WINDOW_TOLERANCE else "불일치", WINDOW_TOLERANCE))
    lines.append("video mean vs video_predictions: max abs diff %s over %d videos -> %s (tolerance %.0e)" % (
        "-" if gap is None else "%.3e" % gap, len(mean_gaps), "미확정" if gap is None else ("일치" if gap <= MEAN_TOLERANCE else "불일치"), MEAN_TOLERANCE))
    lines.append("elapsed: %.1f s" % elapsed)
    return lines


def conclusions(editions, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED):
    lines = ["== conclusions read from the numbers (3 lines) =="]
    names = [("padded windows included", editions[True]), ("padded windows excluded", editions[False])]
    for number, text in enumerate(("class 10 videos with max score >= 6.68", "class 0 videos with an alarm (T=6.68, m=3)",
                                   "AUC class 10 vs 0 on video max score"), 1):
        parts = []
        for name, records in names:
            if number == 1:
                values = [video_value(r, "max") for r in records if r.label == 10]
                parts.append("%s: %s (%d/%d)" % (name, pct(fraction_at_least(values, 6.68)), sum(1 for v in values if v >= 6.68), len(values)))
            elif number == 2:
                stats = alarm_stats(records, 0, 6.68, 3)
                parts.append("%s: %s (%d/%d)" % (name, pct(stats["alarmed"] / float(stats["videos"]) if stats["videos"] else None),
                                                 stats["alarmed"], stats["videos"]))
            else:
                rows = [(r.person, r.label, video_value(r, "max")) for r in records if r.label in (10, 0)]
                point, low, high, _valid = person_bootstrap_auc(rows, 10, 0, draws, seed)
                parts.append("%s: %s [%s, %s]" % (name, num(point), num(low), num(high)))
        lines.append("(%d) %s -> %s" % (number, text, "; ".join(parts)))
    return lines


LIMITS = [
    "== what this analysis cannot say ==",
    "- Window labels are copies of the video label (0, 5, 10); a window inside a sleepy video is not known to be a sleepy moment. Per-window shares are not event rates.",
    "- Only the seed 1 checkpoints are used. Seeds 2 and 3 and the spread between runs are not looked at.",
    "- The alarm rule works on window numbers; windows have no time, so no seconds are reported. Blink counts depend on each person's blink rate.",
    "- Videos are scored on B1 re-extracted features (windows rebuilt from B1 blinks files and checked against the stored arrays); how live webcam input compares is 미검증.",
    "- Persons excluded by the B1 pipeline and the variant videos are not in the evaluation arrays and are not scored here.",
    "- AUC intervals resample persons; they describe sampling spread over persons in this data set, not other data.",
]


def build_report(by_fold, exclusions, problems, window_gap, mean_gaps, elapsed, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED):
    all_records = [r for fold in sorted(by_fold) for r in by_fold[fold]]
    editions = {True: make_edition(all_records, True), False: make_edition(all_records, False)}
    lines = conclusions(editions, draws, seed) + [""] + input_summary(by_fold, exclusions, problems, window_gap, mean_gaps, elapsed)
    names = {True: "padded windows included", False: "padded windows excluded"}
    for title, builder in (("T1 window score distribution by class", table_t1), ("T2 video max score by class", table_t2),
                           ("T3 alarm simulation", table_t3), ("T4 video-level AUC", lambda r: table_t4(r, draws, seed))):
        for include in (True, False):
            lines.extend(["", "== %s [%s] ==" % (title, names[include])])
            lines.extend(builder(editions[include]))
    for include in (True, False):
        lines.extend(["", "== T5 per fold [%s] ==" % names[include]])
        lines.extend(table_t5(dict((fold, make_edition(by_fold[fold], include)) for fold in sorted(by_fold))))
    return lines + [""] + LIMITS


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="오프라인 점수 분석(표준출력만, 파일을 쓰지 않음)")
    parser.add_argument("--blink-dir", dest="blink_dir", required=True, help="B1 blinks 폴더(<OUT_DIR>/batch)")
    parser.add_argument("--seq-dir", dest="seq_dir", required=True, help="시퀀스 산출물 루트(<SEQ_DIR>)")
    parser.add_argument("--ckpt-root", dest="ckpt_root", required=True, help="학습 출력 루트(<CKPT_DIR>)")
    parser.add_argument("--eval-dir", dest="eval_dir", required=True, help="video_predictions_seed{s}.csv 가 있는 폴더")
    parser.add_argument("--seed", type=int, default=1, help="체크포인트 시드(기본 1)")
    parser.add_argument("--stats", default=None, help="2단계 통계 JSON(기본: 패키지 안의 stage2_stats.json)")
    parser.add_argument("--folds", default="1,2,3,4,5", help="분석할 폴드(기본 모두)")
    parser.add_argument("--bootstrap-draws", dest="draws", type=int, default=BOOTSTRAP_DRAWS)
    parser.add_argument("--bootstrap-seed", dest="bootstrap_seed", type=int, default=BOOTSTRAP_SEED)
    return parser.parse_args(argv)


def main(argv=None):
    options = parse_args(argv)
    for path in (options.blink_dir, options.seq_dir, options.ckpt_root, options.eval_dir):
        if not os.path.isdir(path):
            print("폴더를 찾을 수 없습니다: %s" % os.path.basename(path))
            return 2
    started = time.time()
    try:
        folds = [int(part) for part in options.folds.split(",") if part.strip()]
        stats = S2.load_stats(options.stats)
        predictions = VF.read_predictions(options.eval_dir, options.seed)
        network = SC.load_network()
    except (OSError, ValueError, SC.ScoringError) as error:
        print("%s: %s" % (type(error).__name__, error))
        return 2
    by_fold, problems, window_gap, mean_gaps = {}, [], 0.0, []
    for fold in folds:
        subjects = VF.subjects_of(VF.read_metadata(options.seq_dir, fold))
        test_array = np.load(os.path.join(options.seq_dir, "npy", "BlinksTest_30_Fold%d.npy" % fold), mmap_mode="r")
        means, stds = stats[fold]
        with SC.FoldScorer(fold, SC.checkpoint_prefix(options.ckpt_root, options.seed, fold), means, stds, network) as scorer:
            records, fold_problems, fold_gap, fold_mean_gaps = collect_fold(fold, subjects, options.blink_dir, test_array, predictions, scorer)
        by_fold[fold] = records
        problems.extend(fold_problems)
        window_gap = max(window_gap, fold_gap)
        mean_gaps.extend(fold_mean_gaps)
    for line in build_report(by_fold, read_exclusions(options.seq_dir), problems, window_gap, mean_gaps, time.time() - started,
                             options.draws, options.bootstrap_seed):
        print(line)
    return 1 if (window_gap > WINDOW_TOLERANCE or (mean_gaps and max(mean_gaps) > MEAN_TOLERANCE)) else 0


if __name__ == "__main__":
    sys.exit(main())
