# live_score_recheck.py
# 역할: live_combined 가 --out-json 으로 남긴 숫자 JSON 의 행을 새 LiveSession 에 다시 넣고, 세션이 만든 창을 자식 프로세스 없이 이 프로세스 안에서 점수화해
#       기록된 라이브 점수와 비교한다(같은 행의 재채점 비교). 콘솔 요약만 쓰고 파일은 저장하지 않는다.
# 실행: PYTHONPATH=src python -m improve.hmlstm.live_score_recheck --live-json <입력 JSON> --ckpt-root <CKPT_DIR> [--tolerance 1e-4]
# 입력 JSON: live_combined.build_run_document 의 형식(kind live_combined_run, schema_version 1). 쓰는 키: fps_assumed, rows[{row, frame_index, t, face_frames}],
#       reference{rows, end_time, means, stds}(없으면 null), scores[{k, end_row, frame_index, mean, min, max}], 선택 키 config{reference_s, extension_s, min_reference_rows}.
#       설정값(기준 구간 길이 등)은 live_combined 가 기록하지 않으므로 config 가 없으면 live_session 의 기본값을 쓰고 요약에 어느 쪽인지 적는다.
# 시간: 행의 t 를 쓴다(scores[].time_sec 는 frame_index / 가정 fps 라서 쓰지 않는다). scores[].end_row 는 기준 이후 행 기준 상대 인덱스이다.
# advance 의존성(live_session 소스로 확인): 기준 구간 종료는 행이 없는 틱의 advance(t) 로도 일어나지만, add_row 가 행을 넣기 전에 먼저 advance(t) 를 부르므로(157줄)
#       기준 구간 뒤의 첫 행은 어느 쪽이든 기준 이후 행이 된다. 기준 행 수와 기준 통계는 행의 t 만으로 같게 재현된다. 다른 것은 reference_end_time 하나이다:
#       라이브는 종료를 일으킨 틱의 시각을, 재계산은 그 뒤 첫 행의 시각을 가진다. 그래서 종료 시각은 참고로만 보여 주고 판정에 쓰지 않는다.
# 이 모듈은 import 시 tensorflow, cv2, dlib 을 가져오지 않는다. TensorFlow 는 실제 채점기를 만들 때(build_scorer)만 불린다.
"""라이브 점수 재채점 비교(Python 3.6 호환)."""
import argparse
import json
import math
import sys

import numpy as np

from . import live_session as LS

KIND = "live_combined_run"
SCHEMA_VERSION = 1
DEFAULT_TOLERANCE = 1e-4
REFERENCE_STATS_TOLERANCE = 1e-9
NOTE = "이 값은 같은 행의 재채점 비교이며 B1 오프라인 경로와의 비교가 아님"
MATCH, MISMATCH, UNDECIDED = "일치", "불일치", "미확정"
SEED = 1


class RecheckError(ValueError):
    """입력 JSON 이나 인자가 맞지 않을 때."""


# ----- 입력 -----
def _reject_constant(name):
    raise RecheckError("입력 JSON 에 숫자가 아닌 값(%s)이 있습니다" % name)


def load_document(path):
    try:
        with open(path, encoding="utf-8") as handle:
            document = json.load(handle, parse_constant=_reject_constant)
    except (OSError, ValueError) as error:
        if isinstance(error, RecheckError):
            raise
        raise RecheckError("입력 JSON 을 읽을 수 없습니다: %s" % error)
    return validate_document(document)


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _integer(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _require(condition, field):
    if not condition:
        raise RecheckError("입력 JSON 에 필요한 필드가 없거나 형식이 맞지 않습니다: %s" % field)


def validate_document(document):
    _require(isinstance(document, dict), "(최상위 객체)")
    _require(document.get("kind") == KIND, "kind (%s 이어야 함)" % KIND)
    _require(document.get("schema_version") == SCHEMA_VERSION, "schema_version (%d 이어야 함)" % SCHEMA_VERSION)
    _require(_number(document.get("fps_assumed")) and document["fps_assumed"] > 0, "fps_assumed")
    _require(isinstance(document.get("rows"), list), "rows")
    for index, entry in enumerate(document["rows"]):
        prefix = "rows[%d]" % index
        _require(isinstance(entry, dict), prefix)
        row = entry.get("row")
        _require(isinstance(row, list) and len(row) == LS.ROW_LENGTH and all(_number(v) for v in row), prefix + ".row")
        _require(_integer(entry.get("frame_index")), prefix + ".frame_index")
        _require(_number(entry.get("t")), prefix + ".t")
        _require(_integer(entry.get("face_frames")), prefix + ".face_frames")
    _require(isinstance(document.get("scores"), list), "scores")
    for index, entry in enumerate(document["scores"]):
        prefix = "scores[%d]" % index
        _require(isinstance(entry, dict), prefix)
        _require(_integer(entry.get("k")), prefix + ".k")
        _require(_integer(entry.get("frame_index")), prefix + ".frame_index")
        for key in ("mean", "min", "max"):
            _require(_number(entry.get(key)), "%s.%s" % (prefix, key))
        _require("end_row" not in entry or _integer(entry["end_row"]), prefix + ".end_row")
    reference = document.get("reference")
    if reference is not None:
        _require(isinstance(reference, dict), "reference")
        _require(_integer(reference.get("rows")), "reference.rows")
        for key in ("means", "stds"):
            _require(isinstance(reference.get(key), list) and all(_number(v) for v in reference[key]), "reference." + key)
        _require(reference.get("end_time") is None or _number(reference["end_time"]), "reference.end_time")
    config = document.get("config")
    _require(config is None or isinstance(config, dict), "config")
    return document


def session_settings(document):
    """(LiveSession 인자 dict, 출처 문장). JSON 의 config 가 있으면 그 값을, 없으면 live_session 의 기본값을 쓴다."""
    defaults = {"reference_s": LS.REFERENCE_S, "extension_s": LS.EXTENSION_S, "min_reference_rows": LS.MIN_REFERENCE_ROWS}
    config = document.get("config")
    if not config:
        return defaults, "기본값(JSON 에 기록 없음)"
    settings = dict(defaults)
    for key in defaults:
        if key in config:
            _require(_number(config[key]), "config." + key)
            settings[key] = config[key]
    return settings, "JSON 의 config"


# ----- 다시 넣기 -----
def replay_session(document):
    """행을 기록 순서대로 새 LiveSession 에 넣는다. 반환: (세션, 만들어진 WindowJob 목록, 설정 출처 문장)."""
    settings, source = session_settings(document)
    session = LS.LiveSession(0.0, fps=document["fps_assumed"], reference_s=settings["reference_s"], extension_s=settings["extension_s"],
                             min_reference_rows=int(settings["min_reference_rows"]))
    jobs = []
    for entry in document["rows"]:
        frame, t = entry["frame_index"], float(entry["t"])
        faces = entry["face_frames"]
        try:
            jobs.extend(session.add_row(entry["row"], frame, t, faces if faces > 0 else None))
        except LS.SessionError as error:
            raise RecheckError("행을 세션에 넣을 수 없습니다(frame_index %d): %s" % (frame, error))
    return session, jobs, source


def score_jobs(session, jobs, scorer):
    """창 작업을 한 개씩 scorer.score 에 넣어(라이브와 같은 조건) 세션에 반영한다."""
    for job in jobs:
        result = scorer.score(np.asarray([job.window], dtype=np.float64))
        folds = [float(np.asarray(result.per_fold[fold])[0]) for fold in sorted(result.per_fold)] if getattr(result, "per_fold", None) else None
        session.apply_score(job.seq, float(result.mean[0]), float(result.min[0]), float(result.max[0]), folds)


def build_scorer(ckpt_root):
    """실제 5폴드 채점기(여기서 처음 TensorFlow 가 불릴 수 있다). score_service 가 쓰는 것과 같은 생성이다."""
    from . import scoring as SC
    from . import stage2 as S2
    return SC.EnsembleScorer(ckpt_root, SEED, stats=S2.load_stats(None))


# ----- 비교 -----
def compare(document, session, tolerance):
    """기록된 점수와 재계산 점수를 비교한 결과 dict."""
    recorded = sorted(document["scores"], key=lambda item: item["k"])
    recomputed = sorted(session.section["windows"], key=lambda item: item["k"])
    recorded_by_k = dict((item["k"], item) for item in recorded)
    recomputed_by_k = dict((item["k"], item) for item in recomputed)
    common = sorted(set(recorded_by_k) & set(recomputed_by_k))
    frame_mismatches = [k for k in common if recorded_by_k[k]["frame_index"] != recomputed_by_k[k]["frame_index"]]
    row_mismatches = [k for k in common if "end_row" in recorded_by_k[k] and recorded_by_k[k]["end_row"] != recomputed_by_k[k]["end_row"]]
    differences = dict((key, [abs(recorded_by_k[k][key] - recomputed_by_k[k][key]) for k in common]) for key in ("mean", "min", "max"))
    worst = [max(differences[key][i] for key in differences) for i in range(len(common))]
    exceeded = [common[i] for i in range(len(common)) if worst[i] > tolerance]
    reference = document.get("reference")
    reference_info = None
    if reference is not None:
        means_diff = stds_diff = None
        if session.reference is not None and len(reference["means"]) == len(session.reference.means) and len(reference["stds"]) == len(session.reference.stds):
            means_diff = float(np.max(np.abs(np.array(reference["means"]) - np.asarray(session.reference.means)))) if reference["means"] else 0.0
            stds_diff = float(np.max(np.abs(np.array(reference["stds"]) - np.asarray(session.reference.stds)))) if reference["stds"] else 0.0
        stats_match = means_diff is not None and max(means_diff, stds_diff) <= REFERENCE_STATS_TOLERANCE
        end_time = reference.get("end_time")
        reference_info = {"rows_recorded": reference["rows"], "rows_recomputed": len(session.ref_rows), "rows_match": reference["rows"] == len(session.ref_rows),
                          "means_diff": means_diff, "stds_diff": stds_diff, "stats_match": stats_match,
                          "end_time_recorded": end_time, "end_time_recomputed": session.reference_end_time,
                          "end_time_diff": (abs(end_time - session.reference_end_time) if end_time is not None and session.reference_end_time is not None else None)}
    last = None
    if recorded and recomputed:
        last = {"recorded": recorded[-1], "recomputed": recomputed[-1]}
    counts_match = len(recorded) == len(recomputed)
    if not recorded and not recomputed:
        verdict = UNDECIDED
    elif counts_match and not frame_mismatches and not row_mismatches and not exceeded:
        verdict = MATCH
    else:
        verdict = MISMATCH
    return {"verdict": verdict, "tolerance": tolerance, "count_recorded": len(recorded), "count_recomputed": len(recomputed), "counts_match": counts_match,
            "frame_mismatches": len(frame_mismatches), "first_frame_mismatch": frame_mismatches[0] if frame_mismatches else None,
            "row_mismatches": len(row_mismatches), "first_row_mismatch": row_mismatches[0] if row_mismatches else None,
            "max_diff": dict((key, max(values) if values else None) for key, values in differences.items()),
            "mean_diff": dict((key, float(np.mean(values)) if values else None) for key, values in differences.items()),
            "exceeded": len(exceeded), "first_exceeded": exceeded[0] if exceeded else None, "compared": len(common), "last": last, "reference": reference_info,
            "phase": session.phase, "reason": session.reason}


def _fmt(value, spec="%.6g"):
    return "-" if value is None else spec % value


def _score_text(entry):
    return "%.4f [%.4f-%.4f] (창 %d, frame %d)" % (entry["mean"], entry["min"], entry["max"], entry["k"], entry["frame_index"])


def format_report(document, result, settings_source):
    lines = ["live json: status %s ; rows %d ; scores recorded %d ; frames %s" % (document.get("status", "-"), len(document["rows"]), len(document["scores"]),
                                                                              document.get("frames", "-")),
             "settings: %s ; fps assumed %g" % (settings_source, document["fps_assumed"]),
             "session after replay: phase %s%s (recorded phase %s)" % (result["phase"], "" if not result["reason"] else " (%s)" % result["reason"], document.get("session_phase", "-"))]
    reference = result["reference"]
    if reference is None:
        lines.append("reference: JSON 에 기준 정보가 없어 비교하지 않음")
    else:
        lines.append("reference rows: recorded %d ; recomputed %d ; %s" % (reference["rows_recorded"], reference["rows_recomputed"], MATCH if reference["rows_match"] else MISMATCH))
        lines.append("reference stats: max diff means %s ; stds %s ; %s" % (_fmt(reference["means_diff"], "%.3g"), _fmt(reference["stds_diff"], "%.3g"),
                                                                           MATCH if reference["stats_match"] else MISMATCH))
        lines.append("reference end time (참고, 판정에 쓰지 않음: 라이브는 틱 시각, 재계산은 다음 행 시각): recorded %s ; recomputed %s ; diff %s s" % (
            _fmt(reference["end_time_recorded"], "%.3f"), _fmt(reference["end_time_recomputed"], "%.3f"), _fmt(reference["end_time_diff"], "%.3f")))
    lines.append("windows: recorded %d ; recomputed %d ; %s" % (result["count_recorded"], result["count_recomputed"], MATCH if result["counts_match"] else MISMATCH))
    lines.append("frame numbers: mismatches %d of %d ; first mismatch window %s" % (result["frame_mismatches"], result["compared"], _fmt(result["first_frame_mismatch"], "%d")))
    lines.append("window position (end_row): mismatches %d ; first mismatch window %s" % (result["row_mismatches"], _fmt(result["first_row_mismatch"], "%d")))
    lines.append("score diff (0 to 10 scale): max diff mean %s min %s max %s ; average diff mean %s min %s max %s" % (
        _fmt(result["max_diff"]["mean"]), _fmt(result["max_diff"]["min"]), _fmt(result["max_diff"]["max"]),
        _fmt(result["mean_diff"]["mean"]), _fmt(result["mean_diff"]["min"]), _fmt(result["mean_diff"]["max"])))
    lines.append("over tolerance %g: windows %d ; first window %s" % (result["tolerance"], result["exceeded"], _fmt(result["first_exceeded"], "%d")))
    if result["last"] is None:
        lines.append("last window: -")
    else:
        lines.append("last window recorded: %s" % _score_text(result["last"]["recorded"]))
        lines.append("last window recomputed: %s" % _score_text(result["last"]["recomputed"]))
    lines.append("verdict: %s" % result["verdict"])
    lines.append(NOTE)
    return lines


# ----- 실행 -----
def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="live_combined 의 숫자 JSON 을 읽어 같은 행을 다시 채점하고 기록된 라이브 점수와 비교한다(콘솔 요약만).")
    parser.add_argument("--live-json", dest="live_json", required=True, help="live_combined --out-json 으로 만든 JSON")
    parser.add_argument("--ckpt-root", dest="ckpt_root", required=True, help="학습 출력 루트(<CKPT_DIR>, EnsembleScorer 에 전달)")
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE, help="점수 차이 허용값(0에서 10 눈금, 기본 %g)" % DEFAULT_TOLERANCE)
    return parser.parse_args(argv)


def run(document, scorer, tolerance):
    """문서와 채점기로 비교 결과와 보고 줄을 만든다(시험에서 가짜 채점기를 넣는 진입점)."""
    session, jobs, source = replay_session(document)
    score_jobs(session, jobs, scorer)
    result = compare(document, session, tolerance)
    return result, format_report(document, result, source)


def main(argv=None, scorer_factory=None, out=print):
    arguments = parse_args(argv)
    if not (arguments.tolerance >= 0 and math.isfinite(arguments.tolerance)):
        out("live_score_recheck: --tolerance 는 0 이상의 유한한 수여야 합니다")
        return 2
    scorer = None
    try:
        document = load_document(arguments.live_json)
        session, jobs, source = replay_session(document)
        if jobs:
            scorer = (scorer_factory or build_scorer)(arguments.ckpt_root)
            score_jobs(session, jobs, scorer)
        result = compare(document, session, arguments.tolerance)
        for line in format_report(document, result, source):
            out(line)
        return 0 if result["verdict"] == MATCH else 1
    except RecheckError as error:
        out("live_score_recheck: %s" % error)
        return 2
    finally:
        if scorer is not None and hasattr(scorer, "close"):
            scorer.close()


if __name__ == "__main__":
    sys.exit(main())
