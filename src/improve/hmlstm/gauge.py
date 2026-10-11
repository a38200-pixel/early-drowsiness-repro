# gauge.py
# 역할: HM-LSTM 점수 게이지가 영상의 어느 프레임에서 무엇을 보여 줄지 정하는 순수 함수 모음(결정 I-33: 알람은 눈 닫힘 규칙으로만 울리고 점수는 0~10 게이지로
#       참고용 표시, I-27: 5개 폴드 평균과 최솟값~최댓값). 입력은 timeline.py score 가 만든 시간선 JSON 의 한 구간(평소영상 또는 대본영상)이다.
#       TensorFlow, cv2, PIL 을 가져오지 않고 화면을 그리지도 않는다(그리는 일은 이후 GUI 단계가 한다). 알람과 연결하지 않는다.
# 시간선 구간(dict)에서 쓰는 키: fps, blink_frame_indices(행별 프레임 번호, 비내림차순), windows(각 항목에 k, frame_index, mean, min, max).
#       창 k 의 frame_index 는 그 창의 마지막 행이 확정된 프레임이다. 행 확정이 블링크 끝보다 8~15프레임 늦는 것은 보정하지 않는다(I-W-28 판단).
# 구간 이름(깨어 있음 등)은 쓰지 않는다. 눈금선 위치만 상수로 둔다(구간 이름 표시 여부는 I-W-35 미결).
"""게이지 표시 상태와 문구(순수 함수, Python 3.6 호환)."""
import bisect

LABEL = "HM-LSTM score (0~10), 참고용, 알람 아님"
SCALE_MIN = 0.0
SCALE_MAX = 10.0
BOUNDARIES = (3.34, 6.68)           # 눈금선 위치(라벨 경계). 이름은 붙이지 않는다
WINDOW_BLINKS = 30                  # 첫 창이 나오려면 확정된 블링크가 이만큼 필요하다
STAGE_COLLECTING = "collecting"
STAGE_VALUE = "value"
STAGE_NO_BASELINE = "no_baseline"


def blinks_confirmed(section, frame_index):
    """frame_index 프레임까지 확정된 블링크 수(blink_frame_indices 중 값이 frame_index 이하인 것의 개수)."""
    return bisect.bisect_right(section["blink_frame_indices"], frame_index)


def latest_window(section, frame_index):
    """frame_index 프레임까지 확정된 마지막 창(dict). 아직 없으면 None. 창의 frame_index 는 비내림차순이라고 가정한다."""
    windows = section["windows"]
    position = bisect.bisect_right([window["frame_index"] for window in windows], frame_index)
    return windows[position - 1] if position > 0 else None


def display_state(section, frame_index, baseline_missing=False):
    """프레임 frame_index 에서의 게이지 상태(dict)를 돌려준다.

    stage: "no_baseline"(호출자가 기준이 없다고 알림), "value"(최신 창이 있음), "collecting"(아직 창이 없음).
    blinks: 확정된 블링크 수, needed: 첫 창에 필요한 수(30).
    window, mean, min, max: 최신 창 번호와 점수(없으면 None). updated_frame: 그 창이 확정된 프레임.
    elapsed_sec: 갱신 후 경과 시간(초) = (frame_index - updated_frame) / fps. 영상 끝 이후의 프레임에서는 계속 커진다. 창이 없으면 None.
    """
    blinks = blinks_confirmed(section, frame_index)
    state = {"stage": STAGE_COLLECTING, "blinks": blinks, "needed": WINDOW_BLINKS, "window": None, "mean": None, "min": None, "max": None,
             "updated_frame": None, "elapsed_sec": None}
    if baseline_missing:
        state["stage"] = STAGE_NO_BASELINE
        return state
    window = latest_window(section, frame_index)
    if window is not None:
        state.update(stage=STAGE_VALUE, window=window["k"], mean=window["mean"], min=window["min"], max=window["max"],
                     updated_frame=window["frame_index"], elapsed_sec=(frame_index - window["frame_index"]) / float(section["fps"]))
    return state


def format_value(state):
    """"6.6 [4.8-8.0]" 형식(평균 [최솟값-최댓값], 소수 한 자리)."""
    return "%.1f [%.1f-%.1f]" % (state["mean"], state["min"], state["max"])


def format_gauge_lines(state):
    """게이지 문구를 줄 목록으로 돌려준다. 첫 줄은 항상 라벨이다."""
    lines = [LABEL]
    if state["stage"] == STAGE_NO_BASELINE:
        lines.append("기준 없음")
    elif state["stage"] == STAGE_VALUE:
        lines.append(format_value(state))
        lines.append("창 %d · 갱신 후 %.1f초" % (state["window"], state["elapsed_sec"]))
    else:
        lines.append("수집 중 %d/%d 블링크" % (min(state["blinks"], state["needed"]), state["needed"]))
    return lines


def format_gauge_text(state):
    """format_gauge_lines 를 줄바꿈으로 이은 문자열."""
    return "\n".join(format_gauge_lines(state))
