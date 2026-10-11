# timeline_plot.py
# 역할: 게이지 점수 시간선(timeline.py score 의 출력)을 그림 한 장으로 그리고, 정답 사건 구간(e02_truth.json)과의 관계를 숫자로 출력하는 도구
#       (결정 I-34 (나): 숫자 집계 그래프만 만든다. 영상 프레임, 얼굴, 얼굴이 식별되는 파생 이미지는 넣지 않으며 입력은 숫자 JSON 뿐이다).
#       점수가 졸음이나 동작을 반영한다는 주장은 하지 않는다. 출력은 서술 통계와 순위뿐이고 해석 문장과 유의성 검정은 없다.
# 입력: --timeline(score 출력), --target-rows(대본영상 행 JSON), --truth(e02_truth.json, 읽기만), --out-png, 선택 --out-baseline-png 와 --baseline-rows.
# 그림(v3, 처음 보는 사람이 그림만 보고 읽을 수 있게): 영어 라벨, 글자 10pt 이상, 폭 2100px(14in, dpi 150). 위에서 아래로 제목, 범례, 점수 패널,
#       (대본영상만) 사건 패널, "How to read this figure" 글상자. 점수 패널에는 사건을 그리지 않고 사건은 아래 패널에 따로 그린다.
#       점수 = 5개 모델의 평균 선과 최솟값~최댓값 띠, 3.34 와 6.68 점선(이름 없음). 점수가 없는 앞 구간은 빗금, 평소영상의 기준 구간은 회색 음영.
#       matplotlib 은 Agg 백엔드로 파일에만 그린다(창을 띄우지 않는다). 출력 파일이 이미 있으면 덮어쓰지 않고 오류로 끝낸다(독점 생성).
# 표준출력: a 창별 표, b 창이 덮는 구간과 사건 겹침(개수, 그룹 통계), c 인접 창 사이 평균 변화 상위, d Dur 상위 행(가장 가까운 눈 감기 사건과 시각 차이 포함),
#       e 평소영상 같은 점검, f 검증.
# 정의: 창 k 의 구간 = [시작 행의 프레임 번호 ÷ fps, 끝 행의 프레임 번호 ÷ fps] 초(시작 행 = 끝 행 - 29). 겹침은 닫힌 구간 기준(끝점이 닿으면 겹침).
#       행 확정은 깜빡임이 끝난 시점보다 8~15프레임 뒤이며 보정하지 않는다(I-W-28 판단). 긴 깜빡임 = Dur 가 30 프레임 이상인 행.
# 종료 코드: 0 = 완료, 1 = 검증 불일치, 2 = 인자나 입력 오류
"""시간선 그림과 사건 대조(Python 3.6, matplotlib 은 그릴 때만 지연 임포트)."""
import argparse
import json
import os
import sys
import textwrap

from . import gauge as G
from . import timeline as TL

WINDOW_ROWS = 30
GROUPS = ("closure", "down", "frown")
PLOT_COLORS = {"closure": "#D55E00", "down": "#0072B2", "frown": "#009E73"}       # 색약 사용자도 구분하기 쉬운 팔레트(Okabe-Ito)
EVENT_ROWS = (("closure", "Eye closure (script)"), ("down", "Looking down (script)"), ("frown", "Grimacing (script)"))
LONG_BLINK_DUR = 30                                                               # Dur(프레임) 이 이 값 이상이면 긴 깜빡임(약 1초)
DUR_COLUMN = TL.COLUMNS.index("dur")
AMP_COLUMN = TL.COLUMNS.index("amp")
VEL_COLUMN = TL.COLUMNS.index("vel")

# 그림 글귀(영어). 구간 이름은 쓰지 않는다.
SUBTITLE = "HM-LSTM score (0-10) - reference only, not an alarm"
LEGEND_LINE = "Average of the 5 models"
LEGEND_BAND = "Range of the 5 models (lowest to highest)"
LEGEND_DOTS = "3.34 and 6.68: thirds of the 0-10 scale"
END_HIGH = "10 = drowsy end of the scale"
END_LOW = "0 = alert end of the scale"
NO_SCORE = "No score yet: waiting for the first 30 blinks to be confirmed"
REFERENCE_TEXT = "Last third of blinks: used as this person's reference (not scored)"
LONG_BLINK_LABEL = "Blinks of 1 s or longer (detected)"
EVENT_NOTE = "Script times are approximate (+-1 s). Triangles appear when a blink is confirmed, about 0.5-1 s after it ends."
HOW_TITLE = "How to read this figure"

# 그림 치수(인치). 폭 14in x dpi 150 = 2100px.
FIG_W = 14.0
LEFT = 2.6
RIGHT = 2.9
AXES_W = FIG_W - LEFT - RIGHT
SCORE_H = 4.8
EVENT_H = 2.0
TOP = 1.55
GAP = 0.15
X_AREA = 0.8
NOTE_H = 0.35
FONT = 10
DPI = 150


# ----- 순수 함수(그리기 제외) -----
def event_group(item):
    """사건 한 개의 종류: 눈 감기(B, C)는 closure, 구간 D 는 down, 구간 E 는 frown. 알 수 없으면 TimelineError."""
    if item.get("kind") == "눈 감기":
        return "closure"
    if item.get("interval") == "D":
        return "down"
    if item.get("interval") == "E":
        return "frown"
    raise TL.TimelineError("알 수 없는 사건 종류입니다: %r" % (item.get("id"),))


def parse_truth(document):
    """e02_truth 의 events 를 [{id, group, start_s, end_s}] 로 바꾼다."""
    if not isinstance(document, dict) or not isinstance(document.get("events"), list):
        raise TL.TimelineError("정답 문서에 events 목록이 없습니다")
    events = []
    for item in document["events"]:
        events.append({"id": item["id"], "group": event_group(item), "start_s": float(item["start_s"]), "end_s": float(item["end_s"])})
    return events


def overlaps(a_start, a_end, b_start, b_end):
    """닫힌 구간 [a_start, a_end] 와 [b_start, b_end] 가 겹치는가(끝점만 닿아도 True)."""
    return a_start <= b_end and b_start <= a_end


def window_spans(windows, frame_indices, fps):
    """창마다 {k, start_row, end_row, start_s, end_s}. 시작 행 = 끝 행 - 29, 초 = 그 행의 프레임 번호 ÷ fps."""
    spans = []
    for window in windows:
        end = window["end_row"]
        start = max(0, end - (WINDOW_ROWS - 1))
        spans.append({"k": window["k"], "start_row": start, "end_row": end, "start_s": frame_indices[start] / float(fps),
                      "end_s": frame_indices[end] / float(fps)})
    return spans


def overlap_counts(span, events):
    """창 구간이 종류별 사건 몇 개와 겹치는지: {closure, down, frown}."""
    counts = dict((group, 0) for group in GROUPS)
    for event in events:
        if overlaps(span["start_s"], span["end_s"], event["start_s"], event["end_s"]):
            counts[event["group"]] += 1
    return counts


def describe(values):
    """(개수, 평균, 최솟값, 최댓값). 값이 없으면 평균, 최솟값, 최댓값은 None."""
    if not values:
        return (0, None, None, None)
    return (len(values), sum(values) / float(len(values)), min(values), max(values))


def group_stats(windows, counts_list):
    """겹침 그룹별 창 수와 창 평균 점수의 평균, 최솟값, 최댓값. 반환: [(그룹 이름, 개수, 평균, 최솟값, 최댓값)]. 서술 통계만."""
    groups = [("overlaps an eye-closing event", lambda c: c["closure"] > 0),
              ("no eye-closing event overlap", lambda c: c["closure"] == 0),
              ("overlaps looking-down or frowning", lambda c: c["down"] + c["frown"] > 0),
              ("no looking-down or frowning overlap", lambda c: c["down"] + c["frown"] == 0),
              ("no event overlap at all", lambda c: sum(c.values()) == 0)]
    result = []
    for name, test in groups:
        values = [window["mean"] for window, counts in zip(windows, counts_list) if test(counts)]
        result.append((name,) + describe(values))
    return result


def adjacent_changes(windows, top=3):
    """창 평균이 앞 창보다 가장 크게 내려간 곳과 올라간 곳 상위 top 개. 반환: (내려감, 올라감), 항목은 (k, time_sec, 변화량).
    변화량 = 평균[k] - 평균[k-1] 이고 내려감은 음수만, 올라감은 양수만 센다. 같으면 k 가 작은 쪽이 먼저."""
    changes = [(windows[i]["k"], windows[i]["time_sec"], windows[i]["mean"] - windows[i - 1]["mean"]) for i in range(1, len(windows))]
    down = sorted([c for c in changes if c[2] < 0], key=lambda c: (c[2], c[0]))[:top]
    up = sorted([c for c in changes if c[2] > 0], key=lambda c: (-c[2], c[0]))[:top]
    return down, up


def event_at(second, events):
    """초 second 가 속한 사건 id(닫힌 구간). 없으면 "-"."""
    for event in events:
        if event["start_s"] <= second <= event["end_s"]:
            return event["id"]
    return "-"


def nearest_closure(second, events):
    """눈 감기 사건 중 끝 시각이 second 와 가장 가까운 것: (사건 id, second - 끝 시각). 양수면 사건 끝보다 늦다. 눈 감기 사건이 없으면 None.
    가까운 정도가 같으면 시작 시각이 이른 사건. 대응 판정이 아니라 숫자일 뿐이다."""
    closures = [e for e in events if e["group"] == "closure"]
    if not closures:
        return None
    best = min(closures, key=lambda e: (abs(second - e["end_s"]), e["start_s"]))
    return best["id"], second - best["end_s"]


def top_dur_rows(rows, frame_indices, fps, events, count, row_limit=None):
    """Dur 열이 큰 행 상위 count 개. 항목: {row, frame, second, dur, amp, vel, event, nearest, delta_s}. 같으면 행 번호(0부터)가 작은 쪽이 먼저.
    nearest/delta_s 는 nearest_closure 의 결과(눈 감기 사건이 없으면 None).
    row_limit 가 있으면 앞쪽 row_limit 개 행만 후보로 한다(평소영상 샘플 구간)."""
    limit = len(rows) if row_limit is None else min(row_limit, len(rows))
    order = sorted(range(limit), key=lambda i: (-rows[i][DUR_COLUMN], i))[:count]
    result = []
    for i in order:
        second = frame_indices[i] / float(fps)
        near = nearest_closure(second, events)
        result.append({"row": i, "frame": frame_indices[i], "second": second, "dur": rows[i][DUR_COLUMN], "amp": rows[i][AMP_COLUMN],
                       "vel": rows[i][VEL_COLUMN], "event": event_at(second, events),
                       "nearest": near[0] if near else None, "delta_s": near[1] if near else None})
    return result


def verify_section(section, rows_document=None):
    """시간선 구간의 점검. 반환: [(이름, 일치 여부, 설명)].
    time_sec 단조(비내림차순), 마지막 time_sec 가 영상 길이(frames_read ÷ fps) 이내, 그리고 행 JSON 이 있으면
    각 창의 frame_index 가 행 JSON 의 frame_indices[end_row] 와 같고 blink_frame_indices 도 같은지."""
    checks = []
    times = [window["time_sec"] for window in section["windows"]]
    decreasing = [k + 1 for k in range(1, len(times)) if times[k] < times[k - 1]]
    checks.append(("time_sec is non-decreasing", not decreasing, "decreasing at window %s" % decreasing[0] if decreasing else "windows %d" % len(times)))
    length = section["frames_read"] / float(section["fps"])
    last = times[-1] if times else None
    checks.append(("last time_sec within video length", last is None or last <= length,
                   "last %s s, length %.2f s" % ("-" if last is None else "%.2f" % last, length)))
    if rows_document is not None:
        indices = rows_document["frame_indices"]
        bad = [w["k"] for w in section["windows"] if not 0 <= w["end_row"] < len(indices) or indices[w["end_row"]] != w["frame_index"]]
        checks.append(("window end_row/frame_index match the rows file", not bad, "mismatch at window %s" % bad[0] if bad else "windows %d" % len(section["windows"])))
        same = list(section["blink_frame_indices"]) == list(indices)
        checks.append(("blink_frame_indices equal the rows file", same, "rows %d vs %d" % (len(section["blink_frame_indices"]), len(indices))))
        meta = section["fps"] == rows_document["fps"] and section["frames_read"] == rows_document["frames_read"]
        checks.append(("fps and frames_read equal the rows file", meta, "fps %s/%s frames %s/%s" % (
            section["fps"], rows_document["fps"], section["frames_read"], rows_document["frames_read"])))
    return checks


# ----- 그림 글귀와 값(순수 함수) -----
def video_title(kind, frames_read, fps):
    """그림 제목 1줄. kind 는 "script" 또는 "normal". 길이(초)는 frames_read ÷ fps 를 반올림한다. 파일 이름은 쓰지 않는다."""
    seconds = int(frames_read / float(fps) + 0.5)
    name = "Script test video" if kind == "script" else "Normal everyday video"
    return "%s (%d s)" % (name, seconds)


def first_window_info(spans, windows):
    """첫 점의 시각과 값, 그 창이 요약하는 구간의 시작 초와 끝 초."""
    return {"time_sec": windows[0]["time_sec"], "mean": windows[0]["mean"], "start_s": spans[0]["start_s"], "end_s": spans[0]["end_s"]}


def window_length_range(spans):
    """모든 창의 [시작 초, 끝 초] 길이(끝 - 시작)의 (최솟값, 최댓값)."""
    lengths = [span["end_s"] - span["start_s"] for span in spans]
    return min(lengths), max(lengths)


def reference_boundary_second(frame_indices, sample_blinks, fps):
    """평소영상에서 마지막 샘플 행(sample_blinks 번째 행, 0부터 센 번호 sample_blinks - 1)이 확정된 시각(초). 이 뒤가 기준 구간이다."""
    return frame_indices[sample_blinks - 1] / float(fps)


def duration_text(start, end):
    """사건 막대 안의 지속시간 글자: 정수면 "6 s", 아니면 "2.5 s"."""
    length = end - start
    return "%d s" % length if length == int(length) else "%.1f s" % length


def event_bars(events):
    """종류별 막대 목록 {group: [(시작 초, 끝 초, 지속시간 글자)]}. 사건 코드는 그림에 쓰지 않으므로 담지 않는다."""
    bars = dict((group, []) for group in GROUPS)
    for event in events:
        bars[event["group"]].append((event["start_s"], event["end_s"], duration_text(event["start_s"], event["end_s"])))
    return bars


def long_blink_seconds(rows, frame_indices, fps, minimum=LONG_BLINK_DUR):
    """Dur 가 minimum(프레임) 이상인 행의 확정 시각(초) 목록. 29 는 빠지고 30 은 들어간다."""
    return [frame_indices[i] / float(fps) for i, row in enumerate(rows) if row[DUR_COLUMN] >= minimum]


def how_to_read_lines(length_range, script):
    """"How to read this figure" 글상자의 번호 붙은 문장들. script 가 참이면 개인 기준에 대한 4번째 줄을 더한다."""
    low, high = length_range
    lines = ["Each dot is one score, computed from the 30 most recent blinks at that moment, so it summarizes the preceding stretch of video "
             "(%d-%d s long here), not a single instant." % (int(low + 0.5), int(high + 0.5)),
             "The line is the average of 5 models (the same network trained 5 times on different parts of the training data); "
             "the grey band shows how far apart the 5 models were.",
             "One recording of one person, shown for illustration only; it is not evidence that the score detects drowsiness."]
    if script:
        lines.append("Scores use the personal reference taken from the same person's normal video.")
    return lines


def how_to_read_text(length_range, script):
    """글상자 전체 글(제목 줄 + 줄바꿈한 번호 문장). 줄 수를 세어 그림 높이를 정하는 데도 쓴다."""
    body = []
    for number, sentence in enumerate(how_to_read_lines(length_range, script), 1):
        body.append(textwrap.fill("%d. %s" % (number, sentence), width=150, subsequent_indent="    "))
    return HOW_TITLE + "\n" + "\n".join(body)


# ----- 표 문자열 -----
def format_table(headers, rows):
    """열 너비를 맞춘 표 줄 목록(첫 줄이 머리글)."""
    texts = [[str(cell) for cell in row] for row in rows]
    widths = [max([len(headers[c])] + [len(row[c]) for row in texts]) for c in range(len(headers))]
    lines = ["  ".join(headers[c].ljust(widths[c]) for c in range(len(headers))).rstrip()]
    for row in texts:
        lines.append("  ".join(row[c].rjust(widths[c]) for c in range(len(headers))))
    return lines


def f2(value):
    return "-" if value is None else "%.2f" % value


def window_table(windows):
    return format_table(["k", "time_sec", "mean", "min", "max"],
                        [(w["k"], f2(w["time_sec"]), f2(w["mean"]), f2(w["min"]), f2(w["max"])) for w in windows])


def overlap_table(windows, spans, counts_list):
    return format_table(["k", "start_s", "end_s", "closure", "down", "frown", "mean"],
                        [(s["k"], f2(s["start_s"]), f2(s["end_s"]), c["closure"], c["down"], c["frown"], f2(w["mean"]))
                         for w, s, c in zip(windows, spans, counts_list)])


def group_table(stats):
    return format_table(["group", "windows", "mean of window means", "min", "max"], [(n, c, f2(m), f2(lo), f2(hi)) for n, c, m, lo, hi in stats])


def change_table(changes):
    return format_table(["window (k-1 -> k)", "time_sec", "change"], [("%d -> %d" % (k - 1, k), f2(t), "%+.2f" % d) for k, t, d in changes])


def dur_table(rows, with_event=True):
    """Dur 상위 행 표. with_event 가 참이면 event 열과 함께 "nearest closure event", "second minus event end"(양수면 사건 끝보다 늦음) 열이 붙는다."""
    headers = ["row", "frame", "second", "dur", "amp", "vel"]
    if with_event:
        headers += ["event", "nearest closure event", "second minus event end"]
    body = []
    for r in rows:
        line = [r["row"], r["frame"], f2(r["second"]), "%.4f" % r["dur"], "%.4f" % r["amp"], "%.4f" % r["vel"]]
        if with_event:
            line += [r["event"], r["nearest"] if r["nearest"] else "-", "%+.2f" % r["delta_s"] if r["delta_s"] is not None else "-"]
        body.append(line)
    return format_table(headers, body)


# ----- 그리기(matplotlib 은 여기서만) -----
class Canvas(object):
    """인치 단위 배치를 돕는 작은 도구. 위에서 잰 인치를 그림 비율 좌표로 바꾼다."""

    def __init__(self, plt, height):
        self.height = height
        self.figure = plt.figure(figsize=(FIG_W, height), dpi=DPI)

    def fx(self, inches):
        return inches / FIG_W

    def fy(self, inches_from_top):
        return 1.0 - inches_from_top / self.height

    def axes(self, top, height, sharex=None):
        return self.figure.add_axes([LEFT / FIG_W, self.fy(top + height), AXES_W / FIG_W, height / self.height], sharex=sharex)


def _score_panel(plt, matplotlib, canvas, section, spans, length, first_note_y):
    """점수 패널: 평균 선과 점, 5개 모델 범위 띠, 3.34 와 6.68 점선, 점수가 없는 앞 구간의 빗금, 첫 점 주석, 오른쪽 여백 글."""
    axes = canvas.axes(TOP, SCORE_H)
    windows = section["windows"]
    times = [w["time_sec"] for w in windows]
    info = first_window_info(spans, windows)
    axes.set_xlim(0, length)
    axes.set_ylim(G.SCALE_MIN, G.SCALE_MAX)
    axes.set_yticks(range(0, 11))
    axes.tick_params(labelsize=FONT)
    axes.grid(axis="x", color="#E3E3E3", linewidth=0.8, zorder=0)
    axes.axvspan(0, info["time_sec"], facecolor="#F4F4F4", edgecolor="#B5B5B5", hatch="///", linewidth=0, zorder=1)
    axes.fill_between(times, [w["min"] for w in windows], [w["max"] for w in windows], color="#9A9A9A", alpha=0.35, linewidth=0, zorder=2)
    for boundary in G.BOUNDARIES:
        axes.axhline(boundary, color="#4D4D4D", linestyle=(0, (2, 3)), linewidth=1.2, zorder=3)
    axes.plot(times, [w["mean"] for w in windows], color="black", marker="o", markersize=4.5, linewidth=1.6, zorder=4)
    axes.set_ylabel("HM-LSTM score (0-10)", fontsize=FONT)
    # 점수가 없는 앞 구간의 설명(구간이 좁으면 세로로 쓴다)
    region_in = info["time_sec"] / length * AXES_W
    box = dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="none", alpha=0.92)
    if region_in >= 3.2:
        axes.text(info["time_sec"] / 2.0, 5.0, "No score yet:\nwaiting for the first\n30 blinks to be confirmed", ha="center", va="center",
                  fontsize=FONT, bbox=box, zorder=5)
    else:
        axes.text(info["time_sec"] / 2.0, 5.0, "No score yet:\nwaiting for the first 30 blinks to be confirmed", ha="center", va="center",
                  rotation=90, fontsize=FONT, bbox=box, zorder=5)
    # 첫 점 주석
    note = "First score at %.0f s:\nuses the 30 blinks seen from %.0f to %.0f s" % (info["time_sec"], info["start_s"], info["end_s"])
    axes.annotate(note, xy=(info["time_sec"], info["mean"]), xytext=(info["time_sec"] + 0.04 * length, first_note_y), fontsize=FONT,
                  ha="left", va="center", zorder=6,
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#777777"),
                  arrowprops=dict(arrowstyle="-|>", color="black", linewidth=1.1, shrinkA=2, shrinkB=5))
    # 오른쪽 바깥 여백: 눈금선 숫자와 0, 10 의 뜻(구간 이름은 쓰지 않는다)
    blend = matplotlib.transforms.blended_transform_factory(axes.transAxes, axes.transData)
    for boundary in G.BOUNDARIES:
        axes.text(1.012, boundary, "%.2f" % boundary, transform=blend, fontsize=FONT, va="center", color="#333333")
    axes.text(1.012, G.SCALE_MAX, END_HIGH, transform=blend, fontsize=FONT, va="center", color="#333333")
    axes.text(1.012, G.SCALE_MIN, END_LOW, transform=blend, fontsize=FONT, va="center", color="#333333")
    return axes


def _titles_and_legend(canvas, plt, matplotlib, title):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    figure = canvas.figure
    figure.text(0.5, canvas.fy(0.32), title, ha="center", va="center", fontsize=14, fontweight="bold")
    figure.text(0.5, canvas.fy(0.70), SUBTITLE, ha="center", va="center", fontsize=11)
    handles = [Line2D([0], [0], color="black", marker="o", markersize=4.5, linewidth=1.6, label=LEGEND_LINE),
               Patch(facecolor="#9A9A9A", alpha=0.35, label=LEGEND_BAND),
               Line2D([0], [0], color="#4D4D4D", linestyle=(0, (2, 3)), linewidth=1.2, label=LEGEND_DOTS)]
    figure.legend(handles=handles, loc="center", bbox_to_anchor=(0.5, canvas.fy(1.13)), ncol=3, fontsize=FONT, frameon=False,
                  handlelength=2.2, columnspacing=1.8)


def _how_to_box(canvas, top_inches, text):
    canvas.figure.text(canvas.fx(0.25), canvas.fy(top_inches), text, ha="left", va="top", fontsize=FONT, linespacing=1.4,
                       bbox=dict(boxstyle="round,pad=0.5", facecolor="#F6F6F6", edgecolor="#9A9A9A"))


def _box_height(text):
    return 0.30 + 0.20 * (text.count("\n") + 1)


def _save(canvas, plt, path):
    try:
        with open(path, "xb") as handle:
            canvas.figure.savefig(handle, format="png", dpi=DPI)
    finally:
        plt.close(canvas.figure)


def draw_target_png(path, section, events, rows_document):
    """대본영상 그림: 점수 패널, 사건 패널(눈 감기, 아래 보기, 찡그리기 막대와 긴 깜빡임 삼각형), 읽는 법 글상자. 파일이 이미 있으면 FileExistsError.
    영상 프레임이나 얼굴은 그리지 않는다."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import patheffects
    windows = section["windows"]
    spans = window_spans(windows, section["blink_frame_indices"], section["fps"])
    length = section["frames_read"] / float(section["fps"])
    text = how_to_read_text(window_length_range(spans), True)
    score_bottom = TOP + SCORE_H
    events_top = score_bottom + GAP
    panel_bottom = events_top + EVENT_H
    box_top = panel_bottom + X_AREA + NOTE_H + 0.1
    canvas = Canvas(plt, box_top + _box_height(text) + 0.35)
    _titles_and_legend(canvas, plt, matplotlib, video_title("script", section["frames_read"], section["fps"]))
    score_axes = _score_panel(plt, matplotlib, canvas, section, spans, length, 0.9)
    plt.setp(score_axes.get_xticklabels(), visible=False)
    axes = canvas.axes(events_top, EVENT_H, sharex=score_axes)
    axes.set_xlim(0, length)
    axes.set_ylim(-0.6, 3.6)
    axes.set_yticks([])
    axes.tick_params(labelsize=FONT)
    axes.grid(axis="x", color="#E3E3E3", linewidth=0.8, zorder=0)
    for spine in ("top", "right", "left"):
        axes.spines[spine].set_visible(False)
    blend = matplotlib.transforms.blended_transform_factory(axes.transAxes, axes.transData)
    bars = event_bars(events)
    for index, (group, label) in enumerate(EVENT_ROWS):
        y = 3 - index
        axes.text(-0.012, y, label, transform=blend, ha="right", va="center", fontsize=FONT)
        axes.axhline(y - 0.5, color="#DDDDDD", linewidth=0.8, zorder=1)
        for start, end, duration in bars[group]:
            axes.barh(y, end - start, left=start, height=0.62, color=PLOT_COLORS[group], alpha=0.9, linewidth=0, zorder=2)
            axes.text((start + end) / 2.0, y, duration, ha="center", va="center", fontsize=FONT, zorder=3, color="black", fontweight="bold",
                      path_effects=[patheffects.withStroke(linewidth=3, foreground="white")])      # 좁은 막대 밖으로 글자가 나가도, 어두운 막대 위에서도 읽히도록 흰 테두리
    axes.text(-0.012, 0, LONG_BLINK_LABEL.replace(" (", "\n("), transform=blend, ha="right", va="center", fontsize=FONT)
    seconds = long_blink_seconds(rows_document["rows"], rows_document["frame_indices"], section["fps"])
    axes.plot(seconds, [0] * len(seconds), linestyle="none", marker="^", markersize=10, color="black", zorder=3)
    axes.set_xlabel("Time in video (s)", fontsize=FONT, labelpad=4)
    canvas.figure.text(canvas.fx(LEFT), canvas.fy(panel_bottom + X_AREA + 0.12), EVENT_NOTE, ha="left", va="center", fontsize=FONT, color="#333333")
    _how_to_box(canvas, box_top, text)
    _save(canvas, plt, path)


def draw_baseline_png(path, section, summary, rows_document=None):
    """평소영상 그림: 점수 패널과 읽는 법 글상자. 마지막 샘플 행 뒤부터 영상 끝까지를 개인 기준 구간으로 회색 음영 처리한다."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    windows = section["windows"]
    spans = window_spans(windows, section["blink_frame_indices"], section["fps"])
    length = section["frames_read"] / float(section["fps"])
    text = how_to_read_text(window_length_range(spans), False)
    panel_bottom = TOP + SCORE_H
    box_top = panel_bottom + X_AREA + 0.1
    canvas = Canvas(plt, box_top + _box_height(text) + 0.35)
    _titles_and_legend(canvas, plt, matplotlib, video_title("normal", section["frames_read"], section["fps"]))
    axes = _score_panel(plt, matplotlib, canvas, section, spans, length, 0.9)
    boundary = reference_boundary_second(section["blink_frame_indices"], summary["sample_blinks"], section["fps"])
    if boundary < length:
        axes.axvspan(boundary, length, facecolor="#BDBDBD", alpha=0.55, linewidth=0, zorder=1)
        axes.text((boundary + length) / 2.0, 7.0, "Last third of blinks:\nused as this person's\nreference (not scored)", ha="center", va="center",
                  fontsize=FONT, zorder=5, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="none", alpha=0.92))
    axes.set_xlabel("Time in video (s)", fontsize=FONT, labelpad=4)
    _how_to_box(canvas, box_top, text)
    _save(canvas, plt, path)


# ----- CLI -----
def read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def report_target(section, rows_document, events, lines):
    windows = section["windows"]
    spans = window_spans(windows, section["blink_frame_indices"], section["fps"])
    counts = [overlap_counts(span, events) for span in spans]
    lines += ["", "== a. target windows =="] + window_table(windows)
    lines += ["", "== b. window spans and event overlap (closed intervals; counts of events per kind) =="] + overlap_table(windows, spans, counts)
    lines += ["", "group statistics over window mean scores (descriptive only)"] + group_table(group_stats(windows, counts))
    down, up = adjacent_changes(windows)
    lines += ["", "== c. largest decreases between adjacent windows (mean score) =="] + change_table(down)
    lines += ["", "largest increases between adjacent windows (mean score)"] + change_table(up)
    lines += ["", "== d. top 8 rows by dur =="] + dur_table(top_dur_rows(rows_document["rows"], rows_document["frame_indices"], section["fps"], events, 8))
    lines.append("note: a row is confirmed 8 to 15 frames after the blink ends, so the frame number is later than the blink itself.")
    lines.append("note: 'second minus event end' is positive when the row's confirmation time is later than the end time of the nearest eye-closure event; numbers only, no matching is claimed.")


def report_baseline(section, summary, rows_document, lines):
    windows = section["windows"]
    count, mean, low, high = describe([w["mean"] for w in windows])
    lines += ["", "== e. baseline video, sample part =="]
    lines.append("windows %d; window mean score: mean %s, min %s, max %s" % (count, f2(mean), f2(low), f2(high)))
    if summary.get("fold_means"):
        lines.append("fold means (folds 1-5): " + " ".join(f2(v) for v in summary["fold_means"]))
    if rows_document is not None:
        top = top_dur_rows(rows_document["rows"], rows_document["frame_indices"], section["fps"], [], 5, summary.get("sample_blinks"))
        lines += ["top 5 sample rows by dur"] + dur_table(top, with_event=False)
        lines.append("note: a row is confirmed 8 to 15 frames after the blink ends, so the frame number is later than the blink itself.")
    else:
        lines.append("(dur rows skipped: --baseline-rows not given)")


def report_verification(timeline, target_rows, baseline_rows, lines):
    lines += ["", "== f. verification =="]
    failed = False
    for label, section, rows_document in (("target", timeline["target"], target_rows), ("baseline", timeline["baseline"], baseline_rows)):
        for name, ok, detail in verify_section(section, rows_document):
            failed = failed or not ok
            lines.append("%s: %s -> %s (%s)" % (label, name, "일치" if ok else "불일치", detail))
    return failed


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="게이지 점수 시간선 그림과 정답 사건 대조(숫자 JSON 입력, 출력은 PNG 와 표준출력).")
    parser.add_argument("--timeline", required=True, help="timeline.py score 가 만든 시간선 JSON")
    parser.add_argument("--target-rows", dest="target_rows", required=True, help="대본영상 행 JSON(timeline.py extract 출력)")
    parser.add_argument("--truth", required=True, help="정답 사건 JSON(e02_truth.json, 읽기만 함)")
    parser.add_argument("--out-png", dest="out_png", required=True, help="대본영상 그림 PNG(이미 있으면 거부)")
    parser.add_argument("--out-baseline-png", dest="out_baseline_png", default=None, help="평소영상 그림 PNG(선택, 이미 있으면 거부)")
    parser.add_argument("--baseline-rows", dest="baseline_rows", default=None, help="평소영상 행 JSON(선택, Dur 상위 행과 검증에 쓴다)")
    return parser.parse_args(argv)


def main(argv=None):
    options = parse_args(argv)
    try:
        for path in (options.out_png, options.out_baseline_png):
            if path and os.path.exists(path):
                raise TL.TimelineError("이미 있는 경로에는 덮어쓰지 않습니다: %s" % os.path.basename(path))
        timeline = TL.read_timeline_document(options.timeline)
        target_rows = TL.read_rows_document(options.target_rows)
        baseline_rows = TL.read_rows_document(options.baseline_rows) if options.baseline_rows else None
        events = parse_truth(read_json(options.truth))
        lines = ["== inputs ==", "timeline target=%s baseline=%s; events %d (%s)" % (
            timeline["target"]["video_name"], timeline["baseline"]["video_name"], len(events), " ".join(e["id"] for e in events))]
        report_target(timeline["target"], target_rows, events, lines)
        report_baseline(timeline["baseline"], timeline["summary"]["baseline"], baseline_rows, lines)
        failed = report_verification(timeline, target_rows, baseline_rows, lines)
        for path in (options.out_png, options.out_baseline_png):
            if path:
                os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        draw_target_png(options.out_png, timeline["target"], events, target_rows)
        lines.append("")
        lines.append("figure written: %s" % os.path.basename(options.out_png))
        if options.out_baseline_png:
            draw_baseline_png(options.out_baseline_png, timeline["baseline"], timeline["summary"]["baseline"], baseline_rows)
            lines.append("figure written: %s" % os.path.basename(options.out_baseline_png))
        for line in lines:
            print(line)
        return 1 if failed else 0
    except (TL.TimelineError, FileExistsError, FileNotFoundError, KeyError, ValueError) as error:
        print("timeline_plot: %s" % error)
        return 2


if __name__ == "__main__":
    sys.exit(main())
