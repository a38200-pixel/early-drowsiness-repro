# 역할: 눈 감김 구간 상태기계(C2). docs/improvement/RULE_SPEC.md 5절과 결정 I-04~I-08, I-15, I-W-11(확정)을 따른다.
# 입력: (timestamp_s, ear_mean 또는 None, source) 시퀀스. source 는 detected / tracked / none 이다(RULE_SPEC 7.1절).
# 출력: Result(alarms, episodes, unjudgeable, threshold, params). 파일을 읽거나 쓰지 않는다.
# 시간은 모두 타임스탬프(초)로 센다. 프레임 수를 쓰지 않는다(I-05).
#
# 규칙 요약
#   - 닫힘: ear_mean <= 임계값(기본 0.21, I-04). 열림: ear_mean > 임계값. 이동평균은 쓰지 않는다(I-08).
#   - 구간: 닫힘이 시작된 첫 샘플의 시각이 구간 시작이다.
#   - 알람(I-05, 확정): 구간 길이가 3.0초에 도달한 시점의 첫 샘플에서 구간당 1회.
#   - 해제(I-06): 열림이 연속 0.3초 이상이면 구간을 해제한다. 구간 끝은 열린 구간의 시작 시각이다(확정).
#     0.3초 미만의 열림은 구간 시간에 포함한다(확정).
#   - 얼굴 없음(I-07, I-15): 유효 EAR 이 없는 샘플. 1초 미만이면 직전 상태를 유지하고, 1초 이상이면 "판정 불가"로
#     진행 중인 구간을 폐기한다. 이미 울린 알람은 끄지 않는다.
#   - 영상 끝(I-W-11, 확정): 진행 중인 구간이 3.0초 미만이면 폐기하고 이미 울린 알람은 유지한다.
#
# 문서가 정하지 않아 "제안"으로 정한 해석(바꾸려면 RULE_SPEC.md 에서 먼저 정해야 한다)
#   (제안 1) 시간 측정 규약: 샘플은 다음 샘플의 시각까지를 대표한다. 그래서 닫힘 N프레임 뒤에 열림 프레임이 오면
#            구간 끝 = 그 열림 프레임의 시각이고, 열림 연속 시간은 "열림이 시작된 시각 ~ 열림이 끝난 뒤 첫 샘플의 시각"
#            이다(예: 30 fps 에서 열림 9프레임 뒤 닫힘이 오면 0.3초).
#   (제안 2) 구간 길이를 재는 끝: 닫힘 샘플이면 그 샘플의 시각, 열림 샘플이면 열림이 시작된 시각이다(열린 간격이 이어질지
#            확정되기 전까지는 간격을 구간 시간에 넣지 않는다).
#   (제안 3) 알람은 유효 EAR 이 있는 샘플에서만 판정한다(얼굴 없음 샘플에서는 판정하지 않는다).
#   (제안 4) 유효 출처의 기본값은 detected 하나이다(I-W-14, 미확정). tracked 를 유효로 볼지는 호출자가 고른다.
#   (제안 5) 판정 불가 구간의 끝은 얼굴이 다시 유효해진 첫 샘플의 시각이고, 입력이 끝날 때까지 이어지면 마지막 시각이다.
#   (제안 6) 알람 reliability 필드의 값("정상" 등)은 호출자가 정한다. 기본값 "정상"은 제안이다(I-W-12).
import math
from collections import namedtuple

EPS = 1e-9                      # 시간 비교의 부동소수점 오차 허용
SOURCES = ("detected", "tracked", "none")
DEFAULT_VALID_SOURCES = ("detected",)   # 제안(미확정, I-W-14)
DEFAULT_THRESHOLD = 0.21
ALARM_AFTER_S = 3.0
RELEASE_OPEN_S = 0.3
UNJUDGEABLE_AFTER_S = 1.0
DEFAULT_RELIABILITY = "정상"            # 제안(I-W-12)

STATUS_RELEASED = "released"
STATUS_KEPT_AT_END = "kept_at_end"
STATUS_DISCARDED_AT_END = "discarded_at_end"
STATUS_DISCARDED_UNJUDGEABLE = "discarded_unjudgeable"

Alarm = namedtuple("Alarm", ["episode_id", "alarm_time_s", "episode_start_s", "duration_at_alarm_s",
                             "threshold", "alarm_after_s", "reliability"])
Episode = namedtuple("Episode", ["episode_id", "start_s", "end_s", "length_s", "status", "alarmed"])
Unjudgeable = namedtuple("Unjudgeable", ["start_s", "end_s", "length_s"])
Result = namedtuple("Result", ["alarms", "episodes", "unjudgeable", "threshold", "params"])


def _at_least(value, limit):
    """value 가 limit 이상인지 오차 허용으로 본다."""
    return value >= limit - EPS


class EpisodeMachine(object):
    """감김 구간 상태기계. feed 로 샘플을 시각 순서대로 넣고 마지막에 finish 를 부른다."""

    def __init__(self, threshold=DEFAULT_THRESHOLD, alarm_after_s=ALARM_AFTER_S, release_open_s=RELEASE_OPEN_S,
                 unjudgeable_after_s=UNJUDGEABLE_AFTER_S, valid_sources=DEFAULT_VALID_SOURCES,
                 reliability=DEFAULT_RELIABILITY):
        """설정을 받는다.

        입력: threshold - 닫힘 임계값(이하이면 닫힘). alarm_after_s - 알람 지속 시간(초). release_open_s - 해제에 필요한
              열림 연속 시간(초). unjudgeable_after_s - 판정 불가로 보는 얼굴 없음 시간(초). valid_sources - 유효 EAR 로 보는
              출처. reliability - 알람에 붙일 신뢰 표시(제안).
        """
        for name, value in (("threshold", threshold), ("alarm_after_s", alarm_after_s),
                            ("release_open_s", release_open_s), ("unjudgeable_after_s", unjudgeable_after_s)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError("%s 는 0 보다 큰 유한한 수여야 합니다." % name)
        valid_sources = tuple(valid_sources)
        for source in valid_sources:
            if source not in SOURCES or source == "none":
                raise ValueError("valid_sources 는 detected, tracked 중에서 골라야 합니다: %r" % (source,))
        if not valid_sources:
            raise ValueError("valid_sources 가 비어 있습니다.")
        self.threshold = float(threshold)
        self.alarm_after_s = float(alarm_after_s)
        self.release_open_s = float(release_open_s)
        self.unjudgeable_after_s = float(unjudgeable_after_s)
        self.valid_sources = valid_sources
        self.reliability = reliability
        self.state = "open"                 # open(구간 없음), closed(구간 진행 중), unjudgeable(판정 불가)
        self.episode_id = 0
        self.t_start = None
        self.open_run_start = None
        self.last_closed_ts = None
        self.alarmed = False
        self.missing_start = None
        self.unjudgeable_start = None
        self.last_ts = None
        self.finished = False
        self.alarms = []
        self.episodes = []
        self.unjudgeable = []

    # ----- 입력 -----
    def feed(self, timestamp_s, ear_mean, source):
        """샘플 하나를 넣는다.

        입력: timestamp_s - 초(앞 샘플보다 작으면 오류). ear_mean - 양쪽 눈 EAR 평균 또는 None(없음).
              source - detected / tracked / none
        """
        if self.finished:
            raise RuntimeError("finish 뒤에는 샘플을 넣을 수 없습니다.")
        if isinstance(timestamp_s, bool) or not isinstance(timestamp_s, (int, float)) or not math.isfinite(timestamp_s):
            raise ValueError("timestamp_s 가 유한한 수가 아닙니다: %r" % (timestamp_s,))
        if source not in SOURCES:
            raise ValueError("source 는 %s 중 하나여야 합니다: %r" % (", ".join(SOURCES), source))
        if ear_mean is not None and (isinstance(ear_mean, bool) or not isinstance(ear_mean, (int, float))
                                     or not math.isfinite(ear_mean)):
            raise ValueError("ear_mean 은 유한한 수 또는 None 이어야 합니다: %r" % (ear_mean,))
        timestamp_s = float(timestamp_s)
        if self.last_ts is not None and timestamp_s < self.last_ts - EPS:
            raise ValueError("timestamp_s 가 거꾸로 갑니다: %r 다음에 %r" % (self.last_ts, timestamp_s))
        self.last_ts = timestamp_s
        if ear_mean is None or source not in self.valid_sources:
            self._on_missing(timestamp_s)
        else:
            self._on_valid(timestamp_s, float(ear_mean))

    # ----- 얼굴 없음 -----
    def _on_missing(self, ts):
        if self.missing_start is None:
            self.missing_start = ts
        if self.state != "unjudgeable" and _at_least(ts - self.missing_start, self.unjudgeable_after_s):
            self._enter_unjudgeable()

    def _enter_unjudgeable(self):
        """얼굴 없음이 1초 이상이 되어 판정 불가로 들어간다. 진행 중인 구간은 폐기하고 알람은 유지한다."""
        if self.state == "closed":
            self.episodes.append(Episode(self.episode_id, self.t_start, None, None, STATUS_DISCARDED_UNJUDGEABLE,
                                         self.alarmed))
            self._reset_episode()
        self.state = "unjudgeable"
        self.unjudgeable_start = self.missing_start

    def _close_unjudgeable(self, end_ts):
        """판정 불가 구간을 끝낸다(제안 5)."""
        self.unjudgeable.append(Unjudgeable(self.unjudgeable_start, end_ts, end_ts - self.unjudgeable_start))
        self.unjudgeable_start = None
        self.state = "open"

    # ----- 유효 EAR -----
    def _on_valid(self, ts, ear):
        if self.state == "unjudgeable":
            self._close_unjudgeable(ts)
        elif self.missing_start is not None and _at_least(ts - self.missing_start, self.unjudgeable_after_s):
            # 샘플 사이가 1초 이상 비었다(얼굴 없음 샘플이 1초에 닿기 전에 다음 유효 샘플이 온 경우).
            self._enter_unjudgeable()
            self._close_unjudgeable(ts)
        self.missing_start = None
        if ear <= self.threshold:
            self._on_closed(ts)
        else:
            self._on_open(ts)

    def _start_episode(self, ts):
        self.state = "closed"
        self.episode_id += 1
        self.t_start = ts
        self.open_run_start = None
        self.alarmed = False

    def _reset_episode(self):
        self.t_start = None
        self.open_run_start = None
        self.alarmed = False

    def _release(self):
        """열림이 0.3초 이상 이어져 구간을 해제한다. 구간 끝은 열린 구간의 시작 시각이다."""
        end = self.open_run_start
        self.episodes.append(Episode(self.episode_id, self.t_start, end, end - self.t_start, STATUS_RELEASED,
                                     self.alarmed))
        self._reset_episode()
        self.state = "open"

    def _on_closed(self, ts):
        if self.state == "open":
            self._start_episode(ts)
        elif self.state == "closed" and self.open_run_start is not None:
            if _at_least(ts - self.open_run_start, self.release_open_s):
                self._release()
                self._start_episode(ts)
            else:
                self.open_run_start = None      # 0.3초 미만의 열림은 구간 시간에 포함한다
        self.last_closed_ts = ts
        self._check_alarm(ts, ts)

    def _on_open(self, ts):
        if self.state != "closed":
            return
        if self.open_run_start is None:
            self.open_run_start = ts
        if _at_least(ts - self.open_run_start, self.release_open_s):
            self._release()
            return
        self._check_alarm(ts, self.open_run_start)

    def _check_alarm(self, ts, confirmed_end):
        """구간 길이가 3.0초에 도달했으면 알람을 낸다(구간당 1회).

        입력: ts - 현재 샘플의 시각. confirmed_end - 구간 길이를 재는 끝(제안 2).
        """
        length = confirmed_end - self.t_start
        if not self.alarmed and _at_least(length, self.alarm_after_s):
            self.alarmed = True
            self.alarms.append(Alarm(self.episode_id, ts, self.t_start, length, self.threshold, self.alarm_after_s,
                                     self.reliability))

    # ----- 끝 -----
    def finish(self):
        """입력이 끝났음을 알리고 결과를 돌려준다."""
        if self.finished:
            return self.result()
        if self.state != "unjudgeable" and self.missing_start is not None and self.last_ts is not None \
                and _at_least(self.last_ts - self.missing_start, self.unjudgeable_after_s):
            self._enter_unjudgeable()
        if self.state == "unjudgeable":
            self._close_unjudgeable(self.last_ts)
        elif self.state == "closed":
            end = self.open_run_start if self.open_run_start is not None else self.last_closed_ts
            status = STATUS_KEPT_AT_END if self.alarmed else STATUS_DISCARDED_AT_END
            self.episodes.append(Episode(self.episode_id, self.t_start, end, end - self.t_start, status, self.alarmed))
            self._reset_episode()
            self.state = "open"
        self.finished = True
        return self.result()

    def result(self):
        """지금까지의 결과를 Result 로 만든다."""
        params = {"threshold": self.threshold, "alarm_after_s": self.alarm_after_s,
                  "release_open_s": self.release_open_s, "unjudgeable_after_s": self.unjudgeable_after_s,
                  "valid_sources": list(self.valid_sources)}
        return Result(list(self.alarms), list(self.episodes), list(self.unjudgeable), self.threshold, params)


def run_samples(samples, **kwargs):
    """샘플 시퀀스 전체를 상태기계에 넣고 결과를 돌려준다.

    입력: samples - (timestamp_s, ear_mean 또는 None, source) 시퀀스. kwargs - EpisodeMachine 설정
    반환: Result
    """
    machine = EpisodeMachine(**kwargs)
    for timestamp_s, ear_mean, source in samples:
        machine.feed(timestamp_s, ear_mean, source)
    return machine.finish()
