# windows.py
# 역할: B1 깜빡임 행([TOTAL_BLINKS, Freq*100, Amp, Dur, Vel])에서 HM-LSTM 입력 창([N, 30, 4], 열 순서 [Freq, Amp, Dur, Vel])까지의
#       계산(1단계 사람별 정규화와 창 만들기)을 순수 numpy 로 구현한다(결정 I-28, I-30). 파일을 읽거나 쓰지 않는다.
# 같은 계산의 출처: src/repro/reproduction_build_sequences.py 의 compute_reference(159~184행), normalize_table(215~225행),
#       unroll_in_time(228~243행), attach_sequences(265~266행)와 저자 Preprocessing.py 51~75행. 그 모듈은 같은 폴더 모듈을 이름으로
#       가져오는 구조라 import 하지 않고 같은 계산을 새로 구현했다. 코드는 저자 코드를 복사하지 않았다.
# B1 과 같은 점: 쓰는 열은 1~4(0열 TOTAL_BLINKS 는 쓰지 않음), 기준 통계는 기준 표의 마지막 n//3 개(n<3 이면 전체),
#       np.std ddof=0, 표준편차 0 이면 기준 표 전체 열의 표준편차로 대체, 퇴행(n<2, 대체 후에도 0 또는 비유한)은 기준 없음,
#       창 길이 30 이동 2, 마지막 불완전 창은 버림, 기준 영상은 앞쪽 n - n//3 개만 샘플로 사용.
# B1 과 다른 점: 30개 미만 영상의 처리가 선택 사항이다. pad_short=False(기본)이면 창을 만들지 않고(대기), True 이면 B1 처럼
#       앞쪽 0 패딩 창 1개를 만든다(B1 재현용, 결정 I-30 D3).
"""행 -> 창 계산(Python 3.6 호환, numpy 만 사용)."""
import collections

import numpy as np

WINDOW_SIZE = 30
STRIDE = 2
FEATURE_COLUMNS = (1, 2, 3, 4)       # 행의 열 번호: Freq*100, Amp, Dur, Vel
FEATURE_NAMES = ("Freq", "Amp", "Dur", "Vel")

Reference = collections.namedtuple("Reference", ["ref_n", "first", "means", "stds", "fallback", "fallback_kind", "reason"])


def as_table(rows):
    """행 목록을 (n, 5) float64 배열로 바꾼다. 열이 5개가 아니거나 2차원이 아니면 ValueError. 빈 입력은 (0, 5)."""
    table = np.asarray(rows, dtype=np.float64)
    if table.size == 0:
        return np.zeros((0, 5))
    if table.ndim != 2 or table.shape[1] != 5:
        raise ValueError("행은 (n, 5) 이어야 합니다: %s" % (table.shape,))
    return table


def degenerate_reason(means, stds, count):
    """기준 통계를 쓸 수 없는 이유(문자열) 또는 None. B1 degenerate_reason 과 같다: 1행 이하, 표준편차 0 또는 비유한, 평균 비유한."""
    if count < 2:
        return "row count %d" % count
    for mean, sigma in zip(means, stds):
        if sigma == 0 or not np.isfinite(sigma) or not np.isfinite(mean):
            return "std is zero or not finite (rows %d)" % count
    return None


def fallback_kind(count, fallback_columns):
    """표준편차 0 대체 방식: whole_array(n<3 이라 전체 배열 통계) / whole_std(전체 열 표준편차로 대체) / none."""
    if count < 3:
        return "whole_array"
    if fallback_columns > 0:
        return "whole_std"
    return "none"


def reference_details(rows):
    """기준 표(alert)에서 B1 compute_reference 와 같은 방식으로 정규화 기준을 계산해 Reference 로 돌려준다.

    기준 구간은 마지막 n//3 개이고 n<3 이면 값[-0:] 처럼 전체이다(B1 의 동작). 빈 표이면 None.
    reason 이 None 이 아니면 기준으로 쓸 수 없다(means, stds 는 B1 이 계산한 값 그대로 담긴다).
    """
    table = as_table(rows)
    count = len(table)
    if count == 0:
        return None
    bunch = count // 3
    ref_n = bunch if bunch > 0 else count
    means, stds, fallback = [], [], 0
    with np.errstate(all="ignore"):
        for column in FEATURE_COLUMNS:
            values = np.ascontiguousarray(table[:, column])
            mean = np.mean(values[-bunch:])
            sigma = np.std(values[-bunch:])
            if sigma == 0:
                sigma = np.std(values)
                fallback += 1
            means.append(mean)
            stds.append(sigma)
    return Reference(ref_n=ref_n, first=count - ref_n, means=np.array(means), stds=np.array(stds), fallback=fallback,
                     fallback_kind=fallback_kind(count, fallback), reason=degenerate_reason(means, stds, count))


def compute_reference(rows):
    """(means, stds) 두 길이 4 배열을 돌려준다. 기준을 쓸 수 없으면(빈 표, 퇴행) None."""
    details = reference_details(rows)
    if details is None or details.reason is not None:
        return None
    return details.means, details.stds


def normalize_table(rows, means, stds, n_used=None):
    """표의 앞쪽 n_used 개 행을 (값-평균)/표준편차 로 정규화한다. 반환: (n_used, 4) float64, 열 순서 [Freq, Amp, Dur, Vel]."""
    table = as_table(rows)
    if n_used is None:
        n_used = len(table)
    normalized = np.zeros([n_used, 4])
    for position, column in enumerate(FEATURE_COLUMNS):
        values = np.ascontiguousarray(table[:, column])
        normalized[:, position] = (values[0:n_used] - means[position]) / stds[position]
    return normalized


def unroll_in_time(normalized, pad_short=False):
    """길이 30 창을 폭 2 로 민다. 반환: (N, 30, 4) float64.

    n > 30 이면 (n-30)//2 + 1 개이고 마지막 불완전 창은 버린다. n == 30 이면 1개. n < 30 이면 pad_short=False(기본)는 창 0개,
    pad_short=True 는 B1 처럼 앞쪽 0 패딩 창 1개. 입력이 비면 창 0개.
    """
    count = len(normalized)
    if count == 0 or (count < WINDOW_SIZE and not pad_short):
        return np.zeros([0, WINDOW_SIZE, 4])
    if count <= WINDOW_SIZE:
        out = np.zeros([1, WINDOW_SIZE, 4])
        out[0, -count:, :] = normalized
        return out
    total = ((count - WINDOW_SIZE) // STRIDE) + 1
    out = np.zeros([total, WINDOW_SIZE, 4])
    for index in range(total):
        out[index, :, :] = normalized[index * STRIDE:index * STRIDE + WINDOW_SIZE, :]
    return out


def window_ranges(n_used, pad_short=False):
    """창마다 사용한 행 번호 범위(0부터, 끝 포함)의 list. 앞쪽 0 패딩은 범위에 넣지 않는다. 창 수는 unroll_in_time 과 같다."""
    if n_used == 0 or (n_used < WINDOW_SIZE and not pad_short):
        return []
    if n_used <= WINDOW_SIZE:
        return [(0, n_used - 1)]
    return [(index * STRIDE, index * STRIDE + WINDOW_SIZE - 1) for index in range(((n_used - WINDOW_SIZE) // STRIDE) + 1)]


def rows_used(count, is_reference, hold_out_reference_tail=True):
    """영상에서 샘플로 쓰는 행 수. 기준 영상(alert)은 hold_out_reference_tail=True(B1) 이면 앞쪽 n - n//3 개, 나머지는 전부."""
    if is_reference and hold_out_reference_tail:
        return count - count // 3
    return count


def video_windows(rows, means, stds, is_reference=False, pad_short=False, hold_out_reference_tail=True):
    """영상 하나의 행에서 창을 만든다(B1 attach_sequences 와 같다). 반환: ((N, 30, 4) 배열, 창마다 행 범위 list)."""
    table = as_table(rows)
    n_used = rows_used(len(table), is_reference, hold_out_reference_tail)
    normalized = normalize_table(table, means, stds, n_used)
    return unroll_in_time(normalized, pad_short), window_ranges(n_used, pad_short)
