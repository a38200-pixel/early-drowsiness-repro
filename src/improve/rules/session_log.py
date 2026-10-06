# 역할: 판정 로그(JSON Lines)와 사용자 메모를 출력 폴더에 쓰는 함수(C3). docs/improvement/D1_REQUIREMENTS.md C3 를 따른다.
# 입력: out_dir(필수, 저장소 안의 기본값 없음), run_id(실행 식별자), 판정 로그 레코드, 메모(타임스탬프, 텍스트, 선택 라벨).
# 출력: <out_dir>/decisions_<run_id>.jsonl, <out_dir>/memos_<run_id>.jsonl. 한 줄에 JSON 하나.
# 영상, 프레임, 이미지를 저장하는 기능은 만들지 않는다. 영상 파일 이름은 run_id 로 쓸 수 없다(확장자와 경로 구분자 거부).
# 알람을 보기 전에 메모를 먼저 기록할지는 미확정이다(I-W-09). 이 모듈은 순서를 강제하지 않는다.
# 판정 로그 레코드의 열은 제안이다(미확정): timestamp_s, ear_mean, face_detected, source, state, alarm, threshold.
import json
import math
import os
import re

RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
DECISION_FIELDS = ("timestamp_s", "ear_mean", "face_detected", "source", "state", "alarm", "threshold")


def _check_run_id(run_id):
    """run_id 는 영문자, 숫자, 밑줄, 하이픈 1~64자만 허용한다(점과 경로 구분자 거부)."""
    if not isinstance(run_id, str) or not RUN_ID_PATTERN.match(run_id):
        raise ValueError("run_id 는 영문자, 숫자, 밑줄, 하이픈 1~64자여야 합니다: %r" % (run_id,))


def _check_timestamp(timestamp_s):
    if isinstance(timestamp_s, bool) or not isinstance(timestamp_s, (int, float)) or not math.isfinite(timestamp_s):
        raise ValueError("timestamp_s 가 유한한 수가 아닙니다: %r" % (timestamp_s,))


def make_decision_record(timestamp_s, ear_mean, face_detected, source, state, alarm, threshold):
    """판정 로그 레코드(dict)를 만든다. 열은 DECISION_FIELDS 와 같다."""
    _check_timestamp(timestamp_s)
    return {"timestamp_s": float(timestamp_s), "ear_mean": ear_mean, "face_detected": bool(face_detected),
            "source": source, "state": state, "alarm": bool(alarm), "threshold": threshold}


class SessionLog(object):
    """한 실행의 판정 로그와 메모를 출력 폴더에 이어 쓴다."""

    def __init__(self, out_dir, run_id):
        """출력 폴더와 실행 식별자를 받는다. 이미 있는 로그 파일에는 이어 쓰지 않고 오류를 낸다.

        입력: out_dir - 출력 폴더(필수, 없으면 만듦). run_id - 실행 식별자(영상 이름을 쓰지 않는다)
        """
        if not out_dir:
            raise ValueError("출력 폴더가 필요합니다(기본값 없음).")
        _check_run_id(run_id)
        os.makedirs(out_dir, exist_ok=True)
        self.decisions_path = os.path.join(out_dir, "decisions_%s.jsonl" % run_id)
        self.memos_path = os.path.join(out_dir, "memos_%s.jsonl" % run_id)
        for path in (self.decisions_path, self.memos_path):
            if os.path.exists(path):
                raise FileExistsError("이미 있는 로그 파일에는 쓰지 않습니다: %s" % os.path.basename(path))
        # 두 파일을 비워 둔 채 만들어 두면 읽는 쪽이 파일 유무로 실행을 알 수 있다.
        for path in (self.decisions_path, self.memos_path):
            with open(path, "x", encoding="utf-8"):
                pass

    def log_decision(self, record):
        """판정 로그 레코드 하나를 한 줄로 이어 쓴다.

        입력: record - make_decision_record 가 만든 dict(DECISION_FIELDS 를 모두 가져야 함)
        """
        missing = [name for name in DECISION_FIELDS if name not in record]
        if missing:
            raise ValueError("판정 로그 레코드에 열이 없습니다: %s" % ", ".join(missing))
        _check_timestamp(record["timestamp_s"])
        with open(self.decisions_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    def add_memo(self, timestamp_s, text, label=None):
        """사용자 메모 하나를 한 줄로 이어 쓴다.

        입력: timestamp_s - 메모가 가리키는 시각(초). text - 메모 내용. label - 선택 라벨(없으면 None)
        """
        _check_timestamp(timestamp_s)
        if not isinstance(text, str) or not text.strip():
            raise ValueError("메모 text 는 비어 있지 않은 문자열이어야 합니다.")
        record = {"timestamp_s": float(timestamp_s), "text": text, "label": label}
        with open(self.memos_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path):
    """JSON Lines 파일을 읽어 dict 목록으로 돌려준다(빈 줄은 건너뜀).

    입력: path - 파일 경로
    반환: list of dict
    """
    records = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records
