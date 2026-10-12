# 문서 지도

작성일: 2026-10-05. 이 폴더의 문서를 **기록류**와 **정리 문서**로 나누고, 독자별 읽는 순서를 적는다. 상태 표기는 "일치/불일치/미확정/미검증"만 쓴다. 확실도는 [확실], [가능성 높음], [추측]이다.

## 1. 기록류와 정리 문서

| 문서 | 구분 | 성격 | 고치는 방식 |
|---|---|---|---|
| [REPRODUCTION_LOG.md](REPRODUCTION_LOG.md) | 기록류 | 날짜순 단계별 기록(사전 등록, 결과, 정정) | **추가만 한다.** 정정은 날짜 항목으로 쓴다 (2026-10-06 앞부분 번역 1회 예외, D-15) |
| [IMPLEMENTATION_DIFF.md](IMPLEMENTATION_DIFF.md) | 기록류 | 원본 대비 차이, 문제점 목록(P-01\~P-23), 단계별 메모(9\~16절) | 오래된 서술은 지우지 않고 "(날짜 갱신: ...)"을 덧붙인다 |
| [REPRODUCTION_REPORT.md](REPRODUCTION_REPORT.md) | 정리 문서 | 재현 보고서(요약, 방법, 결과, 해석, 문제점, 한계) | 기록류가 바뀌면 다시 정리한다 |
| [CLAIMS_LEDGER.md](CLAIMS_LEDGER.md) | 정리 문서 | 말해도 되는 주장과 금지 표현 | 같음 |
| [PIPELINE_RUNBOOK.md](PIPELINE_RUNBOOK.md) | 정리 문서 | S0~S5 실행 순서, 명령, 환경변수, 함정 | 같음 |
| [DECISIONS.md](DECISIONS.md) | 정리 문서 | 사용자 결정과 방침(날짜, 내용, 이유, 출처) | 로그에 있는 결정만 옮긴다 |
| [EVALUATION_PROTOCOL.md](EVALUATION_PROTOCOL.md) | 규칙 문서 | 개선 실험 전에 고정한 평가 규칙 | 결과를 본 뒤 바꾸지 않는다. 바꾸면 변경 이력에 날짜와 이유를 남긴다 |
| [improvement/](improvement/README.md) 폴더 | 개선 트랙 | 개선 트랙 안내, 추가 전용 기록(IMPROVEMENT_LOG.md), 개선 결정(DECISIONS.md) | 2026-10-06부터. 개선 작업의 사전 등록, 결과, 결정은 이 폴더에 쓴다 |

정리 문서의 숫자는 모두 기록류의 항목(로그 항목 제목과 날짜, 또는 IMPLEMENTATION_DIFF 절 번호)에서 가져온 것이다. 문서가 서로 다르면 기록류(날짜가 있는 항목)가 우선한다.

## 2. 독자별 읽는 순서

| 독자 | 순서 |
|---|---|
| 보고서 독자(결과만 보려는 사람) | 저장소 [README.md](../README.md)의 결과 요약 → [REPRODUCTION_REPORT.md](REPRODUCTION_REPORT.md)의 요약과 5~8절 → [CLAIMS_LEDGER.md](CLAIMS_LEDGER.md) |
| 논문 작성자(수치와 표현을 쓰는 사람) | [CLAIMS_LEDGER.md](CLAIMS_LEDGER.md) → [REPRODUCTION_REPORT.md](REPRODUCTION_REPORT.md)의 부록 B, C → 근거가 필요한 항목은 [REPRODUCTION_LOG.md](REPRODUCTION_LOG.md)의 해당 항목. 논문 원문 대조가 필요한 항목은 보고서 8절 |
| 포트폴리오 독자(재현 절차와 판단 근거를 보려는 사람) | [PIPELINE_RUNBOOK.md](PIPELINE_RUNBOOK.md) → [DECISIONS.md](DECISIONS.md) → [EVALUATION_PROTOCOL.md](EVALUATION_PROTOCOL.md) → [IMPLEMENTATION_DIFF.md](IMPLEMENTATION_DIFF.md) |
| 개선 실험을 하려는 사람 | [EVALUATION_PROTOCOL.md](EVALUATION_PROTOCOL.md)(규칙, 열린 결정) → [REPRODUCTION_REPORT.md](REPRODUCTION_REPORT.md)의 5절(기준선 수치) → [improvement/README.md](improvement/README.md) |

포트폴리오 형식으로의 재구성은 데모 이후에 한다(이 지도는 현재 문서의 읽는 순서만 적는다).

## 3. 트랙 구조

2026-10-06부터 작업을 두 트랙으로 나눈다(결정 D-14, 재현 로그 "재현 트랙 동결과 개선 트랙 분리 (2026-10-06)").

| 구분 | 위치 | 상태 |
|---|---|---|
| 재현 트랙 | `src/repro/`, `third_party/`, `experiments/00_reproduction/`, `01_fold1_seeds/`, `02_fold_seeds/`, 이 폴더의 문서 8개(README 제외 기록류, 정리 문서, 규칙 문서) | 동결. 정정은 날짜 항목으로만 추가한다 |
| 개선 트랙 | `docs/improvement/`, `src/improve/`, `experiments/improve/`, `configs/improve/` | 진행 |
| 경계 문서 | [EVALUATION_PROTOCOL.md](EVALUATION_PROTOCOL.md) | 평가 규칙. 변경은 이력에 기록한다 |

의존 방향: 개선 쪽은 재현 쪽의 산출물을 읽기만 하고, 재현 쪽은 개선 쪽을 import하지 않는다.
