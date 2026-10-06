# 개선 트랙 안내

작성일: 2026-10-06. 이 폴더는 재현 트랙이 동결된 뒤의 개선 작업을 기록한다. 상태 표기는 "일치/불일치/미확정/미검증"만 쓴다. 확실도는 [확실], [가능성 높음], [추측]이다.

## 1. 범위

- 눈 기반 3초 규칙(눈 감김 지속 알람)과 영상 데모. [확실] (사용자 결정, [IMPROVEMENT_LOG.md](IMPROVEMENT_LOG.md)의 첫 항목)
- 하품과 고개 떨굼은 지금 검출하지 않는다. [확실]

## 2. 재현 트랙과의 경계

| 구분 | 재현 트랙 (동결) | 개선 트랙 |
|---|---|---|
| 코드 | `src/repro/`, `third_party/` | `src/improve/rules/`, `src/improve/demo/` |
| 실험 | `experiments/00_reproduction/`, `01_fold1_seeds/`, `02_fold_seeds/` | `experiments/improve/` |
| 설정 | 기존 `configs/` 파일 | `configs/improve/` |
| 문서 | `docs/`의 기존 문서 8개 | `docs/improvement/` |
| 결정 ID | D-xx | 결정 I-xx, 실험 E-xx |
| 경계 문서 | [EVALUATION_PROTOCOL.md](../EVALUATION_PROTOCOL.md) (평가 규칙, 변경은 이력에 기록) | 같음 |

의존 방향: 개선 쪽은 재현 쪽의 산출물(B1 체크포인트, 시퀀스, 기준선 수치)을 **읽기만** 하고 수정하지 않는다. 재현 쪽은 개선 쪽을 import하지 않는다. [확실] (방침)

## 3. 문서

| 문서 | 성격 | 고치는 방식 |
|---|---|---|
| [IMPROVEMENT_LOG.md](IMPROVEMENT_LOG.md) | 날짜순 기록(사전 등록, 결과, 정정) | **추가만 한다** |
| [DECISIONS.md](DECISIONS.md) | 개선 결정과 결정 대기 | 로그에 있는 결정만 옮긴다 |
| [RULE_SPEC.md](RULE_SPEC.md) | 규칙 가지 명세(EAR 정의, 닫힘 판정, 감김 구간 상태기계, 유효성 검사, 민감도 분석 계획, 부록 문헌 표) | 버전을 올리고 변경 이력에 남긴다 |
| [BASELINE_OUTPUTS.md](BASELINE_OUTPUTS.md) | B1 기준 출력 보관 명세(생성은 후속 작업) | 변경 이력에 남긴다 |
| [D1_REQUIREMENTS.md](D1_REQUIREMENTS.md) | D1 구현 단계 C1~C5의 요구사항(코드 없음) | 변경 이력에 남긴다 |
| [PREREG_E-01.md](PREREG_E-01.md) | E-01 사전 등록(규칙 가지의 데모 영상 작동 확인, 개선 주장 아님) | 커밋 뒤에는 기존 문장을 고치지 않고 일탈 표에 추가한다 |

## 4. 읽는 순서

1. [../EVALUATION_PROTOCOL.md](../EVALUATION_PROTOCOL.md): 평가 규칙과 기준선 수치
2. [IMPROVEMENT_LOG.md](IMPROVEMENT_LOG.md): 개선 트랙의 시작과 범위 결정
3. [DECISIONS.md](DECISIONS.md): 확정된 결정과 아직 정하지 않은 결정
4. [../README.md](../README.md): 재현 트랙 문서 지도
