# early-drowsiness-repro

졸음 조기 탐지 논문(arXiv 1904.07312)의 눈 깜빡임 기반 baseline을 공개 코드로 재현해 점검하고, 재현한 부분을 눈 감김 규칙 알람과 참고용 점수 게이지로 나누어 활용하는 방법을 만든 개인 프로젝트입니다.

**핵심 요약**

- **무엇을 한 프로젝트인가:** 졸음 조기 탐지 논문의 눈 깜빡임 기반 baseline을 공개 코드로 재현하고, 그 위에 눈 감김 규칙 알람과 참고용 점수 게이지를 만든 개인 프로젝트입니다.
- **확인한 것:** 공개 체크포인트 평가는 논문 표와 14칸 중 11칸이 일치하고 3칸이 일치하지 않았습니다. 점수 시간선 도구는 저장된 B1 결과와 종단 대조에서 일치했습니다.
- **한계:** 부분 재현이며, 점수는 참고용이고 알람이 아니며, 졸음 검출을 주장하지 않습니다. 사건 대조는 한 사람의 영상 한 건이고 Windows에서만 실행해 봤습니다.

## 1. 프로젝트 개요

### 문제 정의

**논문이 다루는 문제.** 영상에서 졸음의 초기 단계를 알아채는 것입니다. 논문은 눈 깜빡임에서 뽑은 특징(깜빡임 빈도, 진폭, 지속시간, 속도)의 시퀀스를 HM-LSTM 기반 시간 모델에 넣어 졸음 정도 라벨(클래스 0, 5, 10)을 추정하고, 이를 위한 데이터셋 UTA-RLDD를 함께 제안합니다. 논문 원문(PDF)은 이 저장소에 없습니다.

**이 저장소가 다루는 문제.** 논문 방법의 눈 기반 부분을 공개 코드로 재현해 일치하는 것과 일치하지 않는 것을 점검하고(재현 트랙), 재현한 부분을 둘로 나누어 활용하는 방법을 만드는 것입니다(개선 트랙). 하나는 눈 감김이 3.0초 이상 이어지면 울리는 실시간 규칙 알람이고, 다른 하나는 HM-LSTM 점수를 0에서 10 눈금으로 보여 주는 참고용 게이지입니다. 점수는 알람과 연결하지 않습니다(결정 I-33). 하품과 고개 떨굼은 다루지 않습니다(결정 I-01).

### 목표

| 목표 | 근거 문서 | 상태 |
|---|---|---|
| 기준선 B1 재현: 우리가 영상에서 다시 뽑은 blink로 저자 코드와 하이퍼파라미터 그대로 학습 | [EVALUATION_PROTOCOL.md](docs/EVALUATION_PROTOCOL.md), [REPRODUCTION_REPORT.md](docs/REPRODUCTION_REPORT.md) | 일치 11칸, 불일치 3칸(공개 체크포인트와 논문 표 대조). 영상 집합이 달라 B1과 논문은 같은 비교가 아님(미확정) |
| 눈 감김 규칙 알람(3.0초) 구현과 사전 등록 실험 | [DECISIONS.md](docs/improvement/DECISIONS.md) I-03에서 I-08, [PREREG_E-01.md](docs/improvement/PREREG_E-01.md), [PREREG_E-02.md](docs/improvement/PREREG_E-02.md) | E-01 불일치(영상 1개), E-02 일치(영상 1개). 3.0초 경계의 실제 영상 시험은 미검증 |
| 실시간 웹캠 데모 | I-20에서 I-22 | 사용자가 직접 실행해 처리 속도를 측정(결정 I-W-21). 웹캠 라이브 점수 표시는 구현 전(I-W-33 미결) |
| HM-LSTM 점수 게이지(참고용) | I-24, I-27, I-33 | 오프라인 점수 시간선과 표시 규칙은 일치(종단 대조 D4). GUI 통합은 미확정(I-W-33 미결) |
| 재현 가능한 검증 절차 | 사전 등록 문서, 오라클 기록 | 일치(32/0, 32/5의 종단 대조, 32/10_2의 행 대조) |

### 기간

2026-09-30에서 2026-10-11까지 (커밋 이력 기준, 양 끝 날짜를 포함해 12일).

### 역할

개인 프로젝트(1인). 연구 설계, 검증 기준과 사전 등록, 결과 해석과 설계 결정은 본인이 담당했고, 코드 구현과 문서 초안 작성에는 AI 코딩 도구(Claude Code)를 활용했습니다.

### 기술 스택

[envs/environment_tf1.yml](envs/environment_tf1.yml)에서 확인되는 버전입니다.

| 구분 | 이 저장소가 직접 만든 코드(`src/`) | 저자 코드(`third_party/ghoddoosian/`, 수정 없이 vendoring) |
|---|---|---|
| 언어와 도구 | Python 3.6.13, conda 환경 `ghoddoosian_tf1`, `unittest` 시험 | Python |
| 학습과 모델 | 학습 러너(저자 `Training.py`를 메모리에서 치환해 실행), 시퀀스 생성, 평가와 진단(scikit-learn 0.19.2의 로지스틱, RF) | HM-LSTM 구현과 학습 코드, TensorFlow 1.8.0 |
| 영상 처리 | 눈 감김 규칙(EAR), 웹캠 데모, 점수 시간선 도구 | blink 추출(`blink_video.py`): OpenCV 3.4.5(opencv-python 3.4.5.20), dlib 19.17 얼굴 랜드마크, imutils 0.5.4, blink 판별 SVM(`.sav`) |
| 수치와 그림 | numpy 1.14.6, scipy 1.2.1, pandas 1.1.5, matplotlib 2.2.5 | numpy |

### 실행 환경 및 지원 범위

| 구분 | 지원 | 미지원 | 미검증 |
|---|---|---|---|
| 운영체제와 환경 | Windows, conda 환경(Python 3.6, TensorFlow 1.8) | - | Linux, macOS |
| 연산 장치 | CPU(TensorFlow 1.8.0 CPU 패키지로 체크포인트 복원과 추론 확인) | - | GPU |
| 웹캠 실시간 데모 | `live_demo.py`: 화면 26에서 27fps(학습 종료 후), 학습이 CPU 코어 4개를 쓰던 시기에는 6에서 7fps(둘 다 사용자 측정, 결정 I-W-21) | 점수 기반 알람(결정 I-33) | 웹캠 입력으로 HM-LSTM 점수 표시(게이지 GUI 통합은 미결, I-W-33) |
| 녹화 영상 처리 | 규칙 알람 재생, 점수 시간선 도구. 회전 메타데이터는 `blink_video_dump.py --rotate auto`로 처리(결정 I-19: E-02 영상 2개는 메타데이터 270도) | - | 회전 메타데이터가 있는 영상의 `timeline.py extract --rotate auto` 경로 |
| 추출 속도(원본 blink 추출) | 13.5에서 15.8fps(무부하: 데이터셋 영상 15.5, 15.8fps, 사용자 영상 14.3, 13.5fps) | - | 다른 하드웨어에서의 속도 |
| 신호 | 눈 기반(눈 감김 EAR, 깜빡임 특징 4개)만 | 하품, 고개 떨굼 | 다른 촬영 조건 |

학습이 동시에 실행 중일 때 원본 blink 추출은 2.8에서 4.7fps였습니다(실측, 부하 조건).

**필요한 준비물(저장소에 포함되지 않음, 자리표시자 경로).**

| 자리표시자 | 내용 |
|---|---|
| `<DATA_DIR>` | UTA-RLDD 영상 폴더(`RLDD_DATA_DIR`) |
| `<PREDICTOR>` | dlib 68점 랜드마크 모델 `shape_predictor_68_face_landmarks.dat`(`DLIB_LANDMARK_PATH`) |
| `<CKPT_DIR>` | B1 학습 출력 루트(점수 도구가 체크포인트를 읽는 곳, 저장소 밖) |
| `<VIDEO_DIR>` | 처리할 영상 폴더 |
| `<TIMELINE_DIR>` | 저장소 밖 숫자 JSON과 그림 출력 폴더 |

## 2. 한 것 / 하지 않은 것

| 한 것 | 하지 않은 것 |
|---|---|
| 공개 체크포인트 5개를 5-fold로 평가하고 논문 표와 대조(14칸 중 11칸 일치, 3칸 불일치) | 영상 수 174/180의 확정(저자가 어떤 영상을 제외했는지는 미확인) |
| 저자 npy로 재학습(시드 3개, 80 epoch, 저자 하이퍼파라미터 그대로) | 저자 npy와 우리 재추출 특징의 영상 단위 일치(일치한 영상은 피험자 2명, 원인 미검증) |
| 영상 182개 조사, 174개 blink 재추출, 167개(피험자 57명)로 B1 학습(시드 3개, 5-fold) | B1 대비 개선 실험과 개선 주장 |
| 눈 감김 3.0초 규칙 알람, 사전 등록 실험 2건(E-01, E-02), 웹캠 데모 | 3.0초 경계의 실제 영상 시험(합성 시험 12개로만 고정) |
| 원본 blink 추출기를 메모리에서 치환 실행해 B1 행과 대조, 행별 프레임 번호 기록 | 점수 기반 알람, 졸음 검출 주장, 점수의 타당성 검증 |
| 5개 폴드 체크포인트로 점수 시간선을 만드는 도구와 표시 규칙(참고용), 종단 대조(D4) | 게이지 GUI 통합, 웹캠 라이브 점수 |
| | 하품과 고개 떨굼 검출 |

## 3. 배경

대상 논문은 *A Realistic Dataset and Baseline Temporal Model for Early Drowsiness Detection*(Ghoddoosian, Galib, Athitsos, CVPRW 2019, arXiv 1904.07312)입니다. 이 저장소는 그 baseline을 원저자의 공개 코드로 UTA-RLDD에서 다시 돌려 보고 어디까지 같은 값이 나오는지 기록하는 **부분 재현**입니다. 원본 코드는 수정하지 않았고(바이트 동일, 보호 목록 44개 파일), 재현을 위해 추가한 것은 별도 스크립트와 로그로 분리했습니다([docs/REPRODUCTION_REPORT.md](docs/REPRODUCTION_REPORT.md)). 논문과 원본 코드 사이에는 이산화 경계(3.3과 6.6 대 3.34와 6.68), 피험자별 정규화 구간(처음 1/3 대 마지막 1/3), 영상 수(180 대 174) 같은 불일치가 있고 원본을 그대로 따랐습니다([docs/CLAIMS_LEDGER.md](docs/CLAIMS_LEDGER.md) C-13).

## 4. 저장소 구조

```
third_party/ghoddoosian/   원본 코드, 체크포인트, 특징 파일(44개 파일, 수정 없음, MIT; THIRD_PARTY.md 참고)
src/repro/                 재현 트랙 스크립트 27개(영상 조사, blink 재추출, 시퀀스 생성, 학습, 평가, 진단). 각 파일 맨 위 주석에 역할과 사용법
src/improve/rules/         눈 감김 규칙 알람과 웹캠 데모: 모듈 11개, 시험 파일 11개(live_demo.py, episode_machine.py, replay.py 등)
src/improve/hmlstm/        HM-LSTM 점수 도구: 모듈 11개, 시험 파일 9개(timeline.py, gauge.py, scoring.py, verify_blinks.py 등)
scripts/                   run_fold1_seeds.bat, run_folds_seed.bat (재학습 일괄 실행)
experiments/00_reproduction/   공개 체크포인트 평가 결과(CSV, 요약, 로그, regression_eval)와 Fold1 학습 기록
experiments/01_fold1_seeds/    Fold1 시드 1에서 5 재학습 결과(seed1에서 seed5)와 스모크 시험
experiments/02_fold_seeds/     Fold2에서 5의 시드 1에서 3 재학습 결과, 시드별 요약, 집계 파일(aggregate_seeds123.md, .csv)
configs/                   protected_manifest.sha256(44개 파일 해시), image_publishable.csv, 영상 목록과 회전 조사 CSV 4개
envs/environment_tf1.yml   conda 환경(Python 3.6, TensorFlow 1.8)
data/README.md             데이터 입수 방법(저장소에는 데이터를 넣지 않음)
docs/                      재현 트랙 문서 8개와 개선 트랙 문서(docs/improvement/, 마크다운 9개와 e02_truth.json)
```

`src/improve/demo/`, `experiments/improve/`, `configs/improve/`, `docs/figures/`는 자리만 있고 비어 있습니다(`.gitkeep`). 최상위 폴더는 8개(`third_party`, `src`, `scripts`, `experiments`, `configs`, `envs`, `data`, `docs`)입니다. 재현 트랙과 개선 트랙의 경계와 의존 방향은 [docs/improvement/README.md](docs/improvement/README.md)에 있습니다.

## 5. 기준선 B1 재현

### 무엇을 다시 만들었나

영상에서 blink 특징을 다시 추출하고(저자 `blink_video.py`를 수정 없이 실행), 시퀀스를 만들고, 저자 `Training.py`를 메모리에서 치환해 하이퍼파라미터 그대로 학습했습니다. 영상 182개를 조사해 174개를 추출했고, B1에는 167개(피험자 57명)를 포함했습니다(제외 15개는 재추출 데이터에서의 기계적 결과). 산출물(blink txt, 시퀀스 npy, 학습 출력)은 용량과 데이터 이용 지침 때문에 저장소 밖에 둡니다([docs/PIPELINE_RUNBOOK.md](docs/PIPELINE_RUNBOOK.md)).

### 결과

수치는 [aggregate_seeds123.md](experiments/02_fold_seeds/aggregate_seeds123.md)와 [docs/REPRODUCTION_REPORT.md](docs/REPRODUCTION_REPORT.md)의 값 그대로입니다. 약어는 원본 `Training.py`가 출력하는 지표 이름이고 VA는 영상 단위 정답률입니다.

| 항목 | 값 | 상태 |
|---|---|---|
| 공개 체크포인트 평가 vs 논문 Table 2와 3 | 비교 가능한 14칸 중 11칸 일치, 3칸 불일치(Fold3 VA 논문 0.700 대 0.69, Fold3 VRE 0.580 대 0.67, 평균 VRE 1.140 대 1.16). 원인 미확인 | 일치 11, 불일치 3 |
| B0: 저자 npy 재학습 5-fold 평균 VA | 0.556 ± 0.036(시드 3개, SD는 n−1). 공개 체크포인트 0.649, 논문 0.652 | 불일치(원인 미검증) |
| B1: 우리 재추출 데이터 5-fold 평균 VA | 0.586 ± 0.015(시드 3개, 마지막 epoch), VRE 1.498 ± 0.282 | 영상 집합이 달라 위 값과 같은 비교가 아님(미확정) |
| B1 클래스별 영상 정답률(시드 합산) | 클래스 0 0.731, 클래스 5 0.327, 클래스 10 0.698 | 클래스 5가 세 시드 모두 가장 낮음 |
| 영상 요약 특징만 쓴 진단 분류기(재현이 아닌 한계 분석) | 로지스틱 0.540, RF 0.559, 다수 클래스 0.341 | 참고 |
| B1 VA의 95% 구간(피험자 군집 부트스트랩) | 0.5848 [0.5302, 0.6380] | - |
| B1 − RF의 VA 차이 | +0.0279 [−0.0575, 0.1118] | 구간이 0을 포함해 구분할 수 없음(차이가 없다는 증거는 아님) |

- 보고값은 epoch 80의 단일값이며 사후 선택을 하지 않았습니다. 시드가 3개뿐이어서 SD는 참고 수준입니다.
- 재추출한 특징은 저자 npy와 영상 단위로 거의 일치하지 않습니다(값이 일치한 영상은 피험자 2명, 원인 미검증).
- 원본 대비 차이 목록: [docs/IMPLEMENTATION_DIFF.md](docs/IMPLEMENTATION_DIFF.md). 오류를 포함한 단계별 기록: [docs/REPRODUCTION_LOG.md](docs/REPRODUCTION_LOG.md).

### 실행 방법

conda가 필요합니다. 명령은 저장소 루트(`<REPO>`)에서 실행합니다.

```
conda env create -f envs/environment_tf1.yml
conda activate ghoddoosian_tf1

python src/repro/check_environment.py      # 인터프리터와 패키지 버전
python src/repro/check_dataset.py          # .npy 입력과 my_model1..5 체크포인트
python src/repro/reproduction_eval.py 5    # 공개 체크포인트 5-fold 평가
python src/repro/aggregate_seeds.py        # 재학습 3시드 집계
```

`reproduction_eval.py`는 `experiments/00_reproduction/regression_eval/`의 기존 파일을 덮어씁니다. 보존하려면 먼저 옮겨 두세요. 실행 시작 시 vendoring된 44개 파일을 `configs/protected_manifest.sha256`과 대조하고, 하나라도 다르면 중단합니다.

영상이 필요한 선택 단계(`data/README.md` 참고):

```
set DLIB_LANDMARK_PATH=<PREDICTOR>
set RLDD_DATA_DIR=<DATA_DIR>
python src/repro/check_video_assets.py
python src/repro/reproduction_blink_one_video.py <영상 경로>
```

영상 재추출, 시퀀스 생성, B1 학습, 평가의 명령과 환경변수는 [docs/PIPELINE_RUNBOOK.md](docs/PIPELINE_RUNBOOK.md)에 있습니다. `src/repro/reproduction_train_fold1.py`는 레거시이며 기존 결과를 덮어쓸 수 있으니 쓰지 마세요.

## 6. 눈 감김 규칙 알람과 live_demo

**동작(RULE_SPEC v1, [RULE_SPEC.md](docs/improvement/RULE_SPEC.md)).** 프레임마다 양쪽 눈의 EAR 평균을 구해 임계값 이하이면 닫힘으로 봅니다. 닫힘이 3.0초 이상 이어지면 알람이고(결정 I-05), 눈이 뜬 상태가 연속 0.3초 이상이면 감김 구간을 해제합니다(I-06). 얼굴이 1초 이상 검출되지 않으면 "판정 불가"로 구간을 폐기합니다(I-07). 시간은 프레임이 들어온 시각을 씁니다. 데모의 기본 임계값은 0.15이고(I-22, 사용자 1명의 체험 판단으로 고정), 사전 등록 실험은 0.15, 0.21, 0.25를 함께 비교했습니다.

**사전 등록 실험 결과.**

| 실험 | 영상 | 결과 | 상태 |
|---|---|---|---|
| E-01 | 운전자 4명을 이어 붙인 편집본 1개, 감김 사건 6건 | 주 조건(0.21)에서 일치 2건. 사전 예측 3개 중 1개 일치 | 불일치(규칙 정의 재검토) |
| E-02 | 대본 영상 1개(눈 감기 3건: 길이 6, 6, 7초로 양성 기준 3.5초 이상을 넘음, 짧은 감기 2건, 아래 보기, 찡그리기) | 양성 사건 3건 중 3건 일치, 음성 창 알람 0(6개 조합 모두), 사전 예측 5개 중 5개 일치. 0.21에서 추가 알람 2개는 모두 아래 보기 구간(사용자 육안, 블라인드 아님) | 일치(이 영상에서) |

**실행 방법**(`<REPO>`에서 `PYTHONPATH=src`, `<EAR_DIR>`는 저장소 밖 출력 폴더):

```
python -m improve.rules.live_demo --predictor <PREDICTOR> [--camera 0] [--beep] [--thresholds 0.15,0.21]
python -m improve.rules.blink_video_dump --video <VIDEO> --out-dir <EAR_DIR> --predictor <PREDICTOR> --svm <SVM>
python -m improve.rules.replay --input <EAR_DIR>/ear_dump.csv --out-dir <EAR_DIR>/replay --thresholds 0.15 0.21 0.25 --valid-sources detected
```

데모는 아무것도 저장하지 않으며 알람 표시는 R 키로 끕니다(q 종료, m 음소거).

**한계.** 3.0초 경계는 합성 입력 시험 12개로만 고정되어 있고 실제 영상으로 시험하지 않았습니다(미검증). E-02는 영상 한 개이고 정답 시각에 ±1초 오차가 있으며 사용자가 알람 시각을 알고 본 기록입니다. 임계값 선택은 사용자 1명의 체험 판단입니다.

## 7. HM-LSTM 점수 게이지(참고용)

**확정 문구: "HM-LSTM score (0~10), 참고용, 알람 아님".** 점수는 5개 폴드 체크포인트(B1 시드 1)의 평균과 최솟값에서 최댓값 범위로 보이며(I-27), 눈 감김 규칙 알람과 연결하지 않습니다. 알람과 분리한 이유: 점수가 졸음을 반영하는지 검증하지 못했고, 클래스 5와 10의 구분이 약하며 폴드 사이 차이가 크기 때문입니다(I-33, 아래 수치). 0 쪽은 학습 라벨의 깨어 있는 쪽 끝, 10 쪽은 졸음 쪽 끝입니다.

**도구**(`src/improve/hmlstm/`, 영상 처리와 TensorFlow는 별도 프로세스):

| 도구 | 하는 일 |
|---|---|
| `timeline.py extract` | 영상에서 깜빡임 행과 행별 프레임 번호를 뽑아 저장소 밖 JSON으로 저장(저장소 안 경로와 덮어쓰기는 거부) |
| `timeline.py score` | 평소영상 행의 뒤 1/3로 기준을 구하고, 평소영상 앞쪽과 대상 영상을 같은 기준으로 정규화해 5개 폴드 점수 시간선 JSON을 만듦 |
| `gauge.py` | 시간선에서 프레임별 게이지 표시 상태와 문구를 정하는 순수 함수(블링크 30개가 확정되기 전에는 값을 내지 않음) |
| `timeline_plot.py` | 시간선 그림(PNG)과 사건 대조 표를 만듦. 입력은 숫자 JSON뿐이며 영상 프레임을 읽지 않음 |

```
python -m improve.hmlstm.timeline extract --video <VIDEO_DIR>/<영상>.mp4 --svm <SVM> --predictor <PREDICTOR> --out <TIMELINE_DIR>/rows.json --rotate auto
python -m improve.hmlstm.timeline score --baseline <TIMELINE_DIR>/baseline_rows.json --target <TIMELINE_DIR>/target_rows.json --ckpt-root <CKPT_DIR> --out <TIMELINE_DIR>/timeline.json
```

**오프라인 분석에서 확인된 한계**(영상 167개, 폴드별 검증 체크포인트, 시드 1, 패딩 창 포함판, 로그 "오프라인 점수 분석 결과 (2026-10-10)"):

| 비교(영상 최댓값 기준 AUC, 95% 구간) | 값 |
|---|---|
| 클래스 10 대 0 | 0.93 [0.88, 0.97] |
| 클래스 5 대 0 | 0.89 [0.84, 0.94] |
| 클래스 10 대 5 | 0.61 [0.53, 0.70] |

클래스 0 영상은 창이 적어(영상당 평균 클래스 0 24.8개, 클래스 10 59.4개) 최댓값 기준 지표는 영상 길이의 영향을 받습니다.

- 길이 영향이 섞여 있을 수 있는 클래스 0과 나머지의 구분 지표(AUC)는 0.89에서 0.93이지만, 클래스 5와 10의 구분 지표는 0.61입니다.
- 사용자 영상 2개에서 같은 창의 5개 모델 점수 차이(최대 − 최소)의 평균은 2.76(평소영상)과 3.25(대본영상)입니다.
- 점수 기반 알람은 채택하지 않았고(결정 I-33, I-W-30) 모의 결과는 [IMPROVEMENT_LOG.md](docs/improvement/IMPROVEMENT_LOG.md)의 오프라인 점수 분석 항목에 있습니다.

**사용자 영상 사례**(로그 "도구 구현, 측정 기록과 결정 I-34", "D4 종단 오라클 결과, 대본영상 대조, 그림 도구, 실측 시간"의 값):

- 사례 한 건이며 졸음 반영 여부는 이 자료로 말하지 않습니다.
- 평소 영상: 프레임 9101, 블링크 행 193, 창 50, 창 평균 점수 1.93에서 5.31(평균 3.20). 대본 영상: 프레임 5733, 블링크 행 108, 창 40, 창 평균 점수 2.72에서 9.50.
- 대본 영상의 첫 점수는 70.00초에 나옵니다(그때 블링크 30개가 확정). 눈 감기 구간에는 블링크 행이 거의 생기지 않습니다.
- 눈 감기 사건과 겹친 창 14개의 평균은 8.44, 겹치지 않은 창 26개는 5.57입니다. 겹치는 창이 모두 영상 앞부분이어서 사건과 시간 순서를 분리할 수 없습니다.
- 아래 보기 또는 찡그리기와 겹친 창 20개의 평균은 7.95, 둘 다 겹치지 않은 창 20개는 5.20입니다.

## 8. 검증과 재현성

| 검증 | 결과 | 상태 |
|---|---|---|
| 종단 대조 D4(영상 → 훅 → 행 → 창 → 점수), 32/0 | 행 58개가 B1과 같고 열별 최대 차이 0, 폴드 3 영상 평균 4.621983(B1 4.621983, 차이 4.732e-07, 허용 1e-5) | 일치 |
| 종단 대조 D4, 32/5 | 행 140개가 B1과 같고 열별 최대 차이 0, 폴드 3 영상 평균 7.463169(B1 7.463170, 차이 6.041e-07) | 일치 |
| 프레임 번호 대조, 32/10_2 | 행 9개, 프레임 3789, 첫 행 945, 마지막 행 3547, B1 행과 최대 차이 0 | 일치 |
| 폴드 1에서 5 점수 대조(2a) | 창 값 차이 0, 점수 영상 평균 차이 2.4e-07에서 8.7e-07 | 일치 |

시험은 `rules` 233개(`src/improve/rules/tests`)와 `hmlstm` 191개(`src/improve/hmlstm/tests`)입니다(문서 기록 시점 기준). `hmlstm`은 환경변수 `HMLSTM_CKPT_ROOT`를 설정하면 전부 통과하고, 설정하지 않으면 3개를 건너뜁니다. 실행: `python -m unittest discover -s src/improve/hmlstm/tests -t src` (`PYTHONPATH=src`). 평가 규칙과 사전 등록 문서는 결과를 본 뒤 바꾸지 않고 변경 이력이나 일탈 표에 추가합니다.

## 9. 한계와 정직한 서술

- 사건과 점수의 대조는 대본영상 한 건(한 사람의 한 번 촬영)입니다. 이 자료로는 점수가 졸음을 반영하는지 말할 수 없습니다.
- 졸음 검출을 주장하지 않습니다. 규칙 알람은 눈 감김 규칙 하나뿐이고 점수는 참고용입니다. 의료 기기도, 안전 장치도 아닙니다.
- B1 대비 개선을 주장하지 않습니다. B1과 논문의 수치는 영상 집합이 달라 같은 비교가 아닙니다.
- 재현은 부분 재현입니다. 공개 체크포인트와의 차이의 원인은 확인하지 못했습니다.
- 환경이 오래되었고(Python 3.6, TensorFlow 1.8) Windows에서만 실행해 봤습니다.
- 확정되지 않은 수치는 이 문서에 적은 것 외에는 의도적으로 싣지 않았습니다.

## 10. 데이터와 개인정보 정책

- UTA-RLDD 영상, dlib 랜드마크 모델, 그로부터 만든 프레임은 **포함하지 않습니다**. 받는 방법은 [data/README.md](data/README.md)에 있습니다.
- 데이터셋의 얼굴 이미지는 `configs/image_publishable.csv`에 있는 36명(공식 데이터셋 페이지 표의 'YES')만 공개할 수 있고, 반드시 `src/repro/publish_guard.py`를 통해 저장합니다(결정 I-32). 본인 웹캠 화면의 스크린샷은 올릴 수 있고, 사용자가 촬영한 원본 영상 파일과 데이터셋 영상은 올리지 않습니다(결정 I-32, I-34).
- 점수 도구가 만드는 숫자 JSON(깜빡임 행, 프레임 번호, 창별 점수)은 저장소 밖 폴더(`<TIMELINE_DIR>`)에만 저장하며 도구가 저장소 안 경로를 거부합니다. 이 숫자로 그린 집계 그래프(PNG)는 저장소에 올릴 수 있지만 영상 프레임과 얼굴이 식별되는 이미지는 그래프에도 넣지 않습니다(결정 I-34).
- `third_party/`의 `.npy` 20개, `.sav` 1개, 체크포인트 15개를 포함한 44개 파일(합계 55,020,834 B)은 저장소에 유지한 채 공개합니다(결정 I-W-31). 이 파일에 MIT가 적용되는 범위는 미확정이며 [THIRD_PARTY.md](THIRD_PARTY.md)에 적었습니다.

## 11. 라이선스와 인용

- 이 저장소가 직접 만든 코드와 문서(`src/`, `docs/`, `configs/`, `envs/`, `data/`): MIT, [LICENSE](LICENSE). 파일 머리에 자체 저작권 고지가 있는 파생 파일은 그 고지를 따릅니다.
- `third_party/` 아래 코드: 원저자의 MIT 라이선스(저자 고지는 `third_party/ghoddoosian/LICENSE`), 수정하지 않은 사본입니다. 수정 사본, 데이터와 모델 파일의 범위, dlib 모델과 데이터셋 영상을 포함하지 않는다는 내용은 [THIRD_PARTY.md](THIRD_PARTY.md)에 있습니다.
- 이 저장소, 사전학습 체크포인트, 원본 코드를 사용할 때는 원논문(arXiv 1904.07312)을 인용하세요. 공식 5-fold 분할로 평가하고 단일 fold가 아니라 5-fold 평균을 보고하세요.

```bibtex
@inproceedings{ghoddoosian2019realistic,
  title={A Realistic Dataset and Baseline Temporal Model for Early Drowsiness Detection},
  author={Ghoddoosian, Reza and Galib, Marnim and Athitsos, Vassilis},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops},
  pages={0--0},
  year={2019}
}
```

## 12. 향후 과제

- 게이지 GUI 통합과 재생 전용 뷰어 선택(I-W-33)
- 게이지 눈금의 구간 이름 표시 여부(I-W-35)
- 시간선 그림의 가독성 재설계와 저장소 반영
- 회전 영상 경로(`timeline.py extract --rotate auto`) 검증
- 행 확정 지연 검증(I-W-28)

## 13. 문서 안내

| 문서 | 내용 |
|---|---|
| [docs/README.md](docs/README.md) | 문서 지도와 독자별 읽는 순서 |
| [docs/REPRODUCTION_REPORT.md](docs/REPRODUCTION_REPORT.md) | 재현 보고서 |
| [docs/REPRODUCTION_LOG.md](docs/REPRODUCTION_LOG.md) | 오류를 포함한 날짜순 기록 |
| [docs/IMPLEMENTATION_DIFF.md](docs/IMPLEMENTATION_DIFF.md) | 원본 대비 차이와 문제점 목록 |
| [docs/CLAIMS_LEDGER.md](docs/CLAIMS_LEDGER.md) | 써도 되는 주장과 금지 표현 |
| [docs/PIPELINE_RUNBOOK.md](docs/PIPELINE_RUNBOOK.md) | 실행 순서와 환경변수 |
| [docs/DECISIONS.md](docs/DECISIONS.md) | 재현 트랙의 결정 기록 |
| [docs/EVALUATION_PROTOCOL.md](docs/EVALUATION_PROTOCOL.md) | 평가 규칙(결과를 본 뒤 바꾸지 않음) |
| [docs/improvement/README.md](docs/improvement/README.md) | 개선 트랙 안내 |
| [docs/improvement/IMPROVEMENT_LOG.md](docs/improvement/IMPROVEMENT_LOG.md) | 개선 트랙의 날짜순 기록(추가만) |
| [docs/improvement/DECISIONS.md](docs/improvement/DECISIONS.md) | 개선 트랙의 결정과 결정 대기 |
| [docs/improvement/RULE_SPEC.md](docs/improvement/RULE_SPEC.md) | 규칙 알람 명세 |
| [docs/improvement/PREREG_E-01.md](docs/improvement/PREREG_E-01.md), [PREREG_E-02.md](docs/improvement/PREREG_E-02.md) | 사전 등록 |
| [docs/improvement/DATA_REGISTRY.md](docs/improvement/DATA_REGISTRY.md) | 데이터 등록부 |
| [THIRD_PARTY.md](THIRD_PARTY.md), [LICENSE](LICENSE) | 서드파티 고지와 라이선스 |

참고: 날짜순 기록 문서(IMPROVEMENT_LOG.md, REPRODUCTION_LOG.md)와 사전 등록, 평가 규칙 문서는 기록 보존을 위해 고치지 않아서, 범위를 적은 물결표가 GitHub 미리보기에서 취소선으로 보이는 줄이 있습니다. 원문 텍스트는 그대로입니다.
