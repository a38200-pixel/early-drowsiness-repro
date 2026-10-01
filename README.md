# early-drowsiness-repro

졸음 조기 탐지 논문(arXiv:1904.07312)의 baseline을 UTA-RLDD 데이터셋에서 재현한 프로젝트입니다. 원저자의 공개 코드를
기반으로 하며, 투명하고 검증 가능한 재현 기록을 남기는 것이 목적입니다. 원본 코드는 수정 없이 vendoring했고,
재현을 위해 추가한 것은 모두 별도의 스크립트와 로그로 분리했습니다.

## 현재 상태

- **완료:** 공개 체크포인트(`my_model1..5`) 5-fold 평가와 논문 Table 2·3 대조, 재학습 3시드(80 epoch) 집계. 결과는 아래 "결과" 절에 있습니다.
- **하지 않은 것:** 영상→특징 추출 재현, 영상 수 174/180 확정, 개선 실험.

## 결과

표의 숫자는 [aggregate_seeds123.md](experiments/02_fold_seeds/aggregate_seeds123.md)(CSV: [aggregate_seeds123.csv](experiments/02_fold_seeds/aggregate_seeds123.csv))의 값 그대로입니다.
약어는 원본 `Training.py`가 출력하는 지표 이름입니다. 오류를 포함한 단계별 기록은 `docs/REPRODUCTION_LOG.md`에 있습니다.

### 1. 공개 체크포인트 평가 (학습 없음)

`src/repro/reproduction_eval.py`로 공개 `my_model1..5`를 복원해 평가만 했습니다(출력: `experiments/00_reproduction/regression_eval/`).

- **논문 Table 2·3과의 대조**(소수 둘째 자리 반올림): 비교 가능한 14칸 중 **11칸 일치, 3칸 불일치**입니다. 불일치는 Fold3 VA, Fold3 VRE, 평균 VRE이고 원인은 확인하지 못했습니다. 논문 값은 arXiv 1904.07312v1 Table 2·3(p.8)입니다.
- 이 평가를 새 폴더 구조에서 다시 실행했을 때 이 프로젝트의 이전 실행 결과와 25개 값(5 fold × 5 지표)이 모두 같았습니다. 이것은 **이전 실행 결과와의 자체 일관성 확인이며 논문과의 비교가 아닙니다.** 논문에 값이 있는 칸은 fold별 VA·VRE 10칸과 평균 4칸뿐입니다.

| 항목 | 논문 | 공개(원값) | 공개(소수 둘째 자리 반올림) | 판정 |
|---|---|---|---|---|
| Fold1 VA | 0.640 | 0.636 | 0.64 | 일치 |
| Fold1 VRE | 2.420 | 2.420 | 2.42 | 일치 |
| Fold2 VA | 0.610 | 0.611 | 0.61 | 일치 |
| Fold2 VRE | 1.040 | 1.040 | 1.04 | 일치 |
| Fold3 VA | 0.700 | 0.694 | 0.69 | 불일치 |
| Fold3 VRE | 0.580 | 0.667 | 0.67 | 불일치 |
| Fold4 VA | 0.640 | 0.639 | 0.64 | 일치 |
| Fold4 VRE | 0.850 | 0.850 | 0.85 | 일치 |
| Fold5 VA | 0.670 | 0.667 | 0.67 | 일치 |
| Fold5 VRE | 0.810 | 0.812 | 0.81 | 일치 |
| 평균 BSRE | 1.900 | 1.901 | 1.90 | 일치 |
| 평균 VRE | 1.140 | 1.158 | 1.16 | 불일치 |
| 평균 BSA acc | 0.540 | 0.541 | 0.54 | 일치 |
| 평균 VA | 0.652 | 0.649 | 0.65 | 일치 |

### 2. 재학습 (시드 1~3, 80 epoch, 저자 코드·하이퍼파라미터 무변경)

시드 설정과 지표 기록만 추가했고 모델·하이퍼파라미터·전처리는 바꾸지 않았습니다(차이 목록: [docs/IMPLEMENTATION_DIFF.md](docs/IMPLEMENTATION_DIFF.md)).
Fold1은 시드 1~5도 실행했습니다. 보고값은 각 실행의 마지막 epoch(80) 값입니다.

**시드별 5-fold 평균** (SD는 표본 표준편차 n-1, n=3)

| 지표 | 시드1 | 시드2 | 시드3 | 평균 ± SD(n-1, n=3) | 공개 체크포인트 | 논문(Table 2) |
|---|---|---|---|---|---|---|
| BSA loss | 13.957 | 14.101 | 13.736 | 13.931 ± 0.184 | 12.271 | 논문에 없음 |
| BSA acc | 0.517 | 0.512 | 0.489 | 0.506 ± 0.015 | 0.541 | 0.540 |
| VA | 0.535 | 0.597 | 0.534 | 0.556 ± 0.036 | 0.649 | 0.652 |
| BSRE | 3.157 | 2.748 | 2.718 | 2.874 ± 0.245 | 1.901 | 1.900 |
| VRE | 2.580 | 1.587 | 1.692 | 1.953 ± 0.546 | 1.158 | 1.140 |

**fold별 재학습 평균과 공개 체크포인트 차이**

| 구분 | BSA loss | BSA acc | VA | BSRE | VRE |
|---|---|---|---|---|---|
| Fold1 재학습(시드1~3 평균) | 16.013 | 0.449 | 0.556 | 3.478 | 2.743 |
| Fold1 재학습(시드1~5 평균) | 16.006 | 0.461 | 0.545 | 3.504 | 2.840 |
| Fold1 공개 | 15.711 | 0.484 | 0.636 | 3.185 | 2.420 |
| Fold1 차이(시드1~3 평균 - 공개) | +0.301 | -0.035 | -0.081 | +0.293 | +0.324 |
| Fold1 차이(시드1~5 평균 - 공개) | +0.295 | -0.023 | -0.091 | +0.318 | +0.421 |
| Fold2 재학습(시드1~3 평균) | 10.978 | 0.574 | 0.602 | 1.690 | 1.471 |
| Fold2 공개 | 10.574 | 0.588 | 0.611 | 1.391 | 1.040 |
| Fold2 차이(시드1~3 평균 - 공개) | +0.404 | -0.014 | -0.009 | +0.299 | +0.432 |
| Fold3 재학습(시드1~3 평균) | 20.368 | 0.494 | 0.481 | 6.287 | 3.898 |
| Fold3 공개 | 13.315 | 0.561 | 0.694 | 2.153 | 0.667 |
| Fold3 차이(시드1~3 평균 - 공개) | +7.053 | -0.066 | -0.213 | +4.134 | +3.230 |
| Fold4 재학습(시드1~3 평균) | 12.240 | 0.493 | 0.574 | 1.726 | 1.005 |
| Fold4 공개 | 11.508 | 0.544 | 0.639 | 1.599 | 0.850 |
| Fold4 차이(시드1~3 평균 - 공개) | +0.732 | -0.051 | -0.065 | +0.127 | +0.155 |
| Fold5 재학습(시드1~3 평균) | 10.059 | 0.519 | 0.566 | 1.191 | 0.648 |
| Fold5 공개 | 10.250 | 0.529 | 0.667 | 1.176 | 0.812 |
| Fold5 차이(시드1~3 평균 - 공개) | -0.191 | -0.010 | -0.101 | +0.014 | -0.164 |

### 3. 읽는 법과 한계

- 보고값은 epoch 80의 단일값이며 사후 선택을 하지 않았습니다.
- 시드가 3개뿐이어서 표본 SD(n-1)는 참고 수준입니다.
- 5-fold 평균 기준으로 BSA acc와 VA는 공개 체크포인트가 재학습 평균보다 높고, BSA loss·BSRE·VRE는 낮습니다. fold별로는 Fold5의 BSA loss와 VRE에서 재학습 평균이 더 낮은 등 방향이 다른 칸도 있습니다(차이 열 참고). 이 차이의 원인은 확인하지 못했습니다.
- 공개 Fold3 체크포인트는 논문 Table 3의 VA·VRE와도 일치하지 않습니다(1절 표).
- BSA loss는 논문에 정의가 없는 항목입니다. 정의는 [docs/IMPLEMENTATION_DIFF.md](docs/IMPLEMENTATION_DIFF.md)를 참고하세요.
- 지표·전처리 정의가 논문 서술과 다른 부분이 있습니다. 세부는 [docs/IMPLEMENTATION_DIFF.md](docs/IMPLEMENTATION_DIFF.md)를 참고하세요.

### 4. 이번 단계에서 하지 않은 것

- 영상→특징 추출 재현(입력은 저자가 제공한 특징 파일 `.npy`입니다)
- 영상 수 174/180의 확정
- 개선 실험

## 구조

```
third_party/ghoddoosian/   원본 코드, 체크포인트, 특징 파일(바이트 단위로 동일, MIT; THIRD_PARTY.md 참고)
src/repro/                 환경/데이터셋/자산 점검, 공개 체크포인트 평가, 시드별 재학습, 집계(aggregate_seeds.py), 영상 1개 blink 실행
experiments/00_reproduction/   보관된 로그와 공개 체크포인트 평가 결과
experiments/01_fold1_seeds/    Fold1 시드 1~5 재학습 결과
experiments/02_fold_seeds/     Fold2~5 재학습 결과와 집계 파일(aggregate_seeds123.md/.csv)
configs/                   protected_manifest.sha256(해시 44개), image_publishable.csv
envs/environment_tf1.yml   conda 환경(Python 3.6, TensorFlow 1.8)
data/README.md             데이터 입수 방법(저장소에는 데이터를 넣지 않음)
docs/REPRODUCTION_LOG.md   오류를 포함한 단계별 기록
```

원본 대비 차이: [docs/IMPLEMENTATION_DIFF.md](docs/IMPLEMENTATION_DIFF.md)

## 재현 방법

conda가 필요합니다. 명령은 저장소 루트에서 실행합니다.

```
conda env create -f envs/environment_tf1.yml
conda activate ghoddoosian_tf1

python src/repro/check_environment.py      # 인터프리터/패키지 버전
python src/repro/check_dataset.py          # .npy 입력과 my_model1..5 체크포인트
python src/repro/reproduction_eval.py 5    # 사전학습 체크포인트 5-fold 평가
python src/repro/aggregate_seeds.py        # 재학습 3시드 집계(experiments/02_fold_seeds/aggregate_seeds123.md/.csv 생성)
```

`reproduction_eval.py`는 `experiments/00_reproduction/regression_eval/`에 결과를 쓰며 기존 파일을 덮어씁니다.
보존하려면 먼저 옮겨 두세요. 실행 시작 시 vendoring된 44개 파일을 `configs/protected_manifest.sha256`과
대조하고, 하나라도 다르면 중단합니다.

선택 사항(이 저장소에 없는 데이터가 필요, `data/README.md` 참고):

```
set DLIB_LANDMARK_PATH=<shape_predictor_68_face_landmarks.dat 경로>
set RLDD_DATA_DIR=<UTA-RLDD 영상 폴더>
python src/repro/check_video_assets.py
python src/repro/reproduction_blink_one_video.py <영상 경로>
```

재학습은 `src/repro/train_fold1_seeds.py`(Fold1)와 `src/repro/train_fold_seeds.py`(지정 fold)로 하고,
`scripts/run_fold1_seeds.bat`, `scripts/run_folds_seed.bat`가 이를 순서대로 실행합니다.
`src/repro/reproduction_train_fold1.py`는 **레거시**입니다. 다시 실행하면 `experiments/00_reproduction/fold1_training/`의
기존 파일을 덮어쓸 수 있으니 쓰지 마세요.

## 데이터

UTA-RLDD 영상, dlib 랜드마크 모델, 그로부터 만든 프레임은 **포함하지 않습니다**. `data/README.md`를 참고하세요.

## 이용 지침

- 이 저장소, 사전학습 체크포인트, 원본 코드를 사용할 때는 원논문(arXiv:1904.07312)을 반드시 인용하세요.
  `THIRD_PARTY.md`도 참고하세요.
- 공식 5-fold 분할로 평가하고, 단일 fold가 아니라 5-fold 평균을 보고하세요.
- UTA-RLDD의 얼굴 이미지는 `configs/image_publishable.csv`에 있는 36명(공식 데이터셋 페이지 표의 'YES')만 공개할 수
  있습니다. 이런 이미지는 반드시 `src/repro/publish_guard.py`를 통해 저장하세요. 목록에 없는 대상은 거부됩니다.

## 알려진 한계와 미확정 사항

- 재학습: 시드 1~3 결과는 위 "결과" 절에 있고, 공개 체크포인트와의 차이의 원인은 확인하지 못했습니다. 이전 Fold1 단일 실행 로그가
  `experiments/00_reproduction/fold1_training/`에 보관돼 있지만 위 집계에는 포함하지 않았습니다.
- 입력은 저자가 제공한 특징 파일(`.npy`)입니다. 영상에서 특징을 뽑는 파이프라인(`blink_video.py`)은 자산 로딩만 점검했고,
  이 저장소에서 데이터셋 전체에 다시 돌리지 않았습니다.
- 환경이 오래되었고(Python 3.6, TensorFlow 1.8) Windows에서만 실행해 봤습니다.
- 확정되지 않은 수치는 위 "결과" 절에 적은 것 외에는 의도적으로 싣지 않았습니다.

## 인용

```bibtex
@inproceedings{ghoddoosian2019realistic,

title={A Realistic Dataset and Baseline Temporal Model for Early Drowsiness Detection},

author={Ghoddoosian, Reza and Galib, Marnim and Athitsos, Vassilis},

booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops},

pages={0--0},

year={2019}

}
```

## 라이선스

이 저장소의 자체 코드: MIT, `LICENSE` 참고. `third_party/` 아래 코드: MIT, Copyright (c) 2019 Reza Ghoddoosian,
`THIRD_PARTY.md` 참고.
