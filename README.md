# early-drowsiness-repro

졸음 조기 탐지 논문(arXiv:1904.07312)의 baseline을 UTA-RLDD 데이터셋에서 재현한 프로젝트입니다. 원저자의 공개 코드를
기반으로 하며, 투명하고 검증 가능한 재현 기록을 남기는 것이 목적입니다. 원본 코드는 수정 없이 vendoring했고,
재현을 위해 추가한 것은 모두 별도의 스크립트와 로그로 분리했습니다.

## 현재 상태

- **완료:** 사전학습 체크포인트(`my_model1..5`, 공식 5-fold 분할)의 평가를 이 저장소 구조에서 다시 실행했고, 이 프로젝트의
  이전 실행 기록과 25개 값이 모두 일치합니다(자체 기대값과의 비교). 논문 Table 2·3과는 비교 가능한 14칸 중 11칸이 일치하고
  3칸(Fold3 VA, Fold3 VRE, 평균 VRE)이 다릅니다. 원인은 미확인이며 자세한 대조는
  [docs/IMPLEMENTATION_DIFF.md](docs/IMPLEMENTATION_DIFF.md)의 5절을 참고하세요. 아래 회귀 테스트 표를 참고하세요.
- **진행 중:** 재학습. Fold1 재학습 결과는 다중 시드로 확인 중이며 **TBD**입니다. 재학습 수치는 여기에 싣지 않았습니다.

## 회귀 테스트 결과: 사전학습 체크포인트 5-fold 평가 (학습 없음)

`src/repro/reproduction_eval.py`로 생성했습니다(체크포인트 복원 후 평가만 수행). 원본 출력은
`experiments/00_reproduction/regression_eval/`에 있습니다.

| Fold | BSA loss | BSA acc | VA     | BSRE  | VRE   |
| ---- | -------- | ------- | ------ | ----- | ----- |
| 1    | 15.711   | 0.4838  | 0.6364 | 3.185 | 2.420 |
| 2    | 10.574   | 0.5879  | 0.6111 | 1.391 | 1.040 |
| 3    | 13.315   | 0.5606  | 0.6944 | 2.153 | 0.667 |
| 4    | 11.508   | 0.5445  | 0.6389 | 1.599 | 0.850 |
| 5    | 10.250   | 0.5287  | 0.6667 | 1.176 | 0.812 |
| 평균 | 12.271   | 0.5411  | 0.6495 | 1.901 | 1.158 |

약어는 원본 `Training.py`가 출력하는 지표 이름입니다. 정의는 원본 README와 논문을 참고하세요.
오류를 포함한 단계별 진행 기록은 `docs/REPRODUCTION_LOG.md`에 있습니다.

## 구조

```
third_party/ghoddoosian/   원본 코드, 체크포인트, 특징 파일(바이트 단위로 동일, MIT; THIRD_PARTY.md 참고)
src/repro/                 환경/데이터셋/자산 점검, 평가, Fold1 재학습, 영상 1개 blink 실행
experiments/00_reproduction/   보관된 로그와 결과 표
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

`src/repro/reproduction_train_fold1.py`는 원래 설정으로 Fold1을 재학습합니다. 위 표를 재현하는 데는 필요하지 않으며,
결과는 보고하지 않습니다(현재 상태 참고).

## 데이터

UTA-RLDD 영상, dlib 랜드마크 모델, 그로부터 만든 프레임은 **포함하지 않습니다**. `data/README.md`를 참고하세요.

## 이용 지침

- 이 저장소, 사전학습 체크포인트, 원본 코드를 사용할 때는 원논문(arXiv:1904.07312)을 반드시 인용하세요.
  `THIRD_PARTY.md`도 참고하세요.
- 공식 5-fold 분할로 평가하고, 단일 fold가 아니라 5-fold 평균을 보고하세요.
- UTA-RLDD의 얼굴 이미지는 `configs/image_publishable.csv`에 있는 36명(공식 데이터셋 페이지 표의 'YES')만 공개할 수
  있습니다. 이런 이미지는 반드시 `src/repro/publish_guard.py`를 통해 저장하세요. 목록에 없는 대상은 거부됩니다.

## 알려진 한계와 미확정 사항

- Fold1 재학습: 결과는 **TBD**이며 다중 시드로 확인 중입니다. 원본 코드는 난수 시드를 설정하지 않으므로 재학습은
  실행마다 결과가 다릅니다. 이전 Fold1 재학습 로그가 `experiments/00_reproduction/`에 보관돼 있지만, 아직 그로부터
  결론을 내리지 않았습니다.
- 사전학습 체크포인트만 평가합니다. 영상에서 특징을 뽑는 파이프라인(`blink_video.py`)은 자산 로딩만 점검했고,
  이 저장소에서 데이터셋 전체에 다시 돌리지 않았습니다.
- 환경이 오래되었고(Python 3.6, TensorFlow 1.8) Windows에서만 실행해 봤습니다.
- 확정되지 않은 수치는 위 회귀 테스트 표 외에는 의도적으로 싣지 않았습니다.

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
