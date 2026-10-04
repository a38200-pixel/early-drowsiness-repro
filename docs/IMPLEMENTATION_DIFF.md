# 원본 대비 구현·환경 차이

## 1. 목적과 범위

저자 원본(`third_party/ghoddoosian/`)과 이 저장소의 재구현·실행 환경, 그리고 논문(arXiv:1904.07312v1)과 원본 코드 사이의 대응을 한 곳에 정리한다.
코드와 환경만 다루며, 성능 수치는 5절의 논문 대조표를 제외하면 본문에 복사하지 않고 결과 파일 경로만 안내한다(6절).
확인하지 못한 값은 추측하지 않고 "미확인" 또는 "문서에서 확인되지 않음"으로 적고, 어떤 차이의 원인도 추정하지 않는다.

## 2. 기준점

| 항목 | 값 | 근거 |
|---|---|---|
| 원본 저장소 | https://github.com/rezaghoddoosian/Early-Drowsiness-Detection.git | `THIRD_PARTY.md` |
| 기준 커밋 | `2bafc985c46526fa0dfe3a93bb96fcee1d8d7ac4` | `THIRD_PARTY.md` |
| 라이선스 | MIT, Copyright (c) 2019 Reza Ghoddoosian | `THIRD_PARTY.md`, `third_party/ghoddoosian/LICENSE` |
| 논문 | arXiv:1904.07312, "A Realistic Dataset and Baseline Temporal Model for Early Drowsiness Detection" (CVPRW 2019) | `README.md`(인용 절) |
| 논문 기준값의 출처 | 논문 PDF는 저장소에 없다. 이 문서의 "논문 근거"(쪽/절/식)는 작업 지시에 제공된 기준값(arXiv 1904.07312v1)을 인용한 것이다. 작업 지시 기준으로 Table 2·3, Eq.10~13, §5.2 수치는 논문 PDF 원문과 대조가 끝났다. Eq.1~5와 Eq.7~9는 작업 지시에 적힌 식을 기준으로 3.3절 g의 코드 대조 결과를 따른다. 그 밖의 인용(Fig.2, Eq.6, §3.2·§3.3의 값)도 작업 지시 기준으로 논문 PDF 원문과 대조가 끝났다(이 저장소에서 PDF를 다시 열어 확인한 것은 아니다) | 이 문서 3.3절, 5절 |

## 3. 비교표

상태(원본↔이 저장소): **동일** / **의도적 변경** / **미확인**.
판정(논문↔원본 코드): **일치** / **불일치** / **논문에 없음** / **코드에서 확인 불가** / **해당 없음**.
"Training.py:N"은 `third_party/ghoddoosian/Training.py`의 N행이다. 항목별 상세 대조는 3.3절이다.

| 항목 | 원본(저자) | 이 저장소 | 상태 | 논문 근거 | 코드 근거(파일:줄) | 판정 |
|---|---|---|---|---|---|---|
| 모델 코드(원본 .py 6개: `HMSLSTM_Main`, `HMSLSTM_cell`, `MultiHMSLSTM`, `Preprocessing`, `Training`, `blink_video`) | 기준 커밋의 파일 | 바이트 동일, 수정 없음. 44개 파일 blob 해시가 원본과 일치하고, 학습·평가 스크립트는 시작과 종료 시 `configs/protected_manifest.sha256`로 재검사한다. 원본의 알려진 오타(`hstate_size_size`)도 고치지 않았다 | 동일 | Fig.2 p.3, Eq.7-9 p.6 | `THIRD_PARTY.md`, `configs/protected_manifest.sha256`, `src/repro/common.py`, `third_party/ghoddoosian/HMSLSTM_cell.py:26,30`(오타 위치) | 구조 단위는 일치(3.3절 a). 식 기준은 3.3절 g: Eq.7·Eq.8 불일치(batch norm 삽입), Eq.9 일치 |
| 학습 하이퍼파라미터 | epoch 80, batch 64, Adam lr 0.000053, 은닉 `[32,32,32,32]`, 4층, step 30, 입력 특징 4, `drop_out_p=1.0`, Pre_fc1 32, Post_fc1 16, embedding 16/16, Post_fc2 8, th 1.253 | 위 값을 치환하지 않는다. 바꾸는 것은 epoch 수 인자뿐이며 기본값이 80이다(보고용 학습은 80, 검증용 스모크는 2) | 동일 | §5.2 p.7, Fig.2 p.3 | Training.py:503-506, 335 | 일치(`drop_out_p`는 논문에 없음) |
| 난수 시드 | 고정하지 않음. 전역 시드 호출이 없고 Xavier 초기화가 `seed=None`이다. 학습 데이터 셔플은 없다(`batch_gen`). 원본 주석은 fold를 여러 번 돌려 보라고 안내한다 | 시드를 인자로 받아 `random.seed`, `np.random.seed`를 설정하고, `Train()` 안 `tf.reset_default_graph()` 바로 다음 줄에 `tf.set_random_seed(시드)`를 메모리 치환으로 넣는다. 공개 체크포인트 평가(`reproduction_eval.py`)는 시드를 쓰지 않는다 | 의도적 변경 | 논문에 없음(시드, 반복 횟수) | Training.py:155,182,202,216,232,245(`seed=None`), 304, 408(주석). `src/repro/train_fold_seeds.py`, `src/repro/train_fold1_seeds.py` | 논문에 없음 |
| 실행 방식 | `Training.py`를 직접 실행하면 5개 fold를 순환(`for i in range(5)`)하고 모델·로그를 현재 폴더에 쓴다. 한 fold씩 지정하는 인자는 없다 | 래퍼 스크립트가 `Training.py`를 텍스트로 읽어 `load=False` 줄을 기준으로 정의부와 fold 반복부로 나눈 뒤 메모리에서 치환해 `exec`한다. 디스크의 원본은 바뀌지 않는다. 치환 목록은 3.1절 | 의도적 변경 | — | Training.py:407-413. `src/repro/train_fold_seeds.py`, `src/repro/train_fold1_seeds.py`, `src/repro/reproduction_eval.py`, `src/repro/reproduction_train_fold1.py` | 해당 없음 |
| epoch별 지표 기록 | 화면에 출력하고, 학습이 끝나면 `x/loss/accuracy*.npy` 6개를 저장한다. BSRE/VRE는 epoch 번호가 15의 배수이거나 79일 때만 출력된다 | `metrics.csv`에 epoch마다 1행을 추가한다. 열: `epoch`(1부터), `train_loss_sum`(epoch 안 배치 손실 합), `train_moving_acc`(배치 정확도 이동 평균), `BSA_loss`, `BSA_acc`, `VA`, `BSRE`, `VRE`(테스트 세트 기준). 기록 호출만 추가하고 계산은 바꾸지 않는다. 보고값은 마지막 epoch(80)의 값이며 최선 epoch 선택은 하지 않는다. 마지막 epoch가 80이 아니면 요약에서 미완료로 제외한다 | 의도적 변경 | — | Training.py:48,81,509-515. `src/repro/train_fold_seeds.py`(상단 열 설명), `src/repro/summarize_fold1_seeds.py`, `src/repro/summarize_folds.py` | 해당 없음 |
| 체크포인트 | 공개 `my_model1~5`. 저장 호출은 학습 종료 후 1회뿐이다. 같은 줄의 주석은 "Saving every epoch"라고 되어 있어 코드와 다르다. `load=False`로 그대로 돌리면 `ii=i`(0부터)라 `my_model0`부터 저장되어 공개 `my_model1~4`를 덮어쓸 수 있다 | 공개 체크포인트는 `third_party/ghoddoosian/`에 있고 git이 추적하며 해시로 보호한다. 재학습 체크포인트는 `ii=i+1` 치환으로 `my_model<fold>.*`를 `experiments/01_fold1_seeds/`, `experiments/02_fold_seeds/`의 실행 폴더에 저장하며 `.gitignore`로 추적하지 않는다(`.npy`, `checkpoint`, `train.log`도 같다). 저장되는 모델은 마지막 epoch의 것이다. 공개 체크포인트가 어떤 실행·epoch에서 선택되었는지는 원본 저장소에 명시되지 않음 | 의도적 변경(저장 위치·이름), 공개 체크포인트의 선택 이력은 미확인 | 논문에 없음(체크포인트 선택 기준) | Training.py:395-396, 411. `.gitignore` | 논문에 없음 |
| 입력 특징 | 저자가 제공한 `{Blinks, BlinksTest, Labels, LabelsTest}_30_Fold{1..5}.npy` 20개 | 같은 파일을 그대로 쓴다. 실행마다 해당 fold 입력 4개의 SHA-256을 `run_meta.json`에 기록한다 | 동일 | §5.2 p.7(T=30, stride 2) | `third_party/ghoddoosian/README.md:58-62`, Preprocessing.py:263-264. `src/repro/check_dataset.py` | 일치(윈도우). 피험자별 정규화 구간은 불일치, 제공 `.npy`는 1차 정규화 후·2차 정규화 전으로 판단(3.3절 g) |
| 영상→특징 추출(`blink_video.py`, `Preprocessing.py`) | 영상에서 blink 특징을 뽑아 `.npy`를 만드는 단계. `Preprocessing.py`는 import하면 Fold3 고정으로 실행되고(262-272) 학습용 `np.save`는 주석 처리되어 있다(269-270). 학습 데이터 셔플은 시드가 없다(5) | 영상이 없어 실행하지 않았다. dlib 모델과 blink SVM이 로드되는지만 확인했다. `.npy`는 재생성하지 않았다 | 미확인(미재현) | Eq.1-5 | blink_video.py:74-92,179-181,455,464. Preprocessing.py:5,262-272. `src/repro/check_video_assets.py`, `src/repro/reproduction_blink_one_video.py` | 3.3절 g: Eq.1 불일치, Eq.2·3 일치, Eq.4 불일치, Eq.5 일치(구조)·세부 확인 불가. 이 저장소에서는 미재현 |
| 데이터셋 | UTA-RLDD에 적용했다고만 적혀 있고 영상 수는 명시되지 않음. 원본 README·주석에 제외 영상이나 피험자 언급이 없다 | 영상은 저장소에 없고 사용하지 않는다(특징 `.npy`만 사용). 원본 `Training.py`의 fold별 `start_indices` 길이는 33, 36, 36, 36, 33이며 합은 174이다. 논문의 180이라는 수치와의 관계는 **미확정** | 미확인 | 60명, 영상 180개, 5 fold×12명(fold당 36개) p.5 §3.3. 일부 영상은 품질 사유로 무시(수 없음) p.4 §3.2 | Training.py:440-443,445,456,465,477,489. `third_party/ghoddoosian/README.md`, `data/README.md` | 불일치(fold당 영상 수, 합계). 제외 영상은 논문에 없음 |
| 환경(요약) | Ubuntu 16.04, TensorFlow 1.8에서 테스트. 설치 명령은 `tensorflow-gpu==1.8.0`. 패키지 버전은 TensorFlow 외에는 명시되지 않음. requirements 파일 없음 | Windows(버전은 문서에 적혀 있지 않음), CPU 전용, Python 3.6.13, TensorFlow 1.8.0. 자세한 버전은 3.2절 | OS는 의도적이지 않은 차이, TensorFlow는 동일, 나머지는 미확인 | 논문에 없음(소프트웨어 버전) | `third_party/ghoddoosian/README.md:7,23-43`, `envs/environment_tf1.yml`, `docs/REPRODUCTION_LOG.md`, `src/repro/check_environment.py` | 논문에 없음 |
| 결정성 확인 | 원본은 시드가 없어 해당 없음 | 같은 시드로 2 epoch 스모크를 두 번 돌려 `metrics.csv`의 모든 열이 소수 6자리 문자열까지 같음을 확인했다. 시드가 다르면 epoch 1의 `train_loss_sum`이 달랐다. 이 비교는 `%.6f` 출력 기준이라 비트 단위 비교는 아니다. 다른 CPU, OS, GPU에서의 결정성은 확인하지 않았다 | 확인 범위가 한정됨 | 논문에 없음 | `docs/REPRODUCTION_LOG.md`(스모크 테스트 절) | 논문에 없음 |
| 평가 지표 정의 | BSA는 `calc_accuracy_per_batch`, VA는 `vote_accuracy_per_video`, BSRE는 `regression_per_blink`, VRE는 `regression_per_video`. "BSA loss"는 테스트 세트에 대한 학습 손실 `loss`를 그대로 출력한 값이다 | 지표 함수는 수정하지 않고, 반환값만 감싸서 `metrics.csv`에 기록한다. 수치를 논문 값과 맞추려는 변경은 하지 않았다 | 동일(구현), 논문 대조는 3.3절 d | §5.1 p.7, Eq.12, Eq.13 | Training.py:13-51, 54-84, 87-117, 123-131, 313-335, 371-379 | BSA·BSRE·VA 일치, **VRE 불일치**, BSA loss는 논문에 없음 |
| 재현 범위 밖(하지 않은 것) | - | 영상→특징 추출과 `.npy` 재생성, UTA-RLDD 영상 사용, 원본과 같은 OS·GPU 환경 재구성, 공개 체크포인트의 선택 이력 추적, 특징 추출 식(Eq.1~5)의 수치 재현(코드와의 읽기 대조만 수행) | 미확인 | — | 위 각 행 | 해당 없음 |

### 3.1 래퍼 스크립트의 메모리 치환 지점

원본 파일은 디스크에서 바뀌지 않는다. 치환은 텍스트를 읽은 뒤 문자열 수준에서 일어난다. 학습 스크립트는 대상 문자열이 정확히 1번(`open('./` 6곳은 6번) 나오는지 단언하고, 아니면 실행을 멈춘다.

| 치환 대상(원본) | 바꾼 내용 | 목적 | 적용 스크립트 |
|---|---|---|---|
| `tf.reset_default_graph()` 다음 줄 | `tf.set_random_seed(<시드>)` 추가 | 새 그래프에 시드 적용 | `train_fold_seeds.py`, `train_fold1_seeds.py` |
| `variable_path='./'` | 출력 폴더 경로 | 체크포인트를 실행 폴더에 저장 | `train_fold_seeds.py`, `train_fold1_seeds.py`, `reproduction_train_fold1.py` |
| `y_v[idx]=accuracy_per_videoV` 다음 줄 | `_on_epoch_end(...)` 호출 추가 | epoch별 지표 기록 | `train_fold_seeds.py`, `train_fold1_seeds.py` |
| `for i in range(5):` | `range(fold-1, fold)` / `range(1)` / `range(NUM_FOLDS)` | 지정 fold만 학습, 또는 평가할 fold 수 지정 | `train_fold_seeds.py`(지정 fold), `train_fold1_seeds.py`·`reproduction_train_fold1.py`(Fold1), `reproduction_eval.py`(평가) |
| `ii=i    #` | `ii=i+1    #` | 저장 이름을 `my_model<fold>`로 맞춤 | 학습 스크립트 3종 |
| `num_epochs=80` | 인자로 받은 epoch 수 | 스모크 검증 가능하게 함. 기본값 80 | 학습 스크립트 3종 |
| `open('./...')` 6곳 | 출력 폴더 경로 | `x/loss/accuracy*.npy`를 실행 폴더에 저장 | 학습 스크립트 3종 |
| `load=False` | `load=True` | 학습 없이 공개 체크포인트 복원·평가 | `reproduction_eval.py` |
| 지표 함수 호출(네임스페이스 교체, 원문 치환 아님) | `regression_per_blink`, `regression_per_video`, `Train` 등을 반환값을 기록하는 함수로 감쌈 | BSRE, VRE와 평가 결과 수집 | 학습 스크립트 3종, `reproduction_eval.py` |
| `print`(네임스페이스 교체) | 화면과 `train.log`에 함께 쓰는 함수 | 로그 보존 | `train_fold_seeds.py`, `train_fold1_seeds.py` |
| `blink_video.py`의 `output_file = 'alert.txt'`, `path = ''`, dlib `.dat` 상대경로 | 출력 파일, 영상 경로, 환경변수로 받은 `.dat` 경로 | 영상 1개 실행. 영상이 없어 실행하지 않음 | `reproduction_blink_one_video.py` |

공통으로, 모든 래퍼는 작업 디렉터리를 `third_party/ghoddoosian/`로 바꿔 원본이 상대경로로 읽는 `.npy`·체크포인트가 열리게 하고, `sys.dont_write_bytecode = True`로 `__pycache__`가 생기지 않게 한다.
근거: 위 표의 각 스크립트(`src/repro/` 아래).

### 3.2 환경 버전 비교

| 항목 | 원본 저장소가 명시한 것 | 이 저장소(conda 환경 `ghoddoosian_tf1`) | 상태 |
|---|---|---|---|
| OS | Ubuntu 16.04 | Windows(버전은 문서에 적혀 있지 않음) | 의도적이지 않은 차이 |
| CPU/GPU | 설치 명령에 `tensorflow-gpu==1.8.0`. 실제 테스트 하드웨어는 명시되지 않음 | CPU 전용(`tensorflow==1.8.0` pip 패키지) | 미확인(원본 하드웨어) |
| Python | 명시되지 않음 | 3.6.13 | 미확인 |
| TensorFlow | 1.8 | 1.8.0 | 동일 |
| NumPy | 명시되지 않음 | 1.14.6 | 미확인 |
| SciPy | 명시되지 않음 | 1.2.1 | 미확인 |
| scikit-learn | 명시되지 않음 | 0.19.2 | 미확인 |
| scikit-image | 명시되지 않음 | 0.14.2 | 미확인 |
| dlib | 명시되지 않음 | 19.17(conda-forge). 19.16과 19.15는 설치할 수 없었다 | 미확인 |
| OpenCV | 명시되지 않음 | 3.4.5.20(`opencv-python`) | 미확인 |
| imutils | 명시되지 않음 | 0.5.4 | 미확인 |
| matplotlib | 명시되지 않음 | 2.2.5 | 미확인 |
| protobuf | 명시되지 않음 | 3.19.6 | 미확인 |

근거: `third_party/ghoddoosian/README.md`, `envs/environment_tf1.yml`, `docs/REPRODUCTION_LOG.md`(환경과 설치 이슈 절), `src/repro/check_environment.py`(ENVIRONMENT_OK). 위 이 저장소 버전은 `ghoddoosian_tf1`에서 `python --version`과 `pip list`로 다시 확인했다.
원본 blink SVM 파일은 scikit-learn 0.18보다 오래된 형식이며 0.19.2에서 로드는 되지만 영상 예측은 호환성에 민감하다고 `docs/REPRODUCTION_LOG.md`에 적혀 있다.

### 3.3 논문 기준값 대 원본 코드 대조(항목별)

"논문 값"은 작업 지시에 적힌 기준값과 식 그대로이며, PDF 원문 대조의 범위는 2절에 적었다. 코드는 읽기만 했고 실행·수정하지 않았다.
`P`는 `third_party/ghoddoosian/Preprocessing.py`, `T`는 `third_party/ghoddoosian/Training.py`, `B`는 `third_party/ghoddoosian/blink_video.py`, `R`은 `third_party/ghoddoosian/README.md`를 뜻한다.

**a. 학습 설정과 모델 구조**

| 항목 | 논문 값(근거) | 원본 코드 값 | 코드 근거 | 판정 |
|---|---|---|---|---|
| epoch | 80 (§5.2 p.7) | `num_epochs=80` | T:504 | 일치 |
| batch size | 64 (§5.2 p.7) | `batch_size=64` | T:504, 265-276 | 일치 |
| 학습률, optimizer | 0.000053, Adam (§5.2 p.7) | `lr=0.000053`, `tf.train.AdamOptimizer(lr)` | T:506, 335 | 일치 |
| Delta | 1.253 (§5.2 p.7) | `th=1.253`, `max(0, error² - th)` | T:506, 314 | 일치 |
| L2 계수 | 0.1 (§5.2 p.7) | `0.1 * (tf.nn.l2_loss(...)` 6군 가중치(last_fc, pre_fc1, post_fc1 각 층, post_fc2, embeddings, embeddings2). HM-LSTM 내부 가중치와 bias는 합에 없음 | T:317-334 | 일치(계수). L2 대상 범위와 `tf.nn.l2_loss`의 1/2 계수는 기준값에 없음 |
| batch norm | 사용 (§5.2 p.7) | `batchNorm`(배치 평균·분산, EMA decay 0.5, epsilon 1e-3, gamma 없음)을 pre_fc1, post_fc1, embeddings, embeddings2, post_fc2에 적용. last_fc에는 없음 | T:133-144, 162, 190, 210, 224, 238 | 일치(사용 여부). decay·epsilon은 논문에 없음 |
| 가중치 초기화 | 논문에 없음 | FC 가중치는 Xavier(`uniform=False`, `seed=None`), bias는 0. HM-LSTM 셀 내부 선형층은 코드에 가중치 초기화 지정이 없고 bias 초기화만 인자로 넘김 | T:155, 157, 182, 202, 216, 232, 245. `third_party/ghoddoosian/HMSLSTM_cell.py:42` | 논문에 없음. 셀 내부 가중치 초기화는 코드에서 확인 불가 |
| Pre-FC | 32 (Fig.2 p.3) | `Pre_fc1_size=32`, 가중치 `[4, 32]` | T:154, 505 | 일치 |
| HM-LSTM | 4층, 은닉 32 (Fig.2 p.3, Eq.7-9 p.6) | `num_layers=4`, `hstate_size=[32,32,32,32]`, 층별 셀 생성. 셀 식(f, i, o, g, z)은 HM-LSTM 형태 | T:173-176, 505. `third_party/ghoddoosian/HMSLSTM_Main.py:18-36`, `third_party/ghoddoosian/HMSLSTM_cell.py:42-70` | 일치(층 수·은닉). 식 대조는 3.3절 g의 Eq.7·8·9 |
| FC1 | 16 (Fig.2 p.3) | `Post_fc1_size_per_layer=16`. 4개 층 출력에 각각 `[32,16]` 가중치를 적용해 이어 붙여 64차원 | T:181, 192-195, 505 | 일치(층별 16). 이어 붙임은 기준값에 서술 없음 |
| FC2, FC3 | 16, 16 (Fig.2 p.3) | `embb_size=16`(입력 64), `embb_size2=16` | T:201, 215, 505 | 일치 |
| FC4 | 8 (Fig.2 p.3) | `Post_fc2_size=8` | T:230, 505 | 일치 |
| 출력, 스케일 | 출력 1, out = 10·Sigmoid (Eq.7-9 p.6) | `output_size=1`, `10 * tf.sigmoid(output)` | T:243, 255 | 일치 |
| dropout | 논문에 없음 | `drop_out_p=1.0`(keep 확률 1.0, 사실상 미사용) | T:505, 165, 198 | 논문에 없음 |
| 손실 | Eq.11: 합(max(0, \|out-t\|² - Delta))/N | `tf.reduce_mean(tf.maximum(0.0, tf.square(error) - th))` 뒤에 L2 항을 더함 | T:313-315, 333-334 | 일치(Eq.11 부분). L2 항은 별도 |
| 학습 시퀀스 수 | 약 7000 (§5.2 p.7) | 이 저장소 문서에는 fold 2~5의 학습 시퀀스가 6.5k~7.2k로 기록되어 있다. Fold1 개수는 이 저장소 문서에서 확인되지 않음 | `docs/REPRODUCTION_LOG.md`("Full five-fold training estimate" 절) | 일치(대략, fold 2~5 한정). 정확한 개수는 확인되지 않음 |

**b. 윈도우와 라벨**

| 항목 | 논문 값(근거) | 원본 코드 값 | 코드 근거 | 판정 |
|---|---|---|---|---|
| 윈도우 길이 T | 30 (§5.2 p.7) | `window_size=30`, 모델 `step_size=30` | P:263, T:505 | 일치 |
| stride | 2 (§5.2 p.7) | `stride=2`, 윈도우 수 `((n-30)//2)+1` | P:264, 32 | 일치 |
| zero padding | 30개 미만이면 zero padding (§5.2 p.7) | `n <= 30`이면 0으로 채운 `[1,30,4]`의 뒤쪽에 입력을 넣음(앞쪽이 0). `n > 30`이면 완전한 윈도우만 만들고 남는 꼬리 깜빡임은 버림 | P:27-30, 32-38 | 일치(30개 미만 패딩). 꼬리 처리는 기준값에 없음 |
| 시퀀스 라벨 | 영상 라벨 (§5.2 p.7) | alert 0, semisleepy 5, sleepy 10을 해당 영상의 모든 윈도우에 부여 | P:88, 106, 125 | 일치 |

**c. 라벨 값과 이산화**

| 항목 | 논문 값(근거) | 원본 코드 값 | 코드 근거 | 판정 |
|---|---|---|---|---|
| 라벨 값 | 0, 5, 10 (§3.3) | `labels_pool = [0, 5, 10]` | T:16, 57, 90, 124 | 일치 |
| 이산화 경계 | Alert 0 ≤ out < 3.3, LowVigilant 3.3 ≤ out ≤ 6.6, Drowsy 6.6 < out ≤ 10 (Eq.10 p.7) | 예측을 0~10으로 `np.clip`한 뒤 `out // 3.34`의 몫(0, 1, 2)으로 라벨을 고름. 즉 [0, 3.34)→0, [3.34, 6.68)→5, [6.68, 10]→10(하한 포함). 회귀 오차의 S 상수는 3.3과 6.6을 따로 씀 | T:126-127, 99-100, 72-73, 28-29. S 상수: T:31, 33, 40, 42, 68, 70, 75-76 | **불일치**(경계값 3.3/6.6 대 3.34/6.68, 그에 따른 포함 구간) |

**d. 평가 지표**

| 항목 | 논문 값(근거) | 원본 코드 값 | 코드 근거 | 판정 |
|---|---|---|---|---|
| BSA | 투표 전, 이산화 후 전체 테스트 시퀀스 정확도 (§5.1 p.7) | 테스트 시퀀스 전체(`TestB`)에 대해 이산화한 라벨과 정답이 같은 비율 | T:123-131, 371-372 | 일치 |
| BSRE | Eq.12: 오분류 시퀀스의 \|out-S\|² 합 / M | 영상별로 오분류 시퀀스(`predicted_labels != Y`)만 (out-S)²를 합하고 전체 테스트 시퀀스 수(`len(labels)`)로 나눔. 맞은 시퀀스 기여 0 | T:54-84 | 일치. M의 정의는 기준값에 없고 코드는 전체 테스트 시퀀스 수 |
| VA | 영상 정확도 (§5.1 p.7) | 영상 안 시퀀스 라벨의 다수결로 영상 라벨을 정하고, 동률이면 영상 평균 예측으로 정함. 맞은 영상 수 / 영상 수 | T:87-117 | 일치(영상 정확도). 투표·동률 처리 방식은 기준값에 없음 |
| VRE | Eq.13: 오분류 영상의 시퀀스별 (out-S)² 평균의 합 / Q, 정답 영상 기여 0 | 영상의 시퀀스 예측을 먼저 평균(`ave_predicts`)하고, 그 평균으로 영상을 판정하며, 오분류 영상에 대해 **평균 예측의 제곱 오차** (ave-S)²를 더해 영상 수로 나눔. 정답 영상 기여 0 | T:13-51 (평균 27, 판정 28-29, 손실 35-44, 합 46, 분모 51) | **불일치**(시퀀스별 (out-S)²의 평균 대 영상 평균 예측의 (ave-S)²). 정답 영상 기여 0과 분모(영상 수)는 일치 |
| S 정의 | 참 상태 구간의 가장 가까운 경계 (§5.1 p.7) | 참 라벨 0 → 3.3, 10 → 6.6. 참 라벨 5이면 예측(BSRE는 예측 라벨, VRE는 영상 평균 예측)이 5 미만이면 3.3, 이상이면 6.6 | T:30-33, 39-42, 67-70, 74-76 | 일치 |
| BSA loss | 논문에 없음 | 테스트 세트 전체에 대한 `loss`(Eq.11 형태 + 0.1·L2 항), BN 추론 모드, keep 확률 1.0으로 계산해 `BSA:` 줄로 출력 | T:313-335, 371, 379 | 논문에 없음 |

**e. fold 정의와 영상 수**

| 항목 | 논문 값(근거) | 원본 코드 값 | 코드 근거 | 판정 |
|---|---|---|---|---|
| fold 구성 | 60명, 영상 180개, 5 fold × 12명(fold당 36개) (§3.3 p.5) | fold별 테스트 영상 시작 인덱스 목록 `start_indices`의 길이가 33, 36, 36, 36, 33이다. 목록은 "내 경우의 Blinks·BlinksTest 배열 기준"으로 하드코딩되어 있다고 주석에 적혀 있다 | T:440-443, 445, 456, 465, 477, 489 | **불일치**(Fold1·Fold5가 33개, 36개가 아님) |
| 영상 수 합계 | 180 | 33+36+36+36+33 = 174(이번에 다시 계산). `regression_eval`의 영상별 출력 줄 수도 174 | T 위 줄. `experiments/00_reproduction/regression_eval/pretrained_eval_stdout.log` | **불일치**(174 대 180). 차이의 원인은 미확인 |
| 제외 영상 | 일부 영상은 품질 사유로 무시했다고만 서술, 수와 목록은 없음 (§3.2 p.4) | 원본 README와 주석에 제외 영상이나 피험자 언급이 없다(키워드 검색 결과 없음) | R, T, P 전체 | 논문에 없음. 코드와 README에서도 확인 불가 |

**f. 시드, 체크포인트, 평가 진입**

| 항목 | 논문 값(근거) | 원본 코드 값 | 코드 근거 | 판정 |
|---|---|---|---|---|
| 시드 고정 | 논문에 없음 | `Training.py`에 시드 호출이 없다(`set_random_seed`, `random.seed`, `np.random.seed` 검색 0건). Xavier 초기화 `seed=None`. 학습 데이터 순서는 고정(셔플 없음). 학습 데이터 파일 자체의 순서는 `Preprocessing.py`가 시드 없는 셔플로 만든다 | T:155 외, 265-281. P:4-6, 255 | 논문에 없음 |
| 반복 횟수 | 논문에 없음 | 주석이 fold를 여러 번 돌려 최선 성능을 보라고 안내 | T:408-409 | 논문에 없음 |
| 체크포인트 저장 시점 | 논문에 없음(선택 기준) | `save_variables`는 epoch 루프가 끝난 뒤 1회 호출. 주석 "Saving every epoch"와 불일치 | T:283-286, 395-396 | 논문에 없음 |
| 평가 진입 방식 | 논문에 없음 | `load=True`면 `my_model<fold>`를 복원하고 첫 평가 후 `break`. `load=True`일 때 `ii=i+1` | T:338-341, 391-392, 407, 411-413 | 논문에 없음 |

**g. 정규화와 특징 추출**

| 항목 | 논문 값(근거) | 원본 코드 값 | 코드 근거 | 판정 |
|---|---|---|---|---|
| 피험자별 정규화(1차) | Eq.6 p.6: alert 영상 깜빡임의 **처음 1/3**로 피험자별 평균·표준편차를 구하고, 그 구간은 학습·테스트에 쓰지 않음 | `bunch_size = 깜빡임 수 // 3`. 평균·표준편차는 **마지막** `bunch_size`개(`Freq[-bunch_size:]` 등)로 구하고, 표준편차가 0이면 alert 전체의 표준편차로 대체. 정규화·시퀀스화는 앞쪽 `remained_size`개(처음 2/3)만 사용하므로 마지막 1/3은 쓰이지 않음. semisleepy·sleepy는 같은 통계로 정규화 | P:56-77(`gen`), 168-190(`Preprocess`), 98-99, 116-117 | **불일치**(기준 구간이 처음 1/3이 아니라 마지막 1/3). 기준 구간을 쓰지 않는 점은 일치. 표준편차 0 대체는 논문에 없음 |
| 2차 정규화 | 이후 전 피험자 2차 정규화 (Eq.6 p.6) | 학습 `Blinks`의 특징별 평균·표준편차로 학습·테스트 모두 z-score | T:420-431 | 일치(특징별, 학습 통계 사용) |
| 제공 `.npy`의 정규화 상태(Eq.6과의 관계) | Eq.6 p.6: 피험자별 정규화 후 전 피험자 2차 정규화 | 코드상 `Preprocessing.py`는 `normalize_blinks`로 피험자별 z-score를 적용한 값만 시퀀스로 만들어 저장 대상(`blinksTest`, 학습용 `output`)으로 삼는다. 코드 주석도 이 단계를 "First Phase"라 부른다. 2차 정규화는 `.npy`에 저장되지 않고 `Training.py`가 파일을 읽은 뒤 메모리에서 적용한다("Second phase"). 값 분포 보조 확인(읽기 전용): 제공 `Blinks_30_Fold{1..5}.npy`에는 진폭·지속시간·속도에 음수가 있는데, `blink_video.py`는 진폭이 `MIN_AMPLITUDE`(0.04)보다 크고 속도가 0보다 큰 깜빡임만 기록하고 지속시간은 `end-start+1`이므로 원래 값은 음수일 수 없다. 또 특징별 평균·표준편차가 0·1이 아니다(Fold1 학습 `Blinks`, 패딩 포함 전체 값 기준 평균 [19.418, -0.070, 0.701, -0.162], 표준편차 [46.004, 1.642, 3.868, 1.597]. Fold2~5도 평균은 0이 아니고 표준편차는 1이 아님) | P:77-79, 98-99, 116-117, 127-135, 189-190, 213-214, 231-232, 242-259, 267, 271-272. T:419-431. B:51, 179, 220, 461 | **1차 정규화 후, 2차 정규화 전**으로 판단(코드와 값 분포가 모두 이 해석과 일치). 다만 제공 `.npy`가 이 코드로 만들어졌다는 점은 원본 README(R:58-62)의 서술에 의존하며, 코드만으로는 확정 불가(`Preprocessing.py`는 Fold3 고정이고 학습용 `np.save`가 주석 처리됨, P:262-272) |
| Eq.1 EAR | EAR[k] = (‖p2-p6‖ + ‖p3-p5‖) / ‖p1-p4‖. 분모에 2배 계수가 없다(보충 Eq.14도 같음) | `ear = (A + B) / (2.0 * C)`. A = ‖eye[1]-eye[5]‖, B = ‖eye[2]-eye[4]‖, C = ‖eye[0]-eye[3]‖(eye[0..5]를 p1..p6에 대응). 이어서 C < 0.1이면 0.3으로, 0.45 초과면 0.45로 고정하고, 좌우 눈의 EAR을 평균해 사용 | B:74-92(식 88), 417-421 | **불일치**(분모의 2배 계수). 좌우 평균과 0.3·0.45 고정은 기준값에 없음 |
| Eq.2 Duration | Duration = end - start + 1 | `duration = end - start + 1` | B:179(같은 식 67, 248, 267, 283) | 일치 |
| Eq.3 Amplitude | Amplitude = (EAR[start] - 2·EAR[bottom] + EAR[end]) / 2 | `amplitude = 0.5*(startEAR - peakEAR) + 0.5*(endEAR - peakEAR)`. 전개하면 (startEAR - 2·peakEAR + endEAR)/2이며 후보 단계의 식도 같은 형태. peakEAR은 눈 감는 구간 직후의 낮은 극값 | B:180(후보 단계 66, 247, 266) | 일치(수식 동치, bottom = peakEAR로 대응) |
| Eq.4 EyeOpeningVelocity | (EAR[end] - EAR[bottom]) / (end - bottom) | `velocity = (endEAR - selected_EAR_First[2*j+1]) / (end - selected_index_First[2*j+1] + 1)`. 분모에 +1이 있다. 빼는 값은 눈 뜨는 쪽으로 넘어가기 직전의 낮은 극값(`selected_*_First[2*j+1]`)이며, 진폭에 쓰는 `peakEAR`(`selected_*_Sec[2*j]`)과는 코드상 별도 변수이다. 낮은 극값이 하나뿐이면 같은 지점이고 여럿이면 다를 수 있는지는 코드 구조상 확정 불가. 주석은 "eye opening ave velocity"이고 속도가 0 이하인 깜빡임은 기록하지 않는다 | B:181(극값 선택 159-162, 171-177), 461 | **불일치**(분모 +1). 분자의 bottom이 Eq.3의 bottom과 같은 지점인지는 확인 불가 |
| Eq.5 Frequency | Frequency = 100 · (blink_i까지의 깜빡임 수) / (end_i까지의 프레임 수) | `BLINK_FRAME_FREQ = TOTAL_BLINKS / number_of_frames`를 기록할 때 ×100. `TOTAL_BLINKS`는 확정된 깜빡임을 누적하며 한 번에 여러 개가 확정되면 묶음 전체를 더한 값을 그 묶음의 모든 깜빡임에 기록한다. `number_of_frames`는 얼굴이 검출된 프레임만 센 값이고 깜빡임이 확정되어 기록되는 시점의 값이다 | B:455, 464(누적 229, 456-464, 프레임 수 350, 380, 확정 호출 447) | 일치(구조: 100 × 누적 깜빡임 수 / 누적 프레임 수). 분모가 end_i까지의 프레임 수와 같은 값인지는 확인 불가. 묶음으로 확정된 경우 분자가 blink_i까지의 개수와 다를 수 있음(코드 사실) |
| Eq.7 Pre-FC | F = ReLU(B·W + b), W는 4×L(L=32), 시간 스텝 공유 | `pre_fc1_weights`가 `[4, 32]`이고, 입력을 `[-1, 4]`로 펼쳐 모든 시간 스텝에 같은 W를 곱한 뒤 시간 축으로 되돌린다. 곱한 결과에 batch norm을 적용하며 `pre_fc1_biases`는 BN의 offset 자리에 들어간다(`batchNorm(x, beta=biases, gamma=None)`). 그다음 ReLU, dropout(keep 확률 1.0) | T:154, 160-165, 133-144 | **불일치**(식 기준: 식에 없는 batch norm이 들어가고 b가 BN offset 위치). W의 크기, 시간 스텝 공유, ReLU는 일치. 논문 §5.2의 batch norm 사용 서술과는 일치 |
| Eq.8 FC1(층별) | e_1l = ReLU(h_l·W_1l + b_1l). h_l은 각 HM-LSTM 층의 마지막 시간 스텝 은닉 상태, 층별 16 | h_l은 `output_RNN_set[lay]`의 마지막 시간 스텝(`split`으로 `[step-1, 1]`)이고 이 값은 층별 은닉 h(셀 출력의 앞 `hstate_size`열)이다. `W_1l`은 `[32, 16]`이며 곱한 뒤 batch norm(`post_fc1_biases`가 BN offset 자리), 4개 층 결과를 이어 붙인 뒤 ReLU와 dropout을 적용한다(ReLU는 원소별이라 층별 적용과 같다) | T:180-198, 181. `third_party/ghoddoosian/MultiHMSLSTM.py`:37,46. `third_party/ghoddoosian/HMSLSTM_cell.py`:80 | **불일치**(식 기준: batch norm 삽입, b가 BN offset 위치). h_l의 정의와 층별 16은 일치 |
| Eq.9 출력 | out = 10 · Sigmoid(e_4·W_o + b_o) | e_4를 FC4 출력(`post_fc2_out`, 8차원, matmul → BN → ReLU → dropout)으로 대응하고, `output = matmul(post_fc2_out, last_fc_weights) + last_fc_biases`(`[8, 1]`, bias는 BN 없이 더함) 뒤에 `10 * tf.sigmoid(output)`을 적용 | T:229-241, 243, 247, 250, 255 | 일치(마지막 층에는 batch norm이 없고 식과 같은 형태) |
| 특징 추출·입력 열(기타 사실) | 논문 기준값에 없음 | `blink_video.py`는 [누적 깜빡임 수, 빈도×100, 진폭, 지속시간, 속도]를 한 깜빡임당 한 줄로 쓰고, `Preprocessing.py`는 2~5번째 열(`usecols=1..4`)을 사용한다 | B:464. P:51-54 | 논문에 없음 |

**h. 원본 README의 환경·사용법**

| 항목 | 논문 값(근거) | 원본 README | 코드 근거 | 판정 |
|---|---|---|---|---|
| OS, TensorFlow | 논문에 없음 | Ubuntu 16.04, TensorFlow 1.8에서 테스트 | R:7 | 논문에 없음 |
| 요구 패키지 | 논문에 없음 | `tensorflow-gpu==1.8.0`, scipy, dlib(menpo), scikit-image, imutils, scikit-learn, opencv 설치 명령과 apt 패키지. 버전 고정은 TensorFlow뿐이고 requirements 파일은 없음 | R:23-43 | 논문에 없음 |
| 사용법 | 논문에 없음 | 1) `blink_video.py`(영상 → 깜빡임 특징 텍스트, dlib `.dat` 별도 다운로드) 2) `Preprocessing.py`(`.npy` 생성, 편의를 위해 제공) 3) `Training.py`(학습·평가, 사전학습 모델 5개 제공) | R:46-71 | 논문에 없음 |

## 4. 요약

각 항목은 3절 비교표 또는 3.3절 소항목의 이름으로 참조한다.

### 의도적 변경

- **실행 방식**: 래퍼의 메모리 치환(3.1절). 모델 코드는 바이트 동일하다.
- **난수 시드**: 시드를 고정하는 학습 래퍼 추가.
- **epoch별 지표 기록**: `metrics.csv`, 보고값은 마지막 epoch 값.
- **체크포인트**: 재학습 결과의 저장 위치와 이름(`my_model<fold>`), git 비추적.

### 비의도적 차이

- **환경(요약)**: OS(Windows, 원본은 Ubuntu 16.04). 환경 제약에서 온 차이다.
- **환경(요약)**: dlib는 19.16, 19.15를 설치할 수 없어 19.17을 썼다. 원본이 dlib 버전을 명시하지 않아 실제 차이 여부는 아래 미확인에 속한다.

### 미확인

- **환경(요약)**: CPU 전용 실행이 원본과 다른지(원본 하드웨어가 명시되지 않음), Python, NumPy, SciPy, scikit-learn, scikit-image, dlib, OpenCV, imutils, matplotlib, protobuf 버전(원본이 명시하지 않음).
- **체크포인트**: 공개 `my_model1~5`가 어떤 실행·epoch에서 선택되었는지.
- **영상→특징 추출**: `blink_video.py`, `Preprocessing.py`의 재현 여부(미재현). Eq.4의 bottom 지점이 Eq.3과 같은지, Eq.5의 분모 프레임 수가 end_i까지의 값과 같은지(3.3절 g).
- **제공 `.npy`의 생성 이력**: 제공 파일이 `Preprocessing.py`로 만들어졌는지는 원본 README의 서술에 의존하며 코드만으로 확정할 수 없다(3.3절 g).
- **데이터셋**: 174와 180의 관계, 제외 영상의 수와 목록.
- **결정성 확인**: 다른 하드웨어·OS에서의 결정성.
- **HM-LSTM 셀 내부**: 셀 내부 가중치 초기화(3.3절 a).
- **재현 범위 밖(하지 않은 것)**: 3절의 해당 행에 적은 항목들.

### 논문(기준값)과 원본 코드 사이의 불일치

이 저장소가 만든 차이가 아니라 원본 코드가 논문 기준값과 다르게 쓰여 있는 지점이다. 이 저장소는 원본을 바이트 그대로 쓰므로 그대로 따른다. 원인은 추정하지 않으며 모두 3.3절 근거를 따른다.

- **이산화 경계**(3.3절 c): 논문 3.3/6.6, 코드 3.34/6.68.
- **VRE 식**(3.3절 d): 논문 시퀀스별 (out-S)²의 평균, 코드 영상 평균 예측의 (ave-S)².
- **fold별 영상 수와 합계**(3.3절 e): 논문 36개씩·180, 코드 33/36/36/36/33·174.
- **피험자별 정규화 구간**(3.3절 g): 논문 처음 1/3, 코드 마지막 1/3.
- **Eq.1 EAR**(3.3절 g): 논문 분모에 2배 계수 없음, 코드 `(A+B)/(2.0*C)`.
- **Eq.4 열림 속도**(3.3절 g): 논문 분모 `end - bottom`, 코드 분모에 +1.
- **Eq.7·Eq.8 batch norm**(3.3절 g): 논문 식에 없는 batch norm이 코드에 들어 있고 b가 BN offset 자리에 쓰인다. 논문 §5.2의 batch norm 사용 서술과는 일치한다.

식 기준으로 일치한 것은 Eq.2, Eq.3, Eq.9이고, Eq.5는 구조만 일치한다.

## 5. 공개 체크포인트 평가 대 논문 Table 2·3

비교 대상: `experiments/00_reproduction/regression_eval/pretrained_5fold_results.csv`(공개 `my_model1~5`를 복원해 평가만 한 출력, 학습 없음).
기준: 소수 둘째 자리 반올림 일치 여부. 논문 값(Table 2·3)은 작업 지시 기준으로 논문 PDF 원문과 대조가 끝난 값이다.

### 5.1 fold별 (논문 Table 3: VA / VRE)

| Fold | VA 논문 | VA 측정(원값 → 반올림) | VA 일치 | VRE 논문 | VRE 측정(원값 → 반올림) | VRE 일치 |
|---|---|---|---|---|---|---|
| 1 | 0.64 | 0.636364 → 0.64 | 일치 | 2.42 | 2.419539 → 2.42 | 일치 |
| 2 | 0.61 | 0.611111 → 0.61 | 일치 | 1.04 | 1.039631 → 1.04 | 일치 |
| 3 | 0.70 | 0.694444 → 0.69 | **불일치** | 0.58 | 0.667389 → 0.67 | **불일치** |
| 4 | 0.64 | 0.638889 → 0.64 | 일치 | 0.85 | 0.849650 → 0.85 | 일치 |
| 5 | 0.67 | 0.666667 → 0.67 | 일치 | 0.81 | 0.812155 → 0.81 | 일치 |

### 5.2 평균 (논문 Table 2)

| 지표 | 논문 | 측정(5-fold 평균, 원값 → 반올림) | 일치 |
|---|---|---|---|
| BSRE | 1.90 | 1.901040 → 1.90 | 일치 |
| VRE | 1.14 | 1.157673 → 1.16 | **불일치** |
| BSA | 54% (0.54) | 0.541082 → 0.54 | 일치 |
| VA | 65.2% (0.652) | 0.649495 → 0.65 | 일치(소수 둘째 자리 기준). 백분율 소수 첫째 자리(64.9% 대 65.2%)로는 다름 |

### 5.3 집계

- 논문과 비교 가능한 칸은 14칸(Table 3의 VA·VRE 10칸, Table 2의 평균 4칸)이다. 11칸이 일치하고 3칸이 불일치한다. 불일치 칸은 **Fold3 VA, Fold3 VRE, 평균 VRE**이다.
- 논문에 값이 없어 비교하지 못하는 칸: 5개 fold의 BSA loss(논문에 정의 자체가 없음), 5개 fold의 BSA acc와 BSRE(논문 Table 3은 VA·VRE만 보고). 5 fold × 5 지표 = 25칸 중 논문과 비교 가능한 것은 fold별 VA·VRE 10칸뿐이다.
- 산술 사실: 기준값 Table 3의 fold별 VA 평균은 0.652, VRE 평균은 1.14로 Table 2와 같다.
- 원인은 추정하지 않는다. 불일치는 모두 미확인이다.

### 5.4 Fold3 영상 수와 정답 영상 수

`experiments/00_reproduction/regression_eval/pretrained_eval_stdout.log`의 영상별 출력("True label is … detected label is …")에서 Fold3는 영상 36개 중 25개가 정답이다(25/36 = 0.694444, CSV의 VA와 같다). 영상 부분집합을 바꾸어 논문 값에 맞추는 계산은 하지 않았다.
같은 로그에서 fold별 영상 수는 33, 36, 36, 36, 33(합 174)이고, 정답 영상 수는 21, 22, 25, 23, 22이며 각각 CSV의 VA와 일치한다.

### 5.5 "25/25 재현"의 비교 대상과 허용오차

- **비교 대상은 논문이 아니다.** `docs/REPRODUCTION_LOG.md`의 "Stage 3" 절("all 25 values match")과 이 문서 이전 판의 "25/25"는, 새 폴더 구조에서 평가를 다시 실행한 결과가 이 프로젝트의 이전 실행 결과(같은 환경, `docs/REPRODUCTION_LOG.md`의 "Pretrained 5-fold evaluation" 절 표, 245-253행)를 표 표기 자릿수로 반올림한 기대값과 일치하는지 본 것이다. 자체 기대값과의 일치이며 재실행 재현성 확인이다.
- **허용오차**: 표의 표기 자릿수 반올림(BSA loss, BSRE, VRE는 소수 3자리, BSA acc, VA는 소수 4자리). `src/repro/reproduction_eval.py`에는 Fold1에 한해 이전 실행 값(`FOLD1_REFERENCE`)과의 차이가 1e-4 미만인지 보는 검사(33-34, 151-152행)가 있고, 이 역시 자체 기대값이다. 논문 값과의 비교 코드는 없다.
- **정정**: 논문 Table 2·3과의 대조 결과는 5.1~5.3절이다. 14칸 중 11칸 일치, 3칸 불일치이며 "25/25"를 논문 재현의 근거로 쓰지 않는다.

## 6. 결과 파일 안내

수치는 아래 파일에서 확인한다(5절을 제외하면 이 문서에는 복사하지 않는다).

- 공개 체크포인트 5-fold 평가: [`experiments/00_reproduction/regression_eval/`](../experiments/00_reproduction/regression_eval/)
  (이전 실행 기록과 25/25 일치한다는 기록: [`REPRODUCTION_LOG.md`](REPRODUCTION_LOG.md)의 "Stage 3 - regression test in the new structure" 절. 비교 대상의 의미는 5.5절)
- Fold1 다중 시드 재학습 요약: [`experiments/01_fold1_seeds/summary.md`](../experiments/01_fold1_seeds/summary.md)
- fold별 재학습 요약(시드별 `summary_seed*.md`): [`experiments/02_fold_seeds/summary_seed1.md`](../experiments/02_fold_seeds/summary_seed1.md)

## 7. 알려진 미해결 사항

- **공개 체크포인트 대비 재학습 격차의 원인**: 미확인.
- **논문 대비 공개 체크포인트 평가의 불일치 3칸(Fold3 VA, Fold3 VRE, 평균 VRE)의 원인**: 미확인(5절).
- **논문과 원본 코드 사이의 불일치 7건**(이산화 경계, VRE 식, fold별 영상 수, 정규화 구간, Eq.1 계수, Eq.4 분모, Eq.7·8 batch norm)의 원인과 영향: 미확인(4절).
- **영상 수 174/180**: 미확정(3.3절 e).
- **`configs/protected_manifest.sha256`의 CRLF 문제**: `* text=auto`로 Windows에서 이 파일이 CRLF로 체크아웃되어 `sha256sum -c`가 실패했었다. `.gitattributes`에 `configs/*.sha256 text eol=lf`가 현재 적용되어 있고, 저장소 기록(`git ls-files --eol`)상 작업 폴더와 인덱스 모두 LF이다. 근거: `.gitattributes`, `docs/REPRODUCTION_LOG.md`(후속 수정 절).
- **`src/repro/reproduction_train_fold1.py`는 레거시**: 출력 폴더가 `experiments/00_reproduction/fold1_training/`로 고정되어 있고, 폴더 안에 `my_model1*`가 있을 때만 실행을 거부한다. 체크포인트는 저장소에 없으므로 다시 실행하면 같은 폴더의 `training_history*.csv`, `training_stdout.log`(git이 추적하는 파일)를 덮어쓸 수 있다. 새 실험은 `train_fold_seeds.py`, `train_fold1_seeds.py`를 쓴다.

## 8. 발견된 문제점 목록 (2026-10-02)

이 절은 추가만 한다. 확실도 표기: **[확실]** = 이 세션에서 해당 파일을 직접 읽었거나 직접 측정한 값, **[가능성 높음]** = 코드나 측정에서 도출했지만 실행하지 않았거나 논문 PDF를 열어 확인하지 못한 것, **[추측]** = 근거가 약한 해석. 직접 확인하지 못한 항목은 "미검증"이라고 쓴다. 논문 서술: 이 저장소에 논문 PDF가 없고(17행) 이 세션에서도 열지 않았다. 사용자가 논문 PDF 본문(4~8쪽 및 §3.2)을 확인했다고 알려 준 항목(P-08, P-09, P-10, P-13, P-15)은 [확실] (사용자 확인)으로 적었고, 그 밖의 논문 서술은 [가능성 높음] 이하이다. `third_party/ghoddoosian/` 파일의 줄 번호는 이 세션에서 해당 파일을 직접 읽어 확인한 줄만 [확실]로 적었다. 원본 파일은 수정하지 않았고 아래 "처리"는 모두 이 원칙을 따른다.

| ID | 구분 | 내용 | 근거 | 확실도 | 영향 | 상태·처리 |
|---|---|---|---|---|---|---|
| P-01 | 저자 코드 | 얼굴 미검출 프레임 분기에서 `Blink_Tracker` 호출이 `if Counter4blinks==0:` 안에 들어 있다. 얼굴 검출 분기는 그 `if` 바깥에서 호출한다. [확실] (코드 읽기). (2026-10-03 갱신) 추적 분기는 첫 얼굴 검출 이후에만(`third_party/ghoddoosian/blink_video.py:491`) 두 눈 6점씩 12점을 Lucas-Kanade로 추적하고(`third_party/ghoddoosian/blink_video.py:494-495`), 12점이 모두 성공할 때만 EAR을 갱신하며(`third_party/ghoddoosian/blink_video.py:497-507`) 입은 추적하지 않는다(486~566줄에 입 관련 코드가 없다). [확실] (코드 읽기) | `third_party/ghoddoosian/blink_video.py:531-536`(미검출), `third_party/ghoddoosian/blink_video.py:445-450`(검출) | 구조 [확실], 의도 [추측] | blink 진행 중(`Counter4blinks>0`)에 얼굴이 사라지면 그 프레임에서 `Blink_Tracker`가 호출되지 않아 blink가 갱신·종료되지 않는다. [가능성 높음] (실행하지 않음). 저자 의도 여부는 [추측] | 수정하지 않음 |
| P-02 | 저자 코드 | 출력 기록 시 `velocity>0` 필터는 얼굴 검출 분기에만 있고 미검출(광류 추적) 분기에는 없다. [확실] (코드 읽기). (2026-10-03 갱신) 필터는 `third_party/ghoddoosian/blink_video.py:461`(얼굴 분기)에만 있고 `third_party/ghoddoosian/blink_video.py:546-548`(추적 분기)에는 없다. 누적 blink 수는 필터와 무관하게 센다(`third_party/ghoddoosian/blink_video.py:229`). 측정: 시험 출력(Fold1_part2/10/0.MOV, `cw90`)의 마지막 행 누적 값은 20인데 기록된 행은 19개였다. [확실] (측정). 필터로 버려진 blink가 전체에서 몇 개인지는 미검증이다. | `third_party/ghoddoosian/blink_video.py:461`, `third_party/ghoddoosian/blink_video.py:546-548` | [확실] | 미검출 분기에서 확정된 blink는 속도가 0 이하여도 기록된다. [확실] (코드 읽기). 결과 파일에 미치는 규모는 미검증 | 수정하지 않음 |
| P-03 | 저자 코드 | 회전 미처리. 90° 회전 코드는 주석 처리돼 있고, 켜도 출력 크기를 `(cols, rows)`로 고정해 가로 프레임이 잘린다. [가능성 높음] (코드 읽기에서 도출, 실행하지 않음). OpenCV 3.4.5는 회전 태그를 적용하지 않는다. [가능성 높음]. 사례: Fold1_part2/10/0.MOV(세로 촬영, 저장 568×320, 배치 시험에서 종료 코드 0, blink 0행, 367.2초). [확실] (사용자 보고·육안). 45번 피험자 영상(상하 반전)도 같은 계열이다. [추측]. 회전 조사(규칙 v1, 182개): `0` 80, `cw90` 37, `ccw90` 32, `180` 4, `ambiguous` 26, `none` 3. `none` 3개는 Fold2_part2/23/0.mp4, Fold2_part2/23/10.mp4, Fold5_part1/49/0.mp4. `faces_orig==0` 84개 중에서는 cw90 37, ccw90 32, 180 3, ambiguous 8, none 3, 0 1. [확실] (측정). 규칙 v2(`resolved_orientation`, 사후 설계) 기준: `0` 97, `cw90` 37, `ccw90` 30, `180` 4, `review` 14. [확실] (측정). 사전 등록 예측 (a)(b) 적중, (c)는 대조군이 v1에서 `ambiguous`로 빗나가 사전 규칙상 기각 판정이 났다. 이 기각은 v1 판정 규칙의 신뢰도 문제로 정리했다(P-18, `docs/REPRODUCTION_LOG.md`의 정정·보강 항목). [확실] | `third_party/ghoddoosian/blink_video.py:361-366`, `configs/video_orientation_survey.csv`, `docs/REPRODUCTION_LOG.md:537-575`, `docs/REPRODUCTION_LOG.md:590-632` | 측정 [확실], 원인(회전 태그) 미검증 | 누운 프레임에서는 얼굴 검출이 0에 가까워 blink 0행이 될 수 있다. [가능성 높음]. 정상 방향 영상 18개도 180° 회전 검출이 5~8프레임 나와 `ambiguous`가 된다(방법의 한계). [확실] (측정) | 회전 옵션 추가(기본 none), 저자 코드 resize 표현식 1곳(`blink_video.py:359`)을 메모리에서 치환하는 의도적 변경(`src/repro/reproduction_blink_headless.py`의 `--rotate`, 원본 파일은 수정하지 않음). 이 방향은 회전 태그를 적용하는 디코더가 주는 프레임과 같다고 [추측]하며 회전 태그는 읽지 않았다. 회전 태그가 원인인지는 미검증. 시험 2건(cw90, 180) 통과, 기본값 none은 GUI 기준과 동일(사전 등록 `docs/REPRODUCTION_LOG.md:634-651`, 결과 `docs/REPRODUCTION_LOG.md:663-675`). [확실] (측정). ccw90은 시험하지 않았다(미검증). 통과 기준은 "blink 행이 나오고 얼굴이 검출되는 프레임이 대부분"이며 cw90·180 실행의 blink 값 정확성은 미검증. review 14개 육안 판정 완료(관찰자: 사용자 육안, `configs/video_orientation_overrides.csv`, 결과 `docs/REPRODUCTION_LOG.md:677-696`): `0` 3개, `cw90` 7개, `ccw90` 4개. [확실] (측정). 판정한 방향으로 실제 추출한 결과는 미검증 |
| P-04 | 저자 코드 | 출력 파일은 blink를 쓸 때만 만들어진다(`open(output_file, 'ab')`가 기록 블록 안). blink 0개 영상은 파일이 없다. [확실] (코드 읽기). `Preprocessing.py`는 파일이 없는 피험자의 값을 이전 반복의 값으로 조용히 재사용한다(첫 피험자이면 미정의 변수 오류). [가능성 높음] (코드 읽기, 실행하지 않음). blink 0개 파일을 읽으면 n=0에서 `ValueError`. [확실] (numpy 메모리 시험). 저자 데이터의 중복(Fold3 test 영상 24와 27의 시퀀스 일치)과의 연관은 [추측], 미검증 | `third_party/ghoddoosian/blink_video.py:462`, `third_party/ghoddoosian/blink_video.py:546`, `third_party/ghoddoosian/Preprocessing.py:49-125`(변수 대입), `third_party/ghoddoosian/Preprocessing.py:127-128`, `third_party/ghoddoosian/Preprocessing.py:161-239`, `third_party/ghoddoosian/Preprocessing.py:242-243`(연결), `third_party/ghoddoosian/Preprocessing.py:29`(n=0) | 코드 사실 [확실], 결과 [가능성 높음], 데이터와의 연관 [추측] | 저자 데이터에서 파일이 빠진 피험자가 있었다면 이전 피험자의 시퀀스가 중복될 수 있다. [가능성 높음]. 저자의 데이터 폴더 구성은 미검증 | 수정하지 않음 |
| P-05 | 우리 도구 | 배치 도구의 상태 `no_output`은 처리 실패가 아니라 "종료 코드 0이면서 출력 파일이 없음", 곧 blink가 하나도 기록되지 않은 영상이다. `ok` = 종료 코드 0이고 출력 있음, `failed` = 종료 코드가 0이 아님. [확실] (코드 읽기). 종료 코드 0이면서 출력이 없는 이유(blink 0개, 얼굴 미검출, 회전)는 구분하지 않는다. [확실]. 예: Fold1_part2/10/0.MOV 배치 시험 결과. [확실] (사용자 보고) | `src/repro/extract_blinks_batch.py:455-465`(`judge_result`, `no_output`은 464행), `src/repro/extract_blinks_batch.py:50` | [확실] | 기본 실행은 `no_output`도 실패 기록으로 보고 건너뛰며 `--retry-failed`로만 다시 처리한다. [확실] (코드 읽기) | 상태 의미를 이 표에 정의. 코드는 이번 작업에서 바꾸지 않음 |
| P-06 | 저자 데이터 | SVM(`Trained_SVM_C=1000_gamma=0.1_for 7kNegSample.sav`)은 scikit-learn 0.18 이전 버전의 pickle이라 0.19.2에서 로드할 때 UserWarning이 난다("Trying to unpickle estimator SVC from version pre-0.18 when using version 0.19.2 ..."). [확실] (측정). 저자 환경과의 수치 동등성은 미검증 | `third_party/ghoddoosian/Trained_SVM_C=1000_gamma=0.1_for 7kNegSample.sav`, `src/repro/check_video_assets.py:27-32` | 경고 [확실], 동등성 미검증 | 예측이 저자 환경과 다를 수 있다. [추측]. 헤드리스와 GUI 출력이 일치한 것은 같은 환경 안의 비교일 뿐이다. [확실] | 환경(scikit-learn 0.19.2)을 고정해 사용. 미검증 |
| P-07 | 저자 코드 | 전처리 상수: resize width=450, 감마 1.5, `detector(gray, 0)`(업샘플 0), 첫 얼굴만 사용(`rects[0]`), 미검출 시 Lucas-Kanade 추적. [확실] (코드 읽기) | `third_party/ghoddoosian/blink_video.py:359`, `third_party/ghoddoosian/blink_video.py:370`, `third_party/ghoddoosian/blink_video.py:378`, `third_party/ghoddoosian/blink_video.py:386`, `third_party/ghoddoosian/blink_video.py:486-566` | [확실] | 영상 해상도와 방향이 다르면 검출률이 달라진다. [가능성 높음]. 측정: `faces_orig==0` 비율이 fold별 41.7, 47.2, 32.4, 41.7, 67.6%이다. [확실] (측정, `configs/video_inventory.csv`) | 수정하지 않음 |
| P-08 | 논문-코드 차이 | frequency 분모 `number_of_frames`는 얼굴이 검출된 프레임만 센다. [확실] (코드 읽기). 논문 Eq.5는 100 × (blink_i까지의 blink 수) / (end_i까지의 프레임 수)이고 누적 정의이다. [확실] (사용자 확인, 논문 PDF 본문 확인). 구조(100 × 누적 blink 수 / 누적 프레임 수)는 같고, 코드는 분모에 얼굴이 검출된 프레임만 센다는 점이 다르다. [확실]. (2026-10-03 갱신) 출력 한 줄은 `%.4f` 5열 [누적 blink 수, 100 × 누적 blink 수 / number_of_frames, 진폭, 지속 시간, 속도]이다(`third_party/ghoddoosian/blink_video.py:455-464`, 추적 분기 `third_party/ghoddoosian/blink_video.py:541-548`). 분모 `number_of_frames`는 검출기가 얼굴을 낸 프레임만 센다(`third_party/ghoddoosian/blink_video.py:379-380`). 추적으로 이어진 프레임은 세지 않는다. [확실] (코드 읽기) | `third_party/ghoddoosian/blink_video.py:350`, `third_party/ghoddoosian/blink_video.py:380`, `third_party/ghoddoosian/blink_video.py:455`, 논문 PDF 본문 확인(저장소에는 PDF 없음, 4~8쪽 및 §3.2) | [확실] (논문은 사용자 확인) | 얼굴 미검출 프레임이 많은 영상은 분모가 작아져 frequency 값이 커진다. [가능성 높음] (산술). 크기는 미검증 | 수정하지 않음 |
| P-09 | 논문-코드 차이 | 피험자별 정규화 기준 구간: 코드는 alert blink의 **마지막 1/3**(`Freq[-bunch_size:]` 등)으로 평균·표준편차를 구하고 앞쪽 2/3만 학습·테스트에 쓴다. [확실] (코드 읽기). 논문 Eq.6의 정규화는 alert 상태 blink의 **처음 1/3**로 평균·표준편차를 구하며 그 구간은 학습·평가에 쓰지 않는다. [확실] (사용자 확인, 논문 PDF 본문 확인) | `third_party/ghoddoosian/Preprocessing.py:56-77`(`gen`), `third_party/ghoddoosian/Preprocessing.py:168-190`(`Preprocess`), `docs/IMPLEMENTATION_DIFF.md:157`, 논문 PDF 본문 확인(저장소에는 PDF 없음, 4~8쪽 및 §3.2) | [확실] (논문은 사용자 확인) | 정규화 통계가 달라져 특징 값이 달라진다. [가능성 높음]. 학습 성능에 미치는 영향은 미검증 | 수정하지 않음. 제공 `.npy`를 그대로 사용 |
| P-10 | 논문-코드 차이 | 이산화 경계: 코드는 예측을 0~10으로 클립한 뒤 `out // 3.34`의 몫(0, 1, 2)으로 라벨을 고르므로 경계가 3.34와 6.68이다. 논문 Eq.10의 경계는 3.3과 6.6이다. [확실] (사용자 확인, 논문 PDF 본문 확인). 코드의 `6.68`은 상수로 적혀 있지 않고 몫 2의 시작점으로 정해진다. [확실] (코드 읽기). 회귀 오차의 S 상수는 3.3과 6.6을 따로 쓴다. [확실] | `third_party/ghoddoosian/Training.py:28`, `third_party/ghoddoosian/Training.py:72`, `third_party/ghoddoosian/Training.py:99`, `third_party/ghoddoosian/Training.py:108`, `third_party/ghoddoosian/Training.py:126`(3.34), `third_party/ghoddoosian/Training.py:31`, `third_party/ghoddoosian/Training.py:33`, `third_party/ghoddoosian/Training.py:40`, `third_party/ghoddoosian/Training.py:42`(S 상수), 논문 PDF 본문 확인(저장소에는 PDF 없음, 4~8쪽 및 §3.2) | [확실] (논문은 사용자 확인) | 3.3~3.34, 6.6~6.68 사이 예측의 라벨이 논문 정의와 달라진다. [확실] (산술). 지표 수치에 미치는 크기는 미검증 | 수정하지 않음 |
| P-11 | 저자 코드 | 테스트 영상별 시퀀스 경계 `start_indices`는 fold별로 하드코딩돼 있고, 주석은 저자의 `Blinks`/`BlinksTest` 배열 기준이며 특징을 새로 만들면 다시 맞춰야 한다고 적는다. [확실] (코드 읽기) | `third_party/ghoddoosian/Training.py:440-443`(주석), `third_party/ghoddoosian/Training.py:444-498`(목록), `third_party/ghoddoosian/Training.py:502` | [확실] | 특징을 다시 만들면 경계를 재계산해야 VA, VRE, BSRE가 맞는다. [확실] (경계가 `Training.py:13-117`에서 쓰임). 재계산 도구는 없다 | 수정하지 않음. 재계산 도구는 미구현 |
| P-12 | 저자 데이터 | 제공 `.npy`는 피험자별 정규화(1차) 후 값이라 원시 특징을 복원할 수 없다. 피험자별 평균·표준편차는 어디에도 저장되지 않는다. [가능성 높음] (코드에서 도출). 측정: 열 0의 최솟값 -108.51, 최댓값 705.64 등 음수를 포함한다. [확실] (측정) | `third_party/ghoddoosian/Preprocessing.py:11-21`, `third_party/ghoddoosian/Preprocessing.py:59-77`(통계는 지역 변수), `docs/IMPLEMENTATION_DIFF.md:33`, `third_party/ghoddoosian/Blinks_30_Fold1.npy` 등 20개 | [가능성 높음] | 영상에서 새로 뽑은 특징과 제공 `.npy`를 직접 비교하려면 같은 방식의 1차 정규화를 다시 해야 한다. [가능성 높음] | 수정하지 않음 |
| P-13 | 논문-코드 일치(정정) | 2차 정규화는 코드와 논문 모두에 있다. 코드는 학습 `Blinks`의 특징별 평균·표준편차로 학습·테스트 모두 z-score를 한다. [확실] (코드 읽기). 논문에도 개인별 정규화 뒤 개인 간 2차 정규화(평균 0, 분산 1)가 있다. [확실] (사용자 확인, 논문 PDF 본문 확인). 평균·표준편차는 패딩 행(전부 0)을 포함한 전체 배열로 구한다. [확실] (코드 읽기). 패딩 행은 Fold1 train 221370행 중 193행(0.09%)이다. [확실] (측정) | `third_party/ghoddoosian/Training.py:420-431`, `docs/IMPLEMENTATION_DIFF.md:158`, `docs/REPRODUCTION_LOG.md:577-588`(정정 항목), 논문 PDF 본문 확인(저장소에는 PDF 없음, 4~8쪽 및 §3.2) | [확실] (논문은 사용자 확인) | `.npy`(1차 정규화 후)에 한 번 더 z-score가 걸린다. [확실] (코드 읽기). 패딩 포함이 통계에 주는 영향은 작아 보이나 미검증 | 정정으로 기록. 수정하지 않음 |
| P-14 | 저자 데이터 | Fold3 test(`BlinksTest_30_Fold3.npy`)에 Fold1, 2, 4, 5의 train 파일 어디에도 없는 시퀀스 13개(인덱스 2183~2195 = 영상 34~36, 라벨 0/5/10이 1/6/6)가 있다. [확실] (측정). 영상 24와 27(둘 다 라벨 10, 시퀀스 93개씩)과 영상 35와 36(라벨 5와 10, 6개씩)은 시퀀스 단위로 바이트까지 일치한다. [확실] (측정). Fold3 train에는 중복이 없고 Fold1, 2, 4, 5 train에는 93쌍의 중복이 있다. [확실] (측정). 원인은 미검증이다. H1/H2의 정의와 사전 등록은 Fold3 재평가 작업 시 `docs/REPRODUCTION_LOG.md`에 기록될 예정이다(현재 로그에는 없다). | `third_party/ghoddoosian/BlinksTest_30_Fold3.npy`, `third_party/ghoddoosian/LabelsTest_30_Fold3.npy`, `third_party/ghoddoosian/Training.py:465-466`(Fold3 `start_indices`) | 측정 [확실], 원인 미검증 | Fold3의 VA·VRE가 논문과 다른 것(5.1절)과의 관련은 [추측], 미검증 | 검증 대기(미검증). 가설을 쓰려면 별도 사전 등록이 필요하다 |
| P-15 | 논문-코드 차이 | 영상 수: 디스크 182개. [확실] (측정). 논문은 60명 × 3 = 180개 영상이다. [확실] (사용자 확인, 논문 PDF 본문 확인). §3.2에 품질 때문에 일부 영상을 제외했다는 서술은 있으나 제외 개수는 논문에 명시적으로 보고되지 않음. 저자 코드의 테스트 영상 174개(fold별 33/36/36/36/33). [확실] (코드 읽기). ID 32(Fold3_part2)와 49(Fold5_part1)는 클래스 10 파일 `10_1`, `10_2`를 갖는다. [확실] (측정). 디스크 fold별 영상 수는 36/36/37/36/37이다. [확실] (측정). 저자가 어떤 영상을 제외했는지는 [추측], 미검증 | `configs/video_inventory.csv`, `third_party/ghoddoosian/Training.py:445-446`, `third_party/ghoddoosian/Training.py:456-457`, `third_party/ghoddoosian/Training.py:465-466`, `third_party/ghoddoosian/Training.py:477-478`, `third_party/ghoddoosian/Training.py:489-490`, 논문 PDF 본문 확인(저장소에는 PDF 없음, 4~8쪽 및 §3.2) | 측정·코드·논문 [확실] (논문은 사용자 확인), 제외 영상 [추측] | 영상 목록이 저자의 것과 다르면 영상 단위 지표를 직접 비교할 수 없다. [가능성 높음] | 미해결(3.3절 e) |
| P-16 | 우리 도구 | 기존 단일 영상 래퍼는 `os.chdir(ROOT)`로 작업 폴더를 바꾼 뒤 출력 경로를 `os.path.abspath`로 만들어, 상대 출력 경로가 `third_party/ghoddoosian/` 안으로 해석된다. [확실] (코드 읽기). 헤드리스 러너는 출력 경로를 절대경로로 만드는 `resolve_output_path`를 `prepare_runtime`(`os.chdir`)보다 먼저 호출해 해결했다. [확실] (코드 읽기). 기존 래퍼는 수정하지 않았다 | `src/repro/reproduction_blink_one_video.py:19`(`chdir`), `src/repro/reproduction_blink_one_video.py:39`(`abspath`), `src/repro/reproduction_blink_headless.py:74`, `src/repro/reproduction_blink_headless.py:189`, `src/repro/reproduction_blink_headless.py:236`, `src/repro/reproduction_blink_headless.py:243` | [확실] | 기존 래퍼에 상대 출력 경로를 주면 추적 파일 영역에 쓸 수 있다. [확실] (코드 읽기) | 헤드리스 러너에서 해결됨. 기존 래퍼는 미수정(상대 출력 경로를 쓰지 않는다) |
| P-17 | 우리 도구 | 재학습(시드 3개, 5-fold 평균)의 VA가 공개 체크포인트보다 낮다: 0.556 ± 0.036 대 0.649(논문 0.652), 표기값 차이 0.093. [확실] (측정, 표기값). BSRE 2.874 대 1.901, VRE 1.953 대 1.158도 같은 방향이다. [확실] (측정, 표기값). 원인은 알려지지 않았다 | `README.md:48-54`, `experiments/02_fold_seeds/aggregate_seeds123.md:14`, `docs/IMPLEMENTATION_DIFF.md:272` | 수치 [확실], 원인 미상 | 재현이 완전하지 않다. [확실] | 열린 문제. 원인 미검증 |
| P-18 | 우리 도구(조사 방법) | dlib 정면 얼굴 검출기가 틀린 방향에서도 얼굴을 검출한다. 예: Fold2_part1/17/0.mp4는 0°에서 8프레임, 180°에서도 6프레임 검출. Fold1_part2/10/0.MOV는 cw90에서 8프레임, ccw90에서도 4프레임 검출. 같은 양상의 v1 `ambiguous` 18개(0°=8, 180°=5~8). [확실] (측정). 그래서 방향 판정은 단순 최댓값(v1)이 아니라 규칙 v2 + 육안 확인이 필요하다. [확실] (측정). 규칙 v2 자체는 v1 결과를 본 뒤에 만든 탐색적 규칙이다. [확실] | `configs/video_orientation_survey.csv`, `src/repro/survey_orientation.py:157`(v1 `choose_best`), `src/repro/survey_orientation.py:173`(v2 `resolve_orientation`), `docs/REPRODUCTION_LOG.md:590-632` | [확실] (측정) | 최댓값 규칙으로 방향을 정하면 틀린 방향으로 blink를 추출할 수 있다. [가능성 높음] (추출을 실행하지 않음) | 규칙 v2와 review 영상의 육안 확인(`configs/video_orientation_overrides.csv`)으로 대응. 규칙 v2의 신뢰도는 미검증 |
| P-19 | 저자 코드(검출 설정) | 얼굴은 육안으로 보이나 dlib 정면 검출기가 검출하지 못하는 영상이 있다. Fold2_part2/23의 0·5·10 3개는 사용자가 방향 `0`으로 판정하고 note에 "얼굴은 보이나 검출기 미검출"을 남겼고, 조사 카운트는 0, 1, 0이다. Fold5_part1/49/0.mp4는 사용자가 육안으로 얼굴이 보인다고 한 영상으로 육안 방향은 `ccw90`이며 조사 카운트는 네 방향 모두 0이다. [확실] (CSV, 사용자 보고). 이 영상들의 공통점으로 "고개를 든 자세일 수 있다"는 가설이 있다. [추측]. 원인 미검증. (2026-10-03 갱신) 배치 추출(174개) 결과 `face_frames/frame_count`가 0.5 미만인 영상은 5개이다: Fold2_part2/23/5.mp4 0.1147, 23/0.mp4 0.0019(no_output), 23/10.mp4 0.0062, Fold5_part2/58/0.mp4 0.4184, Fold5_part1/49/0.mp4 0.0316. 5개 모두 육안 판정한 review 14개에 속한다. [확실] (측정). 사용자의 육안 인상(정량 확인 아님, 영상 ID 미기록): 피험자 23은 "육안으로는 문제가 없어 보이는데 눈·입 윤곽을 거의 잡지 못함". 미검증. 사용자 방침(2026-10-03)에 따라 기록만 하고 검출기 설정과 파이프라인은 바꾸지 않는다. [확실] | `configs/video_orientation_overrides.csv`, `configs/video_orientation_survey.csv`, `third_party/ghoddoosian/blink_video.py:378`(검출기 호출), `docs/REPRODUCTION_LOG.md:677-696`, `docs/REPRODUCTION_LOG.md:734-784` | 육안 판정 [확실] (사용자 육안), 원인 미검증 | 저자 설정 그대로 처리하면 blink가 0 또는 거의 0일 수 있다. [가능성 높음] (이 영상들은 아직 추출하지 않았고, 미검출 프레임에서 광류 추적은 직전 얼굴이 있어야 시작되는 코드 구조에서 도출. `blink_video.py:491-497`) | 설정은 바꾸지 않는다. 원인은 미검증이다 |
| P-20 | 우리 도구(배치 실행 관측) | 서로 다른 영상의 소요 시간이 0.1초 차이로 묶인다(사용자 보고: 1920.5/1920.5/1920.6, 3748.7/3748.7/3748.8, 1188.0/1188.1, 1865.1/1865.1). 매니페스트에서 같은 시각에 시작해 같은 시각에 끝난 묶음 4개를 확인했다(예: Fold1_part1/04/0.mp4, Fold1_part1/04/10.mp4, Fold5_part1/49/10_1.mp4가 19:45:57에 시작해 20:17:58에 종료). 같은 묶음의 영상은 frame_count와 파일 크기가 달라 같은 파일이 아니다. 시각이 다른데 소요 값만 비슷한 경우(Fold1_part1/05/10.MOV, Fold1_part2/09/10.mp4와 Fold5_part1/52/10.MOV)도 있어 일부는 우연일 수 있다. [확실] (측정). 같은 시각에 끝난 묶음의 원인은 미검증이다. 여러 프로세스에 공통인 요인이 끝나는 시각을 정했을 가능성이 있다. [추측] | `docs/REPRODUCTION_LOG.md:738-755`(1절), 측정(매니페스트의 `started`, `ended`, `elapsed_s` 열, 저장소 밖) | 측정 [확실], 원인 미검증 | 소요 시간과 fps를 영상별 처리 속도로 해석할 수 없다. [가능성 높음]. 합산 처리량(전체 프레임 / 경과)은 영향이 적다. [가능성 높음] | 기록만 함. 원인 조사는 보류(방침, 8.3절) |
| P-21 | 저자 코드(blink 추출 결과) | blink 행이 30 미만인 영상이 20개이다(alert 11, semisleepy 5, sleepy 4. blink 0행은 Fold2_part2/23/0.mp4 하나). [확실] (측정). 원인은 규명하지 않았다. 두 가설을 병기한다. (가설 1) 실제로 눈을 덜 깜빡이는 영상이다. 미검증. (가설 2) blink 후보가 SVM·진폭 기준으로 거부되거나 얼굴 검출이 부족해 기록이 적다. 미검증. 두 가설은 구분되지 않았다(사람이 직접 센 횟수와 콘솔의 승인·거부 로그를 비교하지 않았다). 20개 중 Fold2_part2/23/0, 23/5, 23/10과 Fold5_part1/49/0은 얼굴 검출 비율도 0.5 미만이다(P-19). [확실] (측정) | `docs/REPRODUCTION_LOG.md:738-755`(1절), `docs/REPRODUCTION_LOG.md:769-773`(4절), `third_party/ghoddoosian/blink_video.py:220-221`, `third_party/ghoddoosian/blink_video.py:232`, `third_party/ghoddoosian/blink_video.py:236`, `third_party/ghoddoosian/blink_video.py:239` | 측정 [확실], 원인 미검증 | blink가 30개 이하인 영상은 `Preprocessing.py`가 시퀀스 1개(앞쪽 0 패딩)로 만든다(`third_party/ghoddoosian/Preprocessing.py:27-30`). [확실] (코드 읽기). alert 영상은 앞 2/3만 쓰므로(n_eff = n − n//3, `third_party/ghoddoosian/Preprocessing.py:56-57`) blink 45행 이하이면 n_eff가 30 이하이다. [확실] (산술) | 기록만 함. 검출기 설정과 파이프라인은 바꾸지 않는다(방침) |
| P-22 | 우리 도구(배치 매니페스트) | 매니페스트에서 `face_frames`가 비어 있는 영상이 2개이다: Fold4_part1/40/10.mp4, Fold1_part2/12/5.mp4. 둘 다 status ok, 종료 코드 0, blink 218행과 109행이다. [확실] (측정). 도구는 자식 출력에서 `not grabbed` 다음 줄의 숫자를 읽지 못하면 이 칸을 비운다. [확실] (코드 읽기). 읽지 못한 이유는 조사하지 않았다. 미검증 | `src/repro/extract_blinks_batch.py:294-305`(`parse_face_frames`), `third_party/ghoddoosian/blink_video.py:353-356`, `docs/REPRODUCTION_LOG.md:738-755`(1절) | 측정 [확실], 원인 미검증 | face_frames 집계(95.6%)에서 이 2개가 빠진다(frame_count 합 36,867). [확실] (계산) | 미조사. 보류(8.3절) |
| P-23 | 우리 도구·데이터 구성 | 재추출 데이터의 피험자·영상 구성이 저자 데이터와 다를 수 있다. 우리는 디스크의 영상 182개 중 manual_flags 8개를 뺀 174개를 추출했고 저자 코드의 테스트 영상은 174개(33/36/36/36/33)이다(P-15). 개수가 같아도 같은 집합이라는 뜻은 아니다. 저자가 어떤 영상을 제외했는지는 확인된 바 없다. 미확인 [확실]. 검출이 잘 안 된 영상(피험자 23 포함)은 사용자 방침(2026-10-03)에 따라 기록만 하고 검출기 설정과 파이프라인은 바꾸지 않는다. 이런 영상을 포함한 결과와 뺀 결과를 나란히 보고(B0/B1 병행 보고)하는 것을 권고로 표시한다. B0/B1의 정의는 이 저장소 문서에 없다(미정의). 이후 단계에서 영상이 제외되더라도 기준 영상이 없다는 기계적 결과이며 저자의 제외에 대한 주장이 아니다. [확실] | `docs/REPRODUCTION_LOG.md:762-768`(3절), `third_party/ghoddoosian/Training.py:445-446`, `third_party/ghoddoosian/Training.py:456-457`, `third_party/ghoddoosian/Training.py:465-466`, `third_party/ghoddoosian/Training.py:477-478`, `third_party/ghoddoosian/Training.py:489-490`, 본 절의 P-15 | 저자 제외 여부 미확인 | 비교 대상이 달라지면 저자 값과의 직접 비교가 어려워진다. [가능성 높음] | 권고(병행 보고)이며 이번 단계에서는 하지 않음 |

### 8.1 --show 사용 메모 (2026-10-03)

- 용도: `reproduction_blink_headless.py`의 `--show`는 `cv2.imshow`(`third_party/ghoddoosian/blink_video.py:476`, `third_party/ghoddoosian/blink_video.py:558`)와 `cv2.waitKey`(`third_party/ghoddoosian/blink_video.py:480`, `third_party/ghoddoosian/blink_video.py:562`) 치환을 하지 않아 저자 코드가 그리는 화면 프레임을 창으로 본다. 기본(`--show` 없음)은 이전과 완전히 같다. `plot_frame.draw()` 치환(474, 556줄)은 유지하므로 Tk의 EAR 그래프 창은 갱신되지 않는다. [확실] (코드 읽기, 측정)
- 사용 조건: 창이 열리는 로컬 환경 전용이다. 처음 1~2분만 보고 Ctrl+C로 종료한다. 출력 파일이 이미 있으면 이어 쓰지 않고 거부한다(`--show`에서도 확인). `--rotate`와 함께 쓸 수 있다. [확실]
- 점검 (a) 화면에 그려지는 것: 입 윤곽(파란색, `third_party/ghoddoosian/blink_video.py:396`)과 양쪽 눈 윤곽(초록색, `third_party/ghoddoosian/blink_video.py:429`, `third_party/ghoddoosian/blink_video.py:430`)뿐이다. 얼굴을 검출하지 못해 광류 추적으로 이어진 프레임에는 눈 윤곽만 그려지고(`third_party/ghoddoosian/blink_video.py:511`, `third_party/ghoddoosian/blink_video.py:512`) 입 윤곽은 없다. 랜드마크 점, EAR 값 텍스트, 사각형을 그리는 호출은 파일 전체에서 찾지 못했다(`putText`, `rectangle`, `circle`, `line`, `polylines` 검색 0건). [확실] (코드 읽기)
- 점검 (a) 표시되는 프레임: `frameMinus7=Q.get()`(`third_party/ghoddoosian/blink_video.py:475`, `third_party/ghoddoosian/blink_video.py:557`)으로 큐에서 꺼낸 이전 프레임이다. 큐에 넣는 `Q.put(frame)`(`third_party/ghoddoosian/blink_video.py:371`)은 같은 배열을 참조하므로 윤곽은 그 프레임을 처리할 때 이미 그려져 있다. 창은 `Q.full()`이고 `reference_frame>15`일 때부터 갱신된다(`third_party/ghoddoosian/blink_video.py:442`, `third_party/ghoddoosian/blink_video.py:528`). [확실] (코드 읽기). 큐가 가득 찬 뒤에는 창이 현재보다 약 6프레임 이전 프레임을 보여 준다. [가능성 높음] (큐 크기 7에서 도출, 실행하지 않음)
- 점검 (b) `face_frames`의 의미: 매니페스트의 `face_frames`는 원본의 `number_of_frames`이다. 얼굴 검출기가 얼굴을 한 개 이상 낸 프레임에서만 1씩 더한다(`third_party/ghoddoosian/blink_video.py:379`, `third_party/ghoddoosian/blink_video.py:380`). 검출 실패 후 광류 추적으로 이어진 프레임(`third_party/ghoddoosian/blink_video.py:486-566`)은 세지 않고, 첫 프레임부터 센다. 영상 끝에서만 `not grabbed` 다음 줄로 출력된다(`third_party/ghoddoosian/blink_video.py:353-356`). [확실] (코드 읽기). 이 값은 frequency의 분모에도 쓰인다(`third_party/ghoddoosian/blink_video.py:455`, `third_party/ghoddoosian/blink_video.py:541`, P-08 참조). [확실]
- 점검 (c) 종료 경로: `key = cv2.waitKey(1) & 0xFF` 뒤 `if key != 0xFF: break`(`third_party/ghoddoosian/blink_video.py:480-483`, 미검출 분기는 `third_party/ghoddoosian/blink_video.py:562-566`)로 **아무 키**나 누르면 루프가 끝난다. 주석(482줄)은 `q`라고 하지만 코드는 키를 가리지 않는다. 이 경로로 끝나면 `not grabbed`와 `face_frames`는 출력되지 않고, 이어서 `stream.release()`와 `cv2.destroyAllWindows()`가 실행된다(`third_party/ghoddoosian/blink_video.py:569`, `third_party/ghoddoosian/blink_video.py:570`). 기본 무화면 실행은 waitKey를 255로 치환하므로 이 경로가 없고, `--show`에서는 살아 있다. [확실] (코드 읽기)
- 제약: 이 창은 저자 코드가 그리는 것만 보여 준다. 실제 창 표시와 키 입력 동작은 아직 시험하지 못했다. 미검증 `docs/REPRODUCTION_LOG.md:711-732`에 사전 등록과 결과가 있다. [확실]

### 8.2 저자 blink 추출 로직 요약 (줄 번호, 2026-10-03 검증)

줄 번호는 모두 `third_party/ghoddoosian/blink_video.py` 기준이며 이 세션에서 파일을 직접 읽어 확인했다. 사용자가 준 요약에서 줄 번호를 확인하다가 표현을 고친 항목은 "(정정)"으로 표시했다. 랜드마크 번호 범위는 `imutils` 0.5.4의 값을 실행해 확인했다.

1. 전처리: 폭 450으로 resize(359), 회색조(368), 감마 1.5(370, `adjust_gamma` 정의는 29~37). 90° 회전 코드는 주석 처리돼 있다(361~366). [확실]
2. 검출: `detector(gray, 0)`(378). 얼굴이 하나 이상이면(379) `number_of_frames`에 1을 더하고(380) 첫 얼굴만 쓴다(`predictor(gray, rects[0])`, 386). 추적으로 이어진 프레임은 세지 않는다(486~566줄에 증가가 없다). [확실]
3. MAR (정정): `Mouth = shape[mStart:mEnd]`(391)이고 `(mStart, mEnd)`는 `face_utils.FACIAL_LANDMARKS_IDXS["mouth"]`(320)로 (48, 68)이다. 세로는 `mouth[14]`와 `mouth[18]`(96줄, shape 62와 66), 가로는 `mouth[12]`와 `mouth[16]`(98줄, shape 60과 64)의 거리이다. 가로가 0.1 미만이면 MAR을 0.2로 두고(100~101), 아니면 세로/가로이다(104). 임계값은 0.35(52), 0.30(53), 20(54)이다. 카운터는 MAR가 0.35를 넘는 프레임마다 MCOUNTER를 1 올리고(398~399), MAR가 0.30 미만이 되면 MCOUNTER가 20 이상일 때 MTOTAL을 1 올린 뒤 MCOUNTER를 0으로 되돌린다(401~406). 0.30~0.35 사이에서는 카운터가 유지되므로 "연속 20프레임"이 아니라 임계 초과 프레임의 누적이다. MAR, MTOTAL, MCOUNTER는 출력 5열과 blink 판정에 쓰이지 않는다(사용처가 302, 304, 392, 398~406줄뿐이다). [확실]
4. EAR: `A = |eye[1]-eye[5]|`(77), `B = |eye[2]-eye[4]|`(78), `C = |eye[0]-eye[3]|`(82)이고 `C < 0.1`이면 EAR을 0.3으로 두며(84~85), 아니면 `(A+B)/(2.0*C)`(88), 0.45를 넘으면 0.45로 자른다(89~90). 양쪽 눈의 평균을 쓴다(417~421, 추적 분기 503~506). 눈 랜드마크는 오른쪽 (36, 42), 왼쪽 (42, 48)이다(318~319). [확실]
5. EAR 시리즈와 SVM (정정): `EAR_series`는 길이 13(330)이고 매 프레임 한 칸씩 밀고 새 값을 맨 끝에 넣는다(423, 추적 분기 507). 프레임 큐는 `Queue(maxsize=7)`(48)이다. `IF_Closed_Eyes = loaded_svm.predict(EAR_series.reshape(1,-1))`(444, 530)는 13개 값 전체를 입력으로 받고, 그 결과가 `Blink_Tracker(EAR_series[6], IF_Closed_Eyes, ...)`(447, 533)에서 가운데(6번째) 프레임의 감김 여부로 쓰인다. 저자 주석이 가운데 프레임을 기준 프레임이라고 적는다(199~200, 442). 호출 조건은 `Q.full() and reference_frame>15`(442, 528)이다. [확실] (코드 읽기). SVM이 학습한 내용은 이 코드로는 알 수 없다. 미검증
6. `Blink_Tracker`(190~291) (정정: 병합 조건 표현): 감김(`int(IF_Closed_Eyes)==1`, 193)일 때 EAR을 후보에 쌓는다(194~204). 눈이 다시 뜨였을 때 감김이 2프레임 이상(`Counter4blinks>1`, 244)이면 후보로 삼아 끝, 진폭, 지속 시간을 계산한다(245~248). 병합(250~267): 직전 지속 시간이 15를 넘으면 margin을 8, 아니면 1로 두고(250~253) `Current.start − Last.end ≤ margin + 1`(254)이면 합친다. 두 후보 사이의 프레임 수(256)가 margin 이하인 경우이다. 승인(213~233): `Counter4blinks<2 and skip==False`(213)이고 `(reference_frame−6) − Last_Blink.end > margin`(218)일 때 검사한다. 조건은 `peakEAR < startEAR and peakEAR < endEAR and amplitude > MIN_AMPLITUDE(0.04, 51) and start < peak`(220)와 균형 `(start−peak) > 0.25·(end−peak) and 0.25·(start−peak) < (end−peak)`(221, 한쪽 낙폭이 반대쪽의 4배 이내)이다. 통과하면 3탭 평활 `convolve1d([1/3,1/3,1/3], mode='nearest')`(225), `Ultimate_Blink_Check`(227, 정의 128~186), 누적 blink 수 가산(229), 반환(233) 순이다. 거부하면 `skip=True`로 두고 메시지를 출력한다(235~236, 238~240). [확실]
7. 출력 한 줄은 `%.4f` 5열 [누적 blink 수, 100 × 누적 blink 수 / `number_of_frames`, 진폭, 지속 시간, 속도]이다(얼굴 분기 455~464, 추적 분기 541~548). 분모는 검출기가 얼굴을 낸 프레임 수이다. [확실]
8. 추적 분기(486~515): 첫 얼굴 검출 이후에만(491) 두 눈 6점씩 12점을 Lucas-Kanade로 추적하고(494~495, 파라미터 327~329), 12점이 모두 성공할 때만(497) EAR을 갱신한다(503~507). 입은 추적하지 않는다. [확실]
9. 비대칭: 추적 분기는 `Blink_Tracker` 호출이 `if Counter4blinks==0` 안에 있고(531~536) 얼굴 분기는 그 바깥에 있다(445~450). `velocity>0` 필터는 얼굴 분기에만 있다(461, 추적 분기 546~548에는 없다). 효과의 크기는 미검증이다. [확실] (코드 읽기)
10. 콘솔 메시지 (정정과 보강): "MISSED BLINKS= n"(232)의 n은 승인된 후보에서 추출한 blink 개수(`len(retrieved_blinks)`)이지 놓친 blink가 아니다. n은 0일 수 있다(추출된 blink가 없으면 165~168줄이 배열을 출력하고 빈 목록을 돌려준다). `Ultimate_Blink_Check`의 `MISSED_BLINKS`는 후보 하나에서 blink가 둘 이상 나오면 True이고(163~164) 출력에 쓰이지 않는다(227줄에서 받기만 한다). 후보 거부 메시지는 "rejected due to imbalance"(236)와 "rejected due to noise"(239)이다. [확실]
11. 화면 (정정): 입 윤곽(파란색, 396)과 눈 윤곽(초록색, 429~430, 추적 분기 511~512)만 그린다. 점이나 EAR 텍스트는 없다. 보이는 프레임은 `frameMinus7=Q.get()`(475, 557)으로 꺼낸 큐의 가장 오래된 프레임이고 `imshow`가 바로 뒤에 있다(476, 558). 큐 크기가 7이므로 현재보다 약 6프레임 전이다(변수 이름은 frameMinus7). [확실] (코드 읽기). 약 6프레임이라는 값은 큐 크기에서 도출했다. [가능성 높음]

### 8.3 보류 목록 (방침상 이번 단계에서 하지 않음)

사용자 방침(2026-10-03, `docs/REPRODUCTION_LOG.md:762-768`)에 따라 아래는 이번 단계에서 하지 않고 기록만 한다. [확실]

- 요약 스크립트(`summarize_batch.py`)의 Fold2_part1/17/0 보강: 이전 시험 산출물이라 매니페스트에 행이 없고 출력 txt는 있다. 현재는 이 영상을 "미처리 1개"로 세어 남은 시간 4~6분을 표시한다.
- 수집 루프 점검: 사용자 지시의 항목이며 대상 코드는 지시에 명시되지 않았다. 소요 시간 묶음(P-20)과 관련된 것으로 추정한다. [추측]
- `face_frames`가 비어 있는 2건 점검(P-22).
- 검출기 진단과 화면 오버레이: `--show` 위에 EAR 값이나 랜드마크를 그리는 일(8.1절). 구현하지 않았다.
- 피험자 23 영상의 얼굴 크기 가설 확인: 얼굴 크기가 검출 실패와 관련된다는 가설을 뜻하는 것으로 이해한다. 이 저장소에는 근거가 되는 측정이 없다. [추측]. 미검증


## 9. 2단계(시퀀스 생성) 구현 메모 (2026-10-03)

`src/repro/reproduction_build_sequences.py`는 재추출한 blink txt(`<BLINK_OUT_DIR>/Fold<N>_part<M>/<피험자>/<클래스>[_<변형>].txt`)에서 저자 `Preprocessing.py`(PP)와 같은 계산으로 시퀀스 .npy와 메타데이터를 만든다. 입력은 `BLINK_OUT_DIR`(읽기 전용), 출력은 `SEQ_OUT_DIR`(저장소 밖, 기본값 없음)에만 쓴다.

### 9.1 저자 PP와 같게 동작하는 부분
- txt의 1~4열(빈도×100, 진폭, 지속 시간, 속도)을 쓴다. 0열(누적 blink 수)은 쓰지 않는다(PP `usecols=1..4`). [확실]
- 피험자별 정규화: alert의 마지막 `n//3`개 blink로 평균과 표준편차(ddof=0)를 구한다. `n < 3`이면 PP의 `[-0:]`처럼 전체를 쓴다. 기준 구간 표준편차가 0이면 전체 표준편차로 대체한다. alert 데이터는 앞쪽 `n − n//3`개, semisleepy·sleepy는 전부이며 같은 통계로 정규화한다. [확실]
- `unroll_in_time`: 길이 30, 폭 2. n ≤ 30이면 앞쪽을 0으로 채운 시퀀스 1개, 그보다 크면 `((n−30)//2)+1`개. [확실]
- 순서: 피험자마다 alert, semisleepy, sleepy를 이어 붙인다. 테스트 세트는 섞지 않는다. 라벨은 `label*np.ones([N,1])`(float64). [확실]
- 파일 이름과 형상: `Blinks_30_Fold<k>.npy`, `Labels_30_Fold<k>.npy`, `BlinksTest_30_Fold<k>.npy`, `LabelsTest_30_Fold<k>.npy`. X는 (N,30,4) float64, y는 (N,1). 테스트는 그 fold의 피험자 전체, 훈련은 나머지 fold의 피험자. [확실]
- 합성 데이터로 저자 함수와 비교해 `np.array_equal`로 같음을 확인했다(REPRODUCTION_LOG.md의 "시퀀스 생성 구현 결과 (2026-10-03)"). [확실]

### 9.2 의도적으로 다른 부분
1. 제외 정책 5가지. 영상마다 사유를 기록하고 앞 피험자의 값으로 대체하지 않는다. 우선순위는 `variant_deferred`, `not_extracted`, `no_blink_file`, `no_alert_reference`, `degenerate_reference` 순이다. 피험자 단위 사유(`no_alert_reference`, `degenerate_reference`)는 그 피험자의 나머지 영상도 제외한다. 이 목록은 재추출 데이터에서의 기계적 결과이며 저자의 제외와 관련지을 근거가 없다. [확실]
2. `degenerate_reference` (2026-10-03 변경): 피험자 제외는 저자 코드가 예외를 내거나 비유한 값을 만드는 경우로 한정한다. 해당하는 경우는 alert 1행(저자 TypeError)과 전체 열 표준편차로 대체한 뒤에도 0 또는 비유한인 경우(저자 NaN)뿐이다. alert n = 2(전체 배열 통계), n = 3~5(마지막 1개 평균과 전체 열 표준편차 대체), 기준 구간 표준편차 0의 대체는 저자 동작을 그대로 따르고, 사용한 대체 종류를 `video_summary.csv`의 `author_fallback_used`(`none` / `whole_array` / `whole_std`)에 기록한다. 이전 규칙(기준 구간 blink 수 2 미만이면 제외)은 저자가 하지 않은 제외를 만들어 취소했다. 합성 시험에서 alert n = 2~5의 결과가 저자 `Preprocess`와 `np.array_equal`로 같음을 확인했다. [확실]
3. 메타데이터 출력: `sequence_metadata_fold<k>_<train|test>.csv`, `video_summary.csv`, `exclusions.csv`, `coverage.txt`. 저자 PP에는 없다. [확실]
4. `start_indices_fold<k>.json`을 메타데이터에서 계산한다. 시작이 0이고 증가하며 마지막 영상이 N까지 이어지고, 합이 테스트 시퀀스 수와 같다는 것을 확인한다. `Training.py`의 하드코딩 목록과 같은 형식이다. [확실]
5. 변형 파일(`10_1`, `10_2` 등)은 이번 단계에서 보류한다(`variant_deferred`). 저자가 변형 파일을 어떻게 다뤘는지 확인된 바 없다. 확인되지 않았다.
6. 훈련 세트는 fold 번호를 시드로 고정해 섞는다(`np.random.RandomState(fold).permutation`, PP는 시드 없는 `np.random.permutation`). `Training.py`의 `batch_gen`은 섞지 않으므로 .npy가 섞여 있어야 한다는 판단이다(그 파일 288줄 주석). 같은 시드를 쓰면 저자 `Preprocess`와 같은 순열이다. [가능성 높음]
7. 기존 출력 파일은 덮어쓰지 않고 중단한다(`--force` 없음). `--check-only`는 아무것도 쓰지 않는다.

### 9.3 확인하지 못한 점
- 실제 데이터에서 저자 결과와 같은지. 저자의 .npy는 이 저장소의 원본과 우리 재추출이 달라 직접 비교하지 않았다. 미검증
- 저자 PP가 실제로 사용한 피험자 순서와 훈련 세트 섞기 여부. 미검증
- 우리 테스트 영상 수(2026-10-03 정책 변경 뒤 33, 33, 35, 32, 34)와 `Training.py`의 start_indices 길이(33, 36, 36, 36, 33)가 다른 이유. 미검증
- `ref_small` 피험자 9명의 정규화 안정성. 미검증
- 위의 합성 비교는 하네스에서 저자 함수 정의를 읽어 메모리에서 정의해 수행했다. 이 방식은 구현 파일에는 쓰지 않았다.

### 9.4 확인한 항목 (2026-10-03)
- (i) 훈련 셔플과 시드: 저자 `Preprocessing.py`의 셔플은 `np.random.permutation`이고(5줄, 호출은 255줄) 이 폴더의 `.py` 어디에도 numpy 시드를 설정하는 줄이 없다(Grep에서 `seed`는 `Training.py`의 TensorFlow 초기화 `seed=None`뿐). 따라서 저자의 훈련 순서는 실행마다 달랐을 것이다. [확실] (코드 읽기). 우리 구현은 fold 번호를 시드로 고정한다(`np.random.RandomState(fold).permutation`). 이는 재현성(두 번 실행해 같은 .npy)을 위한 의도한 차이다. 저자 순서를 복원하는 방법은 없다. 훈련 순서가 학습 결과에 미치는 영향은 미검증이다. 훈련 세트의 평균과 표준편차(`Training.py` 2차 정규화)는 순서와 무관하다. [가능성 높음]
- (ii) "usable subjects" 정의: `coverage.txt`와 출력의 "사용 가능한 피험자"는 상태가 `ok`인 피험자, 곧 alert 영상이 포함되었고(txt가 있고 blink 1행 이상이며 변형이 아님) 정규화 기준이 `degenerate_reference`가 아닌 피험자다. 정책 변경 전 56명, 변경 뒤 57명이다. 제외되는 피험자는 06, 45(alert가 `not_extracted`), 23(alert txt 없음)이다. 인벤토리 전체 피험자가 60명이라는 값은 57 + 3에서 따진 것이고 직접 세어 확인하지는 않았다. [가능성 높음]
- (iii) 20개 npy와 저자 main 블록: 저자 main 블록은 `test_fold='Fold3'`를 하드코딩하고(268) 테스트 배열만 저장한다(271~272, `BlinksTest_30_Fold3.npy`, `LabelsTest_30_Fold3.npy`). 훈련 저장(269~270)은 주석이다. 즉 배포된 `Preprocessing.py`는 20개 중 2개만 만든다. 20개라는 구성은 `Training.py`가 `range(5)` 반복에서 fold마다 4개 파일(`Blinks_30_Fold%d`, `Labels_30_Fold%d`, `BlinksTest_30_Fold%d`, `LabelsTest_30_Fold%d`)을 읽는 것에서 온다(408, 414~417). 우리 구현은 이 20개를 모두 만든다. 저자가 나머지 파일을 어떻게 만들었는지(fold 값을 바꿔 반복 실행하고 주석을 풀었는지 등)는 확인하지 못했다. [확실] (코드 읽기). 미검증
- 같은 동작과 다른 동작의 구분은 9.1과 9.2를 따른다. 9.2의 2번이 이번 변경으로 바뀌었다.


## 10. 저자 .npy 대비 비교 결과 (2026-10-03)

비교 도구는 `src/repro/compare_with_author_npy.py`이다(읽기 전용, 표와 CSV는 저장소 밖 `SEQ_OUT_DIR\compare\`). 자세한 사전 등록과 결과는 `docs/REPRODUCTION_LOG.md`의 "저자 .npy 대비 비교 사전 등록·결과 (2026-10-03)"이다. 아래는 요약 수치와 갱신이다.

### 10.1 요약
- 우리 산출물 정합성(작업 0): shape, dtype, 메타데이터 행 수, start_indices가 5개 fold 모두 일치했다. [확실]
- 저자 테스트 배열 구조(작업 1): fold별 영상 수 33, 36, 36, 36, 33, 모두 (0,5,10) 반복이고 경계가 `Training.py`의 start_indices와 일치한다. [확실]
- 위치 대응 비교(작업 2): 비교 161개 중 시퀀스 수가 같은 것 29개(0.180, 그중 14개는 시퀀스 1개), allclose 3개(모두 최대 차이 0, 피험자 09의 세 영상). 회전별 count_equal 비율은 none 0.179, cw90 0.114, ccw90 0.290, 180은 1개(0). [확실]
- 이 비교는 "저자 피험자 순서가 폴더 이름 오름차순이다"라는 가정에 의존하며, 그 가정이 맞는지는 확인하지 못했다. 위치별 시퀀스 수의 관찰은 Fold1이 끝쪽(피험자 12 위치), Fold5가 49 쪽에서 모자란 모양이고 Fold3은 오름차순이 맞지 않을 가능성을 보인다([추측]). 미검증

### 10.2 P-14 갱신 (2026-10-03)
- 저자 테스트 배열 전체(5개 fold)에서 서로 다른 두 영상의 X 블록이 일치하는 쌍은 2개이고 둘 다 Fold3 안이다(0부터 센 위치 23과 26, 시퀀스 93개; 34와 35, 시퀀스 6개). 기존 P-14가 적은 쌍(1부터 센 24와 27, 35와 36)과 같다. 새 쌍과 fold 사이 쌍은 없었다. [확실]
- 우리 영상 ID로 옮기면(오름차순 가정) 첫 쌍은 32의 클래스 10과 33의 클래스 10, 둘째 쌍은 36의 클래스 5와 10이다. 대응이 가정에 의존하므로 영상 ID는 미검증이다. 첫 쌍은 인접 피험자 사이라 `Preprocessing.py`가 파일이 없는 피험자에게 이전 값을 쓰는 동작(P-04)과 일관되지만, 둘째 쌍은 같은 피험자의 클래스 5와 10 사이라 그 동작만으로는 설명되지 않는다. 원인은 미검증이다.

### 10.3 P-15 갱신 (2026-10-03)
- 저자 코드의 테스트 영상 174개(33/36/36/36/33)는 저자 .npy의 y 런으로 다시 확인했다(구조 [확실]). 우리 재추출은 정책 변경 뒤 포함 167개이다.
- 저자가 어떤 영상을 제외했는지는 여전히 확인되지 않았다. 영상 수로는 Fold2~4가 12명 완전 구성, Fold1과 5가 한 명씩 모자란 구성이다. 그 한 명이 누구인지는 위치 대응의 관찰만 있으며(Fold1은 끝의 피험자 12 쪽, Fold5는 49 쪽이 모자란 모양, [추측]) 확정되지 않았다. 미검증

### 10.4 결정 대기 (위치 대응 결과 기준, 10.6에서 정렬 없는 일치 결과를 반영해 갱신)
- 우리 수동 플래그 영상(06 전체, 45 전체, 42/5, 55/10): 우리는 `not_extracted`로 제외했고 저자 배열에는 대응하는 칸이 있다. 추출할지 결정 대기.
- 피험자 49: 저자 PP와 같은 계산으로 alert, semisleepy를 포함하고 있다(변형 파일 10_1, 10_2는 보류). 저자 쪽에서 빠진 것으로 보이는 관찰이 있다([추측]). 유지할지 결정 대기.
- 변형이 있는 피험자 32(Fold3), 49(Fold5)의 클래스 10 칸 처리(10_1, 10_2 중 무엇을 쓸지, 합칠지): 결정 대기.
- 피험자 순서 가정(폴더 이름 오름차순)을 바꿔 재비교할지: 결정 대기. 이번에는 바꾸지 않았다.

### 10.5 정렬 없는 일치 결과 (2026-10-03)
10.1~10.4의 위치 대응 결과는 "저자 피험자 순서가 폴더 이름 오름차순이다"라는 가정에 의존해 약하다. 아래는 그 가정 없이 저자 영상 블록 174개를 우리 포함 영상 167개 전체와 값으로 직접 비교한 결과이다(`compare_with_author_npy.py --match-free`, 판정 기준 E: `np.allclose(atol=1e-6)`와 y 일치, A: 4개 특징 열 각각 |r| ≥ 0.99999, 사전 등록한 대로 고정). 사전 등록과 결과 전문은 `docs/REPRODUCTION_LOG.md`의 "정렬 없는 일치 탐색 사전 등록·결과 (2026-10-03)"이다.
- 결과: E 3개, A 0개, 일치 없음 171개. E 3개는 모두 우리 Fold1_part2/09의 세 영상(저자 Fold1 위치 24~26, 최대 차이 0)이다. [확실]
- 이 세 영상은 같은 blink txt에서 같은 계산을 하면 저자 .npy와 값까지 같아진다는 근거이다(시퀀스 생성이 저자 PP와 같은 계산이라는 근거). 근거의 전부가 피험자 1명이다. [가능성 높음]
- 아핀 일치(정규화 통계만 다른 같은 blink)가 하나도 없으므로 나머지 불일치는 정규화가 아니라 원 blink 데이터의 차이로 보는 쪽이 자연스럽다. 원인은 미검증이다. [가능성 높음]
- 회전별(시퀀스 2개 이상 우리 영상) 일치율: none 79개 중 E 3, 회전 영상 62개 중 0. 일치한 영상이 한 피험자뿐이라 회전과의 관련은 판단할 수 없다. 미검증
- 위치 대응(10.1)과 구분: 위치 대응의 시퀀스 수 일치 29개(0.180)는 14개가 시퀀스 1개의 자명한 일치이고 값 일치는 3개뿐이었다. 정렬 없는 탐색은 같은 3개만 찾았으므로 위치 대응에서 값까지 일치한 것은 우연이 아니라 같은 영상이다. 나머지 시퀀스 수 일치는 값이 달랐다. [확실]
- 오름차순 가정과 비교: 값으로 확인된 칸은 Fold1 0부터 8번째 칸(피험자 09) 하나이며 오름차순 가정과 같다. 나머지 칸은 unknown이어서 가정의 맞고 틀림을 판정할 수 없다. 미검증
- Fold1은 저자 11명 중 09의 앞에 8명이 있으므로 오름차순이면 01~08(06 포함)이 있고 빠진 한 명은 10~12 중 하나이다([가능성 높음]). 12가 빠졌는지는 미검증이다. Fold5의 49는 저자에 있는지 알 수 없다(Fold5 블록이 모두 불일치). 미검증

### 10.6 P-14, P-15 갱신과 결정 대기 (2026-10-03)
- P-14 (확정된 부분만): 동일 X 블록 쌍 2개(Fold3, 0부터 센 위치 23과 26, 34와 35)는 X만으로도 일치하는 우리 영상이 없다. 어느 위치가 정상 슬롯이고 어느 쪽이 사본인지는 판정할 수 없다. 원인은 미검증이다. [확실] (일치 없음 자체)
- P-15 (확정된 부분만): 저자 블록 174개 중 우리 영상과 값이 일치한 것은 3개(피험자 09)이다. 저자가 어떤 영상을 제외했는지, 저자 영상 집합이 우리 집합과 어떻게 다른지는 값 일치로는 확정할 수 없다(일치 영상이 너무 적다). 미검증
- 결정 대기 1, 수동 플래그 영상(06, 45, 42/5, 55/10)의 추출 여부: 이번 결과로는 "추출하면 저자와 일치할 것"이라고도 "일치하지 않을 것"이라고도 말할 수 없다(우리가 추출한 영상 167개 중 164개가 일치하지 않는다). 판단 기준은 값 일치 여부가 아니라 평가 영상 집합의 범위(저자 코드의 174개 구성에 가깝게 할지)이다. 추출한다면 기존 방침대로 사유 없는 제외가 아니라 그 영상들의 결과를 따로 표시하는 병행 보고를 권한다. 결정 대기
- 결정 대기 2, 49 유지 여부: 현재는 저자 PP와 같은 계산(alert 3행, 전체 표준편차 대체)으로 alert와 semisleepy를 포함한다. 저자에 49가 있는지는 값으로 확인되지 않았다(미검증). 판단 기준은 "저자가 하지 않은 제외를 만들지 않는다"(이전 정책 결정)이며, 이 결과가 그 방침을 바꿀 근거는 없다. 유지가 기본이고 49를 뺀 결과를 병행 보고할지는 사용자의 결정을 기다린다. 결정 대기
- 결정 대기 3, 32와 49의 변형 파일 처리: 이번 결과로는 저자 Fold3의 32 자리(위치 23)가 33의 클래스 10 자리(위치 26)와 같은 블록이라는 것만 알 수 있고 둘 다 우리 영상과 일치하지 않는다. 변형 파일 중 어느 것을 쓸지에 대한 근거는 얻지 못했다. 현재는 보류(`variant_deferred`)를 유지한다. 판단 기준 없이 정할 수 없으므로 사용자의 결정을 기다린다. 결정 대기
- 결정 대기 4, 추출 단계 차이의 조사 여부: 나머지 171개가 값까지 다르므로, 같은 영상에서 저자와 우리의 blink 추출이 어디서 갈라지는지를 조사할지(예: 피험자 09가 일치하는 이유를 먼저 살피는 것) 결정 대기. 이 조사는 이번 단계에서 하지 않았다.

### 10.7 탐색적 근사 일치 결과 (2026-10-03)
- 성격: 이 소절은 **탐색적** 분석이다. 앞의 E/A 판정(10.5)은 사전 계획한 판정이고, 이번 풀링 통계와 단계적 일치는 기준(shift −3..3행, 겹침 30 이상, amplitude·duration·velocity의 피어슨 상관 모두 0.9 이상)을 결과를 보기 전에 고정했지만 사전 계획한 판정은 아니다. 사전 등록과 결과는 `docs/REPRODUCTION_LOG.md`의 "클래스별 풀링 통계와 단계적 일치(탐색적) 사전 등록·결과 (2026-10-03)"이다. 구현은 `compare_with_author_npy.py --graded`이다.
- E/A 판정의 한계: E(`allclose(atol=1e-6)`와 y 일치)는 사실상 비트 단위 일치이고 A는 네 열 모두 |r| ≥ 0.99999를 요구한다. 그래서 "같은 영상에서 amplitude는 같고 duration, velocity가 아주 조금 다른" 경우를 구분하지 못한다. 이번 탐색에서 우리 Fold3_part2/31의 세 영상이 그런 경우로 새로 나타났다(E/A로는 일치 없음으로 분류됨). 근사 일치는 상관이 높다는 사실일 뿐 같은 영상이라는 증명이 아니다. [확실]
- 풀링 통계(시퀀스 2개 이상 블록): 영상당 시퀀스 수 평균의 비(우리/저자)는 클래스 0이 0.921, 5가 0.989, 10이 0.998이고, 클래스 10과 0의 특징 평균 차이의 부호가 4개 특징에서 모두 같다. 크기는 freq, amp, vel에서 우리가 1.4~1.6배, dur에서 약 0.85배이다. 원인은 미검증이다. [확실] (수치)
- 근사 일치: 저자 블록(시퀀스 2개 이상) 153개 중 6개(0.039)가 우리 영상과 근사 일치한다. 09의 세 블록(정확 일치, 상관 1.0)과 Fold3의 31의 세 블록(상관 0.9991~1.0000, shift 0, 복원 길이 차이 최대 0.9%)이다. 근사 일치는 모두 회전 none 영상이다. 일치한 영상이 피험자 2명뿐이라 회전과의 관련은 판단할 수 없다. [확실] (수치), 해석은 미검증
- 해석의 한계: (1) 풀링 수준에서는 저자와 우리가 비슷하고 개별 영상 수준에서는 거의 일치하지 않는다. 두 사실은 모순되지 않지만 원인은 이 분석으로 알 수 없다. (2) 표본이 사실상 피험자 2명이므로 Q4, Q5의 적중은 일반화할 수 없다. (3) 오름차순 가정과의 비교에서 저자 Fold3은 31이 0부터 5번째 칸에 있어(가정에서는 6번째) 31 앞에 5명뿐이다. 빠진 피험자가 있는지, 순서가 오름차순이 아닌지는 확인하지 못했다. 미검증
- P-15 갱신(확정된 부분만): 저자 Fold3의 0부터 5번째 피험자 칸이 우리 31과 근사 일치한다(위 (3)). Fold1의 0부터 8번째 칸이 09와 정확 일치한다(10.5). 두 칸 외에는 값으로 확인된 대응이 없다. 저자가 어떤 영상을 제외했는지는 확인되지 않았다. 미검증
- 결정 대기(10.6의 4번 보강): 추출 단계 차이를 조사한다면 09(정확 일치)와 31(amplitude 일치, duration·velocity 근소한 차이)이 출발점이 될 수 있다. 조사 여부는 사용자의 결정을 기다린다. 이번 단계에서는 하지 않았다.


## 11. 기준선 학습 러너 메모(B1) (2026-10-03)

`src/repro/reproduction_train_baseline.py`는 우리 시퀀스 산출물로 저자 `Training.py`와 같은 학습·평가를 하는 러너이다. `Training.py`는 텍스트로 읽어 메모리에서만 치환해 exec한다(디스크의 `third_party`는 수정하지 않는다). 사전 등록과 결과는 `docs/REPRODUCTION_LOG.md`의 "기준선 학습 러너(B1) 구현 사전 등록·결과 (2026-10-03)"이다. 에이전트는 학습을 실행하지 않았다.

### 11.1 치환 항목 (줄 번호는 원본 `Training.py` 기준, 기대 개수)
| 치환 | 줄 | 개수 | 내용 |
|---|---|---|---|
| 시드 | 304 | 1 | `tf.reset_default_graph()` 다음 줄에 `tf.set_random_seed(<시드>)` 삽입(그래프를 새로 만든 뒤에 걸어야 적용된다) |
| 저장 경로 | 316 | 1 | `variable_path='./'` → 출력 폴더 |
| epoch 기록 | 390 | 1 | `y_v[idx]=...` 다음에 `_on_epoch_end(...)` 호출 삽입(계산에는 영향 없음) |
| fold 반복 | 408 | 1 | `for i in range(5):` → 지정 fold 하나 |
| 체크포인트 이름 | 411 | 1 | `ii=i` → `ii=i+1`(`my_model<fold>`) |
| 데이터 경로 | 414~417 | 각 1 | `np.load` 4곳을 `--data-dir`로 |
| start_indices | 502 | 1 | `start_indices=np.asarray(...)` 앞에서 우리 JSON 값으로 덮어쓰기(하드코딩 블록 444~498은 그대로 두고 덮어씀) |
| 저장 | 510~515 | 6 | `open('./...')`를 출력 폴더로 |
| epoch(스모크 전용) | 504 | 1 | `num_epochs=80` → 지정 값 |
치환마다 원본에서 나오는 개수를 검증하고 다르면 중단한다. 메모리 비교로 변경이 위 8개 구간뿐임을 확인했다(삭제 13줄, 추가 16줄). [확실]

### 11.2 저자 코드와 같은 동작
- 모델, 손실(Δ=1.253, L2 0.1), Adam(lr 0.000053), batch 64, 4×32, 2차 정규화(훈련 세트 평균·표준편차), 지표 함수(영상 다수결 VA와 동률일 때 평균 예측, VRE, BSRE, 임계값 3.34)는 원본 소스를 그대로 실행한다. `--smoke`가 없으면 epoch도 80 그대로이고, 하이퍼파라미터 문자열이 원본에서 저자 값 그대로인지 시작할 때 검사한다. 체크포인트는 원본처럼 학습이 끝난 뒤 한 번만 저장된다(396줄). 보고값은 마지막 epoch이다. [확실] (코드 읽기)
- 출력 형식은 기존 래퍼(`train_fold_seeds.py`)와 같은 `metrics.csv`(열 `epoch, train_loss_sum, train_moving_acc, BSA_loss, BSA_acc, VA, BSRE, VRE`)이다. 지표는 원본 함수의 반환값을 감싸 기록한다. [확실] (코드 읽기)

### 11.3 의도한 차이
- 데이터 경로와 start_indices를 우리 산출물로 바꾼다(시험에서 JSON의 합이 테스트 시퀀스 수와 같고 구간마다 라벨이 하나임을 확인). 저자의 start_indices 길이(33, 36, 36, 36, 33)와 우리 길이(33, 33, 35, 32, 34)는 다르다(영상 구성이 다르다, 10절).
- 출력은 `TRAIN_OUT_DIR`(저장소 밖, 기본값 없음; 저장소 안이거나 `my_model1~5` 위치이면 중단)에만 쓰고, 이미 있는 fold 폴더는 덮어쓰지 않고 중단한다.
- 난수 시드를 고정한다(원본에는 시드가 없다). 시드는 기존 래퍼와 같은 방식(`random`, `numpy`, `tf.set_random_seed`)이다.
- `--smoke`(fold 하나, `--epochs` 지정 가능, 출력 폴더에 `SMOKE` 표식 파일)는 시험 전용이며 결과 집계가 무시해야 한다(`is_smoke_dir` 함수 제공). 비스모크 실행에서는 `--epochs`를 거부한다.
- 출력 구성: 체크포인트, `x/loss/accuracy*.npy`, `metrics.csv`, `final_metrics.json`, `run_config.json`(데이터 파일 이름과 해시, 시드, 치환 줄 번호, 원본과 치환된 소스의 해시), `train.log`. `run_config.json`에는 경로를 쓰지 않는다.

### 11.4 미검증
- 실제 학습 실행(TF 1.8에서의 동작, 체크포인트 저장, 지표 기록, `--smoke` 실행, `SMOKE` 표식 파일)은 하지 않았다. 미검증
- 체크포인트 이름 `my_model<fold>`(`ii=i+1`)는 기존 래퍼와 같은 선택이다. 저자가 원본 `ii=i`로 만든 실제 이름은 확인하지 못했다. 미검증
- 시드 고정이 결과에 주는 영향, 우리 데이터(영상 구성이 저자와 다름)로 학습한 결과가 저자 결과와 가까울지는 알 수 없다. 미검증


## 12. B1 기준선 결과 메모 (2026-10-04)

집계 도구는 `src/repro/summarize_training.py`(읽기 전용)이고, 사전 등록과 결과는 `docs/REPRODUCTION_LOG.md`의 "B1 기준선 학습 결과 집계 사전 등록·결과 (2026-10-04)"이다. 표와 CSV는 학습 출력 루트 밖의 집계 폴더에 쓰고 저장소에는 요약만 둔다.

### 12.1 집계 기준 (결과를 보기 전에 고정)
- 주 지표는 마지막 epoch(80)의 영상 단위 VA, VRE이다. 최고 epoch 값(fold별 VA 최대, VRE 최소)은 테스트 세트로 고른 값이라 "낙관적 참고치"로만 따로 표시하고 주 결과로 쓰지 않는다.
- 표: 시드별 fold별 값, 시드별 5-fold 평균(fold 평균), 시드 3개의 평균과 표본 표준편차(n−1).
- SMOKE 표식이 있는 폴더는 건너뛰고 이름을 보고한다. 루트마다 SMOKE가 아닌 실행 폴더가 하나여야 한다.
- 논문, 공개 체크포인트, B0(저자 npy 재학습)의 값은 기존 문서에 적힌 값만 인용하고 없는 값은 not explicitly reported로 쓴다. B1(우리 데이터)과 B0(저자 npy)는 서로 다른 데이터셋이므로 재현 성공/실패로 판정하지 않고 차이만 적는다. [확실]

### 12.2 정상성 점검 항목 (결과: 모두 통과)
1. 모든 fold의 `metrics.csv`가 epoch 1~80의 80행. 2. NaN, inf 없음. 3. 3개 시드의 `run_config.json`에서 epoch 80, 하이퍼파라미터 문자열(batch 64, lr 0.000053, 4×32 등), 원본 `Training.py` 해시, 치환 줄 번호, 입력 해시, 시퀀스 수가 같고 시드만 다름(치환된 소스의 해시는 시드와 출력 경로가 문자열에 들어가므로 비교하지 않음). 4. 테스트 시퀀스 수가 우리 start_indices 합(1328, 2059, 1641, 1206, 1648)과 같음. 5. BSA loss가 epoch 1보다 epoch 80에서 낮음. 참고: `final_metrics.json`이 `metrics.csv` 마지막 행과 같음. [확실]

### 12.3 결과 요약 (5-fold 평균, 시드 3개)
- 마지막 epoch VA 0.586 ± 0.015, VRE 1.498 ± 0.282, BSA acc 0.517, BSRE 2.108. 시드별 VA 5-fold 평균은 0.600, 0.587, 0.570. fold별 시드 평균 VA는 0.525, 0.707, 0.495, 0.583, 0.618(Fold2는 시드 간 0.576~0.818로 흩어짐). [확실]
- 기존 문서의 값과 나란히(차이만 기록): VA는 B0 0.556 ± 0.036, 공개 체크포인트 0.6495, 논문 0.652. VRE는 B0 1.953 ± 0.546, 공개 1.158, 논문 1.14. B1의 테스트 영상 구성(33, 33, 35, 32, 34개)이 저자 데이터(33, 36, 36, 36, 33개)와 다르므로 fold별 값은 같은 영상 집합의 비교가 아니다(10절). [확실] (수치)
- 클래스별 정확도: 71번째 epoch(마지막 epoch 아님) 참고치로 클래스 0 0.731, 5 0.327, 10 0.698이다(시드 합산, 영상 501개). 클래스 5 정답 168개 중 78개가 10으로 예측되었다. [확실] (71번째 epoch 기준)

### 12.4 미검증 사항과 한계
- 마지막 epoch(80)의 영상별 예측, 혼동행렬, 클래스별 정확도는 산출 불가이다. `metrics.csv`에 영상별 예측이 없고, 원본 `vote_accuracy_per_video`는 epoch 번호가 10의 배수일 때만 영상별 예측을 출력하는데(`Training.py` 114~115줄) 마지막 epoch의 번호는 79이다. `train.log`의 71번째 epoch 값은 마지막 epoch가 아니다. 저장된 체크포인트(`my_model<fold>`)를 불러 마지막 epoch의 예측을 얻을 수는 있으나 TensorFlow 실행이 필요해 이번에는 하지 않았다. 미검증
- 시드가 3개뿐이어서 표준편차는 참고 수준이다. fold당 테스트 영상이 32~35개라 영상 하나가 fold VA를 약 0.03 움직인다. 시드 고정(원본은 시드가 없다)이 결과에 주는 영향은 알 수 없다. 미검증
- B1과 B0, 공개 체크포인트의 차이가 영상 구성, 추출 결과, 정규화 기준 중 무엇 때문인지는 이 집계로 알 수 없다(10절 비교 결과와 같은 한계). 미검증
- 낙관적 참고치(최고 epoch)는 80개 epoch 중 테스트 세트로 고른 값이므로 마지막 값보다 작을 수 없고 일반화 성능의 근거가 아니다. [확실]


## 13. 체크포인트 평가 메모 (2026-10-04)

`src/repro/evaluate_baseline_checkpoints.py`는 B1 기준선 학습의 fold별 체크포인트(`my_model<k>`)를 불러 테스트 영상 전체의 영상별 예측을 만든다. 체크포인트 로드와 예측만 하고 학습은 하지 않는다. 사전 등록과 결과는 `docs/REPRODUCTION_LOG.md`의 "B1 체크포인트 영상별 평가 사전 등록·결과 (2026-10-04)"이다.

### 13.1 평가 방식
- `reproduction_eval.py`(공개 체크포인트 평가)와 같은 방식이다: `Training.py`를 텍스트로 읽어 메모리에서만 치환하고, 원본 `Train()`이 `load=True`로 체크포인트를 복원해 테스트 세트를 한 번 평가한다. 치환(원본 줄 번호, 기대 개수 각 1): `load=False` → `load=True`(407), `variable_path='./'` → fold 체크포인트 폴더(316), `for i in range(5):` → 지정 fold 하나(408), `np.load` 4곳 → 데이터 폴더(414~417), `start_indices=np.asarray(...)` 앞에서 우리 JSON 값으로 덮어쓰기(502). `ii`는 `load==True`일 때 원본이 `i+1`로 바꾼다(412~413). [확실]
- 저장 호출은 `Training.py` 395~396줄로 학습 루프가 끝난 뒤 한 번 실행된다(마지막 epoch 직후, 최고 성능 기준도 매 epoch도 아님; 396줄 주석의 "every epoch"는 코드와 맞지 않는다). [확실] (코드 읽기)
- 영상별 결정은 원본 `vote_accuracy_per_video`(87~117줄)와 같은 규칙(예측값을 3.34로 나눈 몫으로 0/5/10, 영상 안 다수결, 동률이면 영상 평균 예측; 임계값 3.34와 6.68)을 별도로 구현했다. 원본 함수의 VA와 원본이 출력한 영상별 줄과 일치하는지 매 fold 검증하며 일치하지 않으면 중단한다. 영상 식별자는 우리 테스트 메타데이터 순서와 start_indices 구간을 맞춰 붙이고 순서와 클래스를 검증한다. 실행 전후 `third_party` 보호 목록과 학습 출력 파일의 SHA-256 불변도 확인한다. [확실]

### 13.2 결과 요약
- 재평가 VA가 `metrics.csv` 마지막 epoch VA와 15개 fold(시드 3 × 5)에서 모두 일치했다(VRE, BSA acc도 소수 6자리까지 일치). 해시 불변 확인도 통과했다. [확실]
- 마지막 epoch 체크포인트의 영상 단위 클래스별 정확도(시드 합산, 영상 501개): 클래스 0 0.731(125/171), 5 0.327(55/168), 10 0.698(113/162). 세 시드 모두 클래스 5가 가장 낮고(0.393, 0.321, 0.268), 클래스 5 정답 168개 중 77개가 10으로, 36개가 0으로 예측되었다. 12절의 71번째 epoch 참고치와 시드 합산 클래스별 정확도가 같은 값이다. [확실]

### 13.3 의도한 차이와 미검증
- 의도한 차이: 데이터 경로와 start_indices를 우리 산출물로 바꾼다. `reproduction_eval.py`와 달리 영상별 예측과 회귀 평균값을 CSV로 남긴다. 평가 결과를 학습 출력 루트 밖의 집계 폴더에 쓴다.
- 미검증: 체크포인트 안에 epoch 번호가 없어 마지막 epoch의 가중치라는 점은 지표 일치로만 뒷받침된다. 틀린 영상의 원인 분석(영상 길이, 피험자, 회전, 추출 품질과의 관련)은 하지 않았다. 시드 3개, fold당 32~35개 영상이라 클래스별 정확도는 참고 수준이다. 미검증
- 12절 12.4의 "마지막 epoch의 혼동행렬은 산출 불가"는 이 절의 평가로 산출 가능해졌다(앞 서술은 그 시점의 사실이며 이 절이 갱신한다).
