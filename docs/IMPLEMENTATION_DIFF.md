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
