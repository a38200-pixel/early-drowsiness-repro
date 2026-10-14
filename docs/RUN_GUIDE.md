# 실행 안내서

이 문서만 보고 이 저장소의 도구를 실행해 볼 수 있게 쓴 안내서입니다. 모든 명령은 **저장소 루트**(`<REPO>`, 이 저장소를 내려받은 폴더)에서 실행합니다. 옵션 이름과 기본값은 각 도구의 소스(argparse 정의)와 실제 파일을 열어 확인한 것이고, 확인하지 못한 것은 "미확인"이라고 적었습니다. 시험 결과는 실행한 날짜와 조건(다른 작업의 CPU 부하 여부)을 붙여 적었습니다(1절, 3절 (g), 4절).

용어는 처음 나올 때 풀어 씁니다.

- **conda 환경**: 파이썬과 패키지 버전을 묶어 두는 격리된 실행 환경. 이 저장소는 `ghoddoosian_tf1`이라는 이름을 씁니다.
- **EAR**: 눈 종횡비. 눈이 감기면 작아지는 값이며, 이 저장소의 눈 감김 규칙은 양쪽 눈 EAR의 평균을 씁니다.
- **blink 행**: 영상에서 깜빡임 하나가 확정될 때마다 만들어지는 숫자 한 줄(깜빡임 누적 횟수, 빈도, 진폭, 지속시간, 속도).
- **predictor**: dlib 68점 얼굴 랜드마크 모델 파일 `shape_predictor_68_face_landmarks.dat`. 눈 위치를 찾는 데 쓰며 이 저장소에는 들어 있지 않습니다.
- **체크포인트**: 학습된 HM-LSTM 모델의 저장 파일. 점수 도구는 5개 폴드(5-fold, 데이터를 5조각으로 나눠 따로 학습한 모델 5개)의 체크포인트를 읽습니다.
- **B1**: 이 저장소가 영상에서 blink 특징을 다시 추출해 저자 코드와 하이퍼파라미터 그대로 학습한 모델(재학습 모델).
- **점수 게이지**: HM-LSTM 점수를 0에서 10 눈금으로 보여 주는 표시. 참고용이며 알람이 아닙니다(활용 트랙(코드 폴더 improve)의 결정 I-33).
- **기준 구간**: 라이브 실행의 처음 300초. 이 구간의 깜빡임으로 사람별 기준 통계를 만들고, 이 구간에는 점수를 내지 않습니다.
- **자리표시자**: `<PREDICTOR>`처럼 꺾쇠로 쓴 이름은 사용자가 자기 경로로 바꿔 넣는 값입니다(2절 표).

## 1. 빠른 시작 (데이터와 모델 없이 되는 것만)

저장소에 이미 들어 있는 파일만으로 실행되는 순서입니다. 영상, predictor, 체크포인트는 필요하지 않습니다.

```
conda env create -f envs/environment_tf1.yml
conda activate ghoddoosian_tf1
```

- 의존성은 [envs/environment_tf1.yml](../envs/environment_tf1.yml) 한 파일에 conda 패키지와 pip 패키지(TensorFlow 1.8.0, OpenCV 3.4.5, numpy 1.14.6 등)가 함께 들어 있어서 별도의 pip 설치 명령은 없습니다. 이 파일로 환경을 새로 만들어 보는 것은 이번 문서 작성에서 하지 않았습니다(미확인). 아래 확인은 이미 만들어 둔 같은 이름의 환경에서 했습니다.
- pandas 1.1.5: 코드에서 쓰지 않음. 환경에 설치되어 있지만 numpy 1.14.6과 함께는 불러올 수 없음.
- 파일에는 Windows용 런타임 패키지(`vc14_runtime`, `ucrt` 등)가 들어 있어 Linux와 macOS에서는 만들어지지 않을 수 있습니다(미확인).

환경 확인(읽기만 합니다):

```
python src/repro/check_environment.py
python src/repro/check_dataset.py
```

기대 출력: 앞의 명령은 패키지별 `expected`와 `found` 줄 뒤에 `ENVIRONMENT_OK`, 뒤의 명령은 마지막에 `DATASET_OK`를 냅니다(이번에 둘 다 확인). 주의: `check_environment.py`는 파이썬 실행 파일의 위치가 사용자 홈 폴더 아래 `anaconda3/envs/ghoddoosian_tf1/python.exe`인지도 검사하므로(소스의 `EXPECTED_EXE`), conda를 다른 곳에 설치했으면 패키지 버전이 맞아도 `executable ... [MISMATCH]`와 `ENVIRONMENT_MISMATCH`가 나오고 종료 코드 1이 됩니다.

데이터 없이 통과하는 테스트(환경변수를 설정하지 않은 상태):

```
set PYTHONPATH=src
python -m unittest discover -s src/improve/rules/tests -t src
python -m unittest discover -s src/improve/hmlstm/tests -t src
```

PowerShell에서는 `set PYTHONPATH=src` 대신 `$env:PYTHONPATH = "src"`를 씁니다. 아래는 환경변수를 모두 지우고 실행한 결과입니다. 표기는 "실행 N개 중 통과 N-K개, 건너뜀 K개"이며, unittest 마지막 줄의 `Ran N`은 건너뜀을 포함한 수입니다.

| 명령의 대상 | 결과 | 날짜와 조건 |
|---|---|---|
| `src/improve/rules/tests` | 실행 278개 중 통과 278개, 건너뜀 0개 | 2026-10-14 출력 기준(`OK`). 소요 29.5초(다른 작업이 CPU를 쓰던 중). 2026-10-13의 부하 없는 실행은 4.2초 |
| `src/improve/hmlstm/tests` | 실행 381개, 건너뜀 4개 | 통과 수는 부하가 없던 2026-10-13 실행에서 전부 성공한 값(실행 381개 중 통과 377개, 건너뜀 4개, 약 15초). 2026-10-14 실행은 부하 상태에서 일부 실패해 통과 수로 쓰지 않음(4절) |

건너뛴 4개의 이유는 3절 (g)에 있습니다.

## 2. 준비물과 환경

### 지원 범위

| 구분 | 지원 | 미확인 |
|---|---|---|
| 운영체제 | Windows | Linux, macOS |
| 파이썬과 프레임워크 | Python 3.6.13, TensorFlow 1.8.0 (conda 환경 `ghoddoosian_tf1`) | 다른 버전 |
| 연산 장치 | CPU (TensorFlow 1.8.0 CPU 패키지로 체크포인트 복원과 추론 확인) | GPU |
| 웹캠 도구 | 사용자 본인의 웹캠 한 대(Windows, 640x480 요청) | 다른 카메라와 조명 |
| 한글 화면 | Windows 한글 글꼴(3절 (c), (d)) | 글꼴이 없는 환경 |

### 필요한 외부 파일 (저장소에 포함되지 않음)

| 자리표시자 | 내용 | 주는 방법 |
|---|---|---|
| `<DATA_DIR>` | UTA-RLDD 영상 폴더(공식 데이터셋 페이지에서 내려받음, [data/README.md](../data/README.md)) | 환경변수 `RLDD_DATA_DIR` |
| `<PREDICTOR>` | dlib 68점 랜드마크 모델 `shape_predictor_68_face_landmarks.dat` | 환경변수 `DLIB_LANDMARK_PATH` 또는 `--predictor` 인자 |
| `<CKPT_DIR>` | B1 학습 출력 루트(저장소 밖). 그 아래 `b1_seed{시드}/seed{시드}/fold{k}`에 체크포인트가 있어야 합니다 | `--ckpt-root` 인자(점수 도구), 환경변수 `TRAIN_OUT_DIR`(학습 도구) |
| `<VIDEO_DIR>` | 처리할 영상이 있는 폴더 | 영상 경로 인자 |
| `<TIMELINE_DIR>` | 저장소 밖의 숫자 JSON과 그림 출력 폴더 | `--out` 등 인자 |
| `<OUT_DIR>` | 저장소 밖의 실행 결과(숫자 JSON) 폴더 | `--out-json` 인자 |
| `<SVM>` | 저자 blink 판별 SVM. 저장소 안 `third_party/ghoddoosian/Trained_SVM_C=1000_gamma=0.1_for 7kNegSample.sav` (공백이 있으므로 명령에서는 따옴표로 감쌉니다) | `--svm` 인자 |

`<CKPT_DIR>` 아래의 폴더 구조는 `timeline.py score`의 도움말 문구를 따랐고, 이 문서 작성에서 실제 폴더를 새로 만들어 확인하지는 않았습니다.

### 환경변수 전체 (재현 트랙)

| 환경변수 | 뜻 | 쓰이는 곳 |
|---|---|---|
| `RLDD_DATA_DIR` | UTA-RLDD 영상 루트 | `reproduction_blink_one_video.py`(상대 경로 영상), `extract_blinks_batch.py` |
| `DLIB_LANDMARK_PATH` | predictor 파일 경로 | `check_video_assets.py`, `reproduction_blink_one_video.py`, `extract_blinks_batch.py` |
| `BLINK_OUT_DIR` | blink txt 출력 루트(저장소 밖) | `extract_blinks_batch.py`, `reproduction_build_sequences.py` |
| `SEQ_OUT_DIR` | 시퀀스 산출물 루트(저장소 밖) | `reproduction_build_sequences.py`, `reproduction_train_baseline.py` |
| `TRAIN_OUT_DIR` | B1 학습 출력 루트(저장소 밖) | `reproduction_train_baseline.py` |
| `PYTHONPATH` | 활용 트랙 도구는 `src`로 지정해야 `python -m improve...`가 동작합니다 | 3절 (b)에서 (g) |
| `HMLSTM_CKPT_ROOT`, `TIMELINE_DIR` | 시험 전용. 체크포인트를 쓰는 시험 3개와 실제 시간선을 쓰는 시험 1개를 켭니다 | 3절 (g) |

### 한글 글꼴과 웹캠

- 재생 뷰어와 `live_combined`는 한글 문구를 Pillow(이미지 라이브러리, 환경 파일의 버전 8.3.1)와 Windows 한글 글꼴로 그립니다. 찾는 글꼴 이름은 `malgun.ttf`, `malgunbd.ttf`, `gulim.ttc` 순이고(소스의 `FONT_NAMES`), 하나도 없으면 시작할 때 오류로 끝납니다.
- 웹캠은 프로세스 하나만 열 수 있습니다(활용 트랙 결정 I-W-32). 다른 프로그램이 쓰고 있으면 안 됩니다.

## 3. 실행 방법

활용 트랙 도구(b에서 g)는 먼저 `conda activate ghoddoosian_tf1`을 하고 `PYTHONPATH=src`를 설정합니다. 어떤 도구도 프레임이나 영상을 저장하지 않습니다.

### (a) B1 재현

**목적.** 공개 체크포인트를 논문 표와 대조하고, 영상에서 blink 특징을 다시 뽑아 B1을 학습하는 재현 트랙입니다. 단계별 설명과 알려진 함정은 [PIPELINE_RUNBOOK.md](PIPELINE_RUNBOOK.md), 결과는 [REPRODUCTION_REPORT.md](REPRODUCTION_REPORT.md)에 있습니다.

**A-1. 공개 체크포인트 평가 (영상과 predictor 불필요).**

```
python src/repro/reproduction_eval.py 5
python src/repro/aggregate_seeds.py
```

- `reproduction_eval.py [폴드 수]`: 폴드 수는 위치 인자이고 기본 5입니다(1이면 Fold1만). 시작할 때 vendoring된 44개 파일의 해시를 보호 목록과 대조하고 하나라도 다르면 중단합니다.
- 결과 파일: `experiments/00_reproduction/regression_eval/`. **같은 이름의 파일이 있으면 덮어쓰므로** 보존하려면 먼저 옮겨 두세요.
- `aggregate_seeds.py`: 인자가 없습니다. 재학습 결과(`experiments/01_fold1_seeds`, `02_fold_seeds`)와 위 평가 결과를 읽어 `experiments/02_fold_seeds/aggregate_seeds123.md`와 `.csv`를 실행할 때마다 새로 만듭니다.
- 기대 출력: 이 두 명령은 이번에 실행하지 않았습니다(미확인). `reproduction_eval.py`는 위 결과 파일을 덮어쓰기 때문입니다.

**A-2. 영상 한 개에서 blink 추출 (영상, predictor 필요).**

```
set DLIB_LANDMARK_PATH=<PREDICTOR>
set RLDD_DATA_DIR=<DATA_DIR>
python src/repro/check_video_assets.py
python src/repro/reproduction_blink_one_video.py <영상 경로> [출력 txt]
```

- 영상 경로가 상대 경로이면 `RLDD_DATA_DIR` 아래에서 찾습니다. 변수가 없으면 변수 이름을 알려 주는 메시지와 함께 중단합니다.
- 출력 txt를 생략하면 소스의 기본 위치에 씁니다. 상대 출력 경로는 `third_party` 안으로 해석되므로 절대 경로를 주세요(PIPELINE_RUNBOOK 7절의 함정).

**A-3. B1 전체 파이프라인 (영상 182개, 수 시간 이상).** 순서는 영상 목록과 방향 조사, blink 일괄 추출, 시퀀스 생성, 80 epoch 학습, 평가입니다. 80 epoch 학습은 오래 걸리므로 사용자가 직접 실행합니다.

| 단계 | 명령 | 소스에서 확인한 옵션 |
|---|---|---|
| blink 일괄 추출 | `python src/repro/extract_blinks_batch.py --list`, `... --limit 3 --workers 1` | `--workers`(기본 1), `--list`(실행 없이 대상만 출력), `--only-fold {1,2,3,4,5}`, `--limit K`, `--flagged-only`, `--retry-failed`. `BLINK_OUT_DIR`, `RLDD_DATA_DIR`, `DLIB_LANDMARK_PATH` 필요 |
| 시퀀스 생성 | `python src/repro/reproduction_build_sequences.py --check-only`, 이어서 같은 명령(옵션 없이) | `--check-only`(읽기만), `--inventory`(기본 `configs/video_inventory.csv`). `BLINK_OUT_DIR`, `SEQ_OUT_DIR` 필요. 이미 있는 출력은 덮어쓰지 않고 중단 |
| B1 학습 | `python src/repro/reproduction_train_baseline.py --fold all --seed 1 --dry-run`, 이어서 `--dry-run` 없이 | `--fold`(1에서 5 또는 all, 기본 all), `--seed`(필수), `--data-dir`, `--start-indices-dir`, `--smoke`, `--epochs`(`--smoke`에서만), `--run-name`, `--dry-run`(TensorFlow를 가져오지 않고 치환 결과만 출력). `SEQ_OUT_DIR`, `TRAIN_OUT_DIR` 필요 |
| 평가 | `python src/repro/evaluate_baseline_checkpoints.py --roots <시드1 루트> <시드2 루트> <시드3 루트> --out-dir <집계 폴더>` | `--roots`(3개, 필수), `--out-dir`(필수, 저장소 밖), `--data-dir`, `--start-indices-dir`, `--dry-run` |
| 학습 집계 | `python src/repro/summarize_training.py --roots <3개 루트> --out-dir <집계 폴더>` | `--roots`(3개), `--out-dir`, `--expected-test` |

- 결과 파일: 학습은 `TRAIN_OUT_DIR/<실행이름>/fold<k>/`(체크포인트, `metrics.csv`, `final_metrics.json`, `run_config.json`, `train.log`), 추출은 `BLINK_OUT_DIR`, 시퀀스는 `SEQ_OUT_DIR`의 `npy/`와 `metadata/`입니다. 모두 저장소 밖입니다.
- B1 학습의 소요 시간은 이 저장소의 문서에 기록이 없습니다.
- 작성자 본인의 B1 학습 1회(Fold1, 시드 1, 80 에폭)는 2026-10-04에 974.5초(약 16분)가 걸렸습니다. 이 값은 그 학습의 실행 설정 파일(`run_config.json`)의 경과 시간 항목(`elapsed_seconds`)에 적힌 값이고, 이 파일은 저장소에 없습니다. 다른 폴드와 다른 시드의 학습 시간은 미확인입니다.
- `reproduction_train_fold1.py`는 레거시이며 다시 실행하면 저장소의 추적 파일을 덮어쓸 수 있으니 쓰지 마세요.

### (b) 규칙 알람 데모 live_demo

**목적.** 웹캠 앞에서 눈 감김이 3.0초 이상 이어지면 알람이 울리는 규칙 하나만 보여 줍니다(점수 없음, TensorFlow 없음).

```
python -m improve.rules.live_demo --predictor <PREDICTOR> --clean
```

| 옵션 | 뜻 | 기본값 |
|---|---|---|
| `--predictor` | predictor 파일 경로(필수) | 없음 |
| `--camera` | 웹캠 번호(640x480 요청) | 0 |
| `--beep` | 알람 순간에 경보음(2음 사이렌, 3.0초). 기본은 주 임계값에서만 | 꺼짐 |
| `--beep-all` | `--beep`와 함께 쓰면 모든 임계값에서 경보음 | 꺼짐 |
| `--thresholds` | 닫힘 임계값 목록(쉼표 구분, 1개에서 4개, 0 초과 1 미만, 첫 값이 주 임계값). 예 `0.15,0.21` | `0.15` |
| `--clean` | 화면 녹화용으로 fps 줄, 단계 시간 줄, 하단 키 안내 줄 등을 그리지 않음(키 입력은 그대로) | 꺼짐 |

- 닫힘 판정은 EAR 평균이 임계값 이하일 때입니다. 닫힘이 3.0초 이어지면 알람, 눈이 뜬 상태가 연속 0.3초면 감김 구간을 해제하고, 얼굴이 1초 이상 보이지 않으면 판정 불가로 구간을 버립니다([RULE_SPEC.md](improvement/RULE_SPEC.md)).
- 키: `q` 종료, `r` 알람 표시 끄기(알람 횟수는 그대로), `m` 음소거. 이 도구에서 Esc는 쓰지 않습니다.
- 기대 출력: 창 `live_demo`가 뜹니다(위에 임계값과 알람 횟수, 가운데 얼굴 상태 띠, 아래에 영상). 웹캠에서 프레임을 읽지 못하면 "웹캠에서 프레임을 읽지 못해 끝냅니다."를 출력하고 끝납니다.
- 결과 파일: 없음(아무것도 저장하지 않습니다).

### (c) 재생 뷰어 (녹화 영상에 규칙 알람과 점수 게이지를 겹쳐 재생)

**목적.** 이미 찍은 영상을 영상 fps로 재생하면서 규칙 알람 화면과 점수 게이지를 보여 줍니다. 검출과 점수 계산은 미리 해 두므로 재생 중에는 계산하지 않습니다. 순서는 행 추출, 점수 시간선, 규칙용 선계산, 재생입니다.

```
python -m improve.hmlstm.timeline extract --video <VIDEO_DIR>/<평소영상> --svm "<SVM>" --predictor <PREDICTOR> --out <TIMELINE_DIR>/baseline_rows.json --rotate auto
python -m improve.hmlstm.timeline extract --video <VIDEO_DIR>/<대상영상> --svm "<SVM>" --predictor <PREDICTOR> --out <TIMELINE_DIR>/target_rows.json --rotate auto
python -m improve.hmlstm.timeline score --baseline <TIMELINE_DIR>/baseline_rows.json --target <TIMELINE_DIR>/target_rows.json --ckpt-root <CKPT_DIR> --out <TIMELINE_DIR>/timeline.json
python -m improve.rules.replay_precompute --video <VIDEO_DIR>/<대상영상> --predictor <PREDICTOR> --rotate auto --out <TIMELINE_DIR>/rules.json
python -m improve.rules.replay_viewer --video <VIDEO_DIR>/<대상영상> --rules-json <TIMELINE_DIR>/rules.json --timeline-json <TIMELINE_DIR>/timeline.json
```

| 도구와 옵션 | 뜻 | 기본값 |
|---|---|---|
| `timeline extract --video`, `--svm`, `--predictor`, `--out` | 영상, 저자 SVM, predictor, 출력 JSON(모두 필수) | 없음 |
| `timeline extract --rotate` | `none`, `cw90`, `ccw90`, `180`, `auto`(MP4 메타데이터에서 읽음) 중 하나 | `none` |
| `timeline extract --progress-every N` | N프레임마다 한 줄 출력(0이면 출력 없음) | 500 |
| `timeline extract --max-frames N` | 앞 N프레임만 처리(부분 실행) | 0(전체) |
| `timeline score --baseline` | 평소영상의 행 JSON. 이 행의 뒤 3분의 1로 기준 통계를 만듭니다 | 필수 |
| `timeline score --target`, `--ckpt-root`, `--out` | 점수를 볼 영상의 행 JSON, 체크포인트 루트, 출력 시간선 JSON | 필수 |
| `timeline score --seed`, `--stats` | 체크포인트 시드, 2단계 통계 JSON | 1, 패키지 안의 `stage2_stats.json` |
| `replay_precompute --video`, `--predictor`, `--out` | 영상, predictor, 출력 JSON(필수) | 없음 |
| `replay_precompute --rotate` | `0`, `90`, `180`, `270`, `auto` 중 하나(시계 방향 각도) | `0` |
| `replay_viewer --video`, `--rules-json`, `--timeline-json` | 영상, `replay_precompute` 출력, `timeline score` 출력(필수) | 없음 |
| `replay_viewer --rotate` | 위와 같은 선택지. `auto`는 규칙용 JSON에 기록된 적용 각도 | `auto` |
| `replay_viewer --video-height` | 영상 칸의 높이(폭은 640 고정) | 480 |
| `replay_viewer --thresholds` | 닫힘 임계값 목록(쉼표 구분, 1개에서 4개, 첫 값이 주 임계값) | `0.15` |

- 키(재생 뷰어): `q` 종료, 스페이스 일시정지와 계속, `r` 처음부터.
- 기대 출력: `timeline extract`는 `extract: video=... rotation=... fps=...` 줄과 진행 줄, 끝에 `extract: frames=... rows=... check=...`와 `extract: written <파일 이름>`을 냅니다. `timeline score`는 `score: ...` 줄과 `score: elapsed_s=...; written <파일 이름>`을 냅니다. `replay_precompute`는 `rows written: frames=... fps=... applied_angle=... faces=...`를 냅니다. 재생 뷰어는 창을 엽니다(규칙 화면 아래에 시간 줄과 점수 게이지 띠).
- 결과 파일: 위 세 JSON은 `--out`으로 준 경로입니다. **저장소 안 경로와 이미 있는 파일은 거부**하고 종료 코드 2로 끝납니다. 재생 뷰어는 아무것도 저장하지 않습니다.
- 한글 문구는 Windows 한글 글꼴로 그리며 글꼴이 없으면 시작할 때 오류로 끝납니다. 회전 메타데이터가 있는 영상의 `timeline extract --rotate auto` 경로는 실제 영상으로 검증하지 않았습니다(미확인).

### (d) 웹캠 통합 live_combined (규칙 알람과 점수 게이지를 한 창에)

**목적.** 웹캠 앞에서 규칙 알람과 점수 게이지를 한 창에 같이 보여 줍니다. 점수 계산은 TensorFlow를 쓰는 별도 자식 프로세스가 맡습니다. 처음 300초는 기준 구간이라 **자연스럽게 앉아 있어야 하고, 이 구간에 졸음이 없어야** 점수를 해석할 수 있습니다(한계, 활용 트랙 결정 I-W-29). 점수는 참고용이며 알람이 아닙니다.

```
python -m improve.hmlstm.live_combined --predictor <PREDICTOR> --ckpt-root <CKPT_DIR>
python -m improve.hmlstm.live_combined --predictor <PREDICTOR> --no-score --duration 60
python -m improve.hmlstm.live_combined --predictor <PREDICTOR> --ckpt-root <CKPT_DIR> --out-json <OUT_DIR>/live.json
```

| 옵션 | 뜻 | 기본값 |
|---|---|---|
| 위치 인자 `camera` 또는 `--camera` | 웹캠 번호. 둘 다 주면 값이 같아야 합니다 | 0 |
| `--predictor` | predictor 파일 경로(필수) | 없음 |
| `--ckpt-root` | 체크포인트 루트. `--no-score`가 아니면 필요합니다 | 없음 |
| `--no-score` | 점수 프로세스 없이 규칙 알람 표시만 | 꺼짐 |
| `--duration` | 실행 시간(초). 0보다 커야 합니다 | 없음(q로 종료) |
| `--rotate` | `none`, `cw90`, `180`, `ccw90` 중 하나 | `none` |
| `--preview-hz` | 그리기 주기(초당 횟수, 1 이상 30 이하) | 15 |
| `--beep` | 알람 순간에 경보음(주 임계값만) | 꺼짐 |
| `--out-json` | 종료 시 숫자 JSON을 쓸 새 경로(저장소 밖) | 없음(저장 안 함) |

- 키: `q` 종료, `r` 알람 표시 끄기, `m` 음소거. Esc와 창 닫기도 종료로 처리하도록 코드에 있으나 실제 입력으로는 확인하지 않았습니다(미확인). `Ctrl+C`도 종료이며 상태 `interrupted`로 요약을 출력합니다.
- 화면 흐름: 점수 프로세스를 띄우는 동안 게이지 자리에 "점수 프로세스 로딩 중", 기준 구간에는 "기준 수집 중 N초 / 300초, 행 n개"와 "기준 구간(처음 300초)에는 졸음이 없는 상태여야 함", 이후 "수집 중 k/30 블링크", 점수가 나오면 `5.5 [4.3-7.0]`처럼 평균과 범위가 보입니다(`docs/images`의 시연 화면에서 확인).
- 기대 출력(콘솔): 종료할 때 요약이 가장 먼저 출력됩니다. 줄은 `status: finished`, `frames`, `mean fps`, 프레임 간격, 처리 시간, 프레임 예산 대비 비율, `live session`, `score process`, `self check`, `rule alarms`, `swallowed exceptions`, 출처별 프레임 수와 EAR 분포이며, 마지막 줄은 고정 문장 "이 값은 라이브 실행 요약이며 점수나 알람의 타당성 검증이 아님"입니다. 종료 코드는 정상 종료 0, 프레임이 없거나 오류가 있으면 1, 인자 오류는 2입니다.
- 첫 점수는 기준 구간(300초)이 끝난 뒤 기준 이후 깜빡임 행이 30개 쌓여야 나옵니다. 사용자가 실행한 1회(점수 포함 600초 실행)에서 첫 점수 시각은 348.8초(화면 관찰 5분 50초)였지만 단일 사용자의 한 번 관찰이며 깜빡임 빈도에 따라 달라집니다.
- 결과 파일: `--out-json`을 줄 때만 숫자 JSON(행, 창별 점수, 알람, 처리 시간 요약)을 씁니다. 프레임과 영상은 저장하지 않습니다.
- 알려진 결함: 저장 JSON의 `scores[].time_sec`는 `frame_index`를 가정한 30 fps로 나눈 값이라 실제 시각이 아닙니다(기록에서 확인함. 한 실행의 첫 창은 기록 345.3초 대 행 시각 348.8초). 시간 비교에는 행의 `t`를 쓰세요. 상세는 [IMPROVEMENT_LOG.md](improvement/IMPROVEMENT_LOG.md)의 "live_combined 두 번째 전체 실행과 재채점 비교 (2026-10-13)" D절에 있습니다.
- `--ckpt-root`를 주고 실제 점수 프로세스까지 돌리는 경로는 사용자가 두 번 직접 실행해 확인했습니다. 이 문서를 쓰면서는 웹캠을 쓰지 않았습니다.

### (e) 재채점 비교 live_score_recheck

**목적.** `live_combined --out-json`으로 저장한 숫자 파일의 행을 새 세션에 다시 넣고 점수를 이 프로세스에서 직접 계산해, 기록된 라이브 점수와 같은지 비교합니다. 같은 LiveSession 코드로 같은 행을 다시 채점하는 비교이며 B1 오프라인 경로와의 비교가 아닙니다.

```
python -m improve.hmlstm.live_score_recheck --live-json <OUT_DIR>/live.json --ckpt-root <CKPT_DIR>
```

| 옵션 | 뜻 | 기본값 |
|---|---|---|
| `--live-json` | `live_combined --out-json`이 만든 JSON(필수) | 없음 |
| `--ckpt-root` | 체크포인트 루트(필수) | 없음 |
| `--tolerance` | 점수 차이 허용값(0에서 10 눈금, 0 이상의 유한한 수) | 1e-4 |

- 설정값(기준 구간 300초 등)은 JSON에 기록되지 않으므로 코드의 기본값(300, 600, 30)을 쓰고 요약에 "기본값(JSON 에 기록 없음)"이라고 적습니다.
- 기대 출력: `windows: recorded N ; recomputed N ; 일치`, `frame numbers: mismatches 0 of N`, `score diff` 줄(최대와 평균 차이), `over tolerance`, 마지막 창 기록값과 재계산값, `verdict: 일치`(또는 불일치, 미확정), 마지막 줄은 고정 문장 "이 값은 같은 행의 재채점 비교이며 B1 오프라인 경로와의 비교가 아님". 종료 코드는 일치 0, 불일치나 미확정 1, 입력 오류 2입니다.
- 결과 파일: 없음(콘솔 요약만, 파일을 저장하지 않습니다).

### (f) 측정 도구

**webcam_hook_speed: 원본 blink 추출을 웹캠에 연결했을 때의 처리 속도를 잽니다.**

```
python -m improve.hmlstm.webcam_hook_speed [웹캠 번호] [초] --predictor <PREDICTOR> [--preview-thread]
```

| 옵션 | 뜻 | 기본값 |
|---|---|---|
| 위치 인자 `camera` | 웹캠 번호 | 0 |
| 위치 인자 `seconds` | 측정 시간(초) | 60 |
| `--predictor` | predictor 파일 경로(필수) | 없음 |
| `--preview` | 읽은 프레임을 줄여 창에 보여 줌(그리는 시간이 처리 시간에 들어감) | 꺼짐 |
| `--preview-thread` | 별도 스레드에서 게이지와 비슷한 가짜 화면을 그림(`--preview`와 함께 쓰면 이것이 우선) | 꺼짐 |
| `--preview-hz` | `--preview-thread`의 그리기 주기(1 이상 30 이하) | 15 |

- 기대 출력: 콘솔 요약(`status`, `frames`, `mean fps`, 프레임 간격, `processing median / p95 / max`, `frame budget`, `preview mode` 등)과 고정 문장. 결과 파일은 없습니다.

**ear_hook_compare: 같은 영상에서 후킹이 모은 EAR 열과 `replay_precompute`의 EAR 열을 비교합니다.**

```
python -m improve.hmlstm.ear_hook_compare --video <VIDEO_DIR>/<영상> --predictor <PREDICTOR> --rules-json <TIMELINE_DIR>/rules.json [--rows-json <TIMELINE_DIR>/rows.json]
```

| 옵션 | 뜻 | 기본값 |
|---|---|---|
| `--video`, `--predictor` | 영상, predictor(필수) | 없음 |
| `--rules-json` | 같은 영상에서 `replay_precompute`가 만든 JSON(필수) | 없음 |
| `--rows-json` | (선택) 같은 영상에서 `timeline extract`가 만든 행 JSON. 있으면 후킹이 만든 깜빡임 행과도 대조 | 없음 |
| `--rotate` | 후킹의 회전 옵션. `auto`는 규칙 JSON에 기록된 적용 각도 | `auto` |
| `--threshold` | LiveRules 닫힘 임계값(0 초과 1 미만) | 0.15 |

- 기대 출력: 후킹과 규칙의 프레임 수, EAR 최대 차이, 알람 비교, `verdict: 일치`(또는 불일치). 영상 전체를 읽는 동안 창은 뜨지 않습니다. 결과 파일은 없습니다.

### (g) 테스트 실행

**목적.** 활용 트랙의 시험을 실행합니다. 카메라, 창, TensorFlow 없이 도는 시험이 대부분입니다.

```
set PYTHONPATH=src
python -m unittest discover -s src/improve/rules/tests -t src
python -m unittest discover -s src/improve/hmlstm/tests -t src
```

- 실행 결과(환경변수 없음): rules는 실행 278개 중 통과 278개, 건너뜀 0개(2026-10-14). hmlstm은 실행 381개, 건너뜀 4개이고 통과 수는 부하가 없던 2026-10-13 실행에서 전부 성공한 값(실행 381개 중 통과 377개, 건너뜀 4개, 약 15초)입니다. 부하가 있을 때의 실패는 4절을 보세요.
- 건너뛴 4개의 이유(출력된 문구 그대로의 요지):

| 시험 | 건너뛴 이유 |
|---|---|
| `test_live_session`의 실제 데이터 시험 1개 | `TIMELINE_DIR`에 `timeline.json`이 없음 |
| `test_score_analysis`, `test_scoring`, `test_timeline`의 체크포인트 시험 3개 | 환경변수 `HMLSTM_CKPT_ROOT`(`<CKPT_DIR>`)가 없음 |

- 건너뛴 시험을 켜려면 `HMLSTM_CKPT_ROOT`를 `b1_seed1` 폴더의 **부모** 폴더로, `TIMELINE_DIR`를 `timeline.json`이 있는 폴더로 지정합니다. 두 변수를 모두 설정하면 체크포인트 시험 3개가 실행되고, `timeline.json`이 없으면 실제 데이터 시험 1개는 계속 건너뜁니다(이 문서 작성에서는 `timeline.json`이 있는 폴더가 없어 이 시험이 실제로 도는 것은 확인하지 못했습니다, 미확인).
- 재현 트랙(`src/repro`)에는 단위 시험 파일이 없습니다(`test_*.py` 검색 결과, 활용 트랙 두 폴더에만 있음). 재현 트랙은 1절의 `check_environment.py`, `check_dataset.py`와 A-1의 평가로 점검합니다.
- 규칙 알람의 3.0초 경계는 합성 입력 시험으로만 고정되어 있고 실제 영상으로는 시험하지 않았습니다(미확인).

## 4. 자주 겪는 문제

| 증상 | 원인과 해결 |
|---|---|
| `python -m improve...`가 "No module named improve" 오류 | `PYTHONPATH=src`를 설정하지 않았습니다. 저장소 루트에서 `set PYTHONPATH=src`(PowerShell은 `$env:PYTHONPATH = "src"`)를 먼저 실행하세요. |
| 환경을 활성화하지 않고 환경의 `python.exe`를 직접 부르면 NumPy 관련 DLL 오류 | `conda activate ghoddoosian_tf1` 뒤에 실행하거나 `conda run -n ghoddoosian_tf1 ...`을 쓰세요([PIPELINE_RUNBOOK.md](PIPELINE_RUNBOOK.md) 7절). |
| 웹캠이 열리지 않거나 프레임을 읽지 못한다는 메시지 | 웹캠은 프로세스당 하나만 열립니다(결정 I-W-32). 다른 프로그램이나 이전에 띄운 도구가 아직 떠 있는지 확인하세요. 번호가 맞는지(`--camera`)도 확인하세요. |
| 한글이 깨지거나 시작할 때 글꼴 오류로 끝남 | 재생 뷰어와 `live_combined`는 `malgun.ttf`, `malgunbd.ttf`, `gulim.ttc` 중 하나가 필요합니다(Windows 한글 글꼴). 글꼴이 없는 환경은 미확인입니다. |
| "set the environment variable DLIB_LANDMARK_PATH ..." 같은 메시지 | 재현 트랙 도구에는 환경변수가 필요합니다(`RLDD_DATA_DIR`, `DLIB_LANDMARK_PATH`, `BLINK_OUT_DIR`, `SEQ_OUT_DIR`, `TRAIN_OUT_DIR`). 활용 트랙 도구는 환경변수 대신 `--predictor`와 `--ckpt-root` 인자를 씁니다. |
| `live_combined`에서 `--ckpt-root`를 빼면 종료 코드 2 | `--no-score`를 주지 않으면 `--ckpt-root`가 필요합니다. |
| `--out-json`, `--out`이 거부됨(종료 코드 2) | 숫자 JSON은 **새 경로**에만 쓰고 이미 있는 파일은 덮어쓰지 않습니다. 저장소 안 경로도 거부합니다. 저장소 밖의 새 파일 이름을 주세요. |
| `reproduction_eval.py`를 다시 실행했더니 결과가 사라짐 | 이 도구는 `experiments/00_reproduction/regression_eval/`의 기존 파일을 덮어씁니다. 실행 전에 옮겨 두세요. |
| 점수가 한참 동안 나오지 않음 | 정상입니다. 처음 300초는 기준 구간이고, 그 뒤 깜빡임 행이 30개 쌓여야 첫 점수가 나옵니다. 관찰된 첫 점수 시각은 348.8초였습니다(단일 사용자, 1회). |
| 점수가 오르내림 | 점수는 참고용이며 알람이 아닙니다. 5개 폴드 모델 사이의 차이가 크고(같은 창에서 평균 2.76과 3.25, 사용자 영상 2개), 졸음 반영 여부는 검증하지 못했습니다([README.md](../README.md) 7절). |
| 기준 구간에 졸린 상태로 앉아 있었음 | 기준이 그 상태로 만들어져 점수 해석이 달라집니다. 코드로 막지 않는 한계입니다(결정 I-W-29). 다시 시작하세요. |
| `check_environment.py`가 `MISMATCH` | 패키지 버전 줄을 확인하세요. 버전이 모두 `OK`인데 `executable`만 `MISMATCH`이면 conda 설치 위치가 소스의 기대 위치와 다른 것입니다(1절). |
| hmlstm 시험 중 `test_live_combined`의 `ScoreClientTest`가 실패함 | 이 시험은 실제 자식 프로세스의 상태가 바뀌기를 15초 안에 기다리므로, CPU를 많이 쓰는 다른 작업이 돌고 있으면 시간 초과로 실패할 수 있습니다. 관찰: 2026-10-13에는 다른 작업이 없을 때 약 15초에 전부 성공했고, 2026-10-14에는 다른 학습이 CPU를 쓰던 중에 154초에서 185초가 걸렸으며 실패가 5건(1차)과 2건(2차) 나왔습니다. 2차의 2건은 `ScoreClientTest` 두 시험이었고, 1차의 나머지 3건은 어떤 시험인지 확인하지 못했습니다(미확인). 정확한 원인은 미확인입니다(부하 때문으로 보이지만 확인하지 않음). 다른 작업을 멈추고 다시 실행해 보세요. |
| Esc나 창의 닫기 버튼으로 종료되는지 | `live_combined`에서는 코드에 있으나 실제 확인은 하지 않았습니다(미확인). 확실한 종료는 `q` 또는 `Ctrl+C`입니다. |

개인정보와 데이터 규칙: 시연 도구는 사용자 본인의 웹캠만 쓰고 프레임이나 영상을 저장하지 않습니다. 숫자 JSON은 저장소 밖에만 둡니다. 영상과 dlib 모델은 저장소에 올리지 않습니다([data/README.md](../data/README.md), [README.md](../README.md) 10절). 서드파티 코드와 라이선스는 [THIRD_PARTY.md](../THIRD_PARTY.md)에 있습니다.
