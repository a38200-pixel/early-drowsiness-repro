# 실행 안내 (S0~S5)

작성일: 2026-10-05. 명령은 저장소 루트에서 실행한다. conda 환경 `ghoddoosian_tf1`(Python 3.6, TensorFlow 1.8)을 활성화한 상태를 가정한다. 명령 형식은 각 스크립트 맨 위 주석의 사용법을 따랐다. 소요 시간은 `REPRODUCTION_LOG.md`에 적힌 값만 쓰고 없으면 "미확인"이라 쓴다. 경로는 환경변수 이름과 `<자리표시자>`로만 적는다. 80 epoch 학습은 사용자가 직접 실행한다(에이전트는 실행하지 않는다).

## 0. 환경변수 이름과 폴더 규칙

| 환경변수 | 뜻 | 쓰이는 단계 |
|---|---|---|
| `RLDD_DATA_DIR` | UTA-RLDD 영상 루트(`Fold<N>_part<M>` 폴더가 들어 있는 곳, 저장소 밖) | S2 |
| `DLIB_LANDMARK_PATH` | `shape_predictor_68_face_landmarks.dat` 파일(저장소 밖) | S2 |
| `BLINK_OUT_DIR` | blink txt 출력 루트(저장소 밖, 기본값 없음) | S2(출력), S3(입력) |
| `SEQ_OUT_DIR` | 시퀀스 산출물 루트(저장소 밖, 기본값 없음, 저장소 안이면 중단) | S3(출력), S4, S5(입력) |
| `TRAIN_OUT_DIR` | B1 학습 출력 루트(저장소 밖, 기본값 없음, 저장소 안이거나 `my_model1~5` 위치이면 중단) | S4 |

집계 결과(S5)는 학습 출력 루트 밖의 집계 폴더를 `--out-dir`로 받는다. 출처: 각 스크립트 맨 위 주석, 로그 "시퀀스 생성 구현 사전 등록 (2026-10-03)", "기준선 학습 러너(B1) 구현 사전 등록 (2026-10-03)".

## 1. S0 공개 체크포인트 평가

| 항목 | 내용 |
|---|---|
| 명령 | `python src/repro/check_environment.py`, `python src/repro/check_dataset.py`, `python src/repro/reproduction_eval.py 5` |
| 입력 | `third_party/ghoddoosian/`의 npy와 `my_model1~5`(저장소 안) |
| 출력 | `experiments/00_reproduction/regression_eval/`(안). 같은 이름의 파일이 있으면 덮어쓴다(README) |
| 예상 소요 | 미확인(로그에 없음) |
| 점검 | 시작 시 보호 목록 44개 파일의 해시를 검사하고 다르면 중단한다 |

## 2. S1 저자 npy 재학습(B0)

| 항목 | 내용 |
|---|---|
| 명령 | `python src/repro/train_fold1_seeds.py <시드> [epoch수=80] [출력폴더명]`, `python src/repro/train_fold_seeds.py <fold 1~5> <시드> [epoch수=80] [출력폴더명]`. 일괄: `scripts\run_fold1_seeds.bat`, `scripts\run_folds_seed.bat <시드>`. 요약: `python src/repro/summarize_fold1_seeds.py`, `python src/repro/summarize_folds.py <시드>`, `python src/repro/aggregate_seeds.py` |
| 입력 | 저자 npy(저장소 안) |
| 출력 | `experiments/01_fold1_seeds/`, `experiments/02_fold_seeds/`(안). 체크포인트·npy·train.log는 git이 추적하지 않는다 |
| 예상 소요 | Fold1 1회 20.8분, fold당 약 21분 추정(5 fold 약 1.8시간), 2 epoch 스모크 약 0.8~0.9분(로그 "Session 2 ... step 7, step 9", "Fold1 다중 시드 재학습 준비 (2026-09-30)") |
| 주의 | `reproduction_train_fold1.py`는 레거시이며 다시 실행하면 `experiments/00_reproduction/fold1_training/`의 추적 파일을 덮어쓸 수 있다(문서 7절) |

## 3. S2 영상→blink 재추출

| 순서 | 명령 | 비고 |
|---|---|---|
| 1 | `python src/repro/inventory_videos.py [--subset flagged]` | 영상 목록. `RLDD_DATA_DIR` 필수 |
| 2 | `python src/repro/survey_orientation.py [--only <영상>] [--subset zero] [--limit K] [--resolve-only]` | 방향별 얼굴 검출 조사. 전체 182개 560.2초(로그 "배치 시험과 회전 조사 결과 (2026-10-02)") |
| 3 | `python src/repro/review_orientation.py [--only <영상>]` | review 영상의 육안 판정(대화형) |
| 4 | `python src/repro/reproduction_blink_headless.py <영상> [출력txt] [--rotate none｜cw90｜ccw90｜180] [--dry-run｜--show]`, 비교: `--compare <a.txt> <b.txt>` | 단독 1영상 755.9초(로그 "blink 무화면 실행 검증 (2026-10-02)") |
| 5 | `python src/repro/extract_blinks_batch.py --list`, `--limit 3 --workers 1`, `--workers 4`(옵션 `--only-fold`, `--flagged-only`, `--retry-failed`) | 일괄 추출. 전체는 workers 6으로 12시간 48분 6초(로그 "일괄 추출 실행 결과 (2026-10-03)") |
| 6 | `python src/repro/summarize_batch.py [--recent-minutes N]` | 진행 요약. `BLINK_OUT_DIR` 설정 필요 |

- 입력: 영상(`RLDD_DATA_DIR`), dlib 모델(`DLIB_LANDMARK_PATH`). 출력: blink txt(`BLINK_OUT_DIR`, 밖), `configs/*.csv`(안).
- 방향(회전)은 `configs/video_orientation_overrides.csv`의 육안 판정이 있으면 그 값, 없으면 `video_orientation_survey.csv`의 `resolved_orientation`을 쓴다(`extract_blinks_batch.py` 맨 위 주석).
- 배치는 처리 목록 174개(manual_flags 8개 제외)를 대상으로 했다(로그 "일괄 추출 실행 결과 (2026-10-03)").

## 4. S3 시퀀스 생성

| 항목 | 내용 |
|---|---|
| 명령 | `python src/repro/reproduction_build_sequences.py --check-only`(읽기만, 쓰기 없음), `python src/repro/reproduction_build_sequences.py` |
| 입력 | `BLINK_OUT_DIR`(읽기 전용), `configs/video_inventory.csv`(`--inventory <CSV>`로 바꿀 수 있음) |
| 출력 | `SEQ_OUT_DIR`의 `npy/`(20개), `metadata/`(영상별 요약, 제외 목록, start_indices JSON, coverage). 이미 있는 출력은 덮어쓰지 않고 중단한다 |
| 비교 | `python src/repro/compare_with_author_npy.py [--check-only｜--match-free｜--graded]`(`SEQ_OUT_DIR`, `BLINK_OUT_DIR` 필요, 저자 npy는 읽기 전용) |
| 예상 소요 | 미확인(로그에 없음) |

## 5. S4 B1 학습

| 항목 | 내용 |
|---|---|
| 명령 | `python src/repro/reproduction_train_baseline.py --fold all --seed <시드> --dry-run`(치환 확인, TensorFlow import 없음), `python src/repro/reproduction_train_baseline.py --fold all --seed <시드>`(80 epoch, 저자 하이퍼파라미터 그대로). 스모크: `--fold <k> --seed <시드> --smoke --epochs 2` |
| 입력 | `SEQ_OUT_DIR`의 npy와 start_indices JSON(`--data-dir`, `--start-indices-dir`로 바꿀 수 있음) |
| 출력 | `TRAIN_OUT_DIR/<실행이름>/fold<k>/`(체크포인트, `metrics.csv`, `final_metrics.json`, `run_config.json`, `train.log`). 스모크는 `SMOKE` 표식 파일을 쓰고 집계가 무시한다 |
| 예상 소요 | 미확인(B1 학습 소요는 로그에 없음. B0는 fold당 약 21분 추정) |
| 주의 | 이미 있는 fold 폴더가 있으면 덮어쓰지 않고 중단한다. `--epochs`는 `--smoke`와만 쓴다 |

## 6. S5 평가·영상 진단·불확실성

| 순서 | 명령 | 출력 |
|---|---|---|
| 1 | `python src/repro/summarize_training.py --roots <시드1 루트> <시드2 루트> <시드3 루트> --out-dir <집계 폴더>` | 정상성 점검, 표, `fold_table.csv`, 혼동행렬(71번째 epoch 참고) |
| 2 | `python src/repro/evaluate_baseline_checkpoints.py --roots <3개 루트> --out-dir <집계 폴더> [--dry-run]` | 영상별 예측 CSV, 혼동행렬, 재평가 대 `metrics.csv` 대조(체크포인트 로드와 예측만, 학습 없음) |
| 3 | `python src/repro/diagnose_video_level_baseline.py --out-dir <집계 폴더>` | 영상 단위 진단(로지스틱, RF, 대조군) |
| 4 | `python src/repro/estimate_baseline_uncertainty.py --b1-predictions <시드1 CSV> <시드2 CSV> <시드3 CSV> --diagnose-predictions <진단 CSV> --out-dir <집계 폴더>` | 불확실성 구간 CSV와 요약 |

- 집계 폴더는 저장소 밖이어야 하고 학습 출력 루트·입력 폴더와 겹치면 중단한다. 이미 있는 출력은 덮어쓰지 않는다.
- 예상 소요: 미확인(로그에 없음).

## 7. 알려진 함정 (로그에 적힌 것만)

| 함정 | 설명 | 출처 |
|---|---|---|
| 환경 활성화 없이 환경의 `python.exe`를 직접 호출 | dlib 설치 환경에서 NumPy MKL DLL 검색 오류. `conda activate ghoddoosian_tf1` 또는 `conda run -n ghoddoosian_tf1 ...`를 쓴다 | 로그 "Installation issues and resolutions" |
| 출력 덮어쓰기 거부 | S1 학습 폴더, S3, S4, S5 도구는 이미 있는 출력이 있으면 덮어쓰지 않고 중단한다. 다시 하려면 폴더를 옮기거나 새 이름을 쓴다 | 로그 "Fold2~5 재학습 스크립트 준비 (2026-10-01)", "시퀀스 생성 구현 사전 등록 (2026-10-03)" |
| 일괄 파일의 건너뛰기 | `.bat`은 이미 있는 fold 폴더를 건너뛰고 실패하면 종료 코드 1로 멈춘다. 중간에 멈춘 폴더는 직접 옮겨야 다시 돈다 | `scripts/run_folds_seed.bat`, 로그 "Fold1 다중 시드 재학습 준비 (2026-09-30)" |
| 레거시 스크립트 | `reproduction_train_fold1.py`를 다시 실행하면 추적 파일을 덮어쓸 수 있다 | 문서 7절 |
| 매니페스트 CRLF | `* text=auto`로 Windows에서 `protected_manifest.sha256`이 CRLF로 체크아웃되면 `sha256sum -c`가 실패한다. `.gitattributes`에 `configs/*.sha256 text eol=lf`가 적용되어 있다 | 로그 "Fold1 다중 시드 재학습 준비 (2026-09-30)"의 후속 수정 |
| Python 3.6 제약 | `subprocess.run(capture_output=...)`은 3.6에서 쓸 수 없다 | 로그 "Repository consolidation log, Stage 1" |
| `no_output` 상태의 뜻 | 처리 실패가 아니라 종료 코드 0이면서 blink가 기록되지 않은 영상이다. 기본 실행은 건너뛰고 `--retry-failed`로만 다시 처리한다 | 문서 P-05 |
| 요약 도구의 남은 시간 | 매니페스트에 행이 없지만 출력 txt가 이미 있는 영상을 "미처리"로 세어 남은 시간이 나올 수 있다 | 로그 "일괄 추출 실행 결과 (2026-10-03)" |
| 학습 로그의 영상별 예측 | 원본은 epoch 번호가 10의 배수일 때만 영상별 예측을 출력한다. 마지막 epoch(번호 79)는 로그에 없다 | 로그 "B1 기준선 학습 결과 집계 (2026-10-04)" |
| 단일 영상 래퍼의 상대 출력 경로 | `reproduction_blink_one_video.py`에 상대 출력 경로를 주면 `third_party` 안으로 해석된다. 헤드리스 러너는 절대경로로 해결했다 | 문서 P-16 |
