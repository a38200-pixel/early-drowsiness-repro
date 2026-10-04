# Ghoddoosian et al. CVPRW 2019 reproduction log

Date: 2026-09-30  
Platform: Windows, CPU-only  
Project: `Early-Drowsiness-Detection`

## Environment

- Conda environment: `ghoddoosian_tf1`
- Environment path: `C:\Users\<USER>\anaconda3\envs\ghoddoosian_tf1`
- Python: 3.6.13
- pip: 21.2.2
- setuptools: 58.0.4
- wheel: 0.37.1
- TensorFlow: 1.8.0 (CPU)
- NumPy: 1.14.6

Activate the environment before running any command. Activation is important
because Conda-provided DLLs, including dlib dependencies, are added to `PATH`
during activation.

```powershell
conda activate ghoddoosian_tf1
python --version
where.exe python
```

The first `where.exe python` result must be:

```text
C:\Users\<USER>\anaconda3\envs\ghoddoosian_tf1\python.exe
```

## Installed major packages

| Package | Version | Source |
|---|---:|---|
| tensorflow | 1.8.0 | pip |
| tensorboard | 1.8.0 | pip |
| numpy | 1.14.6 | pip |
| scipy | 1.2.1 | pip |
| scikit-learn | 0.19.2 | pip |
| opencv-python | 3.4.5.20 | pip |
| dlib | 19.17 / runtime 19.17.0 | conda-forge |
| imutils | 0.5.4 | pip |
| matplotlib | 2.2.5 | pip |
| protobuf | 3.19.6 | pip |
| grpcio | 1.48.2 | pip |

`python -m pip check` reports no broken requirements.

## Static source inspection

The baseline consists of `Training.py`, `HMSLSTM_Main.py`,
`HMSLSTM_cell.py`, and `MultiHMSLSTM.py`.

- All four files parse with Python 3.6; no Python 2-only syntax was found.
- The implementation is TensorFlow 1.x-specific.
- It uses `tf.contrib`, `tf.Session`, `tf.placeholder`, `tf.scan`,
  `tf.variable_scope`, and TensorFlow internal RNN APIs.
- The baseline files do not use `tf.nn.rnn_cell` or `np.int`.
- `blink_video.py` uses the removed/deprecated `np.int` alias, but this is
  available in the selected NumPy 1.14.x environment.
- `HMSLSTM_cell.py` contains two references to the undefined
  `self.hstate_size_size` in the `state_size` and `output_size` properties.
  This is a likely source typo. Neither property was reached by the tested
  TensorFlow 1.8 graph/evaluation/training paths, so the source was not changed.

No original Python source file was modified, and no backup source file was
needed.

## Dataset verification

All Fold1 through Fold5 files are present. Every blink input has shape
`(N, 30, 4)` and every label input has shape `(N, 1)`. All arrays have dtype
`float64`.

| Fold | Training input | Test input | Training labels | Test labels |
|---:|---:|---:|---:|---:|
| 1 | `(7379, 30, 4)` | `(1269, 30, 4)` | `(7379, 1)` | `(1269, 1)` |
| 2 | `(6605, 30, 4)` | `(2043, 30, 4)` | `(6605, 1)` | `(2043, 1)` |
| 3 | `(6465, 30, 4)` | `(2196, 30, 4)` | `(6465, 1)` | `(2196, 1)` |
| 4 | `(7186, 30, 4)` | `(1462, 30, 4)` | `(7186, 1)` | `(1462, 1)` |
| 5 | `(6957, 30, 4)` | `(1691, 30, 4)` | `(6957, 1)` | `(1691, 1)` |

## Original Training.py defaults

- `load=False`
- five folds: `for i in range(5)`
- epochs: 80
- batch size: 64
- learning rate: 0.000053
- sequence length: 30
- feature count: 4
- hidden state sizes: `[32, 32, 32, 32]`
- checkpoint path: project root (`./`)

There is no command-line evaluation option. Setting `load=True` restores the
provided checkpoint and breaks after one evaluation epoch. With the original
`load=False`, `save_variables()` writes to the project root. Because training
uses zero-based `ii=i`, an unmodified full training run creates `my_model0`
and can overwrite `my_model1` through `my_model4`. Therefore the original
script was not executed directly in training mode.

## Pretrained Fold1 evaluation

The source was transformed only in memory to use `load=True` and
`range(1)`. No file was edited and no save branch was entered.

Result:

```text
checkpoint restore: success (my_model1)
BSA loss: 15.711236
BSA accuracy: 0.4838455476753349
VA: 0.6363636363636364
BSRE: 3.185322
VRE: 2.419539
```

Graph construction, checkpoint restoration, tensor shapes, and all four
reported evaluation metrics succeeded.

## Fold1 training smoke test

A one-epoch CPU smoke test was run after successful checkpoint evaluation.
The code was transformed only in memory to select Fold1, set one epoch, and
redirect every output to `reproduction_artifacts`. Original checkpoints were
not overwritten.

```text
training moving accuracy: 0.35362551043557167
test BSA loss: 24.905964
test BSA accuracy: 0.33569739952718675
test VA: 0.2727272727272727
test BSRE: 3.132637
test VRE: 2.650519
checkpoint save: success
```

Smoke-test files are under:

- `reproduction_artifacts/checkpoints_smoke/`
- `reproduction_artifacts/training_smoke/`

The complete original 80-epoch, five-fold training was not run.

## Video-stage verification

- `shape_predictor_68_face_landmarks.dat` is present in the project root.
- Predictor size: 99,693,937 bytes.
- `Trained_SVM_C=1000_gamma=0.1_for 7kNegSample.sav` loads successfully.
- Loaded estimator: `sklearn.svm.classes.SVC`
- Expected SVM input width: 13 features.
- A dummy 13-feature prediction completed successfully.

The SVM emits this compatibility warning:

```text
Trying to unpickle estimator SVC from version pre-0.18 when using version
0.19.2. This might lead to breaking code or invalid results.
```

The artifact is therefore older than scikit-learn 0.18. It works with 0.19.2,
but video predictions should be treated as compatibility-sensitive. Do not
upgrade scikit-learn without validating predictions.

`blink_video.py` was not launched because its default input video path is
empty and it opens interactive Tk/OpenCV windows. A real input video path must
be assigned before running it.

## Installation issues and resolutions

1. `dlib=19.16` was not available from the current conda-forge Windows
   channel. `dlib=19.15` was also unavailable.
   - Resolution: use the available Python 3.6 build `dlib=19.17`.
2. The dlib 19.17 Conda package requires `numpy>=1.14.6`.
   - Resolution: move from NumPy 1.14.5 to 1.14.6. TensorFlow 1.8 import and
     Fold1 evaluation were rerun successfully with identical metrics.
3. Invoking the environment's `python.exe` directly after installing the
   Conda dlib package caused a NumPy MKL DLL lookup error.
   - Cause: direct invocation did not add the environment's Conda DLL paths.
   - Resolution: use `conda activate ghoddoosian_tf1` or `conda run -n
     ghoddoosian_tf1 ...`.
4. Conda's NumPy update initially left pip metadata reporting 1.14.5 while the
   imported runtime reported 1.14.6.
   - Resolution: force-reinstall the NumPy 1.14.6 wheel without dependencies.
     Runtime and package metadata now both report 1.14.6.
5. TensorFlow reports that the CPU supports AVX2 but the distributed binary
   was not compiled to use it.
   - This is informational and does not prevent execution.

## Reproduction status (updated in session 2, see below)

- Isolated legacy environment: complete
- Existing Python 3.12 installations: unchanged
- TensorFlow 1.8 baseline graph: successful
- Pretrained Fold1 restore/evaluation: successful
- Pretrained 5-fold evaluation: successful (session 2)
- Fold1 one-epoch training/data-flow smoke test: successful
- Fold1 full 80-epoch retraining: done, ONE run only (session 2)
- Full five-fold retraining: not run (estimate below, awaiting approval)
- Video dependency imports and legacy SVM load: successful
- Actual video processing: pending selection of an input video

---

# Session 2: baseline reproduction (2026-09-30)

## New helper files (originals untouched)

| File | Purpose |
|---|---|
| `check_environment.py` | interpreter + package version check (result: ENVIRONMENT_OK) |
| `check_dataset.py` | source files, Fold1-5 npy shapes/dtypes, checkpoint files (result: DATASET_OK) |
| `reproduction_eval.py` | pretrained evaluation; executes Training.py in memory with `load=True` |
| `reproduction_train_fold1.py` | Fold1 retraining; output redirected to `reproduction_artifacts/fold1_training/` |
| `check_video_assets.py` | loads the dlib predictor and SVM (result: VIDEO_ASSETS_OK) |
| `reproduction_blink_one_video.py` | runs blink_video.py on one video via in-memory path substitution |

All scripts hash-check 41 protected files (my_model*, *.npy, the six core .py files) before and
after each run: **changed: none, added: none** for every run.

## Environment (step 1)

`where.exe python` first hit is `...\envs\ghoddoosian_tf1\python.exe`. Versions: Python 3.6.13, numpy 1.14.6,
scipy 1.2.1, sklearn 0.19.2, tensorflow 1.8.0, matplotlib 2.2.5, cv2 3.4.5, dlib 19.17.0, imutils 0.5.4. All match.

## Data and checkpoints (steps 2-3)

Fold1-5 shapes match the table above (X `(N,30,4)`, y `(N,1)`, float64, label values {0, 5, 10}).
my_model1..5 each have `.meta`, `.index`, `.data-00000-of-00001` and hold 96 variables (~54k non-optimizer parameters,
matching the paper's "approximately 50,000"). The `checkpoint` state file is absent, which is harmless: Training.py
restores by explicit prefix.

Fold-number wiring in Training.py: with `load=True` fold `i` restores `my_model{i+1}`; with `load=False` it trains
fold `i` but saves `my_model{i}` (zero-based), so an unmodified training run would write `my_model0` and overwrite
`my_model1..4`. This is why training was always run through the in-memory wrapper with a redirected path.

## Pretrained 5-fold evaluation (steps 4-5)

Fold1 reproduces the earlier values exactly (max |diff| 2.8e-7). Files:
`reproduction_artifacts/pretrained_5fold_results.csv`, `pretrained_5fold_summary.txt`, `pretrained_eval_stdout.log`.

| Fold | BSA loss | BSA acc | VA | BSRE | VRE |
|---:|---:|---:|---:|---:|---:|
| 1 | 15.711236 | 0.483846 | 0.636364 | 3.185322 | 2.419539 |
| 2 | 10.573708 | 0.587861 | 0.611111 | 1.390765 | 1.039631 |
| 3 | 13.314880 | 0.560565 | 0.694444 | 2.153216 | 0.667389 |
| 4 | 11.508018 | 0.544460 | 0.638889 | 1.599456 | 0.849650 |
| 5 | 10.249504 | 0.528681 | 0.666667 | 1.176440 | 0.812155 |
| mean | 12.271469 | 0.541082 | 0.649495 | 1.901040 | 1.157673 |
| std (ddof=1) | 2.262427 | 0.038759 | 0.031912 | 0.804573 | 0.717788 |

## Comparison with the paper (step 6)

The repository README states no numeric results. Paper numbers below are from arXiv 1904.07312 (Table 2 / Table 3)
as returned by a fetch of the ar5iv HTML page, which was processed by a summarising model, not read verbatim.
**Verify against the PDF before citing.** The PDF itself could not be parsed locally (no PDF tooling installed).

| Metric | Paper result | Reproduced (pretrained, 5-fold mean) | Difference |
|---|---:|---:|---:|
| VA | 65.2% | 64.95% | -0.25 pt |
| BSA | 54% | 54.11% | +0.11 pt |
| BSRE | 1.90 | 1.901 | +0.001 |
| VRE | 1.14 | 1.158 | +0.018 |
| BSA loss | not explicitly reported | 12.27 | n/a |

Per fold (paper reports only VA and VRE per fold; BSA / BSRE / loss per fold: not explicitly reported):

| Fold | VA paper | VA repro | VRE paper | VRE repro |
|---:|---:|---:|---:|---:|
| 1 | 0.64 | 0.636 | 2.42 | 2.420 |
| 2 | 0.61 | 0.611 | 1.04 | 1.040 |
| 3 | 0.70 | 0.694 | 0.58 | 0.667 |
| 4 | 0.64 | 0.639 | 0.85 | 0.850 |
| 5 | 0.67 | 0.667 | 0.81 | 0.812 |

Everything agrees within rounding except Fold3 VRE (0.58 paper vs 0.667 here). The cause is not determined.

## Fold1 original settings and retraining (step 7)

Original Training.py settings (unchanged in the retraining run): epochs 80; batch size 64 (last partial batch is
kept, no shuffling between epochs); Adam, lr 0.000053; hidden sizes [32,32,32,32], 4 HM-LSTM layers; sequence length
30; 4 input features [Freq, Amp, Dur, Vel]; dropout `drop_out_p=1.0` (keep_prob 1.0 = no dropout); loss
`mean(max(0, |y-yhat|^2 - 1.253)) + 0.1 * sum(L2 of 6 weight groups)`; output `10*sigmoid`; Pre_fc1 32, Post_fc1 16
per layer, embeddings 16/16, Post_fc2 8; feature-wise z-score using training statistics; no random seed.

In-memory changes only: fold range 5->1, `ii=i` -> `ii=i+1`, save path and npy outputs redirected.
Output dir: `reproduction_artifacts/fold1_training/` (`my_model1.*`, `training_history.csv`, `training_stdout.log`,
`x1/loss1/accuracy1*.npy`). Wall time 20.8 min on CPU (about 16 s/epoch). Original `my_model1` sha256 unchanged.
`reproduction_artifacts/fold1_training_1epoch_check/` is a 1-epoch trial of the script, kept separate.

Note: Training.py saves the checkpoint only once, after the last epoch (the "saving every epoch" comment is
inaccurate), so the saved retrained checkpoint is the epoch-80 model.

## Comparison A / B / C on Fold1 (step 8)

| Metric | A. Paper | B. Provided pretrained | C. Retrained, epoch 80 | (C at epoch 13, best VA) |
|---|---:|---:|---:|---:|
| VA | 0.64 | 0.636 | **0.485** | 0.636 |
| BSA acc | not explicitly reported | 0.484 | 0.437 | 0.459 |
| BSA loss | not explicitly reported | 15.711 | 15.715 | 17.334 |
| BSRE | not explicitly reported | 3.185 | 2.800 | 2.231 |
| VRE | 2.42 | 2.420 | 1.449 | 1.579 |

Column "epoch 13" is only for illustration: picking an epoch by test VA is selection on the test set and is not a
valid performance estimate.

Training curve: training moving accuracy rises from 0.39 to 0.70 and training loss falls monotonically, while test
BSA peaks around epoch 20 (0.475) and then drifts down (last 20 epochs mean 0.435), and VA fluctuates between 0.45
and 0.64 (last 20 epochs mean 0.49). With only 33 test videos, one video changes VA by 3.0 points.
Regression errors are not uniformly worse: BSRE and VRE of the retrained model are lower than the pretrained one
even though VA/BSA are lower.

### Cause analysis (only one retraining run exists, so nothing below is proven)

- Random seed / weight initialisation / batch ordering: no seed is set (Xavier init unseeded). Batch order is fixed
  (no shuffling), so init is the only stochastic input. Run-to-run spread is not measured yet, so it cannot be
  separated from a systematic gap with n=1. This is the most likely explanation but is a hypothesis.
- Checkpoint selection: the final-epoch model is saved. The provided pretrained model may come from a different
  epoch or a selected run; Training.py itself says to "run each fold a few times to see the best performance".
  How the provided checkpoints were chosen is not documented, so this is unverified.
- CPU vs GPU, Windows vs Ubuntu, TF non-determinism: cannot be quantified without repeat runs; the pretrained model
  evaluates identically to the paper on the same CPU/Windows stack, so the evaluation path itself is sound.
- Preprocessing: the provided .npy files are used unchanged for both A/B/C, so preprocessing is not a factor here.
- Nothing in the code or hyper-parameters was altered to close the gap.

Suggested check (not run): 3-5 repeat Fold1 runs (about 21 min each) to obtain the run-to-run spread.

## Full five-fold training estimate (step 9)

Fold1 ended in a plausible-but-lower range than the pretrained model, so the go/no-go is the user's call.
Estimate if run sequentially on this CPU: about 21 min per fold, roughly 1.8 h for five folds (folds 2-5 have
similar training-set sizes, 6.5k-7.2k sequences). Storage about 2.4 MB per fold (12 MB total). Must use the
redirected-path wrapper (a `range(5)` generalisation of `reproduction_train_fold1.py`); the original script would
write my_model0..4 to the project root. **Not started.**

## blink_video.py static analysis (step 10)

Pipeline with line references (`blink_video.py`):

| Stage | Where |
|---|---|
| Video read + resize to width 450 | 348, 352, 359 |
| Grayscale + gamma 1.5 | 368, 370 (`adjust_gamma` 29-37) |
| dlib frontal face detector | created 311, applied 378 |
| dlib 68 landmarks | predictor 313, applied 386-387 |
| eye landmarks (left/right) | 318-319, sliced 415-416 |
| EAR per eye, averaged | `eye_aspect_ratio` 74-92, used 417-421 (clamped to 0.45) |
| 13-frame EAR window | `EAR_series` 330, updated by `shift` at 423 |
| No face: optical-flow eye tracking (LK) | 487-523 |
| SVM eye-closed classifier | loaded 315, `predict` on the 13-frame window 444 |
| Blink candidate tracking | `Blink_Tracker` 190-291 (frame of interest `EAR_series[6]`, line 447) |
| Noise/imbalance rejection, merging close blinks | 213-240, 244-283 (`MIN_AMPLITUDE=0.04` line 51) |
| 3-tap smoothing + extrema based blink extraction | 225, `Ultimate_Blink_Check` 128-186 |
| Feature file write | 461-464 and 546-548 (`np.savetxt`, append mode) |

Features (columns written: `TOTAL_BLINKS, Freq*100, Amp, Dur, Vel`; Preprocessing.py reads columns 1-4, column 0 is
ignored, and reorders to the model input order [Freq, Amp, Dur, Vel] at `Preprocessing.py:13-19`):

- Duration: `detected_blink.duration = end - start + 1` (line 179); candidate-level 248/267.
- Amplitude: `0.5*(startEAR-peakEAR) + 0.5*(endEAR-peakEAR)` (line 180); candidate-level 247/266.
- Eye opening velocity: `(endEAR - selected_EAR_First[2j+1]) / (end - selected_index_First[2j+1] + 1)` (line 181).
- Frequency: `BLINK_FRAME_FREQ = TOTAL_BLINKS / number_of_frames` (lines 455, 541), written x100; `TOTAL_BLINKS` updated at 229;
  `number_of_frames` counts only frames where a face is detected (line 380).

Observations (not fixed, source untouched): (1) the face-detected branch drops blinks with `velocity <= 0` (line 461)
but the optical-flow branch (541-548) does not; (2) in that optical-flow branch the `Blink_Tracker` call sits inside
`if Counter4blinks==0:` (531-536), unlike the face branch (445-450); (3) yawning/MAR code (94-107, 389-407) runs but
does not feed the 4 features; (4) `np.int` at line 499 works only with NumPy < 1.24.

## Single-video preprocessing test preparation (step 11)

- `shape_predictor_68_face_landmarks.dat` (99,693,937 bytes) and `Trained_SVM_C=1000_gamma=0.1_for 7kNegSample.sav`
  both load. SVM: SVC, C=1000, gamma=0.1, rbf, 13 input features, classes {-1, +1}; a flat 0.30 EAR window
  predicts open (-1), a flat 0.10 window predicts closed (+1). The pre-0.18 pickle warning still applies.
- Input/output variables are hard-coded at `blink_video.py:579` (`output_file`, opened in append mode) and `:580`
  (`path = ''`). The wrapper replaces only those two lines in memory.
- Run (from the project root, env activated):
  `python reproduction_blink_one_video.py "<path\to\one_video.mp4>"`
  Output goes to `reproduction_artifacts/blink_test/<video>_blinks.txt`; it refuses to append to an existing file.
  It opens Tk/OpenCV windows; press any key in the "Frame" window to stop. Not executed: **no UTA-RLDD video is on
  this machine** (only the SUST Driver Drowsiness Dataset was found), and `blink_video.py` was tuned for UTA-RLDD.

## Source modification policy

No original file was modified (git shows no modified tracked files; hash checks pass). No backups or diffs needed.
Known typo `self.hstate_size_size` in HMSLSTM_cell.py is unreachable in the tested paths and left as is.


---

# Repository consolidation log (new portfolio repository)

## Stage 0 - read-only checks (2026-09-30)
- OLD HEAD = 2bafc985c46526fa0dfe3a93bb96fcee1d8d7ac4, `status -sb` = `master...origin/master` (no ahead/behind).
- OLD tracked files: 44 (.py 6, .npy 20, my_model* 15, LICENSE, README.md, Trained_SVM_C=1000_gamma=0.1_for 7kNegSample.sav).
- NEW: git repository, user.name/user.email set, origin = https://github.com/a38200-pixel/early-drowsiness-repro.git.
- OLD `core.autocrlf` = true (system gitconfig). The 6 tracked .py files have identical CR counts in blob and working tree (no difference).

## Stage 1 - skeleton and vendoring
- Commit "Add repository skeleton": .gitattributes (`third_party/** -text`), .gitkeep placeholders.
- 44 blobs written byte-for-byte via `git cat-file blob HEAD:<name>` into third_party/ghoddoosian/.
  `git hash-object --no-filters` == OLD `rev-parse HEAD:<name>` for 44/44 files. The dlib .dat, .vscode and __pycache__ are not included.
- configs/protected_manifest.sha256 generated (44 entries, `sha256sum -c` all OK).
- Note: the first vendoring attempt failed before writing anything (`capture_output` is unavailable in the Python 3.6 env); fixed the helper script and re-ran.

## Stage 2 - scripts and records moved
- 6 scripts copied to src/repro/ (CRLF normalised to LF). Only path constants / env-var handling / manifest-based integrity check changed; see the diffs in the hand-off report.
- reproduction_eval.py writes its regression re-run to experiments/00_reproduction/regression_eval/ so the archived original results are not overwritten.
- 9 text artifacts (.csv/.log/.txt) copied from OLD reproduction_artifacts (77,293 bytes total); checkpoints and .npy files excluded.
- This log was copied from the old working clone with its absolute project paths replaced by `<PROJECT_ROOT>` and the local user name replaced by `<USER>`.

## Stage 3 - regression test in the new structure (no training)
- Command: `python src/repro/reproduction_eval.py 5` in the `ghoddoosian_tf1` env (pretrained my_model1..5, load=True, evaluation only).
- Integrity: 44 protected files checked, changed: none, added: none. third_party/ still 44 files, no __pycache__.
- Results (rounded to the reference-table precision) - all 25 values match:

| Fold | BSA loss | BSA acc | VA | BSRE | VRE |
|---|---|---|---|---|---|
| 1 | 15.711 | 0.4838 | 0.6364 | 3.185 | 2.420 |
| 2 | 10.574 | 0.5879 | 0.6111 | 1.391 | 1.040 |
| 3 | 13.315 | 0.5606 | 0.6944 | 2.153 | 0.667 |
| 4 | 11.508 | 0.5445 | 0.6389 | 1.599 | 0.850 |
| 5 | 10.250 | 0.5287 | 0.6667 | 1.176 | 0.812 |

- Outputs saved to experiments/00_reproduction/regression_eval/ (CSV, summary, stdout log). The CSV is identical (6 decimals) to the archived
  original pretrained_5fold_results.csv; the summary differs only in "41 hashed" (old glob) vs "44 hashed" (manifest).
- Stderr only contained the TensorFlow AVX2 CPU-feature notice (harmless).

## Stage 4 - documents and configuration
- Added LICENSE (MIT, copyright holder left as `<YOUR NAME>`), THIRD_PARTY.md, .gitignore, data/README.md, README.md,
  configs/image_publishable.csv (36 subject_ids), src/repro/publish_guard.py, envs/environment_tf1.yml
  (`conda env export -n ghoddoosian_tf1 --no-builds`, prefix line removed).
- Checks: image_publishable.csv has 36 rows; publish_guard raises for non-listed subjects (nothing written) and saves for listed ones;
  .gitignore leaves third_party/ tracked (44 files) and ignores foo.png, *.dat, *.mp4, new my_model* outside third_party, config.local.yaml, .vscode/.
- New-structure runs: check_environment.py -> ENVIRONMENT_OK, check_dataset.py -> DATASET_OK;
  check_video_assets.py without DLIB_LANDMARK_PATH exits with a message naming that variable (video/dlib assets not run: no data in the repo).
- README reports only the confirmed regression table; Fold1 retraining is stated as TBD; BibTeX left as a TODO block.

## Stage 5 - privacy / size check
- Searched all tracked files (including binaries) for the local user name, `C:\Users` paths, e-mail-like strings and token/key patterns:
  no hits in third_party/; no hits in the rest except one self-referencing line in this log, which was rewritten with placeholders.

# Fold1 다중 시드 재학습 준비 (2026-09-30)

## 작성한 파일
- `src/repro/common.py`: sha256_of_file, check_manifest(해시 44개 + third_party에 목록 밖 파일이 없는지), get_git_info(커밋, 추적 파일 dirty 여부). 기존 스크립트는 import하지 않음.
- `src/repro/train_fold1_seeds.py`: `<시드> [epoch수=80] [출력폴더명=seed<시드>]`. 출력은 `experiments/01_fold1_seeds/<이름>/`이며, 폴더가 이미 있으면 종료.
  원본 Training.py는 메모리에서만 치환한다.
  - `tf.reset_default_graph()` 다음 줄에 `tf.set_random_seed(<시드>)`를 넣음(Train()이 매번 그래프를 새로 만들기 때문).
  - `variable_path`와 `np.save(open('./...'))` 6곳을 출력 폴더로 돌림.
  - Fold1만(`range(1)`, `ii=i+1`), epoch 수 치환.
  - epoch 기록 호출(`_on_epoch_end`)을 추가. 계산에는 영향 없음.
  - random.seed와 np.random.seed도 설정. 하이퍼파라미터는 그대로.
  - metrics.csv는 epoch마다 flush. 열은 `epoch,train_loss,train_acc,BSA_loss,BSA_acc,VA,BSRE,VRE`이며, 기존 training_history.csv와 순서와 의미가 같음.
  - train.log(원본 print 포함)와 run_meta.json을 남기고, 시작과 종료 시 check_manifest를 실행.
- `src/repro/summarize_fold1_seeds.py`: `seed<숫자>` 폴더의 마지막 행으로 표와 통계(평균, ddof=1 표준편차, 최소, 최대)를 만듦.
  - epoch가 80이 아니면 미완료로 보고 통계에서 제외.
  - 공개 체크포인트 Fold1 값은 regression_eval CSV에서 읽음.
  - VA와 BSA acc는 "시드 N개보다 높다"와 범위 안/밖만 출력하고, 범위 밖이면 "결론 보류, 추가 시드 필요"로 적음.
- `scripts/run_fold1_seeds.bat`:
  - Python 3.6 확인, 예상 시간과 공간 출력, 시드 1~5 순차 실행.
  - 이미 있는 seed 폴더는 건너뛰고, 실패 시 종료 코드 1로 중단.
- `.gitignore`에 `experiments/01_fold1_seeds/*/my_model*`와 `experiments/01_fold1_seeds/smoke_*/`를 추가.

## 스모크 테스트 (2 epoch, 학습 소요 약 0.8~0.9분/회)

| 실행 | 시드 | epoch 1 train_loss | epoch 2 train_loss | epoch 2 BSA acc | epoch 2 VA |
|---|---|---|---|---|---|
| smoke_a1 | 1 | 2323.292410 | 2093.058209 | 0.382979 | 0.333333 |
| smoke_a2 | 1 | 2323.292410 | 2093.058209 | 0.382979 | 0.333333 |
| smoke_b | 2 | 2388.162127 | 2187.231386 | 0.423956 | 0.363636 |

- a1과 a2는 두 epoch 모두 train_loss 차이가 0이고, metrics.csv의 모든 값이 문자열까지 같음(1e-6 기준 일치).
- a1과 b는 epoch 1 train_loss가 다름. 시드가 적용됨.
- third_party: 매니페스트 해시 44/44 일치, 파일 수 44, `__pycache__` 없음(시작/종료 시 check_manifest 통과).
- 스모크 폴더는 삭제하지 않음(gitignore 대상).
- 요약 스크립트는 스크래치패드의 가짜 데이터(완료 5, 미완료 1, csv 없음 1, smoke 1)로 제외 규칙, 통계, 범위 판정을 확인함. 저장소에는 summary.md를 만들지 않음.
- 배치 파일은 가짜 저장소 구조로 건너뛰기, 실패 시 중단(종료 코드 1), Python 3.6 외 환경 거부(종료 코드 1), 모두 건너뛴 정상 종료(0)를 확인함.
  - 처음 버전은 for 블록 안의 `exit /b 1`이 종료 코드로 전달되지 않아(0) 루프 밖 `:failed` 레이블로 옮겨 고침.
- 참고: `.gitattributes`의 `* text=auto`로 `configs/protected_manifest.sha256`이 CRLF로 체크아웃되어, THIRD_PARTY.md의 `sha256sum -c` 명령이 Windows에서는 파일명 끝 `\r` 때문에 실패함.
  CR 제거 후 검증하면 44/44 OK. Python 검사(common.check_manifest)는 영향 없음.
- 80 epoch 학습과 배치 실행은 하지 않음(사용자가 직접 실행).

## 후속 수정 (2026-09-30)
- metrics.csv 열 이름을 기존 training_history.csv에 맞춤: `train_loss` -> `train_loss_sum`, `train_acc` -> `train_moving_acc`
  (나머지 열은 그대로). 두 스크립트 상단 주석에 열마다 의미를 한 줄씩 적음.
- 스모크 테스트 `1 2 smoke_c`: 종료 코드 0, 머리줄 `epoch,train_loss_sum,train_moving_acc,BSA_loss,BSA_acc,VA,BSRE,VRE`.
  값은 smoke_a1과 모두 같음(시드 1 재현성 재확인). 요약 스크립트는 이 CSV를 읽어 '미완료(epoch 2)'로 정상 표시함(스크래치패드에서 실행, 저장소에는 summary.md 만들지 않음).
  third_party는 파일 44개, `__pycache__` 없음.
- `.gitattributes`에 `configs/*.sha256 text eol=lf`를 추가하고 `git add --renormalize .`를 실행함. 인덱스 변화는 없었음(blob은 원래 LF).
  작업 폴더의 매니페스트를 지우고 `git checkout`으로 다시 받자 `w/lf`가 됨. THIRD_PARTY.md의 두 검증법을 확인함.
  - `sha256sum -c`: 44 OK, 실패 0.
  - blob 해시 대조 루프: 로컬 원본 clone을 upstream으로 사용. MISMATCH 0, ls-files 44.

# Fold2~5 재학습 스크립트 준비 (2026-10-01)

## 작성한 파일
- `src/repro/train_fold_seeds.py`: `train_fold1_seeds.py`를 복사해 만든 일반화 버전(원본 파일은 수정하지 않음).
  사용법은 `<fold 1~5> <시드> [epoch수=80] [출력폴더명=fold<fold>_seed<시드>]`이고, 출력은 `experiments/02_fold_seeds/<이름>/`이며 폴더가 이미 있으면 종료.
  - fold 반복부를 `range(fold-1, fold)`로 치환. 원본이 반복 변수 i로 입력 파일(`Fold<i+1>`)과 `start_indices`(`if i==0` ...)를 고르므로 입력과 start_indices가 해당 fold에 맞게 선택됨.
  - `ii=i+1`로 `my_model<fold>` 이름 저장, 저장 경로는 출력 폴더로 리다이렉트(기존 방식).
  - metrics.csv 열과 run_meta.json은 기존과 같고 run_meta.json에 `fold`만 추가. 입력 .npy 해시도 해당 fold 파일 기준.
- `scripts/run_folds_seed.bat <시드>`: fold 2,3,4,5 순서 실행. 폴더가 이미 있으면 건너뛰고 실패 시 종료 코드 1로 중단. 시작 시 예상 시간(fold당 약 21분, 총 약 1.4시간)과 공간 출력.
- `src/repro/summarize_folds.py <시드>`: Fold1(`01_fold1_seeds/seed<S>`)과 Fold2~5(`02_fold_seeds/fold<F>_seed<S>`)를 합쳐 재학습 표, 공개 체크포인트 표, 차이(재학습 - 공개) 표를 만들고 `02_fold_seeds/summary_seed<S>.md`로 저장.
  마지막 epoch가 80이 아니면 미완료로 표시하고 평균에서 제외(공개 값과 차이의 평균도 같은 fold 집합으로 계산). 결론 문구 없이 숫자만 출력.
- `.gitignore`에 `experiments/02_fold_seeds/*/my_model*`와 `experiments/02_fold_seeds/smoke_*/` 추가.

## 검증 (학습은 2 epoch 스모크만 실행, 80 epoch 학습은 하지 않음)
- `1 1 2 smoke_f1`: epoch 1,2의 train_loss_sum이 2323.292410, 2093.058209로 `01_fold1_seeds/seed1/metrics.csv`와 같음(epoch 1~2의 모든 열이 동일).
- `2 1 2 smoke_f2`: 오류 없이 종료. 체크포인트가 `my_model2.*`, run_meta.json의 fold=2, 입력 해시는 Fold2 파일 기준. epoch 2: BSA loss 19.896252, VA 0.305556.
- third_party: 매니페스트 해시 44/44 일치, 파일 수 44, `__pycache__` 없음(시작/종료 시 check_manifest 통과).
- 중간에 `upstream_print`의 컴프리헨션을 for 루프로 풀어 쓴 뒤 `1 1 2 smoke_f3`를 다시 실행했고 seed1과 epoch 1~2의 모든 열이 동일.
- 요약 스크립트는 스크래치패드의 가짜 데이터(완료 3, 미완료 1, 없음 1)로 제외 규칙, 평균, 차이 표를 확인함. 저장소에는 summary 파일을 만들지 않음.
- 배치 파일은 가짜 저장소로 인자 없음, Python 3.6 아님, 건너뛰기 후 실패 중단(종료 코드 1), 모두 건너뜀(종료 코드 0)을 확인함.
- 코드 규칙: 상단 3줄 주석과 모듈 docstring, 모든 함수 docstring, 컴프리헨션/람다 없음, 가장 긴 함수 main 31줄, 3.6 문법만 사용.
- 스모크 폴더(smoke_f1, f2, f3)는 삭제하지 않고 남김(gitignore 대상).

## 정정 (2026-10-01)

위쪽 기존 줄은 수정하지 않고, 아래 두 서술을 정정한다.

- **279행** "Everything agrees within rounding except Fold3 VRE (0.58 paper vs 0.667 here)":
  소수 둘째 자리 반올림 기준으로는 Fold3 VRE(논문 0.58, 측정 0.67)만 다른 것이 아니다. Fold3 VA(논문 0.70, 측정 0.694444 → 0.69)와
  5-fold 평균 VRE(논문 1.14, 측정 1.157673 → 1.16)도 논문과 다르다. 논문과 비교 가능한 14칸 중 11칸이 일치하고 3칸이 불일치한다.
  원인은 확인되지 않았다.
- **418행** "all 25 values match": 비교 대상은 논문이 아니라 이 프로젝트의 이전 실행 결과(245-253행 표)를 표 표기 자릿수로 반올림한 값이다.
  새 폴더 구조에서 다시 실행했을 때 이전 실행과 같은 값이 나오는지 보는 자체 일관성 확인이며, 논문 재현을 뜻하지 않는다.
- 상세 대조는 `docs/IMPLEMENTATION_DIFF.md`의 5절에 있다.

# blink 무화면 실행 검증 (2026-10-02)

- `reproduction_blink_headless.py`로 Fold2_part1/17/0.mp4(클래스 0) 1개를 실행: 소요 755.9초(약 12분 36초), 출력 58줄(첫 빈 줄 1 + blink 57행)이 GUI 기준 출력과 `--compare`로 전 줄 일치(종료 코드 0).

# 회전 조사 사전 등록 (2026-10-02)

이 항목은 조사 스크립트 `src/repro/survey_orientation.py`를 **실행하기 전에** 쓴다. 결과는 날짜가 붙은 별도 항목으로 추가하고 이 항목은 고치지 않는다.

- 배경 1: 배치 시험에서 Fold1_part2/10/0.MOV는 종료 코드 0이면서 blink 0행(출력 파일 없음)이었고 367.2초가 걸렸다. [확실] (사용자 보고)
- 배경 2: 같은 영상은 육안으로 세로 촬영이고 저장 크기는 568×320이다. [확실] (사용자 육안 확인). Fold4_part2/45/10.mp4는 육안으로 상하 반전이다. [확실] (`configs/video_manual_flags.csv`)
- 가설: 얼굴 검출 0인 일부 영상은 회전 태그 때문에 누운 프레임으로 읽힌다. OpenCV 3.4.5는 회전 태그를 적용하지 않는다. [추측]
- 조사 방법: 영상마다 5%~95% 구간을 고르게 나눈 8개 프레임 위치만 읽고(전체 디코딩 안 함, 프레임·이미지는 저장하지 않음), 원본 상태에서 0°, 시계 90°, 180°, 반시계 90°로 회전한 뒤 `blink_video.py`와 같은 전처리(imutils.resize width=450, 그레이스케일, 감마 1.5, `detector(gray, 0)`)로 dlib 정면 얼굴 검출기의 얼굴 수를 센다. 회전은 resize 전에 적용한다. [확실] (방법 정의)
- best_orientation 정의: 얼굴 수가 가장 많은 방향. 전부 0이면 `none`, 1위가 2위의 2배 미만이면 `ambiguous`. [확실] (방법 정의)
- 예측 (a): Fold1_part2/10/0.MOV는 best가 `cw90` 또는 `ccw90`이다. [추측]
- 예측 (b): Fold4_part2/45/10.mp4는 best가 `180`이다. [추측]
- 예측 (c): Fold2_part1/17/0.mp4(대조)는 best가 `0`이다. [추측]
- 판정 규칙: (a)(b)(c) 중 하나라도 빗나가면(`none`과 `ambiguous`도 빗나감으로 센다) 이 방법이 틀린 것으로 보고 가설을 기각한다. [확실] (사전 정의)

# 배치 시험과 회전 조사 결과 (2026-10-02)

## 1. 배치 시험 2건

- Fold2_part1/17/0.mp4: 793.0초, blink 57행, GUI 기준 출력과 `--compare`로 58줄(첫 빈 줄 1 + blink 57행) 완전 동일. [확실] (사용자 보고). 위 "blink 무화면 실행 검증" 항목의 755.9초는 단독 실행이고, 793.0초는 배치 시험의 값이다. [확실]
- Fold1_part2/10/0.MOV: 367.2초, 종료 코드 0, 출력 파일 없음(blink 0). [확실] (사용자 보고)

## 2. 회전 조사 (`src/repro/survey_orientation.py`, 결과 `configs/video_orientation_survey.csv`)

- 실행: `--only` 3개 3.6초, `--subset zero` 84개 179.2초, 전체 182개 560.2초(약 9.3분). 전체 실행에서 가장 오래 걸린 영상은 Fold3_part2/34/5.mov 14.2초, Fold4_part1/37/0.mov 13.9초, Fold4_part1/37/10.mov 12.0초. 읽은 프레임이 8개 미만이거나 예외가 난 영상은 없었다. [확실] (측정)
- 예측 (a) Fold1_part2/10/0.MOV: **적중**. best=`cw90` (얼굴 수 0°/cw90/180°/ccw90 = 0/8/0/4). [확실] (측정)
- 예측 (b) Fold4_part2/45/10.mp4: **적중**. best=`180` (0/0/8/0). [확실] (측정)
- 예측 (c) Fold2_part1/17/0.mp4(대조): **빗나감**. best=`ambiguous` (8/0/6/0). 사전 규칙에서 `ambiguous`는 빗나감이다. [확실] (측정, 사전 규칙)
- 판정: 예측 (c)가 빗나갔으므로 사전 등록한 규칙에 따라 가설("얼굴 검출 0인 일부 영상은 회전 태그 때문에 누운 프레임으로 읽힌다")을 **기각**한다. [확실] (사전 규칙 적용)
- 방향별 영상 수(182개): `0` 80, `cw90` 37, `180` 4, `ccw90` 32, `ambiguous` 26, `none` 3. [확실] (측정)
- inventory의 `faces_orig==0` 영상 84개의 분포: `cw90` 37, `ccw90` 32, `180` 3, `ambiguous` 8, `none` 3, `0` 1(Fold2_part2/23/5.mp4, 0° 검출 1프레임). `faces_orig>0` 영상 98개의 분포: `0` 79, `ambiguous` 18, `180` 1. [확실] (측정)
- `none` 3개(어느 방향에서도 얼굴 수 0): Fold2_part2/23/0.mp4, Fold2_part2/23/10.mp4, Fold5_part1/49/0.mp4. [확실] (측정). 이 3개는 회전으로 설명되지 않는다. 원인은 미검증이다.
- `ambiguous` 26개는 두 종류다. [확실] (측정)
  - 0°에서 8프레임 모두 검출되고 180°에서도 5~8프레임 검출됨(18개): Fold1_part2/09/10, Fold2_part1/17/0, 17/5, 17/10, 18/0, 18/10, Fold2_part2/20/0, Fold3_part2/32/0, 35/10, Fold4_part1/40/0, 41/0, 41/5, Fold4_part2/44/5, 46/5, Fold5_part1/51/0, 51/5, Fold5_part2/55/0, 55/5.
  - 0°에서 0이고 cw90과 ccw90 모두 검출됨(8개): Fold4_part2/47/0, Fold5_part1/52/5, Fold5_part2/56/0, 56/5, 57/5, 59/0, 59/5, 59/10.
- manual_flag 영상 8개의 best: Fold1_part1/06의 0·5·10과 Fold5_part2/55/10은 `0`, Fold4_part2/42/5는 `ccw90`, Fold4_part2/45의 0·5·10은 `180`. [확실] (측정). Fold1_part1/06의 3개는 0°에서 6~8프레임 검출되어 회전으로 설명되지 않으며, 재생기에서 화면이 표시되지 않은 원인은 미검증이다.
- 해석: 이 조사는 회전 태그를 읽지 않으므로 방향별 검출 결과가 "회전 태그 때문"임을 보이지 못한다. 원인은 미검증이다. [확실]
- 해석: `faces_orig==0` 영상 84개 중 80개가 0° 이외 방향에서 검출되어(72개는 한 방향이 뚜렷함), 이 영상들이 누운 프레임으로 읽힌다는 것과 측정은 일치한다. [가능성 높음]
- 해석: `blink_video.py`가 회전하지 않으므로(주석 처리, 361~366행) Fold1_part2/10/0.MOV의 blink 0행은 누운 프레임 때문일 수 있다. [가능성 높음]
- 해석: `ambiguous` 첫 번째 종류 18개는 0°에서 정상 검출되는 영상이므로 "1위가 2위의 2배 미만" 규칙 때문에 생긴 값이고, 이 규칙이 대조군을 `ambiguous`로 만들었다. [가능성 높음]. 규칙을 바꾸는 후속 조사는 별도 사전 등록이 필요하다.

# 정정 (2026-10-02)

위쪽 기존 줄은 수정하지 않는다. 아래 네 진술이 이 파일에 있는지 검색했다(검색어: 정규화, normaliz, 언급하지, blink 0, 제외, exclud, 파일이 없, blink 58, 58개, 140, 평균, per video, 영상당). 네 진술 모두 이 파일에서 발견되지 않았다. [확실] (검색)

- 진술 1 "논문은 2차 정규화를 언급하지 않는다": **해당 진술 없음**. 참고: 코드의 2차 정규화는 `Training.py` 420~431행이다. [확실] (직접 읽음). 논문도 개인별 정규화 뒤 개인 간 정규화(평균 0, 분산 1)를 서술한다. [가능성 높음] (근거: `docs/IMPLEMENTATION_DIFF.md` 158행이 인용한 기준값과 사용자 확인. 이 세션에서 논문 PDF를 열어 확인하지는 않아 미검증)
- 진술 2 "코드가 blink 0개 피험자를 제외한다": **해당 진술 없음**. 참고:
  - `Preprocessing.py`에는 피험자를 건너뛰는 코드(`continue`, `try/except`)가 없다. [확실] (직접 읽음)
  - 연결과 시퀀스 생성 변수(`alert_*`, `semi_*`, `sleepy_*`, 정규화 통계)는 `if txt_file == ...` 블록 안에서만 대입되고 반복문 앞에 초기화가 없다(`gen` 49~125행, `Preprocess` 161~239행). 연결은 폴더마다 항상 실행된다(127~128행, 242~243행). [확실] (직접 읽음)
  - 따라서 파일이 없으면 이전 반복의 값이 남아 조용히 재사용되고(첫 피험자이면 미정의 변수 오류) 이전 피험자의 시퀀스가 중복될 수 있다. [가능성 높음] (실행하지 않음)
  - blink가 0개인 파일을 읽는 경우 `unroll_in_time`의 29행 `out_data[0, -n:, :] = in_data`에서 n=0이면 `ValueError`가 난다. [확실] (numpy 1.14.6 메모리 시험). 다만 blink 0개 영상은 출력 파일이 만들어지지 않아(`blink_video.py` 462, 546행) 이 경로가 아니라 "파일 없음" 경로가 된다. [가능성 높음]. 저자가 실제 데이터 폴더를 어떻게 만들었는지는 미검증이다.
- 진술 3 "blink 58개": **해당 진술 없음**. 참고: 이 파일 535행은 이미 "출력 58줄(첫 빈 줄 1 + blink 57행)"로 적혀 있다. [확실]
- 진술 4 "영상당 평균 약 140 blink": **해당 진술 없음**. 참고: 저자 테스트 영상의 시퀀스 수는 1~166으로 넓게 퍼지고 긴 꼬리 분포로 보인다. [가능성 높음]. 알림(라벨 0) 영상 58개 중 16개가 시퀀스 1개이다. [확실] (측정: `Training.py` 444~498행의 `start_indices`와 `third_party/ghoddoosian/LabelsTest_30_Fold*.npy`의 라벨). 영상당 평균 blink 수로의 환산은 하지 않았다. 미검증

# 방향 판정 규칙 v2 사전 등록 (2026-10-02)

이 항목은 규칙 v2를 적용하는 코드(`src/repro/survey_orientation.py --resolve-only`)를 **실행하기 전에** 쓴다. 결과는 날짜가 붙은 별도 항목으로 추가하고 이 항목은 고치지 않는다.

- 배경: 앞의 회전 조사에서 예측 (c)가 빗나갔다. 대조군 Fold2_part1/17/0.mp4는 얼굴 수가 0°=8, 180°=6이라 v1 규칙(최댓값 방향, 1위가 2위의 2배 미만이면 `ambiguous`)으로는 `ambiguous`였다. [확실] (측정, 위 "배치 시험과 회전 조사 결과" 항목)
- 규칙 v2는 탐색적 규칙이다. v1 결과를 본 뒤에 만들었으므로(사후 설계) 아래 예측이 적중해도 독립적인 확인이 아니다. [확실]
- 규칙 v2 정의: 기존 `configs/video_orientation_survey.csv`의 8프레임 카운트만 사용하고 영상은 다시 읽지 않는다. [확실] (규칙 정의)
  1. `faces_0 >= 6`이면 `0`.
  2. 아니면 네 방향 중 최댓값이 5 이상이고 2위의 2배 이상이면 그 방향.
  3. 그 외는 `review`(육안 확인 대상).
- 예측: Fold1_part2/10/0.MOV는 `cw90`. [추측]
- 예측: Fold4_part2/45/10.mp4는 `180`. [추측]
- 예측: Fold2_part1/17/0.mp4는 `0`. [추측]
- 예측: Fold1_part1/06의 0·5·10 3개는 모두 `0`. [추측]
- 예측: v1에서 `ambiguous`였고 0°에서 정상 검출된 18개(`faces_0`가 읽은 프레임 수와 같은 영상)는 모두 `0`. [추측]
- 주: Fold2_part1/17/0.mp4와 위 18개는 `faces_0`가 8이므로 규칙 1에 의해 정의상 `0`이 된다. 이 두 예측은 거의 자동으로 성립한다. [확실] (규칙 정의와 기존 카운트로 도출)
- 적중 판정: `resolved_orientation`이 예측 값과 정확히 같으면 적중이다. [확실] (사전 정의)
- review 영상은 육안 확인 결과를 기록하는 파일(`configs/video_orientation_overrides.csv`, 작업 B에서 만들 예정)에 값이 들어가기 전까지 방향을 정하지 않는다. [확실] (사전 정의)

# 방향 판정 규칙 v2 결과 (2026-10-02)

- 실행: `src/repro/survey_orientation.py --resolve-only`. 영상은 읽지 않고 기존 `configs/video_orientation_survey.csv`의 카운트만 읽어 `resolved_orientation` 열을 추가했다(182행). [확실] (측정)
- 예측 적중: Fold1_part2/10/0.MOV `cw90` 적중, Fold4_part2/45/10.mp4 `180` 적중, Fold2_part1/17/0.mp4 `0` 적중, Fold1_part1/06의 0·5·10 `0` 3/3 적중, v1 `ambiguous`이고 0°에서 정상 검출된 18개 `0` 18/18 적중. 사전 등록한 예측은 모두 적중했다. [확실] (측정)
- 위 적중은 독립적인 확인이 아니다. 규칙 v2는 v1 결과를 본 뒤에 만든 사후 설계이고, 17/0·06의 3개·18개는 `faces_0`가 6 이상이라 규칙 1로 정의상 `0`이 된다. [확실] (사전 등록 항목의 주와 같음)
- `resolved_orientation` 별 영상 수(182개): `0` 97, `cw90` 37, `ccw90` 30, `180` 4, `review` 14. [확실] (측정)
- v1 대비: v1에서 방향이 정해졌는데 v2가 다른 값이 된 영상 3개(Fold2_part2/23/5.mp4는 0에서 `review`, Fold5_part1/49/10_2.mp4와 Fold5_part2/58/0.mp4는 `ccw90`에서 `review`. 모두 최댓값이 5 미만이라 방향을 정하지 않음). v1에서 `ambiguous`·`none`이었다가 방향이 정해진 영상 18개(모두 `ambiguous` 18개가 `0`으로). 합 21개. [확실] (측정)
- `review` 14개(faces 0°/cw90/180°/ccw90 카운트, v1 값): [확실] (측정)
  - Fold2_part2/23/0.mp4 (0/0/0/0, none), Fold2_part2/23/5.mp4 (1/0/0/0, 0), Fold2_part2/23/10.mp4 (0/0/0/0, none)
  - Fold4_part2/47/0.mp4 (0/6/0/8, ambiguous)
  - Fold5_part1/49/0.mp4 (0/0/0/0, none), Fold5_part1/49/10_2.mp4 (0/0/0/4, ccw90), Fold5_part1/52/5.MOV (0/8/0/5, ambiguous)
  - Fold5_part2/56/0.MOV (0/8/0/7, ambiguous), 56/5.MOV (0/8/0/5, ambiguous), 57/5.MOV (0/8/0/5, ambiguous), 58/0.mp4 (0/0/0/4, ccw90), 59/0.MOV (0/8/0/8, ambiguous), 59/5.MOV (0/8/0/8, ambiguous), 59/10.MOV (0/8/0/8, ambiguous)
- `review`는 세 종류다. 모든 방향에서 0(`none`)인 3개, cw90과 ccw90 모두 5프레임 이상 검출된 8개, 최댓값이 5 미만인 3개(23/5, 49/10_2, 58/0). [확실] (측정). 이 14개는 육안 확인 결과가 `configs/video_orientation_overrides.csv`에 기록되기 전까지 방향을 정하지 않는다. [확실] (사전 정의)

# 정정·보강: 회전 조사 "가설 기각"의 정리 (2026-10-02)

위쪽 "배치 시험과 회전 조사 결과" 항목의 "가설 기각" 서술은 고치지 않고, 아래처럼 정리한다.

- 예측 (c)의 실패는 가설이 틀렸다는 증거라기보다 조사 방법 v1(최댓값 규칙)의 신뢰도 문제를 보여 준다. Fold2_part1/17/0.mp4는 0°=8, 180°=6이었고, `ambiguous` 18개가 같은 양상(0°=8, 180°=5~8)이었다. [확실] (측정)
- Fold1_part2/10/0.MOV의 ccw90=4는 반대 방향에서의 오검출로 보인다. [가능성 높음] (cw90=8과 대비되는 측정값에서 도출. 프레임을 직접 보고 확인하지는 않아 미검증)
- 사전 규칙에 따른 판정(예측 하나가 빗나가면 가설 기각)은 기록으로 남아 있고 바꾸지 않는다. 다만 이 기각이 "누운 프레임" 설명 자체를 부정한다고 읽으면 안 된다. [가능성 높음]
- "회전 태그가 원인이다"라는 가설은 회전 태그를 읽지 않았으므로 **미검증**이다. [확실]
- 예측 (a)(b)와 `faces_orig==0` 영상 84개 중 cw90 37개 + ccw90 32개 = 69개라는 분포는 "일부 영상이 누운 프레임으로 읽힌다"는 설명을 지지한다. [가능성 높음] (측정에서 도출)
- 규칙 v2도 사후 설계라 독립적인 확인이 아니다. 방향 확정은 규칙 v2의 값과 육안 확인(`configs/video_orientation_overrides.csv`)을 함께 쓴다. [확실]

# 회전 옵션 시험 사전 등록 (2026-10-02)

이 항목은 사용자가 아래 시험 3개를 실행하기 **전에** 쓴다. 결과는 날짜가 붙은 별도 항목으로 추가하고 이 항목은 고치지 않는다.

- 변경 내용(영상 실행 전): `reproduction_blink_headless.py`에 `--rotate none|cw90|ccw90|180`(기본 `none`)과 `--dry-run`을 추가했다. `none`이면 이전 버전과 같은 6개 치환만 하며, 메모리 소스가 이전 6개 치환 결과와 바이트 단위로 같음을 확인했다. [확실] (측정, 메모리 비교)
- 변경 내용: `--rotate`가 `none`이 아니면 `blink_video.py`의 `frame = imutils.resize(frame, width=450)` 한 곳(정확히 1회 일치해야 함)을 "원본 프레임을 회전한 뒤 같은 resize(width=450)"로 메모리에서 바꾼다. 원본 파일은 수정하지 않는다. 이것은 저자 코드의 resize 표현식 1곳을 메모리에서 치환하는 의도적 변경이다. [확실]
- 변경 내용: 방향 이름과 회전 함수를 `src/repro/orientation_common.py` 한 곳에 두고 조사 스크립트와 `--rotate`가 함께 쓴다. 회전 표현식, 공통 함수, `np.rot90` 기대값이 네 방향 모두 일치함을 확인했다. [확실] (측정, 합성 배열)
- 변경 내용: `extract_blinks_batch.py`는 영상별 회전으로 override(`configs/video_orientation_overrides.csv`, 현재 헤더만 있음)가 있으면 그 값, 없으면 `resolved_orientation`을 쓰고, review이면서 override가 없는 영상은 처리하지 않는다. 자식에는 `--rotate`를 넘기고 매니페스트에 `rotate`, `face_frames` 열을 추가했다. 기존 매니페스트의 열 이름이 다르면 이어 쓰지 않고 중단한다. [확실]
- 검증(영상은 처리하지 않음): `py_compile` 통과, `--dry-run` 4종(치환 대상 모두 기대 횟수와 일치), 가짜 자식 스크립트로 `--rotate` 전달·새 매니페스트 열·`face_frames` 파싱·헤더 불일치 중단 확인. [확실] (측정)
- `face_frames`는 `blink_video.py`가 영상 끝에서 `not grabbed` 다음 줄에 찍는 `number_of_frames`(얼굴이 검출된 프레임 수)이다. [확실] (코드 읽기: `blink_video.py` 354~355행, 380행). 분모로 쓰는 `frame_count`는 `configs/video_inventory.csv`의 값(OpenCV `CAP_PROP_FRAME_COUNT`)이며 실제로 디코딩되는 프레임 수와 다를 수 있다. [가능성 높음]
- 시험 (i): Fold1_part2/10/0.MOV를 `--rotate cw90`으로 실행. 예측: blink 행 수 1 이상, `face_frames`가 `frame_count`(16309)의 50% 이상(8155 이상). [추측] 임계값 50%는 근거 없이 정한 값이다. [추측]. 이전(회전 없음)에는 종료 코드 0, blink 0행, 출력 파일 없음이었다. [확실] (사용자 보고). 이전 `face_frames` 값은 보고되지 않아 미검증이다.
- 시험 (ii): Fold4_part2/45/10.mp4를 `--rotate 180`으로 실행. 예측: `face_frames`가 `frame_count`(9250)의 50% 이상(4625 이상). [추측] (임계값은 같은 이유로 추측)
- 시험 (iii): Fold2_part1/17/0.mp4를 `--rotate none`으로 실행. 예측: `--compare`로 GUI 기준 출력과 완전 동일(회귀 확인). [추측]
- 사용자가 실행할 명령(저장소 루트, `conda activate ghoddoosian_tf1` 후): [확실] (명령 정의)
  - (i) `python src\repro\reproduction_blink_headless.py Fold1_part2/10/0.MOV "%TEMP%\blink_rot_cw90_f1p2_s10_c0.txt" --rotate cw90`
  - (ii) `python src\repro\reproduction_blink_headless.py Fold4_part2/45/10.mp4 "%TEMP%\blink_rot_180_f4p2_s45_c10.txt" --rotate 180`
  - (iii) `python src\repro\reproduction_blink_headless.py Fold2_part1/17/0.mp4 "%TEMP%\blink_rot_none_f2p1_s17_c0.txt" --rotate none`, 이어서 `python src\repro\reproduction_blink_headless.py --compare "%TEMP%\blink_rot_none_f2p1_s17_c0.txt" <GUI 기준 출력 경로>`
- 판정 규칙: 예측이 하나라도 빗나가면 그대로 기록하고, 임계값이나 코드를 바꿔 맞추지 않는다. blink 행 수 1 이상이나 `face_frames` 50% 이상은 추출 결과가 올바르다는 뜻이 아니라 "얼굴이 검출되는 프레임이 있다"는 뜻에 그친다. [확실] (사전 정의)

# 방향 육안 판정 도구 추가 (2026-10-02)

- `src/repro/review_orientation.py`를 추가했다. 대상은 `configs/video_orientation_survey.csv`에서 `resolved_orientation`이 `review`이고 `configs/video_orientation_overrides.csv`에 아직 기록이 없는 영상이다(`--only <rel_path>`로 영상 1개도 가능). [확실]
- 동작: 영상마다 25%/50%/75% 위치의 프레임만 읽어(OpenCV는 회전 태그를 적용하지 않은 원본 프레임) 4방향(0, cw90, ccw90, 180)으로 돌린 모습을 2x2로 한 창에 표시한다. 칸의 높이는 최대 360px이고 회전 함수는 `orientation_common.py`의 정의를 그대로 쓴다. 키는 0/1/2/3(방향 선택), 스페이스(다음 프레임 위치), s(건너뛰기), q(종료)이다. 선택하면 터미널에서 메모를 묻고 overrides CSV에 한 줄(rel_path, orientation, 관찰자 "사용자 육안", 날짜, 메모)을 추가한다. 이미 기록된 영상은 건너뛰고, q로 끝낸 뒤 다시 실행하면 이어서 한다. [확실] (코드 읽기)
- 이 도구는 blink를 추출하지 않고, 영상 전체를 디코딩하지 않으며, 프레임이나 이미지를 파일로 저장하지 않는다(화면에 표시만 한다). [확실] (코드 읽기: 저장 호출 없음)
- 설계 선택: 조사 CSV의 얼굴 수는 화면에 보이지 않는다. 판정이 그 수치에 끌리지 않게 하려는 것이다. 얼굴이 어느 칸에서도 보이지 않으면 사용자가 0을 고르고 메모에 "얼굴 안 보임"을 남긴다. 이 경우 기록된 `0`은 "방향을 모름"이 아니라 "회전 없음"이므로 그 영상은 회전 없이 추출된다. [확실] (설계). 그 영상의 추출 결과가 0행일 수 있다. [가능성 높음]
- 검증(에이전트): `py_compile` 통과. 합성 프레임으로 캔버스의 칸 높이가 360 이하이고 원본 프레임이 바뀌지 않으며 방향별 칸 크기가 맞음을 확인했다. 임시 폴더에서 CSV 헤더 생성, 줄바꿈 없는 파일에 이어 쓰기, 쉼표가 든 한글 메모 인용을 확인하고 `extract_blinks_batch.py`의 `load_overrides`로 다시 읽어 값이 같음을 확인했다. 실제 영상 1개에서 프레임 3장을 읽는 동작을 확인했다. 현재 대상은 14개이다. [확실] (측정)
- 창 표시와 키 입력 동작은 에이전트가 시험하지 못했다. 미검증
- 판정한 영상 수: 미실행. 사용자가 실행한 뒤 결과를 날짜가 붙은 별도 항목으로 추가한다. 미검증

# 회전 옵션 시험 결과 (2026-10-02)

위 "회전 옵션 시험 사전 등록" 항목의 시험 3건 결과이다. 예측과 판정 규칙은 그 항목을 그대로 적용했다. `face_frames`와 소요 시간은 사용자가 보고한 콘솔 출력 값이고, blink 행 수와 비교 결과는 결과 파일에서 이 세션이 다시 확인했다. [확실]

- (i) Fold1_part2/10/0.MOV `--rotate cw90`: `face_frames` 16302(`frame_count` 16309, 99.96%), blink 19행, 소요 876.7초. 기준(`face_frames` 8155 이상, blink 1행 이상) **통과**. 회전 없는 실행은 blink 0행(출력 파일 없음), 367.2초였다(배치 시험 기록). [확실]
- (ii) Fold4_part2/45/10.mp4 `--rotate 180`: `face_frames` 9236(`frame_count` 9250, 99.85%), blink 244행, 소요 199.3초. 기준(`face_frames` 4625 이상) **통과**. 콘솔 마지막 줄의 244는 처음에 어느 창의 출력인지 추정이었으나, 결과 파일이 245줄(첫 빈 줄 1 + blink 244행)이라 이 실행의 값임을 확인했다. [확실]
- (iii) Fold2_part1/17/0.mp4 `--rotate none`: blink 57행(파일 58줄), 소요 991.6초, GUI 기준 출력과 58줄 완전 동일(`--compare`, 종료 코드 0). 이 세션에서 다시 비교해도 같았다. 회귀 시험 **통과**. [확실]
- 재현성: (i)을 같은 조건으로 두 번 실행(run1, run2)해 `--compare`로 비교했고, 두 출력이 20줄(blink 19행 + 첫 빈 줄) 모두 완전히 같았다. 이 영상에서 회전 경로는 결정적이다. [확실] (영상 1개 기준). run1에서는 콘솔 출력(`face_frames`, 소요 시간)을 잃어버려 run2의 값을 위 (i) 결과에 적었다. [확실]
- 소요 시간에 관한 사용자 보고: 세 실행을 동시에 돌렸으므로 소요 시간은 단독 속도가 아니라고 했다. 단독 속도는 약 23 fps(18121프레임 793초, 배치 시험. 단독 헤드리스 실행 755.9초는 약 24 fps)이고, (i)은 약 18.6 fps(16309/876.7), (iii)은 약 18.3 fps(18121/991.6)로 단독의 약 80%였다. [확실] (계산)
- 위 "동시 실행" 설명은 결과 파일의 시각과 맞지 않는다. 생성~마지막 기록 시각으로 본 실행 구간은 (i) run1 16:31~16:45, (ii) 16:55~16:58, (iii) 17:02~17:18, (i) run2 17:23~17:37로 서로 겹치지 않는다. [확실] (파일 시각). 따라서 느려진 원인이 동시 실행인지는 미검증이다. 다른 프로세스가 CPU를 나눠 썼을 가능성은 [추측]. 소요 시간은 단독 속도 기준으로 해석하지 않는다. [확실]
- (i)과 (iii)은 영상(해상도)이 달라 위 fps 비교는 대략적이다. [가능성 높음]. (ii)는 해상도가 작은 영상(480×270)이라 약 46 fps(9250/199.3)로 빨랐고, 80% 비교에서는 뺐다. [가능성 높음]
- 판정: 회전 옵션과 방향 정의(cw90, 180)는 사전 등록 기준을 통과했다. [확실]. 기준은 "blink 행이 나오고 얼굴이 검출되는 프레임이 대부분이다"를 뜻할 뿐 추출된 blink 값이 올바르다는 뜻은 아니다. 이 영상에는 GUI 기준 출력이 없어 (i)과 (ii)의 blink 값의 정확성은 미검증이다. [확실]
- ccw90은 시험하지 않았다. 미검증. 회전 태그가 원인인지도 여전히 미검증이다. [확실]

# review 14개 육안 판정 결과 (2026-10-02)

- 판정 방법: `src/repro/review_orientation.py`가 원본 프레임(OpenCV, 회전 태그 미적용)을 4방향(0, cw90, ccw90, 180)으로 돌려 2x2로 한 창에 보여 주면 사용자가 얼굴이 바로 선 칸을 고른다. 결과는 `configs/video_orientation_overrides.csv`에 기록된다(관찰자 "사용자 육안", 날짜 2026-10-02). [확실]
- 확인한 프레임 위치: 도구가 보여 주는 25%/50%/75% 위치 중 사용자가 본 위치(스페이스로 전환)이다. 영상마다 실제로 어느 위치를 봤는지는 기록되지 않았다. 미검증 [확실]
- 판정 수: 14개(v2 `review` 14개 전부). override가 없는 `review` 영상은 0개이고, override 14행은 모두 v2 `review` 영상이다. 관찰자는 14행 모두 "사용자 육안", 날짜는 14행 모두 2026-10-02이다. [확실] (측정)
- orientation별 영상 수: `0` 3, `cw90` 7, `ccw90` 4, `180` 0. [확실] (측정)
- 영상별 판정 값(상대 경로): [확실] (측정)
  - `0` (3개): Fold2_part2/23/0.mp4, Fold2_part2/23/5.mp4, Fold2_part2/23/10.mp4
  - `cw90` (7개): Fold5_part1/52/5.MOV, Fold5_part2/56/0.MOV, Fold5_part2/56/5.MOV, Fold5_part2/57/5.MOV, Fold5_part2/59/0.MOV, Fold5_part2/59/5.MOV, Fold5_part2/59/10.MOV
  - `ccw90` (4개): Fold4_part2/47/0.mp4, Fold5_part1/49/0.mp4, Fold5_part1/49/10_2.mp4, Fold5_part2/58/0.mp4
- note 열: Fold2_part2/23의 0·5·10 3개에 "방향O, 얼굴은 보이나 검출기 미검출"이 적혀 있고 나머지 11개는 비어 있다. [확실] (CSV)
- 육안으로 얼굴이 보이나 조사(dlib 검출)에서는 0 또는 거의 0이던 영상(조사 카운트는 8프레임에서 검출된 얼굴 수의 합): [확실] (CSV, 사용자 보고)
  - Fold2_part2/23/0.mp4: 선택한 0°의 카운트 0. Fold2_part2/23/10.mp4: 0°의 카운트 0. 둘 다 note에 명시됨.
  - Fold2_part2/23/5.mp4: 선택한 0°의 카운트 1. note에 명시됨.
  - Fold5_part1/49/0.mp4: 사용자가 이 작업에서 예로 든 영상이다. note는 비어 있고 조사는 네 방향 모두 카운트 0이다(육안은 `ccw90`).
- 조사 카운트와의 대조: 카운트가 0이 아닌 11개는 육안 선택 방향이 카운트 최댓값 방향과 같거나(8개) 최댓값 동률 중 하나(3개: Fold5_part2/59/0·5·10은 cw90=ccw90=8이고 육안은 `cw90`)였다. 네 방향 모두 카운트 0이라 대조할 수 없는 영상은 3개(23/0, 23/10, 49/0)이다. [확실] (측정)
- 위 대조에서 최댓값 방향 자체가 육안과 어긋난 영상은 없었다. `review`로 남은 이유는 임계값(최댓값 5 미만)과 동률이었다. [가능성 높음] (이 14개 한정, 독립 검증은 아님)
- 가설: 얼굴이 육안으로 보이는데 검출되지 않은 영상(23/*, 49/0)의 공통점은 "고개를 든 자세일 수 있다"는 것이다. [추측]. 고개 자세는 확인하거나 측정하지 않았다. 원인 미검증 [확실]
- 반영 상태: override 14개가 반영되어 배치 도구 `--list`에서 `review`는 0개, override 적용 14개이고 처리 목록은 174개(회전 `none` 96, `cw90` 44, `ccw90` 33, `180` 1)이다. [확실] (측정). 육안 판정한 방향으로 실제 blink를 추출한 결과는 아직 시험하지 않았다. 미검증
- dlib 검출기 설정과 코드는 바꾸지 않았다. [확실]

# ccw90 회전 시험 결과 (2026-10-02)

- 사후 기록: 이 시험의 통과 기준(`face_frames`가 `frame_count`의 50% 이상)은 시험 전 대화에서 정했으나, 로그의 "회전 옵션 시험 사전 등록" 항목에는 시험 (i)~(iii)만 있고 ccw90 시험은 없었다. 따라서 이 항목은 결과와 함께 쓰는 사후 기록이며 사전 등록된 예측이 아니다. [확실]
- 시험: Fold4_part2/47/0.mp4를 `--rotate ccw90`으로 실행. 방향 `ccw90`은 사용자 육안 판정이다(위 "review 14개 육안 판정 결과" 항목). [확실]
- 결과: `face_frames` 18027, blink 27행, 소요 1164.1초(약 19분 24초). `face_frames`와 소요 시간은 사용자가 보고한 콘솔 값이고, blink 27행은 결과 파일(27행)에서 이 세션이 확인했다. [확실]
- 비율: `configs/video_inventory.csv`의 `frame_count`는 18027(1920×1080, avc1)이므로 `face_frames`/`frame_count` = 100.00%(18027/18027)이다. 기준(50% 이상) **통과**. [확실] (계산). `face_frames`가 `frame_count`와 정확히 같으므로 모든 프레임에서 얼굴이 검출되었고 영상 끝까지 처리된 것으로 보인다. [가능성 높음]
- 회전 없이 처리할 경우: 조사 CSV에서 이 영상의 0° 얼굴 카운트는 0이었다(카운트 0°/cw90/180°/ccw90 = 0/6/0/8, v1 `ambiguous`, v2 `review`). [확실] (조사 CSV). 회전 없이 실제 추출은 하지 않았다. 미검증
- 속도: 18027프레임/1164.1초 ≈ 15.5 fps. [확실] (계산). 사용자 보고로는 단독 실행이었다. `%TEMP%`의 다른 시험 결과 파일과는 실행 시각이 겹치지 않음을 확인했다(이 시험의 파일 시각 19:12~19:30, 앞선 시험은 17:37 이전). [확실] (파일 시각). 저장소 밖 다른 출력 폴더에서 다른 작업이 돌았는지는 확인하지 못했다. 미검증
- 비교 기준 정정: 앞서 단독 약 23 fps라고 한 기록은 Fold2_part1/17/0.mp4(18121프레임 793초)이고, 이 영상의 저장 크기는 480×640으로 가로형이 아니다. [확실] (조사 CSV). 폭 450으로 줄인 처리 크기는 17/0 약 450×600, 10/0.MOV(cw90 회전, 저장 568×320) 약 450×798, 이 영상(ccw90 회전, 저장 1920×1080) 450×800이다. [확실] (계산)
- 같은 처리 크기인 10/0.MOV는 약 18.6 fps(16309/876.7)였고 이 영상은 약 15.5 fps였다. 같은 영상(17/0)도 실행마다 약 18.3~24 fps였으므로 속도 비교는 대략적이다. [확실] (계산)
- 원인: 회전 영상이 가로형 단독보다 느린 것은 화소 수 증가 때문이라는 설명은 [추측]이다. 처리 크기가 같은 10/0.MOV보다도 느렸으므로 처리 크기만으로는 설명되지 않고, 저장 크기가 큰(1920×1080) 원본 프레임의 디코딩·리사이즈 비용이 더 클 수 있다. [추측]. 원인 미검증
- 판정: ccw90 방향 정의는 이 영상 1개에서 기준을 통과했다(`face_frames` 100%, blink 27행). 육안 판정(`ccw90`)과 조사 카운트(ccw90 8)도 같은 방향을 가리킨다. [확실]. 1영상 한정이며 다른 ccw90 영상(예: Fold5_part1/49/0.mp4, Fold5_part1/49/10_2.mp4, Fold5_part2/58/0.mp4)으로 일반화는 미검증이다. 이 영상의 blink 값 정확성은 GUI 기준 출력이 없어 미검증이다. [확실]

# --show 옵션 사전 등록 (2026-10-03)

이 항목은 `reproduction_blink_headless.py`에 `--show`를 추가하는 코드를 쓰기 **전에** 쓴다. 결과는 날짜가 붙은 별도 항목으로 추가하고 이 항목은 고치지 않는다.

- 목적: `cv2.imshow` 창을 띄워 저자 코드가 그리는 화면 프레임(눈·입 윤곽)을 사람이 볼 수 있게 한다. 기본(무화면) 동작은 바꾸지 않는다. [확실] (작업 정의)
- `--show` 정의: `cv2.imshow`(`blink_video.py` 476, 558줄)와 `cv2.waitKey`(480, 562줄) 치환을 하지 않는다. `plot_frame.draw()`(474, 556줄) 치환은 유지한다. `--rotate`와 함께 쓸 수 있고, 치환 개수 검증의 기대값은 치환 목록에 맞게 줄어든다. 출력 파일 덮어쓰기 거부는 유지한다. [확실] (정의)
- 용어: "치환 대상 항목"은 치환 목록의 항목 수이고, "바뀌는 줄"은 치환으로 내용이 달라지는 소스 줄 수이다. 사용자 지시의 "rotate none은 9개 항목"은 바뀌는 줄 9개로 해석한다. 현재 치환 대상 항목은 6개이다. [확실]
- 예측 (a): `--show` 없이 `--dry-run`의 결과가 변경 전과 같다. rotate `none`은 치환 대상 항목 6개, 바뀌는 줄 9개 [313, 474, 476, 480, 556, 558, 562, 579, 580]이고, rotate `ccw90`은 항목 7개, 바뀌는 줄 10개 [313, 359, 474, 476, 480, 556, 558, 562, 579, 580]이다. 추가로 rotate `none`의 치환된 소스는 변경 전 6개 치환 결과와 바이트 단위로 같다. [추측]
- 예측 (b): `--show --dry-run`에서는 imshow/waitKey 4줄(476, 480, 558, 562)이 치환 목록에서 빠진다. rotate `none`은 항목 4개(출력 경로, 입력 영상, dlib .dat, draw), 바뀌는 줄 5개 [313, 474, 556, 579, 580]이고, rotate `ccw90`은 항목 5개, 바뀌는 줄 6개 [313, 359, 474, 556, 579, 580]이다. [추측]
- 실행(`--dry-run`이 아닌 것)은 사용자가 한다. 창 표시와 키 입력 동작은 에이전트가 시험하지 못한다. 미검증 [확실]
- 판정 규칙: 예측이 하나라도 빗나가면 그대로 기록한다. [확실] (사전 정의)

# --show 옵션 결과 (2026-10-03)

위 "--show 옵션 사전 등록" 항목의 예측 (a), (b)를 `--dry-run`으로 확인했다. 영상은 열지 않았다. [확실]

- 구현: `reproduction_blink_headless.py`에 `--show`를 추가했다. `--show`이면 `cv2.imshow`와 `cv2.waitKey` 치환을 목록에서 뺀다(`plot_frame.draw()` 치환은 유지). `--rotate`와 함께 쓸 수 있다. 도움말(`--help`)에 "창이 열리는 로컬 환경 전용, 처음 1~2분만 보고 Ctrl+C로 종료"를 적었다. [확실]
- 예측 (a) **적중**: `--show` 없이 rotate `none`은 치환 대상 항목 6개(모두 기대 횟수와 일치), 바뀌는 줄 9개 [313, 474, 476, 480, 556, 558, 562, 579, 580]. rotate `ccw90`은 항목 7개, 바뀌는 줄 10개 [313, 359, 474, 476, 480, 556, 558, 562, 579, 580]. rotate `none`의 치환된 소스는 변경 전 6개 치환 결과와 바이트 단위로 같았고, `show=False`를 명시해도 같았다. [확실] (측정)
- 예측 (b) **적중**: `--show --dry-run`에서 imshow/waitKey 4줄(476, 480, 558, 562)이 치환 목록에서 빠졌다. rotate `none`은 항목 4개, 바뀌는 줄 5개 [313, 474, 556, 579, 580]. rotate `ccw90`은 항목 5개, 바뀌는 줄 6개 [313, 359, 474, 556, 579, 580]. `--show` 소스에서 476, 480, 558, 562줄은 원문 그대로이고 474, 556줄은 `None`으로 바뀌었다. [확실] (측정)
- 추가 확인: `py_compile` 통과, `--show` 소스와 `--show --rotate cw90` 소스의 컴파일 통과(실행하지 않음). `--show`에서도 이미 있는 출력 파일에는 이어 쓰지 않고 영상을 열기 전에 종료 코드 1로 끝났으며 그 파일(2128바이트)은 그대로였다. [확실] (측정)
- 아직 확인하지 못한 것: 실제 창 표시, 화면 프레임의 모습, 키 입력과 Ctrl+C 동작은 사용자가 실행해 봐야 한다. 미검증 [확실]
- 코드 읽기로 알 수 있는 동작: `--show`에서는 waitKey가 살아 있어 창에서 아무 키나 눌러도 루프가 끝나고(`blink_video.py` 480~483, 562~566줄), 영상 끝까지 가지 못하면 `not grabbed`와 face_frames 숫자는 찍히지 않는다(353~356줄). Ctrl+C로 끝내도 래퍼는 소요 시간과 blink 행 수를 출력한다. [확실] (코드 읽기). 읽기 전용 점검 결과(화면에 그려지는 것, face_frames의 의미, 종료 경로)는 `docs/IMPLEMENTATION_DIFF.md`의 "--show 사용 메모"에 있다.

# 일괄 추출 실행 결과 (2026-10-03)

이 항목의 수치는 사용자가 보고한 실행 결과이고, 에이전트가 `summarize_batch.py`와 매니페스트, 출력 폴더를 읽어 다시 확인했다. 사용자 수치와 다른 값은 없었다. 출력 위치는 저장소 밖이며 경로는 적지 않는다. [확실]

## 1. 배치 실행 결과

- 실행: workers 6(사용자 보고), 시작 2026-10-02 19:45:57, 종료 2026-10-03 08:34:03, 경과 12시간 48분 6초, 합산 처리량 65.2 fps(frame_count 합 3,002,817 / 경과, 중간에 멈춘 시간을 포함한 값). [확실]
- 대상: 처리 목록 174개(manual_flags 8개 제외, 회전 미정 0개). 매니페스트 기록 173개는 ok 172, no_output 1(Fold2_part2/23/0.mp4), failed 0이고 같은 영상의 재시도 행은 0개이다. [확실]
- 매니페스트에 행이 없는 1개는 Fold2_part1/17/0.mp4이다. 이전 시험 산출물이며, 행은 이전 매니페스트에 있었다(사용자 보고). 출력 txt가 2,128바이트로 존재함을 확인했다(최종 txt 173개 = ok 172 + 이 파일 1, 남은 `.part` 0개). `summarize_batch.py`는 이 영상을 "미처리 1개"로 세어 남은 시간 4~6분을 표시했다(재실행: 최근 처리량 기준 6분 7초, 평균 기준 4분 38초). 실제로는 처리할 영상이 없으므로 이 표시는 요약 도구의 한계이며 보류 목록에 둔다. [확실]
- 같은 크기와 같은 마지막 기록 시각(08:34:03)의 매니페스트 사본이 다른 드라이브에 있음을 확인했다. [확실]
- face_frames 합 2,836,727 / 2,965,950 = 95.6%(face_frames가 기록된 영상 한정). 비어 있는 2개(Fold4_part1/40/10.mp4, Fold1_part2/12/5.mp4)는 status ok, 종료 코드 0, blink 218행과 109행이며, 이 둘의 frame_count 합 36,867(= 3,002,817 − 2,965,950)이 분모에서 빠졌다. 비어 있는 원인은 조사하지 않았다. 미검증 [확실]
- 회전별 face_frames/frame_count: `none` 93개 평균 0.9588 최소 0.0019, `cw90` 44개 평균 0.9900 최소 0.7977, `ccw90` 33개 평균 0.9261 최소 0.0316, `180` 1개 0.9764(Fold3_part1/26/5.mp4). [확실]
- 비율이 0.5 미만인 5개: Fold2_part2/23/5.mp4 0.1147(ok), 23/0.mp4 0.0019(no_output), 23/10.mp4 0.0062(ok), Fold5_part2/58/0.mp4 0.4184(ok), Fold5_part1/49/0.mp4 0.0316(ok). 5개 모두 육안 판정한 review 14개에 속하는 영상이다. [확실]
- blink 행이 30 미만인 영상: 사용자는 약 20개, 그중 alert 약 11개라고 했다(콘솔 로그를 사람이 센 값). 에이전트가 매니페스트의 blink_rows로 센 값은 20개이고 클래스별로 alert(0) 11, semisleepy(5) 5, sleepy(10) 4이며 blink 0행은 Fold2_part2/23/0.mp4 하나이다. 사용자 값과 일치한다. [확실]
  - 행 수와 영상: 0 Fold2_part2/23/0.mp4, 3 Fold5_part1/49/0.mp4, 4 Fold2_part2/23/10.mp4, 9 Fold3_part2/32/10_2.mp4, 10 Fold4_part1/38/5.mp4, 12 Fold2_part1/16/0.MOV, 15 Fold3_part1/25/0.mp4, 15 Fold3_part2/34/0.mov, 15 Fold5_part1/53/5.MOV, 17 Fold3_part1/25/5.mp4, 19 Fold1_part2/10/0.MOV, 19 Fold4_part1/41/0.MP4, 20 Fold3_part1/29/10.mp4, 21 Fold3_part1/26/10.mp4, 22 Fold3_part1/29/5.mp4, 24 Fold5_part1/50/0.MOV, 26 Fold3_part1/27/0.MOV, 26 Fold3_part2/33/0.mp4, 27 Fold4_part2/47/0.mp4, 28 Fold2_part2/23/5.mp4. [확실]
  - 이 20개의 blink 행이 적은 원인은 규명하지 않았다. 미검증 [확실]
- 소요 시간 관측: 서로 다른 영상의 소요가 0.1초 차이로 묶인다(사용자 보고: 1920.5/1920.5/1920.6, 3748.7/3748.7/3748.8, 1188.0/1188.1, 1865.1/1865.1). 매니페스트로 확인한 사실은 다음과 같다. [확실] (측정)
  - 같은 시각에 시작해 같은 시각에 끝난 묶음: Fold1_part1/04/0.mp4, Fold1_part1/04/10.mp4, Fold5_part1/49/10_1.mp4(19:45:57 시작, 20:17:58 종료). Fold1_part2/10/0.MOV, Fold2_part1/17/10.mp4, Fold5_part1/49/10_2.mp4(19:45:57 시작, 20:48:26 종료). Fold3_part2/32/10_2.mp4, Fold5_part1/51/10.mov(20:17:58 시작, 20:37:46 종료). Fold3_part2/32/10_1.mp4, Fold2_part1/17/5.mp4(20:48:26 시작, 21:19:31 종료).
  - 같은 묶음 안의 영상은 frame_count와 파일 크기가 서로 달라 같은 파일이 아니다.
  - 소요 값만 비슷하고 시각이 다른 경우도 있다: Fold1_part1/05/10.MOV(23:44~00:15)는 1865.0초 대, Fold1_part2/09/10.mp4(02:07~02:46)와 Fold5_part1/52/10.MOV(00:15~00:54)는 2334.0~2334.2초이다. 값 하나가 우연히 비슷한 쌍은 영상 173개에서 몇 쌍 나올 수 있다. [가능성 높음]
  - 같은 시각에 시작해 같은 시각에 끝난 묶음의 원인은 미검증이다. 개별 영상의 처리량이 아니라 여러 프로세스에 공통인 요인이 끝나는 시각을 정했을 가능성이 있다. [추측]

## 2. 예측 대비 (사전 등록이 아님)

- 이 항목의 예측은 대화에서 나온 것이고 로그에 사전 등록 항목은 없었다. 사전 등록이 아닌 대화상 추정으로만 기록한다. [확실]
- 처리 시간: 대화에서 6~20시간 범위를 제시했고 실측은 12.8시간(12시간 48분)으로 그 범위 안이다. [확실] (숫자 비교)
- 얼굴 비율 0.5 미만 목록에 Fold2_part2/23/0, 23/5, 23/10, Fold5_part1/49/0이 포함된다는 대화상 예측은 4/4 적중했다. 목록은 5개이며 Fold5_part2/58/0(0.4184)은 예측 밖이었다. [확실]

## 3. 사용자 방침 (2026-10-03)

- 사용자 방침: "재현이 목적이므로 검출이 잘 안 된 영상(피험자 23 포함)은 기록만 하고 검출기 설정과 파이프라인은 바꾸지 않는다. 저자가 해당 영상을 제외했는지 확신이 없는 부분은 그대로 두고 미확인으로 기록한다." [확실] (사용자 방침)
- 저자가 어떤 영상을 제외했는지는 확인된 바 없다. 미확인 [확실]
- 사용자가 "2단계"라 부른 단계에서 어떤 영상이 제외되더라도 그것은 기준 영상이 없다는 기계적 결과이며, 저자가 해당 영상을 제외했다는 주장이 아니다. "2단계"는 사용자의 용어이며 이 저장소 문서에는 정의가 없다. [확실]
- 이 방침에 따라 검출기 설정과 추출 파이프라인은 바꾸지 않았다. [확실]

## 4. 사용자 육안 인상 (정량 확인 아님)

- 피험자 23: "육안으로는 문제가 없어 보이는데 눈·입 윤곽을 거의 잡지 못함". 인상을 받은 영상 ID는 기록하지 않았다. 미검증
- blink 행이 적은 영상: "눈을 잘 안 깜빡이는 것처럼 보임". 실제로 덜 깜빡이는 것인지, SVM이나 진폭 기준(`blink_video.py` 220~221줄)이 후보를 거부한 것인지 구분하지 못했다. 사람이 직접 센 횟수와 콘솔의 승인·거부 로그(232, 236, 239줄)를 비교하지 않았다. 미검증

## 5. "MISSED BLINKS" 해석 점검

- `docs/`와 README에서 "MISSED BLINKS"나 "놓친"을 Grep으로 찾았으나 0건이다. 이전 서술이나 문서에서 이 로그를 놓친 blink의 증거로 해석한 곳은 없어 정정 항목은 만들지 않는다. [확실] (검색)
- 참고: 콘솔의 "MISSED BLINKS= n"(`blink_video.py` 232줄)의 n은 승인된 후보에서 추출한 blink 개수(`len(retrieved_blinks)`)이지 놓친 blink가 아니다. n은 0일 수도 있다(추출된 blink가 없으면 165~168줄이 배열을 출력하고 빈 목록을 돌려준다). `Ultimate_Blink_Check`가 돌려주는 `MISSED_BLINKS`는 후보 하나에서 blink가 둘 이상 나오면 True이고(163~164줄) 출력에는 쓰이지 않는다(227줄에서 받기만 함). 후보를 거부할 때의 메시지는 236줄, 239줄이다. [확실] (코드 읽기)

## 6. 배치 대 단독 비교

- Fold4_part2/47/0.mp4(`ccw90`)의 배치 출력과 단독 실행 출력은 28줄이 모두 같았다(사용자 보고). [확실]
- 에이전트가 같은 방식으로 Fold1_part2/10/0.MOV(`cw90`)의 배치 출력과 앞선 단독 시험 출력도 비교했고, 20줄이 모두 같았다. [확실] (측정)
- `180`은 비교하지 못했다. 단독 시험 영상(Fold4_part2/45/10.mp4)은 manual_flags라 배치 목록에 없고, 배치에서 회전 `180`으로 처리된 영상은 Fold3_part1/26/5.mp4 하나이며 단독 실행과 비교하지 않았다. [확실]
- 비교한 것은 `ccw90` 1영상과 `cw90` 1영상뿐이다. 다른 영상으로 일반화는 미검증이다. [확실]

# 시퀀스 생성 구현 사전 등록 (2026-10-03)

이 항목은 `src/repro/reproduction_build_sequences.py`를 쓰기 **전에** 쓴다. 결과는 날짜가 붙은 별도 항목으로 추가하고 이 항목은 고치지 않는다.

- 목적: 우리가 재추출한 blink txt(`<출력 루트>/Fold<N>_part<M>/<피험자>/<클래스>[_<변형>].txt`)에서 저자 `Preprocessing.py`(PP)와 같은 계산으로 `.npy` 시퀀스를 만든다. 저자 코드는 `exec`하지 않고 직접 구현한다. [확실] (작업 정의)
- 1단계 점검(저자 PP 읽기)에서 알려진 내용과 다른 점은 없었다. 정규화 기준은 alert의 마지막 `n//3`개(PP 168~187줄), 쓰는 열은 txt의 1~4열(163~166줄), 데이터 구간은 alert `n − n//3`개(169, 189줄)이고 semisleepy·sleepy는 전부(213, 231줄)이다. 윈도우 30, 이동 폭 2(263~264줄), `n ≤ 30`이면 앞쪽 0 패딩 시퀀스 1개(27~30줄), 아니면 `(n−30)//2+1`개(32줄)이다. [확실] (코드 읽기)
- 1단계에서 추가로 측정한 저자 코드의 동작: blink가 정확히 1행인 파일은 `np.loadtxt`가 0차원 배열을 돌려줘 `len()`에서 TypeError가 난다. 0행(빈 파일)은 빈 배열과 경고만 나오고 `unroll_in_time`(29줄)에서 ValueError가 난다. `n < 3`이면 `bunch_size`가 0이라 `[-0:]`가 배열 전체이다. 표준편차가 0이면 전체 표준편차로 대체하고(173~174줄 등) 그것도 0이면 NaN이다. [확실] (numpy 1.14.6 측정)
- `Training.py`의 `batch_gen`은 배치를 섞지 않고 순서대로 자른다(265~275줄). 288줄 주석도 입력이 섞여 있다고 적는다. 그래서 훈련 세트는 섞어서 저장해야 하며, 저자 PP는 255줄에서 섞는다. 우리 구현은 재현 가능하도록 fold 번호를 시드로 고정해 섞는다(저자는 시드가 없다). 이것은 의도한 차이이다. [확실]
- 제외 정책(영상별로 `exclusions.csv`가 아니라 `video_summary.csv`와 제외 목록에 사유와 함께 기록하며 앞 피험자 값으로 대체하지 않는다): `variant_deferred`(변형 파일), `not_extracted`(txt가 없고 인벤토리의 manual_flag가 1), `no_blink_file`(그 밖에 txt가 없거나 blink 0행), `no_alert_reference`(그 피험자의 alert를 쓸 수 없음), `degenerate_reference`(기준 구간 blink 수 < 2, 또는 저자식 대체 후에도 표준편차가 0이거나 비유한). 영상마다 앞에서부터 이 순서로 하나만 붙인다. [확실] (정의)
- 제외는 "재추출 데이터에서의 기계적 결과"이다. 저자가 어떤 영상이나 피험자를 제외했는지는 확인된 바 없으며 우리 제외를 저자의 제외로 해석하지 않는다. 미확인 [확실]
- `not_extracted`와 `no_blink_file`은 txt가 없다는 점이 같아서 txt 존재만으로는 구분되지 않는다. 구분은 인벤토리의 `manual_flag`(배치 기본 목록에서 빠지는 영상)로 하며, 아직 추출하지 않은 영상이 `no_blink_file`로 분류될 수 있다. 미검증 [확실]
- 예측 (a): 합성 blink 표(alert n ∈ {2, 29, 30, 31, 32, 33, 57, 100, 200}, semisleepy·sleepy n ∈ {2, 3, 5, 29, 30, 31, 32, 33, 57, 100, 200})로 우리 구현의 X와 y가 저자 PP의 함수·절차(검증 하네스에서만 PP의 함수 정의 1~259줄을 메모리에서 실행하고 실행부 261~272줄은 제외, 파일은 쓰지 않음)와 `np.array_equal`로 같다. 테스트 fold는 순서까지 같고, 훈련 fold는 같은 시드와 같은 순서이면 순서까지 같다. [추측]
- 예측 (a)의 예외(비교에서 빼고 두 동작을 따로 기록): alert n = 1은 저자 TypeError. alert n = 3, 4, 5는 기준 구간이 1개라 우리는 `degenerate_reference`로 제외하지만 저자는 전체 표준편차로 대체해 진행한다. 모든 행이 같은 값이면 저자는 NaN을 내고 우리는 제외한다. semisleepy·sleepy n = 1은 저자 TypeError이고 우리는 포함한다. [추측]
- 예측 (b): 영상 하나의 시퀀스 수는 사용 blink 수가 30 이하이면 1, 30보다 크면 `(n−30)//2+1`이다. alert의 사용 blink 수는 `n − n//3`이다. 메타데이터의 `seq_in_video` 개수와 일치한다. [추측]
- 예측 (c): blink 0행(빈 파일, 빈 줄만 있는 파일)은 `no_blink_file`로 기록되고 예외가 나지 않는다. [추측]
- 예측 (d): 실제 출력 루트를 `--check-only`로 읽기만 해 실행하면 오류 없이 coverage가 나오고 다음과 같다(다르면 그대로 보고). 영상 182개 중 포함 165개, 제외 17개: `no_blink_file` 1개(Fold2_part2/23/0), `variant_deferred` 4개(피험자 32와 49의 10_1, 10_2), `not_extracted` 8개(manual_flags), `no_alert_reference` 2개(피험자 23의 5, 10), `degenerate_reference` 2개(피험자 49의 0, 5; alert가 3행). `ref_small`(기준 구간 blink 수 10 미만)은 피험자 9명(10, 16, 25, 27, 33, 34, 41, 47, 50). [추측]
- 예측 (e): 같은 입력으로 서로 다른 임시 출력 폴더에 두 번 실행한 `.npy`가 `np.array_equal`로 같다. [추측]
- 판정 규칙: 예측이 하나라도 빗나가면 그대로 기록한다. 실제 출력 루트(`SEQ_OUT_DIR`)에는 쓰지 않고, 시험은 임시 폴더의 합성 데이터로만 한다. [확실] (사전 정의)


# 시퀀스 생성 구현 결과 (2026-10-03)

사전 등록(위 "시퀀스 생성 구현 사전 등록 (2026-10-03)")에 대한 결과다. 구현은 `src/repro/reproduction_build_sequences.py`이다. 저자 코드는 exec하지 않고 같은 계산을 직접 구현했다. 산출물(.npy, 메타데이터 CSV)은 저장소에 두지 않았고, 실제 `SEQ_OUT_DIR`에는 쓰지 않았다.

## 시험 방법과 한계
- 합성 데이터 시험은 임시 폴더(저장소 밖)에서만 했다. 하네스 스크립트는 저장소에 넣지 않았다. [확실]
- 비교를 위해 하네스 안에서만 저자 `Preprocessing.py`의 함수 정의(1~259줄)를 텍스트로 읽어 메모리에서 정의했다(`os.listdir`는 정렬하는 것으로 바꿨다). 구현 파일에는 exec가 없다. 하네스에서 exec를 쓴 것은 "exec 금지" 규칙에 대한 예외로, 사용자에게 알린 사항이다. [확실]
- 합성 데이터는 임의 난수이므로 이 시험은 계산 절차의 동일성만 확인한다. 실제 데이터에서 저자 결과와 같은지는 확인하지 못했다. 미검증

## 예측 대조
- (a) 합성 시험 전부 통과(88개 점검 중 실패 0). [확실]
  - 전체 파이프라인: 3개 fold × 2개 part × 2명, 사용 blink 수가 6~200인 합성 표에서 fold마다 저자 `Preprocess`(시드 = fold 번호)와 우리 train/test X, y가 `np.array_equal`로 같았고 dtype도 float64로 같았다.
  - 함수 수준: n = 2, 3, 5, 29, 30, 31, 32, 33, 57, 100, 200을 alert, semisleepy, sleepy 각각에 넣어 저자 `gen`과 우리 함수가 `np.array_equal`로 같았다(alert n = 3, 5는 우리 파이프라인이 제외하는 값이지만 계산 자체는 같다).
  - 퇴화 입력의 저자 동작과 우리 동작을 따로 기록했다: alert n = 1은 저자 TypeError, 우리 `degenerate_reference`. alert n = 3, 4, 5는 저자가 전체 표준편차로 대체해 진행(시퀀스 29개)하고 우리는 `degenerate_reference`. 모든 행이 같은 값이면 저자는 NaN을 내고 우리는 제외. semisleepy n = 1은 저자 TypeError이고 우리는 포함. 예측과 같았다.
- (b) 시퀀스 수가 n ≤ 30이면 1, n > 30이면 `(n−30)//2+1`로 n = 1~200에서 맞았다. alert 사용 blink 수는 `n−n//3`으로 맞았다. 메타데이터 행 수는 모든 fold의 train, test에서 시퀀스 수와 같았다. [확실]
- (c) 0바이트 파일, 빈 줄만 있는 파일, txt 없음은 `no_blink_file`, alert가 0행이면 같은 피험자의 나머지가 `no_alert_reference`, alert 3행은 `degenerate_reference`로 기록되고 예외가 없었다. [확실]
- (d) 실제 출력 루트에 `--check-only`(읽기 전용, `SEQ_OUT_DIR` 없이)를 실행했고 오류 없이 끝났다(종료 코드 0). 예측과 모두 같았다. [확실]
  - 영상 182개: 포함 165, 제외 17. `variant_deferred` 4, `not_extracted` 8, `no_blink_file` 1, `no_alert_reference` 2, `degenerate_reference` 2.
  - ref_small(기준 구간 blink 수 10 미만) 피험자 9명: 10, 16, 25, 27, 33, 34, 41, 47, 50. 기준 구간 표준편차 0으로 전체 표준편차를 쓴 피험자는 0명.
  - 사용 가능한 피험자 56명, ref_n 최소 4, 중앙값 24, 최대 172.
  - fold별 포함 영상 수(클래스 0/5/10): Fold1 11/11/11, Fold2 11/11/11, Fold3 12/12/11, Fold4 11/10/11, Fold5 11/11/10.
  - fold별 시퀀스 수(훈련/테스트)와 테스트 영상 수: Fold1 6506/1328/33, Fold2 5775/2059/33, Fold3 6193/1641/35, Fold4 6628/1206/32, Fold5 6234/1600/32.
- (e) 같은 입력으로 서로 다른 임시 출력 폴더에 두 번 실행한 .npy가 모두 `np.array_equal`로 같았고(float64), 메타데이터 폴더의 해시도 같았다. [확실]
- 가드: `SEQ_OUT_DIR` 미설정, 저장소 안, `BLINK_OUT_DIR` 안이면 중단하고 폴더를 만들지 않았다. 출력 파일이 이미 있으면 덮어쓰지 않고 중단했으며 파일은 그대로였다. `--check-only`는 입력 폴더를 바꾸지 않았고 `SEQ_OUT_DIR`가 있어도 쓰지 않았다. [확실]

## 기록할 점
- 처음 하네스 실행에서 점검 1개가 실패했는데, 하네스의 오류였다. 합성 데이터가 3개 fold라서 npy가 12개인데 20개를 기대하도록 적었다. 기대 개수를 고쳐 다시 실행해 전부 통과했다. 구현 코드는 바꾸지 않았다. [확실]
- 정정: 사전 등록 항목은 제외 사유를 "`exclusions.csv`가 아니라 `video_summary.csv`와 제외 목록에" 기록한다고 적었다. 이는 사용자 지시(정책 줄에 `exclusions.csv`가 명시됨)와 맞지 않는 문구였다. 구현은 `metadata/exclusions.csv`(열: video_id, subject_id, class_label, level, reason, detail)를 쓴다. 이 항목이 사전 등록의 문구를 정정한다. [확실]
- 제외 영상 목록은 "재추출 데이터에서의 기계적 결과"이다. 저자가 어떤 영상이나 피험자를 제외했는지는 확인된 바 없다. 확인되지 않았다.
- 우리 테스트 영상 수(Fold1~5: 33, 33, 35, 32, 32)는 `Training.py`의 start_indices 목록 길이(33, 36, 36, 36, 33)와 Fold1을 빼고 다르다. 원인이 우리 제외인지, 저자 데이터의 구성인지는 확인하지 못했다. 미검증
- 저자 `Preprocess`는 폴더 순서를 `os.listdir`에 맡긴다. 우리는 (fold, part, 피험자 id) 순으로 정렬했다. 저자 실제 데이터의 피험자 순서가 같은지, 저자가 실제로 훈련 세트를 섞었는지는 확인하지 못했다. 합성 비교에서는 순서를 정렬로 맞췄다. 미검증
- `ref_small` 피험자 9명의 정규화는 마지막 n//3 개(4~9개 blink)로 계산한 평균과 표준편차에 의존한다. 이 값이 저자 결과와 가까운지는 알 수 없다. 미검증


# degenerate_reference 정책 변경 사전 등록 (2026-10-03)

## 변경 이유
- 위 "시퀀스 생성 구현 사전 등록 (2026-10-03)"의 `degenerate_reference` 규칙(기준 구간 blink 수 2 미만이면 피험자 제외)은 alert n = 3~5에서 저자 PP보다 엄격하다. 저자 PP는 n = 2에 전체 배열 통계, n = 3~5에 마지막 n//3(= 1)개 평균과 열 전체 표준편차 대체를 쓰며 진행한다(합성 시험으로 확인, n = 3~5는 시퀀스 29개). 사용자 방침은 "저자 코드 재현이 목적이므로 저자가 하지 않은 제외를 만들지 않는다"이다. 그 방침과 맞지 않아 규칙을 바꾼다. [확실]
- 이 항목 이전의 결과 항목(영상 182개 중 포함 165, 제외 17의 `--check-only` 결과 포함)은 고치지 않고 보존한다.

## 변경 내용(구현 전)
- 피험자 제외(`degenerate_reference`)는 저자 코드가 예외를 내거나 비유한 값을 만드는 경우로 한정한다: alert 1행(저자 TypeError), 대체 후에도 표준편차가 0 또는 비유한(저자 NaN).
- 그 밖의 경우(alert n = 2~5, 표준편차 0 대체)는 저자 동작을 따른다. `video_summary.csv`에 열 `author_fallback_used`를 추가한다: `none` / `whole_array`(n < 3이라 전체 배열 통계) / `whole_std`(기준 구간 표준편차가 0이라 전체 열 표준편차로 대체한 열이 있음). `ref_n`, `ref_small`은 유지하고, `coverage.txt`에 `author_fallback_used` 개수를 추가한다.

## 예측
- (a) 합성 시험에서 alert n = 2, 3, 4, 5의 X, y가 저자 PP 함수(`Preprocess`)와 `np.array_equal`로 같다(이전에는 3~5가 제외였다). [추측]
- (b) alert n = 1과 모든 행이 같은 값인 alert는 `degenerate_reference`로 기록되고 예외가 없다. [추측]
- (c) 실제 출력 루트 `--check-only`: `degenerate_reference` 2개 → 0개, 포함 영상 165 → 167(피험자 49의 0, 5). 49의 변형 파일 10_1, 10_2는 `variant_deferred`로 남아 49는 sleepy 없이 alert와 semisleepy만 포함된다. 다른 사유의 개수는 불변(`variant_deferred` 4, `not_extracted` 8, `no_blink_file` 1, `no_alert_reference` 2). 추가로, 사용 가능한 피험자 수는 56 → 57로 예상한다(피험자 49가 들어오므로). [추측]
- (d) 같은 합성 입력으로 두 번 실행한 .npy가 `np.array_equal`로 같다. [추측]
- 판정 규칙: 빗나가면 그대로 기록한다. 실제 `SEQ_OUT_DIR`에는 쓰지 않는다. [확실] (사전 정의)


# degenerate_reference 정책 변경 결과와 정정 (2026-10-03)

## 정정
- 위 "시퀀스 생성 구현 사전 등록 (2026-10-03)"의 `degenerate_reference` 규칙(기준 구간 blink 수 2 미만이면 피험자 제외)은 저자 재현 원칙(저자가 하지 않은 제외를 만들지 않는다)과 맞지 않아 변경했다. 이전 `--check-only` 결과(영상 182개 중 포함 165, 제외 17, `degenerate_reference` 2개 = 피험자 49의 0, 5)는 그 규칙에서 나온 결과로서 위 항목에 그대로 보존한다. [확실]
- 변경 후 규칙: 제외는 alert 1행(저자 TypeError)과 전체 열 표준편차로 대체한 뒤에도 표준편차가 0 또는 비유한인 경우(저자 NaN)로 한정한다. 나머지는 저자 동작을 따르고 `video_summary.csv`의 `author_fallback_used`(`none` / `whole_array` / `whole_std`)에 기록한다. [확실]

## 예측 대조 (위 "degenerate_reference 정책 변경 사전 등록 (2026-10-03)")
- (a) 합성 시험: alert n = 2, 3, 4, 5의 X, y가 저자 함수와 `np.array_equal`로 같았다. 함수 수준(저자 `gen`)과 파이프라인 전체(저자 `Preprocess`, 3개 fold) 모두에서였다. `fallback_kind`는 n = 2 `whole_array`, n = 3~5 `whole_std`로 기록됐다. 적중. [확실]
- (b) alert n = 1과 모든 행이 같은 값인 alert는 `degenerate_reference`로 기록되고 예외가 없었다(저자는 각각 TypeError와 NaN). 적중. [확실]
- (c) 실제 출력 루트 `--check-only`(읽기 전용, 종료 코드 0): `degenerate_reference` 2 → 0, 포함 영상 165 → 167. 49의 변형 10_1, 10_2는 `variant_deferred`로 남아 49는 alert, semisleepy만 포함된다. 다른 사유의 개수는 불변(`variant_deferred` 4, `not_extracted` 8, `no_blink_file` 1, `no_alert_reference` 2). 사용 가능한 피험자 56 → 57. 모두 적중. [확실]
  - `author_fallback_used`: `none` 56명, `whole_array` 0명, `whole_std` 1명(피험자 49, alert 3행). `ref_small`은 10명(이전 9명 + 49), `ref_n` 최소 1(49), 중앙값 24, 최대 172.
  - fold별 포함 영상 수(클래스 0/5/10)는 Fold5만 11/11/10 → 12/12/10으로 바뀌었고 나머지는 같았다. 시퀀스 수(훈련/테스트, 테스트 영상 수): Fold1 6554/1328/33, Fold2 5823/2059/33, Fold3 6241/1641/35, Fold4 6676/1206/32, Fold5 6234/1648/34. 훈련은 Fold1~4가 48씩 늘었고 Fold5 테스트가 48 늘었다(49의 시퀀스 48개). [확실]
- (d) 같은 합성 입력으로 두 번 실행한 .npy가 `np.array_equal`로 같았다(합성 점검 93개 중 실패 0). 적중. [확실]
- 하네스 변경: 정책 변경에 맞춰 점검의 기대값을 바꿨다(alert n = 3 → 포함, n = 1 → 제외로 바꿔 시험). 구현을 점검에 맞추려고 바꾼 것이 아니라 정책에 맞춰 점검을 바꿨다. 실제 `SEQ_OUT_DIR`에는 쓰지 않았다. [확실]
- 피험자 49의 정규화는 alert 3행 중 마지막 1개로 평균을 구하고 표준편차는 alert 열 전체로 대체한 값이다(저자 코드와 같은 절차). 이 값이 의미 있는 기준인지는 알 수 없다. 미검증


# 저자 .npy 대비 비교 사전 등록 (2026-10-03)

## 목적과 범위
- 우리 시퀀스 산출물(`SEQ_OUT_DIR`의 npy 20개)과 저자 원본 .npy의 테스트 배열을 읽기 전용으로 비교한다. 저자 원본은 `np.load(..., mmap_mode='r')`로만 읽는다. 표와 CSV는 `SEQ_OUT_DIR\compare\`에만 쓰고(이미 있으면 중단), 저장소 문서에는 요약 수치만 쓴다. 영상은 열지 않는다. [확실] (사전 정의)
- 사후 맞춤 금지: 정렬 규칙을 바꿔 일치율을 올리는 시도를 하지 않는다. 대응이 어긋나면 어긋나는 위치만 보고하고 재정렬하지 않는다. 저자가 어떤 영상을 제외했는지는 확인된 바 없다. [확실]

## 사전 정의한 절차
- 작업 0: 우리 npy 20개의 shape, dtype, `metadata`의 시퀀스 수 합, `start_indices` 합이 fold별로 일치하는지 확인한다.
- 작업 1: 저자 테스트 X, y(`Training.py` 414~417줄이 읽는 `BlinksTest_30_Fold%d.npy`, `LabelsTest_30_Fold%d.npy`)의 shape, dtype을 보고하고, 테스트 y를 라벨이 같은 연속 구간(런)으로 나눠 영상 경계를 복원한다. (0,5,10) 반복 구조, 영상 수(런 3개 = 피험자 1명)가 `start_indices` 길이(33, 36, 36, 36, 33)와 같은지, 경계가 `Training.py`의 start_indices와 일치하는지 확인한다. 불일치하면 이후 비교를 멈추고 보고한다.
- 피험자 순회 순서: `Preprocessing.py`는 폴더 목록을 `os.listdir`로 얻는다(146, 150, 47줄). `os.listdir`의 순서는 운영체제와 파일 시스템에 달려 있어 코드만으로는 정해지지 않는다. 기본 가정은 폴더 이름 오름차순이다. 저자의 피험자 폴더 이름은 알 수 없으므로, 우리 쪽은 (fold, part, 피험자 id) 오름차순(`reproduction_build_sequences.py`의 정렬)을 같은 순서로 본다. 이 가정이 틀리면 대응이 어긋나는 위치로 드러난다. 미검증
- 작업 2: 저자 영상 구간을 (fold, 피험자 순서, 클래스 0/5/10 순서)로 우리 영상 ID에 대응시킨다. 우리가 제외한 영상은 `ours_excluded:<사유>`로 표시한다. 처음 어긋나는 곳이 있으면 어디서인지만 보고하고 자동으로 재정렬하지 않는다. 대응된 영상마다 저자·우리 시퀀스 수, `count_equal`, `count_equal`이면 X의 `np.allclose(atol=1e-6)`와 최대 절대 차이, y 일치를 `compare\author_vs_ours.csv`에 쓴다. 저자에만 있는 영상, 우리에만 있는 영상은 목록으로 보고한다.
- 회전별(none / cw90 / ccw90 / 180) 집계의 회전은 배치 추출 manifest의 `rotate` 열(추출에 실제 적용된 회전)을 쓴다. 이것이 "최종 회전"과 같다고 보는 것은 [가능성 높음]이다(인벤토리 CSV에는 회전 열이 없고, 방향 조사 CSV의 `resolved_orientation`과 override에서 정해진 값이 추출에 쓰였다). 미검증
- 저자 테스트 배열 안에서 서로 다른 두 영상의 X 블록이 `np.array_equal`로 같은 쌍이 있는지 전수 검사한다(시퀀스 수가 같은 영상끼리만 비교).

## 예측 (결과를 보기 전에 등록)
- P1: 저자 y 런으로 복원한 fold별 영상 수가 start_indices 길이(33, 36, 36, 36, 33)와 같고, 각 fold가 (0,5,10) 반복 구조이다.
- P2 [추측]: 저자 데이터에는 피험자 06(Fold1)과 49(Fold5)가 없고 그 밖의 피험자는 모두 있다. 우리가 제외한 23, 45, 42/5, 55/10, 32의 drowsy에 해당하는 저자 구간이 존재한다. 반증: 오름차순 정렬 가정에서 Fold1 또는 Fold5의 저자 영상 구성이 다른 피험자의 결손을 가리키는 경우.
- P3 [추측]: `count_equal` 비율은 none 영상이 회전 영상(cw90, ccw90, 180)보다 높다. 반증: 회전 영상이 같거나 더 높은 경우.
- P4 [추측]: 저자 테스트 배열에 X 블록이 동일한 영상 쌍이 최소 1쌍 있다(32 또는 49 등에서 파일 누락 시 이전 값 재사용, P-14/P-04). 반증: 0쌍.
- P5 [추측]: `count_equal`인 영상 중 allclose를 만족하는 비율이 `count_equal` 비율보다 낮다(저자 추출 환경과 우리 환경의 수치 차이).
- 판정 규칙: 결과를 본 뒤 적중 여부를 그대로 쓴다. 예측이 빗나가도 정렬 규칙이나 코드를 바꿔 맞추지 않는다. [확실] (사전 정의)


# 저자 .npy 대비 비교 결과 (2026-10-03)

사전 등록(위 "저자 .npy 대비 비교 사전 등록 (2026-10-03)")에 대한 결과다. 구현은 `src/repro/compare_with_author_npy.py`이고, 저자 .npy는 `mmap_mode='r'`로만 열었다. 표와 CSV(`author_vs_ours.csv`, `duplicate_pairs.csv`, `summary.txt`)는 저장소 밖 `SEQ_OUT_DIR\compare\`에만 썼다. 정렬 규칙은 사전 정의한 위치 대응 그대로이며 바꾸거나 재정렬하지 않았다. 저자가 어떤 영상을 제외했는지는 확인된 바 없다. [확실]

## 작업 0: 우리 산출물 점검
- 산출물은 정책 변경(degenerate_reference 변경) 뒤의 실행 결과와 같은 값이다(훈련/테스트 시퀀스 수 Fold1 6554/1328, Fold2 5823/2059, Fold3 6241/1641, Fold4 6676/1206, Fold5 6234/1648). [확실]
- npy 20개의 dtype은 모두 float64이고 X는 (N,30,4), y는 (N,1)이다. 메타데이터 행 수는 훈련·테스트 모두 시퀀스 수와 같고, start_indices는 시작이 0, 증가, 길이 합이 테스트 시퀀스 수와 같으며 영상 수(33, 33, 35, 32, 34)와 개수가 같다. fold 다섯 개 모두 일치했다. [확실]

## 작업 1: 저자 테스트 배열 구조
- 테스트 X, y(`Training.py` 416~417줄이 읽는 파일): Fold1 (1269,30,4), Fold2 (2043,30,4), Fold3 (2196,30,4), Fold4 (1462,30,4), Fold5 (1691,30,4). y는 각각 (N,1)이고 dtype은 모두 float64이다. [확실]
- y를 라벨 런으로 나눈 영상 수는 Fold1~5가 33, 36, 36, 36, 33이다. 모든 fold가 (0,5,10) 반복 구조이고 런 시작 위치가 `Training.py`의 start_indices와 모두 일치했다. [확실] 구조 불일치로 인한 중단은 없었다.
- 피험자 순서: `Preprocessing.py`는 폴더 목록을 `os.listdir`로 얻으므로(146, 150, 47줄) 순서가 코드만으로는 정해지지 않는다. 사전 등록한 기본 가정(폴더 이름 오름차순, 우리 쪽은 (part, id) 오름차순)으로 비교했다. 이 가정이 맞는지는 확인하지 못했다. 미검증

## 작업 2: 위치 대응 비교
- 위치 행 180개 중 비교 161개, 우리 제외 13개(`no_alert_reference` 2, `no_blink_file` 1, `not_extracted` 8, `variant_deferred` 2 칸), 우리에만 있는 칸 6개(Fold1의 피험자 12 세 칸, Fold5의 피험자 60 세 칸), 저자에만 있는 칸 0개. 클래스 라벨 불일치는 0개. [확실]
- 비교 161개 중 시퀀스 수가 같은 것은 29개(0.180)이다. 이 29개 중 14개는 시퀀스가 1개(블록이 앞쪽 0 패딩인 짧은 영상)이고 15개가 2개 이상이다. X가 `np.allclose(atol=1e-6)`를 만족하는 것은 3개(count_equal 중 0.103)이며 세 개 모두 최대 절대 차이가 0이다(Fold1 위치 24~26, 우리 영상 ID 기준 피험자 09의 세 영상). 나머지 count_equal 영상의 최대 절대 차이는 중앙값 24.9, 최대 357이다. y는 count_equal인 영상에서 모두 같았다. [확실]
- 회전별(배치 manifest의 `rotate` 열 기준) 비교 수, count_equal(비율), allclose: none 84, 15(0.179), 3; cw90 44, 5(0.114), 0; ccw90 31, 9(0.290), 0; 180 1, 0(0.000), 0; unknown 1(manifest에 줄이 없는 영상 Fold2_part1/17/0), 0, 0. 회전 영상 전체를 합치면 76개 중 14개(0.184)이다. [확실]
- 처음 어긋나는 위치(재정렬 없이 보고): 시퀀스 수 불일치는 Fold1 위치 0(Fold1_part1/01/0), Fold2 위치 1, Fold3 위치 2, Fold4 위치 0, Fold5 위치 1에서 처음 나타난다. 우리 제외 칸의 첫 위치는 Fold1 15(06), Fold2 30(23), Fold3 23(32의 변형), Fold4 16(42/5), Fold5 2(49의 변형)이다. 한쪽에만 있는 칸은 Fold1 위치 33~35(우리에만)와 Fold5 위치 33~35(우리에만)이다. [확실]
- 동일 블록: 저자 테스트 배열 전체(5개 fold, 같은 shape끼리)에서 X 블록이 `np.array_equal`로 같은 쌍이 2개 있다. 둘 다 Fold3 안이다. (위치 23, 26; 시퀀스 93개; 위치는 0부터이고 우리 칸 기준 32의 클래스 10과 33의 클래스 10), (위치 34, 35; 시퀀스 6개; 36의 클래스 5와 10). 다른 fold나 fold 사이의 쌍은 없었다. 이 두 쌍은 `IMPLEMENTATION_DIFF.md` P-14에 이미 적힌 쌍(1부터 센 영상 24와 27, 35와 36)과 같은 위치이다. 우리 영상 ID는 오름차순 가정을 따른 것이다. [확실] (위치), 미검증 (영상 ID)

## 관찰(사전 등록 밖, 대응에 쓰지 않음)
- 위치별 시퀀스 수 표를 읽었을 때, Fold1의 저자 위치 0~14와 18~32가 우리 같은 위치와 값이 비슷하다(예: 52/56, 54/64, 50/50, 8/8, 39/35, 126/130). 즉 위치 15~17(우리 06)에 해당하는 저자 영상(42, 36, 25개)이 있고, 위치상 빠진 쪽은 끝(우리 피험자 12)이다. [가능성 높음] (표 관찰). 저자 데이터에 06이 없다는 P2의 서술과 맞지 않는다.
- Fold5는 저자 위치 0~2(1, 26, 18개)가 우리 피험자 50의 위치 3~5(1, 29, 21개)와 비슷해 피험자 49가 저자 쪽에서 빠진 것과 일관된다. 이 비교는 위치를 3칸 밀어서 본 것이므로 사전 정의한 대응이 아니며 일치율에 넣지 않았다. [추측]
- Fold3는 저자 위치 12 이후가 우리 위치 15 이후와 값이 비슷하다(예: 39/39, 134/134, 203/203). 오름차순 가정이 이 fold에서 맞지 않거나 피험자 구성이 다를 가능성이 있다. [추측]. 미검증
- 위 세 관찰은 대응 규칙을 바꾸는 근거로 쓰지 않았고, 규칙을 바꿀지는 사용자의 결정을 기다린다.

## 예측 대조
- P1: 적중. fold별 영상 수 33, 36, 36, 36, 33이고 모든 fold가 (0,5,10) 반복이며 경계도 start_indices와 일치했다. [확실]
- P2 [추측]: 부분 적중, 부분 반증. 영상 수는 Fold2, 3, 4가 12명 × 3(= 36)으로 완전한 구성이고 Fold1, 5는 한 명이 모자란 11명 × 3(= 33)이어서 "Fold1과 Fold5에 한 명씩 빠졌다"와 일관된다. 그러나 오름차순 가정에서 Fold1은 06이 아니라 끝의 피험자 12 쪽이 빠진 모양이어서 반증 조건에 해당한다(관찰, [가능성 높음]). Fold5는 49가 빠진 모양과 일관되나 위치를 밀어 본 관찰이라 확정할 수 없다([추측]). 우리가 제외한 23, 45, 42/5, 32의 drowsy에 해당하는 저자 구간은 Fold2~4가 36개로 완전해서 존재하는 것과 일관된다. 55/10은 Fold5가 33개라 판단할 수 없다. 미검증
- P3 [추측]: 반증. none 0.179, 회전 영상 합계 0.184(cw90 0.114, ccw90 0.290, 180은 1개)이므로 회전 영상이 같거나 더 높다. 단 count_equal 29개 중 14개가 시퀀스 1개짜리의 자명한 일치이고 대응 자체도 확정되지 않아 이 비율의 의미는 약하다. [확실] (수치), 해석은 미검증
- P4 [추측]: 적중. 동일 블록 쌍이 2개이다(위 작업 2). 둘 다 P-14의 쌍과 같고 새 쌍은 없다. 원인(파일 누락 시 이전 값 재사용)은 확인하지 못했다. 위치 23과 26의 쌍은 인접 피험자의 같은 클래스(10) 사이이고, 위치 34와 35의 쌍은 같은 피험자의 클래스 5와 10 사이이다. 후자는 "이전 값 재사용" 하나로는 설명되지 않는다. 미검증
- P5 [추측]: 문자 그대로는 적중. allclose 비율 0.103(count_equal 중)이 count_equal 비율 0.180보다 낮다. 그러나 allclose를 만족한 3개는 최대 차이 0(완전 일치)이고, 시퀀스 수가 같은 나머지 12개(2개 이상)는 최대 차이 4.2~357이다. 이 크기는 SVM pickle이나 OpenCV 버전 차이에서 나오는 작은 수치 차이로 보기 어렵다. 그 설명은 뒷받침되지 않았다. 미검증
- 공통: 우리 환경(Python 3.6.13, cv2 3.4.5, dlib 19.17.0, sklearn 0.19.2, 배치 manifest 첫 줄 기록)과 저자 추출 환경의 차이는 확인하지 못했다. 미검증

## 의사결정이 필요한 것 (결정 대기)
- 우리 수동 플래그 영상(45 전체, 42/5, 55/10, 06 전체)은 우리 쪽에서 `not_extracted`로 제외되었고 저자 배열에는 대응하는 칸이 있다(Fold4는 저자 36개). 이 영상들을 추출할지는 사용자의 결정을 기다린다.
- 피험자 49는 이번 정책(저자 PP와 같은 계산)으로 alert, semisleepy가 포함되어 있다. 저자 쪽에서 빠진 것으로 보이는 관찰이 있다([추측]). 유지할지 뺄지는 사용자의 결정을 기다린다.
- 대응 규칙(오름차순 가정)을 바꿔 다시 비교할지는 사용자의 결정을 기다린다. 이번에는 바꾸지 않았다.


# 정렬 없는 일치 탐색 사전 등록 (2026-10-03)

## 목적과 범위
- 앞 비교(위치 대응)는 "저자 피험자 순서가 폴더 이름 오름차순이다"라는 가정에 의존해 약했다. 이번에는 순서 가정 없이 저자 테스트 배열의 영상 블록(y 런 기준 174개)마다 우리 포함 영상 167개 전체(모든 fold)와 값으로 직접 비교한다. 읽기 전용이고 저자 .npy는 `mmap_mode='r'`로만 열며, 결과는 `SEQ_OUT_DIR\compare\`의 새 파일 이름으로만 쓴다(이미 있으면 중단). 코드는 기존 `compare_with_author_npy.py`의 새 모드(`--match-free`)이다. [확실] (사전 정의)
- 우리 영상 블록은 우리 테스트 npy를 start_indices로 잘라 얻는다(영상 ID는 메타데이터 `video_id` 순서). shape이 같은 후보만 비교한다.
- 억지로 대응시키지 않는다. 일치하지 않는 것은 "none"으로 둔다. 저자가 어떤 영상을 제외했는지는 확인된 바 없다.

## 판정 기준 (고정, 결과를 본 뒤 바꾸지 않는다)
- (E) 정확 일치: `np.allclose(atol=1e-6)`이고 y 일치.
- (A) 아핀 일치: (E)가 아니고, 특징 4개 열 각각에서 저자 값과 우리 값을 전 시퀀스·전 시점으로 펼친 벡터의 상관계수 |r| ≥ 0.99999. 한쪽 열의 분산이 0이면 그 열은 양쪽 모두 분산 0일 때만 통과로 본다. A에서는 y 일치를 요구하지 않는다(우리 영상 ID에 클래스가 들어 있어 표에서 확인할 수 있다).
- 시퀀스가 1개(0 패딩)인 블록은 아핀 판정이 성립하지 않으므로 "자명" 범주로 따로 개수만 보고하고 E만 시도한다.
- 일치 종류 집계는 한 저자 블록에 후보가 여러 개이면 모두 기록하고, 종류는 가장 강한 것(E, A, none 순)으로 둔다.
- 복원한 피험자 순서: 저자 fold의 위치 3칸(= 피험자 1명)마다 그 칸의 블록들이 일치한 우리 영상의 (fold, 피험자 id) 집합을 구한다. 일치가 없으면 unknown이다. 이를 (part, id) 오름차순 가정과 비교한다.
- 동일 X 블록 쌍(앞 비교에서 찾은 2쌍)은 X만 `np.allclose(atol=1e-6)`로 비교해 일치하는 우리 영상 ID를 보고한다. 한쪽 위치가 "정상 슬롯"이라는 것은, 그 위치의 칸(피험자 3칸)에 있는 쌍이 아닌 다른 블록들이 일치한 우리 피험자 집합에 일치한 우리 영상의 피험자가 들어 있고, 다른 위치의 칸에는 들어 있지 않은 경우로 정한다. 다른 쪽은 사본이다.
- 우리 영상 기준(시퀀스 2개 이상인 포함 영상)으로 회전(배치 manifest의 `rotate` 열)별 E 일치율, A 일치율, 일치 없음 비율을 낸다. 우리 영상이 어떤 저자 블록과 E 또는 A로 일치하면 일치로 센다.

## 예측 (결과를 보기 전에 등록)
- R1 (점검용): 피험자 09의 세 영상이 E 일치로 재확인된다.
- R2 [추측]: E 또는 A로 일치하는 비율이 none 영상에서 회전 영상(cw90, ccw90, 180)보다 높다. 반증: 회전 영상이 같거나 더 높은 경우(저자 환경이 회전을 처리했을 가능성).
- R3 [추측]: 저자 Fold1의 피험자 구성은 06 포함 11명(01~11)이고 12는 없다. 반증: 12의 영상이 저자 Fold1 블록과 일치하거나, 06 자리의 저자 블록이 다른 피험자와 일치하는 경우. (06은 우리가 추출하지 않았으므로 06 자체와는 일치할 수 없다. 이 예측은 반증 조건으로만 검증하고 06의 존재는 확인하지 못한다.)
- R4 [추측]: 저자 Fold5에는 49가 없다. 반증: 49의 영상이 저자 Fold5 블록과 일치하는 경우.
- R5 [추측]: 동일 블록 2쌍 중 적어도 한 쌍은 두 위치 중 한쪽만 우리 영상 하나와 정상 슬롯으로 일치하고 다른 쪽은 그 영상의 사본이다. 반증: 어느 쪽도 우리 영상과 일치하지 않는 경우.
- R6 [추측]: E나 A로 일치하지 않는 저자 블록이 1개 이상 있다. 반증: 0개.
- 판정 규칙: 결과를 본 뒤 적중 여부를 그대로 쓴다. 판정 기준과 허용 오차는 바꾸지 않는다. [확실] (사전 정의)


# 정렬 없는 일치 탐색 결과 (2026-10-03)

사전 등록(위 "정렬 없는 일치 탐색 사전 등록 (2026-10-03)")에 대한 결과다. 구현은 `src/repro/compare_with_author_npy.py`의 `--match-free` 모드이고, 판정 기준(E, A, 허용 오차)은 사전 등록 그대로이며 바꾸지 않았다. 저자 .npy는 `mmap_mode='r'`로만 열었고 결과는 `SEQ_OUT_DIR\compare\`의 `match_free.csv`, `match_free_summary.txt`에만 썼다. 저자가 어떤 영상을 제외했는지는 확인된 바 없다. [확실]

## 시험
- 구현 점검(합성 블록, 저장소 밖): 열마다 크기·이동을 바꾼 블록은 A, 같은 블록은 E, 무관한 블록은 일치 없음, 시퀀스 1개 블록은 아핀 변환해도 일치 없음, 같은 열이 상수인 경우는 양쪽 모두 상수일 때만 A, 한쪽만 상수이면 일치 없음으로 판정되었다. [확실]

## 결과 요약
- 저자 블록 174개를 우리 포함 영상 167개 전체와 비교했다. E 3개, A 0개, 일치 없음 171개이다. 자명(시퀀스 1개) 블록 21개 중 E는 0개이다. [확실]
- E 일치 3개는 저자 Fold1 위치 24, 25, 26(클래스 0, 5, 10; 시퀀스 23, 35, 51개)과 우리 영상 Fold1_part2/09/0, 09/5, 09/10이다(최대 절대 차이 0). 이 세 영상 외에는 E도 A도 일치하는 저자 블록이 없다. [확실]
- 일치 못한 저자 블록은 171개(fold별로 Fold1 30개, Fold2 36개, Fold3 36개, Fold4 36개, Fold5 33개), 일치 못한 우리 영상은 164개이다. 목록은 `match_free_summary.txt`에 있다.
- 복원한 피험자 순서: 일치가 있는 칸은 Fold1의 저자 위치 24~26 한 칸(피험자 09)뿐이고 이 칸은 오름차순 가정의 같은 칸(0부터 8번째)과 같다. 나머지 칸은 모두 unknown이다(Fold1 10칸, Fold2~4 각 12칸, Fold5 11칸). [확실]
- 동일 X 블록 쌍 2개(Fold3 위치 23과 26, 위치 34와 35)는 X만으로도 일치하는 우리 영상이 없다(두 쌍 모두 "없음"). 따라서 어느 위치가 정상 슬롯이고 어느 쪽이 사본인지는 판정할 수 없었다. [확실]
- 회전별(우리 영상, 시퀀스 2개 이상 142개): none 79개 중 E 3(0.038), A 0, 없음 76(0.962); 회전 영상 합계 62개(cw90 40, ccw90 21, 180 1) 중 E 0, A 0, 없음 62(1.000); 회전 정보가 없는 1개(manifest에 줄이 없는 Fold2_part1/17/0)는 일치 없음. [확실]

## 예측 대조
- R1 (점검용): 적중. 피험자 09의 세 영상이 E로 일치했다. [확실]
- R2 [추측]: 문자 그대로는 적중(none 0.038, 회전 영상 0.000). 그러나 일치한 3개가 모두 한 피험자(09)의 영상이어서 사실상 표본이 피험자 1명이고, 이 비교는 근거로 약하다. 해석은 미검증
- R3 [추측]: 반증되지 않았으나 확정하지도 못했다. 12의 영상이 저자 Fold1 블록과 일치하지 않았고, 06 자리(위치 15~17)의 저자 블록도 다른 피험자와 일치하지 않았다(반증 조건 둘 다 해당 없음). 한편 09가 저자 Fold1의 0부터 8번째 칸에 있으므로 오름차순이면 그 앞에 8명이 있고 이는 01~08(06 포함)과 일관된다. 저자 Fold1은 11명이므로 빠진 한 명은 09 뒤(10~12 중 하나)이다. 12가 빠진 것인지는 이 결과만으로 알 수 없다. 오름차순 가정 아래의 해석이며 [가능성 높음], 12가 없다는 것은 미검증
- R4 [추측]: 반증되지 않았으나 확정하지도 못했다. 49의 영상은 저자 Fold5 블록과 일치하지 않았다(반증 조건에 해당 없음). 단 Fold5의 저자 블록은 33개 모두 우리 영상과 일치하지 않았으므로 49가 없다는 증거로 보기 어렵다. 미검증
- R5 [추측]: 반증. 동일 블록 2쌍 모두 어느 위치도 우리 영상과 일치하지 않는다. 미검증 (사본의 원본이 어떤 영상인지)
- R6 [추측]: 적중. E나 A로 일치하지 않는 저자 블록이 171개이다. [확실]

## 해석 (근거의 강도를 구분)
- 일치하는 영상은 피험자 09 한 명뿐이다. 이 세 영상은 같은 txt에서 같은 계산을 하면 저자 .npy와 값까지 같아진다는 것을 보여 주므로, 시퀀스 생성(정규화 열 순서, 기준 구간, 윈도우, 패딩)이 저자 PP와 같은 계산을 한다는 근거가 된다. [가능성 높음]. 이 세 영상이 이 근거의 전부이다.
- 나머지 171개는 아핀 일치(정규화 통계만 다른 같은 blink 데이터)도 없다. 정규화 차이로 설명할 수 있는 불일치는 이 데이터에서 찾지 못했다. 그러므로 나머지 영상의 불일치는 원 blink 데이터(추출 단계)의 차이로 보는 쪽이 자연스럽다. 이는 [가능성 높음]이고 원인(추출 환경, 영상 판본, 회전 처리, 저자의 다른 선택)은 확인하지 못했다. 미검증
- 위 해석은 유사한 시퀀스 수(앞 비교에서 일부 위치의 시퀀스 수가 가까운 것)가 같은 영상의 서로 다른 추출 결과일 가능성과 일관되지만, 이번 탐색은 값 일치만 판정하므로 이를 확인하지는 않았다. [추측]


# 클래스별 풀링 통계와 단계적 일치(탐색적) 사전 등록 (2026-10-03)

## 성격과 범위
- 이 분석은 **탐색적**이다. 앞의 E/A 판정(사전 계획한 정확·아핀 일치)이 비트 단위 일치만 잡아 "같은 영상에서 조금 다르게 추출된 경우"를 구분하지 못하므로, 근사 일치를 따로 본다. 아래 기준과 허용값은 결과를 보기 전에 고정하며(2026-10-03, 이 항목 작성 시점) 이후 바꾸지 않는다. [확실] (사전 정의)
- 읽기 전용이다. 저자 .npy는 `mmap_mode='r'`로만 열고, 결과는 `SEQ_OUT_DIR\compare\`의 새 이름(`graded.csv`, `graded_summary.txt`)으로만 쓴다(이미 있으면 중단). 코드는 기존 `compare_with_author_npy.py`의 새 모드 `--graded`이다. 영상은 열지 않는다. 저자가 어떤 영상을 제외했는지는 확인된 바 없다.

## 작업 1: 클래스별 풀링 통계 (고정 정의)
- 저자와 우리의 테스트 배열 전체(5개 fold)에서 시퀀스 1개(0 패딩) 블록을 제외하고 클래스(0/5/10)별로 (a) 영상 수, 영상당 시퀀스 수의 평균과 중앙값, (b) 4개 특징(X의 열 0~3)의 평균과 표준편차(모든 시퀀스, 모든 시점을 펼친 값)를 비교한다. 우리 쪽은 포함 영상(167개) 중 시퀀스 2개 이상인 영상이다.

## 작업 2: 단계적 일치 (고정 정의)
- blink 행 복원: 시퀀스 수 N ≥ 2인 블록에서 첫 윈도우 30행과 이후 각 윈도우의 마지막 2행을 이어 붙인다(길이 28 + 2N).
- 쌍 비교: 저자 블록(N ≥ 2)과 우리 영상(N ≥ 2)의 모든 쌍(클래스, fold 제한 없음)에 대해 shift s ∈ {−3..3}행을 허용한다. 저자의 i번째 행과 우리의 (i+s)번째 행을 맞추고, 겹치는 구간의 길이가 30 이상일 때만 판정한다. 특징 amplitude, duration, velocity(X의 열 인덱스 1, 2, 3)의 피어슨 상관을 구한다. 누적 빈도 열(인덱스 0)은 완만한 추세라 우연히 상관이 높아 제외한다. 한쪽 열의 분산이 0이면 그 열의 상관은 0으로 둔다.
- 세 상관의 평균이 최대인 shift를 택하고, 세 상관이 모두 0.9 이상이면 "근사 일치"로 판정한다. 한 저자 블록에 후보가 둘 이상이면 모두 기록한다(대표 후보는 세 상관 평균이 가장 높은 것).
- 점검: 정확 일치 3개(우리 Fold1_part2/09의 세 영상)는 shift 0에서 상관이 1.0(부동소수점 오차 수준)으로 나와야 한다.
- 보고: 근사 일치한 저자 블록 수(N ≥ 2 블록 대비 비율), 대표 후보 쌍의 복원 길이 차이 |Δn|/n(n은 저자 복원 길이)의 분포(중앙값, 최대), 후보가 둘 이상인 모호한 블록 수, 회전별(우리 영상 기준, 근사 일치한 저자 블록이 하나 이상 있는 영상의 비율) 근사 일치율. 근사 일치로 복원한 fold별 피험자 순서를 오름차순과 비교하고, 동일 X 블록 2쌍이 어느 우리 영상에 근사 일치하는지 보고한다. 억지로 대응시키지 않는다.

## 예측 (결과를 보기 전에 등록)
- Q1 [추측]: 클래스 10과 0의 평균 차이(10 − 0)의 부호가 4개 특징 모두에서 저자와 우리가 같다. 반증: 하나라도 다른 경우.
- Q2 [추측]: 영상당 시퀀스 수의 평균이 클래스별로 저자 대비 ±20% 이내이다(우리 평균 / 저자 평균이 0.8~1.2). 반증: 한 클래스라도 벗어나는 경우.
- Q3 [추측]: 저자 블록(N ≥ 2) 중 30% 이상이 근사 일치를 가진다(차이가 작은 섭동 수준이다). 반증: 30% 미만.
- Q4 [추측]: 근사 일치 쌍(피험자 09 제외)의 |Δn|/n 중앙값이 10% 미만이다. 반증: 10% 이상. 09를 뺀 근사 일치 쌍이 없으면 판정 불가로 쓴다.
- Q5 [추측]: 근사 일치율은 none 영상이 회전 영상(cw90, ccw90, 180 합계) 이상이다. 반증: 회전 영상이 더 높은 경우.
- 판정 규칙: 결과를 본 뒤 적중 여부를 그대로 쓴다. 기준과 허용값은 바꾸지 않는다. 근사 일치는 같은 영상이라는 증명이 아니라 상관이 높다는 사실이다. [확실] (사전 정의)


# 클래스별 풀링 통계와 단계적 일치(탐색적) 결과 (2026-10-03)

사전 등록(위 "클래스별 풀링 통계와 단계적 일치(탐색적) 사전 등록 (2026-10-03)")에 대한 결과다. 이 분석은 **탐색적**이고, 기준과 허용값(shift −3..3행, 겹침 30 이상, 상관 열 1~3, 세 상관 모두 0.9 이상)은 사전 등록 시점에 고정했으며 결과를 본 뒤 바꾸지 않았다. 구현은 `compare_with_author_npy.py`의 `--graded` 모드이고, 결과는 저장소 밖 `SEQ_OUT_DIR\compare\`의 `graded.csv`, `graded_summary.txt`에만 썼다. 저자가 어떤 영상을 제외했는지는 확인된 바 없다. [확실]

## 작업 1: 클래스별 풀링 통계 (시퀀스 2개 이상 블록; 저자 153개, 우리 142개)
- 영상 수와 영상당 시퀀스 수(평균, 중앙값): 저자 클래스 0은 42개(38.9, 27.0), 5는 54개(64.6, 48.0), 10은 57개(61.7, 50.0). 우리 클래스 0은 39개(35.8, 25.0), 5는 51개(63.9, 47.0), 10은 52개(61.6, 51.5). 영상당 시퀀스 수 평균의 비(우리/저자)는 클래스 0이 0.921, 5가 0.989, 10이 0.998이다. [확실]
- 특징 평균(freq, amp, dur, vel): 저자 클래스 0 (−2.155, 0.162, 0.030, 0.179), 5 (26.330, 0.088, 0.461, −0.032), 10 (23.330, −0.302, 1.132, −0.392). 우리 클래스 0 (−0.995, 0.140, 0.068, 0.125), 5 (41.105, 0.039, 0.453, −0.027), 10 (34.559, −0.607, 1.007, −0.681). [확실]
- 특징 표준편차: 저자 클래스 0 (9.949, 1.114, 1.130, 1.089), 5 (51.128, 1.700, 2.371, 1.626), 10 (43.341, 1.650, 5.191, 1.661). 우리 클래스 0 (10.282, 1.113, 1.114, 1.111), 5 (67.043, 1.391, 2.956, 1.449), 10 (53.566, 1.578, 6.799, 1.497). [확실]
- 클래스 10과 0의 평균 차이(10 − 0): 저자 (25.485, −0.463, 1.102, −0.571), 우리 (35.554, −0.747, 0.939, −0.805). 네 특징 모두 부호가 같다. 크기는 freq, amp, vel에서 우리가 더 크고(freq 약 1.4배, amp 약 1.6배, vel 약 1.4배) dur는 우리가 더 작다(약 0.85배). 이 차이의 원인은 피험자 구성(우리는 일부 영상이 빠짐), 추출 결과의 차이, 정규화 기준 차이 중 어느 것인지 확인하지 못했다. 미검증

## 작업 2: 단계적(근사) 일치
- 점검: 정확 일치 3개(우리 Fold1_part2/09의 세 영상)는 shift 0, 세 상관이 모두 1.0000000(소수 8자리)이었다. [확실]
- 근사 일치한 저자 블록은 6/153(0.039)이고 후보가 둘 이상인 모호한 블록은 없다. 6개는 09의 세 블록과 새로 나온 세 블록(저자 Fold3 위치 15, 16, 17 = 클래스 0, 5, 10; 시퀀스 134, 103, 203개)이다. 새 세 블록은 우리 Fold3_part2/31의 세 영상(시퀀스 134, 102, 203개)과 shift 0에서 amplitude 상관 1.00000(소수 5자리), duration 0.9994/0.9997/1.0000, velocity 0.9996/0.9991/0.9999이다. [확실]
- 이 세 쌍은 amplitude가 아핀 변환 수준으로 일치하고 duration, velocity가 아주 조금 다르다. 앞의 E/A 판정(비트 단위 허용 오차 1e-6, 열마다 |r| ≥ 0.99999, 누적 빈도 열 포함)으로는 잡히지 않는 종류의 일치이다. 같은 영상에서 다른 추출이 아주 조금 다른 값을 내는 경우일 가능성이 있다. [추측]. 미검증
- 대표 후보 쌍의 복원 길이 차이 |Δn|/n(09 제외 3쌍): 중앙값 0.000, 75퍼센타일 0.004, 최대 0.009(저자 복원 길이 기준; 클래스 5에서 234 대 232). [확실]
- 회전별(우리 영상, 시퀀스 2개 이상) 근사 일치율: none 79개 중 6(0.076), 회전 영상 합계 62개(cw90 40, ccw90 21, 180 1) 중 0, 정보 없음 1개 중 0. 근사 일치한 6개는 모두 회전 none 영상이다. [확실]
- 복원한 피험자 순서: Fold1은 저자 0부터 8번째 칸이 09(오름차순 가정의 같은 칸). Fold3은 저자 0부터 5번째 칸이 31인데 오름차순 가정에서 31은 0부터 6번째 칸이다. 즉 저자 Fold3에서는 31 앞에 5명뿐이다. 나머지 칸은 모두 unknown이고 Fold2, 4, 5는 일치가 없다. [확실] (일치), 가정과의 어긋남의 원인(빠진 피험자인지, 순서가 오름차순이 아닌지)은 미검증
- 동일 X 블록 2쌍(Fold3 위치 23과 26, 34와 35)은 네 위치 모두 근사 일치하는 우리 영상이 없다. [확실]

## 예측 대조
- Q1 [추측]: 적중. 10 − 0 평균 차이의 부호가 4개 특징 모두 저자와 우리가 같다. [확실]
- Q2 [추측]: 적중. 영상당 시퀀스 수 평균의 비가 0.921, 0.989, 0.998로 모두 ±20% 이내이다. [확실]
- Q3 [추측]: 반증. 근사 일치한 저자 블록은 6/153(0.039)로 30% 미만이다. [확실]
- Q4 [추측]: 적중이나 표본이 작다. 09를 뺀 근사 일치 쌍이 3개(모두 한 피험자 31)이고 |Δn|/n 중앙값이 0.000으로 10% 미만이다. 표본이 피험자 1명이라 일반화할 수 없다. 미검증
- Q5 [추측]: 문자 그대로는 적중(none 0.076, 회전 0.000). 일치한 영상이 피험자 2명(09, 31)뿐이라 회전과의 관련을 판단할 수 없다. 미검증

## 해석 (근거의 강도를 구분)
- 풀링 수준에서는 저자와 우리가 비슷하다(영상당 시퀀스 수 평균, 클래스 간 특징 차이의 부호). 개별 영상 수준에서는 거의 일치하지 않는다(근사 일치 6/153). 이는 데이터의 거시 통계는 비슷하지만 대부분의 영상에서 개별 blink 시계열이 다르다는 사실과 일관된다. [확실] (수치), 원인은 미검증
- 피험자 09는 비트 단위로, 31은 amplitude 일치와 duration·velocity의 아주 작은 차이로 일치한다. 31의 경우 추출이 거의 같지만 완전히 같지는 않다는 뜻일 수 있다. 다른 영상에서 같은 종류의 작은 차이가 있는지는 이 탐색의 기준(0.9)이 느슨해도 6개뿐이므로 거의 없다. [추측]
- 시퀀스 수가 비슷한 영상(앞 비교에서 위치별로 가까운 값)이 상관 0.9에 미치지 못했다는 것은 시퀀스 수의 유사성이 같은 영상이라는 증거가 아니라는 뜻이다. [가능성 높음]


# 기준선 학습 러너(B1) 구현 사전 등록 (2026-10-03)

## 목적과 범위
- 우리 시퀀스 산출물(`SEQ_OUT_DIR`의 npy와 start_indices JSON)로 저자 `Training.py`와 같은 학습·평가를 하는 러너 `src/repro/reproduction_train_baseline.py`를 만든다. `Training.py`는 텍스트로 읽어 메모리에서만 치환한다(`third_party`는 수정하지 않는다). 에이전트는 학습을 실행하지 않는다(실제 학습은 사용자가 한다). 하이퍼파라미터(80 epoch, batch 64, lr 0.000053, 4×32 층, L2 0.1, Δ=1.253)는 바꾸지 않는다. 출력은 환경변수 `TRAIN_OUT_DIR`(저장소 밖, 기본값 없음; 저장소 안이거나 `my_model1~5` 폴더이면 중단)에만 쓰고 이미 있는 출력은 덮어쓰지 않고 중단한다. [확실] (사전 정의)

## 1단계 점검 결과 (코드를 쓰기 전, 읽기 전용)
- 기존 재학습 래퍼는 `src/repro/train_fold_seeds.py`(Fold1 한정 선행 버전은 `reproduction_train_fold1.py`, `train_fold1_seeds.py`)이다. 텍스트를 읽어 메모리에서만 치환해 exec하고, 치환마다 기대 개수를 검증한다. 시드는 `random.seed`, `np.random.seed`와 `tf.reset_default_graph()` 다음 줄의 `tf.set_random_seed`이다(원본은 시드를 설정하지 않는다). 지표는 원본 함수(`vote_accuracy_per_video`, `regression_per_video`, `regression_per_blink`)의 반환값을 감싸 기록하며 `metrics.csv` 열은 `epoch, train_loss_sum, train_moving_acc, BSA_loss, BSA_acc, VA, BSRE, VRE`이고 보고값은 마지막 epoch이다. 이 래퍼는 저자 npy와 저자 start_indices를 쓰므로 우리 데이터에는 그대로 쓸 수 없다. [확실] (코드 읽기)
- `Training.py` 줄 번호: 데이터 로드 414~417(상대경로), 2차 정규화 420~431, start_indices 444~498(fold별 하드코딩), `np.asarray` 502, 하이퍼파라미터 503~506, 모델 `Network` 146~258과 `Train` 288~400, L2 계수 0.1은 333, 체크포인트 저장 `save_variables` 283~286(호출 396, 학습 끝에 한 번), 저장 경로 316, 지표 13~131(임계값 3.34, 대표값 3.3/6.6), 영상 다수결 87~117(동률이면 평균), 저장 `np.save(open('./...'))` 6곳 510~515. 시드 설정은 없다. [확실] (코드 읽기)
- 우리 산출물: npy 이름이 저자가 읽는 이름과 같고(`Blinks_30_Fold<k>.npy` 등 20개), start_indices JSON은 정수 리스트이며 첫 값이 0이고 마지막 영상이 끝까지 이어지는 점이 `Training.py`와 같다(길이 33, 33, 35, 32, 34). [확실]
- 1단계에서 본 구조는 구현 가정과 크게 다르지 않아 구현을 진행한다.

## 치환 계획 (기대 개수는 모두 1, `open('./`는 6)
- `tf.reset_default_graph()` 다음 줄에 `tf.set_random_seed(시드)`(304), `variable_path='./'` → 출력 폴더(316), `y_v[idx]=...` 다음에 epoch 기록 호출(390), `for i in range(5):` → 지정 fold 하나(408), `ii=i    #` → `ii=i+1    #`(411, 체크포인트 이름 `my_model<fold>`), `np.load` 4곳의 경로를 데이터 폴더로(414~417), `start_indices=np.asarray(start_indices)` 앞에서 우리 JSON 값으로 덮어쓰기(502; 하드코딩 블록 444~498은 그대로 두고 덮어씀), `open('./` 6곳 → 출력 폴더(510~515), `--smoke`일 때만 `num_epochs=80` → 지정 값(504). `--smoke`가 없는 실행은 epoch 치환을 하지 않고 하이퍼파라미터 문자열이 저자 값 그대로인지 검사한다. [확실] (계획)

## 예측 (결과를 보기 전에 등록)
- (a) `--dry-run`에서 치환 개수가 기대와 같고 `third_party`의 `Training.py`는 해시가 변하지 않는다. [추측]
- (b) 우리 start_indices JSON의 합이 fold별 테스트 시퀀스 수와 같고, 치환된 소스에서 그 값이 쓰인다. [추측]
- (c) 치환된 소스가 컴파일(py_compile 상당)을 통과한다. [추측]
- (d) `TRAIN_OUT_DIR`이 저장소 안이거나 `my_model` 폴더이면 중단한다(설정이 없을 때도 중단). [추측]
- (e) 학습은 실행하지 않는다(스모크 포함). [확실] (사전 정의)
- 판정 규칙: 결과를 본 뒤 적중 여부를 그대로 쓴다. 학습을 실행하지 않았으므로 실제 학습 동작(TF 실행, 체크포인트 저장, 지표 기록, `--smoke` 실행)은 미검증으로 남긴다. [확실] (사전 정의)


# 기준선 학습 러너(B1) 구현 결과 (2026-10-03)

사전 등록(위 "기준선 학습 러너(B1) 구현 사전 등록 (2026-10-03)")에 대한 결과다. 구현은 `src/repro/reproduction_train_baseline.py`이다. 학습은 실행하지 않았다(스모크 포함). `third_party`는 수정하지 않았고, 시험 출력물은 저장소에 만들지 않았다. [확실]

## 예측 대조
- (a) 적중. `--dry-run`(Fold1~5, 실제 `SEQ_OUT_DIR`을 읽기 전용으로 사용)에서 치환 항목의 개수가 모두 기대와 같았다(`seed(304)`, `save_path(316)`, `epoch_hook(390)`, `fold_loop(408)`, `model_name(411)`, `load_*(414~417)`는 각 1번, `start_indices(502)` 1번, `save_files(510-515)` 6번). `third_party/Training.py`의 해시는 실행 전후 같았고 보호 목록(`common.check_manifest`, 44개 파일) 검사도 통과했다. 비스모크 실행은 `num_epochs`를 치환하지 않고, `--smoke --epochs 2`는 `num_epochs=80`을 504줄에서 1번 치환했다. [확실]
- (b) 적중. 우리 start_indices JSON은 fold별 길이 33, 33, 35, 32, 34이고 영상 구간 길이의 합이 테스트 시퀀스 수(1328, 2059, 1641, 1206, 1648)와 같으며(시작 0, 증가, 마지막 영상은 끝까지), 구간마다 라벨이 하나(0, 5, 10)임을 확인했다. 치환된 소스에 그 값이 `start_indices=[...]`로 들어간다. [확실]
- (c) 적중. 치환된 소스의 정의부와 fold 반복부가 모두 컴파일을 통과했다(5개 fold). 메모리에서 원본과 비교하면 변경은 계획한 8개 구간뿐이다(삭제 13줄, 추가 16줄, 원본 위치 304~305 삽입, 316, 390~391 삽입, 408, 411, 414~417, 501~502 삽입, 510~515). [확실]
- (d) 적중. `TRAIN_OUT_DIR`이 설정되지 않았거나, 저장소 안이거나, `my_model1~5` 이름의 폴더·그 하위·`my_model1~5` 체크포인트 파일이 든 폴더이면 중단했다(함수와 CLI 모두, 폴더는 만들어지지 않음). 정상 경로는 통과했다. 이미 있는 fold 출력 폴더가 있으면 덮어쓰지 않고 중단했고 폴더 내용은 그대로였다. [확실]
- (e) 적중. 학습은 실행하지 않았다. `--dry-run` 후 `tensorflow`가 import되지 않았음을 확인했다. [확실]
- 기타 시험(합성 변조): 원본에서 패턴이 2번 또는 0번 나오도록 바꾼 소스는 치환 개수 불일치로, 하이퍼파라미터 문자열(lr)을 바꾼 소스는 가드로 거부했다. `--epochs`를 `--smoke` 없이 주거나, `--smoke`에 `--fold all`을 주거나, `--fold 9`는 거부했다. 점검 22개 중 통과 21개, 1개는 시험 기대값 오류였다(아래). [확실]

## 기록할 점
- 구현 중 하이퍼파라미터 가드 문자열이 너무 일반적이어서(`step_size=30`이 주석에도 6번 나옴) 처음 `--dry-run`이 중단되었다. 가드가 확인하는 값(저자 하이퍼파라미터)은 바꾸지 않고, 문자열을 `Train` 호출부의 한 줄 단위(`batch_size=64,num_epochs=80,`, `hstate_size=[32,32,32,32],num_layers=4,step_size=30,drop_out_p=1.0,`, `lr=0.000053,th=1.253,`, `0.1 * (tf.nn.l2_loss`)로 고쳤다. [확실]
- 시험 하나가 실패로 나왔는데 시험 쪽 오류였다. `difflib.unified_diff`의 자동 무시(빈 줄)가 줄 단위 비교를 어긋나게 해 삭제 16줄, 추가 19줄로 세었고, 처음에 어림잡은 기대값(12, 15)도 틀렸다. 계획에서 따진 기대값(삭제 13줄, 추가 16줄, 구간 8개)을 `SequenceMatcher(autojunk=False)`로 다시 확인해 5개 fold 모두 일치했다. 구현 코드는 이 시험 때문에 바꾸지 않았다. [확실]

## 확인하지 못한 것 (미검증)
- 실제 학습 동작: TensorFlow 1.8에서 치환된 소스의 실행, 체크포인트 저장, `metrics.csv` 기록, `final_metrics.json`과 `run_config.json`의 완료 상태 갱신, `--smoke` 실행과 SMOKE 표식 파일. 미검증
- `ii=i+1`로 체크포인트 이름이 `my_model<fold>`가 되는 것은 기존 래퍼(`train_fold_seeds.py`)와 같은 선택이다. 저자가 원본 `ii=i`로 실제로 어떤 이름의 체크포인트를 만들었는지(`my_model0~4`인지)는 확인하지 못했다. 미검증
- 시드를 고정한 것(원본은 시드가 없다)이 결과에 주는 영향. 미검증


# B1 기준선 학습 결과 집계 사전 등록 (2026-10-04)

## 범위
- 사용자가 실행한 B1 기준선 학습(우리 시퀀스 산출물, 시드 1, 2, 3, 각 5-fold, 80 epoch; 러너는 `src/repro/reproduction_train_baseline.py`)의 결과를 읽기 전용으로 집계한다. 집계 도구는 `src/repro/summarize_training.py`이다. 학습은 실행하지 않고, 학습 출력 폴더의 파일은 읽기만 한다. 표와 CSV는 학습 출력 루트 밖의 집계 폴더(인자로 지정)에만 쓰고, 저장소 문서에는 요약만 쓴다. 저자가 어떤 영상을 제외했는지는 확인된 바 없다. [확실] (사전 정의)
- 집계 기준은 결과를 보기 전에 아래와 같이 고정하며 이후 바꾸지 않는다.

## 집계 기준 (고정)
- 주 지표: 마지막 epoch(80)의 영상 단위 VA, VRE. 같은 epoch의 BSA acc, BSA loss, BSRE도 함께 표시한다. 최고 epoch 값(fold별 VA 최대, VRE 최소와 그 epoch)은 "낙관적 참고치(테스트 기반 선택)"로 따로 표시하고 주 결과로 쓰지 않는다.
- 표: 시드별 fold별 값, 시드별 5-fold 평균(fold 평균, 가중 없음), 시드 3개의 평균과 표본 표준편차(n−1), fold별 시드 간 평균과 표준편차.
- SMOKE 표식이 있는 폴더는 건너뛰고 건너뛴 폴더 이름을 보고한다. 학습 출력 루트마다 SMOKE가 아닌 실행 폴더가 정확히 하나(fold1~5를 모두 가짐)여야 하며 아니면 중단한다.
- 클래스(0/5/10)별 영상 단위 정확도와 영상 단위 혼동행렬: `metrics.csv`에는 영상별 예측이 없으므로 마지막 epoch에 대해서는 산출 불가로 보고하고 추정하지 않는다. 단 원본 `vote_accuracy_per_video`가 epoch 번호(0부터)가 10의 배수일 때 영상별 `True label is :x and the detected label is y`를 출력하고 러너가 이를 `train.log`에 남기므로, `train.log`의 마지막 기록 epoch(epoch 번호 70, 즉 71번째 epoch)에서 영상별 정답과 예측을 읽어 "참고: 71번째 epoch(마지막 epoch 아님)"로 별도 표시한다. 한 기록 블록의 영상 수가 fold의 영상 수와 다르거나 블록을 찾지 못하면 산출 불가로 쓴다.
- 논문 수치, 공개 체크포인트, 저자 npy 재학습(B0)의 값은 기존 로그와 문서에 적힌 값만 인용하고, 없는 숫자는 "not explicitly reported"로 쓴다. B1(우리 데이터)과 B0(저자 npy)는 서로 다른 데이터셋이므로 "재현 성공/실패"로 판정하지 않고 차이를 있는 그대로만 기록한다. [확실] (사전 정의)

## 정상성 점검 (고정, 보고서 맨 앞에 쓴다. 하나라도 실패하면 집계를 계속하되 실패 항목을 맨 앞에 쓴다)
1. 모든 시드와 fold의 `metrics.csv` 행 수가 80이고 epoch가 1~80으로 이어지는지.
2. `metrics.csv`의 모든 지표 열에 NaN과 inf가 없는지(손실 열 `train_loss_sum`, `BSA_loss` 포함).
3. 3개 시드의 `run_config.json`에서 fold마다 `num_epochs`(80), 하이퍼파라미터 문자열(batch 64, lr 0.000053, 4×32 등), 원본 `Training.py` 해시, 치환 목록(이름, 개수, 줄 번호), 입력 파일 해시, 시퀀스 수가 같고 시드만 다른지(스모크가 아님도 확인). 치환된 소스의 해시(`patched_source_sha256`)는 시드와 출력 경로가 소스 문자열에 들어가므로 같지 않은 것이 정상이며 비교하지 않는다.
4. 시드별 fold별 `run_config.json`의 테스트 시퀀스 수가 우리 start_indices 합(1328, 2059, 1641, 1206, 1648)과 같은지.
5. epoch별 BSA loss가 시작(epoch 1) 대비 끝(epoch 80)에 줄었는지(시작과 끝 값만 비교).
- 참고 점검(실패해도 1~5와 구분해 표시): `final_metrics.json`이 `metrics.csv` 마지막 행과 같은지.

## 예측 (제 추측이며 맞는지 틀리는지 기록하기 위한 것, 결과를 보기 전에 등록)
- (a) 시드 평균 5-fold VA(시드별 5-fold 평균 3개의 평균)는 0.45~0.65 사이이다. [추측]
- (b) 시드 간 5-fold VA 평균의 표준편차(n−1)는 0.02 이상이다. [추측]
- (c) 마지막 epoch VA는 최고 epoch VA보다 낮다(시드별 5-fold 평균 기준으로 3개 시드 모두, fold별 최고 epoch VA 평균과 비교). [가능성 높음]
- (d) 클래스 5(중간)의 영상 정확도가 클래스 0과 10보다 낮다(71번째 epoch 참고치의 3개 시드 합산 기준; 산출 불가이면 판정 불가). [가능성 높음]
- (e) 정상성 점검 1~4는 모두 통과한다. [가능성 높음]
- 판정 규칙: 결과를 본 뒤 적중 여부를 그대로 쓴다. 기준은 바꾸지 않는다. [확실] (사전 정의)


# B1 기준선 학습 결과 집계 (2026-10-04)

사전 등록(위 "B1 기준선 학습 결과 집계 사전 등록 (2026-10-04)")의 기준대로 `src/repro/summarize_training.py`로 집계한 결과다. 학습 출력 폴더는 읽기만 했고, 표와 CSV(`summary.txt`, `fold_table.csv`, `confusion_matrix_epoch71.csv`)는 학습 출력 루트 밖의 집계 폴더에 썼다. 집계 도구는 합성 학습 출력(임시 폴더)으로 먼저 시험했다(17개 점검 통과; 한 점검은 시험 쪽의 오탐이었고 시험 조건을 고쳐 통과, 도구 코드는 그 때문에 바꾸지 않았다). 저자가 어떤 영상을 제외했는지는 확인된 바 없다. [확실]

## 정상성 점검
- 실패 0개. (1) 15개 fold 모두 epoch 1~80의 80행. (2) NaN, inf 없음(손실 열 포함). (3) 시드 1, 2, 3의 구성이 fold마다 같고(epoch 80, 하이퍼파라미터 문자열, 원본 `Training.py` 해시, 치환 줄 번호, 입력 파일 해시, 시퀀스 수) 시드 값만 다르며 스모크가 아니다. (4) 테스트 시퀀스 수가 fold별 1328, 2059, 1641, 1206, 1648과 같다. (5) BSA loss가 모든 시드와 fold에서 epoch 1보다 epoch 80에서 낮다(예: 시드1 fold1 19.17 → 15.87, fold2 20.41 → 9.21). 참고 점검(`final_metrics.json` == `metrics.csv` 마지막 행)도 통과. SMOKE 표식 폴더는 입력에 없었다(건너뛴 폴더 없음). [확실]

## 주 결과: 마지막 epoch(80)의 영상 단위 지표
VA(fold1~5, 5-fold 평균): 시드1 0.485, 0.818, 0.486, 0.562, 0.647, 평균 0.600. 시드2 0.515, 0.727, 0.514, 0.562, 0.618, 평균 0.587. 시드3 0.576, 0.576, 0.486, 0.625, 0.588, 평균 0.570. 시드 평균 0.525, 0.707, 0.495, 0.583, 0.618, 5-fold 0.586. 시드 SD(n−1)는 fold별 0.046, 0.122, 0.016, 0.036, 0.029이고 5-fold 평균의 SD는 0.015(n = 3). [확실]

VRE(fold1~5, 5-fold 평균): 시드1 1.839, 0.210, 1.824, 0.668, 1.734, 평균 1.255. 시드2 2.326, 0.878, 1.827, 0.825, 1.298, 평균 1.431. 시드3 2.004, 1.530, 2.126, 1.636, 1.738, 평균 1.807. 시드 평균 2.057, 0.873, 1.926, 1.043, 1.590, 5-fold 1.498 ± 0.282. [확실]

참고 지표(5-fold 평균, 시드 평균 ± SD): BSA acc 0.517 ± 0.018, BSRE 2.108 ± 0.164. [확실]

낙관적 참고치(테스트 세트로 epoch를 고른 값이며 주 결과가 아니다): 5-fold 평균 VA 마지막 대 fold별 최고 epoch 평균이 시드1 0.600 대 0.648, 시드2 0.587 대 0.648, 시드3 0.570 대 0.623이다. VRE 마지막 대 fold별 최저 평균은 1.255 대 1.062, 1.431 대 0.829, 1.807 대 0.694이다. VA 최고 epoch는 fold와 시드마다 19~75로 흩어져 있다. [확실]

## 클래스별 영상 단위 정확도와 혼동행렬
- 마지막 epoch(80)는 `metrics.csv`에 영상별 예측이 없고 `train.log`도 epoch 번호가 10의 배수일 때만 영상별 예측을 기록하므로(마지막 epoch는 번호 79) 산출 불가이다. 추정하지 않았다. [확실]
- 참고로 `train.log`의 71번째 epoch(epoch 번호 70, 마지막 epoch가 아님) 기록에서 읽었다(영상별 기록의 VA가 `metrics.csv` 71번째 epoch VA와 모든 fold에서 일치함을 확인). 5-fold 합산, 행 = 정답 0/5/10, 열 = 예측 0/5/10. 시드 합산(영상 501개): 정답 0은 125/40/6(영상 171), 정답 5는 35/55/78(영상 168), 정답 10은 14/35/113(영상 162). 클래스별 정확도는 0이 0.731, 5가 0.327, 10이 0.698이다. 시드별 클래스 5 정확도는 0.393, 0.268, 0.321이고 클래스 0은 0.702, 0.702, 0.789, 클래스 10은 0.704, 0.722, 0.667이다. 정답 5가 10으로 예측된 영상이 78개로 가장 많다. [확실] (71번째 epoch 기준)

## 기존 문서에 적힌 값과의 나란히 보기 (재현 성공/실패로 판정하지 않는다)
B1(우리 재추출 데이터)과 B0(저자 npy 재학습)는 서로 다른 데이터셋이다. 인용값의 출처는 논문 Table 2, 3, `docs/IMPLEMENTATION_DIFF.md`, `experiments/02_fold_seeds/aggregate_seeds123.md`이다.
- 5-fold 평균 VA: B1 0.586 ± 0.015, B0 0.556 ± 0.036, 공개 체크포인트 0.6495, 논문 0.652.
- 5-fold 평균 VRE: B1 1.498 ± 0.282, B0 1.953 ± 0.546, 공개 체크포인트 1.158, 논문 1.14.
- 5-fold 평균 BSA acc: B1 0.517, B0 0.506, 공개 0.541, 논문 0.54. BSRE: B1 2.108, B0 2.874, 공개 1.901, 논문 1.90. BSA loss는 논문에 not explicitly reported.
- fold별 VA(B1 시드 평균 / B0 3시드 평균 / 공개 / 논문): Fold1 0.525 / 0.556 / 0.636 / 0.64, Fold2 0.707 / 0.602 / 0.611 / 0.61, Fold3 0.495 / 0.481 / 0.694 / 0.70, Fold4 0.583 / 0.574 / 0.639 / 0.64, Fold5 0.618 / 0.566 / 0.667 / 0.67.
- fold별 VRE(B1 / B0 / 공개 / 논문): Fold1 2.057 / 2.743 / 2.420 / 2.42, Fold2 0.873 / 1.471 / 1.040 / 1.04, Fold3 1.926 / 3.898 / 0.667 / 0.58, Fold4 1.043 / 1.005 / 0.850 / 0.85, Fold5 1.590 / 0.648 / 0.812 / 0.81.
- 차이: B1의 5-fold 평균 VA는 B0보다 0.030 높고 공개 체크포인트보다 0.064 낮다. VRE는 B0보다 0.455 낮고 공개 체크포인트보다 0.340 높다. fold별 테스트 영상 구성이 B1과 저자 데이터(B0, 공개)가 다르므로(우리 테스트 영상 33, 33, 35, 32, 34개, 저자 33, 36, 36, 36, 33개; IMPLEMENTATION_DIFF.md 10절) fold별 값은 같은 영상 집합의 비교가 아니다. [확실] (수치), 원인은 미검증

## 예측 대조 (사전 등록 그대로)
- (a) 시드 평균 5-fold VA는 0.45~0.65 사이: 적중(0.586). [추측]
- (b) 시드 간 5-fold VA 평균의 표준편차가 0.02 이상: 반증(0.015, n = 3). 영상 하나가 fold VA를 0.03 안팎으로 움직이는 점(영상 32~35개)도 같이 본다. [추측]
- (c) 마지막 epoch VA가 최고 epoch VA보다 낮다: 적중(3개 시드 모두 0.600 < 0.648, 0.587 < 0.648, 0.570 < 0.623). 다만 최고 epoch 값은 80개 epoch 중 최대라서 마지막 값보다 작을 수 없는 구조이므로 약한 예측이었다. [가능성 높음]
- (d) 클래스 5의 영상 정확도가 클래스 0과 10보다 낮다: 적중(71번째 epoch 참고치, 시드 합산 0.327 대 0.731, 0.698이고 3개 시드 모두 클래스 5가 가장 낮다). 마지막 epoch 값은 아니다. [가능성 높음]
- (e) 정상성 점검 1~4는 모두 통과: 적중(점검 5와 참고 점검도 통과). [가능성 높음]

## 확인하지 못한 것 (미검증)
- 마지막 epoch(80)의 영상별 예측과 혼동행렬: 산출 불가. 체크포인트(`my_model<fold>`)를 불러 평가하면 얻을 수 있으나 이번에는 하지 않았다. 미검증
- 시드가 3개뿐이라 표준편차는 참고 수준이다. 시드 고정(원본에는 시드가 없다)이 결과에 주는 영향도 알 수 없다. 미검증
- B1과 저자 npy 기반 결과의 차이(영상 구성, 추출 결과, 정규화 기준 차이 중 무엇 때문인지)는 이 집계로는 알 수 없다. 미검증


# B1 체크포인트 영상별 평가 사전 등록 (2026-10-04)

## 목적과 범위
- B1 기준선 학습(시드 1, 2, 3 × 5-fold)의 fold별 체크포인트를 불러 테스트 영상 전체의 영상별 예측을 만든다. 마지막 epoch(80)의 영상별 예측과 혼동행렬이 `metrics.csv`에 없어 산출 불가였기 때문이다(앞의 B1 집계 항목). 도구는 `src/repro/evaluate_baseline_checkpoints.py`이다. 체크포인트 로드와 예측만 하고 학습은 하지 않는다. 학습 출력 폴더의 파일은 읽기만 한다. 결과 표와 CSV는 학습 출력 루트 밖의 집계 폴더(인자)에만 쓴다. 저자가 어떤 영상을 제외했는지는 확인된 바 없다. [확실] (사전 정의)

## 1단계 점검 결과 (코드를 쓰기 전, 읽기 전용)
- 체크포인트는 fold마다 `my_model<k>.data-00000-of-00001`, `my_model<k>.index`, `my_model<k>.meta`, `checkpoint`(TF1 Saver 형식)로 15개 fold 모두 있다. 저장 호출은 `Training.py` 395~396줄로 `for` 루프가 끝난 뒤(들여쓰기 8칸) `save_variables`(283~286)를 한 번 부른다. 최고 성능 기준도 매 epoch 저장도 아니며 마지막 epoch 직후 상태이다(396줄 주석의 "every epoch"는 코드와 맞지 않는다). [확실] (코드 읽기)
- fold 폴더 안 파일의 수정 시각이 모두 같아(복사 흔적으로 보임) 수정 시각으로는 저장 시점을 확인할 수 없다. 이 증거는 쓰지 않는다. 미검증
- 기존 평가 스크립트는 `src/repro/reproduction_eval.py`(공개 체크포인트 5-fold 평가)이다. `Training.py`를 텍스트로 읽어 정의부를 메모리에서 exec하고, 반복부의 `load=False`를 `load=True`로 바꿔 체크포인트를 복원한 뒤 원본 `Train()`으로 평가 epoch 하나를 돌린다(398~341줄 복원, 370~373줄 평가). 지표 함수는 감싸 반환값만 기록한다. 영상별 예측은 저장하지 않지만 원본 `vote_accuracy_per_video`가 epoch 번호 0에서 영상별 예측을 출력한다(114~115줄). [확실] (코드 읽기)
- 체크포인트가 마지막 epoch 상태가 아니라는 근거는 없어 구현을 진행한다.

## 평가 방식 (고정)
- 같은 방식으로 `Training.py`를 메모리에서 치환한다: `load=False` → `load=True`(1번), `variable_path='./'` → 그 fold의 체크포인트 폴더(1번), `for i in range(5):` → 지정 fold 하나(1번), `np.load` 4곳(414~417)을 `--data-dir`로, `start_indices=np.asarray(...)` 앞에서 우리 start_indices JSON 값으로 덮어쓰기(1번). `ii`는 `load==True`일 때 원본 코드(412~413줄)가 `i+1`로 바꾼다. 원본 `Train()`이 체크포인트를 복원하고 테스트 세트를 한 번 평가한다(`my_model<fold>`).
- 영상별 결정: 원본 `vote_accuracy_per_video`(87~117줄)와 같은 규칙(시퀀스별 예측 라벨 = 예측값을 3.34로 나눈 몫으로 0/5/10, 영상 안 다수결, 동률이면 영상 평균 예측으로 결정; 임계값 3.34와 6.68)을 별도로 구현하되, 원본 함수의 반환값(VA)과 원본이 출력한 영상별 `True label ... detected label` 줄과 일치하는지 검증한다. 일치하지 않으면 중단한다. 회귀 평균값은 영상 안 예측의 평균이다.
- 영상 식별자: 우리 `sequence_metadata_fold<k>_test.csv`의 테스트 영상 순서(첫 등장 순서)와 start_indices 구간 순서를 맞춰 붙인다. 메타데이터의 영상 시작 위치가 start_indices와 같고, 영상별 클래스가 테스트 라벨 구간의 값과 같음을 검증하며 어긋나면 중단한다. [확실] (사전 정의)
- 출력(집계 폴더): `video_predictions_seed<s>.csv`(열: fold, video_id, true_class, pred_class, regression_mean), `confusion_matrix_seed<s>.csv`, `class_accuracy_summary.csv`, 점검용 `fold_check.csv`와 `evaluation_summary.txt`. 이미 있으면 덮어쓰지 않고 중단한다.
- 무결성: 실행 전후로 `third_party`의 보호 목록 해시와 3개 학습 출력 루트의 모든 파일 SHA-256이 같은지 확인한다.

## 예측 (결과를 보기 전에 등록)
- (a) 재평가한 fold별 VA가 `metrics.csv` 마지막 epoch VA와 15개 모두 일치한다(절대 차이 1e-6 미만). [가능성 높음] (체크포인트가 마지막 epoch일 때). (a)가 어긋나면 "체크포인트가 마지막 epoch가 아님"이나 "평가 방식 차이"로 원인을 추정하지 않고 어긋난 fold와 값만 기록한다.
- (b) 클래스 5의 영상 정확도가 세 시드 모두 클래스 0과 10보다 낮다. [가능성 높음]
- (c) 시드 합산 클래스 5 영상 정확도는 0.40 이하이다. [추측]
- (d) 입력을 바꾸지 않으므로 `third_party` 해시와 학습 출력 파일 해시는 실행 전후 같다. [확실]
- 판정 규칙: 결과를 본 뒤 적중 여부를 그대로 쓴다. 기준은 바꾸지 않는다. 앞의 71번째 epoch 참고치(시드 합산 클래스 5 0.327)와의 차이는 사실로만 기록한다. [확실] (사전 정의)


# B1 체크포인트 영상별 평가 결과 (2026-10-04)

사전 등록(위 "B1 체크포인트 영상별 평가 사전 등록 (2026-10-04)")대로 `src/repro/evaluate_baseline_checkpoints.py`로 시드 1, 2, 3의 fold별 체크포인트(`my_model<k>`)를 불러 테스트 영상 전체의 영상별 예측을 만들었다. 체크포인트 로드와 예측만 했고(학습 없음), 학습 출력 폴더의 파일은 읽기만 했다. 표와 CSV는 학습 출력 루트 밖의 집계 폴더에 썼다. 저자가 어떤 영상을 제외했는지는 확인된 바 없다. [확실]

## 정정과 갱신
- 사전 등록 항목의 "(398~341줄 복원, 370~373줄 평가)"에서 앞의 줄 번호는 오타이고 `Training.py`의 복원은 338~341줄이다. [확실]
- 앞의 "B1 기준선 학습 결과 집계 (2026-10-04)" 항목은 마지막 epoch(80)의 클래스별 정확도와 혼동행렬을 산출 불가로 적었고 71번째 epoch 참고치를 실었다. 이 항목의 체크포인트 평가가 마지막 epoch 값을 제공하므로 그 부분을 이 항목의 값으로 갱신한다. 앞 항목은 고치지 않는다. [확실]

## 시험과 검증
- `--dry-run`(TF import 없음)에서 시드 3개 × 5-fold 모두 영상 식별자 검증(메타데이터 영상 시작 위치 == start_indices, 영상 클래스 == 테스트 라벨 구간), 체크포인트 존재, 치환 개수(각 1), 컴파일을 통과했다. 집계 폴더가 학습 출력 루트 안이거나 저장소 안이거나 `SEQ_OUT_DIR`가 없으면 중단했다. 개발 중 `load=False` 치환 뒤 소스를 나누는 함수가 `load=False` 줄을 찾지 못해 dry-run이 한 번 중단되었고, 이 모듈에 `load=True` 기준의 분할 함수를 추가해 고쳤다(평가 방식은 그대로). [확실]
- 영상별 결정은 원본 `vote_accuracy_per_video`와 같은 규칙으로 별도 구현했고, 모든 fold에서 원본 함수의 VA(차이 1e-12 이내)와 원본이 출력한 영상별 `True label ... detected label` 줄(영상 수와 값 모두)과 일치했다(불일치하면 중단하는 검증). [확실]

## 예측 대조
- (a) 적중. 재평가한 fold별 VA가 `metrics.csv` 마지막 epoch VA와 15개 모두 일치했다(절대 차이 최대 4.85e-07로 `metrics.csv`의 소수 6자리 반올림 오차 이내, 일치 15/15). 같은 재평가의 VRE와 BSA acc도 15개 모두 소수 6자리까지 같았다(BSA loss, BSRE도 `fold_check.csv`에 같은 방식으로 기록). 재평가 지표가 마지막 epoch 지표와 같다는 것은 [확실]이고, 체크포인트가 마지막 epoch의 가중치라는 해석은 간접 근거(지표 일치, 저장 호출이 학습 루프 뒤에 한 번 있음)에 의한 [가능성 높음]이다. [확실] (일치)
- (b) 적중. 클래스 5의 영상 정확도가 세 시드 모두 클래스 0과 10보다 낮다: 시드1 0.393(클래스 0 0.719, 10 0.685), 시드2 0.321(0.719, 0.722), 시드3 0.268(0.754, 0.685). [확실]
- (c) 적중. 시드 합산 클래스 5 영상 정확도는 0.327(55/168)로 0.40 이하이다. [확실]
- (d) 적중. 실행 전후 `third_party` 보호 목록(44개 파일) 검사와 3개 학습 출력 루트의 모든 파일 SHA-256 비교에서 변화가 없었다. [확실]

## 마지막 epoch 체크포인트의 영상 단위 결과 (행 = 정답 0/5/10, 열 = 예측 0/5/10, 5개 fold 합산)
- 시드1: 정답 0은 41/14/2(영상 57), 정답 5는 8/22/26(영상 56), 정답 10은 3/14/37(영상 54). 정확도 0 0.719, 5 0.393, 10 0.685.
- 시드2: 정답 0은 41/14/2, 정답 5는 14/18/24, 정답 10은 6/9/39. 정확도 0 0.719, 5 0.321, 10 0.722.
- 시드3: 정답 0은 43/11/3, 정답 5는 14/15/27, 정답 10은 4/13/37. 정확도 0 0.754, 5 0.268, 10 0.685.
- 시드 합산(영상 501개): 정답 0은 125/39/7(영상 171), 정답 5는 36/55/77(영상 168), 정답 10은 13/36/113(영상 162). 클래스별 정확도 0 0.731, 5 0.327, 10 0.698. 정답 5가 10으로 예측된 영상이 77개로 가장 많고, 정답 5가 0으로 예측된 영상은 36개이다. [확실]
- 앞 항목의 71번째 epoch 참고치(시드 합산 0.731, 0.327, 0.698)와 시드 합산 클래스별 정확도는 같은 값이다. 혼동행렬의 칸은 조금 다르다(예: 정답 0이 5로 예측된 영상 40 대 39, 정답 5가 10으로 예측된 영상 78 대 77). 시드별 값은 다르다(예: 시드1 클래스 0은 0.702 대 0.719, 클래스 10은 0.704 대 0.685). 두 시점 사이의 일치가 우연인지는 확인하지 못했다. 미검증
- 영상별 예측은 집계 폴더의 `video_predictions_seed<s>.csv`(fold, video_id, true_class, pred_class, regression_mean)에 있다. 영상 식별자는 우리 테스트 메타데이터 순서를 start_indices 구간에 맞춰 붙였고 순서와 클래스 검증을 통과했다. [확실]

## 확인하지 못한 것 (미검증)
- 어떤 영상이 어떤 클래스로 틀리는지의 원인(예: 영상 길이, 피험자, 회전, 추출 품질과의 관련)은 이번 평가로 알 수 없다. 영상별 예측과 우리 메타데이터를 나란히 놓아 보는 분석은 하지 않았다. 미검증
- 체크포인트가 마지막 epoch의 가중치라는 점은 지표 일치로 뒷받침되지만 파일 안의 epoch 번호는 확인할 수 없다(`Saver`는 epoch를 저장하지 않는다). 미검증
- 시드가 3개뿐이고 영상이 fold당 32~35개라 영상 하나가 fold VA를 약 0.03 움직인다. 클래스별 정확도의 시드 간 차이(클래스 5가 0.268~0.393)는 참고 수준이다. 미검증
