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
