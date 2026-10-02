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
