# early-drowsiness-repro

Reproduction of the baseline of the early drowsiness detection paper (arXiv:1904.07312) on the
UTA-RLDD dataset, built on the original author's public code. The goal is a transparent,
verifiable reproduction record: the upstream code is vendored unmodified, and everything added for
reproduction lives in separate, readable scripts and logs.

## Status

- **Done:** evaluation of the pretrained checkpoints (`my_model1..5`, official 5-fold split) reproduces in this
  repository's layout. See the regression table below.
- **In progress:** retraining. The Fold1 retraining result is being checked with multiple seeds and is **TBD**;
  no retraining number is reported here.

## Regression results: pretrained checkpoints, 5-fold evaluation (no training)

Produced by `src/repro/reproduction_eval.py` (checkpoints restored and evaluated only). Raw output:
`experiments/00_reproduction/regression_eval/`.

| Fold | BSA loss | BSA acc | VA | BSRE | VRE |
|---|---|---|---|---|---|
| 1 | 15.711 | 0.4838 | 0.6364 | 3.185 | 2.420 |
| 2 | 10.574 | 0.5879 | 0.6111 | 1.391 | 1.040 |
| 3 | 13.315 | 0.5606 | 0.6944 | 2.153 | 0.667 |
| 4 | 11.508 | 0.5445 | 0.6389 | 1.599 | 0.850 |
| 5 | 10.250 | 0.5287 | 0.6667 | 1.176 | 0.812 |
| mean | 12.271 | 0.5411 | 0.6495 | 1.901 | 1.158 |

The abbreviations are the metric names printed by the upstream `Training.py`; see the upstream README
and the paper for their definitions. The step-by-step history, including errors, is in
`docs/REPRODUCTION_LOG.md`.

## Layout

```
third_party/ghoddoosian/   upstream code, checkpoints and features, byte-identical (MIT; see THIRD_PARTY.md)
src/repro/                 environment / dataset / asset checks, evaluation, Fold1 retraining, one-video blink run
experiments/00_reproduction/   archived logs and result tables
configs/                   protected_manifest.sha256 (44 hashes), image_publishable.csv
envs/environment_tf1.yml   conda environment (Python 3.6, TensorFlow 1.8)
data/README.md             how to obtain the data (nothing is stored in the repository)
docs/REPRODUCTION_LOG.md   step-by-step record, including errors
```

## Reproduce

Requires conda. Commands are run from the repository root.

```
conda env create -f envs/environment_tf1.yml
conda activate ghoddoosian_tf1

python src/repro/check_environment.py      # interpreter / package versions
python src/repro/check_dataset.py          # .npy inputs and my_model1..5 checkpoints
python src/repro/reproduction_eval.py 5    # 5-fold evaluation of the pretrained checkpoints
```

`reproduction_eval.py` writes to `experiments/00_reproduction/regression_eval/` and overwrites files
there; move existing results away first if you want to keep them. It first verifies the 44 vendored
files against `configs/protected_manifest.sha256` and stops if any differs.

Optional, needs data that is not in this repository (see `data/README.md`):

```
set DLIB_LANDMARK_PATH=<path to shape_predictor_68_face_landmarks.dat>
set RLDD_DATA_DIR=<folder with the UTA-RLDD videos>
python src/repro/check_video_assets.py
python src/repro/reproduction_blink_one_video.py <video path>
```

`src/repro/reproduction_train_fold1.py` retrains Fold1 with the original settings. It is not needed to
reproduce the table above, and its results are not reported (see Status).

## Data

The UTA-RLDD videos, the dlib landmark model and any frames derived from them are **not** included.
See `data/README.md`.

## Usage guidelines

- Cite the original paper (arXiv:1904.07312) whenever you use this repository, the pretrained
  checkpoints or the upstream code. Also see `THIRD_PARTY.md`.
- Evaluate with the official 5-fold split and report the 5-fold average, not a single fold.
- Face images from UTA-RLDD may be made public only for the 36 subjects listed in
  `configs/image_publishable.csv` (the 'YES' rows of the official dataset page). Save any such image through
  `src/repro/publish_guard.py`, which refuses other subjects.

## Known limitations and open items

- Fold1 retraining: result **TBD**, being verified with multiple seeds. The original code sets no random
  seed, so retraining runs differ from one another. Older Fold1 retraining logs are archived under
  `experiments/00_reproduction/`, but no conclusion is drawn from them yet.
- Only the pretrained checkpoints are evaluated; the video-to-feature pipeline (`blink_video.py`) is checked
  only for asset loading and is not re-run on the dataset here.
- The environment is old (Python 3.6, TensorFlow 1.8) and has been exercised on Windows only.
- Numbers other than the regression table above are intentionally not reported until they are confirmed.

## Citation

<!-- TODO: paste the BibTeX entry given on the official UTA-RLDD page (https://sites.google.com/view/utarldd/home).
     Do not write it from memory. -->

```bibtex
% TODO: paste the official BibTeX here
```

## License

Original code in this repository: MIT, see `LICENSE`. Code under `third_party/`: MIT, Copyright (c) 2019
Reza Ghoddoosian, see `THIRD_PARTY.md`.
