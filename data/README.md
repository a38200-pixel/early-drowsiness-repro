# Data (not included in this repository)

No videos, frames or landmark models are stored here. Keep them **outside** the repository and point
the scripts at them with environment variables.

## UTA-RLDD videos

1. Open the official dataset page: https://sites.google.com/view/utarldd/home
2. Follow the download link(s) listed on that page and read the dataset's terms of use.
3. Store the videos in a folder outside this repository, and set `RLDD_DATA_DIR` to it:

   ```
   # Windows (cmd)
   set RLDD_DATA_DIR=D:\datasets\UTA-RLDD
   # Linux / macOS
   export RLDD_DATA_DIR=/data/UTA-RLDD
   ```

`src/repro/reproduction_blink_one_video.py` resolves a relative video path under `RLDD_DATA_DIR`
(an absolute path also works and does not need the variable).

Do not commit videos or frames extracted from them. `.gitignore` blocks common video/image extensions,
and only `docs/figures/` may contain images. Face images may be published only for the subjects listed in
`configs/image_publishable.csv`; save them through `src/repro/publish_guard.py`.

## dlib 68-point face landmark model

`shape_predictor_68_face_landmarks.dat` is not part of the original author's repository, so it is
not vendored here.

1. Download it separately from the dlib website (https://dlib.net/files/; the file is distributed
   compressed as `.dat.bz2`) and decompress it. Check its terms of use.
2. Store it outside this repository and set `DLIB_LANDMARK_PATH` to the `.dat` file:

   ```
   # Windows (cmd)
   set DLIB_LANDMARK_PATH=D:\models\shape_predictor_68_face_landmarks.dat
   # Linux / macOS
   export DLIB_LANDMARK_PATH=/models/shape_predictor_68_face_landmarks.dat
   ```

Scripts that need it (`check_video_assets.py`, `reproduction_blink_one_video.py`) stop with a message naming
the variable if it is unset or the file does not exist.

## What is already in the repository

The pre-extracted blink features (`Blinks*_30_Fold*.npy`, `Labels*_30_Fold*.npy`), the pretrained
`my_model1..5` checkpoints and the blink SVM come from the upstream repository and live in
`third_party/ghoddoosian/` (see `THIRD_PARTY.md`). Evaluating the pretrained checkpoints needs
neither the videos nor the dlib model.
