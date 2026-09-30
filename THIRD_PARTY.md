# Third-party material

## Early-Drowsiness-Detection (vendored, unmodified)

| | |
|---|---|
| Location in this repository | `third_party/ghoddoosian/` (44 files) |
| Source | https://github.com/rezaghoddoosian/Early-Drowsiness-Detection.git |
| Commit | `2bafc985c46526fa0dfe3a93bb96fcee1d8d7ac4` |
| License | MIT (full text in `third_party/ghoddoosian/LICENSE`) |

```
MIT License

Copyright (c) 2019 Reza Ghoddoosian
```

The complete MIT permission notice and warranty disclaimer are in `third_party/ghoddoosian/LICENSE`
and apply to everything under `third_party/`. Please cite the original paper when you use this
code or the pretrained checkpoints (see `README.md`).

### The files are unmodified

Every file under `third_party/ghoddoosian/` is a byte-for-byte copy of the git blob at the commit
above (`.gitattributes` sets `third_party/** -text` so git never rewrites line endings). Do not edit
these files; all changes for this project live in `src/repro/`, which reads the upstream code as text
and applies substitutions in memory.

Verify it in either of two ways.

1. Against the manifest (`configs/protected_manifest.sha256`, 44 SHA-256 entries), from the repository root:

   ```
   sha256sum -c configs/protected_manifest.sha256
   ```

2. Against the upstream repository itself: all 44 git blob hashes must match.

   ```
   git clone https://github.com/rezaghoddoosian/Early-Drowsiness-Detection.git upstream
   git -C upstream checkout 2bafc985c46526fa0dfe3a93bb96fcee1d8d7ac4
   git -C upstream ls-files | while IFS= read -r f; do
     a=$(git -C upstream rev-parse "HEAD:$f")
     b=$(git hash-object --no-filters "third_party/ghoddoosian/$f")
     [ "$a" = "$b" ] || echo "MISMATCH: $f"
   done
   ```

   No `MISMATCH` line should be printed, and `git -C upstream ls-files | wc -l` should print `44`.

The evaluation script `src/repro/reproduction_eval.py` also refuses to run if any of the 44 files
differs from the manifest.

## dlib 68-point face landmark model (NOT included)

`shape_predictor_68_face_landmarks.dat` (used by `blink_video.py`) is **not** a file of the original
author's repository and is **not** distributed here (`*.dat` is git-ignored). Obtain it separately from
the dlib website and check its own terms before use; see `data/README.md` for how to point the scripts
at it with `DLIB_LANDMARK_PATH`.

## UTA-RLDD videos (NOT included)

The UTA-RLDD dataset and any frames derived from it are not part of this repository. See `data/README.md`.
