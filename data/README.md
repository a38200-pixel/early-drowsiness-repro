# 데이터 (이 저장소에는 포함되지 않음)

영상, 프레임, 랜드마크 모델은 여기에 저장하지 않습니다. 저장소 **밖**에 보관하고 환경변수로 스크립트에 알려 주세요.

## UTA-RLDD 영상

1. 공식 데이터셋 페이지를 엽니다: https://sites.google.com/view/utarldd/home
2. 페이지에 안내된 다운로드 링크를 따라 받고, 데이터셋 이용 약관을 읽습니다.
3. 영상을 이 저장소 밖의 폴더에 보관하고, `RLDD_DATA_DIR`를 그 폴더로 지정합니다.

   ```
   # Windows (cmd)
   set RLDD_DATA_DIR=<DATA_DIR>
   # Linux / macOS
   export RLDD_DATA_DIR=<DATA_DIR>
   ```

`src/repro/reproduction_blink_one_video.py`는 상대경로 영상을 `RLDD_DATA_DIR` 아래에서 찾습니다
(절대경로를 주면 이 변수는 필요 없습니다).

영상이나 그로부터 뽑은 프레임을 커밋하지 마세요. `.gitignore`가 흔한 영상/이미지 확장자를 막으며, 이미지는
`docs/figures/`에만 둘 수 있습니다. 얼굴 이미지는 `configs/image_publishable.csv`에 있는 대상만 공개할 수 있고,
`src/repro/publish_guard.py`를 통해 저장해야 합니다.

## dlib 68점 얼굴 랜드마크 모델

`shape_predictor_68_face_landmarks.dat`는 원저자 저장소의 파일이 아니므로 vendoring하지 않았습니다.

1. dlib 사이트(https://dlib.net/files/)에서 별도로 내려받아(`.dat.bz2`로 압축 배포) 압축을 풉니다.
   이용 조건을 확인하세요.
2. 저장소 밖에 보관하고 `DLIB_LANDMARK_PATH`를 `.dat` 파일로 지정합니다.

   ```
   # Windows (cmd)
   set DLIB_LANDMARK_PATH=<PREDICTOR>
   # Linux / macOS
   export DLIB_LANDMARK_PATH=<PREDICTOR>
   ```

이 파일이 필요한 스크립트(`check_video_assets.py`, `reproduction_blink_one_video.py`)는 변수가 없거나 파일이 없으면
변수 이름을 알려 주는 메시지와 함께 중단됩니다.

## 저장소에 이미 있는 것

미리 추출된 blink 특징(`Blinks*_30_Fold*.npy`, `Labels*_30_Fold*.npy`), 사전학습 `my_model1..5` 체크포인트,
blink SVM은 원본 저장소의 것이며 `third_party/ghoddoosian/`에 있습니다(`THIRD_PARTY.md` 참고).
사전학습 체크포인트 평가에는 영상도 dlib 모델도 필요하지 않습니다.
