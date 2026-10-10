# 서드파티 자료

## Early-Drowsiness-Detection (vendoring, 무수정)

| | |
|---|---|
| 저장소 내 위치 | `third_party/ghoddoosian/` (파일 44개) |
| 출처 | https://github.com/rezaghoddoosian/Early-Drowsiness-Detection.git |
| 커밋 | `2bafc985c46526fa0dfe3a93bb96fcee1d8d7ac4` |
| 라이선스 | MIT (전문은 `third_party/ghoddoosian/LICENSE`) |

```
MIT License

Copyright (c) 2019 Reza Ghoddoosian
```

MIT 허가 문구와 보증 부인 전문은 `third_party/ghoddoosian/LICENSE`에 있으며 `third_party/` 아래 전체에 적용됩니다.
이 코드나 사전학습 체크포인트를 사용할 때는 원논문을 인용하세요(`README.md` 참고).

### 파일은 수정되지 않았습니다

`third_party/ghoddoosian/` 아래 모든 파일은 위 커밋의 git blob을 바이트 단위로 그대로 복사한 것입니다
(`.gitattributes`가 `third_party/** -text`로 지정돼 git이 줄바꿈을 바꾸지 않습니다). 이 파일들은 수정하지 마세요.
`third_party/` 아래 파일은 수정하지 않았고, 이 프로젝트의 추가와 변경은 이 폴더 밖에 있습니다. `src/repro/`의 스크립트와 `src/improve/hmlstm/`의 모듈은 원본 코드를 텍스트로 읽어 메모리에서만 치환합니다. 저자 코드를 수정해 담은 사본은 `src/improve/rules/blink_video_dump.py` 한 파일입니다(저자 코드와 줄 단위로 비교해 같은 줄이 많이 겹친 추적 파일은 이 파일뿐이었습니다). 이 파일은 파일 머리에 저자의 MIT 고지를 포함하며 아래 "수정 사본(파생 파일)" 절에 설명했습니다.

다음 두 가지 방법 중 하나로 검증합니다.

1. 매니페스트(`configs/protected_manifest.sha256`, SHA-256 44개)와 대조. 저장소 루트에서:

   ```
   sha256sum -c configs/protected_manifest.sha256
   ```

2. 원본 저장소와 직접 대조. git blob 해시 44개가 모두 일치해야 합니다.

   ```
   git clone https://github.com/rezaghoddoosian/Early-Drowsiness-Detection.git upstream
   git -C upstream checkout 2bafc985c46526fa0dfe3a93bb96fcee1d8d7ac4
   git -C upstream ls-files | while IFS= read -r f; do
     a=$(git -C upstream rev-parse "HEAD:$f")
     b=$(git hash-object --no-filters "third_party/ghoddoosian/$f")
     [ "$a" = "$b" ] || echo "MISMATCH: $f"
   done
   ```

   `MISMATCH` 줄이 하나도 출력되지 않아야 하고, `git -C upstream ls-files | wc -l`은 `44`여야 합니다.

평가 스크립트 `src/repro/reproduction_eval.py`도 44개 파일 중 하나라도 매니페스트와 다르면 실행을 거부합니다.

## 수정 사본(파생 파일)

| | |
|---|---|
| 파일 | `src/improve/rules/blink_video_dump.py` |
| 원본 | `third_party/ghoddoosian/blink_video.py` |
| 원본과 같은 코드 줄 | 207/249줄(약 83%), 측정 시점 기준. 저자 `.py` 6개와 비교한 값이며 주석과 25자 미만의 줄은 뺐습니다 |
| 고지 위치 | 파일 머리의 주석(저자 MIT 전문과 변경 요약) |
| 변경 내용 | GUI 제거, 명령줄 인자 추가, 프레임별 EAR 덤프와 프레임 회전 추가. `Blink` 클래스, `Blink_Tracker`, SVM 판별, blink 기록 코드는 바꾸지 않았습니다 |

이 파일은 저자 코드의 수정본이므로 루트 `LICENSE`와 별도로 저자의 MIT 고지(Copyright (c) 2019 Reza Ghoddoosian)를 파일 머리에 포함했습니다. 원본은 `third_party/ghoddoosian/`에 바이트 그대로 있습니다.

## 데이터와 모델 파일의 라이선스 적용 범위

저자 `LICENSE`의 문구는 소프트웨어와 관련 문서 파일을 가리킵니다. 같은 저장소의 `.npy` 특징 파일, `.sav` 파일(SVM), 체크포인트 파일(`my_model1~5`)에 MIT가 적용되는지는 이 문구만으로 확정되지 않습니다. 이 파일들은 저자 공개 저장소에서 바이트 그대로 가져왔고 `configs/protected_manifest.sha256`으로 변조 여부를 확인합니다.

## 루트 LICENSE

루트 `LICENSE`는 저장소 소유자 계정명의 저작권 줄(2026)을 두고, 이 저장소를 위해 쓴 코드와 문서에만 적용된다고 밝힙니다. 저자의 MIT 고지는 `third_party/ghoddoosian/LICENSE`와 위 파생 파일의 머리에 따로 있습니다.

## dlib 68점 얼굴 랜드마크 모델 (포함하지 않음)

`shape_predictor_68_face_landmarks.dat`(`blink_video.py`가 사용)는 원저자 저장소의 파일이 **아니며** 여기서
배포하지 않습니다(`*.dat`는 git 무시 대상). dlib 사이트에서 별도로 받고 해당 모델의 이용 조건을 확인하세요.
스크립트가 이 파일을 찾게 하는 `DLIB_LANDMARK_PATH` 설정은 `data/README.md`를 참고하세요.

## UTA-RLDD 영상 (포함하지 않음)

UTA-RLDD 데이터셋과 그로부터 만든 프레임은 이 저장소에 포함되지 않습니다. `data/README.md`를 참고하세요.
