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
이 프로젝트의 모든 변경은 `src/repro/`에 있으며, 그 스크립트는 원본 코드를 텍스트로 읽어 메모리에서만 치환합니다.

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

## dlib 68점 얼굴 랜드마크 모델 (포함하지 않음)

`shape_predictor_68_face_landmarks.dat`(`blink_video.py`가 사용)는 원저자 저장소의 파일이 **아니며** 여기서
배포하지 않습니다(`*.dat`는 git 무시 대상). dlib 사이트에서 별도로 받고 해당 모델의 이용 조건을 확인하세요.
스크립트가 이 파일을 찾게 하는 `DLIB_LANDMARK_PATH` 설정은 `data/README.md`를 참고하세요.

## UTA-RLDD 영상 (포함하지 않음)

UTA-RLDD 데이터셋과 그로부터 만든 프레임은 이 저장소에 포함되지 않습니다. `data/README.md`를 참고하세요.
