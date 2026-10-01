@echo off
rem run_folds_seed.bat
rem 역할: 지정한 시드로 Fold2~5를 80 epoch씩 차례로 재학습한다. 이미 있는 fold 폴더는 건너뛰고, 실패하면 즉시 중단한다.
rem 사용법: Anaconda Prompt에서 conda activate ghoddoosian_tf1 후 저장소 루트에서 scripts\run_folds_seed.bat <시드>
rem         결과: experiments\02_fold_seeds\fold<F>_seed<시드>, 요약은 python src\repro\summarize_folds.py <시드>
chcp 65001 >nul
setlocal EnableDelayedExpansion
rem 이 파일 위치(scripts\)의 상위 폴더 = 저장소 루트로 이동
cd /d "%~dp0.."

if "%~1"=="" (
    echo 사용법: scripts\run_folds_seed.bat ^<시드^>
    exit /b 1
)
set SEED=%~1

rem conda 환경 확인: Python 3.6이 아니면 안내 후 종료
python -c "import sys; sys.exit(0 if sys.version_info[:2] == (3, 6) else 1)" >nul 2>&1
if errorlevel 1 (
    echo [오류] 현재 python이 3.6이 아닙니다. 먼저 conda activate ghoddoosian_tf1 을 실행하세요.
    exit /b 1
)

echo 시드 %SEED%: Fold2~5 순서대로 실행합니다.
echo 예상 시간: fold당 약 21분, 총 약 1.4시간 [fold 4개]
echo 필요 공간: 약 10MB [fold당 체크포인트 약 2.4MB]
echo 중간에 멈춘 fold는 폴더가 남아 다음 실행에서 건너뛰므로, 다시 돌리려면 그 폴더를 직접 옮기거나 지우세요.
echo 시작 시각: %date% %time%

for %%F in (2 3 4 5) do (
    if exist "experiments\02_fold_seeds\fold%%F_seed%SEED%\" (
        echo [건너뜀] experiments\02_fold_seeds\fold%%F_seed%SEED% 폴더가 이미 있습니다.
    ) else (
        echo [실행] fold %%F 시작: !date! !time!
        python src\repro\train_fold_seeds.py %%F %SEED%
        if errorlevel 1 (
            set FAILED_FOLD=%%F
            goto :failed
        )
        echo [완료] fold %%F 종료: !date! !time!
    )
)

echo 종료 시각: %date% %time%
echo 다음 단계: python src\repro\summarize_folds.py %SEED%
endlocal
exit /b 0

rem 실행 실패: 반복을 멈추고 종료 코드 1로 끝낸다(for 블록 밖에서 exit 해야 종료 코드가 확실히 전달된다)
:failed
echo [오류] fold %FAILED_FOLD% 실행이 실패했습니다. 중단합니다. 종료 시각: %date% %time%
endlocal
exit /b 1
