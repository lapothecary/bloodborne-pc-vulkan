@echo off
cd /d "%~dp0"
if exist "Bloodborne.exe" (
    start "" "Bloodborne.exe" --play %*
) else if exist "bbport-run.exe" (
    "bbport-run.exe" --play %*
) else (
    python run.py %*
)
REM Release v1.1.1
