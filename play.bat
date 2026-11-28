@echo off
cd /d "%~dp0"
if exist "Bloodborne.exe" (
    start "" "Bloodborne.exe" %*
) else (
    python run.py %*
)
