@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run the setup batch file in this folder first.
  if "%~1"=="" pause
  exit /b 1
)
set "PYTHONPATH=%~dp0src"
".venv\Scripts\python.exe" -m offtheball gui %*
set "EXIT_CODE=%ERRORLEVEL%"
if not "%~1"=="" exit /b %EXIT_CODE%
if not "%EXIT_CODE%"=="0" pause
exit /b %EXIT_CODE%
