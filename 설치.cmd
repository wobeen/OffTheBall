@echo off
chcp 65001 >nul
cd /d "%~dp0"
set "UV_CACHE_DIR=%~dp0work\uv-cache"
set "PYTHONPATH=%~dp0src"
where uv >nul 2>nul
if errorlevel 1 (
  echo uv is required. See README.md for installation instructions.
  pause
  exit /b 1
)
if not exist ".venv\Scripts\python.exe" (
  uv venv --python 3.13 .venv
  if errorlevel 1 goto failed
)
uv pip sync --python ".venv\Scripts\python.exe" requirements.lock.txt
if errorlevel 1 goto failed
".venv\Scripts\python.exe" scripts\prepare_model.py
if errorlevel 1 goto failed
echo Installation complete. Run the launcher.
pause
exit /b 0
:failed
echo Installation failed. Review the error output above.
pause
exit /b 1
