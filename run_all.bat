@echo off
REM Robust run_all.bat for Windows — put this in project root

REM Move to the folder where this script lives (project root)
pushd "%~dp0" || (
  echo Failed to change directory to "%~dp0"
  pause
  exit /b 1
)

REM Check venv activation script exists
IF NOT EXIST ".venv\Scripts\activate" (
  echo Virtualenv activation not found at .venv\Scripts\activate
  echo Please create or point run_all.bat to your venv.
  popd
  pause
  exit /b 1
)

echo Project root: %CD%
echo Starting FaceSense Live and Streamlit Dashboard...

REM Start FaceSense Live in a new window (no extra cd inside start)
start "FaceSense Live" cmd /k ".venv\Scripts\activate && echo Running FaceSense Live... && python -u src\facesense_live.py"

REM short delay so the camera process starts first
timeout /t 1 /nobreak >nul

REM Start Streamlit Dashboard in a new window
start "FaceSense Dashboard" cmd /k ".venv\Scripts\activate && echo Running Streamlit Dashboard... && streamlit run app\facesense_dashboard.py"

popd
exit /b 0
