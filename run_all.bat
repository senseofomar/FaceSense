@echo off
REM run_all.bat — Start FaceSense live and Streamlit dashboard in two windows
REM Must be located in project root (the folder that contains .venv, src, app, utils)

REM Save the script directory (project root)
SET ROOT=%~dp0
REM Remove trailing backslash for nicer printing
IF "%ROOT:~-1%"=="\" SET ROOT=%ROOT:~0,-1%

echo Project root: %ROOT%
cd /d "%ROOT%"

REM Check venv exist
IF NOT EXIST ".venv\Scripts\activate" (
  echo Virtualenv not found at .venv\Scripts\activate
  echo Please create or point run_all.bat to your venv.
  pause
  exit /b 1
)

REM -- Start FaceSense Live in a new window --
start "FaceSense Live" cmd /k "cd /d \"%ROOT%\" && .venv\Scripts\activate && echo Running FaceSense Live... && python -u src\facesense_live.py"

REM short pause so live starts first (optional)
timeout /t 1 /nobreak >nul

REM -- Start Streamlit Dashboard in a new window --
start "FaceSense Dashboard" cmd /k "cd /d \"%ROOT%\" && .venv\Scripts\activate && echo Running Streamlit Dashboard... && streamlit run app\facesense_dashboard.py"

echo Launched FaceSense Live and Streamlit. Check the two windows.
exit /b 0
