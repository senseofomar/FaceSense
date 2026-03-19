@echo off
color 0a
title FaceSense Bootloader

echo =========================================
echo       INITIALIZING FACESENSE AI
echo =========================================
echo.

REM Ensure we are in the correct directory (D:\PycharmProjects\FaceSense)
cd /d "%~dp0"

REM Check if virtual environment exists
IF NOT EXIST ".venv\Scripts\activate" (
  color 0c
  echo [ERROR] Virtual environment not found at .venv\Scripts\activate
  pause
  exit /b 1
)

echo [1/2] Booting AI Inference Engine (Camera)...
REM Opens a new terminal, activates venv, and runs the camera module
start "FaceSense Camera" cmd /k "call .venv\Scripts\activate && python -m facesense.core.live"

REM Wait 3 seconds to let the camera initialize before launching the dashboard
timeout /t 3 /nobreak >nul

echo [2/2] Booting Analytics Dashboard...
REM Opens a new terminal, activates venv, and runs the Streamlit app
start "FaceSense Dashboard" cmd /k "call .venv\Scripts\activate && streamlit run app\facesense_dashboard.py"

echo.
echo FaceSense components launched successfully!
echo You can minimize or close this green window.
pause