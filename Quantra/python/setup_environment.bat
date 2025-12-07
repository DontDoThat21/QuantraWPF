@echo off
REM Install Quantra Python Dependencies
REM This script installs all required Python packages for Quantra

echo ===============================================================================
echo Quantra Python Environment Setup
echo ===============================================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or later from https://www.python.org/
    pause
    exit /b 1
)

echo Python version:
python --version
echo.

REM Check if pip is available
pip --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: pip is not installed or not in PATH
    echo Please install pip or reinstall Python
    pause
    exit /b 1
)

echo.
echo Installing core dependencies...
echo.

REM Upgrade pip first
python -m pip install --upgrade pip

REM Install requirements
pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo WARNING: Some packages failed to install
    echo This may be normal for optional dependencies
    echo.
)

echo.
echo ===============================================================================
echo Checking installation...
echo ===============================================================================
echo.

REM Run dependency checker
python check_dependencies.py

echo.
echo ===============================================================================
echo Setup complete!
echo ===============================================================================
echo.

pause
