@echo off
chcp 65001 >nul
title ArUco Blank Generator - Setup
echo ========================================
echo  ArUco Blank Generator - Setup
echo ========================================
echo.

where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found.
    echo         Install Python 3.10+ from https://www.python.org/downloads/
    echo         Make sure to check "Add python.exe to PATH".
    pause
    exit /b 1
)

python --version
echo.

echo [1/3] Upgrading pip...
python -m pip install --upgrade pip
if errorlevel 1 (
    echo [ERROR] Failed to upgrade pip.
    pause
    exit /b 1
)

echo.
echo [2/3] Installing dependencies from requirements.txt...
python -m pip install -r "%~dp0requirements.txt"
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b 1
)

echo.
echo [3/3] Verifying imports...
python -c "import cv2, docx; print('cv2', cv2.__version__); print('python-docx OK')"
if errorlevel 1 (
    echo [ERROR] Import check failed.
    pause
    exit /b 1
)

echo.
echo ========================================
echo  Setup completed successfully!
echo ========================================
echo.
echo  Run the generator:
echo    python generate_blank.py --layout template_layout.json --content content_v1.json --output output\blank_v1.docx
echo.
pause