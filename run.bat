@echo off
chcp 65001 >nul
title ArUco Blank Generator - Run
echo ========================================
echo  ArUco Blank Generator
echo ========================================
echo.

if not exist "output" mkdir "output"

echo [1/2] Generating blank_v1.docx ...
python generate_blank.py --layout template_layout.json --content content_v1.json --output output\blank_v1.docx
if errorlevel 1 (
    echo [ERROR] Failed to generate blank_v1.docx
    pause
    exit /b 1
)

echo.
echo [2/2] Generating blank_v2.docx ...
python generate_blank.py --layout template_layout.json --content content_v2.json --output output\blank_v2.docx
if errorlevel 1 (
    echo [ERROR] Failed to generate blank_v2.docx
    pause
    exit /b 1
)

echo.
echo ========================================
echo  Done! Files saved to output\
echo ========================================
echo.
pause