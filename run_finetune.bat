@echo off
REM BART Fine-tuning Script for Windows
REM Quick start script for fine-tuning BART on research papers

setlocal enabledelayedexpansion

echo.
echo ========================================================================
echo BART Model Fine-tuning for Research Paper Summarization
echo ========================================================================
echo.

REM Check if virtual environment is activated
if not defined VIRTUAL_ENV (
    echo Warning: Virtual environment not activated!
    echo Attempting to activate .venv...
    call .venv\Scripts\activate.bat
)

REM Check if venv was activated successfully
if not defined VIRTUAL_ENV (
    echo Error: Could not activate virtual environment
    echo Please activate manually: .venv\Scripts\activate.bat
    pause
    exit /b 1
)

echo Virtual Environment: !VIRTUAL_ENV!
echo.

REM Menu
echo Select fine-tuning option:
echo.
echo 1 - Quick Test (1000 samples, 2 epochs) - ~10 minutes
echo 2 - Medium Training (5000 samples, 3 epochs) - ~30 minutes
echo 3 - Full Training (all data, 3 epochs) - 1-4 hours
echo 4 - Use Fine-tuned Model (Inference)
echo 5 - Exit
echo.

set /p choice="Enter your choice (1-5): "

if "%choice%"=="1" (
    echo.
    echo Running Quick Test...
    echo.
    python finetune_bart_lite.py --samples 1000 --epochs 2 --batch-size 4
    goto end
)

if "%choice%"=="2" (
    echo.
    echo Running Medium Training...
    echo.
    python finetune_bart_lite.py --samples 5000 --epochs 3 --batch-size 8
    goto end
)

if "%choice%"=="3" (
    echo.
    echo Running Full Training...
    echo Warning: This will take 1-4 hours!
    echo.
    set /p confirm="Do you want to continue? (Y/N): "
    if /i "!confirm!"=="Y" (
        python finetune_bart.py
    ) else (
        echo Cancelled.
    )
    goto end
)

if "%choice%"=="4" (
    echo.
    echo Testing Fine-tuned Model...
    echo.
    python finetune_bart_inference.py
    goto end
)

if "%choice%"=="5" (
    echo Exiting...
    goto end
)

echo Invalid choice. Exiting.

:end
echo.
echo Done!
echo.
pause
