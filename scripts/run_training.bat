@echo off
REM Production Training Script for Cardiac Dreamer (Windows)
REM Channel-token version with cross-patient validation

echo 🔥 Starting Cardiac Dreamer Training...
echo ================================================

REM Configuration
set DATA_DIR=data\processed
set OUTPUT_DIR=outputs
set CONFIG_FILE=configs\production.yaml

REM Check if data directory exists
if not exist "%DATA_DIR%" (
    echo ❌ Error: Data directory %DATA_DIR% not found!
    echo Please ensure your processed data is available.
    pause
    exit /b 1
)

REM Create output directory if it doesn't exist
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

echo 📊 Configuration:
echo   Data Directory: %DATA_DIR%
echo   Output Directory: %OUTPUT_DIR%
echo   Config File: %CONFIG_FILE%
echo.

REM Check if config file exists
if exist "%CONFIG_FILE%" (
    echo 🚀 Launching training with custom config...
    python src\train.py --data_dir "%DATA_DIR%" --output_dir "%OUTPUT_DIR%" --config "%CONFIG_FILE%"
) else (
    echo 🚀 Launching training with default config...
    python src\train.py --data_dir "%DATA_DIR%" --output_dir "%OUTPUT_DIR%"
)

REM Check if training was successful
if %errorlevel% equ 0 (
    echo.
    echo 🎉 Training completed successfully!
    echo.
    echo 📈 To view training logs:
    echo   tensorboard --logdir %OUTPUT_DIR%\run_*\logs
    echo.
    echo 📁 Results saved in: %OUTPUT_DIR%
    echo   - Checkpoints: %OUTPUT_DIR%\run_*\checkpoints\
    echo   - Plots: %OUTPUT_DIR%\run_*\plots\
    echo   - Logs: %OUTPUT_DIR%\run_*\logs\
    echo.
    echo Press any key to continue...
    pause >nul
) else (
    echo.
    echo ❌ Training failed! Check the error messages above.
    echo.
    pause
    exit /b 1
) 