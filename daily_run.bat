@echo off
chcp 65001
echo ===================================================
echo 🚀 Smart Stock Selector - Daily Auto Update
echo ===================================================
echo.

echo [1/3] Syncing Stock Data...
python backend/main.py --sync
if %errorlevel% neq 0 goto error

echo.
echo [2/3] Retraining Sniper AI Ensemble...
python backend/train_ai.py
if %errorlevel% neq 0 goto error

echo.
echo [3/3] Recalculating Scores & Ranks...
python backend/recalculate.py
if %errorlevel% neq 0 goto error

echo.
echo ✅ Daily Update Complete! Best picks are ready.
pause
exit /b 0

:error
echo.
echo ❌ An error occurred! Please check the logs.
pause
exit /b 1
