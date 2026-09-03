@echo off
title Hotel Booking Cancellation Predictor
cd /d "%~dp0"

echo ==========================================
echo Hotel Booking Cancellation Predictor
echo ==========================================
echo.

:start_backend
echo Starting FastAPI Backend on http://localhost:8000...
start "Backend" cmd /c "python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --log-level=warning"
timeout /t 3 >nul
if not errorlevel 1 (
    echo Backend process launched.
) else (
    echo Failed to start backend. Press any key to exit...
    pause >nul
    exit /b 1
)

echo.
echo Starting Streamlit Frontend on http://localhost:8501...
start "Frontend" cmd /c "streamlit run app.py --server.port 8501 --server.headless true --browser.gatherUsageStats false"
timeout /t 5 >nul

echo.
echo ==========================================
echo Both services are starting up...
echo Backend:  http://localhost:8000
echo Frontend: http://localhost:8501
echo ==========================================
echo.
echo Press CTRL+C to stop the services, or close this window.
echo.

:wait
timeout /t 30 >nul
goto wait