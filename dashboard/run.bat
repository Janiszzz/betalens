@echo off
setlocal

set "DASHBOARD_DIR=%~dp0"

echo Starting betalens dashboard...
echo Backend:  http://127.0.0.1:8000
echo Frontend: http://127.0.0.1:5173
echo.

start "betalens backend" cmd /k call "%DASHBOARD_DIR%run_backend.bat"
start "betalens frontend" cmd /k call "%DASHBOARD_DIR%run_frontend.bat"

echo Started backend and frontend in separate windows.
echo Open http://127.0.0.1:5173 in your browser.
echo Close both service windows to stop the dashboard.
pause
endlocal
