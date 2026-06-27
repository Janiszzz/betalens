@echo off
cd /d "%~dp0\.."

powershell -NoProfile -ExecutionPolicy Bypass -Command "try { $r = Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8000/api/health -TimeoutSec 2; if ($r.StatusCode -eq 200) { exit 0 } } catch { exit 1 }; exit 1"
if not errorlevel 1 (
  echo Backend already running at http://127.0.0.1:8000
  echo Reusing existing service.
  pause
  exit /b 0
)

if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -m uvicorn dashboard.backend.main:app --host 127.0.0.1 --port 8000 --reload
) else (
  python -m uvicorn dashboard.backend.main:app --host 127.0.0.1 --port 8000 --reload
)
