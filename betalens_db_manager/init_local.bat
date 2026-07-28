@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "PROJECT_ROOT=%%~fI"
set "VENV_PYTHON=%PROJECT_ROOT%\.venv\Scripts\python.exe"

cd /d "%PROJECT_ROOT%"
if exist "%VENV_PYTHON%" (
    set "PYTHON=%VENV_PYTHON%"
) else (
    set "PYTHON=python"
)

echo Betalens Database Manager: schema bootstrap, optional manifest import, final verify
echo Project: %PROJECT_ROOT%
echo Python:  %PYTHON%
echo.

"%PYTHON%" -m betalens_db_manager init --yes %*
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
    echo.
    echo Database initialization/import failed. Review logs\database-manager\run_*.json and jobs.sqlite3.
    echo Required database dependency: "%PYTHON%" -m pip install -e ".[db]"
)

endlocal & exit /b %EXIT_CODE%
