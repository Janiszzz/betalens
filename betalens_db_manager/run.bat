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

echo Starting Betalens Database Manager...
echo Project: %PROJECT_ROOT%
echo Python:  %PYTHON%
echo.

"%PYTHON%" -m betalens_db_manager
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
    echo.
    echo Startup failed. If GUI dependencies are missing, run:
    echo "%PYTHON%" -m pip install -e ".[db,gui]"
    echo.
    pause
)

endlocal & exit /b %EXIT_CODE%
