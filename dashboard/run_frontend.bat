@echo off
cd /d "%~dp0frontend"
where npm.cmd >nul 2>nul
if errorlevel 1 (
  echo npm.cmd not found. Please install Node.js first.
  pause
  exit /b 1
)

if not exist "node_modules" (
  call npm.cmd install
  if errorlevel 1 (
    echo npm install failed.
    pause
    exit /b 1
  )
)

call npm.cmd run dev
if errorlevel 1 (
  echo frontend dev server failed.
  pause
  exit /b 1
)
