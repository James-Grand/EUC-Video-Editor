@echo off
rem Double-click this to process everything in inbox\.
rem Or from a terminal:  run_ride.bat 2026-11-05_ride1 --no-render
cd /d "%~dp0"
".venv\Scripts\python.exe" tools\run_ride.py %*
echo.
pause
