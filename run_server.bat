@echo off
REM ── Start the FastAPI backend from the project root ──
REM This ensures "backend" is a proper top-level package for Python imports.

cd /d "%~dp0"

if exist ".venv\Scripts\activate.bat" (
    echo Activating virtual environment (.venv)...
    call .venv\Scripts\activate.bat
) else if exist ".venv_new\Scripts\activate.bat" (
    echo Activating virtual environment (.venv_new)...
    call .venv_new\Scripts\activate.bat
) else (
    echo WARNING: No virtual environment found, using system Python.
)

echo Starting FastAPI server on port 8080...
python -m uvicorn backend.api:app --port 8080 --reload
