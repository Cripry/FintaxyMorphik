@echo off
REM ============================================================
REM  Morphik Stack Auto-Start Script
REM  Starts: PostgreSQL -> Redis -> Ollama -> Morphik
REM  PostgreSQL and Redis are Windows services (auto-start).
REM  This script ensures Ollama is running, then launches Morphik.
REM ============================================================

setlocal

set MORPHIK_DIR=D:\Fintaxy Codespace\FintaxyMorphik2
set OLLAMA_EXE=C:\Users\Administrator\AppData\Local\Programs\Ollama\ollama app.exe
set PYTHON_EXE=%MORPHIK_DIR%\.venv\Scripts\python.exe
set LOG_FILE=%MORPHIK_DIR%\logs\startup.log

REM -- CUDA / PyTorch fix for Windows WDDM driver
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REM -- Create logs directory if needed
if not exist "%MORPHIK_DIR%\logs" mkdir "%MORPHIK_DIR%\logs"

echo [%date% %time%] === Morphik Stack Starting === >> "%LOG_FILE%"

REM ---- 1. Wait for PostgreSQL service ----
echo [%date% %time%] Waiting for PostgreSQL... >> "%LOG_FILE%"
:wait_pg
sc query postgresql-x64-16 | findstr /i "RUNNING" >nul 2>&1
if errorlevel 1 (
    timeout /t 5 /nobreak >nul
    goto wait_pg
)
echo [%date% %time%] PostgreSQL is running. >> "%LOG_FILE%"

REM ---- 2. Wait for Redis (Memurai) service ----
echo [%date% %time%] Waiting for Redis/Memurai... >> "%LOG_FILE%"
:wait_redis
sc query Memurai | findstr /i "RUNNING" >nul 2>&1
if errorlevel 1 (
    timeout /t 5 /nobreak >nul
    goto wait_redis
)
echo [%date% %time%] Redis/Memurai is running. >> "%LOG_FILE%"

REM ---- 3. Start Ollama if not running ----
echo [%date% %time%] Checking Ollama... >> "%LOG_FILE%"
tasklist /FI "IMAGENAME eq ollama app.exe" 2>nul | findstr /i "ollama" >nul 2>&1
if errorlevel 1 (
    echo [%date% %time%] Starting Ollama... >> "%LOG_FILE%"
    start "" "%OLLAMA_EXE%"
)

REM Wait for Ollama API to respond
:wait_ollama
curl -s http://localhost:11434/api/version >nul 2>&1
if errorlevel 1 (
    timeout /t 5 /nobreak >nul
    goto wait_ollama
)
echo [%date% %time%] Ollama is responding. >> "%LOG_FILE%"

REM ---- 4. Start Morphik ----
echo [%date% %time%] Starting Morphik server... >> "%LOG_FILE%"
cd /d "%MORPHIK_DIR%"
start "" "%PYTHON_EXE%" start_server.py --skip-redis-check --skip-ui

echo [%date% %time%] Morphik start command issued. >> "%LOG_FILE%"
echo [%date% %time%] === Morphik Stack Startup Complete === >> "%LOG_FILE%"

endlocal
