@echo off
cd /d "%~dp0"
echo Starting FX signal app... (close this window to stop)
start "" cmd /c "timeout /t 4 >nul & start http://localhost:8501"
".venv\Scripts\python.exe" -m streamlit run app.py
pause
