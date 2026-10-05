@echo off
cd /d "%~dp0"
echo Registering hourly update task "FXSignalTool_Update" (runs at minute 05 every hour while you are logged in)...
schtasks /Create /TN "FXSignalTool_Update" /TR "\"%~dp0.venv\Scripts\pythonw.exe\" \"%~dp0update_job.py\"" /SC HOURLY /ST 00:05 /F
if errorlevel 1 (
  echo Failed to register the task.
) else (
  echo Done. To remove it, run the unregister batch file.
)
pause
