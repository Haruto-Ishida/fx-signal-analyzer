@echo off
echo Removing scheduled task "FXSignalTool_Update"...
schtasks /Delete /TN "FXSignalTool_Update" /F
pause
