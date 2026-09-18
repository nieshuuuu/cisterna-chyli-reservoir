@echo off
setlocal
title CC/TD painter - sync my work
cd /d "%~dp0"

rem Double-click this before and after a painting session when you are working from a
rem LOCAL copy of this folder. It pulls painter code fixes down from the share and
rem pushes the masks you painted back up. It never deletes anything and never
rem overwrites a mask on the share that is newer than yours.

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Sync_My_Work.ps1" %*

echo.
pause
