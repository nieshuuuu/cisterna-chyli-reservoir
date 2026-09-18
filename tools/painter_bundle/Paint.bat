@echo off
setlocal
title CC/TD painter
cd /d "%~dp0"

rem  THE one thing to double-click.
rem
rem  Syncs your masks up, pulls down the current painter and everyone else's work,
rem  opens the painter on a fast local copy, and pushes your masks back up when you
rem  press Ctrl-C. You do not have to remember any of those steps.
rem
rem  First run on a machine copies the CT volumes locally (~16 GB, once) and asks first.

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Paint.ps1" %*

echo.
pause
