@echo off
setlocal
title CC/TD painter
cd /d "%~dp0"

rem  Kept only for muscle memory -- Paint.bat is the launcher now.
rem
rem  The old version of this file opened the painter with no sync of any kind. On a local
rem  copy that meant masks were saved to one machine and never reached anybody, which is
rem  the most likely reason 8_31_22_Acq2 was painted-looking and arrived empty. Paint.bat
rem  syncs before and after the session so that cannot happen.

echo.
echo   Start_CC_Painter.bat is now a shortcut to Paint.bat, which also
echo   syncs your work to the share before and after the session.
echo.
call "%~dp0Paint.bat" %*
