@echo off
echo Comprehensive System and Python Check
echo =====================================

echo 1. System Information
systeminfo | findstr /C:"OS Name" /C:"OS Version"

echo.
echo 2. Environment Variables
echo PATH=%PATH%

echo.
echo 3. Python Check
where python
if %errorlevel% neq 0 (
    echo Python is NOT in PATH
    goto :end
)

echo.
echo 4. Python Version
python --version

echo.
echo 5. Pip Version
python -m pip --version

echo.
echo 6. Python Executable Location
where python

echo.
echo 7. Pip Executable Location
where pip

:end
pause
