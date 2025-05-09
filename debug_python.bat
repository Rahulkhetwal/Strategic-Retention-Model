@echo off
echo Python Environment Diagnostic
echo ==========================

echo 1. Python Version
python --version 2>&1

echo.
echo 2. Pip Version
pip --version 2>&1

echo.
echo 3. Python Executable Path
where python 2>&1

echo.
echo 4. Pip Executable Path
where pip 2>&1

echo.
echo 5. System PATH
echo %PATH%

echo.
echo 6. Attempting Package Installation
pip install pandas numpy scikit-learn matplotlib seaborn --user 2>&1

echo.
echo 7. Installed Packages
pip list 2>&1

pause
