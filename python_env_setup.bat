@echo off
echo Python Environment Setup and Diagnostic
echo ======================================

echo 1. Checking Python Installation
where python
if %errorlevel% neq 0 (
    echo Python is not installed or not in PATH
    goto :end
)

echo 2. Python Version
python --version

echo 3. Pip Version
python -m pip --version

echo 4. Upgrading Pip
python -m pip install --upgrade pip

echo 5. Installing Required Packages
python -m pip install pandas numpy scikit-learn matplotlib seaborn --user

echo 6. Verifying Installations
python -m pip list | findstr /i "pandas numpy scikit-learn matplotlib seaborn"

:end
pause
