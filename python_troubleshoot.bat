@echo off
echo Python Environment Troubleshooting
echo ===================================

echo 1. Current Directory
cd

echo 2. Directory Contents
dir

echo 3. Python Version
python --version

echo 4. Pip Version
pip --version

echo 5. Python Executable Path
where python

echo 6. Attempting to run Python script with full path
python "%~dp0simple_churn_analysis.py"

pause
