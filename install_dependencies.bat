@echo off
echo Installing dependencies for Churn Prediction project
call conda activate churn_prediction
call conda install -c conda-forge pandas numpy scikit-learn matplotlib seaborn -y

echo Verifying installations
call python -c "import pandas; import numpy; import sklearn; print('Dependencies installed successfully!')"

echo Running Churn Analysis Script
call python employee_churn_analysis.py

pause
echo Installing Python Dependencies
echo ==============================

python -m pip install pandas numpy scikit-learn matplotlib seaborn --user

echo.
echo Verifying Installations
python -c "import pandas; import numpy; import sklearn; import matplotlib; import seaborn; print('All dependencies installed successfully!')"

pause
