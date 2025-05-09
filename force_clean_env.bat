@echo off
echo Removing existing Conda environment
rmdir /s /q "D:\anaconda\envs\churn_prediction"

echo Creating new Conda environment
conda create -n churn_prediction python=3.12 -y

echo Activating environment and installing dependencies
call conda activate churn_prediction
call conda install -c conda-forge pandas numpy scikit-learn matplotlib seaborn -y

echo Verifying installations
call conda run -n churn_prediction python -c "import pandas; import numpy; import sklearn; print('Dependencies installed successfully!')"

echo Running Churn Analysis Script
call conda run -n churn_prediction python employee_churn_analysis.py

pause
