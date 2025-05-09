@echo off
echo Checking and removing existing Conda environment
for /f "delims=" %%i in ('conda info --envs ^| findstr "churn_prediction"') do (
    echo Removing existing environment
    conda env remove -n churn_prediction
)

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
