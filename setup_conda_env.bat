@echo off
echo Creating Conda Environment for Churn Prediction Project
call conda create -n churn_prediction python=3.12 pandas numpy scikit-learn matplotlib seaborn -y

echo Activating Environment
call conda activate churn_prediction

echo Verifying Installations
python -c "import pandas; import numpy; import sklearn; print('Dependencies installed successfully!')"

pause
