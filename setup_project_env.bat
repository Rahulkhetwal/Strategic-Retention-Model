@echo off
echo Creating Conda Environment for Churn Prediction Project
call conda create -n churn_prediction python=3.12 pandas numpy scikit-learn matplotlib seaborn -y

echo Activating Environment
call conda activate churn_prediction

echo Verifying Installations
python -c "import pandas; import numpy; import sklearn; print('Dependencies installed successfully!')"

echo Navigating to Project Directory
D:
cd "D:\finalyearmajorpro\Machine-Learning_Deep-learning_Free-Download-381de74cb080305f43ffb710db13f3e6f5ce54e0\A Learner's Guide to Model Selection and Tuning"

echo Running Churn Prediction Analysis
python employee_churn_analysis.py

pause
