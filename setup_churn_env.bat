@echo off
echo Creating Conda environment for Churn Prediction Project
conda create -n churn_prediction python=3.12 pandas numpy scikit-learn matplotlib seaborn -y

echo Activating environment
conda activate churn_prediction

echo Verifying library installations
python -c "import pandas; import numpy; import sklearn; import matplotlib; import seaborn; print('All libraries imported successfully!')"

echo Environment setup complete
pause
