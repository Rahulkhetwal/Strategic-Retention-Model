@echo on
title Strategic Retention Model
echo Activating Conda Environment...
call conda activate churn_prediction
echo Running Strategic Retention Model...
python "D:\finalyearmajorpro\Machine-Learning_Deep-learning_Free-Download-381de74cb080305f43ffb710db13f3e6f5ce54e0\A Learner's Guide to Model Selection and Tuning\employee_churn_analysis.py"
echo.
echo Press CTRL+C to exit or close the window manually
ping localhost -n -1 >nul
