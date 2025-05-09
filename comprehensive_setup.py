import subprocess
import sys
import os

def run_command(command):
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            capture_output=True, 
            text=True
        )
        print(f"Command: {command}")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        return result.returncode
    except Exception as e:
        print(f"Error running command: {e}")
        return -1

def setup_conda_environment():
    print("Setting up Conda Environment")
    print("===========================")
    
    # Create conda environment
    create_env_cmd = "conda create -n churn_prediction python=3.12 pandas numpy scikit-learn matplotlib seaborn -y"
    run_command(create_env_cmd)
    
    # Activate environment and verify installations
    activate_cmd = "conda run -n churn_prediction python -c \"import pandas; import numpy; import sklearn; print('Dependencies installed successfully!')\""
    run_command(activate_cmd)

def run_churn_analysis():
    print("\nRunning Churn Analysis")
    print("=====================")
    
    # Change to project directory
    os.chdir(r"D:\finalyearmajorpro\Machine-Learning_Deep-learning_Free-Download-381de74cb080305f43ffb710db13f3e6f5ce54e0\A Learner's Guide to Model Selection and Tuning")
    
    # Run analysis script in conda environment
    run_cmd = "conda run -n churn_prediction python employee_churn_analysis.py"
    run_command(run_cmd)

def main():
    setup_conda_environment()
    run_churn_analysis()

if __name__ == "__main__":
    main()
