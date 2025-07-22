import sys
import os
import traceback

def print_system_info():
    print("System Information:")
    print(f"Python Version: {sys.version}")
    print(f"Python Executable: {sys.executable}")
    print(f"Current Working Directory: {os.getcwd()}")

def check_project_files():
    print("\nChecking Project Files:")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    files = ['employee_data.csv', 'Learner_Notebook3.ipynb']
    
    for file in files:
        file_path = os.path.join(current_dir, file)
        if os.path.exists(file_path):
            print(f"✓ {file} exists")
        else:
            print(f"✗ {file} is MISSING")

def main():
    print("Employee Churn Prediction Project Diagnostic")
    print("=" * 40)
    
    print_system_info()
    check_project_files()

if __name__ == "__main__":
    main()
