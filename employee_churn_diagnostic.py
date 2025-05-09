import sys
import os
import traceback

def check_dependencies():
    print("Checking Python Dependencies:")
    dependencies = [
        'pandas', 'numpy', 'sklearn', 'matplotlib', 'seaborn'
    ]
    missing_deps = []
    
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"✓ {dep} is installed")
        except ImportError:
            print(f"✗ {dep} is NOT installed")
            missing_deps.append(dep)
    
    return missing_deps

def check_project_files():
    print("\nChecking Project Files:")
    required_files = [
        'employee_data.csv',
        'Learner_Notebook3.ipynb'
    ]
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    missing_files = []
    
    for file in required_files:
        file_path = os.path.join(current_dir, file)
        if os.path.exists(file_path):
            print(f"✓ {file} exists")
        else:
            print(f"✗ {file} is MISSING")
            missing_files.append(file)
    
    return missing_files

def main():
    print("Employee Churn Prediction Project Diagnostic")
    print("=" * 40)
    
    print("\nSystem Information:")
    print(f"Python Version: {sys.version}")
    print(f"Python Executable: {sys.executable}")
    print(f"Current Working Directory: {os.getcwd()}")
    
    missing_deps = check_dependencies()
    missing_files = check_project_files()
    
    print("\nDiagnostic Summary:")
    if missing_deps:
        print(f"Missing Dependencies: {', '.join(missing_deps)}")
        print("Please install these packages using: pip install " + " ".join(missing_deps))
    
    if missing_files:
        print(f"Missing Files: {', '.join(missing_files)}")
    
    if not missing_deps and not missing_files:
        print("✓ All dependencies and files are present!")
        print("Project is ready to run.")

if __name__ == "__main__":
    main()
