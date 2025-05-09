import sys
import os
import platform
import subprocess

def run_command(command):
    try:
        result = subprocess.run(command, capture_output=True, text=True, shell=True)
        return {
            'stdout': result.stdout,
            'stderr': result.stderr,
            'returncode': result.returncode
        }
    except Exception as e:
        return {
            'stdout': '',
            'stderr': str(e),
            'returncode': -1
        }

def main():
    print("Comprehensive System and Python Diagnostic")
    print("=" * 40)
    
    # System Information
    print("\n1. System Details:")
    print(f"OS: {platform.platform()}")
    print(f"Python Version: {sys.version}")
    print(f"Python Executable: {sys.executable}")
    
    # Current Directory Information
    print("\n2. Directory Information:")
    print(f"Current Working Directory: {os.getcwd()}")
    
    # List Directory Contents
    print("\n3. Directory Contents:")
    try:
        contents = os.listdir('.')
        for item in contents:
            print(item)
    except Exception as e:
        print(f"Error listing directory: {e}")
    
    # Check for specific project files
    print("\n4. Project Files Check:")
    project_files = [
        'employee_data.csv',
        'Learner_Notebook3.ipynb',
        'simple_churn_analysis.py'
    ]
    
    for file in project_files:
        if os.path.exists(file):
            print(f"✓ {file} exists")
        else:
            print(f"✗ {file} is MISSING")
    
    # Python Path and Environment
    print("\n5. Python Environment:")
    print(f"Python Path: {sys.path}")
    
    # Attempt to import key libraries
    print("\n6. Library Import Check:")
    libraries = ['pandas', 'numpy', 'sklearn', 'matplotlib']
    for lib in libraries:
        try:
            __import__(lib)
            print(f"✓ {lib} can be imported")
        except ImportError:
            print(f"✗ {lib} CANNOT be imported")

if __name__ == "__main__":
    main()
