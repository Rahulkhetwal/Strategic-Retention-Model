import sys
import platform
import subprocess

def run_command(command):
    try:
        result = subprocess.run(command, capture_output=True, text=True, shell=True)
        print(f"Command: {command}")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        print("Return Code:", result.returncode)
    except Exception as e:
        print(f"Error running command {command}: {e}")

def check_python_environment():
    print("Python Environment Diagnostic")
    print("=" * 30)
    
    # Python details
    print("\nPython Details:")
    print("Version:", sys.version)
    print("Executable Path:", sys.executable)
    print("Platform:", platform.platform())
    
    # Pip version
    print("\nPip Version:")
    run_command("pip --version")
    
    # List installed packages
    print("\nInstalled Packages:")
    run_command("pip list")
    
    # Check package installation
    print("\nPackage Installation Test:")
    run_command("pip install pandas numpy scikit-learn matplotlib seaborn")

if __name__ == "__main__":
    check_python_environment()
