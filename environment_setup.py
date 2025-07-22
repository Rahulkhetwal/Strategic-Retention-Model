import sys
import subprocess
import platform
import os

def run_command(command):
    try:
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate()
        return {
            'stdout': stdout,
            'stderr': stderr,
            'returncode': process.returncode
        }
    except Exception as e:
        return {
            'stdout': '',
            'stderr': str(e),
            'returncode': -1
        }

def check_environment():
    print("Python Environment Diagnostic")
    print("=" * 30)
    
    # Python details
    print("\n1. Python Details:")
    print(f"Version: {sys.version}")
    print(f"Executable Path: {sys.executable}")
    print(f"Platform: {platform.platform()}")
    
    # Check pip
    print("\n2. Pip Check:")
    pip_check = run_command("pip --version")
    print(f"Pip Version Command Output:\n{pip_check['stdout']}")
    print(f"Pip Version Command Error:\n{pip_check['stderr']}")
    
    # Attempt package installation
    print("\n3. Package Installation Test:")
    install_cmd = "pip install pandas numpy scikit-learn matplotlib seaborn --user"
    install_result = run_command(install_cmd)
    print(f"Installation Command: {install_cmd}")
    print(f"Installation Output:\n{install_result['stdout']}")
    print(f"Installation Error:\n{install_result['stderr']}")
    print(f"Return Code: {install_result['returncode']}")
    
    # Verify package installation
    print("\n4. Verify Package Installation:")
    verify_cmd = "pip list | findstr /i \"pandas numpy scikit-learn matplotlib seaborn\""
    verify_result = run_command(verify_cmd)
    print(f"Installed Packages:\n{verify_result['stdout']}")

if __name__ == "__main__":
    check_environment()
