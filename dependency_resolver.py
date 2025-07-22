import sys
import subprocess
import platform
import os

def run_command(command):
    try:
        process = subprocess.Popen(
            command, 
            shell=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            universal_newlines=True
        )
        stdout, stderr = process.communicate()
        
        print(f"Command: {command}")
        print("STDOUT:", stdout)
        print("STDERR:", stderr)
        
        return process.returncode
    except Exception as e:
        print(f"Error running command: {e}")
        return -1

def system_diagnostic():
    print("System Diagnostic")
    print("================")
    print(f"Python Version: {sys.version}")
    print(f"Python Executable: {sys.executable}")
    print(f"Platform: {platform.platform()}")
    print(f"Architecture: {platform.architecture()[0]}")
    print(f"Current Directory: {os.getcwd()}")

def install_build_dependencies():
    print("\nInstalling Build Dependencies")
    print("===========================")
    
    commands = [
        # Upgrade pip and setuptools
        "python -m pip install --upgrade pip setuptools wheel",
        
        # Install build tools
        "python -m pip install --user build",
        
        # Pre-install numpy first
        "python -m pip install --user numpy==2.2.5",
        
        # Install other dependencies with fallback methods
        "python -m pip install --user --only-binary=:all: pandas scikit-learn matplotlib seaborn",
        
        # Alternative installation method
        "python -m pip install --user pandas scikit-learn matplotlib seaborn"
    ]
    
    for cmd in commands:
        print(f"\nTrying: {cmd}")
        result = run_command(cmd)
        if result == 0:
            print("Command successful!")
            break

def verify_installations():
    print("\nVerifying Installations")
    print("======================")
    
    libraries = [
        'numpy', 
        'pandas', 
        'sklearn', 
        'matplotlib', 
        'seaborn'
    ]
    
    for lib in libraries:
        try:
            __import__(lib)
            print(f"✓ {lib} is successfully installed")
        except ImportError:
            print(f"✗ {lib} CANNOT be imported")

def main():
    system_diagnostic()
    install_build_dependencies()
    verify_installations()

if __name__ == "__main__":
    main()
