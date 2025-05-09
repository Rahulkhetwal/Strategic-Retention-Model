import sys
import subprocess
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

def system_info():
    print("System Information")
    print("==================")
    print(f"Python Version: {sys.version}")
    print(f"Python Executable: {sys.executable}")
    print(f"Platform: {sys.platform}")
    print(f"Architecture: {sys.maxsize > 2**32}")

def install_dependencies():
    print("\nDependency Installation Strategy")
    print("==============================")
    
    # Comprehensive installation commands
    commands = [
        # Upgrade pip and setuptools
        "python -m pip install --upgrade pip setuptools wheel",
        
        # Pre-install numpy with specific version
        "python -m pip install --user numpy==2.2.5 --use-deprecated=legacy-resolver",
        
        # Install other dependencies with legacy resolver
        "python -m pip install --user pandas scikit-learn matplotlib seaborn --use-deprecated=legacy-resolver",
        
        # Alternative installation method
        "python -m pip install --user --only-binary=:all: numpy pandas scikit-learn matplotlib seaborn"
    ]
    
    for cmd in commands:
        print(f"\nAttempting: {cmd}")
        result = run_command(cmd)
        if result == 0:
            print("Installation successful!")
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
    system_info()
    install_dependencies()
    verify_installations()

if __name__ == "__main__":
    main()
