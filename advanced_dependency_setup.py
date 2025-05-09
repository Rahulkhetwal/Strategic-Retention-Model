import sys
import subprocess
import platform

def run_command(command):
    try:
        result = subprocess.run(command, capture_output=True, text=True, shell=True)
        print(f"Command: {command}")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        return result.returncode
    except Exception as e:
        print(f"Error running command: {e}")
        return -1

def check_system_info():
    print("System Diagnostic Information")
    print("=" * 40)
    print(f"Python Version: {sys.version}")
    print(f"Python Executable: {sys.executable}")
    print(f"Platform: {platform.platform()}")
    print(f"Architecture: {platform.architecture()[0]}")

def install_dependencies():
    print("\nAttempting Dependency Installation")
    print("=" * 40)
    
    # List of dependencies to install
    dependencies = [
        'numpy', 
        'pandas', 
        'scikit-learn', 
        'matplotlib', 
        'seaborn'
    ]
    
    # Try different installation methods
    install_methods = [
        f"python -m pip install --user {' '.join(dependencies)}",
        f"python -m pip install {' '.join(dependencies)}",
        f"python -m pip install --upgrade {' '.join(dependencies)}"
    ]
    
    for method in install_methods:
        print(f"\nTrying installation method: {method}")
        result = run_command(method)
        
        if result == 0:
            print("Installation successful!")
            return True
    
    print("All installation attempts failed.")
    return False

def verify_installations():
    print("\nVerifying Installations")
    print("=" * 40)
    
    libraries = ['numpy', 'pandas', 'sklearn', 'matplotlib', 'seaborn']
    
    for lib in libraries:
        try:
            __import__(lib)
            print(f"✓ {lib} is successfully installed")
        except ImportError:
            print(f"✗ {lib} CANNOT be imported")

def main():
    check_system_info()
    install_dependencies()
    verify_installations()

if __name__ == "__main__":
    main()
