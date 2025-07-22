import sys
import platform
import struct

def python_details():
    print("Python Architecture Diagnostic")
    print("==============================")
    
    # Python version and executable details
    print(f"Python Version: {sys.version}")
    print(f"Python Executable: {sys.executable}")
    
    # Bit architecture check
    print(f"\nIs 64-bit Python: {sys.maxsize > 2**32}")
    print(f"Bit Architecture: {struct.calcsize('P') * 8}-bit")
    
    # Platform details
    print(f"\nPlatform: {platform.platform()}")
    print(f"Machine: {platform.machine()}")

def pip_details():
    try:
        import pip
        print("\nPip Details")
        print("===========")
        print(f"Pip Version: {pip.__version__}")
        print(f"Pip Location: {pip.__file__}")
    except ImportError:
        print("\nPip is not properly installed.")

def site_packages_info():
    import site
    print("\nSite Packages Information")
    print("========================")
    print("User Site Packages:", site.getusersitepackages())
    print("Global Site Packages:", site.getsitepackages())

def main():
    python_details()
    pip_details()
    site_packages_info()

if __name__ == "__main__":
    main()
