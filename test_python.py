import sys
import platform

print("Python Version:", sys.version)
print("Python Executable:", sys.executable)
print("Platform:", platform.platform())

try:
    import pandas
    print("Pandas Version:", pandas.__version__)
except ImportError:
    print("Pandas is not installed")

try:
    import numpy
    print("NumPy Version:", numpy.__version__)
except ImportError:
    print("NumPy is not installed")

try:
    import sklearn
    print("Scikit-learn Version:", sklearn.__version__)
except ImportError:
    print("Scikit-learn is not installed")
