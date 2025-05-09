@echo off
echo Python and Dependency Diagnostic
echo ================================

echo 1. Python Version
python --version

echo 2. Pip Version
python -m pip --version

echo 3. Python Architecture
python -c "import platform; print(platform.architecture()[0])"

echo 4. Pip Configuration
python -m pip config list

echo 5. Site Packages Location
python -c "import site; print(site.getsitepackages())"

echo 6. User Site Packages Location
python -c "import site; print(site.getusersitepackages())"

pause
