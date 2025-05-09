import sys
import os
import traceback

def print_system_info():
    print("System Information:")
    print(f"Python Version: {sys.version}")
    print(f"Python Executable: {sys.executable}")
    print(f"Current Working Directory: {os.getcwd()}")
    print(f"Script Directory: {os.path.dirname(os.path.abspath(__file__))}")

def load_data(filename):
    try:
        print(f"Attempting to load file: {filename}")
        print(f"Full file path: {os.path.abspath(filename)}")
        
        if not os.path.exists(filename):
            print(f"ERROR: File {filename} does not exist!")
            return []
        
        with open(filename, 'r') as f:
            lines = f.readlines()
            print(f"File contents:")
            for line in lines[:5]:  # Print first 5 lines
                print(line.strip())
            
            return lines
    except Exception as e:
        print(f"Error loading file: {e}")
        traceback.print_exc()
        return []

def main():
    try:
        print_system_info()
        
        # Hardcoded filename
        filename = 'employee_data.csv'
        
        # Load data
        data = load_data(filename)
        
        if not data:
            print("No data could be loaded. Exiting.")
            return
        
        print(f"Total lines in file: {len(data)}")
    
    except Exception as e:
        print(f"Unexpected error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
