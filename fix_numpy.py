#!/usr/bin/env python3
"""
Fix Numpy Issues
===============
Fixes numpy and pandas compatibility issues
"""

import subprocess
import sys

def run_command(command):
    """Run a command"""
    try:
        print(f"Running: {command}")
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ Success: {command}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {command}")
        print(f"Error: {e.stderr}")
        return False

def main():
    print("🔧 Fixing numpy issues...")
    
    # Step 1: Uninstall problematic packages
    print("\n1. Removing problematic packages...")
    run_command("pip uninstall numpy pandas -y")
    
    # Step 2: Install numpy first
    print("\n2. Installing numpy...")
    run_command("pip install numpy==1.24.3")
    
    # Step 3: Install pandas
    print("\n3. Installing pandas...")
    run_command("pip install pandas==2.0.3")
    
    # Step 4: Install other required packages
    print("\n4. Installing other packages...")
    run_command("pip install torch transformers datasets scikit-learn")
    
    print("\n🎉 Environment fixed!")
    print("Now try running your project again.")

if __name__ == "__main__":
    main()
