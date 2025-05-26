#!/usr/bin/env python
"""
Simple Dataset Analysis Execution Script
"""

import subprocess
import sys
import os

def main():
    # Set default paths
    data_dir = "data/processed"
    output_dir = "dataset_analysis"
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        print(f"[ERROR] Data directory does not exist: {data_dir}")
        print("Please verify the data directory path")
        return
    
    print("[INFO] Starting Cardiac Dreamer Dataset Analysis...")
    print(f"[DATA] Data Directory: {data_dir}")
    print(f"[OUTPUT] Output Directory: {output_dir}")
    
    # Execute analysis script
    try:
        cmd = [sys.executable, "src/analyze_dataset.py", 
               "--data_dir", data_dir, 
               "--output_dir", output_dir]
        
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        
        print("[SUCCESS] Analysis Complete!")
        print(f"[RESULTS] View Results: {output_dir}/")
        
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Analysis Failed: {e}")
        print(f"Error Output: {e.stderr}")
    except Exception as e:
        print(f"[ERROR] Execution Error: {e}")

if __name__ == "__main__":
    main() 