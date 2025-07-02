#!/usr/bin/env python
"""
Cross-Validation Launcher Script for Cardiac Dreamer

Simple script to launch 5-fold cross-validation with proper configuration.
"""

import os
import subprocess
import sys
from datetime import datetime

def main():
    """Launch cross-validation training"""
    
    print("=" * 80)
    print("CARDIAC DREAMER - CROSS-VALIDATION LAUNCHER")
    print("=" * 80)
    print()
    
    # Configuration
    data_dir = "data/processed"
    output_dir = "outputs"
    config_file = "configs/channel_token_cv.yaml"
    
    # Check if required files exist
    if not os.path.exists(config_file):
        print(f" ERROR: Configuration file not found: {config_file}")
        return
    
    if not os.path.exists(data_dir):
        print(f" ERROR: Data directory not found: {data_dir}")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f" Data Directory: {data_dir}")
    print(f" Output Directory: {output_dir}")
    print(f" Configuration: {config_file}")
    print()
    
    print(" STARTING 5-FOLD CROSS-VALIDATION")
    print("   - Each fold trains for 100 epochs")
    print("   - Each patient group serves as validation once")
    print("   - Results will be automatically analyzed and reported")
    print()
    
    # Construct command
    cmd = [
        sys.executable,
        "src/train_cross_validation.py",
        "--data_dir", data_dir,
        "--output_dir", output_dir,
        "--config", config_file
    ]
    
    print(f"🔧 Command: {' '.join(cmd)}")
    print()
    
    # Ask for confirmation
    response = input(" Start cross-validation? This will take several hours. (y/N): ")
    if response.lower() not in ['y', 'yes']:
        print(" Cross-validation cancelled by user.")
        return
    
    print()
    print(" Starting cross-validation at:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print()
    
    try:
        # Run cross-validation
        result = subprocess.run(cmd, check=True)
        
        print()
        print("=" * 80)
        print(" CROSS-VALIDATION COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print()
        print(" Check the outputs directory for:")
        print("   - cv_summary_report.json: Complete numerical results")
        print("   - cv_detailed_results.csv: Tabular results")  
        print("   - cv_results_visualization.png: Result plots")
        print("   - Individual fold directories with detailed logs")
        print()
        
    except subprocess.CalledProcessError as e:
        print()
        print("=" * 80)
        print(" CROSS-VALIDATION FAILED!")
        print("=" * 80)
        print(f"Error code: {e.returncode}")
        print("Check the console output above for error details.")
        print()
        
    except KeyboardInterrupt:
        print()
        print("=" * 80)
        print(" CROSS-VALIDATION INTERRUPTED BY USER")
        print("=" * 80)
        print("Partial results may be available in the outputs directory.")
        print()


if __name__ == "__main__":
    main() 