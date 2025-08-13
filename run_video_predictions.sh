#!/bin/bash

# Video-Specific Prediction Script
# This script runs predictions for each video folder using their corresponding validation models

# Set default values
CV_OUTPUT_DIR="outputs/cross_validation_20250609_212836"
DATA_DIR="data/processed"
OUTPUT_DIR="video_predictions"
BATCH_SIZE=8
DEVICE="auto"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --cv_output_dir)
            CV_OUTPUT_DIR="$2"
            shift 2
            ;;
        --data_dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --skip_existing)
            SKIP_EXISTING="--skip_existing"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --cv_output_dir DIR    Cross-validation output directory"
            echo "                         (default: outputs/cross_validation_20250609_212836)"
            echo "  --data_dir DIR         Data directory (default: data/processed)" 
            echo "  --output_dir DIR       Output directory (default: video_predictions)"
            echo "  --batch_size SIZE      Batch size (default: 8)"
            echo "  --device DEVICE        Device (auto/cpu/cuda, default: auto)"
            echo "  --skip_existing        Skip existing prediction files"
            echo "  -h, --help             Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Run with default settings"
            echo "  $0 --batch_size 16 --device cuda     # Use larger batch size and force GPU"
            echo "  $0 --skip_existing                    # Skip already processed videos"
            echo ""
            echo "This script will generate one CSV file per video folder:"
            echo "  - data_0513_01_predictions.csv"
            echo "  - data_0513_02_predictions.csv"
            echo "  - data_0513_07_predictions.csv"
            echo "  - ... etc"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "VIDEO-SPECIFIC PREDICTION SCRIPT"
echo "=========================================="
echo "CV Output Directory: $CV_OUTPUT_DIR"
echo "Data Directory: $DATA_DIR"
echo "Output Directory: $OUTPUT_DIR"
echo "Batch Size: $BATCH_SIZE"
echo "Device: $DEVICE"
echo "Skip Existing: ${SKIP_EXISTING:-false}"
echo "=========================================="
echo ""

# Check if the Python script exists
if [ ! -f "generate_video_predictions.py" ]; then
    echo "Error: generate_video_predictions.py not found!"
    echo "Please run this script from the Cardiac-Dreamer directory."
    exit 1
fi

# Check if directories exist
if [ ! -d "$CV_OUTPUT_DIR" ]; then
    echo "Error: CV output directory does not exist: $CV_OUTPUT_DIR"
    echo "Make sure you have the correct path to your cross-validation results."
    exit 1
fi

if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory does not exist: $DATA_DIR"
    exit 1
fi

# Show what video folders will be processed
echo "Checking available video folders..."
if [ -d "$DATA_DIR" ]; then
    VIDEO_COUNT=$(ls -d "$DATA_DIR"/data_* 2>/dev/null | wc -l)
    if [ $VIDEO_COUNT -eq 0 ]; then
        echo "Error: No video folders (data_*) found in $DATA_DIR"
        exit 1
    fi
    echo "Found $VIDEO_COUNT video folders to process"
    echo ""
else
    echo "Error: Data directory not accessible: $DATA_DIR"
    exit 1
fi

# Run the Python script
echo "Starting video prediction generation..."
echo "This will create one CSV file per video folder..."
echo ""

python generate_video_predictions.py \
    --cv_output_dir "$CV_OUTPUT_DIR" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size "$BATCH_SIZE" \
    --device "$DEVICE" \
    $SKIP_EXISTING

PYTHON_EXIT_CODE=$?

echo ""
if [ $PYTHON_EXIT_CODE -eq 0 ]; then
    echo "=========================================="
    echo "VIDEO PREDICTION COMPLETED SUCCESSFULLY!"
    echo "=========================================="
    echo "Results saved in: $OUTPUT_DIR"
    echo ""
    echo "Generated files:"
    if [ -d "$OUTPUT_DIR" ]; then
        CSV_COUNT=$(ls "$OUTPUT_DIR"/*.csv 2>/dev/null | wc -l)
        if [ $CSV_COUNT -gt 0 ]; then
            echo "Total CSV files: $CSV_COUNT"
            echo ""
            echo "Example files:"
            ls "$OUTPUT_DIR"/*.csv 2>/dev/null | head -5
            if [ $CSV_COUNT -gt 5 ]; then
                echo "... and $((CSV_COUNT - 5)) more files"
            fi
        else
            echo "No CSV files found"
        fi
        
        if [ -f "$OUTPUT_DIR/video_prediction_summary.json" ]; then
            echo ""
            echo "Summary file: $OUTPUT_DIR/video_prediction_summary.json"
        fi
    fi
    
    echo ""
    echo "Each CSV file contains frame-by-frame predictions for one video,"
    echo "with the same format as demo_patient_07_predictions.csv"
else
    echo "=========================================="
    echo "VIDEO PREDICTION FAILED!"
    echo "=========================================="
    echo "Python script exited with code: $PYTHON_EXIT_CODE"
    exit $PYTHON_EXIT_CODE
fi 