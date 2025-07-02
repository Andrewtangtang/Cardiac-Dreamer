#!/bin/bash

# Batch Patient Prediction Script
# This script runs predictions for all patients using their validation models

# Set default values
CV_OUTPUT_DIR="outputs"
DATA_DIR="data/processed"
OUTPUT_DIR="all_patient_predictions"
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
            echo "  --cv_output_dir DIR    Cross-validation output directory (default: outputs)"
            echo "  --data_dir DIR         Data directory (default: data/processed)" 
            echo "  --output_dir DIR       Output directory (default: all_patient_predictions)"
            echo "  --batch_size SIZE      Batch size (default: 8)"
            echo "  --device DEVICE        Device (auto/cpu/cuda, default: auto)"
            echo "  --skip_existing        Skip existing prediction files"
            echo "  -h, --help             Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Run with default settings"
            echo "  $0 --batch_size 16 --device cuda     # Use larger batch size and force GPU"
            echo "  $0 --skip_existing                    # Skip already processed patients"
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
echo "BATCH PATIENT PREDICTION SCRIPT"
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
if [ ! -f "generate_all_patient_predictions.py" ]; then
    echo "Error: generate_all_patient_predictions.py not found!"
    echo "Please run this script from the Cardiac-Dreamer directory."
    exit 1
fi

# Check if directories exist
if [ ! -d "$CV_OUTPUT_DIR" ]; then
    echo "Error: CV output directory does not exist: $CV_OUTPUT_DIR"
    exit 1
fi

if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory does not exist: $DATA_DIR"
    exit 1
fi

# Run the Python script
echo "Starting batch prediction generation..."
echo ""

python generate_all_patient_predictions.py \
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
    echo "BATCH PREDICTION COMPLETED SUCCESSFULLY!"
    echo "=========================================="
    echo "Results saved in: $OUTPUT_DIR"
    echo ""
    echo "Generated files:"
    if [ -d "$OUTPUT_DIR" ]; then
        ls -la "$OUTPUT_DIR"/*.csv 2>/dev/null || echo "No CSV files found"
        if [ -f "$OUTPUT_DIR/batch_prediction_summary.json" ]; then
            echo ""
            echo "Summary file: $OUTPUT_DIR/batch_prediction_summary.json"
        fi
    fi
else
    echo "=========================================="
    echo "BATCH PREDICTION FAILED!"
    echo "=========================================="
    echo "Python script exited with code: $PYTHON_EXIT_CODE"
    exit $PYTHON_EXIT_CODE
fi 