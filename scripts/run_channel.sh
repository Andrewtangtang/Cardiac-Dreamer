#!/bin/bash
# Production Training Script for Cardiac Dreamer
# Channel-token version with cross-patient validation

echo "🔥 Starting Cardiac Dreamer Training..."
echo "================================================"

# Configuration
DATA_DIR="data/processed"
OUTPUT_DIR="outputs"
BATCH_SIZE=8
MAX_EPOCHS=150
LEARNING_RATE=1e-4

# Check if data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "❌ Error: Data directory $DATA_DIR not found!"
    echo "Please ensure your processed data is available."
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

echo "📊 Configuration:"
echo "  Data Directory: $DATA_DIR"
echo "  Output Directory: $OUTPUT_DIR"
echo "  Batch Size: $BATCH_SIZE"
echo "  Max Epochs: $MAX_EPOCHS"
echo "  Learning Rate: $LEARNING_RATE"
echo ""

# Run training with automatic patient splitting
echo "🚀 Launching training with automatic patient splitting..."
python src/train.py \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR"

# Check if training was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Training completed successfully!"
    echo ""
    echo "📈 To view training logs:"
    echo "  tensorboard --logdir $OUTPUT_DIR/run_*/logs"
    echo ""
    echo "📁 Results saved in: $OUTPUT_DIR"
    echo "  - Checkpoints: $OUTPUT_DIR/run_*/checkpoints/"
    echo "  - Plots: $OUTPUT_DIR/run_*/plots/"
    echo "  - Logs: $OUTPUT_DIR/run_*/logs/"
else
    echo ""
    echo "❌ Training failed! Check the error messages above."
    exit 1
fi 