# Cardiac Dreamer - Production Training Guide

## 🎯 Overview

A comprehensive production training system for the Cardiac Dreamer model, featuring complete visualization, model checkpointing, and training monitoring capabilities. This system is designed for cross-patient validation in medical AI applications.

## 🚀 Quick Start

### Windows Users
```bash
# Double-click to run or execute in command prompt
scripts\run_training.bat
```

### Linux/Mac Users
```bash
# Grant execution permissions
chmod +x scripts/run_channel.sh

# Run training
scripts/run_channel.sh
```

### Direct Python Execution
```bash
python src/train.py --data_dir data/processed --output_dir outputs
```

## 📁 Output Structure

Each training run creates a timestamped directory with complete experimental artifacts:

```
outputs/
└── run_20241216_143022/
    ├── experiment_config.json     # Complete experiment configuration
    ├── dataset_stats.json         # Dataset statistics and splits
    ├── final_model.ckpt          # Final trained model
    ├── checkpoints/              # Training checkpoints
    │   ├── cardiac_dreamer-epoch=XX-val_main_task_loss=X.XXXX.ckpt
    │   └── last.ckpt
    ├── logs/                     # Training logs
    │   ├── cardiac_dreamer/      # TensorBoard logs
    │   └── cardiac_dreamer_csv/  # CSV format logs
    └── plots/                    # Visualization outputs
        ├── dataset_statistics.png
        └── training_summary.png
```

## 📊 Visualization Outputs

### 1. Dataset Statistics (`dataset_statistics.png`)
- Train/validation/test split ratios
- Patient-wise sample distribution
- Sample image previews
- Cross-patient split validation

### 2. Training Summary (`training_summary.png`)
- Training/validation loss curves
- Main task loss progression
- Learning rate scheduling
- Component-wise loss trends
- Final training metrics and statistics

## ⚙️ Configuration

### Using Configuration Files
```bash
python src/train.py --config configs/production.yaml
```

### Key Configuration Parameters

**Model Architecture**:
- `d_model`: Transformer dimension (default: 768)
- `num_heads`: Number of attention heads (default: 12)
- `num_layers`: Number of transformer layers (default: 6)
- `token_type`: Token strategy ("channel" or "spatial")

**Training Parameters**:
- `batch_size`: Batch size for training (default: 16)
- `max_epochs`: Maximum training epochs (default: 150)
- `lr`: Learning rate (default: 2e-4)
- `early_stop_patience`: Early stopping patience (default: 20)
- `precision`: Training precision (16/32 bit)

**Loss Function**:
- `lambda_latent`: Latent reconstruction loss weight (default: 0.2)
- `lambda_t2_action`: Auxiliary task loss weight (default: 1.0)
- `smooth_l1_beta`: Beta parameter for SmoothL1Loss (default: 1.0)
- `primary_task_only`: Use only main task loss (default: false)

## 🎮 Training Modes

### 1. Automatic Patient Splitting (Default)
The system automatically detects patient folders and performs cross-patient splits:
- 70% for training
- 15% for validation
- 15% for testing

**Advantages**: Prevents data leakage, follows medical AI best practices

### 2. Manual Patient Splitting
```bash
python src/train.py \
    --manual_splits \
    --train_patients data_0513_01,data_0513_02 \
    --val_patients data_0513_03 \
    --test_patients data_0513_04
```

### 3. Resume Training
```bash
python src/train.py \
    --resume_from_checkpoint outputs/run_XXX/checkpoints/last.ckpt
```

## 📈 Training Monitoring

### TensorBoard Integration
```bash
tensorboard --logdir outputs/run_*/logs
```

### Key Metrics to Monitor
- `train_main_task_loss`: Main task training loss (primary focus)
- `val_main_task_loss`: Main task validation loss (primary focus)
- `train_total_loss`: Total training loss
- `val_loss`: Total validation loss
- `learning_rate`: Current learning rate
- Individual component losses (latent, auxiliary)

## 💾 Model Saving Strategy

### Automatic Checkpointing
- **Best Models**: Top 3 checkpoints based on `val_main_task_loss`
- **Latest Model**: Every epoch saves `last.ckpt`
- **Final Model**: Training completion saves `final_model.ckpt`

### Checkpoint Contents
Complete model state including:
- Model weights and architecture
- Optimizer state
- Learning rate scheduler state
- Training step and epoch counters
- Hyperparameters and configuration

## 🔧 Advanced Features

### Gradient Management
- **Gradient Clipping**: Prevents gradient explosion
- **Gradient Accumulation**: Simulates larger batch sizes
- **Mixed Precision**: 16-bit training for acceleration

### Hardware Optimization
- **Automatic GPU/CPU Selection**: Intelligent device detection
- **Multi-Core Data Loading**: Parallel data preprocessing
- **Persistent Workers**: Optimized data loader performance

### Early Stopping
- **Validation Loss Monitoring**: Prevents overfitting
- **Configurable Patience**: Customizable early stopping threshold
- **Learning Rate Scheduling**: Adaptive learning rate reduction

## 🚨 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```yaml
training:
  batch_size: 8         # Reduce from 16
  accumulate_grad_batches: 2  # Simulate larger batches
```

**2. Slow Data Loading**
```yaml
training:
  num_workers: 2        # Reduce from 4
  # Ensure data is on fast storage (SSD)
```

**3. Training Not Converging**
- Check learning rate appropriateness
- Verify data preprocessing correctness
- Analyze loss curve trends
- Ensure sufficient training data

**4. Memory Issues**
```yaml
training:
  precision: 16         # Use mixed precision
  batch_size: 4         # Further reduce batch size
```

### Debug Mode
```bash
# Use small subset for testing
python src/train.py --config configs/debug.yaml
```

## 📋 Pre-Training Checklist

Verify before starting training:

- [ ] Data directory `data/processed` exists with patient folders
- [ ] Each patient folder contains `transitions_dataset.json`
- [ ] GPU/CUDA environment is properly configured (optional)
- [ ] Sufficient disk space for checkpoints and logs
- [ ] Configuration parameters match requirements
- [ ] Dependencies are installed and up-to-date

## 🎯 Performance Benchmarks

### Expected Training Times
- **Small Dataset** (3 patients, ~300 samples): 10-30 minutes
- **Medium Dataset** (10 patients, ~1000 samples): 1-3 hours
- **Large Dataset** (20+ patients, ~3000+ samples): 3-12 hours

### Memory Requirements
- **GPU**: 4GB+ VRAM (batch_size=8), 6GB+ recommended (batch_size=16)
- **RAM**: 8GB+ system memory recommended
- **Storage**: ~500MB-2GB per training run

### Scaling Guidelines

**For 10x More Data**:
```yaml
model:
  lr: 2.0e-4              # Scaled from 1e-4

training:
  batch_size: 16          # Scaled from 8
  max_epochs: 100         # May reduce due to more data
  early_stop_patience: 15 # Adjusted for larger dataset
```

## 📚 Dataset Requirements

### Directory Structure
```
data/processed/
├── data_patient_01/
│   ├── transitions_dataset.json
│   ├── ft1/
│   └── ft2/
├── data_patient_02/
│   └── ...
└── data_patient_N/
```

### JSON Format
Each `transitions_dataset.json` should contain:
```json
[
  {
    "ft1_image_path": "ft1/image_001.png",
    "ft2_image_path": "ft2/image_001.png", 
    "at1_6dof": [x, y, z, roll, pitch, yaw],
    "action_change_6dof": [dx, dy, dz, droll, dpitch, dyaw],
    "at2_6dof": [x2, y2, z2, roll2, pitch2, yaw2]
  }
]
```

## 🔬 Medical AI Best Practices

This training system implements several medical AI best practices:

1. **Cross-Patient Validation**: Strict separation of patients across splits
2. **No Data Leakage**: Same patient never appears in both train and test
3. **Reproducible Results**: Fixed random seeds and deterministic training
4. **Comprehensive Logging**: Complete experiment tracking for reproducibility
5. **Model Versioning**: Automatic checkpointing and model archival

## 📊 Results Analysis

### Model Evaluation
```bash
# Load best checkpoint for evaluation
python src/evaluate.py --checkpoint outputs/run_XXX/checkpoints/best.ckpt
```

### Visualization Analysis
- Monitor convergence in loss plots
- Check for overfitting signs
- Verify cross-patient generalization
- Analyze per-patient performance

The production training system is now fully configured to handle large-scale data with comprehensive visualization and monitoring capabilities for medical AI research! 