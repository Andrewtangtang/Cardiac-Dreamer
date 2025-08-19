# Setup and Installation

This guide focuses on two essentials: setting up a clean Python environment, installing PyTorch correctly (via the official site), and then installing project dependencies.

## Environment Setup

### Virtual Environment Creation

Create an isolated Python environment using either conda or venv:

#### Option 1: Conda (Recommended)
```bash
# Create conda environment
conda create -n cardiac-dreamer python=3.10
conda activate cardiac-dreamer
```

#### Option 2: Virtual Environment
```bash
# Create virtual environment
python -m venv cardiac-dreamer-env

# Activate on Windows
cardiac-dreamer-env\Scripts\activate

# Activate on Linux/macOS
source cardiac-dreamer-env/bin/activate
```

#### Option 3: Pipenv
```bash
# Install pipenv if not available
pip install pipenv

# Create environment from Pipfile
pipenv install

# Activate environment
pipenv shell
```

## PyTorch Installation

Install PyTorch from the official website to match your OS, package manager, and CUDA version.

- Go to the PyTorch Get Started page: `https://pytorch.org/get-started/locally/`
- Select your OS, Python, and the correct CUDA compute platform for your GPU drivers
- Install the suggested versions of `torch`, `torchvision`, and `torchaudio`

Notes:
- Prefer a CUDA-enabled build if you have a compatible GPU; CPU-only is supported but not recommended for training
- Keep versions consistent (e.g., PyTorch 2.2.x with matching torchvision/torchaudio)

### Verification

Verify PyTorch installation and CUDA availability:

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
```

## Dependencies Installation

After PyTorch is installed, install the remaining dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

Key libraries used in this project:
- `pytorch-lightning==2.5.1.post0`: Training framework
- `wandb==0.19.11`: Experiment tracking
- `numpy==1.26.4`: Numerical computations
- `pillow==11.2.1`: Image processing
- `matplotlib==3.10.3`: Visualization
- `pandas==2.2.3`: Data manipulation
- `torchmetrics==1.7.1`: Model evaluation metrics

Optional (development/visualization): JupyterLab, seaborn, GPU monitoring tools.

## Configuration Verification

### CUDA Setup Check

Run the CUDA verification script:

```bash
python checkcuda.py
```

Expected output:
```text
CUDA is available: True
CUDA version: 11.8
Number of GPUs: 1
GPU 0: NVIDIA GeForce RTX 4080
```

### Dataset Structure Check (optional)
Ensure `data/processed/<patient>` folders exist (e.g., `data_0513_01`, `data_0513_02`) with `ft1/`, `ft2/`, and `transitions_dataset.json`.

### Model Configuration Test

Test model instantiation:

```python
from src.models.system import get_cardiac_dreamer_system

# Test model creation
model = get_cardiac_dreamer_system(
    d_model=768,
    num_heads=12,
    num_layers=6,
    feature_dim=49,
    in_channels=1,
    lr=5e-5,
    weight_decay=1e-4
)
print("Model created successfully")
```

## Installation Troubleshooting

### Common Issues

#### CUDA Out of Memory
- Reduce batch size in configuration files
- Use gradient accumulation: set `accumulate_grad_batches > 1`
- Enable mixed precision: set `precision: "16-mixed"`

#### PyTorch Version Conflicts
- Remove existing `torch`, `torchvision`, and `torchaudio`
- Reinstall from the official PyTorch site with versions matching your CUDA drivers

#### Missing CUDA Libraries
- Update NVIDIA drivers to a recent version
- Install the appropriate CUDA Toolkit (if required by your environment)
- Ensure CUDA paths are correctly configured on your system

#### Memory Issues During Data Loading
- Reduce `num_workers` in configuration
- Set `persistent_workers: false`
- Disable `pin_memory` if on CPU-only system

### Verification Commands

Complete installation verification:

```bash
# Check Python version
python --version

# Check PyTorch installation
python -c "import torch; print(torch.__version__)"

# Check Lightning installation
python -c "import lightning; print(lightning.__version__)"

# Check CUDA functionality
python -c "import torch; print(torch.cuda.is_available())"

# Test data loading
python -c "from src.data.dataset import CrossPatientTransitionsDataset; print('Dataset import successful')"

# Test model components
python -c "from src.models.backbone import ResNet34Backbone; print('Backbone import successful')"
```

## Next Steps

After successful installation:

1. **Data Preparation**: Follow [Data Management](03_data.md) for dataset setup
2. **Training**: Proceed to training workflows as described in [Model Architecture](04_model_architecture.md)
3. **Configuration**: Review and modify YAML configuration files in `configs/` directory

### Quick Start Verification

Run a minimal training test:

```bash
# Test single epoch training
python -m src.train_modular --data_dir data/processed --output_dir test_output --config configs/channel_token.yaml
```

This should complete without errors and create output directories with logs and checkpoints.
