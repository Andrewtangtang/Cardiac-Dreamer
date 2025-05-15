#!/usr/bin/env python
# Main training script

import os
import sys
import argparse
import yaml
import json
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import glob
from tqdm import tqdm

# Add project root to path
current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_file_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import model components
from src.models.system import get_cardiac_dreamer_system


class UltrasoundTransitionsDataset(Dataset):
    """
    Dataset for ultrasound image transitions with actions
    Reads from the processed data directory structure
    
    Args:
        data_dir: Data directory path (processed data folder)
        transform: Image transformations
        split: Data split ("train", "val", "test")
        small_subset: Whether to use only a small subset of data (for testing)
    """
    def __init__(
        self, 
        data_dir: str, 
        transform=None, 
        split: str = "train",
        small_subset: bool = False
    ):
        self.data_dir = data_dir
        self.transform = transform
        self.split = split
        self.small_subset = small_subset
        
        # Load transitions CSV (contains all the data we need)
        csv_file = os.path.join(data_dir, f"{split}_transitions.csv")
        
        # Check if file exists
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV file not found: {csv_file}")
            
        # Load transitions DataFrame
        self.transitions_df = pd.read_csv(csv_file)
        
        # Store column names for easier access
        self.current_pose_cols = ["at1_X_mm", "at1_Y_mm", "at1_Z_mm", "at1_Roll_deg", "at1_Pitch_deg", "at1_Yaw_deg"]
        self.target_change_cols = ["change_X_mm", "change_Y_mm", "change_Z_mm", "change_Roll_deg", "change_Pitch_deg", "change_Yaw_deg"]
        
        # Get paths to image folders
        self.images_t1_dir = os.path.join(data_dir, "images_ft1")
        self.images_t2_dir = os.path.join(data_dir, "images_ft2")
        
        # Image root check
        if not os.path.exists(self.images_t1_dir) or not os.path.exists(self.images_t2_dir):
            raise FileNotFoundError(f"Image directories not found: {self.images_t1_dir} or {self.images_t2_dir}")
            
        # For small subset testing, limit samples
        if small_subset:
            num_samples = min(10, len(self.transitions_df))
            self.transitions_df = self.transitions_df.iloc[:num_samples]
            print(f"Using small subset for testing: {num_samples} samples")
            
        print(f"Loaded {len(self.transitions_df)} {split} transitions from {data_dir}")
    
    def __len__(self) -> int:
        return len(self.transitions_df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Get transition data from DataFrame
        transition_row = self.transitions_df.iloc[idx]
        
        # Get image paths from row
        img_t1_path = os.path.join(self.data_dir, transition_row["ft1_image_path"])
        
        # Load current image (t1)
        image = Image.open(img_t1_path).convert("L")  # Convert to grayscale
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        # Current action (x, y, z, roll, pitch, yaw)
        current_action = transition_row[self.current_pose_cols].values.astype(np.float32)
        
        # Target action change (the desired action to move from t1 to t2)
        target_action = transition_row[self.target_change_cols].values.astype(np.float32)
        
        # Convert to tensors
        current_action = torch.tensor(current_action, dtype=torch.float32)
        target_action = torch.tensor(target_action, dtype=torch.float32)
        
        return image, current_action, target_action


def load_config(config_path: str) -> Dict:
    """
    Load YAML configuration file
    
    Args:
        config_path: Configuration file path
    
    Returns:
        Configuration dictionary
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_model_config() -> Dict:
    """
    Get model configuration
    
    Returns:
        Model configuration dictionary
    """
    # Load from actual config files, using hardcoded settings for now
    return {
        "token_type": "channel",
        "d_model": 768,
        "num_heads": 12,
        "num_layers": 6,
        "feature_dim": 49,
        "in_channels": 1,
        "use_pretrained": True,
        "lr": 1e-4,
        "weight_decay": 1e-5,
        "lambda_latent": 0.2,
        "use_flash_attn": False,
    }


def get_train_config() -> Dict:
    """
    Get training configuration
    
    Returns:
        Training configuration dictionary
    """
    # Load from actual config files, using hardcoded settings for now
    return {
        "batch_size": 4,
        "num_workers": 4,
        "max_epochs": 100,
        "early_stop_patience": 10,
        "accelerator": "auto",
        "precision": 16,
    }


def main(args):
    # Set up PyTorch
    pl.seed_everything(42)
    
    # Load configurations
    model_config = get_model_config()
    train_config = get_train_config()
    
    # Set up data transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Create datasets
    data_dir = args.data_dir
    train_dataset = UltrasoundTransitionsDataset(
        data_dir=data_dir,
        transform=transform,
        split="train",
        small_subset=args.small_subset
    )
    
    val_dataset = UltrasoundTransitionsDataset(
        data_dir=data_dir,
        transform=transform,
        split="val",
        small_subset=args.small_subset
    )
    
    test_dataset = UltrasoundTransitionsDataset(
        data_dir=data_dir,
        transform=transform,
        split="test",
        small_subset=args.small_subset
    )
    
    print(f"Training set: {len(train_dataset)} samples")
    print(f"Validation set: {len(val_dataset)} samples")
    print(f"Test set: {len(test_dataset)} samples")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config["batch_size"],
        shuffle=True,
        num_workers=train_config["num_workers"],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config["batch_size"],
        shuffle=False,
        num_workers=train_config["num_workers"],
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=train_config["batch_size"],
        shuffle=False,
        num_workers=train_config["num_workers"],
        pin_memory=True
    )
    
    # Create model
    model = get_cardiac_dreamer_system(
        token_type=model_config["token_type"],
        d_model=model_config["d_model"],
        nhead=model_config["num_heads"],
        num_layers=model_config["num_layers"],
        feature_dim=model_config["feature_dim"],
        lr=model_config["lr"],
        lambda_latent=model_config["lambda_latent"],
        use_flash_attn=model_config["use_flash_attn"]
    )
    
    # Set up callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.output_dir, "checkpoints"),
        filename="cardiac_dreamer-{epoch:02d}-{val_loss:.4f}",
        save_top_k=3,
        monitor="val_loss",
        mode="min"
    )
    
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=train_config["early_stop_patience"],
        mode="min",
        verbose=True
    )
    
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    
    # Set up logger
    logger = TensorBoardLogger(
        save_dir=os.path.join(args.output_dir, "logs"),
        name="cardiac_dreamer"
    )
    
    # Set up trainer
    trainer = pl.Trainer(
        max_epochs=train_config["max_epochs"],
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        logger=logger,
        accelerator=train_config["accelerator"],
        precision=train_config["precision"],
        log_every_n_steps=10,
        enable_progress_bar=True
    )
    
    # Start training
    print("Starting training...")
    trainer.fit(model, train_loader, val_loader)
    
    # Test model
    print("Testing model...")
    trainer.test(model, test_loader)
    
    print(f"Done! Model saved to {os.path.join(args.output_dir, 'checkpoints')}")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train cardiac ultrasound guidance model")
    parser.add_argument("--data_dir", type=str, default="data/processed", help="Data directory path")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory path")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Configuration file path")
    parser.add_argument("--small_subset", action="store_true", help="Use small subset for testing")
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "logs"), exist_ok=True)
    
    main(args) 