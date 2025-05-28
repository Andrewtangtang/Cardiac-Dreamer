#!/usr/bin/env python
# Production training script for Cardiac Dreamer

import os
import sys
import argparse
import yaml
import json
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, DeviceStatsMonitor
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import glob
from tqdm import tqdm
import wandb
from datetime import datetime
import shutil
import random

# Add project root to path
current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_file_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import model components
from src.models.system import get_cardiac_dreamer_system


class CrossPatientTransitionsDataset(Dataset):
    """
    Cross-patient dataset for ultrasound image transitions with actions
    Reads from multiple patient folders containing transitions_dataset.json files
    
    Data structure expected:
    data/processed/
    data_0513_01/
    
    Each JSON contains:
    {
        "ft1_image_path": "path/to/image",
        "ft2_image_path": "path/to/image", 
        "at1_6dof": [x, y, z, roll, pitch, yaw],
        "action_change_6dof": [dx, dy, dz, droll, dpitch, dyaw],
        "at2_6dof": [x2, y2, z2, roll2, pitch2, yaw2]
    }
    
    Args:
        data_dir: Root data directory path (e.g., "data/processed")
        transform: Image transformations
        split: Data split ("train", "val", "test")
        train_patients: List of patient IDs for training (e.g., ["data_0513_01", "data_0513_02"])
        val_patients: List of patient IDs for validation (e.g., ["data_0513_03"])
        test_patients: List of patient IDs for testing (e.g., ["data_0513_04"])
        small_subset: Whether to use only a small subset of data (for testing)
        normalize_actions: Whether to normalize 6DOF actions (default: True)
    """
    def __init__(
        self, 
        data_dir: str, 
        transform=None, 
        split: str = "train",
        train_patients: Optional[List[str]] = None,
        val_patients: Optional[List[str]] = None,
        test_patients: Optional[List[str]] = None,
        small_subset: bool = False,
        normalize_actions: bool = True
    ):
        self.data_dir = data_dir
        self.transform = transform
        self.split = split
        self.small_subset = small_subset
        self.normalize_actions = normalize_actions
        
        # Automatically detect patient splits if not provided
        if train_patients is None or val_patients is None or test_patients is None:
            print(f"Patient splits not provided, auto-detecting from {data_dir}...")
            auto_train, auto_val, auto_test = get_patient_splits(data_dir)
            
            # Use auto-detected splits as defaults
            if train_patients is None:
                train_patients = auto_train
            if val_patients is None:
                val_patients = auto_val
            if test_patients is None:
                test_patients = auto_test
            
            print(f"Auto-detected patient splits:")
            print(f"  Train: {train_patients}")
            print(f"  Val: {val_patients}")
            print(f"  Test: {test_patients}")
            
        # Select patients based on split
        if split == "train":
            self.patient_ids = train_patients
        elif split == "val":
            self.patient_ids = val_patients
        elif split == "test":
            self.patient_ids = test_patients
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'")
        
        print(f"Loading {split} data from patients: {self.patient_ids}")
        
        # Load all transitions from selected patients
        self.transitions = []
        self.load_transitions()
        
        # Compute normalization statistics if enabled
        if self.normalize_actions and split == "train":
            self.compute_normalization_stats()
        elif self.normalize_actions:
            # For val/test, load the normalization stats from training data
            norm_file = os.path.join(self.data_dir, "normalization_stats.json")
            if os.path.exists(norm_file):
                print(f"📊 Loading normalization stats from: {norm_file}")
                with open(norm_file, 'r') as f:
                    norm_stats = json.load(f)
                
                self.action_mean = np.array(norm_stats["action_mean"])
                self.action_std = np.array(norm_stats["action_std"])
                print(f"✅ Successfully loaded normalization stats for {split} split")
            else:
                print(f"❌ Normalization stats file not found: {norm_file}")
                print(f"   Using default values (mean=0, std=1) - this may cause issues!")
                self.action_mean = np.array([0.0] * 6)
                self.action_std = np.array([1.0] * 6)
        
        # For small subset testing, limit samples
        if small_subset:
            num_samples = min(10, len(self.transitions))
            self.transitions = self.transitions[:num_samples]
            print(f"Using small subset for testing: {num_samples} samples")
            
        print(f"Loaded {len(self.transitions)} {split} transitions from {len(self.patient_ids)} patients")
        
        if self.normalize_actions:
            print(f"📊 Action normalization enabled:")
            print(f"   Mean: {self.action_mean}")
            print(f"   Std: {self.action_std}")
    
    def compute_normalization_stats(self):
        """Compute normalization statistics for 6DOF actions"""
        print("Computing normalization statistics for 6DOF actions...")
        
        at1_actions = []
        at2_actions = []
        action_changes = []
        
        for transition in self.transitions:
            at1_actions.append(transition["at1_6dof"])
            at2_actions.append(transition["at2_6dof"])
            action_changes.append(transition["action_change_6dof"])
        
        # Convert to numpy arrays
        at1_actions = np.array(at1_actions)
        at2_actions = np.array(at2_actions)
        action_changes = np.array(action_changes)
        
        # 🎯 統一處理所有動作向量：將三種類型的動作向量合併
        # 這樣可以確保所有動作向量使用相同的統計量進行正規化
        all_actions = np.vstack([at1_actions, at2_actions, action_changes])
        
        # 對每個維度分別計算統計量
        self.action_mean = np.mean(all_actions, axis=0)
        self.action_std = np.std(all_actions, axis=0)
        
        # Prevent division by zero
        self.action_std = np.where(self.action_std < 1e-6, 1.0, self.action_std)
        
        print(f"Action statistics computed from {len(self.transitions)} transitions:")
        print(f"  Total action vectors: {all_actions.shape[0]} (AT1: {at1_actions.shape[0]}, AT2: {at2_actions.shape[0]}, Changes: {action_changes.shape[0]})")
        print(f"  Combined range: {all_actions.min():.3f} to {all_actions.max():.3f}")
        print(f"  Mean per dimension: {self.action_mean}")
        print(f"  Std per dimension: {self.action_std}")
        
        # 顯示各維度的詳細統計
        dimension_names = ['X (mm)', 'Y (mm)', 'Z (mm)', 'Roll (rad)', 'Pitch (rad)', 'Yaw (rad)']
        print(f"\n📊 Detailed statistics per dimension:")
        for i, name in enumerate(dimension_names):
            print(f"  {name}: mean={self.action_mean[i]:.4f}, std={self.action_std[i]:.4f}, range=[{all_actions[:, i].min():.3f}, {all_actions[:, i].max():.3f}]")
        
        # Save normalization stats for use in val/test
        norm_stats = {
            "action_mean": self.action_mean.tolist(),
            "action_std": self.action_std.tolist(),
            "dimension_names": dimension_names,
            "total_samples": int(all_actions.shape[0]),
            "at1_samples": int(at1_actions.shape[0]),
            "at2_samples": int(at2_actions.shape[0]),
            "change_samples": int(action_changes.shape[0])
        }
        
        norm_file = os.path.join(self.data_dir, "normalization_stats.json")
        with open(norm_file, 'w') as f:
            json.dump(norm_stats, f, indent=2)
        print(f"📁 Normalization stats saved to: {norm_file}")
    
    def normalize_action(self, action):
        """Normalize a 6DOF action using computed statistics"""
        if not self.normalize_actions:
            return action
        return (action - self.action_mean) / self.action_std
    
    def load_transitions(self):
        """Load transitions from all patient folders"""
        for patient_id in self.patient_ids:
            patient_dir = os.path.join(self.data_dir, patient_id)
            json_file = os.path.join(patient_dir, "transitions_dataset.json")
            
            if not os.path.exists(json_file):
                print(f"Warning: JSON file not found for patient {patient_id}: {json_file}")
                continue
                
            # Load transitions for this patient
            with open(json_file, 'r') as f:
                patient_transitions = json.load(f)
            
            # Add patient_id and full paths to each transition
            for transition in patient_transitions:
                # Convert relative paths to absolute paths
                ft1_path = os.path.join(patient_dir, transition["ft1_image_path"])
                ft2_path = os.path.join(patient_dir, transition["ft2_image_path"])
                
                # Verify files exist
                if not os.path.exists(ft1_path):
                    print(f"Warning: Image file not found: {ft1_path}")
                    continue
                    
                # Create complete transition record
                complete_transition = {
                    "patient_id": patient_id,
                    "ft1_image_path": ft1_path,
                    "ft2_image_path": ft2_path,
                    "at1_6dof": transition["at1_6dof"],
                    "action_change_6dof": transition["action_change_6dof"], 
                    "at2_6dof": transition["at2_6dof"]
                }
                
                self.transitions.append(complete_transition)
            
            print(f"Loaded {len(patient_transitions)} transitions from patient {patient_id}")
    
    def __len__(self) -> int:
        return len(self.transitions)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a single transition sample
        
        Returns:
            image_t1: Input image at time t1 [C, H, W]
            a_hat_t1_to_t2_gt: Ground truth relative action from t1 to t2 [6] (normalized)
            at1_6dof_gt: Ground truth action at t1 [6] (main task target, normalized)
            at2_6dof_gt: Ground truth action at t2 [6] (auxiliary task target, normalized)
        """
        # Get transition data
        transition = self.transitions[idx]
        
        # Load image at t1
        image_t1 = Image.open(transition["ft1_image_path"]).convert("L")  # Convert to grayscale
        
        # Apply transformations
        if self.transform:
            image_t1 = self.transform(image_t1)
        
        # Extract actions from transition
        at1_6dof = np.array(transition["at1_6dof"], dtype=np.float32)
        action_change_6dof = np.array(transition["action_change_6dof"], dtype=np.float32) 
        at2_6dof = np.array(transition["at2_6dof"], dtype=np.float32)
        
        # Normalize actions if enabled
        if self.normalize_actions:
            at1_6dof = self.normalize_action(at1_6dof)
            at2_6dof = self.normalize_action(at2_6dof)
            action_change_6dof = self.normalize_action(action_change_6dof)
        
        # Convert to tensors
        a_hat_t1_to_t2_gt = torch.tensor(action_change_6dof, dtype=torch.float32)
        at1_6dof_gt = torch.tensor(at1_6dof, dtype=torch.float32)
        at2_6dof_gt = torch.tensor(at2_6dof, dtype=torch.float32)
        
        return image_t1, a_hat_t1_to_t2_gt, at1_6dof_gt, at2_6dof_gt
    
    def get_patient_stats(self) -> Dict[str, int]:
        """Get statistics about patients in this split"""
        patient_counts = {}
        for transition in self.transitions:
            patient_id = transition["patient_id"]
            patient_counts[patient_id] = patient_counts.get(patient_id, 0) + 1
        return patient_counts


def get_patient_splits(data_dir: str, shuffle_patients: bool = True, balance_by_samples: bool = True, random_seed: int = 40) -> Tuple[List[str], List[str], List[str]]:
    """
    Automatically detect patient folders and create train/val/test splits
    
    Args:
        data_dir: Root data directory
        shuffle_patients: Whether to shuffle patients before splitting (recommended)
        balance_by_samples: Whether to consider sample counts when splitting patients
        random_seed: Random seed for reproducible splits
        
    Returns:
        Tuple of (train_patients, val_patients, test_patients)
    """
    # Set random seed for reproducible splits
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Find all patient directories
    patient_dirs = []
    patient_sample_counts = {}
    
    for item in os.listdir(data_dir):
        patient_path = os.path.join(data_dir, item)
        if os.path.isdir(patient_path) and item.startswith("data_"):
            # Check if it contains the required JSON file
            json_file = os.path.join(patient_path, "transitions_dataset.json")
            if os.path.exists(json_file):
                patient_dirs.append(item)
                
                # Count samples for this patient if balance_by_samples is True
                if balance_by_samples:
                    try:
                        with open(json_file, 'r') as f:
                            transitions = json.load(f)
                        patient_sample_counts[item] = len(transitions)
                    except:
                        patient_sample_counts[item] = 0
    
    if not patient_dirs:
        raise ValueError(f"No valid patient directories found in {data_dir}")
    
    print(f"Found {len(patient_dirs)} patient directories: {patient_dirs}")
    
    if balance_by_samples:
        total_samples = sum(patient_sample_counts.values())
        print(f"Total samples across all patients: {total_samples}")
        print(f"Sample distribution per patient: {patient_sample_counts}")
    
    # Shuffle patients for randomness (unless explicitly disabled)
    if shuffle_patients:
        random.shuffle(patient_dirs)
        print(f"Shuffled patient order: {patient_dirs}")
    else:
        patient_dirs.sort()  # For consistent ordering
        print(f"Using sorted patient order: {patient_dirs}")
    
    # Split patients based on strategy
    if balance_by_samples and len(patient_dirs) >= 6:  # Only use smart splitting if we have enough patients
        # Smart splitting: try to balance sample counts across splits
        train_patients, val_patients, test_patients = _smart_patient_split(
            patient_dirs, patient_sample_counts
        )
    else:
        # Simple splitting: just divide patients by count
        train_patients, val_patients, test_patients = _simple_patient_split(patient_dirs)
    
    print(f"Patient splits:")
    print(f"  Train: {train_patients}")
    print(f"  Validation: {val_patients}")
    print(f"  Test: {test_patients}")
    
    # Print final sample distribution
    if balance_by_samples:
        train_samples = sum(patient_sample_counts.get(p, 0) for p in train_patients)
        val_samples = sum(patient_sample_counts.get(p, 0) for p in val_patients)
        test_samples = sum(patient_sample_counts.get(p, 0) for p in test_patients)
        total = train_samples + val_samples + test_samples
        
        print(f"Final sample distribution:")
        print(f"  Train: {train_samples} samples ({train_samples/total:.1%})")
        print(f"  Val: {val_samples} samples ({val_samples/total:.1%})")
        print(f"  Test: {test_samples} samples ({test_samples/total:.1%})")
    
    return train_patients, val_patients, test_patients


def _simple_patient_split(patient_dirs: List[str]) -> Tuple[List[str], List[str], List[str]]:
    """Simple patient splitting by count"""
    num_patients = len(patient_dirs)
    num_train = max(1, int(0.7 * num_patients))
    num_val = max(1, int(0.15 * num_patients))
    
    train_patients = patient_dirs[:num_train]
    val_patients = patient_dirs[num_train:num_train + num_val]
    test_patients = patient_dirs[num_train + num_val:]
    
    # Ensure we have at least one patient for each split
    if len(val_patients) == 0 and len(train_patients) > 1:
        val_patients = [train_patients.pop()]
    if len(test_patients) == 0 and len(train_patients) > 1:
        test_patients = [train_patients.pop()]
    
    return train_patients, val_patients, test_patients


def _smart_patient_split(patient_dirs: List[str], patient_sample_counts: Dict[str, int]) -> Tuple[List[str], List[str], List[str]]:
    """
    Smart patient splitting that tries to balance sample counts across splits
    Uses a greedy algorithm to approximate 70%/15%/15% sample distribution
    """
    total_samples = sum(patient_sample_counts.values())
    target_train = 0.7 * total_samples
    target_val = 0.15 * total_samples
    target_test = 0.15 * total_samples
    
    # Sort patients by sample count (descending) for better distribution
    sorted_patients = sorted(patient_dirs, key=lambda p: patient_sample_counts[p], reverse=True)
    
    train_patients = []
    val_patients = []
    test_patients = []
    
    train_samples = 0
    val_samples = 0
    test_samples = 0
    
    # Greedy assignment: assign each patient to the split that needs samples most
    for patient in sorted_patients:
        samples = patient_sample_counts[patient]
        
        # Calculate how much each split needs
        train_need = max(0, target_train - train_samples)
        val_need = max(0, target_val - val_samples)
        test_need = max(0, target_test - test_samples)
        
        # Assign to the split with highest need (but ensure minimum 1 patient per split)
        if train_need >= max(val_need, test_need) and (len(train_patients) == 0 or len(val_patients) > 0 and len(test_patients) > 0):
            train_patients.append(patient)
            train_samples += samples
        elif val_need >= test_need and (len(val_patients) == 0 or len(test_patients) > 0):
            val_patients.append(patient)
            val_samples += samples
        else:
            test_patients.append(patient)
            test_samples += samples
    
    # Ensure minimum one patient per split
    if len(val_patients) == 0 and len(train_patients) > 1:
        val_patients = [train_patients.pop()]
        val_samples += patient_sample_counts[val_patients[0]]
        train_samples -= patient_sample_counts[val_patients[0]]
    
    if len(test_patients) == 0 and len(train_patients) > 1:
        test_patients = [train_patients.pop()]
        test_samples += patient_sample_counts[test_patients[0]]
        train_samples -= patient_sample_counts[test_patients[0]]
    
    print(f"Smart splitting achieved:")
    print(f"  Target vs Actual: Train {target_train:.0f} vs {train_samples}, Val {target_val:.0f} vs {val_samples}, Test {target_test:.0f} vs {test_samples}")
    
    return train_patients, val_patients, test_patients




def load_config(config_path: str) -> Dict:
    """
    Load YAML configuration file
    
    Args:
        config_path: Configuration file path
    
    Returns:
        Configuration dictionary
    """
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            print(f"Loaded configuration from {config_path}")
        return config
    else:
        print(f"Configuration file {config_path} not found, using defaults")
        return {}


def get_model_config(config_override: Dict = None) -> Dict:
    """
    Get model configuration with production settings
    
    Returns:
        Model configuration dictionary
    """
    default_config = {
        "token_type": "channel",
        "d_model": 768,
        "num_heads": 12,
        "num_layers": 6,
        "feature_dim": 49,
        "in_channels": 1,
        "use_pretrained": True,
        "lr": 1e-4,
        "weight_decay": 1e-5,
        "lambda_t2_action": 1.0,
        "smooth_l1_beta": 1.0,
        "use_flash_attn": False,
        "primary_task_only": False
    }

    if config_override:
        model_override = config_override.get("model", {})
        default_config.update(model_override)
        
        # Ensure numeric parameters are properly converted to float
        numeric_params = ["lr", "weight_decay", "lambda_t2_action", "smooth_l1_beta"]
        for param in numeric_params:
            if param in default_config:
                try:
                    default_config[param] = float(default_config[param])
                except (ValueError, TypeError):
                    print(f"Warning: Could not convert {param} to float, using default value")
                    # Keep the default value if conversion fails

    return default_config


def get_train_config(config_override: Dict = None) -> Dict:
    """
    Get training configuration with production settings
    
    Returns:
        Training configuration dictionary
    """
    default_config = {
        "batch_size": 16,
        "num_workers": 4,
        "max_epochs": 150,
        "early_stop_patience": 20,
        "accelerator": "auto",
        "precision": 32,
        "gradient_clip_val": 1.0,
        "accumulate_grad_batches": 1,
        "check_val_every_n_epoch": 1,
        "log_every_n_steps": 20
    }
    
    if config_override:
        training_override = config_override.get("training", {})
        default_config.update(training_override)
        
        # Ensure numeric parameters are properly converted to appropriate types
        float_params = ["gradient_clip_val"]
        int_params = ["batch_size", "num_workers", "max_epochs", "early_stop_patience", 
                     "precision", "accumulate_grad_batches", "check_val_every_n_epoch", "log_every_n_steps"]
        
        for param in float_params:
            if param in default_config:
                try:
                    default_config[param] = float(default_config[param])
                except (ValueError, TypeError):
                    print(f"Warning: Could not convert {param} to float, using default value")
        
        for param in int_params:
            if param in default_config:
                try:
                    default_config[param] = int(default_config[param])
                except (ValueError, TypeError):
                    print(f"Warning: Could not convert {param} to int, using default value")
    
    return default_config


class TrainingVisualizer:
    """Handle training visualization and logging"""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.plots_dir = os.path.join(output_dir, "plots")
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def plot_dataset_statistics(self, train_dataset, val_dataset, test_dataset):
        """Plot dataset statistics"""
        print("Creating dataset visualization...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Dataset sizes
        sizes = [len(train_dataset), len(val_dataset), len(test_dataset)]
        labels = ['Train', 'Validation', 'Test']
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        axes[0, 0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        axes[0, 0].set_title('Dataset Distribution')
        
        # Patient distribution
        train_stats = train_dataset.get_patient_stats()
        val_stats = val_dataset.get_patient_stats()
        test_stats = test_dataset.get_patient_stats()
        
        # Bar plot for patient statistics
        patients = list(set(list(train_stats.keys()) + list(val_stats.keys()) + list(test_stats.keys())))
        train_counts = [train_stats.get(p, 0) for p in patients]
        val_counts = [val_stats.get(p, 0) for p in patients]
        test_counts = [test_stats.get(p, 0) for p in patients]
        
        x = np.arange(len(patients))
        width = 0.25
        
        axes[0, 1].bar(x - width, train_counts, width, label='Train', color=colors[0])
        axes[0, 1].bar(x, val_counts, width, label='Val', color=colors[1])
        axes[0, 1].bar(x + width, test_counts, width, label='Test', color=colors[2])
        axes[0, 1].set_xlabel('Patients')
        axes[0, 1].set_ylabel('Number of Samples')
        axes[0, 1].set_title('Samples per Patient')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(patients, rotation=45)
        axes[0, 1].legend()
        
        # Sample a few images for visualization
        sample_images = []
        sample_labels = []
        for i in range(min(4, len(train_dataset))):
            img, _, _, _ = train_dataset[i]
            sample_images.append(img.squeeze().numpy())
            sample_labels.append(f"Sample {i+1}")
        
        for i, (img, label) in enumerate(zip(sample_images, sample_labels)):
            row, col = (i // 2) + 1, i % 2
            if row < 2:
                axes[row, col].imshow(img, cmap='gray')
                axes[row, col].set_title(label)
                axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'dataset_statistics.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save statistics to file
        stats_file = os.path.join(self.output_dir, 'dataset_stats.json')
        stats = {
            'total_samples': {
                'train': len(train_dataset),
                'val': len(val_dataset),
                'test': len(test_dataset)
            },
            'patient_distribution': {
                'train': train_stats,
                'val': val_stats,
                'test': test_stats
            }
        }
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"Dataset statistics saved to {self.plots_dir}")
    
    def create_training_summary(self, trainer, model):
        """Create training summary plots"""
        print("Creating training summary...")
        
        # Get training logs
        csv_logger = None
        for logger in trainer.loggers:
            if isinstance(logger, CSVLogger):
                csv_logger = logger
                break
        
        if csv_logger is None:
            print("No CSV logger found, skipping training plots")
            return
        
        # Read logs
        log_file = os.path.join(csv_logger.log_dir, "metrics.csv")
        if not os.path.exists(log_file):
            print("No metrics file found, skipping training plots")
            return
        
        df = pd.read_csv(log_file)
        
        # Print available columns for debugging
        print(f"Available columns in CSV: {list(df.columns)}")
        
        # Create simplified training plots focusing on essential losses (epoch-based)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Training and validation total loss (epoch-based)
        train_loss = df[df['train_total_loss_epoch'].notna()]
        val_loss = df[df['val_total_loss'].notna()]
        
        if not train_loss.empty and not val_loss.empty:
            axes[0, 0].plot(train_loss['epoch'], train_loss['train_total_loss_epoch'], label='Train', color='#FF6B6B', linewidth=2)
            axes[0, 0].plot(val_loss['epoch'], val_loss['val_total_loss'], label='Validation', color='#4ECDC4', linewidth=2)
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Total Loss')
            axes[0, 0].set_title('Training & Validation Total Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # Main task loss (most important metric) (epoch-based)
        train_main = df[df['train_main_task_loss_epoch'].notna()]
        val_main = df[df['val_main_task_loss'].notna()]
        
        if not train_main.empty and not val_main.empty:
            axes[0, 1].plot(train_main['epoch'], train_main['train_main_task_loss_epoch'], label='Train', color='#FF6B6B', linewidth=2)
            axes[0, 1].plot(val_main['epoch'], val_main['val_main_task_loss'], label='Validation', color='#4ECDC4', linewidth=2)
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Main Task Loss')
            axes[0, 1].set_title('Main Task Loss (AT1 Prediction) - Key Metric')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Learning rate (epoch-based)
        lr_data = df[df['lr-AdamW'].notna()]
        if not lr_data.empty:
            axes[0, 2].plot(lr_data['epoch'], lr_data['lr-AdamW'], color='#45B7D1', linewidth=2)
            axes[0, 2].set_xlabel('Epoch')
            axes[0, 2].set_ylabel('Learning Rate')
            axes[0, 2].set_title('Learning Rate Schedule')
            axes[0, 2].grid(True, alpha=0.3)
            axes[0, 2].set_yscale('log')  # Use log scale for learning rate
        
        # Auxiliary t2 action loss (epoch-based)
        try:
            aux_t2_loss = df[df['train_aux_t2_loss'].notna()]
            val_aux_t2_loss = df[df['val_aux_t2_loss'].notna()]
            
            if not aux_t2_loss.empty:
                axes[1, 0].plot(aux_t2_loss['epoch'], aux_t2_loss['train_aux_t2_loss'], label='Train', color='#FECA57', linewidth=2)
                if not val_aux_t2_loss.empty:
                    axes[1, 0].plot(val_aux_t2_loss['epoch'], val_aux_t2_loss['val_aux_t2_loss'], label='Validation', color='#FFD93D', linewidth=2)
                axes[1, 0].set_xlabel('Epoch')
                axes[1, 0].set_ylabel('Auxiliary T2 Loss')
                axes[1, 0].set_title('T2 Action Prediction Loss (Auxiliary)')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
            else:
                # 如果沒有輔助損失數據，顯示提示信息
                axes[1, 0].text(0.5, 0.5, 'No auxiliary T2 loss data available', 
                               ha='center', va='center', transform=axes[1, 0].transAxes)
                axes[1, 0].set_title('T2 Action Prediction Loss (Auxiliary)')
                
        except KeyError as e:
            print(f"⚠️ Warning: Column {e} not found in training logs")
            axes[1, 0].text(0.5, 0.5, f'Auxiliary loss data not available\n(Column {e} missing)', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('T2 Action Prediction Loss (Auxiliary)')
        
        # Final metrics summary
        if not val_loss.empty:
            final_metrics = val_loss.iloc[-1]
            
            # Format metrics safely
            epoch_num = int(final_metrics.get('epoch', 0))
            
            total_loss_val = final_metrics.get('val_total_loss', 'N/A')
            if isinstance(total_loss_val, (int, float)):
                total_loss_str = f"{total_loss_val:.4f}"
            else:
                total_loss_str = str(total_loss_val)
            
            main_task_loss_val = final_metrics.get('val_main_task_loss', 'N/A')
            if isinstance(main_task_loss_val, (int, float)):
                main_task_loss_str = f"{main_task_loss_val:.4f}"
            else:
                main_task_loss_str = str(main_task_loss_val)
            
            metrics_text = f"""Final Metrics (Epoch {epoch_num}):

Total Loss: {total_loss_str}

Main Task Loss: {main_task_loss_str}
(This is the key metric for AT1 prediction)

Training completed successfully.
Focus on Main Task Loss for performance.

📊 Final validation plots generated at training end"""
            
            axes[1, 1].text(0.1, 0.5, metrics_text, fontsize=11, 
                           verticalalignment='center', fontfamily='monospace')
            axes[1, 1].set_title('Final Training Metrics')
            axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'training_summary.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training summary saved to {self.plots_dir}")
    
    def create_final_validation_summary(self, output_dir: str):
        """Create a summary of the final validation scatter plots"""
        print("Creating final validation scatter plots summary...")
        
        # Find the final validation plots directory
        final_plots_dir = os.path.join(output_dir, "final_validation_plots")
        
        if not os.path.exists(final_plots_dir):
            print("No final validation plots found, skipping summary")
            return
        
        # Find the combined plot
        combined_plot = os.path.join(final_plots_dir, "final_validation_scatter_combined.png")
        if not os.path.exists(combined_plot):
            print("No combined final validation plot found")
            return
        
        # Copy the plot to the main plots directory for easy access
        import shutil
        summary_plot_path = os.path.join(self.plots_dir, 'final_validation_scatter_plots.png')
        shutil.copy2(combined_plot, summary_plot_path)
        
        print(f"📊 Final validation scatter plots copied to: {summary_plot_path}")
        print(f"📁 All final validation plots available in: {final_plots_dir}")
        
        # Create a summary text file with information about the plots
        summary_info = {
            "final_validation_plots": {
                "directory": final_plots_dir,
                "combined_plot": combined_plot,
                "description": "Final validation scatter plots showing predicted vs ground truth for each 6DOF dimension",
                "dimensions": ["X", "Y", "Z", "Roll", "Pitch", "Yaw"],
                "plot_types": [
                    "Individual plots: final_validation_scatter_[dimension].png",
                    "Combined plot: final_validation_scatter_combined.png"
                ],
                "optimization": "Generated only once at training end (not every epoch) to prevent memory issues"
            }
        }
        
        summary_file = os.path.join(output_dir, 'final_validation_plots_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary_info, f, indent=2)
        
        print(f"📄 Final validation plots summary saved to: {summary_file}")


def setup_callbacks(output_dir: str, train_config: Dict) -> List:
    """Set up training callbacks"""
    callbacks = []
    
    # Model checkpoint - save best models
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(output_dir, "checkpoints"),
        filename="cardiac_dreamer-{epoch:02d}-{val_main_task_loss:.4f}",
        save_top_k=3,
        monitor="val_main_task_loss",  # Monitor main task loss
        mode="min",
        save_last=True,
        save_weights_only=False,
        verbose=True
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping
    early_stop_callback = EarlyStopping(
        monitor="val_main_task_loss",
        patience=train_config["early_stop_patience"],
        mode="min",
        verbose=True,
        strict=False
    )
    callbacks.append(early_stop_callback)
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(
        logging_interval="epoch",
        log_momentum=True
    )
    callbacks.append(lr_monitor)
    
    # Device stats monitor
    device_stats = DeviceStatsMonitor()
    callbacks.append(device_stats)
    
    return callbacks


def setup_loggers(output_dir: str, config_override: Dict = None) -> List:
    """Set up training loggers"""
    loggers = []
    
    # TensorBoard logger
    tb_logger = TensorBoardLogger(
        save_dir=os.path.join(output_dir, "logs"),
        name="cardiac_dreamer",
        default_hp_metric=False
    )
    loggers.append(tb_logger)
    
    # CSV logger for easy analysis
    csv_logger = CSVLogger(
        save_dir=os.path.join(output_dir, "logs"),
        name="cardiac_dreamer_csv"
    )
    loggers.append(csv_logger)
    
    # WandB logger (if enabled)
    if config_override and config_override.get("experiment", {}).get("use_wandb", False):
        try:
            from pytorch_lightning.loggers import WandbLogger
            
            experiment_config = config_override.get("experiment", {})
            
            # 簡化配置，避免權限問題
            wandb_config = {
                "project": experiment_config.get("wandb_project", "cardiac_dreamer"),
                "tags": experiment_config.get("tags", []),
                "notes": experiment_config.get("description", ""),
                "name": f"cardiac_dreamer_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            }
            
            # 添加 entity（如果有指定）
            entity = experiment_config.get("wandb_entity")
            if entity:
                wandb_config["entity"] = entity
                
            wandb_logger = WandbLogger(**wandb_config)
            loggers.append(wandb_logger)
            print(f"✅ WandB logger enabled - Project: {experiment_config.get('wandb_project')}")
            
        except ImportError:
            print("⚠️ WandB not available, skipping WandB logger")
        except Exception as e:
            print(f"⚠️ Failed to setup WandB logger: {e}")
            print("   Continuing without WandB...")
    
    return loggers


def save_experiment_config(output_dir: str, args, model_config: Dict, train_config: Dict):
    """Save experiment configuration"""
    config = {
        "experiment_info": {
            "timestamp": datetime.now().isoformat(),
            "data_dir": args.data_dir,
            "output_dir": args.output_dir,
            "manual_splits": args.manual_splits,
            "train_patients": args.train_patients,
            "val_patients": args.val_patients,
            "test_patients": args.test_patients
        },
        "model_config": model_config,
        "training_config": train_config
    }
    
    config_file = os.path.join(output_dir, "experiment_config.json")
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Experiment configuration saved to {config_file}")


def main(args):
    # Set up PyTorch
    pl.seed_everything(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Load configurations
    config_override = load_config(args.config) if args.config else {}
    model_config = get_model_config(config_override)
    train_config = get_train_config(config_override)
    
    # Create output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(run_output_dir, exist_ok=True)
    
    # Save experiment configuration
    save_experiment_config(run_output_dir, args, model_config, train_config)
    
    # Set up data transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Automatically detect patient splits or use manual splits
    data_dir = args.data_dir
    if args.manual_splits:
        # Use manually specified patient splits
        train_patients = args.train_patients.split(',') if args.train_patients else None
        val_patients = args.val_patients.split(',') if args.val_patients else None  
        test_patients = args.test_patients.split(',') if args.test_patients else None
    else:
        # Automatically detect and split patients
        train_patients, val_patients, test_patients = get_patient_splits(data_dir)
    
    # Create datasets using the new cross-patient dataset
    print("Creating datasets...")
    train_dataset = CrossPatientTransitionsDataset(
        data_dir=data_dir,
        transform=transform,
        split="train",
        train_patients=train_patients,
        val_patients=val_patients,
        test_patients=test_patients,
        small_subset=False,  # Use full dataset for production
        normalize_actions=True
    )
    
    val_dataset = CrossPatientTransitionsDataset(
        data_dir=data_dir,
        transform=transform,
        split="val",
        train_patients=train_patients,
        val_patients=val_patients,
        test_patients=test_patients,
        small_subset=False,  # Use full dataset for production
        normalize_actions=True
    )
    
    test_dataset = CrossPatientTransitionsDataset(
        data_dir=data_dir,
        transform=transform,
        split="test",
        train_patients=train_patients,
        val_patients=val_patients,
        test_patients=test_patients,
        small_subset=False,  # Use full dataset for production
        normalize_actions=True
    )
    
    print(f"\nDataset Statistics:")
    print(f"Training set: {len(train_dataset)} samples")
    print(f"Validation set: {len(val_dataset)} samples")
    print(f"Test set: {len(test_dataset)} samples")
    
    # Print patient distribution
    print(f"\nPatient distribution:")
    print(f"Train patients: {train_dataset.get_patient_stats()}")
    print(f"Val patients: {val_dataset.get_patient_stats()}")
    print(f"Test patients: {test_dataset.get_patient_stats()}")
    
    # Create visualization
    visualizer = TrainingVisualizer(run_output_dir)
    visualizer.plot_dataset_statistics(train_dataset, val_dataset, test_dataset)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config["batch_size"],
        shuffle=True,
        num_workers=min(train_config["num_workers"], 2),  # 🔧 limit worker number to prevent memory leaks
        pin_memory=True if torch.cuda.is_available() else False,  # 🔧 use pin_memory only when GPU is available
        persistent_workers=False,  # 🔧 disable persistent_workers to prevent memory accumulation
        drop_last=True  # 🔧 drop last incomplete batch
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config["batch_size"],
        shuffle=False,
        num_workers=min(train_config["num_workers"], 2),  # 🔧 limit worker number to prevent memory leaks
        pin_memory=True if torch.cuda.is_available() else False,  # 🔧 use pin_memory only when GPU is available
        persistent_workers=False,  # 🔧 disable persistent_workers to prevent memory accumulation
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=train_config["batch_size"],
        shuffle=False,
        num_workers=min(train_config["num_workers"], 2),  # 🔧 limit worker number to prevent memory leaks
        pin_memory=True if torch.cuda.is_available() else False,  # 🔧 use pin_memory only when GPU is available
        persistent_workers=False,  # 🔧 disable persistent_workers to prevent memory accumulation
        drop_last=False
    )
    
    # Test a single batch to verify data loading
    print(f"\nTesting data loading...")
    sample_batch = next(iter(train_loader))
    print(f"Batch structure:")
    print(f"  Image: {sample_batch[0].shape}")
    print(f"  Action change (a_hat_t1_to_t2_gt): {sample_batch[1].shape}")
    print(f"  At1 6DOF (at1_6dof_gt): {sample_batch[2].shape}")
    print(f"  At2 6DOF (at2_6dof_gt): {sample_batch[3].shape}")
    
    # Create model
    print("Creating model...")
    model = get_cardiac_dreamer_system(
        token_type=model_config["token_type"],
        d_model=model_config["d_model"],
        nhead=model_config["num_heads"],
        num_layers=model_config["num_layers"],
        feature_dim=model_config["feature_dim"],
        lr=model_config["lr"],
        weight_decay=model_config["weight_decay"],
        lambda_t2_action=model_config["lambda_t2_action"],
        smooth_l1_beta=model_config["smooth_l1_beta"],
        use_flash_attn=model_config["use_flash_attn"],
        primary_task_only=model_config["primary_task_only"]
    )
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model created with {total_params:,} total parameters ({trainable_params:,} trainable)")
    
    # Set up callbacks and loggers
    callbacks = setup_callbacks(run_output_dir, train_config)
    loggers = setup_loggers(run_output_dir, config_override)
    
    # Log experiment config to WandB if enabled
    if config_override and config_override.get("experiment", {}).get("use_wandb", False):
        for logger in loggers:
            if hasattr(logger, 'experiment') and hasattr(logger.experiment, 'config'):
                # Log all configurations to WandB
                logger.experiment.config.update({
                    "model_config": model_config,
                    "training_config": train_config,
                    "dataset_stats": {
                        "train_samples": len(train_dataset),
                        "val_samples": len(val_dataset), 
                        "test_samples": len(test_dataset),
                        "train_patients": list(train_dataset.get_patient_stats().keys()),
                        "val_patients": list(val_dataset.get_patient_stats().keys()),
                        "test_patients": list(test_dataset.get_patient_stats().keys())
                    }
                })
                print("📊 Experiment config logged to WandB")
                break
    
    # Set up trainer
    trainer = pl.Trainer(
        max_epochs=train_config["max_epochs"],
        callbacks=callbacks,
        logger=loggers,
        accelerator=train_config["accelerator"],
        precision=train_config["precision"],
        gradient_clip_val=train_config["gradient_clip_val"],
        accumulate_grad_batches=train_config["accumulate_grad_batches"],
        check_val_every_n_epoch=train_config["check_val_every_n_epoch"],
        log_every_n_steps=train_config["log_every_n_steps"],
        enable_progress_bar=True,
        enable_model_summary=True,
        deterministic=True
    )
    
    # Start training
    print(f"\nStarting training...")
    print(f"Output directory: {run_output_dir}")
    print(f"Logs will be saved to: {os.path.join(run_output_dir, 'logs')}")
    print(f"Checkpoints will be saved to: {os.path.join(run_output_dir, 'checkpoints')}")
    
    try:
        trainer.fit(model, train_loader, val_loader)
            
        # Create training summary
        visualizer.create_training_summary(trainer, model)
        
        # 🔧 optimize: generate final validation scatter plots after training (not every epoch)
        print("📊 generating final validation scatter plots...")
        model.generate_final_validation_plots(output_dir=run_output_dir)
        
        # Create final validation scatter plots summary (update to use final plots)
        visualizer.create_final_validation_summary(run_output_dir)
    
        # Test model
        print("Testing model...")
        trainer.test(model, test_loader)
    
            # Save final model state
        final_model_path = os.path.join(run_output_dir, "final_model.ckpt")
        trainer.save_checkpoint(final_model_path)
        
        print(f"\n🎉 Training completed successfully!")
        print(f"📊 Results saved to: {run_output_dir}")
        print(f"📈 View training logs: tensorboard --logdir {os.path.join(run_output_dir, 'logs')}")
        print(f"💾 Best model checkpoint: {callbacks[0].best_model_path}")
        print(f"📊 Final validation plots: {os.path.join(run_output_dir, 'final_validation_plots')}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        raise


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train cardiac ultrasound guidance model")
    parser.add_argument("--data_dir", type=str, default="data/processed", help="Data directory path")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory path")
    parser.add_argument("--config", type=str, help="Configuration file path (optional)")
    
    # Patient split arguments
    parser.add_argument("--manual_splits", action="store_true", help="Use manual patient splits instead of automatic")
    parser.add_argument("--train_patients", type=str, help="Comma-separated list of training patient IDs")
    parser.add_argument("--val_patients", type=str, help="Comma-separated list of validation patient IDs")
    parser.add_argument("--test_patients", type=str, help="Comma-separated list of test patient IDs")
    
    # Resume training
    parser.add_argument("--resume_from_checkpoint", type=str, help="Path to checkpoint to resume training from")
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    
    main(args) 
