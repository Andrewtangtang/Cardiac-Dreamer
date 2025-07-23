"""
Data loader utilities for cardiac dreamer training
"""

import torch
from torch.utils.data import DataLoader
from typing import Dict, Tuple


def create_data_loaders(
    train_dataset, 
    val_dataset, 
    test_dataset, 
    train_config: Dict
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create optimized data loaders for training, validation, and testing
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset  
        test_dataset: Test dataset (can be None)
        train_config: Training configuration dictionary
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
        Note: test_loader will be None if test_dataset is None
    """
    
    # Create training data loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config["batch_size"],
        shuffle=True,
        num_workers=min(train_config["num_workers"], 2),  # 🔧 limit worker number to prevent memory leaks
        pin_memory=True if torch.cuda.is_available() else False,  # 🔧 use pin_memory only when GPU is available
        persistent_workers=False,  # 🔧 disable persistent_workers to prevent memory accumulation
        drop_last=True  # 🔧 drop last incomplete batch
    )
    
    # Create validation data loader
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config["batch_size"],
        shuffle=False,
        num_workers=min(train_config["num_workers"], 2),  # 🔧 limit worker number to prevent memory leaks
        pin_memory=True if torch.cuda.is_available() else False,  # 🔧 use pin_memory only when GPU is available
        persistent_workers=False,  # 🔧 disable persistent_workers to prevent memory accumulation
        drop_last=False
    )
    
    # Create test data loader only if test dataset exists
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=train_config["batch_size"],
            shuffle=False,
            num_workers=min(train_config["num_workers"], 2),  # 🔧 limit worker number to prevent memory leaks
            pin_memory=True if torch.cuda.is_available() else False,  # 🔧 use pin_memory only when GPU is available
            persistent_workers=False,  # 🔧 disable persistent_workers to prevent memory accumulation
            drop_last=False
        )
    else:
        test_loader = None
    
    return train_loader, val_loader, test_loader


def test_data_loading(train_loader):
    """Test data loading with a single batch"""
    print(f"\nTesting data loading...")
    sample_batch = next(iter(train_loader))
    print(f"Batch structure:")
    print(f"  Image: {sample_batch[0].shape}")
    print(f"  Action change (a_hat_t1_to_t2_gt): {sample_batch[1].shape}")
    print(f"  At1 6DOF (at1_6dof_gt): {sample_batch[2].shape}")
    print(f"  At2 6DOF (at2_6dof_gt): {sample_batch[3].shape}")
    return sample_batch 