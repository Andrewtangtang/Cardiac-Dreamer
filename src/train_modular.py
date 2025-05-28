#!/usr/bin/env python
"""
Modular training script for Cardiac Dreamer

This is a refactored version of the original train.py with improved modularity.
All major components have been separated into dedicated modules for better
maintainability and reusability.
"""

import os
import sys
import argparse
import torch
import pytorch_lightning as pl
import torchvision.transforms as transforms
from datetime import datetime

# Add project root to path
current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_file_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import modular components
from src.models.system import get_cardiac_dreamer_system
from src.data import CrossPatientTransitionsDataset, get_patient_splits
from src.config import load_config, get_model_config, get_train_config, save_experiment_config
from src.training import setup_callbacks, setup_loggers, create_data_loaders
from src.training.data_loaders import test_data_loading
from src.visualization import TrainingVisualizer


def create_datasets(args, transform, train_config):
    """
    Create train, validation, and test datasets
    
    Args:
        args: Command line arguments
        transform: Image transformations
        train_config: Training configuration
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    data_dir = args.data_dir
    
    # Determine patient splits
    if args.manual_splits:
        # Use manually specified patient splits
        train_patients = args.train_patients.split(',') if args.train_patients else None
        val_patients = args.val_patients.split(',') if args.val_patients else None  
        test_patients = args.test_patients.split(',') if args.test_patients else None
    else:
        # Automatically detect and split patients
        train_patients, val_patients, test_patients = get_patient_splits(data_dir)
    
    # Create datasets using the cross-patient dataset
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
    
    return train_dataset, val_dataset, test_dataset


def print_dataset_info(train_dataset, val_dataset, test_dataset):
    """Print dataset statistics and patient distribution"""
    print(f"\nDataset Statistics:")
    print(f"Training set: {len(train_dataset)} samples")
    print(f"Validation set: {len(val_dataset)} samples")
    print(f"Test set: {len(test_dataset)} samples")
    
    # Print patient distribution
    print(f"\nPatient distribution:")
    print(f"Train patients: {train_dataset.get_patient_stats()}")
    print(f"Val patients: {val_dataset.get_patient_stats()}")
    print(f"Test patients: {test_dataset.get_patient_stats()}")


def create_model(model_config):
    """
    Create the cardiac dreamer model
    
    Args:
        model_config: Model configuration dictionary
        
    Returns:
        Configured model instance
    """
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
    
    return model


def setup_trainer(run_output_dir, train_config, config_override):
    """
    Set up the PyTorch Lightning trainer
    
    Args:
        run_output_dir: Output directory for this run
        train_config: Training configuration
        config_override: Configuration overrides
        
    Returns:
        Configured trainer and its components
    """
    # Set up callbacks and loggers
    callbacks = setup_callbacks(run_output_dir, train_config)
    loggers = setup_loggers(run_output_dir, config_override)
    
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
    
    return trainer, callbacks, loggers


def log_wandb_config(loggers, model_config, train_config, train_dataset, val_dataset, test_dataset, config_override):
    """Log experiment configuration to WandB if enabled"""
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
                print(" Experiment config logged to WandB")
                break


def run_training(trainer, model, train_loader, val_loader, test_loader, run_output_dir, visualizer, callbacks):
    """
    Execute the training process
    
    Args:
        trainer: PyTorch Lightning trainer
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader  
        test_loader: Test data loader
        run_output_dir: Output directory
        visualizer: Training visualizer
        callbacks: Training callbacks
    """
    print(f"\nStarting training...")
    print(f"Output directory: {run_output_dir}")
    print(f"Logs will be saved to: {os.path.join(run_output_dir, 'logs')}")
    print(f"Checkpoints will be saved to: {os.path.join(run_output_dir, 'checkpoints')}")
    
    try:
        # Start training
        trainer.fit(model, train_loader, val_loader)
            
        # Create training summary
        visualizer.create_training_summary(trainer, model)
        
        # 🔧 optimize: generate final validation scatter plots after training (not every epoch)
        print(" generating final validation scatter plots...")
        model.generate_final_validation_plots(output_dir=run_output_dir)
        
        # Create final validation scatter plots summary (update to use final plots)
        visualizer.create_final_validation_summary(run_output_dir)
    
        # Test model
        print("Testing model...")
        trainer.test(model, test_loader)
    
        # Save final model state
        final_model_path = os.path.join(run_output_dir, "final_model.ckpt")
        trainer.save_checkpoint(final_model_path)
        
        print(f"\n Training completed successfully!")
        print(f" Results saved to: {run_output_dir}")
        print(f" View training logs: tensorboard --logdir {os.path.join(run_output_dir, 'logs')}")
        print(f" Best model checkpoint: {callbacks[0].best_model_path}")
        print(f" Final validation plots: {os.path.join(run_output_dir, 'final_validation_plots')}")
        
    except Exception as e:
        print(f" Training failed: {e}")
        raise


def main(args):
    """Main training function"""
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
    
    # Create datasets
    train_dataset, val_dataset, test_dataset = create_datasets(args, transform, train_config)
    print_dataset_info(train_dataset, val_dataset, test_dataset)
    
    # Create visualization
    visualizer = TrainingVisualizer(run_output_dir)
    visualizer.plot_dataset_statistics(train_dataset, val_dataset, test_dataset)
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset, val_dataset, test_dataset, train_config
    )
    
    # Test data loading
    test_data_loading(train_loader)
    
    # Create model
    model = create_model(model_config)
    
    # Set up trainer
    trainer, callbacks, loggers = setup_trainer(run_output_dir, train_config, config_override)
    
    # Log experiment config to WandB if enabled
    log_wandb_config(loggers, model_config, train_config, train_dataset, val_dataset, test_dataset, config_override)
    
    # Run training
    run_training(trainer, model, train_loader, val_loader, test_loader, run_output_dir, visualizer, callbacks)


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train cardiac ultrasound guidance model (Modular Version)")
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