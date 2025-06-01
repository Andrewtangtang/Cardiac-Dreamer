#!/usr/bin/env python
"""
Modular training script for Cardiac Dreamer

This is a refactored version of the original train.py with improved modularity.
All major components have been separated into dedicated modules for better
maintainability and reusability.

Enhanced with:
- Data augmentation support
- Advanced regularization strategies  
- Learning rate scheduling
- Model EMA
- MixUp augmentation
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
from src.data import (
    CrossPatientTransitionsDataset, 
    get_patient_splits,
    create_augmented_transform,
    create_mixup_augmentation
)
from src.config import load_config, get_model_config, get_train_config, save_experiment_config
from src.training import (
    setup_callbacks, 
    setup_loggers, 
    create_data_loaders,
    create_training_components
)
from src.training.data_loaders import test_data_loading
from src.visualization import TrainingVisualizer
from src.models.regularization import ModelEMA, create_regularized_loss


def create_enhanced_datasets(args, config, train_config):
    """
    Create enhanced train, validation, and test datasets with augmentation support
    
    Args:
        args: Command line arguments
        config: Full configuration dictionary
        train_config: Training configuration
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset, transform)
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
    
    # Create enhanced transforms with augmentation support
    train_transform = create_augmented_transform(config)
    
    # Validation and test use basic transforms (no augmentation)
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    print("Creating enhanced datasets with augmentation support...")
    
    # Create datasets using the cross-patient dataset
    train_dataset = CrossPatientTransitionsDataset(
        data_dir=data_dir,
        transform=train_transform,  # Enhanced transform with augmentation
        split="train",
        train_patients=train_patients,
        val_patients=val_patients,
        test_patients=test_patients,
        small_subset=False,
        normalize_actions=True
    )
    
    val_dataset = CrossPatientTransitionsDataset(
        data_dir=data_dir,
        transform=val_test_transform,  # Basic transform
        split="val",
        train_patients=train_patients,
        val_patients=val_patients,
        test_patients=test_patients,
        small_subset=False,
        normalize_actions=True
    )
    
    test_dataset = CrossPatientTransitionsDataset(
        data_dir=data_dir,
        transform=val_test_transform,  # Basic transform
        split="test",
        train_patients=train_patients,
        val_patients=val_patients,
        test_patients=test_patients,
        small_subset=False,
        normalize_actions=True
    )
    
    return train_dataset, val_dataset, test_dataset, train_transform


def print_enhanced_dataset_info(train_dataset, val_dataset, test_dataset, config):
    """Print enhanced dataset statistics and configuration info"""
    print(f"\nEnhanced Dataset Statistics:")
    print(f"Training set: {len(train_dataset)} samples")
    print(f"Validation set: {len(val_dataset)} samples")
    print(f"Test set: {len(test_dataset)} samples")
    
    # Print patient distribution
    print(f"\nPatient distribution:")
    print(f"Train patients: {train_dataset.get_patient_stats()}")
    print(f"Val patients: {val_dataset.get_patient_stats()}")
    print(f"Test patients: {test_dataset.get_patient_stats()}")
    
    # Print augmentation info
    augmentation_config = config.get('data', {}).get('augmentation', {})
    if augmentation_config.get('enabled', False):
        print(f"\n🎨 Data Augmentation Enabled:")
        print(f"  Rotation range: ±{augmentation_config.get('rotation_range', 0)}°")
        print(f"  Brightness range: ±{augmentation_config.get('brightness_range', 0)}")
        print(f"  Contrast range: ±{augmentation_config.get('contrast_range', 0)}")
        print(f"  Noise std: {augmentation_config.get('noise_std', 0)}")
    else:
        print(f"\n🎨 Data Augmentation: Disabled")


def create_enhanced_model(model_config, config):
    """
    Create enhanced cardiac dreamer model with regularization
    
    Args:
        model_config: Model configuration dictionary
        config: Full configuration dictionary
        
    Returns:
        Enhanced model instance
    """
    print("Creating enhanced model with regularization...")
    
    # Create base model
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
    
    # Print regularization info
    regularization_config = config.get('regularization', {})
    if regularization_config:
        print(f"\n🛡️ Regularization Strategy:")
        if regularization_config.get('dropout_rate', 0) > 0:
            print(f"  Dropout rate: {regularization_config['dropout_rate']}")
        if regularization_config.get('use_ema', False):
            print(f"  EMA enabled with decay: {regularization_config.get('ema_decay', 0.999)}")
        if regularization_config.get('lr_scheduler', {}):
            scheduler_config = regularization_config['lr_scheduler']
            print(f"  LR Scheduler: {scheduler_config.get('type', 'none')}")
    
    return model


def setup_enhanced_trainer(run_output_dir, train_config, config):
    """
    Set up enhanced PyTorch Lightning trainer with advanced features
    
    Args:
        run_output_dir: Output directory for this run
        train_config: Training configuration
        config: Full configuration dictionary
        
    Returns:
        Enhanced trainer and its components
    """
    # Set up callbacks and loggers
    callbacks = setup_callbacks(run_output_dir, train_config)
    loggers = setup_loggers(run_output_dir, config)
    
    # Enhanced trainer setup
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


def log_enhanced_wandb_config(loggers, model_config, train_config, train_dataset, val_dataset, test_dataset, config):
    """Log enhanced experiment configuration to WandB if enabled"""
    if config and config.get("experiment", {}).get("use_wandb", False):
        for logger in loggers:
            if hasattr(logger, 'experiment') and hasattr(logger.experiment, 'config'):
                # Log comprehensive configuration to WandB
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
                    },
                    "augmentation_config": config.get('data', {}).get('augmentation', {}),
                    "regularization_config": config.get('regularization', {}),
                    "advanced_config": config.get('advanced', {}),
                })
                print("✅ Enhanced experiment config logged to WandB")
                break


def run_enhanced_training(trainer, model, train_loader, val_loader, test_loader, run_output_dir, 
                         visualizer, callbacks, config):
    """
    Execute enhanced training process with advanced features
    
    Args:
        trainer: PyTorch Lightning trainer
        model: Enhanced model to train
        train_loader: Training data loader
        val_loader: Validation data loader  
        test_loader: Test data loader
        run_output_dir: Output directory
        visualizer: Training visualizer
        callbacks: Training callbacks
        config: Full configuration dictionary
    """
    print(f"\n🚀 Starting enhanced training...")
    print(f"📁 Output directory: {run_output_dir}")
    print(f"📊 Logs: {os.path.join(run_output_dir, 'logs')}")
    print(f"💾 Checkpoints: {os.path.join(run_output_dir, 'checkpoints')}")
    
    # Setup EMA if enabled
    ema_model = None
    regularization_config = config.get('regularization', {})
    if regularization_config.get('use_ema', False):
        ema_decay = regularization_config.get('ema_decay', 0.999)
        ema_model = ModelEMA(model, decay=ema_decay)
        print(f"🔄 EMA enabled with decay: {ema_decay}")
    
    # Setup MixUp if enabled
    mixup = create_mixup_augmentation(config)
    if mixup:
        print(f"🌀 MixUp enabled with alpha: {config.get('advanced', {}).get('mixup_alpha', 0.2)}")
    
    try:
        # Start enhanced training
        trainer.fit(model, train_loader, val_loader)
            
        # Apply EMA if used
        if ema_model:
            print("🔄 Applying EMA weights for final evaluation...")
            ema_model.apply_shadow(model)
            
        # Create training summary
        visualizer.create_training_summary(trainer, model)
        
        # Generate final validation scatter plots
        print("📊 Generating final validation scatter plots...")
        model.generate_final_validation_plots(output_dir=run_output_dir)
        
        # Create final validation summary
        visualizer.create_final_validation_summary(run_output_dir)
    
        # Test model
        print("🧪 Testing enhanced model...")
        trainer.test(model, test_loader)
    
        # Save final model state
        final_model_path = os.path.join(run_output_dir, "final_model.ckpt")
        trainer.save_checkpoint(final_model_path)
        
        # Save EMA model if used
        if ema_model:
            ema_model_path = os.path.join(run_output_dir, "final_ema_model.ckpt")
            torch.save(ema_model.state_dict(), ema_model_path)
            print(f"💾 EMA model saved: {ema_model_path}")
        
        print(f"\n🎉 Enhanced training completed successfully!")
        print(f"📁 Results saved to: {run_output_dir}")
        print(f"📈 View training logs: tensorboard --logdir {os.path.join(run_output_dir, 'logs')}")
        print(f"🏆 Best model checkpoint: {callbacks[0].best_model_path}")
        print(f"📊 Final validation plots: {os.path.join(run_output_dir, 'final_validation_plots')}")
        
    except Exception as e:
        print(f"❌ Enhanced training failed: {e}")
        raise


def main(args):
    """Enhanced main training function"""
    # Set up PyTorch with enhanced reproducibility
    pl.seed_everything(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Load enhanced configurations
    config = load_config(args.config) if args.config else {}
    model_config = get_model_config(config)
    train_config = get_train_config(config)
    
    print(f"🔧 Using configuration: {args.config or 'default'}")
    
    # Create output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"enhanced_run_{timestamp}"
    run_output_dir = os.path.join(args.output_dir, base_name)
    os.makedirs(run_output_dir, exist_ok=True)
    
    # Save experiment configuration
    save_experiment_config(run_output_dir, args, model_config, train_config)
    
    # Create enhanced datasets with augmentation
    train_dataset, val_dataset, test_dataset, train_transform = create_enhanced_datasets(
        args, config, train_config
    )
    print_enhanced_dataset_info(train_dataset, val_dataset, test_dataset, config)
    
    # Create visualization
    visualizer = TrainingVisualizer(run_output_dir)
    visualizer.plot_dataset_statistics(train_dataset, val_dataset, test_dataset)
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset, val_dataset, test_dataset, train_config
    )
    
    # Test data loading
    test_data_loading(train_loader)
    
    # Create enhanced model
    model = create_enhanced_model(model_config, config)
    
    # Set up enhanced trainer
    trainer, callbacks, loggers = setup_enhanced_trainer(run_output_dir, train_config, config)
    
    # Log enhanced experiment config to WandB if enabled
    log_enhanced_wandb_config(loggers, model_config, train_config, 
                             train_dataset, val_dataset, test_dataset, config)
    
    # Run enhanced training
    run_enhanced_training(trainer, model, train_loader, val_loader, test_loader, 
                         run_output_dir, visualizer, callbacks, config)


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Enhanced Cardiac Ultrasound Guidance Model Training")
    parser.add_argument("--data_dir", type=str, default="data/processed", help="Data directory path")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory path")
    parser.add_argument("--config", type=str, help="Configuration file path (recommended)")
    
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