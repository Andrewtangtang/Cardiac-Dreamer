#!/usr/bin/env python
"""
Cross-validation training script for Cardiac Dreamer

This script implements 5-fold cross-validation where each patient serves as 
the validation set once, with the other 4 patients used for training.

Patient distribution:
- patient1: videos 1-5 (data_0513_01 to data_0513_05)
- patient2: videos 6-9 (data_0513_06 to data_0513_09)
- patient3: videos 11-14 (data_0513_11 to data_0513_14)
- patient4: videos 16-21 (data_0513_16 to data_0513_21)
- patient5: videos 22-26 (data_0513_22 to data_0513_26)
"""

import os
import sys
import argparse
import torch
import pytorch_lightning as pl
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
import json
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import wandb

# Add project root to path
current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_file_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import modular components
from src.models.system import get_cardiac_dreamer_system
from src.data import CrossPatientTransitionsDataset, create_augmented_transform
from src.config import load_config, get_model_config, get_train_config, save_experiment_config
from src.training import setup_callbacks, setup_loggers, create_data_loaders
from src.training.data_loaders import test_data_loading
from src.visualization import TrainingVisualizer


def get_patient_groups():
    """Define the 5 patient groups for cross-validation"""
    return {
        'patient1': ['data_0513_01', 'data_0513_02', 'data_0513_03', 'data_0513_04', 'data_0513_05'],
        'patient2': ['data_0513_06', 'data_0513_07', 'data_0513_08', 'data_0513_09'],
        'patient3': ['data_0513_11', 'data_0513_12', 'data_0513_13', 'data_0513_14'],
        'patient4': ['data_0513_16', 'data_0513_17', 'data_0513_18', 'data_0513_19', 'data_0513_20','data_0513_21'],
        'patient5': ['data_0513_22', 'data_0513_23', 'data_0513_24', 'data_0513_25', 'data_0513_26']
    }


def create_cv_datasets(args, config, train_config, fold_idx, val_patient_group):
    """
    Create train/validation datasets for a specific cross-validation fold
    
    Args:
        args: Command line arguments
        config: Full configuration dictionary
        train_config: Training configuration
        fold_idx: Current fold index (0-4)
        val_patient_group: Patient group name to use as validation
        
    Returns:
        Tuple of (train_dataset, val_dataset, train_transform)
    """
    data_dir = args.data_dir
    patient_groups = get_patient_groups()
    
    # Get validation patients
    val_patients = patient_groups[val_patient_group]
    
    # Get training patients (all other groups)
    train_patients = []
    for group_name, patients in patient_groups.items():
        if group_name != val_patient_group:
            train_patients.extend(patients)
    
    print(f"\n=== Cross-Validation Fold {fold_idx + 1}/5 ===")
    print(f"Validation Patient Group: {val_patient_group}")
    print(f"Validation Patients: {val_patients}")
    print(f"Training Patients ({len(train_patients)}): {train_patients}")
    
    # Create transforms
    train_transform = create_augmented_transform(config)
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Create datasets
    train_dataset = CrossPatientTransitionsDataset(
        data_dir=data_dir,
        transform=train_transform,
        split="train",
        train_patients=train_patients,
        val_patients=val_patients,
        test_patients=[],  # No test set in CV
        small_subset=False,
        normalize_actions=True
    )
    
    val_dataset = CrossPatientTransitionsDataset(
        data_dir=data_dir,
        transform=val_transform,
        split="val",
        train_patients=train_patients,
        val_patients=val_patients,
        test_patients=[],  # No test set in CV
        small_subset=False,
        normalize_actions=True
    )
    
    # Print dataset statistics
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    return train_dataset, val_dataset, train_transform


def create_cv_fold_statistics(train_dataset, val_dataset, fold_idx, val_patient_group, fold_output_dir):
    """Create detailed statistics visualization for a specific CV fold"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    
    print(f"[FOLD {fold_idx + 1}] Creating detailed fold statistics...")
    
    # Set plotting style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Create figure with more detailed layout
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f'Cross-Validation Fold {fold_idx + 1} - Dataset Statistics\nValidation Group: {val_patient_group}', 
                 fontsize=16, fontweight='bold')
    
    # Get patient statistics
    train_stats = train_dataset.get_patient_stats()
    val_stats = val_dataset.get_patient_stats()
    patient_groups = get_patient_groups()
    
    # 1. Dataset size pie chart
    sizes = [len(train_dataset), len(val_dataset)]
    labels = ['Training', 'Validation']
    colors = ['#FF6B6B', '#4ECDC4']
    explode = (0.05, 0.05)
    
    axes[0, 0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', 
                   startangle=90, explode=explode, shadow=True)
    axes[0, 0].set_title(f'Sample Distribution\nTotal: {sum(sizes)} samples')
    
    # 2. Patient samples bar chart with group coloring
    all_patients = list(train_stats.keys()) + list(val_stats.keys())
    train_counts = [train_stats.get(p, 0) for p in all_patients]
    val_counts = [val_stats.get(p, 0) for p in all_patients]
    
    # Determine colors based on patient groups
    patient_colors = []
    for patient in all_patients:
        for group_name, group_patients in patient_groups.items():
            if patient in group_patients:
                if group_name == val_patient_group:
                    patient_colors.append('#4ECDC4')  # Validation color
                else:
                    patient_colors.append('#FF6B6B')  # Training color
                break
    
    x = np.arange(len(all_patients))
    width = 0.35
    
    bars1 = axes[0, 1].bar(x - width/2, train_counts, width, label='Training', color='#FF6B6B', alpha=0.8)
    bars2 = axes[0, 1].bar(x + width/2, val_counts, width, label='Validation', color='#4ECDC4', alpha=0.8)
    
    axes[0, 1].set_xlabel('Patients')
    axes[0, 1].set_ylabel('Number of Samples')
    axes[0, 1].set_title('Samples per Patient (Cross-Validation Split)')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(all_patients, rotation=45, ha='right')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        if height > 0:
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                           f'{int(height)}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        if height > 0:
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                           f'{int(height)}', ha='center', va='bottom', fontsize=8)
    
    # 3. Patient group breakdown
    group_info = []
    for group_name, group_patients in patient_groups.items():
        is_val_group = (group_name == val_patient_group)
        group_info.append({
            'Group': group_name,
            'Patients': ', '.join(group_patients),
            'Role': 'Validation' if is_val_group else 'Training',
            'Sample_Count': sum(val_stats.get(p, 0) for p in group_patients) if is_val_group 
                           else sum(train_stats.get(p, 0) for p in group_patients)
        })
    
    # Create text summary
    summary_text = f"Cross-Validation Fold {fold_idx + 1} Configuration:\n\n"
    summary_text += f"Validation Group: {val_patient_group}\n"
    summary_text += f"Validation Patients: {', '.join(patient_groups[val_patient_group])}\n"
    summary_text += f"Validation Samples: {len(val_dataset)}\n\n"
    
    train_groups = [g for g in patient_groups.keys() if g != val_patient_group]
    summary_text += f"Training Groups: {', '.join(train_groups)}\n"
    summary_text += f"Training Patients: {len(train_stats)} total\n"
    summary_text += f"Training Samples: {len(train_dataset)}\n\n"
    
    summary_text += "Sample Distribution by Group:\n"
    for info in group_info:
        summary_text += f"  {info['Group']}: {info['Sample_Count']} samples ({info['Role']})\n"
    
    axes[0, 2].text(0.05, 0.95, summary_text, fontsize=10, verticalalignment='top', 
                    fontfamily='monospace', transform=axes[0, 2].transAxes)
    axes[0, 2].set_title('Fold Configuration Summary')
    axes[0, 2].axis('off')
    
    # 4. Sample images from training set
    sample_images = []
    sample_labels = []
    for i in range(min(3, len(train_dataset))):
        img, _, _, _ = train_dataset[i]
        sample_images.append(img.squeeze().numpy())
        # Get patient info for this sample
        transition = train_dataset.transitions[i]
        patient_id = transition["patient_id"]
        sample_labels.append(f"Train Sample {i+1}\n({patient_id})")
    
    for i, (img, label) in enumerate(zip(sample_images, sample_labels)):
        if i < 3:  # Show up to 3 images
            axes[1, i].imshow(img, cmap='gray')
            axes[1, i].set_title(label, fontsize=10)
            axes[1, i].axis('off')
    
    # If we have fewer than 3 images, hide the unused subplot
    for i in range(len(sample_images), 3):
        axes[1, i].axis('off')
    
    plt.tight_layout()
    
    # Save plot
    stats_plot_path = os.path.join(fold_output_dir, "fold_dataset_statistics.png")
    plt.savefig(stats_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save detailed statistics as JSON
    fold_stats = {
        'fold_info': {
            'fold_idx': fold_idx + 1,
            'val_patient_group': val_patient_group,
            'val_patients': patient_groups[val_patient_group],
            'train_patients': list(train_stats.keys()),
        },
        'sample_counts': {
            'total_train': len(train_dataset),
            'total_val': len(val_dataset),
            'train_by_patient': train_stats,
            'val_by_patient': val_stats
        },
        'patient_groups': patient_groups
    }
    
    stats_json_path = os.path.join(fold_output_dir, "fold_statistics.json")
    with open(stats_json_path, 'w') as f:
        json.dump(fold_stats, f, indent=2)
    
    print(f"[FOLD {fold_idx + 1}] Fold statistics saved to: {stats_plot_path}")
    print(f"[FOLD {fold_idx + 1}] Fold statistics JSON saved to: {stats_json_path}")


def create_cv_model(model_config, config, fold_idx):
    """Create model for cross-validation fold"""
    print(f"Creating model for fold {fold_idx + 1}...")
    
    # Extract scheduler configuration
    scheduler_config = config.get('advanced', {}).get('scheduler', {})
    scheduler_type = scheduler_config.get('type', 'cosine')
    scheduler_params = {k: v for k, v in scheduler_config.items() if k != 'type'}
    
    # Create model
    model = get_cardiac_dreamer_system(
        d_model=model_config["d_model"],
        nhead=model_config["num_heads"],
        num_layers=model_config["num_layers"],
        feature_dim=model_config["feature_dim"],
        lr=model_config["lr"],
        weight_decay=model_config["weight_decay"],
        lambda_t2_action=model_config["lambda_t2_action"],
        smooth_l1_beta=model_config["smooth_l1_beta"],
        use_flash_attn=model_config["use_flash_attn"],
        primary_task_only=model_config["primary_task_only"],
        freeze_backbone_layers=model_config.get("freeze_backbone_layers", 0),
        scheduler_type=scheduler_type,
        scheduler_config=scheduler_params
    )
    
    return model


def setup_cv_trainer(fold_output_dir, train_config, config, fold_idx):
    """Set up trainer for cross-validation fold"""
    # Setup callbacks and loggers for this fold
    callbacks = setup_callbacks(fold_output_dir, train_config)
    
    # Modify WandB config for cross-validation
    cv_config = config.copy() if config else {}
    if cv_config.get("experiment", {}).get("use_wandb", False):
        cv_config["experiment"]["wandb_project"] = f"cardiac-dreamer-cv-fold{fold_idx+1}"
        cv_config["experiment"]["tags"] = [
            "cross_validation",
            f"fold_{fold_idx+1}",
            "channel_tokens",
            "100_epochs"
        ]
        cv_config["experiment"]["description"] = f"Cross-validation fold {fold_idx+1}/5 - 100 epochs training"
    
    loggers = setup_loggers(fold_output_dir, cv_config)
    
    # Create trainer - FORCE 100 epochs for CV
    trainer = pl.Trainer(
        max_epochs=100,  # Fixed 100 epochs for CV
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


def run_single_fold(args, config, model_config, train_config, fold_idx, val_patient_group, cv_output_dir):
    """Run training for a single cross-validation fold"""
    
    # Create fold-specific output directory
    fold_name = f"fold_{fold_idx+1}_{val_patient_group}"
    fold_output_dir = os.path.join(cv_output_dir, fold_name)
    os.makedirs(fold_output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Starting Cross-Validation Fold {fold_idx + 1}/5")
    print(f"Validation Patient Group: {val_patient_group}")
    print(f"Output Directory: {fold_output_dir}")
    print(f"{'='*60}")
    
    # Finish any existing WandB run before starting new fold
    if fold_idx > 0:
        try:
            wandb.finish()
            print(f"[WANDB] Finished previous WandB run for fold {fold_idx}")
        except:
            pass
    
    try:
        # Create datasets for this fold
        train_dataset, val_dataset, train_transform = create_cv_datasets(
            args, config, train_config, fold_idx, val_patient_group
        )
        
        # Create data loaders
        train_loader, val_loader, _ = create_data_loaders(
            train_dataset, val_dataset, None, train_config
        )
        
        # Test data loading
        test_data_loading(train_loader)
        
        # Create model for this fold
        model = create_cv_model(model_config, config, fold_idx)
        
        # Setup trainer for this fold
        trainer, callbacks, loggers = setup_cv_trainer(
            fold_output_dir, train_config, config, fold_idx
        )
        
        # Log fold-specific config to WandB with allow_val_change=True
        if config and config.get("experiment", {}).get("use_wandb", False):
            for logger in loggers:
                if hasattr(logger, 'experiment') and hasattr(logger.experiment, 'config'):
                    try:
                        logger.experiment.config.update({
                            "fold_info": {
                                "fold_idx": fold_idx + 1,
                                "val_patient_group": val_patient_group,
                                "val_patients": get_patient_groups()[val_patient_group],
                                "train_samples": len(train_dataset),
                                "val_samples": len(val_dataset)
                            }
                        }, allow_val_change=True)
                        print(f"[WANDB] Successfully updated config for fold {fold_idx + 1}")
                    except Exception as e:
                        print(f"[WANDB] Warning: Could not update config for fold {fold_idx + 1}: {e}")
                    break
        
        # Create visualizer for this fold
        visualizer = TrainingVisualizer(fold_output_dir)
        create_cv_fold_statistics(train_dataset, val_dataset, fold_idx, val_patient_group, fold_output_dir)
        
        # Run training for this fold
        print(f"[FOLD {fold_idx + 1}] Starting training...")
        trainer.fit(model, train_loader, val_loader)
        
        # Generate final validation plots
        model.generate_final_validation_plots(output_dir=fold_output_dir)
        
        # Create training summary
        visualizer.create_training_summary(trainer, model)
        
        # Save final model
        final_model_path = os.path.join(fold_output_dir, "final_model.ckpt")
        trainer.save_checkpoint(final_model_path)
        
        # Collect fold results
        fold_results = {
            'fold_idx': fold_idx + 1,
            'val_patient_group': val_patient_group,
            'val_patients': get_patient_groups()[val_patient_group],
            'train_samples': len(train_dataset),
            'val_samples': len(val_dataset),
            'best_model_path': callbacks[0].best_model_path,
            'best_val_loss': float(callbacks[0].best_model_score.cpu()),
            'total_epochs': trainer.current_epoch + 1,
            'fold_output_dir': fold_output_dir
        }
        
        print(f"[FOLD {fold_idx + 1}] Training completed successfully!")
        print(f"[FOLD {fold_idx + 1}] Best validation loss: {fold_results['best_val_loss']:.6f}")
        print(f"[FOLD {fold_idx + 1}] Results saved to: {fold_output_dir}")
        
        # Finish WandB run for this fold
        try:
            wandb.finish()
            print(f"[WANDB] Finished WandB run for fold {fold_idx + 1}")
        except:
            pass
        
        return fold_results
        
    except Exception as e:
        print(f"[ERROR] Fold {fold_idx + 1} failed: {e}")
        # Make sure to finish WandB even if fold fails
        try:
            wandb.finish()
        except:
            pass
        raise


def generate_cv_report(cv_results, cv_output_dir):
    """Generate comprehensive cross-validation report"""
    
    print(f"\n{'='*80}")
    print(f"GENERATING CROSS-VALIDATION REPORT")
    print(f"{'='*80}")
    
    # Create results dataframe
    results_data = []
    for result in cv_results:
        results_data.append({
            'Fold': result['fold_idx'],
            'Validation_Patient_Group': result['val_patient_group'],
            'Validation_Patients': ', '.join(result['val_patients']),
            'Train_Samples': result['train_samples'],
            'Val_Samples': result['val_samples'],
            'Best_Val_Loss': result['best_val_loss'],
            'Total_Epochs': result['total_epochs']
        })
    
    df = pd.DataFrame(results_data)
    
    # Save detailed results
    results_path = os.path.join(cv_output_dir, "cv_detailed_results.csv")
    df.to_csv(results_path, index=False)
    
    # Calculate summary statistics
    val_losses = [r['best_val_loss'] for r in cv_results]
    mean_loss = np.mean(val_losses)
    std_loss = np.std(val_losses)
    min_loss = np.min(val_losses)
    max_loss = np.max(val_losses)
    
    # Create summary report
    summary_report = {
        'cross_validation_summary': {
            'total_folds': len(cv_results),
            'epochs_per_fold': 100,
            'validation_loss_statistics': {
                'mean': float(mean_loss),
                'std': float(std_loss),
                'min': float(min_loss),
                'max': float(max_loss),
                'all_losses': val_losses
            }
        },
        'fold_details': cv_results,
        'patient_performance': {}
    }
    
    # Analyze per-patient performance
    patient_groups = get_patient_groups()
    for result in cv_results:
        group = result['val_patient_group']
        summary_report['patient_performance'][group] = {
            'validation_loss': result['best_val_loss'],
            'patients': result['val_patients'],
            'sample_count': result['val_samples'],
            'relative_performance': 'best' if result['best_val_loss'] == min_loss else 'worst' if result['best_val_loss'] == max_loss else 'average'
        }
    
    # Save summary report
    summary_path = os.path.join(cv_output_dir, "cv_summary_report.json")
    with open(summary_path, 'w') as f:
        json.dump(summary_report, f, indent=2)
    
    # Create visualization plots
    create_cv_visualizations(cv_results, cv_output_dir)
    
    # Print text report
    print_cv_summary(summary_report, cv_output_dir)
    
    return summary_report


def create_cv_visualizations(cv_results, cv_output_dir):
    """Create visualization plots for cross-validation results"""
    
    # Set style
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle('Cross-Validation Results Analysis - Patient Group Performance\n(Lower Loss = Better Performance)', fontsize=16, fontweight='bold')
    
    # Extract data
    folds = [r['fold_idx'] for r in cv_results]
    val_losses = [r['best_val_loss'] for r in cv_results]
    patient_groups = [r['val_patient_group'] for r in cv_results]
    train_samples = [r['train_samples'] for r in cv_results]
    val_samples = [r['val_samples'] for r in cv_results]
    val_patients_lists = [r['val_patients'] for r in cv_results]
    
    # Create color map for performance (green = better/lower loss, red = worse/higher loss)
    min_loss = min(val_losses)
    max_loss = max(val_losses)
    loss_range = max_loss - min_loss
    
    def get_performance_color(loss):
        if loss_range == 0:
            return 'green'  # All same performance
        # Normalize to [0, 1] then map to color (0=best=green, 1=worst=red)
        normalized = (loss - min_loss) / loss_range
        return plt.cm.RdYlGn_r(normalized)  # Red-Yellow-Green reversed (so green is good)
    
    performance_colors = [get_performance_color(loss) for loss in val_losses]
    
    # Plot 1: Validation Loss per Fold with Performance Color Coding
    bars1 = axes[0, 0].bar(folds, val_losses, color=performance_colors, alpha=0.8, edgecolor='black')
    axes[0, 0].set_title('Best Validation Loss per Fold\n(Green=Better Performance, Red=Worse Performance)')
    axes[0, 0].set_xlabel('Fold')
    axes[0, 0].set_ylabel('Validation Loss (Lower is Better)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add value labels and patient group labels on bars
    for i, (v, group) in enumerate(zip(val_losses, patient_groups)):
        axes[0, 0].text(i + 1, v + max(val_losses) * 0.01, f'{v:.4f}', ha='center', va='bottom', fontweight='bold')
        axes[0, 0].text(i + 1, v - max(val_losses) * 0.02, group, ha='center', va='top', fontsize=8, rotation=0)
    
    # Plot 2: Patient Group Performance (Horizontal Bar Chart with Performance Colors)
    # Sort by performance for better visualization
    sorted_indices = sorted(range(len(val_losses)), key=lambda i: val_losses[i])
    sorted_groups = [patient_groups[i] for i in sorted_indices]
    sorted_losses = [val_losses[i] for i in sorted_indices]
    sorted_patients = [val_patients_lists[i] for i in sorted_indices]
    sorted_colors = [performance_colors[i] for i in sorted_indices]
    
    bars2 = axes[0, 1].barh(sorted_groups, sorted_losses, color=sorted_colors, alpha=0.8, edgecolor='black')
    axes[0, 1].set_title('Patient Group Performance Ranking\n(Lower Loss = Better, Sorted Best to Worst)')
    axes[0, 1].set_xlabel('Best Validation Loss (Lower is Better)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Add performance indicators and patient lists
    for i, (group, loss, patients) in enumerate(zip(sorted_groups, sorted_losses, sorted_patients)):
        patient_str = ', '.join(patients)
        rank_text = "BEST" if i == 0 else "2ND" if i == 1 else "3RD" if i == 2 else f"#{i+1}"
        axes[0, 1].text(loss + max(val_losses) * 0.01, i, f'{rank_text} {loss:.4f}\n({patient_str})', 
                       va='center', ha='left', fontsize=8)
    
    # Plot 3: Sample Distribution with Patient Group Details
    x = np.arange(len(folds))
    width = 0.35
    bars3 = axes[1, 0].bar(x - width/2, train_samples, width, label='Training Samples', color='lightblue', alpha=0.8)
    bars4 = axes[1, 0].bar(x + width/2, val_samples, width, label='Validation Samples', color='lightcoral', alpha=0.8)
    axes[1, 0].set_title('Sample Distribution per Fold\n(Training vs Validation)')
    axes[1, 0].set_xlabel('Fold')
    axes[1, 0].set_ylabel('Number of Samples')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels([f'F{f}\n({g})' for f, g in zip(folds, patient_groups)], fontsize=9)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Add sample count labels
    for bar in bars3:
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + max(train_samples) * 0.01,
                       f'{int(height)}', ha='center', va='bottom', fontsize=8)
    for bar in bars4:
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + max(val_samples) * 0.01,
                       f'{int(height)}', ha='center', va='bottom', fontsize=8)
    
    # Plot 4: Loss Distribution with Statistics
    axes[1, 1].hist(val_losses, bins=max(3, len(val_losses)//2), color='purple', alpha=0.7, edgecolor='black')
    mean_loss = np.mean(val_losses)
    std_loss = np.std(val_losses)
    axes[1, 1].axvline(mean_loss, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_loss:.4f}')
    axes[1, 1].axvline(mean_loss + std_loss, color='orange', linestyle=':', linewidth=2, label=f'+1σ: {mean_loss + std_loss:.4f}')
    axes[1, 1].axvline(mean_loss - std_loss, color='orange', linestyle=':', linewidth=2, label=f'-1σ: {mean_loss - std_loss:.4f}')
    # Add best performance line
    axes[1, 1].axvline(min_loss, color='green', linestyle='-', linewidth=2, label=f'Best: {min_loss:.4f}')
    axes[1, 1].set_title('Distribution of Validation Losses\n(Cross-Validation Consistency)')
    axes[1, 1].set_xlabel('Validation Loss (Lower is Better)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 5: Patient Group Breakdown Table with Performance Indicators
    patient_groups_detail = get_patient_groups()
    table_data = []
    for group_name, group_patients in patient_groups_detail.items():
        # Find the fold result for this group
        fold_result = next((r for r in cv_results if r['val_patient_group'] == group_name), None)
        if fold_result:
            loss = fold_result['best_val_loss']
            rank = sorted(val_losses).index(loss) + 1
            performance_indicator = "BEST" if rank == 1 else "2ND" if rank == 2 else "3RD" if rank == 3 else f"#{rank}"
            table_data.append([
                f"{performance_indicator} {group_name}",
                f"Fold {fold_result['fold_idx']}",
                f"{loss:.4f}",
                f"Rank {rank}/{len(cv_results)}",
                f"{fold_result['val_samples']}",
                ', '.join(group_patients)
            ])
    
    # Sort table data by performance (best first)
    table_data.sort(key=lambda x: float(x[2].split()[0]))
    
    # Create table
    table = axes[2, 0].table(cellText=table_data,
                           colLabels=['Patient Group', 'Fold', 'Val Loss', 'Rank', 'Samples', 'Patients'],
                           cellLoc='center',
                           loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 2.2)
    
    # Color code the table rows based on performance
    for i, row in enumerate(table_data):
        loss_val = float(row[2])
        color = get_performance_color(loss_val)
        for j in range(len(row)):
            table[(i+1, j)].set_facecolor(color)
            table[(i+1, j)].set_alpha(0.3)
    
    axes[2, 0].set_title('Patient Group Performance Ranking\n(Green=Better, Red=Worse)')
    axes[2, 0].axis('off')
    
    # Plot 6: Performance Ranking and Summary
    # Rank patient groups by performance (best first)
    ranked_results = sorted(cv_results, key=lambda x: x['best_val_loss'])
    
    summary_text = f"Cross-Validation Summary (Lower Loss = Better):\n\n"
    summary_text += f"Total Folds: {len(cv_results)}\n"
    summary_text += f"Mean Val Loss: {mean_loss:.4f} ± {std_loss:.4f}\n"
    summary_text += f"BEST Performance: {min_loss:.4f} (Lowest Loss)\n"
    summary_text += f"WORST Performance: {max_loss:.4f} (Highest Loss)\n"
    summary_text += f"Performance Range: {max_loss - min_loss:.4f}\n\n"
    
    summary_text += "Patient Group Ranking (Best to Worst):\n"
    for i, result in enumerate(ranked_results):
        rank_text = "BEST" if i == 0 else "2ND" if i == 1 else "3RD" if i == 2 else f"#{i+1}"
        summary_text += f"{rank_text} {result['val_patient_group']}: {result['best_val_loss']:.4f}\n"
        summary_text += f"   Patients: {', '.join(result['val_patients'])}\n"
    
    summary_text += f"\nData Distribution:\n"
    summary_text += f"Total Training Samples: {sum(train_samples)}\n"
    summary_text += f"Total Validation Samples: {sum(val_samples)}\n"
    summary_text += f"Avg Samples per Fold: Train={np.mean(train_samples):.0f}, Val={np.mean(val_samples):.0f}\n"
    
    axes[2, 1].text(0.05, 0.95, summary_text, fontsize=9, verticalalignment='top', 
                    fontfamily='monospace', transform=axes[2, 1].transAxes)
    axes[2, 1].set_title('Performance Summary & Ranking\n(Lower Loss = Better Performance)')
    axes[2, 1].axis('off')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(cv_output_dir, "cv_results_visualization.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[VISUALIZATION] Enhanced cross-validation plots saved to: {plot_path}")
    
    # Also create a simple patient group mapping visualization
    create_patient_group_mapping_plot(cv_output_dir)


def print_cv_summary(summary_report, cv_output_dir):
    """Print detailed cross-validation summary to console"""
    
    cv_summary = summary_report['cross_validation_summary']
    stats = cv_summary['validation_loss_statistics']
    
    print(f"\n{'='*80}")
    print(f"CROSS-VALIDATION SUMMARY REPORT")
    print(f"{'='*80}")
    
    print(f"\nOVERALL STATISTICS:")
    print(f"   Total Folds: {cv_summary['total_folds']}")
    print(f"   Epochs per Fold: {cv_summary['epochs_per_fold']}")
    print(f"   Mean Validation Loss: {stats['mean']:.6f} ± {stats['std']:.6f}")
    print(f"   Best Validation Loss: {stats['min']:.6f}")
    print(f"   Worst Validation Loss: {stats['max']:.6f}")
    print(f"   Loss Range: {stats['max'] - stats['min']:.6f}")
    
    print(f"\nPER-PATIENT GROUP PERFORMANCE:")
    patient_perf = summary_report['patient_performance']
    for group, perf in patient_perf.items():
        status_text = "BEST" if perf['relative_performance'] == 'best' else "WORST" if perf['relative_performance'] == 'worst' else "AVG"
        print(f"   {status_text} {group}: {perf['validation_loss']:.6f} ({perf['sample_count']} samples)")
        print(f"      Patients: {', '.join(perf['patients'])}")
    
    print(f"\nDETAILED FOLD RESULTS:")
    for fold_detail in summary_report['fold_details']:
        print(f"   Fold {fold_detail['fold_idx']}: {fold_detail['best_val_loss']:.6f} (Group: {fold_detail['val_patient_group']})")
    
    print(f"\nRESULTS SAVED TO:")
    print(f"   Main Report: {cv_output_dir}/cv_summary_report.json")
    print(f"   Detailed CSV: {cv_output_dir}/cv_detailed_results.csv")
    print(f"   Visualizations: {cv_output_dir}/cv_results_visualization.png")
    
    print(f"\n{'='*80}")
    print(f"CROSS-VALIDATION COMPLETED SUCCESSFULLY!")
    print(f"{'='*80}")


def create_patient_group_mapping_plot(cv_output_dir):
    """Create a visualization showing patient group mapping for cross-validation"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    
    print(f"[VISUALIZATION] Creating patient group mapping plot...")
    
    # Set up plotting
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Cross-Validation Patient Group Mapping Overview', fontsize=16, fontweight='bold')
    
    patient_groups = get_patient_groups()
    colors = plt.cm.Set3(np.linspace(0, 1, len(patient_groups)))
    
    # Create a fold overview
    fold_info = []
    for i, (group_name, patients) in enumerate(patient_groups.items()):
        fold_info.append({
            'fold': i + 1,
            'group': group_name,
            'patients': patients,
            'val_group': group_name,
            'train_groups': [g for g in patient_groups.keys() if g != group_name]
        })
    
    # Plot 1: Patient Group Overview (Pie Chart)
    group_names = list(patient_groups.keys())
    group_sizes = [len(patients) for patients in patient_groups.values()]
    
    axes[0, 0].pie(group_sizes, labels=group_names, colors=colors, autopct='%1.0f patients', 
                   startangle=90, explode=[0.05] * len(group_names))
    axes[0, 0].set_title('Patient Groups\n(Each group = 1 validation fold)')
    
    # Plot 2: Fold Configuration Matrix
    matrix_data = np.zeros((len(patient_groups), len(patient_groups)))
    group_list = list(patient_groups.keys())
    
    for i, val_group in enumerate(group_list):
        for j, group in enumerate(group_list):
            if group == val_group:
                matrix_data[i, j] = 2  # Validation
            else:
                matrix_data[i, j] = 1  # Training
    
    im = axes[0, 1].imshow(matrix_data, cmap='RdYlBu_r', aspect='auto')
    axes[0, 1].set_xticks(range(len(group_list)))
    axes[0, 1].set_yticks(range(len(group_list)))
    axes[0, 1].set_xticklabels(group_list, rotation=45)
    axes[0, 1].set_yticklabels([f'Fold {i+1}' for i in range(len(group_list))])
    axes[0, 1].set_title('Cross-Validation Configuration Matrix\n(Red=Validation, Blue=Training)')
    
    # Add text annotations
    for i in range(len(group_list)):
        for j in range(len(group_list)):
            text = 'VAL' if matrix_data[i, j] == 2 else 'TRAIN'
            axes[0, 1].text(j, i, text, ha="center", va="center", 
                           color='white' if matrix_data[i, j] == 2 else 'black', fontweight='bold')
    
    # Plot 3: Patient Distribution Table
    table_data = []
    for group_name, patients in patient_groups.items():
        table_data.append([
            group_name,
            f"{len(patients)} patients",
            ', '.join(patients)
        ])
    
    table = axes[0, 2].table(cellText=table_data,
                           colLabels=['Patient Group', 'Count', 'Patient IDs'],
                           cellLoc='left',
                           loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 2.5)
    
    # Color code table rows
    for i in range(len(table_data)):
        for j in range(3):
            table[(i+1, j)].set_facecolor(colors[i])
            table[(i+1, j)].set_alpha(0.3)
    
    axes[0, 2].set_title('Patient Group Details')
    axes[0, 2].axis('off')
    
    # Plots 4-6: Individual fold details
    for fold_idx in range(min(3, len(fold_info))):
        info = fold_info[fold_idx]
        ax = axes[1, fold_idx]
        
        # Create a simple bar chart showing train vs val groups
        categories = ['Validation', 'Training']
        val_count = len(info['patients'])
        train_count = sum(len(patient_groups[g]) for g in info['train_groups'])
        counts = [val_count, train_count]
        
        bars = ax.bar(categories, counts, color=['coral', 'lightgreen'], alpha=0.8)
        ax.set_title(f'Fold {info["fold"]} Configuration\nVal Group: {info["group"]}')
        ax.set_ylabel('Number of Patients')
        
        # Add value labels
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{count}', ha='center', va='bottom', fontweight='bold')
        
        # Add group details as text
        val_text = f"Val: {info['group']}\n({', '.join(info['patients'])})"
        train_text = f"Train: {', '.join(info['train_groups'])}"
        
        ax.text(0.02, 0.98, val_text, transform=ax.transAxes, fontsize=8,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='coral', alpha=0.3))
        ax.text(0.02, 0.5, train_text, transform=ax.transAxes, fontsize=8,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    
    # If fewer than 3 folds, hide unused subplots
    for i in range(len(fold_info), 3):
        axes[1, i].axis('off')
    
    plt.tight_layout()
    
    # Save plot
    mapping_plot_path = os.path.join(cv_output_dir, "patient_group_mapping.png")
    plt.savefig(mapping_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[VISUALIZATION] Patient group mapping plot saved to: {mapping_plot_path}")


def main(args):
    """Main cross-validation function"""
    # Set up reproducibility
    pl.seed_everything(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Load configuration
    config = load_config(args.config) if args.config else {}
    model_config = get_model_config(config)
    train_config = get_train_config(config)
    
    print(f"[CONFIG] Using configuration: {args.config or 'default'}")
    
    # Create main output directory for cross-validation
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cv_run_name = f"cross_validation_{timestamp}"
    cv_output_dir = os.path.join(args.output_dir, cv_run_name)
    os.makedirs(cv_output_dir, exist_ok=True)
    
    print(f"[OUTPUT] Cross-validation results will be saved to: {cv_output_dir}")
    
    # Save experiment configuration
    save_experiment_config(cv_output_dir, args, model_config, train_config)
    
    # Define cross-validation folds
    patient_groups = get_patient_groups()
    cv_folds = list(patient_groups.keys())
    
    print(f"\n[CV SETUP] Setting up {len(cv_folds)}-fold cross-validation")
    print(f"[CV SETUP] Patient groups: {cv_folds}")
    print(f"[CV SETUP] Each fold will train for 100 epochs")
    
    # Run cross-validation
    cv_results = []
    
    for fold_idx, val_patient_group in enumerate(cv_folds):
        try:
            fold_result = run_single_fold(
                args, config, model_config, train_config, 
                fold_idx, val_patient_group, cv_output_dir
            )
            cv_results.append(fold_result)
            
        except Exception as e:
            print(f"[ERROR] Failed on fold {fold_idx + 1}: {e}")
            # Continue with remaining folds
            continue
    
    # Ensure all WandB runs are finished
    try:
        wandb.finish()
        print(f"[WANDB] Finished all WandB runs for cross-validation")
    except:
        pass
    
    # Generate comprehensive report
    if cv_results:
        print(f"\n[REPORT] Generating cross-validation report...")
        summary_report = generate_cv_report(cv_results, cv_output_dir)
        
        print(f"\n[SUCCESS] Cross-validation completed!")
        print(f"[SUCCESS] {len(cv_results)}/{len(cv_folds)} folds completed successfully")
        print(f"[SUCCESS] Results saved to: {cv_output_dir}")
    else:
        print(f"\n[ERROR] No folds completed successfully!")
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-Validation Training for Cardiac Dreamer")
    parser.add_argument("--data_dir", type=str, default="data/processed", help="Data directory path")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory path")
    parser.add_argument("--config", type=str, required=True, help="Configuration file path")
    
    # Add compatibility parameters for save_experiment_config function
    parser.add_argument("--manual_splits", action="store_true", default=False, help="Compatibility parameter (not used in CV)")
    parser.add_argument("--train_patients", type=str, default=None, help="Compatibility parameter (not used in CV)")
    parser.add_argument("--val_patients", type=str, default=None, help="Compatibility parameter (not used in CV)")
    parser.add_argument("--test_patients", type=str, default=None, help="Compatibility parameter (not used in CV)")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    main(args) 