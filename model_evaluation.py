#!/usr/bin/env python
"""
Comprehensive Model Evaluation Script
Combines automatic checkpoint detection with detailed model evaluation
Supports both training and validation dataset evaluation
Usage: python model_evaluation.py [--use_custom_split] [--split SPLIT] [--auto_find_best] [--run_dir RUN_DIR]
"""

import os
import sys
import argparse
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sklearn.metrics import r2_score
import json
from datetime import datetime
import pandas as pd

# Add project root to path
current_file_dir = os.path.dirname(os.path.abspath(__file__))
if current_file_dir not in sys.path:
    sys.path.insert(0, current_file_dir)

# Import model and dataset
from src.models.system import get_cardiac_dreamer_system
from src.data import CrossPatientTransitionsDataset, get_patient_splits, get_custom_patient_splits_no_test


def find_best_checkpoint(run_dir: str):
    """Find the best checkpoint from a training run directory"""
    checkpoints_dir = os.path.join(run_dir, "checkpoints")
    
    if not os.path.exists(checkpoints_dir):
        print(f"[ERROR] Checkpoints directory not found: {checkpoints_dir}")
        return None
    
    # Find all checkpoint files
    checkpoint_files = glob.glob(os.path.join(checkpoints_dir, "cardiac_dreamer-epoch=*-val_main_task_loss=*.ckpt"))
    
    if not checkpoint_files:
        print(f"[ERROR] No checkpoint files found in: {checkpoints_dir}")
        return None
    
    # Sort by validation loss (lower is better)
    def extract_val_loss(filename):
        try:
            # Extract validation loss from filename
            parts = os.path.basename(filename).split('-')
            for part in parts:
                if part.startswith('val_main_task_loss='):
                    return float(part.split('=')[1].replace('.ckpt', ''))
        except:
            return float('inf')
        return float('inf')
    
    # Sort and select the best
    checkpoint_files.sort(key=extract_val_loss)
    best_checkpoint = checkpoint_files[0]
    
    print(f"[SUCCESS] Found best checkpoint: {best_checkpoint}")
    print(f"[INFO] Validation loss: {extract_val_loss(best_checkpoint):.4f}")
    
    return best_checkpoint


def load_model_from_checkpoint(checkpoint_path: str):
    """Load trained model from checkpoint"""
    print(f"Loading model from: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Try to get hyperparameters from checkpoint
    if 'hyper_parameters' in checkpoint:
        hparams = checkpoint['hyper_parameters']
        print(f"Found hyperparameters in checkpoint: {list(hparams.keys())}")
        
        # Use configuration from checkpoint
        model = get_cardiac_dreamer_system(
            token_type=hparams.get("token_type", "channel"),
            d_model=hparams.get("d_model", 768),
            nhead=hparams.get("nhead", 12),
            num_layers=hparams.get("num_layers", 6),
            feature_dim=hparams.get("feature_dim", 49),
            lr=hparams.get("lr", 1e-4),
            weight_decay=hparams.get("weight_decay", 1e-5),
            lambda_t2_action=hparams.get("lambda_t2_action", 1.0),
            smooth_l1_beta=hparams.get("smooth_l1_beta", 1.0),
            use_flash_attn=hparams.get("use_flash_attn", False),
            primary_task_only=hparams.get("primary_task_only", False)
        )
    else:
        print("Warning: No hyperparameters found in checkpoint, using default configuration")
        # Use default configuration
        model = get_cardiac_dreamer_system(
            token_type="channel",
            d_model=768,
            nhead=12,
            num_layers=6,
            feature_dim=49,
            lr=1e-4,
            weight_decay=1e-5,
            lambda_t2_action=1.0,
            smooth_l1_beta=1.0,
            use_flash_attn=False,
            primary_task_only=False
        )
    
    # Load model weights
    try:
        model.load_state_dict(checkpoint['state_dict'])
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model state dict: {e}")
        print("Trying to load with strict=False...")
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        print("Model loaded with some missing/unexpected keys")
    
    model.eval()
    return model


def create_dataset(data_dir: str = "data/processed", split: str = "val", use_custom_split: bool = False):
    """Create specified dataset"""
    print(f"Creating {split} dataset from: {data_dir}")
    
    # Choose split method
    if use_custom_split:
        print("Using custom split: patients 1-5 as validation set")
        train_patients, val_patients, test_patients = get_custom_patient_splits_no_test(data_dir)
    else:
        print("Using automatic patient splits")
        train_patients, val_patients, test_patients = get_patient_splits(data_dir)
    
    # Set up image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Create dataset
    dataset = CrossPatientTransitionsDataset(
        data_dir=data_dir,
        transform=transform,
        split=split,
        train_patients=train_patients,
        val_patients=val_patients,
        test_patients=test_patients,
        small_subset=False,
        normalize_actions=True
    )
    
    print(f"Dataset created: {len(dataset)} samples")
    print(f"Patients: {dataset.get_patient_stats()}")
    
    return dataset


def generate_predictions(model, data_loader, device):
    """Generate predictions on dataset"""
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    print("Generating predictions...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            if batch_idx % 10 == 0:
                print(f"Processing batch {batch_idx + 1}/{len(data_loader)}")
            
            image_t1, a_hat_t1_to_t2_gt, at1_6dof_gt, at2_6dof_gt = batch
            
            # Move to device
            image_t1 = image_t1.to(device)
            a_hat_t1_to_t2_gt = a_hat_t1_to_t2_gt.to(device)
            at1_6dof_gt = at1_6dof_gt.to(device)
            
            # Forward pass
            outputs = model(image_t1, a_hat_t1_to_t2_gt)
            at1_pred = outputs['predicted_action_composed']  # Main task prediction
            
            # Unified denormalization: all predictions and targets use same statistics
            # Get normalization statistics from dataset
            if hasattr(data_loader.dataset, 'action_mean') and hasattr(data_loader.dataset, 'action_std'):
                action_mean = torch.tensor(data_loader.dataset.action_mean, device=device)
                action_std = torch.tensor(data_loader.dataset.action_std, device=device)
                
                # Denormalize predictions and targets
                at1_pred_denorm = at1_pred * action_std + action_mean
                at1_target_denorm = at1_6dof_gt * action_std + action_mean
            else:
                print("Warning: No normalization stats found, using raw values")
                at1_pred_denorm = at1_pred
                at1_target_denorm = at1_6dof_gt
            
            # Collect results (using denormalized values)
            all_predictions.append(at1_pred_denorm.cpu().numpy())
            all_targets.append(at1_target_denorm.cpu().numpy())
    
    # Combine all batches
    predictions = np.vstack(all_predictions)
    targets = np.vstack(all_targets)
    
    print(f"Generated predictions for {predictions.shape[0]} samples")
    return predictions, targets


def calculate_metrics(predictions, ground_truth):
    """Calculate detailed evaluation metrics"""
    dimension_names = ['X', 'Y', 'Z', 'Roll', 'Pitch', 'Yaw']
    dimension_units = ['mm', 'mm', 'mm', 'deg', 'deg', 'deg']
    
    metrics = {}
    
    for i, (dim_name, unit) in enumerate(zip(dimension_names, dimension_units)):
        pred_dim = predictions[:, i]
        gt_dim = ground_truth[:, i]
        
        # Calculate various metrics
        r2 = r2_score(gt_dim, pred_dim)
        mse = np.mean((pred_dim - gt_dim) ** 2)
        mae = np.mean(np.abs(pred_dim - gt_dim))
        rmse = np.sqrt(mse)
        correlation = np.corrcoef(pred_dim, gt_dim)[0, 1]
        
        # Calculate percentage error
        mape = np.mean(np.abs((pred_dim - gt_dim) / (gt_dim + 1e-8))) * 100  # Avoid division by zero
        
        metrics[dim_name] = {
            'r2_score': float(r2),
            'mse': float(mse),
            'mae': float(mae),
            'rmse': float(rmse),
            'correlation': float(correlation),
            'mape': float(mape),
            'pred_mean': float(np.mean(pred_dim)),
            'pred_std': float(np.std(pred_dim)),
            'gt_mean': float(np.mean(gt_dim)),
            'gt_std': float(np.std(gt_dim)),
            'unit': unit
        }
    
    # Calculate overall metrics
    overall_r2 = np.mean([metrics[dim]['r2_score'] for dim in dimension_names])
    overall_mse = np.mean([metrics[dim]['mse'] for dim in dimension_names])
    overall_correlation = np.mean([metrics[dim]['correlation'] for dim in dimension_names])
    
    metrics['overall'] = {
        'mean_r2': float(overall_r2),
        'mean_mse': float(overall_mse),
        'mean_correlation': float(overall_correlation),
        'total_samples': int(len(predictions))
    }
    
    return metrics


def create_scatter_plots(predictions, ground_truth, model_name, split_name, output_dir):
    """Create detailed scatter plots"""
    print(f"Creating scatter plots for {model_name} on {split_name} set...")
    
    # Create output directory
    plots_dir = os.path.join(output_dir, f"scatter_plots_{model_name}_{split_name}")
    os.makedirs(plots_dir, exist_ok=True)
    
    # 6DOF dimension names
    dimension_names = ['X', 'Y', 'Z', 'Roll', 'Pitch', 'Yaw']
    dimension_units = ['mm', 'mm', 'mm', 'deg', 'deg', 'deg']
    
    # Set plot style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    r2_scores = []
    
    # Create individual scatter plots
    for i, (dim_name, unit) in enumerate(zip(dimension_names, dimension_units)):
        plt.figure(figsize=(10, 8))
        
        pred_dim = predictions[:, i]
        gt_dim = ground_truth[:, i]
        
        # Calculate R² score
        r2 = r2_score(gt_dim, pred_dim)
        r2_scores.append(r2)
        correlation = np.corrcoef(pred_dim, gt_dim)[0, 1]
        
        # Create scatter plot
        plt.scatter(gt_dim, pred_dim, alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
        
        # Add perfect prediction line (y=x)
        min_val = min(np.min(gt_dim), np.min(pred_dim))
        max_val = max(np.max(gt_dim), np.max(pred_dim))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction (y=x)')
        
        # Add trend line
        z = np.polyfit(gt_dim, pred_dim, 1)
        p = np.poly1d(z)
        plt.plot(gt_dim, p(gt_dim), 'g-', linewidth=2, alpha=0.8, label=f'Trend Line (slope={z[0]:.3f})')
        
        # Set labels and title
        plt.xlabel(f'Ground Truth {dim_name} ({unit})', fontsize=12)
        plt.ylabel(f'Predicted {dim_name} ({unit})', fontsize=12)
        plt.title(f'{dim_name} Axis: Predicted vs Ground Truth\nR² = {r2:.4f}, Correlation = {correlation:.4f}', fontsize=14)
        
        # Add grid and legend
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Set equal axis ratio
        plt.axis('equal')
        
        # Save individual plot
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'scatter_{dim_name.lower()}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  {dim_name} axis: R² = {r2:.4f}")
    
    # Create combined plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, (dim_name, unit) in enumerate(zip(dimension_names, dimension_units)):
        pred_dim = predictions[:, i]
        gt_dim = ground_truth[:, i]
        r2 = r2_scores[i]
        
        # Scatter plot
        axes[i].scatter(gt_dim, pred_dim, alpha=0.6, s=20, edgecolors='black', linewidth=0.3)
        
        # Perfect prediction line
        min_val = min(np.min(gt_dim), np.min(pred_dim))
        max_val = max(np.max(gt_dim), np.max(pred_dim))
        axes[i].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, alpha=0.8)
        
        # Trend line
        z = np.polyfit(gt_dim, pred_dim, 1)
        p = np.poly1d(z)
        axes[i].plot(gt_dim, p(gt_dim), 'g-', linewidth=2, alpha=0.8)
        
        # Labels and title
        axes[i].set_xlabel(f'Ground Truth {dim_name} ({unit})')
        axes[i].set_ylabel(f'Predicted {dim_name} ({unit})')
        axes[i].set_title(f'{dim_name}: R² = {r2:.4f}')
        axes[i].grid(True, alpha=0.3)
        axes[i].set_aspect('equal', adjustable='box')
    
    plt.suptitle(f'{model_name} on {split_name} set - Mean R² = {np.mean(r2_scores):.4f}', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'scatter_combined_{model_name}_{split_name}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Scatter plots saved to: {plots_dir}")
    return r2_scores


def evaluate_model(checkpoint_path, model_name, data_loader, device, split_name, output_dir):
    """Evaluate single model"""
    print(f"\nEvaluating {model_name} on {split_name} set...")
    print(f"Checkpoint: {checkpoint_path}")
    
    try:
        # 1. Load model
        model = load_model_from_checkpoint(checkpoint_path)
        model = model.to(device)
        
        # 2. Generate predictions
        predictions, ground_truth = generate_predictions(model, data_loader, device)
        
        # 3. Calculate metrics
        metrics = calculate_metrics(predictions, ground_truth)
        
        # 4. Create scatter plots
        r2_scores = create_scatter_plots(predictions, ground_truth, model_name, split_name, output_dir)
        
        # 5. Save detailed statistics
        stats_file = os.path.join(output_dir, f"detailed_metrics_{model_name}_{split_name}.json")
        with open(stats_file, 'w') as f:
            json.dump({
                'model_name': model_name,
                'split_name': split_name,
                'checkpoint_path': checkpoint_path,
                'timestamp': datetime.now().isoformat(),
                'metrics': metrics
            }, f, indent=2)
        
        print(f"Detailed metrics saved to: {stats_file}")
        
        # 6. Print brief results
        print(f"\n{model_name} on {split_name} Results:")
        dimension_names = ['X', 'Y', 'Z', 'Roll', 'Pitch', 'Yaw']
        for i, dim_name in enumerate(dimension_names):
            r2 = metrics[dim_name]['r2_score']
            mse = metrics[dim_name]['mse']
            print(f"  {dim_name:5s}: R² = {r2:7.4f}, MSE = {mse:8.4f}")
        print(f"  Mean : R² = {metrics['overall']['mean_r2']:7.4f}, MSE = {metrics['overall']['mean_mse']:8.4f}")
        
        return metrics
        
    except Exception as e:
        print(f"Error evaluating {model_name} on {split_name}: {e}")
        return None


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Comprehensive model evaluation with automatic checkpoint detection")
    parser.add_argument("--data_dir", type=str, default="data/processed",
                       help="Data directory path")
    parser.add_argument("--output_dir", type=str, default="model_evaluation_results",
                       help="Output directory for results")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test", "both"],
                       help="Dataset split to evaluate on (use 'both' for train+val)")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size for evaluation")
    parser.add_argument("--use_custom_split", action="store_true",
                       help="Use custom split where patients 1-5 are validation set")
    parser.add_argument("--auto_find_best", action="store_true",
                       help="Automatically find best checkpoint from run directory")
    parser.add_argument("--run_dir", type=str, default="outputs/enhanced_run_20250603_003625",
                       help="Run directory to find best checkpoint (used with --auto_find_best)")
    parser.add_argument("--checkpoint_path", type=str,
                       help="Specific checkpoint path to evaluate")
    
    args = parser.parse_args()
    
    print("🚀 Starting comprehensive model evaluation...")
    print(f"📁 Data directory: {args.data_dir}")
    print(f"📁 Output directory: {args.output_dir}")
    print(f"📊 Evaluating on: {args.split} split(s)")
    if args.use_custom_split:
        print("🔧 Using custom split: patients 1-5 as validation set")
    else:
        print("🔧 Using automatic patient splits")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine checkpoint path
    checkpoint_path = None
    model_name = "model"
    
    if args.auto_find_best and args.run_dir:
        print(f"\n🔍 Looking for best checkpoint in: {args.run_dir}")
        checkpoint_path = find_best_checkpoint(args.run_dir)
        if checkpoint_path:
            # Extract model name from run directory
            model_name = f"best_{os.path.basename(args.run_dir)}"
        else:
            print("❌ Could not find best checkpoint!")
            return
    elif args.checkpoint_path:
        checkpoint_path = args.checkpoint_path
        # Extract model name from checkpoint filename
        model_name = os.path.basename(checkpoint_path).replace('.ckpt', '')
    else:
        print("❌ Please specify either --auto_find_best with --run_dir or --checkpoint_path")
        return
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint file not found: {checkpoint_path}")
        return
    
    print(f"✅ Using checkpoint: {checkpoint_path}")
    
    # Determine which splits to evaluate
    splits_to_evaluate = []
    if args.split == "both":
        splits_to_evaluate = ["train", "val"]
    else:
        splits_to_evaluate = [args.split]
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    # Evaluate on each split
    all_results = {}
    
    for split in splits_to_evaluate:
        print(f"\n{'='*60}")
        print(f"📊 Evaluating on {split.upper()} set")
        print(f"{'='*60}")
        
        # Create dataset
        dataset = create_dataset(args.data_dir, split=split, use_custom_split=args.use_custom_split)
        
        data_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=False,
            persistent_workers=False
        )
        
        # Evaluate model
        results = evaluate_model(
            checkpoint_path=checkpoint_path,
            model_name=model_name,
            data_loader=data_loader,
            device=device,
            split_name=split,
            output_dir=args.output_dir
        )
        
        if results:
            all_results[split] = results
    
    # Create summary report
    if all_results:
        print(f"\n{'='*60}")
        print("📊 EVALUATION SUMMARY")
        print(f"{'='*60}")
        
        summary_data = []
        for split, results in all_results.items():
            if results:
                summary_data.append({
                    'Split': split.upper(),
                    'Samples': results['overall']['total_samples'],
                    'Mean_R2': results['overall']['mean_r2'],
                    'Mean_MSE': results['overall']['mean_mse'],
                    'Mean_Correlation': results['overall']['mean_correlation']
                })
        
        # Save summary
        summary_df = pd.DataFrame(summary_data)
        summary_file = os.path.join(args.output_dir, f'evaluation_summary_{model_name}.csv')
        summary_df.to_csv(summary_file, index=False)
        
        print(summary_df.to_string(index=False))
        print(f"\n📁 Summary saved to: {summary_file}")
    
    print(f"\n🎉 Evaluation completed!")
    print(f"📁 All results saved to: {args.output_dir}")


if __name__ == "__main__":
    main() 