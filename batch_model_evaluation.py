#!/usr/bin/env python
"""
Batch Model Evaluation Script: Test multiple best models and generate scatter plots and statistics
Usage: python batch_model_evaluation.py
"""

import os
import sys
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
from src.train import CrossPatientTransitionsDataset, get_patient_splits


def load_model_from_checkpoint(checkpoint_path: str):
    """Load trained model from checkpoint"""
    print(f"Loading model from: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Try to get hyperparameters from checkpoint
    if 'hyper_parameters' in checkpoint:
        hparams = checkpoint['hyper_parameters']
        print(f"📊 Found hyperparameters in checkpoint: {list(hparams.keys())}")
        
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
        print("⚠️ No hyperparameters found in checkpoint, using default configuration")
        # Use default configuration
        model = get_cardiac_dreamer_system(
            token_type="patch",  # Inferred from experiment directory
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
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model state dict: {e}")
        print("🔧 Trying to load with strict=False...")
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        print("✅ Model loaded with some missing/unexpected keys")
    
    model.eval()
    return model


def create_dataset(data_dir: str = "data/processed", split: str = "train"):
    """Create specified dataset"""
    print(f"Creating {split} dataset from: {data_dir}")
    
    # Automatically detect patient splits
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
    
    print(f"✅ {split.capitalize()} dataset created: {len(dataset)} samples")
    print(f"📊 {split.capitalize()} patients: {dataset.get_patient_stats()}")
    
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
            
            # 🎯 Unified denormalization: all predictions and targets use same statistics
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
        mean_abs_target = np.mean(np.abs(gt_dim))
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


def create_scatter_plots(predictions, ground_truth, model_name, output_dir):
    """Create scatter plots"""
    print(f"Creating scatter plots for {model_name}...")
    
    # Create output directory
    plots_dir = os.path.join(output_dir, f"scatter_plots_{model_name}")
    os.makedirs(plots_dir, exist_ok=True)
    
    # 6DOF dimension names
    dimension_names = ['X', 'Y', 'Z', 'Roll', 'Pitch', 'Yaw']
    dimension_units = ['mm', 'mm', 'mm', 'deg', 'deg', 'deg']
    
    # Set plot style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Create combined plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    r2_scores = []
    
    for i, (dim_name, unit) in enumerate(zip(dimension_names, dimension_units)):
        pred_dim = predictions[:, i]
        gt_dim = ground_truth[:, i]
        
        # Calculate R² score
        r2 = r2_score(gt_dim, pred_dim)
        r2_scores.append(r2)
        
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
    
    plt.suptitle(f'{model_name} - Scatter Plots (Mean R² = {np.mean(r2_scores):.4f})', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'scatter_combined_{model_name}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Scatter plots saved to: {plots_dir}")
    return r2_scores


def evaluate_model(checkpoint_path, model_name, data_loader, device, output_dir):
    """Evaluate single model"""
    print(f"\n🚀 Evaluating {model_name}...")
    print(f"📁 Checkpoint: {checkpoint_path}")
    
    try:
        # 1. Load model
        model = load_model_from_checkpoint(checkpoint_path)
        model = model.to(device)
        
        # 2. Generate predictions
        predictions, ground_truth = generate_predictions(model, data_loader, device)
        
        # 3. Calculate metrics
        metrics = calculate_metrics(predictions, ground_truth)
        
        # 4. Create scatter plots
        r2_scores = create_scatter_plots(predictions, ground_truth, model_name, output_dir)
        
        # 5. Save detailed statistics
        stats_file = os.path.join(output_dir, f"detailed_metrics_{model_name}.json")
        with open(stats_file, 'w') as f:
            json.dump({
                'model_name': model_name,
                'checkpoint_path': checkpoint_path,
                'timestamp': datetime.now().isoformat(),
                'metrics': metrics
            }, f, indent=2)
        
        print(f"📄 Detailed metrics saved to: {stats_file}")
        
        # 6. Print brief results
        print(f"\n📊 {model_name} Results:")
        dimension_names = ['X', 'Y', 'Z', 'Roll', 'Pitch', 'Yaw']
        for i, dim_name in enumerate(dimension_names):
            r2 = metrics[dim_name]['r2_score']
            mse = metrics[dim_name]['mse']
            print(f"  {dim_name:5s}: R² = {r2:7.4f}, MSE = {mse:8.4f}")
        print(f"  Mean : R² = {metrics['overall']['mean_r2']:7.4f}, MSE = {metrics['overall']['mean_mse']:8.4f}")
        
        return metrics
        
    except Exception as e:
        print(f"❌ Error evaluating {model_name}: {e}")
        return None


def create_comparison_report(all_metrics, output_dir):
    """Create model comparison report"""
    print("\n📊 Creating comparison report...")
    
    # Create comparison table
    comparison_data = []
    dimension_names = ['X', 'Y', 'Z', 'Roll', 'Pitch', 'Yaw']
    
    for model_name, metrics in all_metrics.items():
        if metrics is None:
            continue
            
        row = {'Model': model_name}
        
        # Add R² scores for each dimension
        for dim in dimension_names:
            row[f'{dim}_R2'] = metrics[dim]['r2_score']
            row[f'{dim}_MSE'] = metrics[dim]['mse']
        
        # Add overall metrics
        row['Mean_R2'] = metrics['overall']['mean_r2']
        row['Mean_MSE'] = metrics['overall']['mean_mse']
        row['Mean_Correlation'] = metrics['overall']['mean_correlation']
        
        comparison_data.append(row)
    
    # Convert to DataFrame and save
    df = pd.DataFrame(comparison_data)
    
    # Save detailed comparison table
    detailed_csv = os.path.join(output_dir, 'detailed_model_comparison.csv')
    df.to_csv(detailed_csv, index=False)
    
    # Create simplified comparison table (R² scores only)
    r2_columns = ['Model'] + [f'{dim}_R2' for dim in dimension_names] + ['Mean_R2']
    r2_df = df[r2_columns]
    
    simple_csv = os.path.join(output_dir, 'r2_comparison.csv')
    r2_df.to_csv(simple_csv, index=False)
    
    # Print comparison results
    print("\n🏆 Model Ranking by Mean R²:")
    sorted_df = df.sort_values('Mean_R2', ascending=False)
    for idx, row in sorted_df.iterrows():
        print(f"  {row['Model']:20s}: Mean R² = {row['Mean_R2']:.4f}")
    
    print(f"\n📁 Comparison reports saved:")
    print(f"  - Detailed: {detailed_csv}")
    print(f"  - R² only: {simple_csv}")
    
    return df


def main():
    """Main function"""
    print("🚀 Starting batch model evaluation...")
    
    # Set parameters
    data_dir = "data/processed"
    output_dir = "batch_evaluation_results"
    batch_size = 16
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Define models to test
    models_to_test = [
        {
            "name": "epoch9_best",
            "path": "outputs_channel_token/run_20250528_041012\checkpoints\cardiac_dreamer-epoch=05-val_main_task_loss=0.1955.ckpt"
        },
        {
            "name": "epoch11_second",
            "path": "outputs_channel_tokenrun_20250528_041012\checkpoints\cardiac_dreamer-epoch=11-val_main_task_loss=0.1951.ckpt"
        },
        {
            "name": "epoch05_third",
            "path": "outputs_channel_token\run_20250528_041012\checkpoints\cardiac_dreamer-epoch=05-val_main_task_loss=0.1955.ckpt"
        }]
    
    # Check if model files exist
    valid_models = []
    for model_info in models_to_test:
        if os.path.exists(model_info["path"]):
            valid_models.append(model_info)
            print(f"✅ Found: {model_info['name']} - {model_info['path']}")
        else:
            print(f"❌ Missing: {model_info['name']} - {model_info['path']}")
    
    if not valid_models:
        print("❌ No valid model checkpoints found!")
        return
    
    # Create training dataset loader
    print(f"\n📊 Creating training dataset...")
    train_dataset = create_dataset(data_dir, split="train")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=False,
        persistent_workers=False
    )
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    # Evaluate all models
    all_metrics = {}
    
    for model_info in valid_models:
        metrics = evaluate_model(
            checkpoint_path=model_info["path"],
            model_name=model_info["name"],
            data_loader=train_loader,
            device=device,
            output_dir=output_dir
        )
        all_metrics[model_info["name"]] = metrics
    
    # Create comparison report
    if any(metrics is not None for metrics in all_metrics.values()):
        create_comparison_report(all_metrics, output_dir)
    
    print(f"\n🎉 Batch evaluation completed!")
    print(f"📁 All results saved to: {output_dir}")


if __name__ == "__main__":
    main() 