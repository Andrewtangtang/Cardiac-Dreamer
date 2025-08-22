#!/usr/bin/env python
"""
Direct Backbone Evaluation for Cross-Validation Results

This script performs direct backbone evaluation on cross-validation results,
skipping the transformer dreamer for more realistic assessment.

Usage:
python evaluate_cv_direct_backbone.py --cv_output_dir outputs --data_dir data/processed
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import json
import gc
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from src.data import CrossPatientTransitionsDataset, get_patient_splits
from src.models.guidance import pool_features
from src.models.system import CardiacDreamerSystem
from src.models.backbone import get_resnet34_encoder
from src.models.guidance import get_guidance_layer


class DirectBackboneEvaluator(nn.Module):
    """
    Direct Evaluator: ResNet34 Backbone + Guidance Layer
    
    Skip Transformer Dreamer, directly predict action from backbone features
    """
    
    def __init__(
        self,
        backbone: nn.Module,
        guidance: nn.Module,
        device: torch.device = None
    ):
        super().__init__()
        self.backbone = backbone
        self.guidance = guidance
        self.device = device or torch.device('cpu')
        
    def forward(self, image_t1: torch.Tensor) -> torch.Tensor:
        """
        Direct forward pass
        
        Args:
            image_t1: Input image [batch_size, 1, 224, 224]
            
        Returns:
            predicted_action: Predicted action [batch_size, 6]
        """
        # 1. ResNet34 backbone: [B, 1, 224, 224] -> [B, 512, 7, 7]
        feature_map = self.backbone(image_t1)
        
        # 2. Global average pooling: [B, 512, 7, 7] -> [B, 512]
        pooled_features = torch.mean(feature_map.view(feature_map.size(0), 512, -1), dim=2)
        
        # 3. Guidance layer: [B, 512] -> [B, 6]
        predicted_action = self.guidance(pooled_features)
        
        return predicted_action


def find_cv_fold_checkpoints(cv_output_dir: str) -> List[Dict[str, str]]:
    """
    Find all cross-validation fold checkpoints
    
    Args:
        cv_output_dir: Cross-validation output directory
        
    Returns:
        List of fold information dictionaries
    """
    fold_info = []
    
    # Look for fold directories
    for item in os.listdir(cv_output_dir):
        fold_path = os.path.join(cv_output_dir, item)
        if os.path.isdir(fold_path) and item.startswith('fold_'):
            
            # Extract fold number
            try:
                fold_num = int(item.split('_')[1])
            except (IndexError, ValueError):
                continue
            
            # Look for checkpoint files
            checkpoint_dir = os.path.join(fold_path, 'checkpoints')
            if not os.path.exists(checkpoint_dir):
                print(f"Warning: No checkpoints directory found for {item}")
                continue
            
            # Find best checkpoint
            best_checkpoint = None
            for ckpt_file in os.listdir(checkpoint_dir):
                if ckpt_file.endswith('.ckpt') and 'best' in ckpt_file.lower():
                    best_checkpoint = os.path.join(checkpoint_dir, ckpt_file)
                    break
            
            # If no 'best' checkpoint, look for any checkpoint
            if not best_checkpoint:
                for ckpt_file in os.listdir(checkpoint_dir):
                    if ckpt_file.endswith('.ckpt'):
                        best_checkpoint = os.path.join(checkpoint_dir, ckpt_file)
                        break
            
            if best_checkpoint:
                fold_info.append({
                    'fold_num': fold_num,
                    'fold_dir': fold_path,
                    'checkpoint_path': best_checkpoint,
                    'fold_name': item
                })
            else:
                print(f"Warning: No checkpoint found for {item}")
    
    # Sort by fold number
    fold_info.sort(key=lambda x: x['fold_num'])
    
    return fold_info


def infer_val_patient_group_from_fold_dir(fold_dir: str) -> str:
    """
    Infer validation patient group from fold directory name
    
    Args:
        fold_dir: Fold directory path (e.g., 'outputs/.../fold_1_patient1')
        
    Returns:
        Patient group name (e.g., 'patient1')
    """
    # Extract fold directory name
    folder_name = os.path.basename(fold_dir)
    
    # Expected format: fold_N_patientX
    if '_patient' in folder_name:
        # Extract patient group from folder name
        parts = folder_name.split('_')
        for part in parts:
            if part.startswith('patient'):
                return part
    
    # Fallback: try to parse fold number and map to patient groups
    if 'fold_' in folder_name:
        try:
            fold_num_str = folder_name.split('fold_')[1].split('_')[0]
            fold_num = int(fold_num_str)
            # Map fold number to patient group (1-based)
            if 1 <= fold_num <= 5:
                return f'patient{fold_num}'
        except:
            pass
    
    print(f"Warning: Could not infer patient group from folder name: {folder_name}")
    return 'unknown'


def load_fold_components(checkpoint_path: str, device: torch.device) -> Tuple[nn.Module, nn.Module, str]:
    """
    Load backbone and guidance components from fold checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Computing device
        
    Returns:
        Tuple of (backbone, guidance, val_patient_group)
    """
    try:
        print(f"Loading fold checkpoint: {os.path.basename(checkpoint_path)}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Load full system
        system = CardiacDreamerSystem.load_from_checkpoint(
            checkpoint_path, 
            map_location=device,
            strict=False
        )
        
        # Extract components
        backbone = system.backbone
        guidance = system.guidance
        
        # Ensure evaluation mode
        backbone.eval()
        guidance.eval()
        
        # Infer validation patient group from checkpoint directory
        fold_dir = os.path.dirname(checkpoint_path)
        # If the directory is 'checkpoints', go up one more level to get the actual fold directory
        if os.path.basename(fold_dir) == 'checkpoints':
            fold_dir = os.path.dirname(fold_dir)
        val_patient_group = infer_val_patient_group_from_fold_dir(fold_dir)
        
        return backbone, guidance, val_patient_group
        
    except Exception as e:
        print(f"Error loading checkpoint {checkpoint_path}: {e}")
        raise


def load_fold_normalization_stats(fold_dir: str) -> Tuple[List[float], List[float]]:
    """
    Load fold-specific normalization statistics
    
    Args:
        fold_dir: Fold directory path
        
    Returns:
        Tuple of (action_mean, action_std)
    """
    # Try to find normalization stats from fold directory structure
    potential_paths = [
        os.path.join(fold_dir, "normalization_stats.json"),
        os.path.join(fold_dir, "fold_statistics.json"),
    ]
    
    for norm_path in potential_paths:
        if os.path.exists(norm_path):
            try:
                with open(norm_path, 'r') as f:
                    data = json.load(f)
                
                # Extract normalization stats
                if "normalization_stats" in data:
                    norm_stats = data["normalization_stats"]
                    if "action_mean" in norm_stats and "action_std" in norm_stats:
                        return norm_stats["action_mean"], norm_stats["action_std"]
                
                print(f"Warning: No normalization stats found in {norm_path}")
                
            except Exception as e:
                print(f"Warning: Error reading {norm_path}: {e}")
    
    print(f"Warning: No fold-specific normalization stats found for {fold_dir}")
    print("Using default normalization values [0.5], [0.5] for images")
    return None, None


def compute_fold_normalization_stats(
    data_dir: str,
    train_patients: List[str],
    val_patients: List[str]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute normalization statistics for a specific fold's training data
    
    Args:
        data_dir: Data directory
        train_patients: List of training patient IDs for this fold
        val_patients: List of validation patient IDs for this fold
        
    Returns:
        Tuple of (action_mean, action_std) as numpy arrays
    """
    print(f"  Computing fold-specific normalization statistics...")
    print(f"    Training patients: {train_patients}")
    
    # Create a temporary training dataset to compute stats
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Create training dataset without action normalization first
    temp_dataset = CrossPatientTransitionsDataset(
        data_dir=data_dir,
        transform=transform,
        split="train",
        train_patients=train_patients,
        val_patients=val_patients,
        test_patients=[],
        small_subset=False,
        normalize_actions=False  # Don't normalize yet, we need raw data
    )
    
    if len(temp_dataset.transitions) == 0:
        print("    Warning: No transitions found! Using default normalization.")
        return np.array([0.0] * 6), np.array([1.0] * 6)
    
    # Collect all action data from training set
    at1_actions = []
    at2_actions = []
    action_changes = []
    
    for transition in temp_dataset.transitions:
        at1_actions.append(transition["at1_6dof"])
        at2_actions.append(transition["at2_6dof"])
        action_changes.append(transition["action_change_6dof"])
    
    # Convert to numpy arrays
    at1_actions = np.array(at1_actions)
    at2_actions = np.array(at2_actions)
    action_changes = np.array(action_changes)
    
    # Combine all actions for statistics
    all_actions = np.vstack([at1_actions, at2_actions, action_changes])
    
    # Compute statistics
    action_mean = np.mean(all_actions, axis=0)
    action_std = np.std(all_actions, axis=0)
    
    # Prevent division by zero
    action_std = np.where(action_std < 1e-6, 1.0, action_std)
    
    print(f"    Computed from {len(temp_dataset.transitions)} transitions:")
    print(f"    Action mean: {action_mean}")
    print(f"    Action std: {action_std}")
    
    return action_mean, action_std


def create_fold_specific_dataset(
    data_dir: str,
    fold_dir: str,
    val_patient_group: str,
    batch_size: int
) -> DataLoader:
    """
    Create dataset with fold-specific normalization for proper evaluation
    
    Args:
        data_dir: Main data directory
        fold_dir: Fold output directory
        val_patient_group: Validation patient group for this fold
        batch_size: Batch size
        
    Returns:
        DataLoader with fold-specific normalization
    """
    # Get patient splits - we need to reconstruct the fold's patient split
    from src.data import get_patient_groups
    patient_groups = get_patient_groups()
    
    # Reconstruct training patients for this fold
    val_patients = patient_groups[val_patient_group]
    train_patients = []
    for group_name, patients in patient_groups.items():
        if group_name != val_patient_group:
            train_patients.extend(patients)
    
    print(f"  Reconstructing fold dataset:")
    print(f"    Validation group: {val_patient_group}")
    print(f"    Validation patients: {val_patients}")
    print(f"    Training patients: {len(train_patients)} patients")
    
    # Directly compute fold-specific normalization stats from training data
    print(f"  Computing fold-specific normalization dynamically...")
    action_mean, action_std = compute_fold_normalization_stats(
        data_dir, train_patients, val_patients
    )
    
    # Create transform with default image normalization
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Create validation dataset with fold-specific normalization
    dataset = CrossPatientTransitionsDataset(
        data_dir=data_dir,
        transform=transform,
        split="val",
        train_patients=train_patients,
        val_patients=val_patients,
        test_patients=[],
        small_subset=False,
        normalize_actions=True  # Enable normalization
    )
    
    # Override with our computed normalization stats
    dataset.action_mean = action_mean
    dataset.action_std = action_std
    
    print(f"  Final action normalization being used:")
    print(f"    Action mean: {dataset.action_mean}")
    print(f"    Action std: {dataset.action_std}")
    print(f"    Validation samples: {len(dataset)}")
    
    # Create data loader
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return data_loader


def evaluate_fold_direct_backbone(
    evaluator: DirectBackboneEvaluator,
    data_loader: DataLoader,
    device: torch.device,
    fold_num: int,
    val_patient_group: str,
    action_mean: np.ndarray = None,
    action_std: np.ndarray = None
) -> Dict[str, float]:
    """
    Evaluate single fold using direct backbone method
    
    Args:
        evaluator: Direct evaluator
        data_loader: Data loader with fold-specific normalization
        device: Computing device
        fold_num: Fold number
        val_patient_group: Validation patient group
        action_mean: Normalization mean for denormalization
        action_std: Normalization std for denormalization
        
    Returns:
        Evaluation results dictionary
    """
    evaluator.eval()
    
    all_predictions = []
    all_targets = []
    total_samples = 0
    total_loss = 0.0
    
    print(f"  Evaluating fold {fold_num} (val group: {val_patient_group})...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            # Parse batch data
            image_t1, a_hat_t1_to_t2_gt, at1_6dof_gt, at2_6dof_gt = batch
            
            # Move to GPU
            image_t1 = image_t1.to(device)
            at1_6dof_gt = at1_6dof_gt.to(device)
            
            batch_size = image_t1.shape[0]
            
            # Direct prediction
            predicted_action = evaluator(image_t1)
            
            # Calculate loss
            loss = F.smooth_l1_loss(predicted_action, at1_6dof_gt)
            total_loss += loss.item() * batch_size
            
            # Collect predictions and targets
            all_predictions.append(predicted_action.cpu())
            all_targets.append(at1_6dof_gt.cpu())
            
            total_samples += batch_size
            
            # Memory management
            if batch_idx % 50 == 0:
                torch.cuda.empty_cache()
    
    # Concatenate results
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Convert to numpy for metrics calculation
    predictions_np = all_predictions.numpy()
    targets_np = all_targets.numpy()
    
    # Import calculate_metrics with denormalization support
    from model_evaluation import calculate_metrics
    
    # Calculate metrics with denormalization if stats are provided
    if action_mean is not None and action_std is not None:
        print(f"  Using fold-specific denormalization: mean={action_mean}, std={action_std}")
        metrics_dict = calculate_metrics(predictions_np, targets_np, action_mean, action_std)
    else:
        print(f"  WARNING: No denormalization stats provided - MAE will be in normalized units!")
        metrics_dict = calculate_metrics(predictions_np, targets_np)
    
    # Extract overall metrics
    results = {}
    results['fold_num'] = fold_num
    results['val_patient_group'] = val_patient_group
    results['total_samples'] = total_samples
    results['total_loss'] = total_loss / total_samples
    
    # Extract dimension-wise MAE (now in physical units)
    dim_names = ['X', 'Y', 'Z', 'Roll', 'Pitch', 'Yaw']
    for dim in dim_names:
        results[f'mae_{dim.lower()}'] = metrics_dict[dim]['mae']
        results[f'r2_{dim.lower()}'] = metrics_dict[dim]['r2_score']
        results[f'correlation_{dim.lower()}'] = metrics_dict[dim]['correlation']
    
    # Overall metrics
    results['overall_mae'] = metrics_dict['Overall']['mae']
    results['overall_r2'] = metrics_dict['Overall']['r2_score']
    results['overall_correlation'] = metrics_dict['Overall']['correlation']
    
    # Calculate translation and rotation MAE
    translation_mae = np.mean([results['mae_x'], results['mae_y'], results['mae_z']])
    rotation_mae = np.mean([results['mae_roll'], results['mae_pitch'], results['mae_yaw']])
    
    results['translation_mae'] = translation_mae
    results['rotation_mae'] = rotation_mae
    
    print(f"  Results: Translation MAE={translation_mae:.3f}mm, Rotation MAE={rotation_mae:.3f}deg")
    print(f"  Overall MAE={results['overall_mae']:.3f}, R²={results['overall_r2']:.3f}")
    
    return results


def save_cv_evaluation_results(
    all_fold_results: List[Dict],
    summary_stats: Dict,
    output_dir: str
):
    """
    Save CV evaluation results to files
    
    Args:
        all_fold_results: List of fold evaluation results
        summary_stats: Summary statistics
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detailed results CSV
    df = pd.DataFrame(all_fold_results)
    csv_path = os.path.join(output_dir, 'cv_direct_backbone_detailed_results.csv')
    df.to_csv(csv_path, index=False)
    
    # Save summary JSON
    summary_data = {
        'evaluation_type': 'cv_direct_backbone',
        'description': 'Cross-validation direct backbone evaluation (skip transformer dreamer)',
        'timestamp': datetime.now().isoformat(),
        'total_folds': len(all_fold_results),
        'summary_statistics': summary_stats,
        'fold_details': all_fold_results
    }
    
    json_path = os.path.join(output_dir, 'cv_direct_backbone_summary.json')
    with open(json_path, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"CV evaluation results saved:")
    print(f"  - {csv_path}")
    print(f"  - {json_path}")


def print_cv_evaluation_summary(all_fold_results: List[Dict], summary_stats: Dict):
    """
    Print comprehensive CV evaluation summary
    
    Args:
        all_fold_results: List of fold evaluation results
        summary_stats: Summary statistics
    """
    print("\n" + "="*80)
    print("CROSS-VALIDATION DIRECT BACKBONE EVALUATION SUMMARY")
    print("="*80)
    
    print(f"\nOverall Statistics:")
    print(f"   Total Folds: {len(all_fold_results)}")
    print(f"   Total Samples: {sum(r['total_samples'] for r in all_fold_results)}")
    print(f"   Mean Overall MAE: {summary_stats['mean_overall_mae']:.6f}")
    print(f"   Std Overall MAE: {summary_stats['std_overall_mae']:.6f}")
    print(f"   Mean Overall R²: {summary_stats['mean_overall_r2']:.4f}")
    print(f"   Mean Correlation: {summary_stats['mean_overall_correlation']:.4f}")
    
    print(f"\nTranslation vs Rotation Performance:")
    print(f"   Mean Translation MAE: {summary_stats['mean_translation_mae']:.3f} mm")
    print(f"   Mean Rotation MAE: {summary_stats['mean_rotation_mae']:.3f} degrees")
    
    print(f"\nPer-dimension Summary:")
    dim_names = ["X", "Y", "Z", "Roll", "Pitch", "Yaw"]
    print(f"{'Dimension':<10} {'Mean MAE':<12} {'Std MAE':<12} {'Mean R²':<12} {'Mean Corr':<12}")
    print("-" * 68)
    
    for dim_name in dim_names:
        mean_mae = summary_stats[f'mean_mae_{dim_name.lower()}']
        std_mae = summary_stats[f'std_mae_{dim_name.lower()}']
        mean_r2 = summary_stats[f'mean_r2_{dim_name.lower()}']
        mean_corr = summary_stats[f'mean_correlation_{dim_name.lower()}']
        print(f"{dim_name:<10} {mean_mae:<12.6f} {std_mae:<12.6f} {mean_r2:<12.4f} {mean_corr:<12.4f}")
    
    print(f"\nPer-fold Results:")
    print(f"{'Fold':<6} {'Val Group':<12} {'Samples':<8} {'Trans MAE':<10} {'Rot MAE':<10} {'Overall R²':<12}")
    print("-" * 72)
    
    for result in all_fold_results:
        print(f"{result['fold_num']:<6} {result['val_patient_group']:<12} "
              f"{result['total_samples']:<8} {result['translation_mae']:<10.3f} "
              f"{result['rotation_mae']:<10.3f} {result['overall_r2']:<12.4f}")


def calculate_summary_statistics(all_fold_results: List[Dict]) -> Dict:
    """
    Calculate summary statistics across all folds
    
    Args:
        all_fold_results: List of fold evaluation results
        
    Returns:
        Summary statistics dictionary
    """
    summary = {}
    
    # Overall metrics
    overall_mae_values = [r['overall_mae'] for r in all_fold_results]
    overall_r2_values = [r['overall_r2'] for r in all_fold_results]
    overall_corr_values = [r['overall_correlation'] for r in all_fold_results]
    translation_mae_values = [r['translation_mae'] for r in all_fold_results]
    rotation_mae_values = [r['rotation_mae'] for r in all_fold_results]
    
    summary['mean_overall_mae'] = np.mean(overall_mae_values)
    summary['std_overall_mae'] = np.std(overall_mae_values)
    summary['mean_overall_r2'] = np.mean(overall_r2_values)
    summary['std_overall_r2'] = np.std(overall_r2_values)
    summary['mean_overall_correlation'] = np.mean(overall_corr_values)
    summary['std_overall_correlation'] = np.std(overall_corr_values)
    summary['mean_translation_mae'] = np.mean(translation_mae_values)
    summary['std_translation_mae'] = np.std(translation_mae_values)
    summary['mean_rotation_mae'] = np.mean(rotation_mae_values)
    summary['std_rotation_mae'] = np.std(rotation_mae_values)
    
    # Per-dimension metrics
    dim_names = ["X", "Y", "Z", "Roll", "Pitch", "Yaw"]
    for dim_name in dim_names:
        mae_values = [r[f'mae_{dim_name.lower()}'] for r in all_fold_results]
        r2_values = [r[f'r2_{dim_name.lower()}'] for r in all_fold_results]
        corr_values = [r[f'correlation_{dim_name.lower()}'] for r in all_fold_results]
        
        summary[f'mean_mae_{dim_name.lower()}'] = np.mean(mae_values)
        summary[f'std_mae_{dim_name.lower()}'] = np.std(mae_values)
        summary[f'mean_r2_{dim_name.lower()}'] = np.mean(r2_values)
        summary[f'std_r2_{dim_name.lower()}'] = np.std(r2_values)
        summary[f'mean_correlation_{dim_name.lower()}'] = np.mean(corr_values)
        summary[f'std_correlation_{dim_name.lower()}'] = np.std(corr_values)
    
    return summary


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Cross-Validation Direct Backbone Evaluation")
    parser.add_argument("--cv_output_dir", type=str, required=True, 
                       help="Cross-validation output directory containing fold results")
    parser.add_argument("--data_dir", type=str, default="data/processed", 
                       help="Data directory")
    parser.add_argument("--batch_size", type=int, default=16, 
                       help="Batch size for evaluation")
    parser.add_argument("--output_dir", type=str, default="cv_direct_backbone_evaluation", 
                       help="Output directory for evaluation results")
    parser.add_argument("--device", type=str, default="auto", 
                       help="Computing device (auto/cpu/cuda)")
    
    args = parser.parse_args()
    
    print("="*80)
    print("CROSS-VALIDATION DIRECT BACKBONE EVALUATION")
    print("="*80)
    print()
    
    # Set device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    print(f"CV output directory: {args.cv_output_dir}")
    print(f"Data directory: {args.data_dir}")
    print(f"Results will be saved to: {args.output_dir}")
    print()
    
    # Check directories
    if not os.path.exists(args.cv_output_dir):
        raise FileNotFoundError(f"CV output directory does not exist: {args.cv_output_dir}")
    
    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(f"Data directory does not exist: {args.data_dir}")
    
    try:
        # Find fold checkpoints
        print("Discovering fold checkpoints...")
        fold_info = find_cv_fold_checkpoints(args.cv_output_dir)
        
        if not fold_info:
            raise ValueError(f"No valid fold checkpoints found in {args.cv_output_dir}")
        
        print(f"Found {len(fold_info)} fold checkpoints:")
        for info in fold_info:
            print(f"  - Fold {info['fold_num']}: {os.path.basename(info['checkpoint_path'])}")
        print()
        
        # Evaluate each fold
        print("Starting fold evaluations with fold-specific normalization...")
        all_fold_results = []
        all_fold_predictions = []
        all_fold_targets = []
        
        for i, info in enumerate(fold_info):
            print(f"Processing fold {info['fold_num']} ({i+1}/{len(fold_info)})...")
            
            # Load fold components
            backbone, guidance, val_patient_group = load_fold_components(
                info['checkpoint_path'], device
            )
            
            # Create evaluator
            evaluator = DirectBackboneEvaluator(backbone, guidance, device)
            
            # Create fold-specific dataset and data loader
            data_loader = create_fold_specific_dataset(
                args.data_dir, info['fold_dir'], val_patient_group, args.batch_size
            )
            
            # Evaluate fold
            results = evaluate_fold_direct_backbone(
                evaluator, data_loader, device, info['fold_num'], val_patient_group,
                action_mean=data_loader.dataset.action_mean,
                action_std=data_loader.dataset.action_std
            )
            
            all_fold_results.append(results)
            
            # Clean up GPU memory
            del backbone, guidance, evaluator
            torch.cuda.empty_cache()
            gc.collect()
            
            print()
        
        # Calculate summary statistics
        print("Calculating summary statistics...")
        summary_stats = calculate_summary_statistics(all_fold_results)
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Print summary
        print_cv_evaluation_summary(all_fold_results, summary_stats)
        
        # Save results
        print("\nSaving evaluation results...")
        save_cv_evaluation_results(all_fold_results, summary_stats, args.output_dir)
        
        print()
        print("="*80)
        print("CROSS-VALIDATION DIRECT BACKBONE EVALUATION COMPLETED!")
        print("="*80)
        print(f"Results saved in: {args.output_dir}")
        print("  - cv_direct_backbone_detailed_results.csv: Per-fold detailed results")
        print("  - cv_direct_backbone_summary.json: Summary statistics")
        print()
        print("IMPORTANT: MAE values are now in original physical units (mm/degrees)")
        print("           Previous results were in normalized units and too small!")
        print("           This evaluation uses fold-specific denormalization for accuracy.")
        
    except Exception as e:
        print(f"Error occurred during CV evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 