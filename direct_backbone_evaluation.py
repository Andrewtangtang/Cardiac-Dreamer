#!/usr/bin/env python
"""
Direct Backbone Evaluation for Cardiac Dreamer

This script skips the intermediate Transformer Dreamer and directly uses ResNet34 backbone + guidance layer for evaluation.
This evaluation approach may be more realistic for actual applications since it doesn't require predicting next frame features.

Architecture pipeline:
1. ResNet34 backbone: [B, 1, 224, 224] -> [B, 512, 7, 7]
2. Global average pooling: [B, 512, 7, 7] -> [B, 512]  
3. Guidance layer: [B, 512] -> [B, 6]

Usage:
python direct_backbone_evaluation.py --model_path /path/to/checkpoint.ckpt --data_dir data/processed
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
import json
from typing import Dict, List, Tuple
import gc

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import our modules
from src.models.system import CardiacDreamerSystem
from src.models.backbone import get_resnet34_encoder
from src.models.guidance import get_guidance_layer
from src.data import CrossPatientTransitionsDataset, get_patient_splits
import torchvision.transforms as transforms


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


def load_trained_components(checkpoint_path: str, device: torch.device) -> Tuple[nn.Module, nn.Module]:
    """
    Load backbone and guidance components from trained model
    
    Args:
        checkpoint_path: Model checkpoint path
        device: Computing device
        
    Returns:
        backbone, guidance: Trained backbone and guidance layers
    """
    print(f"Loading model checkpoint: {checkpoint_path}")
    
    # Load complete system
    system = CardiacDreamerSystem.load_from_checkpoint(checkpoint_path)
    system.eval()
    system = system.to(device)
    
    print(f"Successfully loaded system model")
    print(f"  - Token type: {system.token_type}")
    print(f"  - Backbone frozen layers: {system.hparams.freeze_backbone_layers}")
    
    # Extract backbone and guidance components
    backbone = system.backbone
    guidance = system.guidance
    
    # Ensure models are in evaluation mode
    backbone.eval()
    guidance.eval()
    
    return backbone, guidance


def evaluate_direct_backbone(
    evaluator: DirectBackboneEvaluator,
    data_loader: DataLoader,
    device: torch.device,
    max_samples: int = None
) -> Dict[str, float]:
    """
    Evaluate model using direct backbone method
    
    Args:
        evaluator: Direct evaluator
        data_loader: Data loader
        device: Computing device
        max_samples: Maximum evaluation samples (None for all)
        
    Returns:
        Evaluation results dictionary
    """
    evaluator.eval()
    
    all_predictions = []
    all_targets = []
    total_samples = 0
    total_loss = 0.0
    
    print(f"Starting direct backbone evaluation...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            # Parse batch data
            image_t1, a_hat_t1_to_t2_gt, at1_6dof_gt, at2_6dof_gt = batch
            
            # Move to GPU
            image_t1 = image_t1.to(device)
            at1_6dof_gt = at1_6dof_gt.to(device)  # Use at1_6dof as target
            
            batch_size = image_t1.shape[0]
            
            # Direct prediction (without transformer)
            predicted_action = evaluator(image_t1)
            
            # Calculate loss (using SmoothL1Loss consistent with training)
            loss = F.smooth_l1_loss(predicted_action, at1_6dof_gt)
            total_loss += loss.item() * batch_size
            
            # Collect predictions and targets
            all_predictions.append(predicted_action.cpu())
            all_targets.append(at1_6dof_gt.cpu())
            
            total_samples += batch_size
            
            # Check if maximum samples reached
            if max_samples and total_samples >= max_samples:
                break
                
            # Progress display
            if (batch_idx + 1) % 20 == 0:
                print(f"  Processed {total_samples} samples...")
                
            # Memory management
            if batch_idx % 50 == 0:
                torch.cuda.empty_cache()
    
    # Concatenate all predictions and targets
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Calculate various evaluation metrics
    results = {}
    
    # Average loss
    results['avg_loss'] = total_loss / total_samples
    
    # Overall MSE
    mse = F.mse_loss(all_predictions, all_targets)
    results['overall_mse'] = mse.item()
    
    # Per-dimension MSE
    per_dim_mse = torch.mean((all_predictions - all_targets) ** 2, dim=0)
    dim_names = ["X", "Y", "Z", "Roll", "Pitch", "Yaw"]
    for i, dim_name in enumerate(dim_names):
        results[f'mse_{dim_name.lower()}'] = per_dim_mse[i].item()
    
    # Per-dimension MAE
    per_dim_mae = torch.mean(torch.abs(all_predictions - all_targets), dim=0)
    for i, dim_name in enumerate(dim_names):
        results[f'mae_{dim_name.lower()}'] = per_dim_mae[i].item()
    
    # Overall MAE
    results['overall_mae'] = torch.mean(torch.abs(all_predictions - all_targets)).item()
    
    # Correlation coefficients
    correlations = []
    for i in range(6):
        pred_dim = all_predictions[:, i].numpy()
        target_dim = all_targets[:, i].numpy()
        corr = np.corrcoef(pred_dim, target_dim)[0, 1]
        correlations.append(corr)
        results[f'corr_{dim_names[i].lower()}'] = corr
    
    results['avg_correlation'] = np.mean(correlations)
    results['total_samples'] = total_samples
    
    print(f"Evaluation completed, processed {total_samples} samples")
    
    return results, all_predictions, all_targets


def create_evaluation_plots(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    results: Dict[str, float],
    output_dir: str
):
    """
    Create visualization plots for evaluation results
    
    Args:
        predictions: Prediction results
        targets: Ground truth targets
        results: Evaluation results
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    dim_names = ["X", "Y", "Z", "Roll", "Pitch", "Yaw"]
    preds_np = predictions.numpy()
    targets_np = targets.numpy()
    
    # Create combined scatter plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, dim_name in enumerate(dim_names):
        pred_dim = preds_np[:, i]
        target_dim = targets_np[:, i]
        
        axes[i].scatter(target_dim, pred_dim, alpha=0.6, s=15, color='blue', edgecolors='black', linewidth=0.3)
        
        # Perfect prediction line
        min_val = min(target_dim.min(), pred_dim.min())
        max_val = max(target_dim.max(), pred_dim.max())
        axes[i].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        
        # Calculate metrics
        mse = results[f'mse_{dim_name.lower()}']
        corr = results[f'corr_{dim_name.lower()}']
        
        axes[i].set_xlabel(f'Ground Truth {dim_name}', fontsize=10)
        axes[i].set_ylabel(f'Predicted {dim_name}', fontsize=10)
        axes[i].set_title(f'{dim_name}\nMSE: {mse:.4f}, Corr: {corr:.3f}', fontsize=11)
        axes[i].grid(True, alpha=0.3)
        axes[i].set_aspect('equal', adjustable='box')
    
    plt.suptitle(f'Direct Backbone Evaluation Results\n(Overall MSE: {results["overall_mse"]:.4f}, {results["total_samples"]} samples)', 
                 fontsize=16)
    plt.tight_layout()
    
    # Save combined plot
    combined_plot_path = os.path.join(output_dir, 'direct_backbone_evaluation_scatter.png')
    plt.savefig(combined_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create error distribution plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, dim_name in enumerate(dim_names):
        errors = preds_np[:, i] - targets_np[:, i]
        
        axes[i].hist(errors, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[i].axvline(0, color='red', linestyle='--', linewidth=2, label='Perfect Prediction')
        axes[i].axvline(np.mean(errors), color='orange', linestyle='-', linewidth=2, 
                       label=f'Mean Error: {np.mean(errors):.4f}')
        
        axes[i].set_xlabel(f'Prediction Error ({dim_name})', fontsize=10)
        axes[i].set_ylabel('Frequency', fontsize=10)
        axes[i].set_title(f'{dim_name} Error Distribution\nStd: {np.std(errors):.4f}', fontsize=11)
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.suptitle('Direct Backbone Evaluation - Error Distributions', fontsize=16)
    plt.tight_layout()
    
    # Save error distribution plot
    error_plot_path = os.path.join(output_dir, 'direct_backbone_evaluation_errors.png')
    plt.savefig(error_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization plots saved to {output_dir}")


def print_evaluation_results(results: Dict[str, float]):
    """
    Print detailed evaluation results
    
    Args:
        results: Evaluation results dictionary
    """
    print("\n" + "="*80)
    print("Direct Backbone Evaluation Results")
    print("="*80)
    
    print(f"\nOverall Performance:")
    print(f"   Total samples: {results['total_samples']}")
    print(f"   Average loss: {results['avg_loss']:.6f}")
    print(f"   Overall MSE: {results['overall_mse']:.6f}")
    print(f"   Overall MAE: {results['overall_mae']:.6f}")
    print(f"   Average correlation: {results['avg_correlation']:.4f}")
    
    print(f"\nPer-dimension Detailed Results:")
    dim_names = ["X", "Y", "Z", "Roll", "Pitch", "Yaw"]
    print(f"{'Dimension':<10} {'MSE':<12} {'MAE':<12} {'Correlation':<12}")
    print("-" * 52)
    
    for dim_name in dim_names:
        mse = results[f'mse_{dim_name.lower()}']
        mae = results[f'mae_{dim_name.lower()}']
        corr = results[f'corr_{dim_name.lower()}']
        print(f"{dim_name:<10} {mse:<12.6f} {mae:<12.6f} {corr:<12.4f}")


def save_evaluation_results(results: Dict[str, float], output_path: str):
    """
    Save evaluation results to JSON file
    
    Args:
        results: Evaluation results dictionary
        output_path: Output file path
    """
    # Convert to serializable format
    serializable_results = {}
    for key, value in results.items():
        if isinstance(value, (int, float)):
            serializable_results[key] = float(value)
        else:
            serializable_results[key] = value
    
    # Add metadata
    serializable_results['evaluation_type'] = 'direct_backbone'
    serializable_results['description'] = 'ResNet34 backbone + guidance layer (skip transformer dreamer)'
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    
    print(f"Evaluation results saved to {output_path}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Direct Backbone Evaluation")
    parser.add_argument("--model_path", type=str, required=True, help="Trained model checkpoint path")
    parser.add_argument("--data_dir", type=str, default="data/processed", help="Data directory")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"], help="Dataset split for evaluation")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum evaluation samples (None for all)")
    parser.add_argument("--output_dir", type=str, default="direct_backbone_evaluation_results", help="Results output directory")
    parser.add_argument("--device", type=str, default="auto", help="Computing device (auto/cpu/cuda)")
    
    args = parser.parse_args()
    
    # Set device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Check model file
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file does not exist: {args.model_path}")
    
    # Check data directory
    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(f"Data directory does not exist: {args.data_dir}")
    
    try:
        # Load trained components
        backbone, guidance = load_trained_components(args.model_path, device)
        
        # Create direct evaluator
        evaluator = DirectBackboneEvaluator(backbone, guidance, device)
        
        # Prepare data loader
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        # Get patient splits
        train_patients, val_patients, test_patients = get_patient_splits(args.data_dir)
        
        # Create dataset
        dataset = CrossPatientTransitionsDataset(
            data_dir=args.data_dir,
            transform=transform,
            split=args.split,
            train_patients=train_patients,
            val_patients=val_patients,
            test_patients=test_patients,
            small_subset=False
        )
        
        print(f"Loaded dataset: {args.split} split, {len(dataset)} samples")
        
        # Create data loader
        data_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # Execute evaluation
        results, predictions, targets = evaluate_direct_backbone(
            evaluator, data_loader, device, args.max_samples
        )
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Print results
        print_evaluation_results(results)
        
        # Create visualizations
        create_evaluation_plots(predictions, targets, results, args.output_dir)
        
        # Save results
        results_path = os.path.join(args.output_dir, "evaluation_results.json")
        save_evaluation_results(results, results_path)
        
        print(f"\nDirect Backbone Evaluation completed!")
        print(f"Results saved in: {args.output_dir}")
        print(f"   - evaluation_results.json: Detailed numerical results")
        print(f"   - direct_backbone_evaluation_scatter.png: Scatter plots")
        print(f"   - direct_backbone_evaluation_errors.png: Error distribution plots")
        
    except Exception as e:
        print(f"Error occurred during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 