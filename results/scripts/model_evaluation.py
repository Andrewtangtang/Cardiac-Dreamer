#!/usr/bin/env python
"""
Model Evaluation Module for Cardiac Dreamer

This module provides evaluation utilities for cross-validation MAE extraction,
including model loading, prediction generation, and metrics calculation.
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
import warnings
from typing import Tuple, Dict, Any
from torch.utils.data import DataLoader


def load_model_from_checkpoint(checkpoint_path: str):
    """
    Load CardiacDreamerSystem model from checkpoint
    
    Args:
        checkpoint_path: Path to the checkpoint file
        
    Returns:
        Loaded model in evaluation mode
    """
    try:
        print(f"Loading model from checkpoint: {os.path.basename(checkpoint_path)}")
        
        # Import here to avoid circular imports
        from src.models.system import CardiacDreamerSystem
        
        # Load model from checkpoint
        model = CardiacDreamerSystem.load_from_checkpoint(
            checkpoint_path,
            map_location='cpu',  # Load to CPU first, then move to device
            strict=False
        )
        
        # Set to evaluation mode
        model.eval()
        
        print(f"Model loaded successfully: {type(model).__name__}")
        return model
        
    except Exception as e:
        print(f"Error loading model from {checkpoint_path}: {e}")
        raise


def generate_predictions(model, data_loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate predictions using the loaded model
    
    Args:
        model: Loaded CardiacDreamerSystem model
        data_loader: DataLoader for validation data
        device: Computing device (GPU/CPU)
        
    Returns:
        Tuple of (predictions, ground_truth) as numpy arrays [N, 6]
    """
    model = model.to(device)
    model.eval()
    
    all_predictions = []
    all_ground_truth = []
    
    print(f"Generating predictions for {len(data_loader)} batches...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            try:
                # Parse batch data - format: (image_t1, a_hat_t1_to_t2_gt, at1_6dof_gt, at2_6dof_gt)
                image_t1, a_hat_t1_to_t2_gt, at1_6dof_gt, at2_6dof_gt = batch
                
                # Move to device
                image_t1 = image_t1.to(device)
                a_hat_t1_to_t2_gt = a_hat_t1_to_t2_gt.to(device)
                at1_6dof_gt = at1_6dof_gt.to(device)
                
                # Generate prediction using the model's forward method
                # CardiacDreamerSystem.forward(image_t1, a_hat_t1_to_t2_gt) returns a dict
                system_outputs = model(image_t1, a_hat_t1_to_t2_gt)
                
                # Extract the main prediction: predicted_action_composed
                predicted_action = system_outputs["predicted_action_composed"]
                
                # Collect predictions and ground truth
                all_predictions.append(predicted_action.cpu().numpy())
                all_ground_truth.append(at1_6dof_gt.cpu().numpy())
                
                if (batch_idx + 1) % 10 == 0:
                    print(f"  Processed {batch_idx + 1}/{len(data_loader)} batches")
                    
            except Exception as e:
                print(f"Error processing batch {batch_idx}: {e}")
                continue
    
    if not all_predictions:
        raise ValueError("No valid predictions generated!")
    
    # Concatenate all results
    predictions = np.vstack(all_predictions)
    ground_truth = np.vstack(all_ground_truth)
    
    print(f"Prediction generation completed: {predictions.shape[0]} samples")
    
    return predictions, ground_truth


def denormalize_actions(normalized_actions: np.ndarray, action_mean: np.ndarray, action_std: np.ndarray) -> np.ndarray:
    """
    Denormalize actions back to original scale
    
    Args:
        normalized_actions: Normalized action values [N, 6]
        action_mean: Mean values used for normalization [6]
        action_std: Standard deviation values used for normalization [6]
        
    Returns:
        Denormalized actions in original physical units [N, 6]
    """
    return normalized_actions * action_std + action_mean


def calculate_metrics(predictions: np.ndarray, ground_truth: np.ndarray, 
                     action_mean: np.ndarray = None, action_std: np.ndarray = None) -> Dict[str, Dict[str, Any]]:
    """
    Calculate comprehensive metrics for each dimension
    
    Args:
        predictions: Predicted values [N, 6] (normalized or denormalized)
        ground_truth: Ground truth values [N, 6] (normalized or denormalized)
        action_mean: Mean values for denormalization [6] (optional)
        action_std: Std values for denormalization [6] (optional)
        
    Returns:
        Dictionary with metrics for each dimension
    """
    print("Calculating metrics for each dimension...")
    
    if predictions.shape != ground_truth.shape:
        raise ValueError(f"Shape mismatch: predictions {predictions.shape} vs ground_truth {ground_truth.shape}")
    
    # Denormalize if normalization stats are provided
    if action_mean is not None and action_std is not None:
        print("Denormalizing predictions and ground truth to original physical units...")
        predictions_denorm = denormalize_actions(predictions, action_mean, action_std)
        ground_truth_denorm = denormalize_actions(ground_truth, action_mean, action_std)
        
        print(f"Original range (normalized): pred [{predictions.min():.3f}, {predictions.max():.3f}], gt [{ground_truth.min():.3f}, {ground_truth.max():.3f}]")
        print(f"Denormalized range: pred [{predictions_denorm.min():.3f}, {predictions_denorm.max():.3f}], gt [{ground_truth_denorm.min():.3f}, {ground_truth_denorm.max():.3f}]")
        
        # Use denormalized values for metrics calculation
        pred_data = predictions_denorm
        gt_data = ground_truth_denorm
        print("USING DENORMALIZED VALUES FOR MAE CALCULATION (original physical units)")
    else:
        print("WARNING: No normalization stats provided - calculating metrics on normalized values!")
        print("This will result in artificially small MAE values that don't reflect physical units.")
        pred_data = predictions
        gt_data = ground_truth
    
    # Define dimension names and units
    dimension_info = {
        'X': {'name': 'X Translation', 'unit': 'mm'},
        'Y': {'name': 'Y Translation', 'unit': 'mm'},
        'Z': {'name': 'Z Translation', 'unit': 'mm'},
        'Roll': {'name': 'Roll Rotation', 'unit': 'degrees'},
        'Pitch': {'name': 'Pitch Rotation', 'unit': 'degrees'},
        'Yaw': {'name': 'Yaw Rotation', 'unit': 'degrees'}
    }
    
    metrics = {}
    
    for i, (dim_key, dim_info) in enumerate(dimension_info.items()):
        pred_dim = pred_data[:, i]
        gt_dim = gt_data[:, i]
        
        # Calculate MAE (now in original physical units)
        mae = np.mean(np.abs(pred_dim - gt_dim))
        
        # Calculate MSE and RMSE
        mse = np.mean((pred_dim - gt_dim) ** 2)
        rmse = np.sqrt(mse)
        
        # Calculate R² score
        try:
            r2 = r2_score(gt_dim, pred_dim)
        except:
            r2 = float('nan')
        
        # Calculate Pearson correlation
        try:
            correlation, p_value = pearsonr(pred_dim, gt_dim)
            if np.isnan(correlation):
                correlation = 0.0
                p_value = 1.0
        except:
            correlation = 0.0
            p_value = 1.0
        
        # Calculate additional metrics
        mean_error = np.mean(pred_dim - gt_dim)  # Bias
        std_error = np.std(pred_dim - gt_dim)    # Standard deviation of errors
        
        # Calculate percentage metrics (avoid division by zero)
        gt_range = np.max(gt_dim) - np.min(gt_dim)
        mape = np.mean(np.abs((gt_dim - pred_dim) / (np.abs(gt_dim) + 1e-8))) * 100  # MAPE
        
        metrics[dim_key] = {
            'name': dim_info['name'],
            'unit': dim_info['unit'],
            'mae': float(mae),
            'mse': float(mse), 
            'rmse': float(rmse),
            'r2_score': float(r2),
            'correlation': float(correlation),
            'correlation_p_value': float(p_value),
            'mean_error': float(mean_error),
            'std_error': float(std_error),
            'mape': float(mape),
            'gt_mean': float(np.mean(gt_dim)),
            'gt_std': float(np.std(gt_dim)),
            'gt_range': float(gt_range),
            'pred_mean': float(np.mean(pred_dim)),
            'pred_std': float(np.std(pred_dim)),
            'sample_count': len(pred_dim)
        }
        
        print(f"  {dim_key}: MAE={mae:.3f} {dim_info['unit']}, R²={r2:.3f}, Corr={correlation:.3f}")
    
    # Calculate overall metrics
    overall_mae = np.mean([metrics[dim]['mae'] for dim in dimension_info.keys()])
    overall_r2 = np.mean([metrics[dim]['r2_score'] for dim in dimension_info.keys() if not np.isnan(metrics[dim]['r2_score'])])
    overall_correlation = np.mean([metrics[dim]['correlation'] for dim in dimension_info.keys()])
    
    metrics['Overall'] = {
        'name': 'Overall Average',
        'unit': 'mixed',
        'mae': float(overall_mae),
        'r2_score': float(overall_r2),
        'correlation': float(overall_correlation),
        'sample_count': pred_data.shape[0]
    }
    
    print(f"  Overall: MAE={overall_mae:.3f}, R²={overall_r2:.3f}, Corr={overall_correlation:.3f}")
    
    return metrics


def evaluate_model_on_dataset(model, dataset, device: torch.device, batch_size: int = 16) -> Dict[str, Any]:
    """
    Complete evaluation pipeline for a model on a dataset
    
    Args:
        model: Loaded model
        dataset: Dataset to evaluate on
        device: Computing device
        batch_size: Batch size for evaluation
        
    Returns:
        Complete evaluation results
    """
    from torch.utils.data import DataLoader
    
    # Create data loader
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Generate predictions
    predictions, ground_truth = generate_predictions(model, data_loader, device)
    
    # Calculate metrics
    metrics = calculate_metrics(predictions, ground_truth)
    
    return {
        'metrics': metrics,
        'predictions': predictions,
        'ground_truth': ground_truth,
        'sample_count': len(dataset)
    }


def print_evaluation_summary(metrics: Dict[str, Dict[str, Any]]):
    """
    Print a formatted summary of evaluation metrics
    
    Args:
        metrics: Metrics dictionary from calculate_metrics
    """
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    
    # Print dimension-wise results
    dimensions = ['X', 'Y', 'Z', 'Roll', 'Pitch', 'Yaw']
    
    print(f"{'Dimension':<12} {'MAE':<10} {'RMSE':<10} {'R²':<8} {'Correlation':<12} {'Unit':<8}")
    print("-" * 70)
    
    for dim in dimensions:
        if dim in metrics:
            m = metrics[dim]
            print(f"{dim:<12} {m['mae']:<10.3f} {m['rmse']:<10.3f} {m['r2_score']:<8.3f} {m['correlation']:<12.3f} {m['unit']:<8}")
    
    if 'Overall' in metrics:
        print("-" * 70)
        m = metrics['Overall']
        print(f"{'Overall':<12} {m['mae']:<10.3f} {'N/A':<10} {m['r2_score']:<8.3f} {m['correlation']:<12.3f} {'mixed':<8}")
    
    print("="*80)


# Backward compatibility aliases
def load_dreamer_model(checkpoint_path: str):
    """Alias for load_model_from_checkpoint for backward compatibility"""
    return load_model_from_checkpoint(checkpoint_path)


def evaluate_dreamer_model(model, data_loader, device):
    """Alias for generate_predictions for backward compatibility"""
    return generate_predictions(model, data_loader, device) 