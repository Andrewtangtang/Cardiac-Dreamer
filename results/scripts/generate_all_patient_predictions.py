#!/usr/bin/env python
"""
Batch Patient Prediction Generator

This script generates frame-by-frame predictions for ALL patients using 
their corresponding validation models (ResNet34 + Guidance Layer).

Usage:
python generate_all_patient_predictions.py --cv_output_dir outputs --data_dir data/processed
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from datetime import datetime

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from src.data import CrossPatientTransitionsDataset
from src.models.system import CardiacDreamerSystem
from model_evaluation import denormalize_actions


class DirectBackbonePredictor(nn.Module):
    """
    Direct Predictor: ResNet34 Backbone + Guidance Layer
    """
    
    def __init__(self, backbone: nn.Module, guidance: nn.Module, device: torch.device = None):
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
        # ResNet34 backbone: [B, 1, 224, 224] -> [B, 512, 7, 7]
        feature_map = self.backbone(image_t1)
        
        # Global average pooling: [B, 512, 7, 7] -> [B, 512]
        pooled_features = torch.mean(feature_map.view(feature_map.size(0), 512, -1), dim=2)
        
        # Guidance layer: [B, 512] -> [B, 6]
        predicted_action = self.guidance(pooled_features)
        
        return predicted_action


def discover_all_patient_groups(cv_output_dir: str) -> List[Dict[str, str]]:
    """
    Discover all patient groups from CV output directory
    
    Args:
        cv_output_dir: Cross-validation output directory
        
    Returns:
        List of dictionaries with fold and patient group information
    """
    print("Discovering all patient groups from CV results...")
    
    patient_folds = []
    
    # Look for fold directories
    for item in os.listdir(cv_output_dir):
        fold_path = os.path.join(cv_output_dir, item)
        if os.path.isdir(fold_path) and item.startswith('fold_'):
            
            # Look for checkpoint files
            checkpoint_dir = os.path.join(fold_path, 'checkpoints')
            if not os.path.exists(checkpoint_dir):
                print(f"  Warning: No checkpoints directory found in {item}")
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
                # Extract fold number and patient group
                try:
                    fold_num = int(item.split('_')[1])
                except (IndexError, ValueError):
                    fold_num = 0
                
                # Infer patient group from folder name
                if '_patient' in item:
                    parts = item.split('_')
                    for part in parts:
                        if part.startswith('patient'):
                            patient_group = part
                            break
                    else:
                        patient_group = f'patient{fold_num}'
                else:
                    patient_group = f'patient{fold_num}'
                
                patient_folds.append({
                    'fold_num': fold_num,
                    'fold_dir': fold_path,
                    'checkpoint_path': best_checkpoint,
                    'fold_name': item,
                    'patient_group': patient_group
                })
                
                print(f"  Found: {item} -> Patient Group: {patient_group}")
            else:
                print(f"  Warning: No checkpoint found in {item}")
    
    # Sort by fold number
    patient_folds.sort(key=lambda x: x['fold_num'])
    
    print(f"Discovered {len(patient_folds)} patient groups")
    return patient_folds


def load_patient_model_components(checkpoint_path: str, device: torch.device) -> Tuple[nn.Module, nn.Module]:
    """
    Load backbone and guidance components from patient's validation model
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Computing device
        
    Returns:
        Tuple of (backbone, guidance)
    """
    print(f"    Loading model from: {os.path.basename(checkpoint_path)}")
    
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
    
    return backbone, guidance


def load_patient_normalization_stats(fold_dir: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Load normalization statistics for the patient's model
    
    Args:
        fold_dir: Fold directory path
        
    Returns:
        Tuple of (action_mean, action_std) or (None, None) if not found
    """
    # Try to find normalization stats
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
                        action_mean = np.array(norm_stats["action_mean"])
                        action_std = np.array(norm_stats["action_std"])
                        print(f"    Loaded normalization stats from {os.path.basename(norm_path)}")
                        return action_mean, action_std
                
            except Exception as e:
                print(f"    Warning: Error reading {norm_path}: {e}")
    
    print(f"    Warning: No normalization stats found for {fold_dir}")
    return None, None


def create_patient_group_dataset(
    data_dir: str,
    patient_group: str,
    batch_size: int = 1
) -> DataLoader:
    """
    Create dataset for specific patient group
    
    Args:
        data_dir: Main data directory
        patient_group: Patient group name
        batch_size: Batch size
        
    Returns:
        DataLoader for the patient group
    """
    from src.data import get_patient_groups
    
    # Get patient groups
    patient_groups = get_patient_groups()
    
    # Find the actual patient list for this patient_group
    target_patients = patient_groups.get(patient_group, None)
    
    if target_patients is None:
        raise ValueError(f"Could not find patients for group {patient_group}")
    
    print(f"    Creating dataset for patients: {target_patients}")
    
    # Create transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Create dataset - we want all data for these patients, so we'll use them as validation
    dataset = CrossPatientTransitionsDataset(
        data_dir=data_dir,
        transform=transform,
        split="val",
        train_patients=[],  # Empty train patients
        val_patients=target_patients,  # Our target patients as validation
        test_patients=[],
        small_subset=False,
        normalize_actions=True
    )
    
    print(f"    Dataset created with {len(dataset)} samples")
    
    # Create data loader
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # Keep order for frame tracking
        num_workers=2,
        pin_memory=True
    )
    
    return data_loader


def generate_patient_group_predictions(
    predictor: DirectBackbonePredictor,
    data_loader: DataLoader,
    device: torch.device,
    action_mean: Optional[np.ndarray] = None,
    action_std: Optional[np.ndarray] = None
) -> pd.DataFrame:
    """
    Generate frame-by-frame predictions for patient group
    
    Args:
        predictor: Direct predictor model
        data_loader: Data loader
        device: Computing device
        action_mean: Normalization mean for denormalization
        action_std: Normalization std for denormalization
        
    Returns:
        DataFrame with frame-by-frame predictions
    """
    predictor.eval()
    
    results = []
    
    print(f"    Generating predictions for {len(data_loader)} batches...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            if batch_idx % 200 == 0 and batch_idx > 0:
                print(f"      Processing batch {batch_idx}/{len(data_loader)}")
            
            # Parse batch data
            image_t1, a_hat_t1_to_t2_gt, at1_6dof_gt, at2_6dof_gt = batch
            
            # Move to GPU
            image_t1 = image_t1.to(device)
            
            # Generate prediction
            predicted_action = predictor(image_t1)
            
            # Convert to CPU numpy
            predicted_action_np = predicted_action.cpu().numpy()
            at1_6dof_gt_np = at1_6dof_gt.numpy()
            
            # Process each sample in batch
            batch_size = image_t1.shape[0]
            for i in range(batch_size):
                # Get frame information
                sample_idx = batch_idx * data_loader.batch_size + i
                
                # Try to get frame ID from dataset
                if hasattr(data_loader.dataset, 'transitions') and sample_idx < len(data_loader.dataset.transitions):
                    transition = data_loader.dataset.transitions[sample_idx]
                    frame_id = transition.get('id', f'frame_{sample_idx:06d}')
                else:
                    frame_id = f'frame_{sample_idx:06d}'
                
                # Get predictions and ground truth (normalized)
                pred_normalized = predicted_action_np[i]
                gt_normalized = at1_6dof_gt_np[i]
                
                # Denormalize if statistics are available
                if action_mean is not None and action_std is not None:
                    pred_denormalized = denormalize_actions(
                        pred_normalized.reshape(1, -1), action_mean, action_std
                    )[0]
                    gt_denormalized = denormalize_actions(
                        gt_normalized.reshape(1, -1), action_mean, action_std
                    )[0]
                else:
                    # If no normalization stats, use normalized values
                    pred_denormalized = pred_normalized
                    gt_denormalized = gt_normalized
                
                # Create result entry
                result = {
                    'frame_t1_id': frame_id,
                    'gt_x': gt_denormalized[0],
                    'gt_y': gt_denormalized[1], 
                    'gt_z': gt_denormalized[2],
                    'gt_roll': gt_denormalized[3],
                    'gt_pitch': gt_denormalized[4],
                    'gt_yaw': gt_denormalized[5],
                    'pred_x': pred_denormalized[0],
                    'pred_y': pred_denormalized[1],
                    'pred_z': pred_denormalized[2],
                    'pred_roll': pred_denormalized[3],
                    'pred_pitch': pred_denormalized[4],
                    'pred_yaw': pred_denormalized[5]
                }
                
                results.append(result)
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    print(f"    Generated predictions for {len(df)} frames")
    
    return df


def save_batch_summary(all_results: Dict[str, Dict], output_dir: str):
    """
    Save summary of all patient group results
    
    Args:
        all_results: Dictionary with results for all patient groups
        output_dir: Output directory
    """
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_patient_groups': len(all_results),
        'results_summary': {}
    }
    
    total_frames = 0
    for patient_group, result_info in all_results.items():
        frames_count = result_info['frames_count']
        total_frames += frames_count
        
        summary['results_summary'][patient_group] = {
            'frames_count': frames_count,
            'output_file': result_info['output_file'],
            'fold_num': result_info['fold_num'],
            'fold_name': result_info['fold_name'],
            'has_denormalization': result_info['has_denormalization']
        }
    
    summary['total_frames_processed'] = total_frames
    
    # Save summary JSON
    summary_path = os.path.join(output_dir, 'batch_prediction_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nBatch summary saved to: {summary_path}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Generate All Patient Predictions")
    parser.add_argument("--cv_output_dir", type=str, required=True, 
                       help="Cross-validation output directory containing fold results")
    parser.add_argument("--data_dir", type=str, default="data/processed", 
                       help="Data directory")
    parser.add_argument("--output_dir", type=str, default="all_patient_predictions", 
                       help="Output directory for prediction results")
    parser.add_argument("--batch_size", type=int, default=8, 
                       help="Batch size for prediction")
    parser.add_argument("--device", type=str, default="auto", 
                       help="Computing device (auto/cpu/cuda)")
    parser.add_argument("--skip_existing", action="store_true",
                       help="Skip patient groups that already have prediction files")
    
    args = parser.parse_args()
    
    print("="*80)
    print("BATCH PATIENT PREDICTION GENERATOR")
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
    print(f"Output directory: {args.output_dir}")
    print(f"Skip existing: {args.skip_existing}")
    print()
    
    # Check directories
    if not os.path.exists(args.cv_output_dir):
        raise FileNotFoundError(f"CV output directory does not exist: {args.cv_output_dir}")
    
    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(f"Data directory does not exist: {args.data_dir}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Discover all patient groups
        patient_folds = discover_all_patient_groups(args.cv_output_dir)
        
        if not patient_folds:
            raise ValueError(f"No valid patient groups found in {args.cv_output_dir}")
        
        print(f"\nProcessing {len(patient_folds)} patient groups...")
        print()
        
        all_results = {}
        
        for i, fold_info in enumerate(patient_folds):
            patient_group = fold_info['patient_group']
            fold_num = fold_info['fold_num']
            
            print(f"Processing {i+1}/{len(patient_folds)}: {patient_group} (Fold {fold_num})")
            
            # Check if output file already exists
            output_filename = f"{patient_group}_predictions.csv"
            output_path = os.path.join(args.output_dir, output_filename)
            
            if args.skip_existing and os.path.exists(output_path):
                print(f"  Skipping {patient_group} - output file already exists")
                continue
            
            try:
                # Load model components
                print(f"  Loading model components...")
                backbone, guidance = load_patient_model_components(fold_info['checkpoint_path'], device)
                
                # Load normalization statistics
                print(f"  Loading normalization statistics...")
                action_mean, action_std = load_patient_normalization_stats(fold_info['fold_dir'])
                
                # Create predictor
                predictor = DirectBackbonePredictor(backbone, guidance, device)
                
                # Create dataset
                print(f"  Creating patient dataset...")
                data_loader = create_patient_group_dataset(
                    args.data_dir, patient_group, args.batch_size
                )
                
                # Generate predictions
                print(f"  Generating frame-by-frame predictions...")
                predictions_df = generate_patient_group_predictions(
                    predictor, data_loader, device, action_mean, action_std
                )
                
                # Save predictions
                predictions_df.to_csv(output_path, index=False)
                
                print(f"  Saved: {output_filename} ({len(predictions_df)} frames)")
                
                # Store results info
                all_results[patient_group] = {
                    'frames_count': len(predictions_df),
                    'output_file': output_filename,
                    'fold_num': fold_num,
                    'fold_name': fold_info['fold_name'],
                    'has_denormalization': action_mean is not None
                }
                
                # Clean up GPU memory
                del backbone, guidance, predictor
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"  Error processing {patient_group}: {e}")
                import traceback
                traceback.print_exc()
                continue
            
            print()
        
        # Save batch summary
        if all_results:
            save_batch_summary(all_results, args.output_dir)
            
            print("="*80)
            print("BATCH PREDICTION GENERATION COMPLETED!")
            print("="*80)
            print(f"Processed {len(all_results)} patient groups")
            print(f"Total frames: {sum(r['frames_count'] for r in all_results.values())}")
            print(f"Results saved in: {args.output_dir}")
            
            # List all generated files
            print("\nGenerated files:")
            for patient_group, result_info in all_results.items():
                status = "(with denormalization)" if result_info['has_denormalization'] else "(normalized values)"
                print(f"  - {result_info['output_file']} ({result_info['frames_count']} frames) {status}")
        else:
            print("No predictions were generated successfully.")
        
    except Exception as e:
        print(f"Error occurred during batch prediction generation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 