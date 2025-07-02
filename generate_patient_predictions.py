#!/usr/bin/env python
"""
Patient-Specific Prediction Generator

This script generates frame-by-frame predictions for specific patients using 
their corresponding validation models (ResNet34 + Guidance Layer).

Usage:
python generate_patient_predictions.py --cv_output_dir outputs --data_dir data/processed --patient_id patient_07
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

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


def find_patient_validation_model(cv_output_dir: str, patient_id: str) -> Optional[Dict[str, str]]:
    """
    Find the validation model corresponding to a specific patient
    
    Args:
        cv_output_dir: Cross-validation output directory
        patient_id: Patient ID (e.g., 'patient_07')
        
    Returns:
        Dictionary with model information or None if not found
    """
    print(f"Looking for validation model for {patient_id}...")
    
    # Convert patient_id to expected group format
    if patient_id.startswith('patient_'):
        # Extract number and convert to group
        try:
            patient_num = int(patient_id.split('_')[1])
            patient_group = f'patient{patient_num}'
        except (IndexError, ValueError):
            patient_group = patient_id
    else:
        patient_group = patient_id
    
    print(f"  Looking for patient group: {patient_group}")
    
    # Look for fold directories
    for item in os.listdir(cv_output_dir):
        fold_path = os.path.join(cv_output_dir, item)
        if os.path.isdir(fold_path) and item.startswith('fold_'):
            
            # Check if this fold has our patient as validation
            if patient_group in item:
                print(f"  Found matching fold directory: {item}")
                
                # Look for checkpoint files
                checkpoint_dir = os.path.join(fold_path, 'checkpoints')
                if not os.path.exists(checkpoint_dir):
                    print(f"    Warning: No checkpoints directory found")
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
                    # Extract fold number
                    try:
                        fold_num = int(item.split('_')[1])
                    except (IndexError, ValueError):
                        fold_num = 0
                    
                    return {
                        'fold_num': fold_num,
                        'fold_dir': fold_path,
                        'checkpoint_path': best_checkpoint,
                        'fold_name': item,
                        'patient_group': patient_group
                    }
                else:
                    print(f"    Warning: No checkpoint found in {item}")
    
    print(f"  No validation model found for {patient_id}")
    return None


def load_patient_model_components(checkpoint_path: str, device: torch.device) -> Tuple[nn.Module, nn.Module]:
    """
    Load backbone and guidance components from patient's validation model
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Computing device
        
    Returns:
        Tuple of (backbone, guidance)
    """
    print(f"Loading model from: {os.path.basename(checkpoint_path)}")
    
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
    import json
    
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
                        print(f"  Loaded normalization stats from {norm_path}")
                        print(f"    Action mean: {action_mean}")
                        print(f"    Action std: {action_std}")
                        return action_mean, action_std
                
            except Exception as e:
                print(f"Warning: Error reading {norm_path}: {e}")
    
    print(f"Warning: No normalization stats found for {fold_dir}")
    return None, None


def create_patient_dataset(
    data_dir: str,
    patient_id: str,
    model_info: Dict[str, str],
    batch_size: int = 1
) -> DataLoader:
    """
    Create dataset for specific patient
    
    Args:
        data_dir: Main data directory
        patient_id: Patient ID
        model_info: Model information dictionary
        batch_size: Batch size
        
    Returns:
        DataLoader for the patient
    """
    from src.data import get_patient_groups
    
    # Get patient groups
    patient_groups = get_patient_groups()
    
    # Find the actual patient list for this patient_id
    target_patients = None
    for group_name, patients in patient_groups.items():
        if group_name == model_info['patient_group']:
            target_patients = patients
            break
    
    if target_patients is None:
        raise ValueError(f"Could not find patients for group {model_info['patient_group']}")
    
    print(f"Creating dataset for patients: {target_patients}")
    
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
    
    print(f"Dataset created with {len(dataset)} samples")
    
    # Create data loader
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # Keep order for frame tracking
        num_workers=2,
        pin_memory=True
    )
    
    return data_loader


def generate_patient_predictions(
    predictor: DirectBackbonePredictor,
    data_loader: DataLoader,
    device: torch.device,
    action_mean: Optional[np.ndarray] = None,
    action_std: Optional[np.ndarray] = None
) -> pd.DataFrame:
    """
    Generate frame-by-frame predictions for patient
    
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
    
    print(f"Generating predictions for {len(data_loader)} batches...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            if batch_idx % 100 == 0:
                print(f"  Processing batch {batch_idx}/{len(data_loader)}")
            
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
                    print("Warning: Using normalized values (no denormalization stats)")
                
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
    print(f"Generated predictions for {len(df)} frames")
    
    return df


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Generate Patient-Specific Predictions")
    parser.add_argument("--cv_output_dir", type=str, required=True, 
                       help="Cross-validation output directory containing fold results")
    parser.add_argument("--data_dir", type=str, default="data/processed", 
                       help="Data directory")
    parser.add_argument("--patient_id", type=str, required=True,
                       help="Patient ID (e.g., patient_07)")
    parser.add_argument("--output_dir", type=str, default="patient_predictions", 
                       help="Output directory for prediction results")
    parser.add_argument("--batch_size", type=int, default=8, 
                       help="Batch size for prediction")
    parser.add_argument("--device", type=str, default="auto", 
                       help="Computing device (auto/cpu/cuda)")
    
    args = parser.parse_args()
    
    print("="*80)
    print("PATIENT-SPECIFIC PREDICTION GENERATOR")
    print("="*80)
    print()
    
    # Set device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Patient ID: {args.patient_id}")
    print(f"Using device: {device}")
    print(f"CV output directory: {args.cv_output_dir}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    # Check directories
    if not os.path.exists(args.cv_output_dir):
        raise FileNotFoundError(f"CV output directory does not exist: {args.cv_output_dir}")
    
    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(f"Data directory does not exist: {args.data_dir}")
    
    try:
        # Find patient's validation model
        print("Finding patient's validation model...")
        model_info = find_patient_validation_model(args.cv_output_dir, args.patient_id)
        
        if model_info is None:
            raise ValueError(f"No validation model found for patient {args.patient_id}")
        
        print(f"Found validation model:")
        print(f"  Fold: {model_info['fold_num']}")
        print(f"  Fold directory: {model_info['fold_name']}")
        print(f"  Patient group: {model_info['patient_group']}")
        print(f"  Checkpoint: {os.path.basename(model_info['checkpoint_path'])}")
        print()
        
        # Load model components
        print("Loading model components...")
        backbone, guidance = load_patient_model_components(model_info['checkpoint_path'], device)
        
        # Load normalization statistics
        print("Loading normalization statistics...")
        action_mean, action_std = load_patient_normalization_stats(model_info['fold_dir'])
        
        # Create predictor
        predictor = DirectBackbonePredictor(backbone, guidance, device)
        
        # Create dataset
        print("Creating patient dataset...")
        data_loader = create_patient_dataset(
            args.data_dir, args.patient_id, model_info, args.batch_size
        )
        
        # Generate predictions
        print("Generating frame-by-frame predictions...")
        predictions_df = generate_patient_predictions(
            predictor, data_loader, device, action_mean, action_std
        )
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Save predictions
        output_filename = f"{args.patient_id}_predictions.csv"
        output_path = os.path.join(args.output_dir, output_filename)
        
        predictions_df.to_csv(output_path, index=False)
        
        print(f"\nPredictions saved to: {output_path}")
        print(f"Total frames processed: {len(predictions_df)}")
        
        # Print sample statistics
        if action_mean is not None:
            print("\nSample statistics (in physical units):")
            print("Translation (mm):")
            print(f"  GT X: {predictions_df['gt_x'].mean():.3f} ± {predictions_df['gt_x'].std():.3f}")
            print(f"  GT Y: {predictions_df['gt_y'].mean():.3f} ± {predictions_df['gt_y'].std():.3f}")
            print(f"  GT Z: {predictions_df['gt_z'].mean():.3f} ± {predictions_df['gt_z'].std():.3f}")
            print("Rotation (degrees):")
            print(f"  GT Roll: {predictions_df['gt_roll'].mean():.3f} ± {predictions_df['gt_roll'].std():.3f}")
            print(f"  GT Pitch: {predictions_df['gt_pitch'].mean():.3f} ± {predictions_df['gt_pitch'].std():.3f}")
            print(f"  GT Yaw: {predictions_df['gt_yaw'].mean():.3f} ± {predictions_df['gt_yaw'].std():.3f}")
        else:
            print("\nNote: Values are in normalized units (no denormalization applied)")
        
        print()
        print("="*80)
        print("PREDICTION GENERATION COMPLETED!")
        print("="*80)
        
    except Exception as e:
        print(f"Error occurred during prediction generation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 