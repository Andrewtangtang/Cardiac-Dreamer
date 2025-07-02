#!/usr/bin/env python3
"""
Video-Specific Prediction Generator for Direct Backbone Evaluation

This script generates predictions for individual video sequences using trained models
from cross-validation, creating CSV files matching the demo patient format.

Usage:
python generate_video_predictions.py --cv_output_dir outputs/cross_validation_20250609_212836 --data_dir data/processed
"""

import os
import sys
import json
import argparse
import warnings
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), "."))

from src.data.dataset import CrossPatientTransitionsDataset
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


def discover_all_video_folders(data_dir: str) -> List[str]:
    """
    Discover all video folders in the data directory
    
    Args:
        data_dir: Data directory path
        
    Returns:
        List of video folder names
    """
    print("Discovering all video folders...")
    
    video_folders = []
    
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")
    
    for item in os.listdir(data_dir):
        item_path = os.path.join(data_dir, item)
        if os.path.isdir(item_path) and item.startswith('data_'):
            video_folders.append(item)
    
    video_folders.sort()
    
    print(f"Found {len(video_folders)} video folders:")
    for folder in video_folders:
        print(f"  - {folder}")
    
    return video_folders


def get_patient_to_video_mapping(data_dir: str) -> Dict[str, List[str]]:
    """
    Create mapping from patient groups to video folders
    
    Args:
        data_dir: Data directory path
        
    Returns:
        Dictionary mapping patient groups to list of video folders
    """
    from src.data import get_patient_groups
    
    # Get patient groups mapping
    patient_groups = get_patient_groups()
    
    # Create reverse mapping: video -> patient_group
    video_to_patient = {}
    patient_to_videos = {}
    
    # Initialize patient_to_videos
    for patient_group in patient_groups.keys():
        patient_to_videos[patient_group] = []
    
    # Discover all video folders
    video_folders = discover_all_video_folders(data_dir)
    
    # Map each video to patient group
    for video_folder in video_folders:
        # Use the complete folder name for matching (e.g., data_0513_07)
        video_id = video_folder  # Keep the full folder name
        
        # Find which patient group this video belongs to
        found_group = None
        for patient_group, patient_list in patient_groups.items():
            if video_id in patient_list:
                found_group = patient_group
                break
        
        if found_group:
            patient_to_videos[found_group].append(video_folder)
            video_to_patient[video_folder] = found_group
            print(f"  {video_folder} -> {found_group}")
        else:
            print(f"  Warning: {video_folder} not found in any patient group")
    
    return patient_to_videos, video_to_patient


def discover_patient_models(cv_output_dir: str) -> Dict[str, Dict[str, str]]:
    """
    Discover all patient models from CV output directory
    
    Args:
        cv_output_dir: Cross-validation output directory
        
    Returns:
        Dictionary mapping patient groups to model information
    """
    print("Discovering patient models from CV results...")
    
    patient_models = {}
    
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
                if ckpt_file.endswith('.ckpt') and 'val_main_task_loss' in ckpt_file:
                    # Find the checkpoint with lowest validation loss
                    if best_checkpoint is None:
                        best_checkpoint = os.path.join(checkpoint_dir, ckpt_file)
                    else:
                        # Compare validation losses in filename
                        current_loss = float(ckpt_file.split('val_main_task_loss=')[1].split('.ckpt')[0])
                        best_loss = float(os.path.basename(best_checkpoint).split('val_main_task_loss=')[1].split('.ckpt')[0])
                        if current_loss < best_loss:
                            best_checkpoint = os.path.join(checkpoint_dir, ckpt_file)
            
            # If no validation loss checkpoint, look for any checkpoint
            if not best_checkpoint:
                for ckpt_file in os.listdir(checkpoint_dir):
                    if ckpt_file.endswith('.ckpt') and ckpt_file != 'last.ckpt':
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
                
                patient_models[patient_group] = {
                    'fold_num': fold_num,
                    'fold_dir': fold_path,
                    'checkpoint_path': best_checkpoint,
                    'fold_name': item
                }
                
                print(f"  Found model for {patient_group}: {os.path.basename(best_checkpoint)}")
            else:
                print(f"  Warning: No checkpoint found in {item}")
    
    return patient_models


def load_patient_model_components(checkpoint_path: str, device: torch.device) -> Tuple[nn.Module, nn.Module]:
    """
    Load backbone and guidance components from patient's validation model
    """
    print(f"    Loading model from: {os.path.basename(checkpoint_path)}")
    
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


def compute_fold_normalization_stats(
    data_dir: str,
    train_patients: List[str]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute normalization statistics for a specific fold's training data
    
    Args:
        data_dir: Data directory
        train_patients: List of training patient IDs for this fold
        
    Returns:
        Tuple of (action_mean, action_std) as numpy arrays
    """
    print(f"    Computing fold-specific normalization statistics...")
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
        split="train",  # Use train split to get training data
        train_patients=train_patients,
        val_patients=[],  # Empty validation
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


def load_patient_normalization_stats(fold_dir: str, train_patients: List[str], data_dir: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Compute normalization statistics for the patient's model dynamically
    """
    print(f"    Computing normalization statistics dynamically...")
    
    # Directly compute normalization stats from training data
    action_mean, action_std = compute_fold_normalization_stats(data_dir, train_patients)
    
    print(f"    Successfully computed fold-specific normalization stats")
    return action_mean, action_std


def create_video_dataset(
    data_dir: str,
    video_folder: str,
    batch_size: int = 1
) -> DataLoader:
    """
    Create dataset for specific video folder
    """
    print(f"    Creating dataset for video: {video_folder}")
    
    # Use the complete folder name as video ID (e.g., data_0513_07)
    video_id = video_folder  # Keep the full folder name
    
    # Create transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Create dataset with this specific video as validation
    dataset = CrossPatientTransitionsDataset(
        data_dir=data_dir,
        transform=transform,
        split="val",
        train_patients=[],  # Empty train patients
        val_patients=[video_id],  # This specific video as validation
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


def generate_video_predictions(
    predictor: DirectBackbonePredictor,
    data_loader: DataLoader,
    device: torch.device,
    action_mean: Optional[np.ndarray] = None,
    action_std: Optional[np.ndarray] = None
) -> pd.DataFrame:
    """
    Generate frame-by-frame predictions for video
    """
    predictor.eval()
    
    results = []
    
    print(f"    Generating predictions for {len(data_loader)} batches...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            if batch_idx % 100 == 0 and batch_idx > 0:
                print(f"      Processing batch {batch_idx}/{len(data_loader)}")
            
            # Parse batch data
            image_t1, a_hat_t1_to_t2_gt, at1_6dof_gt, at2_6dof_gt = batch
            
            # Move to GPU
            image_t1 = image_t1.to(device)
            
            # Generate prediction
            predicted_action = predictor(image_t1)
            
            # Convert to CPU numpy
            predicted_action_np = predicted_action.cpu().numpy()
            
            # Process each sample in batch
            batch_size = image_t1.shape[0]
            for i in range(batch_size):
                # Get frame information
                sample_idx = batch_idx * data_loader.batch_size + i
                
                # Try to get frame ID from dataset
                if hasattr(data_loader.dataset, 'transitions') and sample_idx < len(data_loader.dataset.transitions):
                    transition = data_loader.dataset.transitions[sample_idx]
                    frame_id = transition.get('id', f'frame_{sample_idx:06d}')
                    
                    # Get ORIGINAL ground truth from transition data (NOT normalized)
                    gt_original = np.array(transition["at1_6dof"])  # This is in original physical units
                    
                else:
                    frame_id = f'frame_{sample_idx:06d}'
                    # Fallback: use normalized gt (this shouldn't happen normally)
                    at1_6dof_gt_np = at1_6dof_gt.numpy()
                    gt_original = at1_6dof_gt_np[i]
                    print(f"Warning: Using normalized GT for {frame_id}")
                
                # Get predictions (normalized) - ONLY denormalize predictions
                pred_normalized = predicted_action_np[i]
                
                # Denormalize ONLY predictions if statistics are available
                if action_mean is not None and action_std is not None:
                    pred_denormalized = denormalize_actions(
                        pred_normalized.reshape(1, -1), action_mean, action_std
                    )[0]
                else:
                    # If no normalization stats, use normalized values
                    pred_denormalized = pred_normalized
                    print(f"Warning: No denormalization for prediction of {frame_id}")
                
                # Ground truth is ALREADY in original physical units - DO NOT denormalize!
                gt_denormalized = gt_original
                
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
    Save summary of all video results
    """
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_videos': len(all_results),
        'results_summary': {}
    }
    
    total_frames = 0
    for video_folder, result_info in all_results.items():
        frames_count = result_info['frames_count']
        total_frames += frames_count
        
        summary['results_summary'][video_folder] = {
            'frames_count': frames_count,
            'output_file': result_info['output_file'],
            'patient_group': result_info['patient_group'],
            'fold_name': result_info['fold_name'],
            'has_denormalization': result_info['has_denormalization']
        }
    
    summary['total_frames_processed'] = total_frames
    
    # Save summary JSON
    summary_path = os.path.join(output_dir, 'video_prediction_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nVideo prediction summary saved to: {summary_path}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Generate Video-Specific Predictions")
    parser.add_argument("--cv_output_dir", type=str, required=True, 
                       help="Cross-validation output directory containing fold results")
    parser.add_argument("--data_dir", type=str, default="data/processed", 
                       help="Data directory")
    parser.add_argument("--output_dir", type=str, default="video_predictions", 
                       help="Output directory for prediction results")
    parser.add_argument("--batch_size", type=int, default=8, 
                       help="Batch size for prediction")
    parser.add_argument("--device", type=str, default="auto", 
                       help="Computing device (auto/cpu/cuda)")
    parser.add_argument("--skip_existing", action="store_true",
                       help="Skip videos that already have prediction files")
    
    args = parser.parse_args()
    
    print("="*80)
    print("VIDEO-SPECIFIC PREDICTION GENERATOR")
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
        # Discover patient models
        patient_models = discover_patient_models(args.cv_output_dir)
        
        if not patient_models:
            raise ValueError(f"No valid patient models found in {args.cv_output_dir}")
        
        # Create patient to video mapping
        patient_to_videos, video_to_patient = get_patient_to_video_mapping(args.data_dir)
        
        print(f"\nPatient to Video Mapping:")
        for patient_group, videos in patient_to_videos.items():
            print(f"  {patient_group}: {len(videos)} videos")
        print()
        
        # Process each video
        all_results = {}
        total_videos = sum(len(videos) for videos in patient_to_videos.values())
        processed_count = 0
        
        for patient_group, videos in patient_to_videos.items():
            if patient_group not in patient_models:
                print(f"Warning: No model found for patient group {patient_group}")
                continue
            
            model_info = patient_models[patient_group]
            
            # Get training patients for this fold (reconstruct from fold statistics)
            from src.data import get_patient_groups
            patient_groups = get_patient_groups()
            
            # Get all patients and remove validation group to get training patients
            all_patients = []
            for group_name, patients in patient_groups.items():
                all_patients.extend(patients)
            
            val_patients = patient_groups[patient_group]
            train_patients = [p for p in all_patients if p not in val_patients]
            
            print(f"Loading model for {patient_group}...")
            print(f"  Training patients: {len(train_patients)} patients")
            print(f"  Validation patients: {val_patients}")
            
            # Load model components once for this patient group
            backbone, guidance = load_patient_model_components(model_info['checkpoint_path'], device)
            
            # Compute normalization statistics dynamically
            action_mean, action_std = load_patient_normalization_stats(
                model_info['fold_dir'], train_patients, args.data_dir
            )
            
            predictor = DirectBackbonePredictor(backbone, guidance, device)
            
            for video_folder in videos:
                processed_count += 1
                print(f"Processing {processed_count}/{total_videos}: {video_folder} (using {patient_group} model)")
                
                # Check if output file already exists
                output_filename = f"{video_folder}_predictions.csv"
                output_path = os.path.join(args.output_dir, output_filename)
                
                if args.skip_existing and os.path.exists(output_path):
                    print(f"  Skipping {video_folder} - output file already exists")
                    continue
                
                try:
                    # Create dataset for this video
                    data_loader = create_video_dataset(
                        args.data_dir, video_folder, args.batch_size
                    )
                    
                    if len(data_loader) == 0:
                        print(f"  Warning: No data found for {video_folder}")
                        continue
                    
                    # Generate predictions
                    predictions_df = generate_video_predictions(
                        predictor, data_loader, device, action_mean, action_std
                    )
                    
                    # Save predictions
                    predictions_df.to_csv(output_path, index=False)
                    
                    print(f"  Saved: {output_filename} ({len(predictions_df)} frames)")
                    
                    # Store results info
                    all_results[video_folder] = {
                        'frames_count': len(predictions_df),
                        'output_file': output_filename,
                        'patient_group': patient_group,
                        'fold_name': model_info['fold_name'],
                        'has_denormalization': action_mean is not None
                    }
                    
                except Exception as e:
                    print(f"  Error processing {video_folder}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
                
                print()
            
            # Clean up GPU memory after processing all videos for this patient group
            del backbone, guidance, predictor
            torch.cuda.empty_cache()
        
        # Save batch summary
        if all_results:
            save_batch_summary(all_results, args.output_dir)
            
            print("="*80)
            print("VIDEO PREDICTION GENERATION COMPLETED!")
            print("="*80)
            print(f"Processed {len(all_results)} videos")
            print(f"Total frames: {sum(r['frames_count'] for r in all_results.values())}")
            print(f"Results saved in: {args.output_dir}")
            
            # List some example generated files
            print("\nExample generated files:")
            count = 0
            for video_folder, result_info in all_results.items():
                if count < 5:  # Show first 5 examples
                    status = "(with denormalization)" if result_info['has_denormalization'] else "(normalized values)"
                    print(f"  - {result_info['output_file']} ({result_info['frames_count']} frames) {status}")
                    count += 1
            
            if len(all_results) > 5:
                print(f"  ... and {len(all_results) - 5} more files")
        else:
            print("No predictions were generated successfully.")
        
    except Exception as e:
        print(f"Error occurred during video prediction generation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 