#!/usr/bin/env python
"""
Generate AT1_6DOF Predictions for Single Patient Demo
Extracts prediction vs ground truth data from one patient's video for demo visualization
Output: CSV file with ground truth and predicted at1_6dof values
"""

import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import json

# Add project root to path
current_file_dir = os.path.dirname(os.path.abspath(__file__))
if current_file_dir not in sys.path:
    sys.path.insert(0, current_file_dir)

# Import model and dataset
from src.models.system import get_cardiac_dreamer_system
from src.data import CrossPatientTransitionsDataset, get_custom_patient_splits_no_test


def load_model_from_checkpoint(checkpoint_path: str):
    """Load trained model from checkpoint"""
    print(f"Loading model from: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Try to get hyperparameters from checkpoint
    if 'hyper_parameters' in checkpoint:
        hparams = checkpoint['hyper_parameters']
        print(f"Found hyperparameters in checkpoint")
        
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
        print("No hyperparameters found, using default configuration")
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
        print(f"Loading with strict=False due to: {e}")
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    
    model.eval()
    return model


def list_available_patients(data_dir: str):
    """List available patient directories"""
    patients = []
    for item in os.listdir(data_dir):
        if item.startswith('data_0513_') and os.path.isdir(os.path.join(data_dir, item)):
            patients.append(item)
    return sorted(patients)


def create_single_patient_dataset(data_dir: str, patient_id: str, max_samples: int = None, take_last_n: int = None):
    """Create dataset for a specific patient"""
    print(f"Creating dataset for patient: {patient_id}")
    
    # Get patient splits to determine if this patient is in training set
    train_patients, val_patients, test_patients = get_custom_patient_splits_no_test(data_dir)
    
    # Check which split this patient belongs to
    patient_split = None
    if patient_id in train_patients:
        patient_split = "train"
        print(f"Patient {patient_id} is in TRAINING set")
    elif patient_id in val_patients:
        patient_split = "val"
        print(f"Patient {patient_id} is in VALIDATION set")
    else:
        print(f"Patient {patient_id} not found in any split!")
        return None
    
    # Set up transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Create dataset for the specific split
    dataset = CrossPatientTransitionsDataset(
        data_dir=data_dir,
        transform=transform,
        split=patient_split,
        train_patients=train_patients,
        val_patients=val_patients,
        test_patients=test_patients,
        small_subset=False,
        normalize_actions=True
    )
    
    # Filter samples to only include the target patient
    filtered_samples = []
    # Iterate through the dataset directly (it is iterable)
    for i in range(len(dataset)):
        sample = dataset[i] # Access sample by index
        # The sample itself is a tuple: (image_t1, a_hat_t1_to_t2_gt, at1_6dof_gt, at2_6dof_gt)
        # To get metadata like ft1_image_path, we need to access the original transition info
        # Assuming the dataset object stores the original transitions list that has this info
        # and the order is preserved, or there's a way to map index `i` to original transition data.
        # Let's assume dataset.transitions[i] holds the original dict with 'ft1_image_path'
        if hasattr(dataset, 'transitions') and i < len(dataset.transitions):
            transition_info = dataset.transitions[i]
            if patient_id in transition_info['ft1_image_path']:  # Check if sample belongs to target patient
                filtered_samples.append(transition_info) # Store the original transition dict
                if max_samples and len(filtered_samples) >= max_samples:
                    break
        else:
            # Fallback or error if original transition info isn't accessible as expected
            # This part might need adjustment based on how CrossPatientTransitionsDataset is structured
            # For now, let's assume this won't happen or we can't filter if this info is missing.
            # If we can't filter by path, we might take all samples if patient_id is in train/val list.
            # This is a placeholder for a more robust way to link iterated samples to their metadata.
            pass 

    # If we want to take only the last N samples, slice the list
    if take_last_n and len(filtered_samples) > take_last_n:
        print(f"Taking last {take_last_n} samples from {len(filtered_samples)} total samples")
        filtered_samples = filtered_samples[-take_last_n:]
    
    # If we successfully filtered, we need to create a *new* dataset instance 
    # or modify it in a way that it only uses these filtered_samples.
    # The current CrossPatientTransitionsDataset might not support re-assigning its internal list directly via `dataset.samples`.
    # A safer approach is to pass the filtered list of transition dicts to a new instance if possible, 
    # or adjust the __getitem__ and __len__ of the existing dataset to use this filtered list.
    # For simplicity, if `dataset.transitions` is the source, we modify it.
    if filtered_samples:
        dataset.transitions = filtered_samples # This assumes self.transitions is the primary source for __getitem__
        dataset.num_transitions = len(filtered_samples)
    else:
        # Handle case where no samples were found for the patient or filtering failed.
        print(f"Warning: No samples found or could not filter for patient {patient_id}. Using all samples from the determined split.")

    print(f"Dataset created: {len(dataset)} samples from patient {patient_id} (or all from split if filtering failed)")
    return dataset


def generate_predictions(model, dataset, device, batch_size=16):
    """Generate predictions for the dataset"""
    print(f"Generating predictions...")
    
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=False
    )
    
    model.eval()
    results = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            print(f"Processing batch {batch_idx + 1}/{len(data_loader)}")
            
            image_t1, a_hat_t1_to_t2_gt, at1_6dof_gt, at2_6dof_gt = batch
            
            # Move to device
            image_t1 = image_t1.to(device)
            a_hat_t1_to_t2_gt = a_hat_t1_to_t2_gt.to(device)
            at1_6dof_gt = at1_6dof_gt.to(device)
            
            # Forward pass
            outputs = model(image_t1, a_hat_t1_to_t2_gt)
            at1_pred = outputs['predicted_action_composed']  # This is our main prediction
            
            # Denormalize using dataset statistics
            if hasattr(dataset, 'action_mean') and hasattr(dataset, 'action_std'):
                action_mean = torch.tensor(dataset.action_mean, device=device)
                action_std = torch.tensor(dataset.action_std, device=device)
                
                # Denormalize both predictions and ground truth
                at1_pred_denorm = at1_pred * action_std + action_mean
                at1_gt_denorm = at1_6dof_gt * action_std + action_mean
            else:
                print("No normalization stats found, using raw values")
                at1_pred_denorm = at1_pred
                at1_gt_denorm = at1_6dof_gt
            
            # Convert to numpy and collect results
            batch_results = []
            batch_size_actual = image_t1.shape[0]
            
            for i in range(batch_size_actual):
                # Get sample index in the original dataset
                sample_idx = batch_idx * batch_size + i
                if sample_idx < len(dataset.transitions):
                    transition_info = dataset.transitions[sample_idx]
                    frame_id = os.path.basename(transition_info['ft1_image_path']).replace('.png', '')
                    
                    result = {
                        'frame_t1_id': frame_id,
                        'gt_x': float(at1_gt_denorm[i, 0].cpu().numpy()),
                        'gt_y': float(at1_gt_denorm[i, 1].cpu().numpy()),
                        'gt_z': float(at1_gt_denorm[i, 2].cpu().numpy()),
                        'gt_roll': float(at1_gt_denorm[i, 3].cpu().numpy()),
                        'gt_pitch': float(at1_gt_denorm[i, 4].cpu().numpy()),
                        'gt_yaw': float(at1_gt_denorm[i, 5].cpu().numpy()),
                        'pred_x': float(at1_pred_denorm[i, 0].cpu().numpy()),
                        'pred_y': float(at1_pred_denorm[i, 1].cpu().numpy()),
                        'pred_z': float(at1_pred_denorm[i, 2].cpu().numpy()),
                        'pred_roll': float(at1_pred_denorm[i, 3].cpu().numpy()),
                        'pred_pitch': float(at1_pred_denorm[i, 4].cpu().numpy()),
                        'pred_yaw': float(at1_pred_denorm[i, 5].cpu().numpy()),
                    }
                    batch_results.append(result)
            
            results.extend(batch_results)
    
    print(f"Generated predictions for {len(results)} samples")
    return results


def save_to_csv(results, output_file):
    """Save results to CSV file"""
    print(f"Saving results to: {output_file}")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Sort by frame_t1_id for consistent ordering
    df = df.sort_values('frame_t1_id')
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    
    # Print summary
    print(f"Saved {len(df)} predictions to {output_file}")
    print(f"Sample data preview:")
    print(df.head())
    
    # Calculate and print some basic stats
    if len(df) > 0:
        mse_x = ((df['pred_x'] - df['gt_x']) ** 2).mean()
        mse_y = ((df['pred_y'] - df['gt_y']) ** 2).mean()
        mse_z = ((df['pred_z'] - df['gt_z']) ** 2).mean()
        print(f"\nQuick MSE Preview:")
        print(f"   X: {mse_x:.4f}")
        print(f"   Y: {mse_y:.4f}")
        print(f"   Z: {mse_z:.4f}")
    else:
        print("No data to save!")


def process_single_patient(model, data_dir, patient_id, output_base_dir, device, batch_size=16, take_last_n=20):
    """Process a single patient and save results"""
    print(f"\n{'='*60}")
    print(f"Processing patient: {patient_id}")
    print(f"{'='*60}")
    
    try:
        # Create dataset for specific patient (take last 20 samples)
        dataset = create_single_patient_dataset(data_dir, patient_id, max_samples=None, take_last_n=take_last_n)
        if dataset is None or len(dataset) == 0:
            print(f"Skipping patient {patient_id}: No valid dataset created")
            return False
        
        # Generate predictions
        results = generate_predictions(model, dataset, device, batch_size)
        
        if not results:
            print(f"Skipping patient {patient_id}: No predictions generated")
            return False
        
        # Create output directory for this patient
        patient_output_dir = os.path.join(output_base_dir, patient_id)
        output_file = os.path.join(patient_output_dir, f"{patient_id}_at1_predictions.csv")
        
        # Save to CSV
        save_to_csv(results, output_file)
        
        print(f"Successfully processed patient {patient_id}")
        return True
        
    except Exception as e:
        print(f"Error processing patient {patient_id}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Generate AT1_6DOF predictions for all patients demo")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--data_dir", type=str, default="data/processed",
                       help="Data directory path")
    parser.add_argument("--patient_id", type=str, default=None,
                       help="Specific patient ID (e.g., data_0513_01). If not specified, will process ALL patients.")
    parser.add_argument("--output_base_dir", type=str, default="output/patient_predictions",
                       help="Base output directory (each patient will have a subdirectory)")
    parser.add_argument("--take_last_n", type=int, default=20,
                       help="Number of last samples to take from each patient's video data")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size for inference")
    
    args = parser.parse_args()
    
    print("AT1_6DOF Prediction Generation for All Patients Demo")
    print(f"Data directory: {args.data_dir}")
    print(f"Output base directory: {args.output_base_dir}")
    print(f"Taking last {args.take_last_n} samples per patient")
    
    # Validate checkpoint
    if not os.path.exists(args.checkpoint_path):
        print(f"Checkpoint file not found: {args.checkpoint_path}")
        return
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        # 1. Load model
        print("Loading model...")
        model = load_model_from_checkpoint(args.checkpoint_path)
        model = model.to(device)
        print("Model loaded successfully!")
        
        # 2. Get list of patients to process
        if args.patient_id:
            # Process specific patient
            patients_to_process = [args.patient_id]
            print(f"Processing specific patient: {args.patient_id}")
        else:
            # Process all available patients
            patients_to_process = list_available_patients(args.data_dir)
            print(f"Found {len(patients_to_process)} patients to process")
            print("Patients:", patients_to_process)
        
        if not patients_to_process:
            print("No patients found to process!")
            return
        
        # 3. Process each patient
        successful_patients = []
        failed_patients = []
        
        for i, patient_id in enumerate(patients_to_process):
            print(f"\nProgress: {i+1}/{len(patients_to_process)}")
            success = process_single_patient(
                model=model,
                data_dir=args.data_dir,
                patient_id=patient_id,
                output_base_dir=args.output_base_dir,
                device=device,
                batch_size=args.batch_size,
                take_last_n=args.take_last_n
            )
            
            if success:
                successful_patients.append(patient_id)
            else:
                failed_patients.append(patient_id)
        
        # 4. Summary
        print(f"\n{'='*80}")
        print("PROCESSING SUMMARY")
        print(f"{'='*80}")
        print(f"Total patients: {len(patients_to_process)}")
        print(f"Successfully processed: {len(successful_patients)}")
        print(f"Failed: {len(failed_patients)}")
        
        if successful_patients:
            print(f"\nSuccessful patients:")
            for patient in successful_patients:
                output_file = os.path.join(args.output_base_dir, patient, f"{patient}_at1_predictions.csv")
                print(f"  - {patient} -> {output_file}")
        
        if failed_patients:
            print(f"\nFailed patients:")
            for patient in failed_patients:
                print(f"  - {patient}")
        
        print(f"\nAll results saved to: {args.output_base_dir}")
        print(f"Each patient has its own subdirectory with {args.take_last_n} predictions.")
        print(f"Demo data generation completed!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 