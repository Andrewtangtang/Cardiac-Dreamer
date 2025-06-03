"""
Patient splitting utilities for cardiac dreamer training
"""

import os
import json
import random
import numpy as np
from typing import Dict, List, Tuple


def get_patient_splits(data_dir: str, shuffle_patients: bool = True, balance_by_samples: bool = True, random_seed: int = 40) -> Tuple[List[str], List[str], List[str]]:
    """
    Automatically detect patient folders and create train/val/test splits
    
    Args:
        data_dir: Root data directory
        shuffle_patients: Whether to shuffle patients before splitting (recommended)
        balance_by_samples: Whether to consider sample counts when splitting patients
        random_seed: Random seed for reproducible splits
        
    Returns:
        Tuple of (train_patients, val_patients, test_patients)
    """
    # Set random seed for reproducible splits
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Find all patient directories
    patient_dirs = []
    patient_sample_counts = {}
    
    for item in os.listdir(data_dir):
        patient_path = os.path.join(data_dir, item)
        if os.path.isdir(patient_path) and item.startswith("data_"):
            # Check if it contains the required JSON file
            json_file = os.path.join(patient_path, "transitions_dataset.json")
            if os.path.exists(json_file):
                patient_dirs.append(item)
                
                # Count samples for this patient if balance_by_samples is True
                if balance_by_samples:
                    try:
                        with open(json_file, 'r') as f:
                            transitions = json.load(f)
                        patient_sample_counts[item] = len(transitions)
                    except:
                        patient_sample_counts[item] = 0
    
    if not patient_dirs:
        raise ValueError(f"No valid patient directories found in {data_dir}")
    
    print(f"Found {len(patient_dirs)} patient directories: {patient_dirs}")
    
    if balance_by_samples:
        total_samples = sum(patient_sample_counts.values())
        print(f"Total samples across all patients: {total_samples}")
        print(f"Sample distribution per patient: {patient_sample_counts}")
    
    # Shuffle patients for randomness (unless explicitly disabled)
    if shuffle_patients:
        random.shuffle(patient_dirs)
        print(f"Shuffled patient order: {patient_dirs}")
    else:
        patient_dirs.sort()  # For consistent ordering
        print(f"Using sorted patient order: {patient_dirs}")
    
    # Split patients based on strategy
    if balance_by_samples and len(patient_dirs) >= 6:  # Only use smart splitting if we have enough patients
        # Smart splitting: try to balance sample counts across splits
        train_patients, val_patients, test_patients = _smart_patient_split(
            patient_dirs, patient_sample_counts
        )
    else:
        # Simple splitting: just divide patients by count
        train_patients, val_patients, test_patients = _simple_patient_split(patient_dirs)
    
    print(f"Patient splits:")
    print(f"  Train: {train_patients}")
    print(f"  Validation: {val_patients}")
    print(f"  Test: {test_patients}")
    
    # Print final sample distribution
    if balance_by_samples:
        train_samples = sum(patient_sample_counts.get(p, 0) for p in train_patients)
        val_samples = sum(patient_sample_counts.get(p, 0) for p in val_patients)
        test_samples = sum(patient_sample_counts.get(p, 0) for p in test_patients)
        total = train_samples + val_samples + test_samples
        
        print(f"Final sample distribution:")
        print(f"  Train: {train_samples} samples ({train_samples/total:.1%})")
        print(f"  Val: {val_samples} samples ({val_samples/total:.1%})")
        print(f"  Test: {test_samples} samples ({test_samples/total:.1%})")
    
    return train_patients, val_patients, test_patients


def _simple_patient_split(patient_dirs: List[str]) -> Tuple[List[str], List[str], List[str]]:
    """Simple patient splitting by count"""
    num_patients = len(patient_dirs)
    num_train = max(1, int(0.7 * num_patients))
    num_val = max(1, int(0.15 * num_patients))
    
    train_patients = patient_dirs[:num_train]
    val_patients = patient_dirs[num_train:num_train + num_val]
    test_patients = patient_dirs[num_train + num_val:]
    
    # Ensure we have at least one patient for each split
    if len(val_patients) == 0 and len(train_patients) > 1:
        val_patients = [train_patients.pop()]
    if len(test_patients) == 0 and len(train_patients) > 1:
        test_patients = [train_patients.pop()]
    
    return train_patients, val_patients, test_patients


def _smart_patient_split(patient_dirs: List[str], patient_sample_counts: Dict[str, int]) -> Tuple[List[str], List[str], List[str]]:
    """
    Smart patient splitting that tries to balance sample counts across splits
    Uses a greedy algorithm to approximate 70%/15%/15% sample distribution
    """
    total_samples = sum(patient_sample_counts.values())
    target_train = 0.7 * total_samples
    target_val = 0.15 * total_samples
    target_test = 0.15 * total_samples
    
    # Sort patients by sample count (descending) for better distribution
    sorted_patients = sorted(patient_dirs, key=lambda p: patient_sample_counts[p], reverse=True)
    
    train_patients = []
    val_patients = []
    test_patients = []
    
    train_samples = 0
    val_samples = 0
    test_samples = 0
    
    # Greedy assignment: assign each patient to the split that needs samples most
    for patient in sorted_patients:
        samples = patient_sample_counts[patient]
        
        # Calculate how much each split needs
        train_need = max(0, target_train - train_samples)
        val_need = max(0, target_val - val_samples)
        test_need = max(0, target_test - test_samples)
        
        # Assign to the split with highest need (but ensure minimum 1 patient per split)
        if train_need >= max(val_need, test_need) and (len(train_patients) == 0 or len(val_patients) > 0 and len(test_patients) > 0):
            train_patients.append(patient)
            train_samples += samples
        elif val_need >= test_need and (len(val_patients) == 0 or len(test_patients) > 0):
            val_patients.append(patient)
            val_samples += samples
        else:
            test_patients.append(patient)
            test_samples += samples
    
    # Ensure minimum one patient per split
    if len(val_patients) == 0 and len(train_patients) > 1:
        val_patients = [train_patients.pop()]
        val_samples += patient_sample_counts[val_patients[0]]
        train_samples -= patient_sample_counts[val_patients[0]]
    
    if len(test_patients) == 0 and len(train_patients) > 1:
        test_patients = [train_patients.pop()]
        test_samples += patient_sample_counts[test_patients[0]]
        train_samples -= patient_sample_counts[test_patients[0]]
    
    print(f"Smart splitting achieved:")
    print(f"  Target vs Actual: Train {target_train:.0f} vs {train_samples}, Val {target_val:.0f} vs {val_samples}, Test {target_test:.0f} vs {test_samples}")
    
    return train_patients, val_patients, test_patients


def get_custom_patient_splits_no_test(data_dir: str) -> Tuple[List[str], List[str], List[str]]:
    """
    Custom patient splitting: patients 1-5 as validation, rest as training, no test set
    
    Args:
        data_dir: Root data directory
        
    Returns:
        Tuple of (train_patients, val_patients, test_patients)
        Note: test_patients will be empty list
    """
    # Find all patient directories
    patient_dirs = []
    patient_sample_counts = {}
    
    for item in os.listdir(data_dir):
        patient_path = os.path.join(data_dir, item)
        if os.path.isdir(patient_path) and item.startswith("data_"):
            # Check if it contains the required JSON file
            json_file = os.path.join(patient_path, "transitions_dataset.json")
            if os.path.exists(json_file):
                patient_dirs.append(item)
                
                # Count samples for this patient
                try:
                    with open(json_file, 'r') as f:
                        transitions = json.load(f)
                    patient_sample_counts[item] = len(transitions)
                except:
                    patient_sample_counts[item] = 0
    
    if not patient_dirs:
        raise ValueError(f"No valid patient directories found in {data_dir}")
    
    print(f"Found {len(patient_dirs)} patient directories")
    
    # Sort patients to ensure consistent ordering
    patient_dirs.sort()
    
    # Create splits: patients 1-5 as validation
    val_patients = []
    train_patients = []
    test_patients = []  # Empty test set
    
    for patient in patient_dirs:
        # Extract patient number from name (e.g., "data_0513_01" -> 1)
        try:
            patient_num = int(patient.split('_')[-1])
            if 1 <= patient_num <= 5:
                val_patients.append(patient)
            else:
                train_patients.append(patient)
        except:
            # If we can't parse the number, put in training
            train_patients.append(patient)
    
    # Sort for consistency
    train_patients.sort()
    val_patients.sort()
    
    print(f"\nCustom patient splits (no test set):")
    print(f"  Train patients ({len(train_patients)}): {train_patients}")
    print(f"  Val patients ({len(val_patients)}): {val_patients}")
    print(f"  Test patients: None")
    
    # Print sample distribution
    train_samples = sum(patient_sample_counts.get(p, 0) for p in train_patients)
    val_samples = sum(patient_sample_counts.get(p, 0) for p in val_patients)
    total_samples = train_samples + val_samples
    
    print(f"\nSample distribution:")
    print(f"  Train: {train_samples} samples ({train_samples/total_samples:.1%})")
    print(f"  Val: {val_samples} samples ({val_samples/total_samples:.1%})")
    print(f"  Total: {total_samples} samples")
    
    return train_patients, val_patients, test_patients 