"""
Dataset classes for cardiac dreamer training
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import Dict, List, Tuple, Optional
from .patient_splits import get_patient_splits


class CrossPatientTransitionsDataset(Dataset):
    """
    Cross-patient dataset for ultrasound image transitions with actions
    Reads from multiple patient folders containing transitions_dataset.json files
    
    Data structure expected:
    data/processed/
    data_0513_01/
    
    Each JSON contains:
    {
        "ft1_image_path": "path/to/image",
        "ft2_image_path": "path/to/image", 
        "at1_6dof": [x, y, z, roll, pitch, yaw],
        "action_change_6dof": [dx, dy, dz, droll, dpitch, dyaw],
        "at2_6dof": [x2, y2, z2, roll2, pitch2, yaw2]
    }
    
    Args:
        data_dir: Root data directory path (e.g., "data/processed")
        transform: Image transformations
        split: Data split ("train", "val", "test")
        train_patients: List of patient IDs for training (e.g., ["data_0513_01", "data_0513_02"])
        val_patients: List of patient IDs for validation (e.g., ["data_0513_03"])
        test_patients: List of patient IDs for testing (e.g., ["data_0513_04"])
        small_subset: Whether to use only a small subset of data (for testing)
        normalize_actions: Whether to normalize 6DOF actions (default: True)
    """
    def __init__(
        self, 
        data_dir: str, 
        transform=None, 
        split: str = "train",
        train_patients: Optional[List[str]] = None,
        val_patients: Optional[List[str]] = None,
        test_patients: Optional[List[str]] = None,
        small_subset: bool = False,
        normalize_actions: bool = True
    ):
        self.data_dir = data_dir
        self.transform = transform
        self.split = split
        self.small_subset = small_subset
        self.normalize_actions = normalize_actions
        
        # Automatically detect patient splits if not provided
        if train_patients is None or val_patients is None or test_patients is None:
            print(f"Patient splits not provided, auto-detecting from {data_dir}...")
            auto_train, auto_val, auto_test = get_patient_splits(data_dir)
            
            # Use auto-detected splits as defaults
            if train_patients is None:
                train_patients = auto_train
            if val_patients is None:
                val_patients = auto_val
            if test_patients is None:
                test_patients = auto_test
            
            print(f"Auto-detected patient splits:")
            print(f"  Train: {train_patients}")
            print(f"  Val: {val_patients}")
            print(f"  Test: {test_patients}")
            
        # Select patients based on split
        if split == "train":
            self.patient_ids = train_patients
        elif split == "val":
            self.patient_ids = val_patients
        elif split == "test":
            self.patient_ids = test_patients
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'")
        
        print(f"Loading {split} data from patients: {self.patient_ids}")
        
        # Load all transitions from selected patients
        self.transitions = []
        self.load_transitions()
        
        # Compute normalization statistics if enabled
        if self.normalize_actions and split == "train":
            self.compute_normalization_stats()
        elif self.normalize_actions:
            # For val/test, load the normalization stats from training data
            norm_file = os.path.join(self.data_dir, "normalization_stats.json")
            if os.path.exists(norm_file):
                print(f"📊 Loading normalization stats from: {norm_file}")
                with open(norm_file, 'r') as f:
                    norm_stats = json.load(f)
                
                self.action_mean = np.array(norm_stats["action_mean"])
                self.action_std = np.array(norm_stats["action_std"])
                print(f"✅ Successfully loaded normalization stats for {split} split")
            else:
                print(f"❌ Normalization stats file not found: {norm_file}")
                print(f"   Using default values (mean=0, std=1) - this may cause issues!")
                self.action_mean = np.array([0.0] * 6)
                self.action_std = np.array([1.0] * 6)
        
        # For small subset testing, limit samples
        if small_subset:
            num_samples = min(10, len(self.transitions))
            self.transitions = self.transitions[:num_samples]
            print(f"Using small subset for testing: {num_samples} samples")
            
        print(f"Loaded {len(self.transitions)} {split} transitions from {len(self.patient_ids)} patients")
        
        if self.normalize_actions:
            print(f"📊 Action normalization enabled:")
            print(f"   Mean: {self.action_mean}")
            print(f"   Std: {self.action_std}")
    
    def compute_normalization_stats(self):
        """Compute normalization statistics for 6DOF actions"""
        print("Computing normalization statistics for 6DOF actions...")
        
        # 
        if len(self.transitions) == 0:
            print(" No valid transitions found! Using default normalization values.")
            self.action_mean = np.array([0.0] * 6)
            self.action_std = np.array([1.0] * 6)
            return
        
        at1_actions = []
        at2_actions = []
        action_changes = []
        
        for transition in self.transitions:
            at1_actions.append(transition["at1_6dof"])
            at2_actions.append(transition["at2_6dof"])
            action_changes.append(transition["action_change_6dof"])
        
        # Convert to numpy arrays
        at1_actions = np.array(at1_actions)
        at2_actions = np.array(at2_actions)
        action_changes = np.array(action_changes)
        
        # 
        all_actions = np.vstack([at1_actions, at2_actions, action_changes])
        
        # 
        if all_actions.size == 0:
            print(" No valid action data found! Using default normalization values.")
            self.action_mean = np.array([0.0] * 6)
            self.action_std = np.array([1.0] * 6)
            return
        
        # 
        self.action_mean = np.mean(all_actions, axis=0)
        self.action_std = np.std(all_actions, axis=0)
        
        # Prevent division by zero
        self.action_std = np.where(self.action_std < 1e-6, 1.0, self.action_std)
        
        print(f"Action statistics computed from {len(self.transitions)} transitions:")
        print(f"  Total action vectors: {all_actions.shape[0]} (AT1: {at1_actions.shape[0]}, AT2: {at2_actions.shape[0]}, Changes: {action_changes.shape[0]})")
        print(f"  Combined range: {all_actions.min():.3f} to {all_actions.max():.3f}")
        print(f"  Mean per dimension: {self.action_mean}")
        print(f"  Std per dimension: {self.action_std}")
        
        # 
        dimension_names = ['X (mm)', 'Y (mm)', 'Z (mm)', 'Roll (rad)', 'Pitch (rad)', 'Yaw (rad)']
        print(f"\n📊 Detailed statistics per dimension:")
        for i, name in enumerate(dimension_names):
            print(f"  {name}: mean={self.action_mean[i]:.4f}, std={self.action_std[i]:.4f}, range=[{all_actions[:, i].min():.3f}, {all_actions[:, i].max():.3f}]")
        
        # Save normalization stats for use in val/test
        norm_stats = {
            "action_mean": self.action_mean.tolist(),
            "action_std": self.action_std.tolist(),
            "dimension_names": dimension_names,
            "total_samples": int(all_actions.shape[0]),
            "at1_samples": int(at1_actions.shape[0]),
            "at2_samples": int(at2_actions.shape[0]),
            "change_samples": int(action_changes.shape[0])
        }
        
        norm_file = os.path.join(self.data_dir, "normalization_stats.json")
        with open(norm_file, 'w') as f:
            json.dump(norm_stats, f, indent=2)
        print(f"📁 Normalization stats saved to: {norm_file}")
    
    def normalize_action(self, action):
        """Normalize a 6DOF action using computed statistics"""
        if not self.normalize_actions:
            return action
        return (action - self.action_mean) / self.action_std
    
    def load_transitions(self):
        """Load transitions from all patient folders"""
        for patient_id in self.patient_ids:
            patient_dir = os.path.join(self.data_dir, patient_id)
            json_file = os.path.join(patient_dir, "transitions_dataset.json")
            
            if not os.path.exists(json_file):
                print(f"Warning: JSON file not found for patient {patient_id}: {json_file}")
                continue
                
            # Load transitions for this patient
            with open(json_file, 'r') as f:
                patient_transitions = json.load(f)
            
            # Add patient_id and full paths to each transition
            for transition in patient_transitions:
                # Convert Windows-style paths to Unix-style paths
                ft1_image_path = transition["ft1_image_path"].replace('\\', '/')
                ft2_image_path = transition["ft2_image_path"].replace('\\', '/')
                
                # Convert relative paths to absolute paths
                ft1_path = os.path.join(patient_dir, ft1_image_path)
                ft2_path = os.path.join(patient_dir, ft2_image_path)
                
                # Verify files exist
                if not os.path.exists(ft1_path):
                    print(f"Warning: Image file not found: {ft1_path}")
                    continue
                    
                if not os.path.exists(ft2_path):
                    print(f"Warning: Image file not found: {ft2_path}")
                    continue
                
                # Create complete transition record
                complete_transition = {
                    "patient_id": patient_id,
                    "ft1_image_path": ft1_path,
                    "ft2_image_path": ft2_path,
                    "at1_6dof": transition["at1_6dof"],
                    "action_change_6dof": transition["action_change_6dof"], 
                    "at2_6dof": transition["at2_6dof"]
                }
                
                self.transitions.append(complete_transition)
            
            print(f"Loaded {len(patient_transitions)} transitions from patient {patient_id}")
    
    def __len__(self) -> int:
        return len(self.transitions)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a single transition sample
        
        Returns:
            image_t1: Input image at time t1 [C, H, W]
            a_hat_t1_to_t2_gt: Ground truth relative action from t1 to t2 [6] (normalized)
            at1_6dof_gt: Ground truth action at t1 [6] (main task target, normalized)
            at2_6dof_gt: Ground truth action at t2 [6] (auxiliary task target, normalized)
        """
        # Get transition data
        transition = self.transitions[idx]
        
        # Load image at t1
        image_t1 = Image.open(transition["ft1_image_path"]).convert("L")  # Convert to grayscale
        
        # Apply transformations
        if self.transform:
            image_t1 = self.transform(image_t1)
        
        # Extract actions from transition
        at1_6dof = np.array(transition["at1_6dof"], dtype=np.float32)
        action_change_6dof = np.array(transition["action_change_6dof"], dtype=np.float32) 
        at2_6dof = np.array(transition["at2_6dof"], dtype=np.float32)
        
        # Normalize actions if enabled
        if self.normalize_actions:
            at1_6dof = self.normalize_action(at1_6dof)
            at2_6dof = self.normalize_action(at2_6dof)
            action_change_6dof = self.normalize_action(action_change_6dof)
        
        # Convert to tensors
        a_hat_t1_to_t2_gt = torch.tensor(action_change_6dof, dtype=torch.float32)
        at1_6dof_gt = torch.tensor(at1_6dof, dtype=torch.float32)
        at2_6dof_gt = torch.tensor(at2_6dof, dtype=torch.float32)
        
        return image_t1, a_hat_t1_to_t2_gt, at1_6dof_gt, at2_6dof_gt
    
    def get_patient_stats(self) -> Dict[str, int]:
        """Get statistics about patients in this split"""
        patient_counts = {}
        for transition in self.transitions:
            patient_id = transition["patient_id"]
            patient_counts[patient_id] = patient_counts.get(patient_id, 0) + 1
        return patient_counts 