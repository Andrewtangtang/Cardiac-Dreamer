#!/usr/bin/env python
"""
測試修正後的正規化實現
"""

import sys
import os
sys.path.append('.')

from src.train import CrossPatientTransitionsDataset
import torchvision.transforms as transforms
import numpy as np

def test_normalization():
    print("🧪 Testing normalization implementation...")
    
    # 創建數據變換
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # 創建測試數據集
    dataset = CrossPatientTransitionsDataset(
        data_dir='data/processed',
        transform=transform,
        split='train',
        small_subset=True,
        normalize_actions=True
    )
    
    print(f"✅ Dataset created successfully!")
    print(f"📊 Dataset size: {len(dataset)}")
    print(f"📊 Action mean: {dataset.action_mean}")
    print(f"📊 Action std: {dataset.action_std}")
    
    # 測試幾個樣本
    print(f"\n🔍 Testing samples...")
    for i in range(min(3, len(dataset))):
        image_t1, a_hat_t1_to_t2_gt, at1_6dof_gt, at2_6dof_gt = dataset[i]
        print(f"Sample {i}:")
        print(f"  a_hat_t1_to_t2_gt: {a_hat_t1_to_t2_gt.numpy()}")
        print(f"  at1_6dof_gt: {at1_6dof_gt.numpy()}")
        print(f"  at2_6dof_gt: {at2_6dof_gt.numpy()}")
    
    print(f"\n🎯 Normalization test completed!")

if __name__ == "__main__":
    test_normalization() 