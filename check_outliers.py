#!/usr/bin/env python
"""
快速檢查 AT1 異常值
"""
import numpy as np
import sys
sys.path.append('.')
from src.data import CrossPatientTransitionsDataset, get_patient_splits
import torchvision.transforms as transforms

def main():
    print("檢查 AT1 中的異常值 (>500 或 <-500)...")
    
    # 創建數據集
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    train_patients, val_patients, test_patients = get_patient_splits('data/processed')
    
    # 檢查所有分割
    for split_name, patients in [('train', train_patients), ('val', val_patients), ('test', test_patients)]:
        print(f"\n=== 檢查 {split_name} 集 ===")
        
        dataset = CrossPatientTransitionsDataset(
            data_dir='data/processed',
            transform=transform,
            split=split_name,
            train_patients=train_patients,
            val_patients=val_patients,
            test_patients=test_patients,
            small_subset=False,
            normalize_actions=False
        )
        
        outliers_found = 0
        axis_names = ['X (mm)', 'Y (mm)', 'Z (mm)', 'Roll (deg)', 'Pitch (deg)', 'Yaw (deg)']
        
        for i in range(len(dataset)):
            try:
                _, _, at1_6dof, _ = dataset[i]
                at1_action = at1_6dof.numpy() if hasattr(at1_6dof, 'numpy') else at1_6dof
                
                for j, axis in enumerate(axis_names):
                    value = at1_action[j]
                    if abs(value) > 200:
                        transition = dataset.transitions[i]
                        print(f"[OUTLIER] 異常值發現:")
                        print(f"   軸: {axis}")
                        print(f"   值: {value:.3f}")
                        print(f"   患者: {transition['patient_id']}")
                        print(f"   樣本索引: {i}/{len(dataset)}")
                        print(f"   圖片路徑: {transition['ft1_image_path']}")
                        print()
                        outliers_found += 1
                        
            except Exception as e:
                print(f"警告: 處理樣本 {i} 時出錯: {e}")
                continue
        
        print(f"{split_name} 集找到 {outliers_found} 個異常值")

if __name__ == "__main__":
    main() 