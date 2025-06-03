#!/usr/bin/env python
"""
獨立程式：載入訓練好的模型並生成驗證集散點圖
使用方法: python generate_validation_plots.py --checkpoint_path outputs_safe/run_20250527_044827/checkpoints/cardiac_dreamer-epoch=45-val_main_task_loss=0.4943.ckpt
"""

import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sklearn.metrics import r2_score
import json
from datetime import datetime

# Add project root to path
current_file_dir = os.path.dirname(os.path.abspath(__file__))
if current_file_dir not in sys.path:
    sys.path.insert(0, current_file_dir)

# Import model and dataset
from src.models.system import get_cardiac_dreamer_system
from src.data import CrossPatientTransitionsDataset, get_patient_splits, get_custom_patient_splits_no_test


def load_model_from_checkpoint(checkpoint_path: str):
    """載入訓練好的模型"""
    print(f"Loading model from: {checkpoint_path}")
    
    # 載入 checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 嘗試從 checkpoint 中獲取超參數
    if 'hyper_parameters' in checkpoint:
        hparams = checkpoint['hyper_parameters']
        print(f"📊 Found hyperparameters in checkpoint: {list(hparams.keys())}")
        
        # 使用 checkpoint 中的配置
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
        print("⚠️ No hyperparameters found in checkpoint, using default configuration")
        # 使用預設配置
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
    
    # 載入模型權重
    try:
        model.load_state_dict(checkpoint['state_dict'])
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model state dict: {e}")
        print("🔧 Trying to load with strict=False...")
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        print("✅ Model loaded with some missing/unexpected keys")
    
    model.eval()
    return model


def create_validation_dataset(data_dir: str = "data/processed", use_custom_split: bool = False):
    """創建驗證集"""
    print(f"Creating validation dataset from: {data_dir}")
    
    # 選擇分割方法
    if use_custom_split:
        print("Using custom split: patients 1-5 as validation set")
        train_patients, val_patients, test_patients = get_custom_patient_splits_no_test(data_dir)
    else:
        print("Using automatic patient splits")
        train_patients, val_patients, test_patients = get_patient_splits(data_dir)
    
    # 設定圖像轉換
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # 創建驗證集
    val_dataset = CrossPatientTransitionsDataset(
        data_dir=data_dir,
        transform=transform,
        split="val",
        train_patients=train_patients,
        val_patients=val_patients,
        test_patients=test_patients,
        small_subset=False,
        normalize_actions=True
    )
    
    print(f"✅ Validation dataset created: {len(val_dataset)} samples")
    print(f"📊 Validation patients: {val_dataset.get_patient_stats()}")
    
    return val_dataset


def generate_predictions(model, val_loader, device):
    """Generate predictions on validation set"""
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    print("Generating predictions...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if batch_idx % 10 == 0:
                print(f"Processing batch {batch_idx + 1}/{len(val_loader)}")
            
            image_t1, a_hat_t1_to_t2_gt, at1_6dof_gt, at2_6dof_gt = batch
            
            # 移動到設備
            image_t1 = image_t1.to(device)
            a_hat_t1_to_t2_gt = a_hat_t1_to_t2_gt.to(device)
            at1_6dof_gt = at1_6dof_gt.to(device)
            
            # 前向傳播
            outputs = model(image_t1, a_hat_t1_to_t2_gt)
            at1_pred = outputs['predicted_action_composed']  # 主任務預測
            
            # 🎯 統一反正規化：所有預測和目標都使用相同的統計量
            # 從驗證集獲取正規化統計量
            if hasattr(val_loader.dataset, 'action_mean') and hasattr(val_loader.dataset, 'action_std'):
                action_mean = torch.tensor(val_loader.dataset.action_mean, device=device)
                action_std = torch.tensor(val_loader.dataset.action_std, device=device)
                
                # 反正規化預測值和目標值
                at1_pred_denorm = at1_pred * action_std + action_mean
                at1_target_denorm = at1_6dof_gt * action_std + action_mean
            else:
                print("Warning: No normalization stats found, using raw values")
                at1_pred_denorm = at1_pred
                at1_target_denorm = at1_6dof_gt
            
            # 收集結果（使用反正規化後的值）
            all_predictions.append(at1_pred_denorm.cpu().numpy())
            all_targets.append(at1_target_denorm.cpu().numpy())
    
    # 合併所有批次
    predictions = np.vstack(all_predictions)
    targets = np.vstack(all_targets)
    
    print(f"Generated predictions for {predictions.shape[0]} samples")
    return predictions, targets


def create_scatter_plots(predictions, ground_truth, output_dir: str):
    """創建6軸散點圖"""
    print("Creating scatter plots...")
    
    # 創建輸出目錄
    plots_dir = os.path.join(output_dir, "validation_scatter_plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # 6DOF 維度名稱
    dimension_names = ['X', 'Y', 'Z', 'Roll', 'Pitch', 'Yaw']
    dimension_units = ['mm', 'mm', 'mm', 'deg', 'deg', 'deg']
    
    # 設定繪圖風格
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # 計算每個維度的 R² 和統計資訊
    r2_scores = []
    stats_info = {}
    
    # 創建個別散點圖
    for i, (dim_name, unit) in enumerate(zip(dimension_names, dimension_units)):
        plt.figure(figsize=(10, 8))
        
        pred_dim = predictions[:, i]
        gt_dim = ground_truth[:, i]
        
        # 計算 R² 分數
        r2 = r2_score(gt_dim, pred_dim)
        r2_scores.append(r2)
        
        # 計算統計資訊
        stats_info[dim_name] = {
            'r2_score': float(r2),
            'pred_mean': float(np.mean(pred_dim)),
            'pred_std': float(np.std(pred_dim)),
            'gt_mean': float(np.mean(gt_dim)),
            'gt_std': float(np.std(gt_dim)),
            'correlation': float(np.corrcoef(pred_dim, gt_dim)[0, 1])
        }
        
        # 創建散點圖
        plt.scatter(gt_dim, pred_dim, alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
        
        # 添加理想線 (y=x)
        min_val = min(np.min(gt_dim), np.min(pred_dim))
        max_val = max(np.max(gt_dim), np.max(pred_dim))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction (y=x)')
        
        # 添加趨勢線
        z = np.polyfit(gt_dim, pred_dim, 1)
        p = np.poly1d(z)
        plt.plot(gt_dim, p(gt_dim), 'g-', linewidth=2, alpha=0.8, label=f'Trend Line (slope={z[0]:.3f})')
        
        # 設定標籤和標題
        plt.xlabel(f'Ground Truth {dim_name} ({unit})', fontsize=12)
        plt.ylabel(f'Predicted {dim_name} ({unit})', fontsize=12)
        plt.title(f'{dim_name} Axis: Predicted vs Ground Truth\nR² = {r2:.4f}, Correlation = {stats_info[dim_name]["correlation"]:.4f}', fontsize=14)
        
        # 添加網格和圖例
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # 設定相等的軸比例
        plt.axis('equal')
        
        # 保存個別圖
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'scatter_{dim_name.lower()}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ {dim_name} axis: R² = {r2:.4f}")
    
    # 創建組合圖
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, (dim_name, unit) in enumerate(zip(dimension_names, dimension_units)):
        pred_dim = predictions[:, i]
        gt_dim = ground_truth[:, i]
        r2 = r2_scores[i]
        
        # 散點圖
        axes[i].scatter(gt_dim, pred_dim, alpha=0.6, s=20, edgecolors='black', linewidth=0.3)
        
        # 理想線
        min_val = min(np.min(gt_dim), np.min(pred_dim))
        max_val = max(np.max(gt_dim), np.max(pred_dim))
        axes[i].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, alpha=0.8)
        
        # 趨勢線
        z = np.polyfit(gt_dim, pred_dim, 1)
        p = np.poly1d(z)
        axes[i].plot(gt_dim, p(gt_dim), 'g-', linewidth=2, alpha=0.8)
        
        # 標籤和標題
        axes[i].set_xlabel(f'Ground Truth {dim_name} ({unit})')
        axes[i].set_ylabel(f'Predicted {dim_name} ({unit})')
        axes[i].set_title(f'{dim_name}: R² = {r2:.4f}')
        axes[i].grid(True, alpha=0.3)
        axes[i].set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'scatter_combined.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 保存統計資訊
    stats_file = os.path.join(plots_dir, 'validation_statistics.json')
    with open(stats_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'overall_stats': {
                'mean_r2': float(np.mean(r2_scores)),
                'total_samples': int(len(predictions)),
                'dimensions': dimension_names
            },
            'dimension_stats': stats_info
        }, f, indent=2)
    
    print(f"📊 Scatter plots saved to: {plots_dir}")
    print(f"📄 Statistics saved to: {stats_file}")
    print(f"🎯 Overall mean R²: {np.mean(r2_scores):.4f}")
    
    return r2_scores, stats_info


def main():
    parser = argparse.ArgumentParser(description="Generate validation scatter plots from trained model")
    parser.add_argument("--checkpoint_path", type=str, required=True, 
                       help="Path to model checkpoint (.ckpt file)")
    parser.add_argument("--data_dir", type=str, default="data/processed",
                       help="Data directory path")
    parser.add_argument("--output_dir", type=str, default="validation_analysis",
                       help="Output directory for plots")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size for inference")
    parser.add_argument("--use_custom_split", action="store_true",
                       help="Use custom split where patients 1-5 are validation set")
    
    args = parser.parse_args()
    
    # 檢查 checkpoint 是否存在
    if not os.path.exists(args.checkpoint_path):
        print(f"❌ Checkpoint file not found: {args.checkpoint_path}")
        return
    
    # 創建輸出目錄
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("🚀 Starting validation analysis...")
    print(f"📁 Checkpoint: {args.checkpoint_path}")
    print(f"📁 Data directory: {args.data_dir}")
    print(f"📁 Output directory: {args.output_dir}")
    
    try:
        # 1. 載入模型
        model = load_model_from_checkpoint(args.checkpoint_path)
        
        # 2. 創建驗證集
        val_dataset = create_validation_dataset(args.data_dir, args.use_custom_split)
        
        # 創建 DataLoader
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=False,
            persistent_workers=False
        )
        
        # 3. 生成預測
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        predictions, ground_truth = generate_predictions(model, val_loader, device)
        
        # 4. 創建散點圖
        r2_scores, stats_info = create_scatter_plots(predictions, ground_truth, args.output_dir)
        
        # 5. 打印總結
        print("\n🎉 Analysis completed successfully!")
        print("\n📊 Final Results:")
        dimension_names = ['X', 'Y', 'Z', 'Roll', 'Pitch', 'Yaw']
        for i, dim_name in enumerate(dimension_names):
            print(f"  {dim_name:5s}: R² = {r2_scores[i]:.4f}")
        print(f"  Mean : R² = {np.mean(r2_scores):.4f}")
        
        print(f"\n📁 All plots saved to: {args.output_dir}/validation_scatter_plots/")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        raise


if __name__ == "__main__":
    main() 