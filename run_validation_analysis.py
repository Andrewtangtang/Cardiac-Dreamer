#!/usr/bin/env python
"""
簡化的驗證分析執行腳本
自動找到最佳的 checkpoint 並生成散點圖
"""

import os
import glob
import subprocess
import sys

def find_best_checkpoint(run_dir: str):
    """找到最佳的 checkpoint"""
    checkpoints_dir = os.path.join(run_dir, "checkpoints")
    
    if not os.path.exists(checkpoints_dir):
        print(f"❌ Checkpoints directory not found: {checkpoints_dir}")
        return None
    
    # 找到所有 checkpoint 文件
    checkpoint_files = glob.glob(os.path.join(checkpoints_dir, "cardiac_dreamer-epoch=*-val_main_task_loss=*.ckpt"))
    
    if not checkpoint_files:
        print(f"❌ No checkpoint files found in: {checkpoints_dir}")
        return None
    
    # 根據 validation loss 排序（越小越好）
    def extract_val_loss(filename):
        try:
            # 從檔名中提取 validation loss
            parts = os.path.basename(filename).split('-')
            for part in parts:
                if part.startswith('val_main_task_loss='):
                    return float(part.split('=')[1].replace('.ckpt', ''))
        except:
            return float('inf')
        return float('inf')
    
    # 排序並選擇最佳的
    checkpoint_files.sort(key=extract_val_loss)
    best_checkpoint = checkpoint_files[0]
    
    print(f"✅ Found best checkpoint: {best_checkpoint}")
    print(f"📊 Validation loss: {extract_val_loss(best_checkpoint):.4f}")
    
    return best_checkpoint

def main():
    # 預設的運行目錄
    default_run_dir = "outputs_channel_token/run_20250528_041012"
    
    if len(sys.argv) > 1:
        run_dir = sys.argv[1]
    else:
        run_dir = default_run_dir
    
    print(f"🔍 Looking for checkpoints in: {run_dir}")
    
    # 找到最佳 checkpoint
    best_checkpoint = find_best_checkpoint(run_dir)
    
    if best_checkpoint is None:
        print("❌ No valid checkpoint found!")
        return
    
    # 執行驗證分析（使用 pipenv run 確保在正確的環境中執行）
    cmd = [
        "pipenv", "run", "python", "generate_validation_plots.py",
        "--checkpoint_path", best_checkpoint,
        "--data_dir", "data/processed",
        "--output_dir", "validation_analysis_results"
    ]
    
    print(f"🚀 Running validation analysis...")
    print(f"📝 Command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
        print("\n🎉 Validation analysis completed successfully!")
        print("📁 Results saved to: validation_analysis_results/validation_scatter_plots/")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running validation analysis: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main() 