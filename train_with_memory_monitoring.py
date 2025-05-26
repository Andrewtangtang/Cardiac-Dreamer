#!/usr/bin/env python3
"""
帶記憶體監控的安全訓練腳本
整合記憶體洩漏檢測，防止系統crash
"""

import os
import sys
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
import gc
import time

# 添加項目根目錄到路徑
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# 導入記憶體檢測器
from memory_leak_detector import MemoryLeakDetector, start_memory_monitoring, stop_memory_monitoring, check_memory, cleanup_memory

# 導入訓練模組
from src.train import main as original_main, load_config


class MemoryAwareCallback(pl.Callback):
    """記憶體感知的回調函數"""
    
    def __init__(self, memory_detector: MemoryLeakDetector):
        super().__init__()
        self.memory_detector = memory_detector
        self.last_cleanup_epoch = 0
        self.cleanup_interval = 5  # 每5個epoch清理一次
        
    def on_train_epoch_start(self, trainer, pl_module):
        """訓練epoch開始時檢查記憶體"""
        current_epoch = trainer.current_epoch
        self.memory_detector.manual_check(epoch=current_epoch, step=None)
        
        # 定期清理記憶體
        if current_epoch - self.last_cleanup_epoch >= self.cleanup_interval:
            print(f"🧹 Epoch {current_epoch}: 執行定期記憶體清理")
            self.memory_detector.force_cleanup()
            self.last_cleanup_epoch = current_epoch
    
    def on_train_epoch_end(self, trainer, pl_module):
        """訓練epoch結束時檢查記憶體"""
        current_epoch = trainer.current_epoch
        self.memory_detector.manual_check(epoch=current_epoch, step=None)
        
        # 檢查記憶體使用情況
        summary = self.memory_detector.get_summary()
        if "cpu_memory_growth_mb" in summary:
            cpu_growth = summary["cpu_memory_growth_mb"]
            if cpu_growth > 2000:  # 超過2GB增長
                print(f"⚠️ CPU記憶體增長過多: {cpu_growth:.1f} MB，執行強制清理")
                self.memory_detector.force_cleanup()
        
        if "gpu_memory_growth_mb" in summary:
            gpu_growth = summary["gpu_memory_growth_mb"]
            if gpu_growth > 1000:  # 超過1GB增長
                print(f"⚠️ GPU記憶體增長過多: {gpu_growth:.1f} MB，執行強制清理")
                self.memory_detector.force_cleanup()
    
    def on_validation_epoch_end(self, trainer, pl_module):
        """驗證epoch結束時清理記憶體"""
        # 驗證後清理記憶體，因為驗證會累積大量輸出
        if hasattr(pl_module, 'validation_step_outputs'):
            if len(pl_module.validation_step_outputs) > 0:
                print(f"🧹 清理驗證輸出: {len(pl_module.validation_step_outputs)} 個樣本")
                pl_module.validation_step_outputs.clear()
        
        # 強制清理GPU緩存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        gc.collect()


def setup_memory_safe_training(config_path: str, data_dir: str, output_dir: str):
    """設置記憶體安全的訓練環境"""
    
    print("🔧 設置記憶體安全訓練環境...")
    
    # 1. 啟動記憶體監控
    memory_detector = MemoryLeakDetector(
        check_interval=30.0,  # 每30秒檢查一次
        warning_threshold_mb=500,  # 500MB警告
        critical_threshold_mb=1000  # 1GB危險
    )
    memory_detector.start_monitoring()
    
    # 2. 設置PyTorch記憶體管理
    if torch.cuda.is_available():
        # 設置GPU記憶體分配策略
        torch.cuda.empty_cache()
        # 設置記憶體分片以減少碎片化
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
        print("✅ GPU記憶體管理已優化")
    
    # 3. 設置垃圾回收
    gc.set_threshold(700, 10, 10)  # 更積極的垃圾回收
    print("✅ 垃圾回收已優化")
    
    # 4. 設置環境變量
    os.environ['PYTHONHASHSEED'] = '42'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    return memory_detector


def safe_train_main():
    """安全訓練主函數"""
    parser = argparse.ArgumentParser(description="帶記憶體監控的安全訓練")
    parser.add_argument("--config", type=str, default="configs/safe_training.yaml",
                       help="配置文件路徑")
    parser.add_argument("--data_dir", type=str, default="data/processed",
                       help="數據目錄")
    parser.add_argument("--output_dir", type=str, default="outputs_safe_training",
                       help="輸出目錄")
    parser.add_argument("--max_epochs", type=int, default=None,
                       help="最大訓練輪數（覆蓋配置文件）")
    parser.add_argument("--batch_size", type=int, default=None,
                       help="批次大小（覆蓋配置文件）")
    parser.add_argument("--enable_monitoring", action="store_true", default=True,
                       help="啟用記憶體監控")
    
    args = parser.parse_args()
    
    print("🚀 啟動帶記憶體監控的安全訓練")
    print(f"   配置文件: {args.config}")
    print(f"   數據目錄: {args.data_dir}")
    print(f"   輸出目錄: {args.output_dir}")
    
    memory_detector = None
    
    try:
        # 設置記憶體安全環境
        if args.enable_monitoring:
            memory_detector = setup_memory_safe_training(
                args.config, args.data_dir, args.output_dir
            )
        
        # 載入配置並進行安全調整
        config_override = load_config(args.config) if args.config else {}
        
        # 安全配置調整
        if "training" not in config_override:
            config_override["training"] = {}
        
        # 強制使用安全設置
        safe_training_config = {
            "batch_size": args.batch_size or 4,  # 小批次大小
            "num_workers": 2,  # 限制worker數量
            "precision": 16,  # 混合精度
            "gradient_clip_val": 0.5,  # 梯度裁剪
            "accumulate_grad_batches": 4,  # 梯度累積
            "max_epochs": args.max_epochs or 20,  # 限制epoch數量
            "check_val_every_n_epoch": 2,  # 減少驗證頻率
            "log_every_n_steps": 50  # 減少日誌頻率
        }
        
        config_override["training"].update(safe_training_config)
        
        print("🔧 應用安全訓練配置:")
        for key, value in safe_training_config.items():
            print(f"   {key}: {value}")
        
        # 創建臨時配置文件
        import tempfile
        import yaml
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_override, f)
            temp_config_path = f.name
        
        # 修改參數以使用臨時配置
        original_args = argparse.Namespace(
            config=temp_config_path,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            manual_splits=False,
            train_patients=None,
            val_patients=None,
            test_patients=None
        )
        
        # 添加記憶體監控回調
        if memory_detector:
            # 這裡需要修改原始訓練函數以支持自定義回調
            # 暫時使用原始訓練函數
            print("⚠️ 注意：記憶體監控回調需要修改原始訓練函數才能完全整合")
        
        # 執行訓練
        print("🎯 開始安全訓練...")
        original_main(original_args)
        
        print("✅ 訓練完成！")
        
    except KeyboardInterrupt:
        print("\n⚠️ 訓練被用戶中斷")
    except Exception as e:
        print(f"\n❌ 訓練過程中發生錯誤: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理資源
        if memory_detector:
            print("🧹 執行最終記憶體清理...")
            memory_detector.force_cleanup()
            memory_detector.stop_monitoring()
            
            # 生成記憶體報告
            report_path = os.path.join(args.output_dir, "memory_usage_report.png")
            os.makedirs(args.output_dir, exist_ok=True)
            memory_detector.generate_report(report_path)
            
            # 顯示記憶體摘要
            summary = memory_detector.get_summary()
            print("\n📊 最終記憶體使用摘要:")
            for key, value in summary.items():
                if isinstance(value, float):
                    print(f"   {key}: {value:.1f}")
                else:
                    print(f"   {key}: {value}")
        
        # 清理臨時文件
        try:
            if 'temp_config_path' in locals():
                os.unlink(temp_config_path)
        except:
            pass
        
        # 最終清理
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("🏁 安全訓練腳本執行完畢")


def quick_memory_test():
    """快速記憶體測試"""
    print("🧪 執行快速記憶體測試...")
    
    detector = MemoryLeakDetector(check_interval=5.0)
    detector.start_monitoring()
    
    try:
        # 模擬訓練過程
        for epoch in range(3):
            print(f"\n--- 模擬 Epoch {epoch} ---")
            
            # 模擬創建一些tensor
            tensors = []
            for step in range(5):
                if torch.cuda.is_available():
                    tensor = torch.randn(100, 100).cuda()
                else:
                    tensor = torch.randn(100, 100)
                tensors.append(tensor)
                
                if step % 2 == 0:
                    detector.manual_check(epoch=epoch, step=step)
            
            # 模擬記憶體洩漏（不清理tensor）
            if epoch < 2:
                print(f"   保留 {len(tensors)} 個tensors（模擬記憶體洩漏）")
            else:
                print(f"   清理 {len(tensors)} 個tensors")
                del tensors
                detector.force_cleanup()
            
            time.sleep(2)
    
    finally:
        detector.stop_monitoring()
        detector.generate_report("memory_test_report.png")
        
        summary = detector.get_summary()
        print("\n📊 測試記憶體摘要:")
        for key, value in summary.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.1f}")
            else:
                print(f"   {key}: {value}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        quick_memory_test()
    else:
        safe_train_main() 