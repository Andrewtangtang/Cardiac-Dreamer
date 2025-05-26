#!/usr/bin/env python
"""
Safe Training Script - 安全訓練腳本
包含資源監控和自動停止功能，防止系統crash
"""

import subprocess
import sys
import os
import time
import threading
from monitor_training import TrainingMonitor

class SafeTrainer:
    def __init__(self):
        self.training_process = None
        self.monitor = TrainingMonitor(
            check_interval=5,
            max_gpu_memory=0.85,  # 85% GPU記憶體警告
            max_cpu_percent=85,   # 85% CPU警告
            max_ram_percent=85    # 85% RAM警告
        )
        self.should_stop = False
    
    def start_training(self, config_file="configs/safe_training.yaml"):
        """啟動安全訓練"""
        print("🚀 啟動安全訓練模式...")
        print(f"📋 使用配置: {config_file}")
        
        # 檢查配置檔案是否存在
        if not os.path.exists(config_file):
            print(f"❌ 配置檔案不存在: {config_file}")
            return False
        
        # 啟動監控
        self.monitor.start_monitoring()
        
        # 構建訓練命令
        cmd = [
            sys.executable, "src/train.py",
            "--config", config_file,
            "--data_dir", "data/processed",
            "--output_dir", "outputs_safe_training"
        ]
        
        print(f"🔧 執行命令: {' '.join(cmd)}")
        
        try:
            # 啟動訓練進程
            self.training_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            print("✅ 訓練進程已啟動")
            print("📊 監控系統資源中...")
            print("⚠️  如果出現嚴重警告，請立即按 Ctrl+C 停止訓練")
            
            # 監控訓練輸出
            self.monitor_training_output()
            
        except Exception as e:
            print(f"❌ 啟動訓練失敗: {e}")
            return False
        
        return True
    
    def monitor_training_output(self):
        """監控訓練輸出"""
        try:
            while True:
                output = self.training_process.stdout.readline()
                if output == '' and self.training_process.poll() is not None:
                    break
                if output:
                    print(f"[TRAIN] {output.strip()}")
                
                # 檢查是否需要停止
                if self.should_stop:
                    print("🛑 收到停止信號，終止訓練...")
                    self.stop_training()
                    break
            
            # 獲取最終返回碼
            return_code = self.training_process.poll()
            if return_code == 0:
                print("✅ 訓練正常完成")
            else:
                print(f"⚠️  訓練異常結束，返回碼: {return_code}")
                
        except KeyboardInterrupt:
            print("\n🛑 用戶中斷訓練")
            self.stop_training()
        except Exception as e:
            print(f"❌ 監控訓練輸出時發生錯誤: {e}")
            self.stop_training()
    
    def stop_training(self):
        """停止訓練"""
        if self.training_process:
            print("🛑 正在停止訓練進程...")
            try:
                self.training_process.terminate()
                self.training_process.wait(timeout=10)
                print("✅ 訓練進程已停止")
            except subprocess.TimeoutExpired:
                print("⚠️  強制終止訓練進程...")
                self.training_process.kill()
                self.training_process.wait()
                print("✅ 訓練進程已強制終止")
            except Exception as e:
                print(f"❌ 停止訓練進程時發生錯誤: {e}")
        
        # 停止監控
        self.monitor.stop_monitoring()
    
    def emergency_stop(self):
        """緊急停止"""
        print("🚨 緊急停止訓練！")
        self.should_stop = True
        self.stop_training()

def main():
    """主函數"""
    print("🛡️  Cardiac Dreamer 安全訓練器")
    print("=" * 50)
    print("🔧 特性:")
    print("   - 降低的模型大小和批次大小")
    print("   - 實時系統資源監控")
    print("   - 自動警告和建議")
    print("   - 安全停止機制")
    print("=" * 50)
    
    # 檢查必要的依賴
    try:
        import psutil
        import GPUtil
        print("✅ 系統監控依賴已安裝")
    except ImportError as e:
        print(f"❌ 缺少必要依賴: {e}")
        print("請安裝: pip install psutil GPUtil")
        return
    
    # 創建安全訓練器
    trainer = SafeTrainer()
    
    try:
        # 啟動訓練
        success = trainer.start_training("configs/safe_training.yaml")
        
        if not success:
            print("❌ 訓練啟動失敗")
            return
        
    except KeyboardInterrupt:
        print("\n🛑 用戶中斷")
        trainer.stop_training()
    except Exception as e:
        print(f"❌ 發生未預期的錯誤: {e}")
        trainer.emergency_stop()
    finally:
        print("🏁 安全訓練器已關閉")

if __name__ == "__main__":
    main() 