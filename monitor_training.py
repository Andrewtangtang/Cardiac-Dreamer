#!/usr/bin/env python
"""
Training Monitor - 監控訓練過程中的系統資源使用
防止系統crash的監控工具
"""

import psutil
import GPUtil
import time
import threading
import signal
import sys
from datetime import datetime

class TrainingMonitor:
    def __init__(self, check_interval=5, max_gpu_memory=0.9, max_cpu_percent=90, max_ram_percent=90):
        """
        初始化訓練監控器
        
        Args:
            check_interval: 檢查間隔（秒）
            max_gpu_memory: GPU記憶體使用上限（比例）
            max_cpu_percent: CPU使用率上限（百分比）
            max_ram_percent: RAM使用率上限（百分比）
        """
        self.check_interval = check_interval
        self.max_gpu_memory = max_gpu_memory
        self.max_cpu_percent = max_cpu_percent
        self.max_ram_percent = max_ram_percent
        self.monitoring = False
        self.monitor_thread = None
        
        # 設置信號處理
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        """處理中斷信號"""
        print(f"\n[MONITOR] 收到信號 {signum}，停止監控...")
        self.stop_monitoring()
        sys.exit(0)
    
    def get_gpu_info(self):
        """獲取GPU資訊"""
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # 使用第一個GPU
                return {
                    'memory_used': gpu.memoryUsed,
                    'memory_total': gpu.memoryTotal,
                    'memory_percent': gpu.memoryUtil,
                    'temperature': gpu.temperature,
                    'load': gpu.load
                }
            else:
                return None
        except Exception as e:
            print(f"[WARNING] 無法獲取GPU資訊: {e}")
            return None
    
    def get_system_info(self):
        """獲取系統資訊"""
        try:
            # CPU資訊
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # 記憶體資訊
            memory = psutil.virtual_memory()
            
            # 磁碟資訊
            disk = psutil.disk_usage('/')
            
            return {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_used_gb': memory.used / (1024**3),
                'memory_total_gb': memory.total / (1024**3),
                'disk_percent': disk.percent
            }
        except Exception as e:
            print(f"[WARNING] 無法獲取系統資訊: {e}")
            return None
    
    def check_safety_limits(self, gpu_info, system_info):
        """檢查安全限制"""
        warnings = []
        critical = []
        
        # 檢查GPU
        if gpu_info:
            if gpu_info['memory_percent'] > self.max_gpu_memory:
                critical.append(f"GPU記憶體使用過高: {gpu_info['memory_percent']:.1%}")
            elif gpu_info['memory_percent'] > 0.8:
                warnings.append(f"GPU記憶體使用較高: {gpu_info['memory_percent']:.1%}")
            
            if gpu_info['temperature'] > 85:
                critical.append(f"GPU溫度過高: {gpu_info['temperature']}°C")
            elif gpu_info['temperature'] > 80:
                warnings.append(f"GPU溫度較高: {gpu_info['temperature']}°C")
        
        # 檢查系統
        if system_info:
            if system_info['cpu_percent'] > self.max_cpu_percent:
                critical.append(f"CPU使用率過高: {system_info['cpu_percent']:.1f}%")
            elif system_info['cpu_percent'] > 80:
                warnings.append(f"CPU使用率較高: {system_info['cpu_percent']:.1f}%")
            
            if system_info['memory_percent'] > self.max_ram_percent:
                critical.append(f"RAM使用率過高: {system_info['memory_percent']:.1f}%")
            elif system_info['memory_percent'] > 80:
                warnings.append(f"RAM使用率較高: {system_info['memory_percent']:.1f}%")
        
        return warnings, critical
    
    def print_status(self, gpu_info, system_info):
        """打印狀態資訊"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"\n[{timestamp}] === 系統狀態 ===")
        
        # GPU狀態
        if gpu_info:
            print(f"🎮 GPU: {gpu_info['memory_used']:.0f}MB/{gpu_info['memory_total']:.0f}MB "
                  f"({gpu_info['memory_percent']:.1%}) | "
                  f"負載: {gpu_info['load']:.1%} | "
                  f"溫度: {gpu_info['temperature']}°C")
        else:
            print("🎮 GPU: 無法檢測")
        
        # 系統狀態
        if system_info:
            print(f"💻 CPU: {system_info['cpu_percent']:.1f}% | "
                  f"RAM: {system_info['memory_used_gb']:.1f}GB/{system_info['memory_total_gb']:.1f}GB "
                  f"({system_info['memory_percent']:.1f}%)")
        else:
            print("💻 系統: 無法檢測")
    
    def monitor_loop(self):
        """監控循環"""
        print("[MONITOR] 開始監控系統資源...")
        print(f"[MONITOR] 檢查間隔: {self.check_interval}秒")
        print(f"[MONITOR] 安全限制: GPU記憶體<{self.max_gpu_memory:.0%}, CPU<{self.max_cpu_percent}%, RAM<{self.max_ram_percent}%")
        
        while self.monitoring:
            try:
                # 獲取資源資訊
                gpu_info = self.get_gpu_info()
                system_info = self.get_system_info()
                
                # 打印狀態
                self.print_status(gpu_info, system_info)
                
                # 檢查安全限制
                warnings, critical = self.check_safety_limits(gpu_info, system_info)
                
                # 顯示警告
                for warning in warnings:
                    print(f"⚠️  [WARNING] {warning}")
                
                # 顯示嚴重警告
                for crit in critical:
                    print(f"🚨 [CRITICAL] {crit}")
                
                if critical:
                    print("🚨 [CRITICAL] 建議立即停止訓練以防止系統crash！")
                
                # 等待下次檢查
                time.sleep(self.check_interval)
                
            except Exception as e:
                print(f"[ERROR] 監控過程中發生錯誤: {e}")
                time.sleep(self.check_interval)
    
    def start_monitoring(self):
        """開始監控"""
        if not self.monitoring:
            self.monitoring = True
            self.monitor_thread = threading.Thread(target=self.monitor_loop, daemon=True)
            self.monitor_thread.start()
            print("[MONITOR] 監控已啟動")
    
    def stop_monitoring(self):
        """停止監控"""
        if self.monitoring:
            self.monitoring = False
            if self.monitor_thread:
                self.monitor_thread.join(timeout=2)
            print("[MONITOR] 監控已停止")

def main():
    """主函數 - 獨立運行監控器"""
    print("🔍 Cardiac Dreamer 訓練監控器")
    print("=" * 50)
    
    # 創建監控器
    monitor = TrainingMonitor(
        check_interval=3,      # 每3秒檢查一次
        max_gpu_memory=0.85,   # GPU記憶體85%警告
        max_cpu_percent=85,    # CPU 85%警告
        max_ram_percent=85     # RAM 85%警告
    )
    
    try:
        # 開始監控
        monitor.start_monitoring()
        
        print("\n按 Ctrl+C 停止監控...")
        
        # 保持主線程運行
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n[INFO] 用戶中斷監控")
    except Exception as e:
        print(f"[ERROR] 監控器錯誤: {e}")
    finally:
        monitor.stop_monitoring()
        print("[INFO] 監控器已關閉")

if __name__ == "__main__":
    main() 