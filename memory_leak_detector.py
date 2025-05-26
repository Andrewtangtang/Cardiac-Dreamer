#!/usr/bin/env python3
"""
記憶體洩漏檢測器
監控訓練過程中的記憶體使用情況，檢測潛在的記憶體洩漏問題
"""

import gc
import psutil
import torch
import time
import threading
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class MemorySnapshot:
    """記憶體快照"""
    timestamp: float
    cpu_memory_mb: float
    gpu_memory_mb: float
    gpu_allocated_mb: float
    gpu_cached_mb: float
    num_tensors: int
    epoch: Optional[int] = None
    step: Optional[int] = None


class MemoryLeakDetector:
    """記憶體洩漏檢測器"""
    
    def __init__(self, 
                 check_interval: float = 10.0,  # 檢查間隔（秒）
                 warning_threshold_mb: float = 1000,  # 警告閾值（MB）
                 critical_threshold_mb: float = 2000,  # 危險閾值（MB）
                 max_snapshots: int = 1000):  # 最大快照數量
        self.check_interval = check_interval
        self.warning_threshold = warning_threshold_mb
        self.critical_threshold = critical_threshold_mb
        self.max_snapshots = max_snapshots
        
        self.snapshots: List[MemorySnapshot] = []
        self.monitoring = False
        self.monitor_thread = None
        self.start_time = time.time()
        
        # 基線記憶體使用量
        self.baseline_cpu = None
        self.baseline_gpu = None
        
    def start_monitoring(self):
        """開始監控"""
        if self.monitoring:
            return
            
        self.monitoring = True
        self.start_time = time.time()
        
        # 記錄基線
        snapshot = self._take_snapshot()
        self.baseline_cpu = snapshot.cpu_memory_mb
        if torch.cuda.is_available():
            self.baseline_gpu = snapshot.gpu_memory_mb
            
        self.snapshots.append(snapshot)
        
        # 啟動監控線程
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        print(f"🔍 記憶體洩漏檢測器已啟動")
        print(f"   基線CPU記憶體: {self.baseline_cpu:.1f} MB")
        if self.baseline_gpu:
            print(f"   基線GPU記憶體: {self.baseline_gpu:.1f} MB")
    
    def stop_monitoring(self):
        """停止監控"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        print("🔍 記憶體洩漏檢測器已停止")
    
    def _monitor_loop(self):
        """監控循環"""
        while self.monitoring:
            try:
                snapshot = self._take_snapshot()
                self.snapshots.append(snapshot)
                
                # 限制快照數量
                if len(self.snapshots) > self.max_snapshots:
                    self.snapshots.pop(0)
                
                # 檢查記憶體洩漏
                self._check_memory_leak(snapshot)
                
                time.sleep(self.check_interval)
                
            except Exception as e:
                print(f"⚠️ 記憶體監控錯誤: {e}")
                time.sleep(self.check_interval)
    
    def _take_snapshot(self) -> MemorySnapshot:
        """拍攝記憶體快照"""
        # CPU記憶體
        process = psutil.Process()
        cpu_memory_mb = process.memory_info().rss / 1024 / 1024
        
        # GPU記憶體
        gpu_memory_mb = 0
        gpu_allocated_mb = 0
        gpu_cached_mb = 0
        num_tensors = 0
        
        if torch.cuda.is_available():
            gpu_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
            gpu_allocated_mb = torch.cuda.memory_allocated() / 1024 / 1024
            gpu_cached_mb = torch.cuda.memory_reserved() / 1024 / 1024
            
            # 計算tensor數量
            for obj in gc.get_objects():
                if torch.is_tensor(obj) and obj.is_cuda:
                    num_tensors += 1
        
        return MemorySnapshot(
            timestamp=time.time() - self.start_time,
            cpu_memory_mb=cpu_memory_mb,
            gpu_memory_mb=gpu_memory_mb,
            gpu_allocated_mb=gpu_allocated_mb,
            gpu_cached_mb=gpu_cached_mb,
            num_tensors=num_tensors
        )
    
    def _check_memory_leak(self, snapshot: MemorySnapshot):
        """檢查記憶體洩漏"""
        if len(self.snapshots) < 5:  # 需要足夠的歷史數據
            return
        
        # 計算記憶體增長趨勢
        recent_snapshots = self.snapshots[-5:]
        cpu_growth = snapshot.cpu_memory_mb - recent_snapshots[0].cpu_memory_mb
        gpu_growth = snapshot.gpu_memory_mb - recent_snapshots[0].gpu_memory_mb
        
        # CPU記憶體洩漏檢查
        if cpu_growth > self.warning_threshold:
            if cpu_growth > self.critical_threshold:
                print(f"🚨 嚴重CPU記憶體洩漏警告!")
                print(f"   過去 {len(recent_snapshots)} 次檢查增長: {cpu_growth:.1f} MB")
                print(f"   當前使用量: {snapshot.cpu_memory_mb:.1f} MB")
                self._suggest_cpu_fixes()
            else:
                print(f"⚠️ CPU記憶體增長警告: +{cpu_growth:.1f} MB")
        
        # GPU記憶體洩漏檢查
        if torch.cuda.is_available() and gpu_growth > self.warning_threshold:
            if gpu_growth > self.critical_threshold:
                print(f"🚨 嚴重GPU記憶體洩漏警告!")
                print(f"   過去 {len(recent_snapshots)} 次檢查增長: {gpu_growth:.1f} MB")
                print(f"   當前使用量: {snapshot.gpu_memory_mb:.1f} MB")
                print(f"   GPU tensor數量: {snapshot.num_tensors}")
                self._suggest_gpu_fixes()
            else:
                print(f"⚠️ GPU記憶體增長警告: +{gpu_growth:.1f} MB")
    
    def _suggest_cpu_fixes(self):
        """建議CPU記憶體修復方法"""
        print("💡 CPU記憶體洩漏修復建議:")
        print("   1. 檢查是否有未釋放的大型數據結構")
        print("   2. 確保validation_step_outputs和test_step_outputs被正確清理")
        print("   3. 減少DataLoader的num_workers")
        print("   4. 禁用persistent_workers")
        print("   5. 調用gc.collect()強制垃圾回收")
    
    def _suggest_gpu_fixes(self):
        """建議GPU記憶體修復方法"""
        print("💡 GPU記憶體洩漏修復建議:")
        print("   1. 確保tensor使用.detach().cpu()移到CPU")
        print("   2. 調用torch.cuda.empty_cache()清理GPU緩存")
        print("   3. 檢查是否有tensor沒有正確釋放")
        print("   4. 減少batch_size")
        print("   5. 使用混合精度訓練(precision=16)")
    
    def manual_check(self, epoch: Optional[int] = None, step: Optional[int] = None):
        """手動檢查記憶體狀態"""
        snapshot = self._take_snapshot()
        snapshot.epoch = epoch
        snapshot.step = step
        self.snapshots.append(snapshot)
        
        print(f"📊 記憶體狀態檢查 (Epoch {epoch}, Step {step}):")
        print(f"   CPU記憶體: {snapshot.cpu_memory_mb:.1f} MB")
        if torch.cuda.is_available():
            print(f"   GPU記憶體: {snapshot.gpu_memory_mb:.1f} MB")
            print(f"   GPU緩存: {snapshot.gpu_cached_mb:.1f} MB")
            print(f"   GPU Tensor數量: {snapshot.num_tensors}")
        
        # 與基線比較
        if self.baseline_cpu:
            cpu_diff = snapshot.cpu_memory_mb - self.baseline_cpu
            print(f"   CPU增長: {cpu_diff:+.1f} MB")
            
        if self.baseline_gpu and torch.cuda.is_available():
            gpu_diff = snapshot.gpu_memory_mb - self.baseline_gpu
            print(f"   GPU增長: {gpu_diff:+.1f} MB")
    
    def force_cleanup(self):
        """強制清理記憶體"""
        print("🧹 執行強制記憶體清理...")
        
        # Python垃圾回收
        collected = gc.collect()
        print(f"   Python GC回收對象: {collected}")
        
        # GPU記憶體清理
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("   GPU緩存已清理")
        
        # 拍攝清理後快照
        snapshot = self._take_snapshot()
        print(f"   清理後CPU記憶體: {snapshot.cpu_memory_mb:.1f} MB")
        if torch.cuda.is_available():
            print(f"   清理後GPU記憶體: {snapshot.gpu_memory_mb:.1f} MB")
    
    def generate_report(self, save_path: str = "memory_leak_report.png"):
        """生成記憶體使用報告"""
        if len(self.snapshots) < 2:
            print("⚠️ 記憶體快照數據不足，無法生成報告")
            return
        
        timestamps = [s.timestamp / 60 for s in self.snapshots]  # 轉換為分鐘
        cpu_memory = [s.cpu_memory_mb for s in self.snapshots]
        
        plt.figure(figsize=(12, 8))
        
        # CPU記憶體圖
        plt.subplot(2, 1, 1)
        plt.plot(timestamps, cpu_memory, 'b-', linewidth=2, label='CPU Memory')
        if self.baseline_cpu:
            plt.axhline(y=self.baseline_cpu, color='b', linestyle='--', alpha=0.7, label='Baseline')
        plt.xlabel('時間 (分鐘)')
        plt.ylabel('CPU記憶體 (MB)')
        plt.title('CPU記憶體使用趨勢')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # GPU記憶體圖
        if torch.cuda.is_available() and any(s.gpu_memory_mb > 0 for s in self.snapshots):
            plt.subplot(2, 1, 2)
            gpu_memory = [s.gpu_memory_mb for s in self.snapshots]
            gpu_cached = [s.gpu_cached_mb for s in self.snapshots]
            
            plt.plot(timestamps, gpu_memory, 'r-', linewidth=2, label='GPU Allocated')
            plt.plot(timestamps, gpu_cached, 'orange', linewidth=2, label='GPU Cached')
            if self.baseline_gpu:
                plt.axhline(y=self.baseline_gpu, color='r', linestyle='--', alpha=0.7, label='Baseline')
            plt.xlabel('時間 (分鐘)')
            plt.ylabel('GPU記憶體 (MB)')
            plt.title('GPU記憶體使用趨勢')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 記憶體使用報告已保存: {save_path}")
    
    def get_summary(self) -> Dict:
        """獲取記憶體使用摘要"""
        if not self.snapshots:
            return {}
        
        latest = self.snapshots[-1]
        
        summary = {
            "monitoring_duration_minutes": latest.timestamp / 60,
            "current_cpu_memory_mb": latest.cpu_memory_mb,
            "current_gpu_memory_mb": latest.gpu_memory_mb,
            "current_gpu_tensors": latest.num_tensors,
            "total_snapshots": len(self.snapshots)
        }
        
        if self.baseline_cpu:
            summary["cpu_memory_growth_mb"] = latest.cpu_memory_mb - self.baseline_cpu
            
        if self.baseline_gpu:
            summary["gpu_memory_growth_mb"] = latest.gpu_memory_mb - self.baseline_gpu
        
        return summary


# 全局檢測器實例
_global_detector = None

def get_memory_detector() -> MemoryLeakDetector:
    """獲取全局記憶體檢測器"""
    global _global_detector
    if _global_detector is None:
        _global_detector = MemoryLeakDetector()
    return _global_detector

def start_memory_monitoring():
    """啟動記憶體監控"""
    detector = get_memory_detector()
    detector.start_monitoring()

def stop_memory_monitoring():
    """停止記憶體監控"""
    detector = get_memory_detector()
    detector.stop_monitoring()

def check_memory(epoch: Optional[int] = None, step: Optional[int] = None):
    """檢查記憶體狀態"""
    detector = get_memory_detector()
    detector.manual_check(epoch, step)

def cleanup_memory():
    """清理記憶體"""
    detector = get_memory_detector()
    detector.force_cleanup()

def generate_memory_report(save_path: str = "memory_leak_report.png"):
    """生成記憶體報告"""
    detector = get_memory_detector()
    detector.generate_report(save_path)


if __name__ == "__main__":
    # 測試記憶體檢測器
    print("🧪 測試記憶體洩漏檢測器...")
    
    detector = MemoryLeakDetector(check_interval=2.0)
    detector.start_monitoring()
    
    try:
        # 模擬記憶體使用
        tensors = []
        for i in range(5):
            print(f"創建tensor {i+1}...")
            if torch.cuda.is_available():
                tensor = torch.randn(1000, 1000).cuda()
            else:
                tensor = torch.randn(1000, 1000)
            tensors.append(tensor)
            
            detector.manual_check(epoch=i)
            time.sleep(3)
        
        # 清理
        print("清理tensors...")
        del tensors
        detector.force_cleanup()
        detector.manual_check(epoch=5)
        
        time.sleep(5)
        
    finally:
        detector.stop_monitoring()
        detector.generate_report("test_memory_report.png")
        
        summary = detector.get_summary()
        print("\n📊 記憶體使用摘要:")
        for key, value in summary.items():
            print(f"   {key}: {value}") 