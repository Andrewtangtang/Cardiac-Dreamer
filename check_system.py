#!/usr/bin/env python
"""
System Check - 訓練前系統檢查
檢查硬體資源和環境是否適合訓練
"""

import psutil
import torch
import sys
import os

def check_gpu():
    """檢查GPU狀態"""
    print("🎮 GPU 檢查:")
    
    if not torch.cuda.is_available():
        print("   ❌ CUDA 不可用")
        return False
    
    gpu_count = torch.cuda.device_count()
    print(f"   ✅ 發現 {gpu_count} 個GPU")
    
    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        print(f"   📱 GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
        
        # 檢查GPU記憶體
        if gpu_memory < 4:
            print(f"   ⚠️  GPU {i} 記憶體較少 ({gpu_memory:.1f}GB)，建議使用更小的批次大小")
        elif gpu_memory >= 8:
            print(f"   ✅ GPU {i} 記憶體充足")
    
    return True

def check_memory():
    """檢查系統記憶體"""
    print("\n💾 系統記憶體檢查:")
    
    memory = psutil.virtual_memory()
    total_gb = memory.total / (1024**3)
    available_gb = memory.available / (1024**3)
    used_percent = memory.percent
    
    print(f"   📊 總記憶體: {total_gb:.1f}GB")
    print(f"   📊 可用記憶體: {available_gb:.1f}GB")
    print(f"   📊 使用率: {used_percent:.1f}%")
    
    if total_gb < 8:
        print("   ⚠️  系統記憶體較少，建議關閉其他程式")
        return False
    elif used_percent > 80:
        print("   ⚠️  記憶體使用率過高，建議關閉其他程式")
        return False
    else:
        print("   ✅ 記憶體狀態良好")
        return True

def check_cpu():
    """檢查CPU狀態"""
    print("\n🖥️  CPU 檢查:")
    
    cpu_count = psutil.cpu_count()
    cpu_percent = psutil.cpu_percent(interval=1)
    
    print(f"   📊 CPU 核心數: {cpu_count}")
    print(f"   📊 CPU 使用率: {cpu_percent:.1f}%")
    
    if cpu_count < 4:
        print("   ⚠️  CPU 核心數較少，建議減少 num_workers")
        return False
    elif cpu_percent > 80:
        print("   ⚠️  CPU 使用率過高，建議關閉其他程式")
        return False
    else:
        print("   ✅ CPU 狀態良好")
        return True

def check_disk():
    """檢查磁碟空間"""
    print("\n💿 磁碟空間檢查:")
    
    disk = psutil.disk_usage('.')
    total_gb = disk.total / (1024**3)
    free_gb = disk.free / (1024**3)
    used_percent = (disk.used / disk.total) * 100
    
    print(f"   📊 總空間: {total_gb:.1f}GB")
    print(f"   📊 可用空間: {free_gb:.1f}GB")
    print(f"   📊 使用率: {used_percent:.1f}%")
    
    if free_gb < 5:
        print("   ⚠️  磁碟空間不足，建議清理空間")
        return False
    elif used_percent > 90:
        print("   ⚠️  磁碟使用率過高")
        return False
    else:
        print("   ✅ 磁碟空間充足")
        return True

def check_dependencies():
    """檢查依賴套件"""
    print("\n📦 依賴套件檢查:")
    
    required_packages = [
        'torch', 'torchvision', 'pytorch_lightning',
        'wandb', 'numpy', 'PIL', 'psutil'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} (缺失)")
            missing_packages.append(package)
    
    # 檢查可選套件
    optional_packages = ['GPUtil']
    for package in optional_packages:
        try:
            __import__(package)
            print(f"   ✅ {package} (可選)")
        except ImportError:
            print(f"   ⚠️  {package} (可選，建議安裝用於GPU監控)")
    
    if missing_packages:
        print(f"   ❌ 缺少必要套件: {missing_packages}")
        return False
    else:
        print("   ✅ 所有必要套件已安裝")
        return True

def check_data():
    """檢查數據狀態"""
    print("\n📁 數據檢查:")
    
    data_dir = "data/processed"
    if not os.path.exists(data_dir):
        print(f"   ❌ 數據目錄不存在: {data_dir}")
        return False
    
    # 檢查患者數據
    patient_dirs = [d for d in os.listdir(data_dir) 
                   if os.path.isdir(os.path.join(data_dir, d)) and d.startswith("data_")]
    
    print(f"   📊 發現 {len(patient_dirs)} 個患者目錄")
    
    valid_patients = 0
    for patient_dir in patient_dirs:
        patient_path = os.path.join(data_dir, patient_dir)
        json_file = os.path.join(patient_path, "transitions_dataset.json")
        ft1_dir = os.path.join(patient_path, "ft1")
        
        if os.path.exists(json_file) and os.path.exists(ft1_dir):
            ft1_files = len([f for f in os.listdir(ft1_dir) if f.endswith('.png')])
            if ft1_files > 0:
                valid_patients += 1
                print(f"   ✅ {patient_dir}: {ft1_files} 個影像")
            else:
                print(f"   ⚠️  {patient_dir}: 無影像檔案")
        else:
            print(f"   ❌ {patient_dir}: 缺少必要檔案")
    
    if valid_patients == 0:
        print("   ❌ 沒有有效的患者數據")
        return False
    elif valid_patients == 1:
        print("   ⚠️  只有1個有效患者，將無法進行交叉驗證")
        return True
    else:
        print(f"   ✅ {valid_patients} 個有效患者")
        return True

def get_recommendations():
    """獲取訓練建議"""
    print("\n💡 訓練建議:")
    
    # 檢查GPU記憶體
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if gpu_memory < 6:
            print("   🔧 建議使用 batch_size=2 或更小")
        elif gpu_memory < 8:
            print("   🔧 建議使用 batch_size=4")
        else:
            print("   🔧 可以使用 batch_size=8 或更大")
    
    # 檢查系統記憶體
    memory = psutil.virtual_memory()
    total_gb = memory.total / (1024**3)
    if total_gb < 16:
        print("   🔧 建議使用 num_workers=2")
    else:
        print("   🔧 可以使用 num_workers=4")
    
    print("   🔧 建議使用 precision=16 (混合精度) 節省記憶體")
    print("   🔧 建議關閉不必要的程式釋放資源")

def main():
    """主函數"""
    print("🔍 Cardiac Dreamer 系統檢查")
    print("=" * 50)
    
    checks = [
        check_dependencies(),
        check_gpu(),
        check_memory(),
        check_cpu(),
        check_disk(),
        check_data()
    ]
    
    print("\n" + "=" * 50)
    print("📋 檢查結果:")
    
    passed = sum(checks)
    total = len(checks)
    
    if passed == total:
        print(f"   ✅ 所有檢查通過 ({passed}/{total})")
        print("   🚀 系統準備就緒，可以開始訓練！")
    elif passed >= total - 1:
        print(f"   ⚠️  大部分檢查通過 ({passed}/{total})")
        print("   🔧 建議修復警告後再開始訓練")
    else:
        print(f"   ❌ 多項檢查失敗 ({passed}/{total})")
        print("   🛠️  請修復問題後再嘗試訓練")
    
    get_recommendations()
    
    print("\n🛡️  建議使用安全訓練模式:")
    print("   python safe_train.py")

if __name__ == "__main__":
    main() 