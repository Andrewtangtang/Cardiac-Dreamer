import torch
import sys

def check_cuda_status():
    print("=== CUDA 系統檢查報告 ===")
    
    # 檢查 PyTorch 版本
    print(f"PyTorch 版本: {torch.__version__}")
    
    # 檢查 CUDA 是否可用
    cuda_available = torch.cuda.is_available()
    print(f"CUDA 是否可用: {cuda_available}")
    
    if cuda_available:
        # 顯示 CUDA 版本
        print(f"CUDA 版本: {torch.version.cuda}")
        
        # 獲取當前設備
        current_device = torch.cuda.current_device()
        print(f"當前使用的設備編號: {current_device}")
        
        # 獲取所有可用的 GPU 數量
        device_count = torch.cuda.device_count()
        print(f"可用的 GPU 數量: {device_count}")
        
        # 顯示每個 GPU 的資訊
        for i in range(device_count):
            print(f"\nGPU {i} 資訊:")
            print(f"設備名稱: {torch.cuda.get_device_name(i)}")
            print(f"記憶體分配: {torch.cuda.memory_allocated(i) / 1024**2:.2f} MB")
            print(f"記憶體快取: {torch.cuda.memory_reserved(i) / 1024**2:.2f} MB")
            
        # 測試 CUDA 是否正常運作
        try:
            x = torch.rand(10, 10).cuda()
            print("\nCUDA 測試: 成功創建 CUDA tensor")
        except Exception as e:
            print(f"\nCUDA 測試失敗: {str(e)}")
    else:
        print("\n警告: CUDA 不可用！請檢查：")
        print("1. 是否已安裝 NVIDIA GPU 驅動程式")
        print("2. 是否已正確安裝 CUDA Toolkit")
        print("3. PyTorch 是否安裝了 CUDA 版本")

if __name__ == "__main__":
    try:
        check_cuda_status()
        print(torch.version.cuda)
    except Exception as e:
        print(f"執行檢查時發生錯誤: {str(e)}")
    
    print("\n=== 檢查完成 ===")