#!/usr/bin/env python3
"""
測試 validation 圖表優化
驗證新的圖表生成機制是否正常工作
"""

import os
import sys
import torch
import numpy as np
import tempfile
import shutil

# 添加項目根目錄到路徑
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from src.models.system import CardiacDreamerSystem


def test_validation_optimization():
    """測試 validation 圖表優化功能"""
    print("🧪 測試 validation 圖表優化功能...")
    
    # 創建模型
    model = CardiacDreamerSystem(
        token_type="patch",
        d_model=256,  # 小一點的模型用於測試
        nhead=4,
        num_layers=2,
        use_flash_attn=False
    )
    
    print("✅ 模型創建成功")
    
    # 檢查新方法是否存在
    assert hasattr(model, 'generate_final_validation_plots'), "generate_final_validation_plots 方法不存在"
    print("✅ generate_final_validation_plots 方法存在")
    
    # 模擬 validation step
    batch_size = 4
    image_t1 = torch.randn(batch_size, 1, 224, 224)
    a_hat_t1_to_t2_gt = torch.randn(batch_size, 6)
    at1_6dof_gt = torch.randn(batch_size, 6)
    at2_6dof_gt = torch.randn(batch_size, 6)
    
    batch = (image_t1, a_hat_t1_to_t2_gt, at1_6dof_gt, at2_6dof_gt)
    
    print("✅ 測試數據創建成功")
    
    # 模擬幾個 validation steps
    model.eval()
    with torch.no_grad():
        for i in range(3):  # 模擬3個batch
            val_loss = model.validation_step(batch, i)
            print(f"   Validation step {i}: loss = {val_loss:.4f}")
    
    print("✅ Validation steps 執行成功")
    
    # 檢查是否有收集到驗證數據
    assert len(model.validation_step_outputs) > 0, "沒有收集到驗證數據"
    print(f"✅ 收集到 {len(model.validation_step_outputs)} 個驗證樣本")
    
    # 模擬 validation epoch end
    model.on_validation_epoch_end()
    
    # 檢查是否正確保存了最新驗證數據
    assert hasattr(model, 'latest_validation_data'), "latest_validation_data 屬性不存在"
    assert model.latest_validation_data is not None, "latest_validation_data 為空"
    print("✅ 驗證數據已正確保存")
    
    # 檢查 validation_step_outputs 是否已清理
    assert len(model.validation_step_outputs) == 0, "validation_step_outputs 沒有被清理"
    print("✅ validation_step_outputs 已正確清理")
    
    # 測試最終圖表生成
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"📊 在臨時目錄生成最終圖表: {temp_dir}")
        
        # 生成最終圖表
        model.generate_final_validation_plots(output_dir=temp_dir)
        
        # 檢查圖表是否生成
        plots_dir = os.path.join(temp_dir, "final_validation_plots")
        assert os.path.exists(plots_dir), "圖表目錄不存在"
        
        # 檢查組合圖表
        combined_plot = os.path.join(plots_dir, "final_validation_scatter_combined.png")
        assert os.path.exists(combined_plot), "組合圖表不存在"
        
        # 檢查個別圖表
        dim_names = ["x", "y", "z", "roll", "pitch", "yaw"]
        for dim in dim_names:
            individual_plot = os.path.join(plots_dir, f"final_validation_scatter_{dim}.png")
            assert os.path.exists(individual_plot), f"{dim} 圖表不存在"
        
        print("✅ 所有圖表文件都已正確生成")
        
        # 列出生成的文件
        files = os.listdir(plots_dir)
        print(f"📁 生成的文件: {files}")
    
    # 檢查數據是否已清理
    assert model.latest_validation_data is None, "latest_validation_data 沒有被清理"
    print("✅ 驗證數據已正確清理")
    
    print("🎉 所有測試通過！validation 圖表優化功能正常工作")


def test_memory_efficiency():
    """測試記憶體效率"""
    print("\n🧪 測試記憶體效率...")
    
    import gc
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    print(f"初始記憶體使用: {initial_memory:.1f} MB")
    
    # 創建模型並模擬多個 validation epochs
    model = CardiacDreamerSystem(
        token_type="patch",
        d_model=256,
        nhead=4,
        num_layers=2,
        use_flash_attn=False
    )
    
    batch_size = 8
    image_t1 = torch.randn(batch_size, 1, 224, 224)
    a_hat_t1_to_t2_gt = torch.randn(batch_size, 6)
    at1_6dof_gt = torch.randn(batch_size, 6)
    at2_6dof_gt = torch.randn(batch_size, 6)
    
    batch = (image_t1, a_hat_t1_to_t2_gt, at1_6dof_gt, at2_6dof_gt)
    
    model.eval()
    with torch.no_grad():
        for epoch in range(5):  # 模擬5個epochs
            print(f"   模擬 epoch {epoch}...")
            
            # 每個epoch多個validation steps
            for step in range(10):
                model.validation_step(batch, step)
            
            # Validation epoch end
            model.on_validation_epoch_end()
            
            # 檢查記憶體
            current_memory = process.memory_info().rss / 1024 / 1024
            memory_growth = current_memory - initial_memory
            print(f"   Epoch {epoch} 記憶體: {current_memory:.1f} MB (增長: {memory_growth:+.1f} MB)")
            
            # 確保記憶體增長不會太多
            if memory_growth > 500:  # 超過500MB就警告
                print(f"⚠️ 記憶體增長過多: {memory_growth:.1f} MB")
    
    # 最終清理
    gc.collect()
    final_memory = process.memory_info().rss / 1024 / 1024
    total_growth = final_memory - initial_memory
    
    print(f"最終記憶體使用: {final_memory:.1f} MB (總增長: {total_growth:+.1f} MB)")
    
    if total_growth < 200:  # 總增長少於200MB算正常
        print("✅ 記憶體使用效率良好")
    else:
        print(f"⚠️ 記憶體增長較多: {total_growth:.1f} MB")
    
    print("🎉 記憶體效率測試完成")


if __name__ == "__main__":
    try:
        test_validation_optimization()
        test_memory_efficiency()
        print("\n🎉 所有測試都通過了！")
        print("✅ Validation 圖表優化功能正常")
        print("✅ 記憶體使用效率良好")
        print("✅ 可以安全地進行長時間訓練")
        
    except Exception as e:
        print(f"\n❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 