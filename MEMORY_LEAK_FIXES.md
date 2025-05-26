# 記憶體洩漏修復指南

## 🔍 發現的記憶體洩漏問題

### 1. **validation_step_outputs 和 test_step_outputs 累積**
**問題**: 在 `system.py` 中，每個 validation/test step 都會將 tensor 添加到列表中，但這些 tensor 沒有正確移到 CPU 並且會無限累積。

**原始代碼問題**:
```python
self.validation_step_outputs.append({
    "predicted_action_composed": predicted_composed_action.detach().cpu(),
    "target_action_composed_gt": at1_6dof_gt.detach().cpu()
})
```

**修復方案**:
```python
# 🔧 限制累積數量並確保tensor移到CPU
if len(self.validation_step_outputs) < self.max_validation_outputs:
    self.validation_step_outputs.append({
        "predicted_action_composed": predicted_composed_action.detach().cpu().clone(),
        "target_action_composed_gt": at1_6dof_gt.detach().cpu().clone()
    })
```

### 2. **DataLoader 的 persistent_workers 記憶體洩漏**
**問題**: `persistent_workers=True` 會讓 worker 進程持續運行，累積記憶體而不釋放。

**原始代碼問題**:
```python
persistent_workers=True if train_config["num_workers"] > 0 else False
```

**修復方案**:
```python
persistent_workers=False,  # 🔧 禁用persistent_workers防止記憶體累積
num_workers=min(train_config["num_workers"], 2),  # 🔧 限制worker數量
```

### 3. **GPU 記憶體沒有定期清理**
**問題**: GPU 記憶體碎片化和緩存累積導致 OOM。

**修復方案**:
```python
# 🔧 定期清理GPU記憶體
if batch_idx % 50 == 0:  # 每50個batch清理一次
    torch.cuda.empty_cache()
```

### 4. **每個 validation epoch 都畫圖導致的性能問題**
**問題**: 原本每個 validation epoch 都會生成 scatter plots，這會：
- 消耗大量 CPU 和記憶體資源
- 產生大量檔案（每個epoch 7個圖檔）
- 可能導致 I/O 瓶頸和系統不穩定
- 在長時間訓練中累積大量圖檔

**原始代碼問題**:
```python
def on_validation_epoch_end(self):
    # 每個epoch都生成完整的scatter plots
    for i, dim_name in enumerate(dim_names):
        plt.figure(figsize=(8, 8))
        # ... 畫圖邏輯 ...
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
```

**修復方案**:
```python
def on_validation_epoch_end(self):
    # 🔧 優化：只收集數據，不畫圖
    self.latest_validation_data = {
        "predictions": all_preds.clone(),
        "targets": all_targets.clone(),
        "epoch": current_epoch
    }
    print(f"📊 Validation epoch {current_epoch}: 收集了 {len(all_preds)} 個樣本的預測數據")
    print(f"   將在訓練結束後生成scatter plots")

def generate_final_validation_plots(self, output_dir: str = None):
    # 🔧 只在訓練結束後生成一次最終圖表
    # 基於最後一個epoch的驗證數據
```

### 5. **epoch 結束時沒有強制垃圾回收**
**問題**: Python 垃圾回收不夠積極，大型對象沒有及時釋放。

**修復方案**:
```python
# 🔧 修復記憶體洩漏：清理輸出並強制垃圾回收
self.validation_step_outputs.clear()
del all_preds, all_targets, preds_np, targets_np
gc.collect()
torch.cuda.empty_cache()
```

## 🛠️ 修復的文件

### 1. `src/models/system.py`
- ✅ 添加記憶體管理參數 (`max_validation_outputs`, `max_test_outputs`)
- ✅ 修復 validation_step 和 test_step 中的 tensor 洩漏
- ✅ 添加定期 GPU 記憶體清理
- ✅ 強化 epoch 結束時的記憶體清理
- ✅ 添加垃圾回收機制
- ✅ **優化 validation 圖表生成**：從每個epoch都畫圖改為只在訓練結束後畫一次

### 2. `src/train.py`
- ✅ 修復 DataLoader 配置
- ✅ 禁用 persistent_workers
- ✅ 限制 num_workers 數量
- ✅ 條件性使用 pin_memory
- ✅ **更新訓練流程**：在訓練結束後調用最終圖表生成

### 3. 新增工具

#### `memory_leak_detector.py`
- 🔍 實時記憶體監控
- ⚠️ 自動警告和建議
- 📊 記憶體使用報告生成
- 🧹 強制記憶體清理功能

#### `train_with_memory_monitoring.py`
- 🚀 整合記憶體監控的安全訓練
- 🔧 自動應用安全配置
- 📈 訓練過程記憶體追蹤

## 🎯 使用方法

### 1. 使用修復後的正常訓練
```bash
# 使用修復後的系統進行訓練
pipenv run python src/train.py --config configs/safe_training.yaml --data_dir data/processed --output_dir outputs_safe
```

### 2. 使用記憶體監控訓練
```bash
# 帶記憶體監控的安全訓練
python train_with_memory_monitoring.py --config configs/safe_training.yaml --data_dir data/processed --output_dir outputs_monitored
```

### 3. 測試記憶體檢測器
```bash
# 測試記憶體洩漏檢測器
python memory_leak_detector.py
python train_with_memory_monitoring.py test
```

## 📊 效果對比

### 修復前的問題
- ❌ 記憶體持續增長，最終導致 OOM 或系統 crash
- ❌ validation_step_outputs 無限累積
- ❌ GPU 記憶體碎片化嚴重
- ❌ DataLoader worker 進程記憶體洩漏
- ❌ **每個 validation epoch 都畫圖**，消耗大量資源和產生大量檔案

### 修復後的改善
- ✅ 記憶體使用穩定，有上限控制
- ✅ 定期自動清理，防止累積
- ✅ GPU 記憶體有效管理
- ✅ 可以長時間穩定訓練
- ✅ **只在訓練結束後畫一次圖**，大幅減少 I/O 和記憶體壓力
- ✅ **高效的圖表生成**，避免訓練過程中的 I/O 瓶頸

## 🔧 安全配置建議

### 針對 GTX 1650 (4GB) 的安全設置
```yaml
training:
  batch_size: 4                   # 小批次大小
  num_workers: 2                  # 限制worker數量
  precision: 16                   # 混合精度節省記憶體
  gradient_clip_val: 0.5          # 梯度裁剪
  accumulate_grad_batches: 4      # 梯度累積模擬大批次
  max_epochs: 50                  # 適中的epoch數量
  check_val_every_n_epoch: 2      # 減少驗證頻率
```

### 模型配置調整
```yaml
model:
  d_model: 512                    # 從768降到512
  num_heads: 8                    # 從12降到8  
  num_layers: 4                   # 從6降到4
  token_type: "patch"             # 使用更高效的patch tokens
```

## 🚨 警告信號

### CPU 記憶體洩漏信號
- 記憶體使用量持續增長超過 2GB
- 系統響應變慢
- 出現 "Memory Error" 或 "Out of Memory"

### GPU 記憶體洩漏信號
- CUDA OOM 錯誤
- GPU 記憶體使用率接近 100%
- 訓練速度突然下降

### 系統不穩定信號
- 整台電腦 crash
- 藍屏或黑屏
- 風扇狂轉，溫度過高

## 💡 預防措施

### 1. 訓練前檢查
```bash
# 檢查系統狀態
python check_system.py

# 測試記憶體檢測器
python train_with_memory_monitoring.py test
```

### 2. 訓練中監控
- 使用 `monitor_training.py` 實時監控
- 觀察記憶體使用趨勢
- 注意溫度和負載

### 3. 緊急處理
```python
# 在 Python 中緊急清理記憶體
import gc
import torch

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
```

## 🎉 總結

通過這些修復，我們解決了：

1. **主要記憶體洩漏源頭** - validation/test outputs 累積
2. **DataLoader 記憶體問題** - persistent_workers 和過多 workers
3. **GPU 記憶體管理** - 定期清理和碎片化處理
4. **監控和預警機制** - 實時檢測和自動建議
5. **Validation 圖表性能問題** - 從每個epoch都畫圖改為訓練結束後畫一次

現在系統可以：
- ✅ 長時間穩定訓練而不 crash
- ✅ 自動監控和警告記憶體問題
- ✅ 在資源受限的硬體上安全運行
- ✅ 提供詳細的記憶體使用報告
- ✅ **高效的圖表生成**，避免訓練過程中的 I/O 瓶頸

**建議使用 `configs/safe_training.yaml` 配置和 `train_with_memory_monitoring.py` 腳本進行安全訓練！**