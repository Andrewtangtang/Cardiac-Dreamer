# Echocardio‑ViT Guidance – **Channel‑Token (512 + Action) Branch**

> **目的**：以 **512 個卷積通道 token + 1 顆 Action‑CLS token** (共 513 token) 進入 Transformer，檢驗「Channel 自注意力」對心臟探頭導航的效果。


---

## 0 · 目錄 

```text
├── README.md               # ← 你正在看
├── requirements.txt        # 同 baseline
├── data/                   # raw / processed
├── notebooks/
│   ├── 01_channel_vis.ipynb  # 通道 token 可視化
│   └── 90_report.ipynb
├── configs/
│   ├── default.yaml          # include model/channel.yaml + train/base
│   ├── model/
│   │   └── channel.yaml      # token_type=channel, d=768, L=6, heads=12
│   └── train/
│       └── base.yaml
├── src/
│   ├── models/
│   │   ├── backbone.py       # 共用 ResNet34
│   │   ├── dreamer_channel.py# ★ 512 token 版本
│   │   ├── guidance.py       # 共用 (512+q → 1024 → 512 → 6)
│   │   └── system.py         # LitModule   token_type 驅動不同 dreamer
│   └── train.py
└── scripts/run_channel.sh
```

---

## 1 · 資料流

```mermaid
flowchart TB
    A[US image 224×224] -->|ResNet34| B[B,512,7,7]
    B -->|flatten 3rd dim| Tok512[B,512,49]
    Tok512 -->|permute| Tok[B,512,49]
    Tok -->|Linear 49→768| TokD[B,512,768]
    act[6‑DoF] -->|Linear 6→768| CLS[B,1,768]
    CLS --> Concat
    TokD --> Concat
    Concat[[B,513,768]\n+ pos_emb] --> Enc(Transformer L=6 H=12)
    Enc --> Out[[B,513,768]]
    Out -->|index 1‑512| Reshape
    Reshape -->|Linear 768→512| Map[B,512,49]
    Map -->|view| Map2D[B,512,7,7]
    Map2D --> GAP[Global AvgPool] --> F[B,512]
    qi[plane query] -->|concat| Fuse
    F --> Fuse
    Fuse --> MLP[Guidance MLP] --> a_hat[Pred 6‑DoF]
```

**重點**

1. `token_type=channel` → **token 數 512**，每 token 向量長 49。
2. Positional encoding `pos_emb` shape `[1,513,768]`：index 0 給 Action‑CLS，其餘 1‑512 為 channel‑ID。
3. Self‑Attention 複雜度 ≈ 262 k 關係 → 建議啟用 **Flash‑Attention** 或減 batch。
4. Guidance MLP 與 baseline 相同：`512+q_dim → 1024 → 512 → 6`。

---

## 2 · 安裝與執行

### 環境設置

```bash
# 安裝依賴
pipenv install

# 進入虛擬環境
pipenv shell
```

### 訓練模型

```bash
# 使用小型數據集進行測試訓練（使用10個樣本）
python src/train.py --data_dir data/processed --small_subset

# 使用小型數據集進行訓練，並指定輸出目錄
python src/train.py --data_dir data/processed --small_subset --output_dir outputs/test_run

# 使用完整數據集進行訓練
python src/train.py --data_dir data/processed --output_dir outputs/full_train

# 自定義訓練參數後訓練
python src/train.py --data_dir data/processed --output_dir outputs/custom_train
```

### 參數說明

- `--data_dir`: 指定資料目錄路徑
- `--output_dir`: 指定輸出目錄路徑（模型檢查點和日誌）
- `--small_subset`: 使用小型資料集進行測試（僅使用前10個樣本）
- `--config`: 指定配置檔案路徑（默認為 configs/default.yaml）

訓練過程會輸出到指定的輸出目錄，包含：
1. 模型檢查點 (`checkpoints/`)
2. TensorBoard 日誌 (`logs/`)

---

## 3 · 主要超參

| key                  | 預設值  | 說明                        |
| -------------------- | ---- | ------------------------- |
| `model.d_model`      | 768  | token hidden size         |
| `model.num_heads`    | 12   | Multi‑head Attention      |
| `model.num_layers`   | 6    | Encoder blocks            |
| `model.flash_attn`   | true | 使用 Flash‑Attention 降 VRAM |
| `train.batch`        | 4    | → 16 GB GPU 建議 2          |
| `loss.lambda_latent` | 0.2  | Dreamer latent loss 權重    |

---
