## Data

### 1. Preprocessing (Required)

#### Raw Data to Processed Format
To convert your raw ultrasound data into the required format, please refer to our preprocessing repository: [Image_6D_Preprocessing](https://github.com/Yun0921/Image_6D_Preprocessing/tree/main).

**Input Requirements:**
- Raw ultrasound image frames
- Corresponding 6D position data (CSV format)

**Processing Steps:**
The preprocessing script handles:
- Image resizing to 224×224 pixels
- Single-channel grayscale conversion
- File organization into the required directory structure
- Generation of `transitions_dataset.json` with proper 6-DOF annotations

**Expected Input Structure:**
```
Input/
├── frames_date_N/                     # Raw image files
├── new_{frames_date_N}_final_data.csv # 6D position data
└── data_processing.py                 # Preprocessing script
```

**Generated Output Structure:**
```
process_data/
└── data_date_N/
    ├── ft1/                          # t1 images (source frames)
    ├── ft2/                          # t2 images (target frames)  
    └── transitions_dataset.json      # Paired annotations
```

#### Technical Requirements
- Convert all ultrasound images to single-channel grayscale with a fixed spatial size of 224×224 prior to training.
- Recommended storage formats: PNG or JPG. Avoid spaces and special characters in file names.
- Although the training pipeline also applies `Resize(224, 224)` and normalization at load time, offline preprocessing to 224×224 is required to ensure consistency and reproducibility.
- Runtime normalization uses mean=0.5 and std=0.5. See the transform utilities in [`src/data/augmentation.py`](../src/data/augmentation.py) and the data loader setup in [`src/training/data_loaders.py`](../src/training/data_loaders.py).

### 2. Dataset Location and Directory Layout
- Root directory: `data/processed/`
- One folder per patient, preferably reusing the original naming convention (e.g., `data_0513_01`).
- Each patient folder must contain:
  - `ft1/`: t1 images (source frames)
  - `ft2/`: t2 images (target/future frames)
  - `transitions_dataset.json`: paired image records and 6‑DOF annotations (relative paths and numeric values)

Example (reflecting the actual structure):
```
Cardiac-Dreamer/
  data/
    processed/
      data_0513_01/
        ft1/
          000001.png
          000002.png
          ...
        ft2/
          000001.png
          000002.png
          ...
        transitions_dataset.json
      data_0513_02/
        ft1/
        ft2/
        transitions_dataset.json
      ...
```

Recommendation: Use paths relative to the patient folder inside `transitions_dataset.json` (e.g., `ft1/000001.png`). The loader will resolve them into absolute paths (see [`src/data/dataset.py`](../src/data/dataset.py)). Windows backslashes (\\) are normalized to forward slashes (/).

### 3. transitions_dataset.json Format
Each patient folder contains a `transitions_dataset.json` file with an array of transition entries:
```json
[
  {
    "ft1_image_path": "ft1/000001.png",
    "ft2_image_path": "ft2/000001.png",
    "at1_6dof": [x, y, z, roll, pitch, yaw],
    "action_change_6dof": [dx, dy, dz, droll, dpitch, dyaw],
    "at2_6dof": [x2, y2, z2, roll2, pitch2, yaw2]
  }
]
```
- `ft1_image_path`: relative path to the t1 image (recommended: `ft1/<filename>`)
- `ft2_image_path`: relative path to the t2 image (recommended: `ft2/<filename>`)
- `at1_6dof`: absolute 6‑DOF pose at t1, ordered as `[X, Y, Z, Roll, Pitch, Yaw]`
- `action_change_6dof`: relative motion from t1 to t2 (primary action input)
- `at2_6dof`: absolute 6‑DOF pose at t2

Units and coordinate frames must match your annotation pipeline (translations typically in mm, rotations in rad). Conversions between 6‑DOF vectors and 4×4 homogeneous matrices are implemented in [`src/utils/transformation_utils.py`](../src/utils/transformation_utils.py).

Note: The dataset loader normalizes 6‑DOF values using statistics computed on the training set. These are saved to `data/processed/normalization_stats.json` during training initialization; validation/test subsequently load the same statistics. See [`src/data/dataset.py`](../src/data/dataset.py).

### 4. Loading and Splitting (For Reference and Validation)
- Primary dataset class: `CrossPatientTransitionsDataset` (see [`src/data/dataset.py`](../src/data/dataset.py)).
- Patient splitting: automatic detection or manual specification; cross‑validation uses predefined groups (see [`src/data/patient_splits.py`](../src/data/patient_splits.py) and [`src/train_cross_validation.py`](../src/train_cross_validation.py)).
- Transforms and augmentation:
  - Training: `Resize(224,224) → (optional) custom augmentation → ToTensor → Normalize(0.5,0.5)` (see [`src/data/augmentation.py`](../src/data/augmentation.py)).
  - Validation/Test: `Resize(224,224) → ToTensor → Normalize(0.5,0.5)` (see [`src/train_modular.py`](../src/train_modular.py)).

### 5. Minimal Verification Checklist
- All images are 224×224, single‑channel grayscale.
- Each `data/processed/<patient_id>/` contains `ft1/`, `ft2/`, and `transitions_dataset.json`.
- `ft1_image_path` and `ft2_image_path` in `transitions_dataset.json` point to existing files via relative paths.
- 6‑DOF values are reasonable (consistent ranges and units).
- `CrossPatientTransitionsDataset` can load and batch successfully (use `test_data_loading` in [`src/training/data_loaders.py`](../src/training/data_loaders.py) for a quick check). 