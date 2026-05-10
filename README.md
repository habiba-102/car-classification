# Car Classification with Image Preprocessing

A complete PyTorch pipeline that classifies car images into **3 brands**
(Audi · Lamborghini · Mercedes) using transfer learning (ResNet-18) with
**22 image preprocessing techniques** applied end-to-end.

---

## Project Structure

```
car_classification/
├── dataset/
│   └── Images/
│       ├── Train/
│       │   ├── audi/          # 20 images
│       │   ├── lamborghini/   # 19 images
│       │   └── mercedes/      # 25 images
│       └── Test/
│           ├── audi/          # 9 images
│           ├── lamborghini/   # 30 images
│           └── mercedes/      # 19 images
├── src/
│   ├── preprocessing.py           # All 22 preprocessing techniques
│   ├── dataset.py                 # PyTorch Dataset + DataLoader factory
│   ├── model.py                   # ResNet-18 transfer-learning head
│   ├── train.py                   # Training loop
│   ├── evaluate.py                # Test-set evaluation + confusion matrix
│   └── visualize_preprocessing.py # Save technique grids per class
├── outputs/                       # Auto-created; holds models + plots
├── requirements.txt
└── README.md
```

---

## Preprocessing Techniques

All techniques are implemented in `src/preprocessing.py`:

| # | Technique | Purpose |
|---|-----------|---------|
| 1 | **Resize / Rescale** | Standardise spatial dimensions to 224×224 |
| 2 | **Grayscale Conversion** | Remove colour bias |
| 3 | **CLAHE Histogram Equalisation** | Boost local contrast in LAB space |
| 4 | **Gaussian Blur** | Denoise; smooth high-frequency artifacts |
| 5 | **Median Blur** | Remove salt-and-pepper noise |
| 6 | **Unsharp Mask Sharpening** | Enhance edges and fine detail |
| 7 | **Min-Max Normalisation** | Scale pixel values to [0, 1] |
| 8 | **ImageNet Standardisation** | Zero-mean / unit-std per channel |
| 9 | **Sobel Edge Detection** | Gradient magnitude map |
| 10 | **Canny Edge Detection** | Thin, clean edge map |
| 11 | **Morphological Erosion** | Shrink bright regions |
| 12 | **Morphological Dilation** | Expand bright regions |
| 13 | **Morphological Opening** | Remove small bright noise |
| 14 | **Morphological Closing** | Fill small dark holes |
| 15 | **Pad to Square** | Zero-pad to equal dimensions before resize |
| 16 | **Center Crop** | Focus on central region |
| 17 | **Horizontal Flip** | Augmentation: mirror image |
| 18 | **Random Rotation (±20°)** | Augmentation: orientation invariance |
| 19 | **Color Jitter** | Augmentation: brightness/contrast/saturation |
| 20 | **Random Crop + Resize** | Augmentation: scale invariance |
| 21 | **Perspective Warp** | Augmentation: viewpoint variation |
| 22 | **Random Erasing** | Augmentation: occlusion robustness |

During **training**, techniques 15–22 are applied stochastically via the
`torchvision.transforms` pipeline in `src/dataset.py`.
During **inference**, only resize + centre-crop + ImageNet normalisation are used.

---

## Setup

### 1. Clone / download the project

```bash
# Unzip the downloaded archive into car_classification/
unzip archive.zip -d car_classification/
cd car_classification/
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **Python 3.9+** recommended. CUDA is optional but speeds up training significantly.

---

## Usage

### A. Visualise all preprocessing techniques

Generates a 6×4 grid image per class saved to `outputs/`.

```bash
python src/visualize_preprocessing.py \
    --data_dir dataset/Images \
    --out_dir  outputs
```

Output files:
- `outputs/preprocessing_audi.png`
- `outputs/preprocessing_lamborghini.png`
- `outputs/preprocessing_mercedes.png`

---

### B. Train the model

```bash
python src/train.py \
    --data_dir    dataset/Images \
    --out_dir     outputs \
    --epochs      30 \
    --batch_size  16 \
    --img_size    224 \
    --lr          1e-4
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_dir` | `dataset/Images` | Root with `Train/` and `Test/` |
| `--out_dir` | `outputs` | Where to save checkpoints |
| `--epochs` | `20` | Number of training epochs |
| `--batch_size` | `16` | Batch size |
| `--img_size` | `224` | Input resolution |
| `--lr` | `1e-4` | Initial learning rate |
| `--num_workers` | `0` | DataLoader worker processes |

Training saves:
- `outputs/best_model.pth` — checkpoint with highest validation accuracy
- `outputs/training_curves.png` — loss and accuracy plots
- `outputs/history.json` — raw metrics per epoch
- `outputs/class_to_idx.json` — class label mapping

> **Tip:** With pretrained ResNet-18 weights and ≥20 epochs you should reach
> **80–95% test accuracy**. Without pretrained weights (no internet) expect
> ~50% as a baseline that improves with more epochs.

---

### C. Evaluate on the test set

```bash
python src/evaluate.py \
    --checkpoint outputs/best_model.pth \
    --data_dir   dataset/Images \
    --out_dir    outputs
```

Prints a per-class precision/recall/F1 report and saves:
- `outputs/confusion_matrix.png`

---

## Model Architecture

```
ResNet-18 (pretrained ImageNet backbone)
    └── FC head replaced with:
        Dropout(0.4) → Linear(512 → 256) → ReLU → Dropout(0.3) → Linear(256 → 3)
```

**Optimizer:** AdamW  |  **Scheduler:** Cosine Annealing  |  **Loss:** CrossEntropy + label smoothing 0.1

---

## Quick-start (all steps at once)

```bash
pip install -r requirements.txt

# 1. Visualise preprocessing
python src/visualize_preprocessing.py

# 2. Train
python src/train.py --epochs 30

# 3. Evaluate
python src/evaluate.py
```

All outputs land in the `outputs/` folder.
