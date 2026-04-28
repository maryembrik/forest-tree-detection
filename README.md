# Forest Dead Tree Detection

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15%2B-orange?logo=tensorflow)](https://tensorflow.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red?logo=pytorch)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?logo=streamlit)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

Automatic detection of dead and diseased trees from aerial RGB satellite imagery using deep learning segmentation models.

---

## Overview

This project performs **binary pixel-level segmentation** — classifying every pixel as either dead/stressed tree or healthy background — directly from 224×224 aerial RGB patches. Three model families are compared:

| Model | Dice | IoU | Precision | Recall | Accuracy |
|-------|:----:|:---:|:---------:|:------:|:--------:|
| **U-Net (original)** | 0.548 | 0.377 | 0.568 | 0.529 | 98.1% |
| U-Net (EfficientNetB0) | >0.75* | >0.60* | ~0.78* | ~0.73* | — |
| Mask R-CNN (ResNet50-FPN) | 0.22 | 0.14 | — | — | — |
| Random Forest (RGB+LBP+NDVI) | 0.47 | 0.32 | 0.55 | 0.42 | — |

> *Estimated after fine-tuning with EfficientNetB0 encoder + attention gates + combined Dice+Focal loss. Retrain to verify.

---

## Pipeline

```
  Aerial RGB Images (224×224 px)
           │
           ▼
  ┌─────────────────────────────┐
  │       Pre-processing        │
  │  Normalize + Augmentation   │
  │  (flip, rotate, elastic,    │
  │   CLAHE, Gaussian noise)    │
  └────────────┬────────────────┘
               │
     ┌─────────┼──────────────────────┐
     ▼         ▼                      ▼
┌──────────┐ ┌────────────────┐ ┌──────────────┐
│  U-Net   │ │  Mask R-CNN    │ │    Random    │
│EfficientB0│ │ ResNet50-FPN  │ │    Forest    │
│+Attention│ │Instance Seg.  │ │RGB+LBP+NDVI │
│+Dice+Focal│ │               │ │200 estimators│
└────┬─────┘ └────────────────┘ └──────────────┘
     │
     ▼
  ┌──────────────────┐
  │ Post-processing  │
  │ Threshold 0.5    │
  │ Connected blobs  │
  │ Tree count       │
  └────────┬─────────┘
           ▼
  ┌─────────────────────────────────────┐
  │        Streamlit Dashboard          │
  │  Map Explorer · Analytics           │
  │  Model Comparison · Report Export   │
  └─────────────────────────────────────┘
```

---

## Dataset

| Attribute | Value |
|-----------|-------|
| Images | 444 RGB PNG, 224×224 px |
| Masks | Binary — dead tree pixel = 1, background = 0 |
| Counties | AR037, AR039, AR041, AR081, AR145 (Arkansas) · MO025, MO049 (Missouri) |
| Years | 2018, 2019, 2021 |
| Class balance | ~2% positive pixels (severely imbalanced) |
| Split | 70% train / 15% val / 15% test |

The dataset is **not included** in this repository due to size. Place it at `../USA_segmentation/` relative to the project root:

```
FORESTesprit/
├── forest-tree-detection/   ← this repo
└── USA_segmentation/
    ├── RGB_images/
    ├── masks/
    └── best_unet_model.h5
```

---

## Project Structure

```
forest-tree-detection/
├── app/
│   └── streamlit_app.py       # 4-page dark-theme dashboard
├── src/
│   ├── dataset.py             # Data loading, augmentation, TTA
│   ├── metrics.py             # Dice, IoU, Precision, Recall, F1
│   ├── train.py               # Training CLI (U-Net + Random Forest)
│   ├── inference.py           # Batch inference + TTA CLI
│   └── models/
│       ├── unet.py            # EfficientNetB0 U-Net + attention gates
│       ├── maskrcnn.py        # Mask R-CNN ResNet50-FPN
│       └── random_forest.py   # RF with per-pixel RGB+LBP+NDVI features
├── notebooks/
│   ├── forestEsprit.ipynb     # Original notebook (bugs fixed)
│   └── demo.ipynb             # Quick demo: load → predict → visualize
├── tests/                     # pytest unit tests
├── docs/
│   ├── METHODOLOGY.md         # Technical explanations
│   └── RESULTS.md             # Full metrics + bug analysis
├── outputs/                   # Model checkpoints, predictions (git-ignored)
├── config.yaml                # All hyperparameters
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── Makefile
└── setup.py
```

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/<your-username>/forest-tree-detection.git
cd forest-tree-detection

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch dashboard  (auto-loads best_unet_model.h5 from ../USA_segmentation/)
python -m streamlit run app/streamlit_app.py

# 4. Train U-Net (optional — GPU recommended)
python src/train.py --config config.yaml --model unet

# 5. Train Random Forest baseline
python src/train.py --config config.yaml --model rf
```

---

## Models

### U-Net (original baseline)
- Custom 5-level encoder-decoder (64→128→256→512→1024 channels)
- Trained from scratch with Dice loss only
- **Dice: 0.548 · IoU: 0.377 · Accuracy: 98.1%** on 67 test images

### U-Net (improved — EfficientNetB0)
- EfficientNetB0 encoder pretrained on ImageNet
- Attention gates on all skip connections
- Combined loss: `0.5 × Dice + 0.5 × Focal` (handles 2% class imbalance)
- AdamW optimizer · ReduceLROnPlateau · EarlyStopping(patience=15)
- **Expected Dice > 0.75** after full training

### Mask R-CNN
- ResNet50-FPN backbone (torchvision)
- Instance segmentation (counts individual trees)
- **Dice: 0.22** with pre-trained backbone only

### Random Forest baseline
- 11 per-pixel features: R, G, B, pseudo-NDVI, LBP, 3×3 patch mean/std
- 200 trees · `class_weight='balanced'`
- **Dice: 0.47 · IoU: 0.32** (no GPU needed, fully interpretable)

---

## Key Bug Fixed

The original notebook showed **U-Net Dice = 0.03** (vs Mask R-CNN Dice = 0.22), making it look 7× worse. This was **entirely a filename mismatch**, not a model failure:

```python
# BUG — sequential names that never match ground truth:
cv2.imwrite(f"unet_mask_{i+1}.png", mask)

# FIX — use original identifiers:
cv2.imwrite(f"unet_{original_stem}.png", mask)
# e.g.  unet_ar037_2019_n_06_04_0.png
```

Ground-truth masks are named `mask_ar037_2019_n_06_04_0.png`. The intersection of the two filename sets was empty → Dice computed on zero samples ≈ 0.03 (smoothing term only). Real U-Net Dice = **0.548**.

---

## Dashboard Pages

| Page | What you see |
|------|-------------|
| **Map Explorer** | Upload any aerial PNG → instant dead-tree segmentation overlay + VARI heatmap + probability map |
| **Analytics** | Pixel-class pie chart, stacked bar chart, Forest Health Index gauge |
| **Model Comparison** | Metrics table + grouped bar chart comparing all 4 models + root-cause explanation of the Dice=0.03 bug |
| **Report & Export** | One-click PDF report, mask PNG download, metrics CSV, GeoJSON export |

---

## Configuration

Edit `config.yaml` to change paths, hyperparameters, or thresholds:

```yaml
data:
  image_dir: "../USA_segmentation/RGB_images"
  mask_dir:  "../USA_segmentation/masks"
  img_height: 224
  img_width:  224

training:
  batch_size: 16
  epochs: 100
  learning_rate: 0.001
  patience_lr: 5
  patience_stop: 15

model:
  threshold: 0.5
  backbone: "EfficientNetB0"
```

---

## Requirements

```
tensorflow>=2.15
torch>=2.0
torchvision>=0.15
streamlit>=1.32
opencv-python
Pillow
numpy
scikit-learn
scikit-image
albumentations
plotly
folium
streamlit-folium
pyyaml
tqdm
joblib
reportlab
scipy
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

## License

MIT — see [LICENSE](LICENSE).
