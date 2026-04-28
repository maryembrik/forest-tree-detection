# Methodology

## 1. How Dead Trees Appear in RGB Satellite Imagery

Dead and dying trees exhibit distinctive visual signatures in aerial RGB imagery:

| Stage | RGB Appearance | Cause |
|-------|---------------|-------|
| **Healthy** | Deep green canopy, high VARI | Active chlorophyll, full leaf cover |
| **Stressed** | Pale green / yellow-green | Reduced photosynthesis, partial leaf loss |
| **Recently dead** | Brown / tan, textured canopy | Dried foliage still attached |
| **Snag (long-dead)** | Grey-white skeleton | No foliage, bare branches |
| **Fallen** | Dark green patch (debris) | Ground-level decomposition |

**VARI (Visible Atmospherically Resistant Index)** is the key spectral proxy for RGB:
```
VARI = (Green - Red) / (Green + Red + ε)
```
Healthy trees → VARI ≈ 0.2–0.5. Dead/stressed → VARI < 0.

## 2. Why Dice + Focal Loss for Sparse Binary Masks

The dataset has ~12% positive pixels (dead trees), making it severely imbalanced.

**Dice Loss** directly optimizes the F1/Dice score:
```
L_dice = 1 - (2 * |Y_true ∩ Y_pred| + ε) / (|Y_true| + |Y_pred| + ε)
```
Advantage: class-imbalance agnostic; penalizes missing dead tree pixels equally.

**Binary Focal Loss** down-weights easy negatives (the dominant background):
```
L_focal = -α * (1 - p_t)^γ * log(p_t)
```
With α=0.25, γ=2.0 — severely misclassified pixels get 4× more weight.

**Combined Loss = 0.5 × Dice + 0.5 × Focal** captures both:
- Global segmentation quality (Dice)
- Hard-example mining (Focal)

## 3. EfficientNetB0 Transfer Learning vs Training from Scratch

**Original approach (from-scratch 5-level U-Net):**
- ~31M parameters initialized randomly
- Needs >1000 images to learn general visual features
- Converges slowly; susceptible to overfitting on 345 images

**Improved approach (EfficientNetB0 encoder):**
- Pretrained on 1.2M ImageNet images → rich feature hierarchy
- ~5.3M encoder parameters (frozen or fine-tuned)
- Decoder learns only task-specific upsampling (~2M params)
- Reaches good convergence in 20–30 epochs vs 60–80 from scratch

## 4. Attention Gates in Skip Connections

Standard U-Net skip connections blindly concatenate encoder features with decoder features. This introduces noise from background regions.

**Attention gate mechanism:**
```
θ(x) = W_θ · x + b_θ           (encoder feature projection)
φ(g) = W_φ · g + b_φ           (decoder gate projection)
α = σ(ReLU(θ(x) + φ(g)))       (soft attention weight ∈ [0,1])
output = α ⊙ x                  (attended features)
```
Where σ is sigmoid. The gate learns to suppress uninformative background pixels, focusing on dead tree regions. This is especially effective for sparse targets.

## 5. Test-Time Augmentation (TTA)

TTA averages predictions over multiple transformed versions of the test image:
1. Original image
2. Horizontal flip
3. Vertical flip
4. Both flips combined

Predictions are un-transformed (flipped back) then averaged:
```
P_TTA = mean(P_orig, flip_h(P_h), flip_v(P_v), flip_hv(P_hv))
```
Typically improves Dice by 1–3% with no additional training cost.

## 6. Mask R-CNN vs U-Net

| Property | U-Net | Mask R-CNN |
|----------|-------|-----------|
| Task | Semantic segmentation | Instance segmentation |
| Output | Per-pixel class | Per-instance binary mask + bounding box |
| Best for | Dense connected regions | Countable discrete objects |
| Our dataset | Better fit (binary masks) | Better for counting individual trees |
| Training data | 345 image-level masks | Needs instance-level annotations |

For this dataset (binary masks, not instance-labeled), U-Net is the primary model.

## 7. Random Forest Features

Per-pixel feature vector (11 dimensions):

| Feature | Dim | Description |
|---------|-----|-------------|
| R, G, B | 3 | Raw channel values [0,1] |
| Pseudo-NDVI | 1 | (R-G)/(R+G+ε) — vegetation proxy |
| LBP | 1 | Local Binary Pattern (radius=1, P=8, uniform) |
| Patch mean (3 channels) | 3 | 3×3 box-filter mean per channel |
| Patch std (3 channels) | 3 | 3×3 local standard deviation per channel |

200-tree ensemble with `class_weight='balanced'`.

## 8. Limitations

| Limitation | Impact | Future Solution |
|-----------|--------|-----------------|
| RGB only (no NIR/NDVI) | Cannot compute true NDVI | Add Sentinel-2 NIR band |
| Small dataset (345 images) | Risk of overfitting | Semi-supervised learning / data collection |
| Binary only (dead vs healthy) | No severity grades | 4-class: healthy/stressed/diseased/dead |
| 224×224 patches | Misses large-scale patterns | Multi-scale inference |
| No temporal analysis | Single-date snapshot | Time-series change detection |
| Arkansas/Missouri only | Geographic bias | Pan-continental training data |

## 9. Future Work

1. **Multispectral bands**: Add Sentinel-2 NIR (B8) for true NDVI
2. **Semi-supervised**: Use 10,000+ unlabeled aerial images with pseudo-labels
3. **Larger backbone**: EfficientNetB4 / ResNet50 — trade speed for accuracy
4. **Temporal modeling**: ConvLSTM over multi-date image stacks for change detection
5. **Active learning**: Human-in-the-loop labeling of hard examples
6. **Uncertainty quantification**: Monte Carlo dropout for confidence intervals
