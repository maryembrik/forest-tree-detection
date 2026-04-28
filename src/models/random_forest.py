"""
Random Forest baseline for dead-tree binary segmentation.
Extracts per-pixel features: RGB, pseudo-NDVI, LBP texture, patch stats.
"""

from __future__ import annotations

import numpy as np
import joblib
from pathlib import Path
from typing import Tuple, Dict

try:
    from skimage.feature import local_binary_pattern
    SKIMAGE_OK = True
except ImportError:
    SKIMAGE_OK = False


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_features(image_float: np.ndarray) -> np.ndarray:
    """Extract per-pixel features from a (H, W, 3) float32 RGB image.

    Features per pixel (11 total):
      - R, G, B                         → 3
      - pseudo_NDVI = (R-G)/(R+G+1e-6)  → 1
      - LBP (radius=1, n_points=8)       → 1
      - 3x3 patch mean per channel       → 3
      - 3x3 patch std per channel        → 3

    Returns
    -------
    features : np.ndarray, shape (H*W, 11)
    """
    import cv2
    H, W, _ = image_float.shape
    r = image_float[:, :, 0]
    g = image_float[:, :, 1]
    b = image_float[:, :, 2]

    # Spectral features
    pseudo_ndvi = (r - g) / (r + g + 1e-6)

    # LBP texture
    if SKIMAGE_OK:
        gray = (0.299 * r + 0.587 * g + 0.114 * b)
        lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
        lbp = lbp / lbp.max() if lbp.max() > 0 else lbp
    else:
        lbp = np.zeros((H, W), dtype=np.float32)

    # Patch stats: 3x3 mean and std per channel (using box filter)
    patch_means = []
    patch_stds = []
    for ch in range(3):
        ch_img = image_float[:, :, ch].astype(np.float32)
        mean_filt = cv2.boxFilter(ch_img, -1, (3, 3))
        mean_sq_filt = cv2.boxFilter(ch_img ** 2, -1, (3, 3))
        std_filt = np.sqrt(np.maximum(mean_sq_filt - mean_filt ** 2, 0))
        patch_means.append(mean_filt)
        patch_stds.append(std_filt)

    feature_maps = [r, g, b, pseudo_ndvi, lbp] + patch_means + patch_stds
    features = np.stack(feature_maps, axis=-1).reshape(H * W, len(feature_maps))
    return features.astype(np.float32)


# ── Training ──────────────────────────────────────────────────────────────────

def train_random_forest(
    X_images: np.ndarray,
    Y_masks: np.ndarray,
    n_estimators: int = 200,
    class_weight: str = "balanced",
    max_samples: int = 50_000,
    random_state: int = 42,
):
    """Train Random Forest on pixel-level features.

    Parameters
    ----------
    X_images : (N, H, W, 3) float32
    Y_masks  : (N, H, W, 1) float32  {0, 1}
    max_samples : int
        Max pixels to subsample for training (avoids memory explosion).

    Returns
    -------
    Fitted RandomForestClassifier
    """
    from sklearn.ensemble import RandomForestClassifier

    print(f"Extracting features from {len(X_images)} images…")
    all_features, all_labels = [], []
    for i, (img, mask) in enumerate(zip(X_images, Y_masks)):
        feats = extract_features(img)      # (H*W, 11)
        labels = mask.flatten().astype(int)
        all_features.append(feats)
        all_labels.append(labels)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(X_images)}")

    X_feat = np.concatenate(all_features, axis=0)
    Y_lab  = np.concatenate(all_labels, axis=0)

    # Subsample to avoid memory issues
    if len(X_feat) > max_samples:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(X_feat), max_samples, replace=False)
        X_feat, Y_lab = X_feat[idx], Y_lab[idx]

    print(f"Training RF on {len(X_feat)} pixels…")
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight=class_weight,
        n_jobs=-1,
        random_state=random_state,
        max_depth=20,
    )
    model.fit(X_feat, Y_lab)
    print("RF training complete.")
    return model


# ── Inference ─────────────────────────────────────────────────────────────────

def predict_random_forest(
    model,
    image_float: np.ndarray,
    threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Predict on a single (H, W, 3) float32 image.

    Returns
    -------
    mask   : (H, W) uint8  {0, 1}
    proba  : (H, W) float32 — probability of class 1 (dead tree)
    """
    H, W, _ = image_float.shape
    feats = extract_features(image_float)                # (H*W, 11)
    proba = model.predict_proba(feats)[:, 1]             # (H*W,)
    proba = proba.reshape(H, W).astype(np.float32)
    mask = (proba >= threshold).astype(np.uint8)
    return mask, proba


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_random_forest(
    model,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Evaluate on test set, return dict of metrics."""
    from sklearn.metrics import (accuracy_score, f1_score,
                                 precision_score, recall_score)

    all_true, all_pred = [], []
    for img, mask in zip(X_test, Y_test):
        pred_mask, _ = predict_random_forest(model, img, threshold)
        all_true.append(mask.flatten().astype(int))
        all_pred.append(pred_mask.flatten().astype(int))

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)

    intersection = np.sum(y_true * y_pred)
    dice = (2 * intersection + 1e-6) / (y_true.sum() + y_pred.sum() + 1e-6)
    union = y_true.sum() + y_pred.sum() - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)

    return {
        "dice":      float(dice),
        "iou":       float(iou),
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
    }


# ── Persistence ───────────────────────────────────────────────────────────────

def save_rf(model, path: str | Path) -> None:
    """Save RF model with joblib."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"RF model saved to {path}")


def load_rf(path: str | Path):
    """Load RF model."""
    return joblib.load(path)
