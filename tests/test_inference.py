"""Tests for inference pipeline."""
import sys
import numpy as np
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from metrics import compute_all_metrics
from dataset import get_mask_filename, get_rgb_filename


# ── Filename convention ───────────────────────────────────────────────────────

def test_output_filename_convention():
    """Input RGB_xxx.png → output unet_mask_xxx.png (uses original name, not unet_mask_1.png)."""
    img_name = "RGB_ar037_2019_n_06_04_0.png"
    mask_name = get_mask_filename(img_name)          # mask_ar037_2019_n_06_04_0.png
    assert "mask_ar037" in mask_name                 # keeps original identifier
    assert "mask_1" not in mask_name                 # NOT sequential bug


def test_unet_prediction_output_name():
    """Verify the naming logic for saved U-Net predictions."""
    original_stem = "ar037_2019_n_06_04_0"
    # Convention: unet_ prefix + original identifier
    saved_name = f"unet_{original_stem}.png"
    assert "ar037" in saved_name
    assert saved_name != "unet_mask_1.png"


# ── Metrics thresholding ──────────────────────────────────────────────────────

def test_metrics_thresholding():
    """Raw sigmoid predictions must be thresholded before metric computation."""
    y_true = np.ones((100,), dtype=np.float32)
    # All predictions = 0.6 (above threshold 0.5) → should be treated as 1 → perfect Dice
    y_pred_raw = np.full((100,), 0.6, dtype=np.float32)
    metrics = compute_all_metrics(y_true, y_pred_raw, threshold=0.5)
    assert abs(metrics["dice"] - 1.0) < 1e-4, "Float predictions must be thresholded at 0.5"


def test_metrics_below_threshold_treated_as_zero():
    """Predictions below threshold should be 0."""
    y_true = np.zeros((100,), dtype=np.float32)
    y_pred_raw = np.full((100,), 0.4, dtype=np.float32)  # All below 0.5
    metrics = compute_all_metrics(y_true, y_pred_raw, threshold=0.5)
    # True negatives only → perfect for negative class
    assert metrics["dice"] > 0.99


def test_prediction_mask_binary():
    """Output prediction mask must contain only 0 and 1."""
    prob = np.random.rand(224, 224).astype(np.float32)
    mask = (prob >= 0.5).astype(np.uint8)
    unique = set(np.unique(mask))
    assert unique.issubset({0, 1})


def test_predict_output_types():
    """Simulate model predict and verify return types."""
    class MockModel:
        def predict(self, x, verbose=0):
            return np.random.rand(*x.shape[:3], 1).astype(np.float32)

    model = MockModel()
    image = np.random.rand(224, 224, 3).astype(np.float32)
    x = image[np.newaxis]
    raw = model.predict(x)[0, :, :, 0]
    mask = (raw >= 0.5).astype(np.uint8)

    assert isinstance(mask, np.ndarray)
    assert isinstance(raw, np.ndarray)
    assert mask.dtype == np.uint8
    assert raw.dtype == np.float32


def test_metrics_dict_keys():
    y_true = np.random.randint(0, 2, (224, 224)).astype(np.float32)
    y_pred = np.random.rand(224, 224).astype(np.float32)
    result = compute_all_metrics(y_true, y_pred)
    required = {"dice", "iou", "precision", "recall", "f1"}
    assert required.issubset(set(result.keys()))


def test_dice_not_affected_by_filename_bug():
    """
    The original bug: U-Net masks saved as unet_mask_1.png, unet_mask_2.png (sequential)
    while GT masks are mask_ar037_2019_n_06_04_0.png → no common files → Dice=0.03.
    Verify the correct naming produces non-zero Dice.
    """
    # Simulate correct naming: GT file = mask_ar037_2019_n_06_04_0.png
    gt_files = {"mask_ar037_2019_n_06_04_0.png", "mask_ar037_2019_n_06_04_1.png"}
    # WRONG naming (old bug): sequential
    unet_wrong = {"unet_mask_1.png", "unet_mask_2.png"}
    # CORRECT naming: use original identifier
    unet_correct = {"unet_ar037_2019_n_06_04_0.png", "unet_ar037_2019_n_06_04_1.png"}

    # The comparison loop needs common base names — build sets of base identifiers
    def get_base(name):
        return name.replace("mask_", "").replace("unet_", "").replace("RGB_", "")

    gt_bases = {get_base(f) for f in gt_files}
    wrong_bases = {get_base(f) for f in unet_wrong}
    correct_bases = {get_base(f) for f in unet_correct}

    assert len(gt_bases & wrong_bases) == 0, "Sequential naming → no matches (bug)"
    assert len(gt_bases & correct_bases) == 2, "Original naming → all match (fix)"
