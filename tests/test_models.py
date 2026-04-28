"""Tests for src/metrics.py and src/models/"""
import sys
import numpy as np
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from metrics import dice_coefficient, iou_score, precision_recall_f1, compute_all_metrics


# ── Metric tests (NumPy) ──────────────────────────────────────────────────────

def test_dice_perfect():
    y = np.ones((224, 224, 1), dtype=np.float32)
    assert abs(dice_coefficient(y, y) - 1.0) < 1e-4


def test_dice_zero_overlap():
    y_true = np.zeros((100, 100), dtype=np.float32)
    y_true[:50, :] = 1.0
    y_pred = np.zeros((100, 100), dtype=np.float32)
    y_pred[50:, :] = 1.0
    assert dice_coefficient(y_true, y_pred) < 0.05


def test_dice_thresholds_floats():
    # Raw sigmoid outputs (0.6) should be thresholded to 1
    y_true = np.ones((10, 10), dtype=np.float32)
    y_pred = np.full((10, 10), 0.6, dtype=np.float32)
    d = dice_coefficient(y_true, y_pred)
    assert abs(d - 1.0) < 1e-4


def test_iou_range():
    y_true = np.random.randint(0, 2, (50, 50)).astype(np.float32)
    y_pred = np.random.rand(50, 50).astype(np.float32)
    score = iou_score(y_true, y_pred)
    assert 0.0 <= score <= 1.0


def test_precision_recall_f1_keys():
    y_true = np.random.randint(0, 2, (100,)).astype(np.float32)
    y_pred = np.random.rand(100).astype(np.float32)
    result = precision_recall_f1(y_true, y_pred)
    assert "precision" in result and "recall" in result and "f1" in result


def test_compute_all_metrics_keys():
    y_true = np.random.randint(0, 2, (224, 224)).astype(np.float32)
    y_pred = np.random.rand(224, 224).astype(np.float32)
    result = compute_all_metrics(y_true, y_pred)
    for key in ("dice", "iou", "precision", "recall", "f1"):
        assert key in result, f"Missing key: {key}"


def test_metrics_all_correct():
    y_true = np.ones((100,), dtype=np.float32)
    y_pred = np.ones((100,), dtype=np.float32)
    result = compute_all_metrics(y_true, y_pred)
    assert abs(result["dice"] - 1.0) < 1e-3
    assert abs(result["precision"] - 1.0) < 1e-3
    assert abs(result["recall"] - 1.0) < 1e-3


# ── Random Forest tests ───────────────────────────────────────────────────────

def test_rf_features_shape():
    from models.random_forest import extract_features
    image = np.random.rand(224, 224, 3).astype(np.float32)
    feats = extract_features(image)
    assert feats.ndim == 2
    assert feats.shape[0] == 224 * 224
    assert feats.shape[1] >= 5  # at minimum RGB + NDVI + LBP


@pytest.mark.slow
def test_rf_train_predict():
    from models.random_forest import train_random_forest, predict_random_forest
    N = 10
    X = np.random.rand(N, 32, 32, 3).astype(np.float32)
    Y = np.random.randint(0, 2, (N, 32, 32, 1)).astype(np.float32)
    model = train_random_forest(X, Y, n_estimators=10, max_samples=5000)
    mask, proba = predict_random_forest(model, X[0])
    assert mask.shape == (32, 32)
    assert set(np.unique(mask)).issubset({0, 1})
    assert proba.shape == (32, 32)
    assert proba.min() >= 0.0 and proba.max() <= 1.0


# ── U-Net / Loss tests ────────────────────────────────────────────────────────

@pytest.mark.slow
def test_unet_output_shape():
    """Build EfficientNet UNet and check output shape (slow)."""
    try:
        import tensorflow as tf
        from models.unet import build_efficientnet_unet, compile_model
        model = build_efficientnet_unet(input_shape=(224, 224, 3))
        compile_model(model)
        dummy = np.random.rand(1, 224, 224, 3).astype(np.float32)
        out = model.predict(dummy, verbose=0)
        assert out.shape == (1, 224, 224, 1)
        assert out.min() >= 0.0 and out.max() <= 1.0
    except ImportError:
        pytest.skip("TensorFlow not available")


def test_combined_loss_tf_range():
    try:
        import tensorflow as tf
        from models.unet import combined_loss
        loss_fn = combined_loss(dice_w=0.5, focal_w=0.5)
        y_true = tf.constant(np.random.randint(0, 2, (2, 224, 224, 1)), dtype=tf.float32)
        y_pred = tf.constant(np.random.rand(2, 224, 224, 1), dtype=tf.float32)
        val = loss_fn(y_true, y_pred).numpy()
        assert 0.0 <= val <= 2.0
    except ImportError:
        pytest.skip("TensorFlow not available")
