"""Tests for src/dataset.py"""
import sys
import os
import tempfile
import numpy as np
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from dataset import (
    get_mask_filename, get_rgb_filename, build_augmentation,
    augment_image_mask, split_dataset, compute_class_weights,
    predict_with_tta,
)


def test_get_mask_filename():
    assert get_mask_filename("RGB_ar037_2019_n_06_04_0.png") == "mask_ar037_2019_n_06_04_0.png"


def test_get_rgb_filename():
    assert get_rgb_filename("mask_ar037_2019_n_06_04_0.png") == "RGB_ar037_2019_n_06_04_0.png"


def test_filename_roundtrip():
    orig = "RGB_mo025_2021_f_03_07_1.png"
    assert get_rgb_filename(get_mask_filename(orig)) == orig


def test_augmentation_preserves_shape():
    aug = build_augmentation("train")
    image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    mask_float = np.zeros((224, 224, 1), dtype=np.float32)
    aug_img, aug_mask = augment_image_mask(image.astype(np.float32) / 255.0, mask_float, aug)
    assert aug_img.shape == (224, 224, 3)
    assert aug_mask.shape == (224, 224, 1)


def test_augmentation_values_range():
    aug = build_augmentation("train")
    image = np.random.rand(224, 224, 3).astype(np.float32)
    mask = np.random.randint(0, 2, (224, 224, 1)).astype(np.float32)
    aug_img, aug_mask = augment_image_mask(image, mask, aug)
    assert aug_img.min() >= 0.0 and aug_img.max() <= 1.0
    assert set(np.unique(aug_mask)).issubset({0.0, 1.0})


def test_build_augmentation_none_for_val():
    assert build_augmentation("val") is None
    assert build_augmentation("test") is None


def test_split_ratios():
    N = 100
    X = np.zeros((N, 224, 224, 3))
    Y = np.zeros((N, 224, 224, 1))
    result = split_dataset(X, Y)
    total = (len(result["X_train"]) + len(result["X_val"]) + len(result["X_test"]))
    assert total == N
    assert abs(len(result["X_train"]) - 70) <= 2
    assert abs(len(result["X_val"]) - 15) <= 2
    assert abs(len(result["X_test"]) - 15) <= 2


def test_split_no_overlap():
    N = 100
    X = np.arange(N).reshape(N, 1, 1, 1).astype(np.float32)
    Y = np.zeros((N, 1, 1, 1))
    result = split_dataset(X, Y)
    train_idx = set(result["idx_train"])
    val_idx = set(result["idx_val"])
    test_idx = set(result["idx_test"])
    assert len(train_idx & val_idx) == 0
    assert len(train_idx & test_idx) == 0
    assert len(val_idx & test_idx) == 0


def test_class_weights_imbalanced():
    # 5% positive pixels → weight for positives should be >> 1
    Y = np.zeros((100, 224, 224, 1))
    Y[:, :11, :11, :] = 1.0  # ~5% positive
    weights = compute_class_weights(Y)
    assert weights[1] > weights[0]
    assert weights[1] > 1.0


def test_tta_returns_correct_shape():
    class MockModel:
        def predict(self, x, verbose=0):
            return np.zeros((*x.shape[:3], 1), dtype=np.float32)

    model = MockModel()
    image = np.random.rand(224, 224, 3).astype(np.float32)
    mask, prob = predict_with_tta(model, image, threshold=0.5)
    assert mask.shape == (224, 224)
    assert prob.shape == (224, 224)
    assert set(np.unique(mask)).issubset({0, 1})
