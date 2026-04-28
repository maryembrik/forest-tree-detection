"""Dataset loading and augmentation for dead tree detection."""
import os
import numpy as np
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import albumentations as A

IMG_HEIGHT = 224
IMG_WIDTH = 224


def get_mask_filename(img_filename):
    """Robustly convert RGB image filename to mask filename.
    RGB_ar037_2019_n_06_04_0.png -> mask_ar037_2019_n_06_04_0.png
    """
    return img_filename.replace("RGB_", "mask_")


def get_rgb_filename(mask_filename):
    """Robustly convert mask filename to RGB image filename."""
    return mask_filename.replace("mask_", "RGB_")


def load_dataset(image_dir, mask_dir, img_height=IMG_HEIGHT, img_width=IMG_WIDTH, verbose=True):
    """Load all image/mask pairs. Returns (X, Y, filenames).

    X: float32 (N, H, W, 3) in [0, 1]
    Y: float32 (N, H, W, 1) in {0, 1}
    filenames: list of stem names (without extension)
    """
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)

    image_files = sorted(image_dir.glob("*.png"))
    images, masks, filenames = [], [], []

    iterator = tqdm(image_files, desc="Loading dataset") if verbose else image_files
    for img_path in iterator:
        mask_name = get_mask_filename(img_path.name)
        mask_path = mask_dir / mask_name
        if not mask_path.exists():
            continue

        img = Image.open(img_path).convert("RGB").resize((img_width, img_height))
        img_arr = np.array(img, dtype=np.float32) / 255.0

        mask = Image.open(mask_path).convert("L").resize((img_width, img_height))
        mask_arr = np.array(mask, dtype=np.float32) / 255.0
        mask_arr = np.round(mask_arr)[..., np.newaxis]  # (H, W, 1)

        images.append(img_arr)
        masks.append(mask_arr)
        filenames.append(img_path.stem)

    X = np.array(images, dtype=np.float32)
    Y = np.array(masks, dtype=np.float32)
    return X, Y, filenames


def split_dataset(X, Y, filenames=None, test_size=0.30, val_size=0.50, random_state=42):
    """70/15/15 train/val/test split."""
    idx = np.arange(len(X))
    idx_train, idx_temp = train_test_split(idx, test_size=test_size, random_state=random_state)
    idx_val, idx_test = train_test_split(idx_temp, test_size=val_size, random_state=random_state)

    result = {
        "X_train": X[idx_train], "Y_train": Y[idx_train],
        "X_val": X[idx_val],   "Y_val": Y[idx_val],
        "X_test": X[idx_test], "Y_test": Y[idx_test],
        "idx_train": idx_train, "idx_val": idx_val, "idx_test": idx_test,
    }
    if filenames is not None:
        filenames = np.array(filenames)
        result["fn_train"] = filenames[idx_train]
        result["fn_val"]   = filenames[idx_val]
        result["fn_test"]  = filenames[idx_test]
    return result


def build_augmentation(mode="train"):
    """Albumentations pipeline. Uses A.Affine instead of deprecated ShiftScaleRotate."""
    if mode == "train":
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.Rotate(limit=30, p=0.5),
            A.Affine(translate_percent=0.05, scale=(0.95, 1.05), rotate=0, p=0.5),
            A.RandomCrop(height=224, width=224, p=1.0),
            A.ElasticTransform(alpha=1, sigma=50, p=0.3),
            A.GridDistortion(p=0.3),
            A.GaussNoise(p=0.2),
            A.CLAHE(p=0.3),
            A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.2),
        ])
    return None  # val/test: no augmentation


def augment_image_mask(image_float, mask_float, augmentation):
    """Apply augmentation to (H,W,3) float image and (H,W,1) float mask.

    Returns augmented (image_float, mask_float).
    """
    img_uint8 = (image_float * 255).astype(np.uint8)
    mask_sq   = (mask_float.squeeze() * 255).astype(np.uint8)
    aug = augmentation(image=img_uint8, mask=mask_sq)
    img_out  = aug["image"].astype(np.float32) / 255.0
    mask_out = aug["mask"].astype(np.float32) / 255.0
    mask_out = np.round(mask_out)[..., np.newaxis]
    return img_out, mask_out


def keras_generator(X, Y, batch_size=16, shuffle=True, augmentation=None):
    """Keras-compatible infinite generator yielding (X_batch, Y_batch)."""
    n   = len(X)
    idx = np.arange(n)
    while True:
        if shuffle:
            np.random.shuffle(idx)
        for start in range(0, n, batch_size):
            b_idx = idx[start:start + batch_size]
            X_b   = X[b_idx].copy()
            Y_b   = Y[b_idx].copy()
            if augmentation is not None:
                for i in range(len(b_idx)):
                    X_b[i], Y_b[i] = augment_image_mask(X_b[i], Y_b[i], augmentation)
            yield X_b, Y_b


def compute_class_weights(Y_train):
    """Compute class weights to handle imbalanced masks."""
    n_pixels = Y_train.size
    n_pos    = float(Y_train.sum())
    n_neg    = n_pixels - n_pos
    w_pos    = n_pixels / (2.0 * n_pos + 1e-8)
    w_neg    = n_pixels / (2.0 * n_neg + 1e-8)
    return {0: w_neg, 1: w_pos}


def predict_with_tta(model, image, threshold=0.5):
    """Test-Time Augmentation: average predictions over 4 flips.

    image: (H, W, 3) float32 numpy array
    Returns (mask, prob_map) both (H, W)
    """
    variants = [
        image,
        np.flip(image, axis=0),
        np.flip(image, axis=1),
        np.flip(np.flip(image, axis=0), axis=1),
    ]
    probs = []
    for v in variants:
        x = v[np.newaxis]
        p = model.predict(x, verbose=0)[0, :, :, 0]
        probs.append(p)

    # Undo flips on probability maps before averaging
    probs[1] = np.flip(probs[1], axis=0)
    probs[2] = np.flip(probs[2], axis=1)
    probs[3] = np.flip(np.flip(probs[3], axis=0), axis=1)

    avg_prob = np.mean(probs, axis=0)
    mask     = (avg_prob >= threshold).astype(np.uint8)
    return mask, avg_prob
