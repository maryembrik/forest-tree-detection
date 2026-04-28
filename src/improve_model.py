"""
improve_model.py
================
Fine-tunes the existing U-Net with three targeted fixes for the 1:53
class imbalance found in the dataset:

  1. Tversky loss  (beta=0.7 -> penalises missing dead trees 2.3x more)
  2. Positive-patch oversampling  (guarantees ≥50 % of each batch has
     dead-tree pixels)
  3. Optimal threshold search post-training (best Dice, not fixed 0.5)

Usage:
    python src/improve_model.py
"""

import json, os, pickle, sys, warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

# ── project path ──────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
OUT  = os.path.join(ROOT, "outputs")
os.makedirs(OUT, exist_ok=True)

# ── paths ──────────────────────────────────────────────────────────────────────
DATA_ROOT  = os.path.join(ROOT, "..", "USA_segmentation")
IMG_DIR    = os.path.join(DATA_ROOT, "RGB_images")
MASK_DIR   = os.path.join(DATA_ROOT, "masks")
BASE_MODEL = os.path.join(DATA_ROOT, "best_unet_model.h5")
SAVE_PATH  = os.path.join(OUT, "best_improved_model.keras")

# ── losses ────────────────────────────────────────────────────────────────────

def dice_coef(y_true, y_pred, smooth=1.0):
    y_true_f = K.flatten(tf.cast(y_true, tf.float32))
    y_pred_f = K.flatten(y_pred)
    inter = K.sum(y_true_f * y_pred_f)
    return (2. * inter + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def tversky_loss(y_true, y_pred, alpha=0.3, beta=0.7, smooth=1e-6):
    """
    Tversky index generalises Dice:
      alpha = FP weight, beta = FN weight
    With beta=0.7 the model is penalised 2.3x harder for missing dead pixels
    than for false alarms — crucial for 1:53 imbalance.
    """
    y_true_f = K.flatten(tf.cast(y_true, tf.float32))
    y_pred_f = K.flatten(y_pred)
    tp = K.sum(y_true_f * y_pred_f)
    fp = K.sum((1 - y_true_f) * y_pred_f)
    fn = K.sum(y_true_f * (1 - y_pred_f))
    return 1.0 - (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)


def focal_tversky_loss(y_true, y_pred, gamma=1.5):
    """Focal Tversky: raises Tversky to power gamma -> focuses on hard examples."""
    tv = tversky_loss(y_true, y_pred)
    return K.pow(tv, 1.0 / gamma)


def binary_focal_loss(y_true, y_pred, alpha=0.85, gamma=3.0):
    """
    alpha=0.85 puts 85 % of the loss weight on the positive (dead-tree) class.
    gamma=3.0 down-weights easy background pixels hard.
    """
    eps = 1e-7
    y_pred = K.clip(y_pred, eps, 1 - eps)
    y_true = tf.cast(y_true, tf.float32)
    p_t    = y_true * y_pred + (1 - y_true) * (1 - y_pred)
    alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
    return K.mean(-alpha_t * K.pow(1 - p_t, gamma) * K.log(p_t))


def combined_loss(y_true, y_pred):
    """0.5 x Focal-Tversky  +  0.5 x Focal-BCE"""
    return 0.5 * focal_tversky_loss(y_true, y_pred) + \
           0.5 * binary_focal_loss(y_true, y_pred)


# ── oversampling generator ────────────────────────────────────────────────────

def positive_oversample_generator(X, Y, batch_size=8, pos_ratio=0.5,
                                   augmentation=None):
    """
    Infinite generator that ensures `pos_ratio` of every batch comes from
    images that contain at least one dead-tree pixel.  This prevents the
    model from ignoring the rare class.
    """
    pos_idx = np.where(Y.reshape(len(Y), -1).sum(axis=1) > 0)[0]
    neg_idx = np.where(Y.reshape(len(Y), -1).sum(axis=1) == 0)[0]
    n_pos   = max(1, int(batch_size * pos_ratio))
    n_neg   = batch_size - n_pos

    print(f"  Oversampling: {len(pos_idx)} positive / "
          f"{len(neg_idx)} negative images  "
          f"-> {n_pos} pos + {n_neg} neg per batch")

    while True:
        p_idx = np.random.choice(pos_idx, n_pos, replace=len(pos_idx) < n_pos)
        n_idx = np.random.choice(neg_idx, n_neg, replace=len(neg_idx) < n_neg)
        idx   = np.concatenate([p_idx, n_idx])
        np.random.shuffle(idx)

        X_b = X[idx].copy()
        Y_b = Y[idx].copy()

        if augmentation is not None:
            from dataset import augment_image_mask
            for i in range(len(idx)):
                X_b[i], Y_b[i] = augment_image_mask(X_b[i], Y_b[i], augmentation)

        yield X_b, Y_b


# ── threshold search ──────────────────────────────────────────────────────────

def find_best_threshold(model, X_val, Y_val):
    preds = model.predict(X_val, batch_size=8, verbose=0)
    y_true = Y_val.flatten()
    y_pred = preds.flatten()

    best_t, best_dice = 0.5, 0.0
    for t in np.arange(0.15, 0.75, 0.02):
        pred_bin   = (y_pred >= t).astype(float)
        inter      = np.sum(y_true * pred_bin)
        dice       = (2 * inter + 1e-6) / (y_true.sum() + pred_bin.sum() + 1e-6)
        if dice > best_dice:
            best_dice, best_t = dice, t

    print(f"  Best threshold: {best_t:.2f}  ->  val Dice = {best_dice:.4f}")
    return best_t


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    from dataset import (load_dataset, split_dataset,
                         build_augmentation, keras_generator)
    from metrics import compute_all_metrics

    # ── data ──────────────────────────────────────────────────────────────────
    print("\n[1/6] Loading dataset …")
    X, Y, fns = load_dataset(IMG_DIR, MASK_DIR, verbose=True)
    splits = split_dataset(X, Y, filenames=fns, test_size=0.30, val_size=0.50)
    X_train, Y_train = splits["X_train"], splits["Y_train"]
    X_val,   Y_val   = splits["X_val"],   splits["Y_val"]
    X_test,  Y_test  = splits["X_test"],  splits["Y_test"]

    pos = float(Y_train.mean())
    print(f"  Train {len(X_train)} | Val {len(X_val)} | Test {len(X_test)}")
    print(f"  Positive ratio: {pos*100:.2f}%  (1:{int((1-pos)/pos)} imbalance)")

    # ── model ─────────────────────────────────────────────────────────────────
    print("\n[2/6] Loading base model …")
    model = tf.keras.models.load_model(
        BASE_MODEL,
        custom_objects={"dice_coefficient": dice_coef, "dice_coef": dice_coef},
        compile=False,
    )
    print(f"  Parameters: {model.count_params():,}")

    # ── baseline metrics ──────────────────────────────────────────────────────
    print("\n[3/6] Baseline evaluation (threshold=0.50) …")
    base_preds = model.predict(X_test, batch_size=8, verbose=0)
    base_m = compute_all_metrics(Y_test, base_preds, threshold=0.5)
    print("  Baseline -> " +
          " | ".join(f"{k}={v:.4f}" for k, v in base_m.items()))

    # ── compile with improved loss ────────────────────────────────────────────
    print("\n[4/6] Recompiling with Focal-Tversky + Focal-BCE loss …")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
        loss=combined_loss,
        metrics=[dice_coef],
    )

    # ── generators ────────────────────────────────────────────────────────────
    aug        = build_augmentation(mode="train")
    batch_size = 8
    train_gen  = positive_oversample_generator(
        X_train, Y_train, batch_size=batch_size,
        pos_ratio=0.6, augmentation=aug)
    val_gen = keras_generator(X_val, Y_val, batch_size=batch_size,
                              shuffle=False, augmentation=None)

    steps     = max(1, len(X_train) // batch_size)
    val_steps = max(1, len(X_val)   // batch_size)

    # ── callbacks ─────────────────────────────────────────────────────────────
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            SAVE_PATH, monitor="val_dice_coef",
            save_best_only=True, mode="max", verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=4, min_lr=1e-8, verbose=1),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_dice_coef", patience=10,
            mode="max", restore_best_weights=True, verbose=1),
    ]

    # ── fine-tune ─────────────────────────────────────────────────────────────
    print("\n[5/6] Fine-tuning …")
    history = model.fit(
        train_gen, steps_per_epoch=steps,
        validation_data=val_gen, validation_steps=val_steps,
        epochs=40, callbacks=callbacks, verbose=1,
    )

    # save history
    with open(os.path.join(OUT, "finetune_history.pkl"), "wb") as f:
        pickle.dump(history.history, f)

    # ── final evaluation ──────────────────────────────────────────────────────
    print("\n[6/6] Final evaluation …")
    best_threshold = find_best_threshold(model, X_val, Y_val)

    test_preds = model.predict(X_test, batch_size=8, verbose=0)
    final_m    = compute_all_metrics(Y_test, test_preds, threshold=best_threshold)

    print("\n" + "="*60)
    print(f"{'Metric':<14} {'Baseline (thr=0.50)':>20} {'Improved':>12} {'Delta':>8}")
    print("="*60)
    for k in final_m:
        b = base_m[k]
        f = final_m[k]
        print(f"{k:<14} {b:>20.4f} {f:>12.4f} {f-b:>+8.4f}")
    print("="*60)
    print(f"Optimal threshold: {best_threshold:.2f}")

    # save metrics
    result = {
        "baseline":  {k: float(v) for k, v in base_m.items()},
        "improved":  {k: float(v) for k, v in final_m.items()},
        "threshold": float(best_threshold),
        "history": {k: [float(x) for x in v]
                    for k, v in history.history.items()},
    }
    with open(os.path.join(OUT, "improvement_results.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved -> {SAVE_PATH}")
    print(f"Saved -> outputs/improvement_results.json")


if __name__ == "__main__":
    main()
