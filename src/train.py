"""
train.py - Training script for forest dead-tree detection.

Usage:
    python train.py --config config.yaml --model unet --output-dir outputs/
    python train.py --config config.yaml --model rf   --output-dir outputs/
"""

import argparse
import json
import os
import pickle
import sys

import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as fh:
        return yaml.safe_load(fh)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# ---------------------------------------------------------------------------
# UNet training
# ---------------------------------------------------------------------------

def train_unet(config: dict) -> None:
    """End-to-end training pipeline for EfficientNet-UNet."""

    output_dir = config.get("output_dir", "outputs/")
    ensure_dir(output_dir)

    # ------------------------------------------------------------------ #
    # 1. Deferred TF import (so RF path doesn't need TF)                 #
    # ------------------------------------------------------------------ #
    import tensorflow as tf  # noqa: F401

    # Project modules (inserted so relative imports work from any cwd)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from dataset import (
        build_augmentation,
        compute_class_weights,
        keras_generator,
        load_dataset,
        split_dataset,
    )
    from models.unet import (
        build_efficientnet_unet,
        compile_model,
        dice_coefficient,
        get_callbacks,
    )
    from metrics import compute_all_metrics

    # ------------------------------------------------------------------ #
    # 2. Dataset                                                          #
    # ------------------------------------------------------------------ #
    data_cfg = config.get("data", {})
    image_dir   = data_cfg.get("image_dir",  "../../USA_segmentation/RGB_images")
    mask_dir    = data_cfg.get("mask_dir",   "../../USA_segmentation/masks")
    img_height  = data_cfg.get("img_height", 224)
    img_width   = data_cfg.get("img_width",  224)

    print("[train_unet] Loading dataset …")
    X, Y, filenames = load_dataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        img_height=img_height,
        img_width=img_width,
        verbose=True,
    )
    print(f"[train_unet] Loaded {len(X)} samples  shape={X.shape}")

    # ------------------------------------------------------------------ #
    # 3. Train / Val / Test split (70/15/15)                              #
    # ------------------------------------------------------------------ #
    splits = split_dataset(X, Y, filenames=filenames, test_size=0.30, val_size=0.50)
    X_train, Y_train = splits["X_train"], splits["Y_train"]
    X_val,   Y_val   = splits["X_val"],   splits["Y_val"]
    X_test,  Y_test  = splits["X_test"],  splits["Y_test"]

    print(
        f"[train_unet] Split → train: {len(X_train)}, "
        f"val: {len(X_val)}, test: {len(X_test)}"
    )

    # ------------------------------------------------------------------ #
    # 4. Class weights                                                    #
    # ------------------------------------------------------------------ #
    class_weights = compute_class_weights(Y_train)
    print(f"[train_unet] Class weights: {class_weights}")

    # ------------------------------------------------------------------ #
    # 5. Augmentation (train only)                                        #
    # ------------------------------------------------------------------ #
    train_aug = build_augmentation(mode="train")

    # ------------------------------------------------------------------ #
    # 6. Keras generators                                                 #
    # ------------------------------------------------------------------ #
    train_cfg  = config.get("training", {})
    batch_size = train_cfg.get("batch_size", 8)

    train_gen = keras_generator(X_train, Y_train, batch_size=batch_size,
                                shuffle=True,  augmentation=train_aug)
    val_gen   = keras_generator(X_val,   Y_val,   batch_size=batch_size,
                                shuffle=False, augmentation=None)
    test_gen  = keras_generator(X_test,  Y_test,  batch_size=batch_size,
                                shuffle=False, augmentation=None)

    steps_per_epoch  = max(1, len(X_train) // batch_size)
    validation_steps = max(1, len(X_val)   // batch_size)

    # ------------------------------------------------------------------ #
    # 7. Model                                                            #
    # ------------------------------------------------------------------ #
    model_cfg    = config.get("model", {})
    dropout_rate = model_cfg.get("dropout_rate", 0.3)

    print("[train_unet] Building EfficientNet-UNet …")
    model = build_efficientnet_unet(
        input_shape=(img_height, img_width, 3),
        dropout_rate=dropout_rate,
        freeze_encoder=False,
    )
    model.summary()

    # ------------------------------------------------------------------ #
    # 8. Compile                                                          #
    # ------------------------------------------------------------------ #
    lr           = train_cfg.get("learning_rate", 1e-3)
    weight_decay = train_cfg.get("weight_decay", 1e-4)
    model = compile_model(model, lr=lr, weight_decay=weight_decay)

    # ------------------------------------------------------------------ #
    # 9. Callbacks                                                        #
    # ------------------------------------------------------------------ #
    best_model_path = os.path.join(output_dir, "best_unet_model.keras")
    patience_lr   = train_cfg.get("patience_lr",   5)
    patience_stop = train_cfg.get("patience_stop", 15)

    callbacks = get_callbacks(
        save_path=best_model_path,
        patience_lr=patience_lr,
        patience_stop=patience_stop,
    )

    # ------------------------------------------------------------------ #
    # 10. Training                                                        #
    # ------------------------------------------------------------------ #
    epochs = train_cfg.get("epochs", 100)
    print(f"[train_unet] Training for up to {epochs} epochs …")

    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_gen,
        validation_steps=validation_steps,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1,
    )

    # ------------------------------------------------------------------ #
    # 11. Save history                                                    #
    # ------------------------------------------------------------------ #
    history_path = os.path.join(output_dir, "history.pkl")
    with open(history_path, "wb") as fh:
        pickle.dump(history.history, fh)
    print(f"[train_unet] History saved → {history_path}")

    # ------------------------------------------------------------------ #
    # 12. Evaluate on test set (best checkpoint)                          #
    # ------------------------------------------------------------------ #
    print("[train_unet] Evaluating on test set …")
    best_model = tf.keras.models.load_model(
        best_model_path,
        custom_objects={"dice_coefficient": dice_coefficient},
    )

    threshold  = model_cfg.get("threshold", 0.5)
    all_preds, all_targets = [], []
    test_steps = max(1, len(X_test) // batch_size)

    for step, (batch_imgs, batch_masks) in enumerate(test_gen):
        if step >= test_steps:
            break
        preds = best_model.predict(batch_imgs, verbose=0)
        all_preds.append(preds)
        all_targets.append(batch_masks)

    all_preds   = np.concatenate(all_preds,   axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    metrics = compute_all_metrics(all_targets, all_preds, threshold=threshold)

    print("\n[train_unet] Test metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # ------------------------------------------------------------------ #
    # 13. Save metrics                                                    #
    # ------------------------------------------------------------------ #
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as fh:
        json.dump({k: float(v) for k, v in metrics.items()}, fh, indent=2)
    print(f"[train_unet] Metrics saved → {metrics_path}")


# ---------------------------------------------------------------------------
# Random Forest baseline
# ---------------------------------------------------------------------------

def train_rf(config: dict) -> None:
    """Training pipeline for Random Forest baseline."""

    output_dir = config.get("output_dir", "outputs/")
    ensure_dir(output_dir)

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from dataset import load_dataset, split_dataset
    from models.random_forest import (
        evaluate_random_forest,
        save_rf,
        train_random_forest,
    )

    # ------------------------------------------------------------------ #
    # Dataset                                                             #
    # ------------------------------------------------------------------ #
    data_cfg   = config.get("data", {})
    image_dir  = data_cfg.get("image_dir",  "../../USA_segmentation/RGB_images")
    mask_dir   = data_cfg.get("mask_dir",   "../../USA_segmentation/masks")
    img_height = data_cfg.get("img_height", 224)
    img_width  = data_cfg.get("img_width",  224)

    print("[train_rf] Loading dataset …")
    X, Y, filenames = load_dataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        img_height=img_height,
        img_width=img_width,
        verbose=True,
    )
    print(f"[train_rf] Loaded {len(X)} samples")

    # ------------------------------------------------------------------ #
    # Split                                                               #
    # ------------------------------------------------------------------ #
    splits = split_dataset(X, Y, filenames=filenames, test_size=0.30, val_size=0.50)
    X_train, Y_train = splits["X_train"], splits["Y_train"]
    X_test,  Y_test  = splits["X_test"],  splits["Y_test"]

    print(f"[train_rf] Split → train: {len(X_train)}, test: {len(X_test)}")

    # ------------------------------------------------------------------ #
    # Train                                                               #
    # ------------------------------------------------------------------ #
    rf_cfg      = config.get("rf", {})
    n_estimators = rf_cfg.get("n_estimators", 200)
    max_samples  = rf_cfg.get("max_samples", 50_000)

    rf_model = train_random_forest(
        X_train, Y_train,
        n_estimators=n_estimators,
        class_weight="balanced",
        max_samples=max_samples,
    )

    # ------------------------------------------------------------------ #
    # Save model                                                          #
    # ------------------------------------------------------------------ #
    rf_path = os.path.join(output_dir, "rf_model.pkl")
    save_rf(rf_model, rf_path)

    # ------------------------------------------------------------------ #
    # Evaluate                                                            #
    # ------------------------------------------------------------------ #
    model_cfg = config.get("model", {})
    threshold = model_cfg.get("threshold", 0.5)

    metrics = evaluate_random_forest(rf_model, X_test, Y_test, threshold=threshold)

    print("\n[train_rf] Test metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    metrics_path = os.path.join(output_dir, "rf_metrics.json")
    with open(metrics_path, "w") as fh:
        json.dump({k: float(v) for k, v in metrics.items()}, fh, indent=2)
    print(f"[train_rf] Metrics saved → {metrics_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train forest dead-tree detection models."
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to YAML config file (default: config.yaml)",
    )
    parser.add_argument(
        "--model",
        default="unet",
        choices=["unet", "rf"],
        help="Model to train: 'unet' (EfficientNet-UNet) or 'rf' (Random Forest). "
             "Default: unet",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/",
        help="Directory for checkpoints, history and metrics (default: outputs/)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config_path = os.path.abspath(args.config)
    if not os.path.isfile(config_path):
        print(f"[ERROR] Config file not found: {config_path}")
        sys.exit(1)

    config = load_config(config_path)
    config["output_dir"] = os.path.abspath(args.output_dir)

    print(f"[main] Config : {config_path}")
    print(f"[main] Model  : {args.model}")
    print(f"[main] Output : {config['output_dir']}")

    if args.model == "unet":
        train_unet(config)
    elif args.model == "rf":
        train_rf(config)
    else:
        print(f"[ERROR] Unknown model: {args.model}")
        sys.exit(1)


if __name__ == "__main__":
    main()
