"""
inference.py - Inference and prediction saving for forest dead-tree detection.

Usage (single image):
    python inference.py --config config.yaml --image path/to/image.png --output-dir preds/

Usage (batch folder):
    python inference.py --config config.yaml --folder path/to/folder/ --output-dir preds/ --tta

Usage flags:
    --tta           Enable test-time augmentation
    --threshold     Binarisation threshold (default 0.5)
"""

import argparse
import os
import sys

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    import yaml

    with open(config_path, "r") as fh:
        config = yaml.safe_load(fh)
    return config


def ensure_dir(path: str) -> None:
    """Create directory (and parents) if it does not exist."""
    os.makedirs(path, exist_ok=True)


def _resolve_src_dir() -> None:
    """Insert the src/ directory into sys.path so project modules are importable."""
    src_dir = os.path.dirname(os.path.abspath(__file__))
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_for_inference(config: dict):
    """
    Load the trained Keras model with all required custom objects.

    Search order:
      1. <output_dir>/best_unet_model.keras
      2. <output_dir>/best_unet_model.h5
      3. ../../USA_segmentation/best_unet_model.h5  (project fallback)

    Returns
    -------
    model : tf.keras.Model
    """
    import tensorflow as tf

    _resolve_src_dir()
    from metrics import dice_coefficient

    # ------------------------------------------------------------------ #
    # Custom loss / metric objects                                         #
    # ------------------------------------------------------------------ #
    def dice_loss(y_true, y_pred, smooth: float = 1.0):
        y_true_f = tf.keras.backend.flatten(tf.cast(y_true, tf.float32))
        y_pred_f = tf.keras.backend.flatten(tf.cast(y_pred, tf.float32))
        intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
        return 1.0 - (2.0 * intersection + smooth) / (
            tf.keras.backend.sum(y_true_f)
            + tf.keras.backend.sum(y_pred_f)
            + smooth
        )

    def combined_loss(y_true, y_pred):
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        dl = dice_loss(y_true, y_pred)
        return bce + dl

    custom_objects = {
        "dice_coefficient": dice_coefficient,
        "combined_loss": combined_loss,
        "dice_loss": dice_loss,
    }

    # ------------------------------------------------------------------ #
    # Candidate paths                                                      #
    # ------------------------------------------------------------------ #
    output_dir = config.get("output_dir", "outputs/")
    candidate_paths = [
        os.path.join(output_dir, "best_unet_model.keras"),
        os.path.join(output_dir, "best_unet_model.h5"),
        # Absolute fallback relative to the dataset root
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "..",
            "USA_segmentation",
            "best_unet_model.h5",
        ),
    ]

    for path in candidate_paths:
        path = os.path.normpath(path)
        if os.path.isfile(path):
            print(f"[load_model] Loading model from: {path}")
            model = tf.keras.models.load_model(path, custom_objects=custom_objects)
            return model

    raise FileNotFoundError(
        "Could not find a trained model. Searched:\n"
        + "\n".join(f"  {os.path.normpath(p)}" for p in candidate_paths)
        + "\nTrain the model first with train.py."
    )


# ---------------------------------------------------------------------------
# Single-image prediction
# ---------------------------------------------------------------------------

def predict_single(
    model,
    image_path: str,
    output_dir: str,
    threshold: float = 0.5,
    use_tta: bool = False,
    config: dict = None,
):
    """
    Run inference on a single image and save the binary mask PNG.

    The output filename mirrors the input filename with a 'unet_mask_' prefix
    replacing the original stem prefix, e.g.:
        RGB_ar037_2019_n_06_04_0.png  →  unet_mask_ar037_2019_n_06_04_0.png

    Parameters
    ----------
    model       : loaded tf.keras.Model
    image_path  : absolute path to the RGB image
    output_dir  : directory where prediction mask is saved
    threshold   : binarisation threshold (default 0.5)
    use_tta     : whether to use test-time augmentation
    config      : optional config dict

    Returns
    -------
    (binary_mask, probability_map) : tuple of np.ndarray, both shape (H, W)
    """
    import cv2
    from PIL import Image

    if config is None:
        config = {}

    img_size = config.get("img_size", 224)

    # ------------------------------------------------------------------ #
    # 1. Load + pre-process                                               #
    # ------------------------------------------------------------------ #
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    img_norm = img_resized.astype(np.float32) / 255.0

    # ------------------------------------------------------------------ #
    # 2. Predict                                                          #
    # ------------------------------------------------------------------ #
    if use_tta:
        _resolve_src_dir()
        from dataset import predict_with_tta

        prob_map = predict_with_tta(model, img_norm, config=config)
    else:
        batch = np.expand_dims(img_norm, axis=0)          # (1, H, W, 3)
        prob_map = model.predict(batch, verbose=0)[0]     # (H, W, 1) or (H, W)

    prob_map = np.squeeze(prob_map)                       # (H, W)
    binary_mask = (prob_map >= threshold).astype(np.uint8) * 255

    # ------------------------------------------------------------------ #
    # 3. Build output filename                                            #
    # ------------------------------------------------------------------ #
    base_name = os.path.basename(image_path)               # e.g. RGB_ar037_…png
    # Strip the leading "RGB_" prefix if present (dataset convention), keep the rest
    if base_name.startswith("RGB_"):
        stem = base_name[len("RGB_"):]
    else:
        stem = base_name
    output_filename = f"unet_mask_{stem}"
    mask_save_path = os.path.join(output_dir, output_filename)

    # ------------------------------------------------------------------ #
    # 4. Save binary mask                                                 #
    # ------------------------------------------------------------------ #
    ensure_dir(output_dir)
    Image.fromarray(binary_mask).save(mask_save_path)
    print(f"[predict_single] Saved mask → {mask_save_path}")

    return binary_mask, prob_map


# ---------------------------------------------------------------------------
# Batch-folder prediction
# ---------------------------------------------------------------------------

def predict_folder(
    model,
    folder_path: str,
    output_dir: str,
    threshold: float = 0.5,
    use_tta: bool = False,
    config: dict = None,
) -> None:
    """
    Batch-process all PNG images in folder_path.

    Parameters
    ----------
    model       : loaded tf.keras.Model
    folder_path : directory containing PNG images
    output_dir  : directory where prediction masks are saved
    threshold   : binarisation threshold (default 0.5)
    use_tta     : whether to use test-time augmentation
    config      : optional config dict
    """
    if not os.path.isdir(folder_path):
        raise NotADirectoryError(f"Not a directory: {folder_path}")

    png_files = sorted(
        f for f in os.listdir(folder_path) if f.lower().endswith(".png")
    )

    if not png_files:
        print(f"[predict_folder] No PNG files found in: {folder_path}")
        return

    print(f"[predict_folder] Processing {len(png_files)} images …")
    ensure_dir(output_dir)

    for idx, fname in enumerate(png_files, start=1):
        img_path = os.path.join(folder_path, fname)
        print(f"  [{idx}/{len(png_files)}] {fname}")
        try:
            mask, prob_map = predict_single(
                model,
                img_path,
                output_dir=output_dir,
                threshold=threshold,
                use_tta=use_tta,
                config=config,
            )

            # Optional: save visualisation alongside the mask
            vis_dir = os.path.join(output_dir, "visualisations")
            ensure_dir(vis_dir)
            vis_name = os.path.splitext(fname)[0] + "_vis.png"
            vis_path = os.path.join(vis_dir, vis_name)

            import cv2
            img_bgr = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            save_prediction_visualization(img_rgb, mask, prob_map, vis_path)

        except Exception as exc:
            print(f"  [WARN] Failed on {fname}: {exc}")

    print(f"[predict_folder] Done. Masks saved to: {output_dir}")


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def save_prediction_visualization(
    image: np.ndarray,
    mask: np.ndarray,
    prob_map: np.ndarray,
    save_path: str,
) -> None:
    """
    Save a 3-panel figure: original image | probability map | binary mask.

    Parameters
    ----------
    image    : RGB image array, shape (H, W, 3), uint8 or float [0,1]
    mask     : binary mask array, shape (H, W), values 0 or 255
    prob_map : probability map array, shape (H, W), float [0, 1]
    save_path: absolute path for the PNG output
    """
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend — safe in server/script context
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: Original image
    axes[0].imshow(image if image.max() <= 1.0 else image.astype(np.uint8))
    axes[0].set_title("Original Image", fontsize=13)
    axes[0].axis("off")

    # Panel 2: Probability map
    im = axes[1].imshow(prob_map, cmap="RdYlGn", vmin=0.0, vmax=1.0)
    axes[1].set_title("Probability Map", fontsize=13)
    axes[1].axis("off")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # Panel 3: Binary mask
    axes[2].imshow(mask, cmap="gray", vmin=0, vmax=255)
    axes[2].set_title("Predicted Mask (binary)", fontsize=13)
    axes[2].axis("off")

    fig.suptitle("Dead-Tree Detection — Prediction", fontsize=14, fontweight="bold")
    plt.tight_layout()

    ensure_dir(os.path.dirname(save_path) or ".")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualisation] Saved → {save_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference for forest dead-tree detection."
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to YAML config file (default: config.yaml)",
    )
    parser.add_argument(
        "--image",
        default=None,
        help="Path to a single PNG image for inference.",
    )
    parser.add_argument(
        "--folder",
        default=None,
        help="Path to a folder of PNG images for batch inference.",
    )
    parser.add_argument(
        "--output-dir",
        default="predictions/",
        help="Directory where prediction masks are saved (default: predictions/)",
    )
    parser.add_argument(
        "--tta",
        action="store_true",
        default=False,
        help="Enable test-time augmentation (TTA) for more robust predictions.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Binarisation threshold for the probability map (default: 0.5).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------ #
    # Validate arguments                                                  #
    # ------------------------------------------------------------------ #
    if args.image is None and args.folder is None:
        print("[ERROR] Provide at least one of --image or --folder.")
        sys.exit(1)

    if args.image is not None and args.folder is not None:
        print("[ERROR] Provide either --image or --folder, not both.")
        sys.exit(1)

    # ------------------------------------------------------------------ #
    # Config                                                              #
    # ------------------------------------------------------------------ #
    config_path = os.path.abspath(args.config)
    if not os.path.isfile(config_path):
        print(f"[ERROR] Config file not found: {config_path}")
        sys.exit(1)

    config = load_config(config_path)
    config["output_dir"] = os.path.abspath(args.output_dir)
    output_dir = config["output_dir"]

    print(f"[main] Config     : {config_path}")
    print(f"[main] Output dir : {output_dir}")
    print(f"[main] TTA        : {args.tta}")
    print(f"[main] Threshold  : {args.threshold}")

    # ------------------------------------------------------------------ #
    # Load model                                                          #
    # ------------------------------------------------------------------ #
    model = load_model_for_inference(config)

    # ------------------------------------------------------------------ #
    # Inference                                                           #
    # ------------------------------------------------------------------ #
    if args.image is not None:
        # Single image
        image_path = os.path.abspath(args.image)
        if not os.path.isfile(image_path):
            print(f"[ERROR] Image not found: {image_path}")
            sys.exit(1)

        mask, prob_map = predict_single(
            model,
            image_path,
            output_dir=output_dir,
            threshold=args.threshold,
            use_tta=args.tta,
            config=config,
        )

        # Save visualisation
        import cv2
        img_bgr = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        vis_name = os.path.splitext(os.path.basename(image_path))[0] + "_vis.png"
        vis_path = os.path.join(output_dir, "visualisations", vis_name)
        save_prediction_visualization(img_rgb, mask, prob_map, vis_path)

    else:
        # Batch folder
        folder_path = os.path.abspath(args.folder)
        predict_folder(
            model,
            folder_path,
            output_dir=output_dir,
            threshold=args.threshold,
            use_tta=args.tta,
            config=config,
        )

    print("[main] Inference complete.")


if __name__ == "__main__":
    main()
