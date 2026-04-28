"""Visualization utilities for dead tree segmentation."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2


# ---------------------------------------------------------------------------
# Prediction display
# ---------------------------------------------------------------------------

def display_predictions(X, Y_true, Y_pred_bin, indices=None, n=3, save_path=None):
    """Display side-by-side Original | Ground Truth | Prediction subplots.

    Parameters
    ----------
    X          : float32 ndarray (N, H, W, 3), values in [0, 1]
    Y_true     : float32 ndarray (N, H, W, 1), binary {0, 1}
    Y_pred_bin : float32/uint8 ndarray (N, H, W, 1) or (N, H, W), binary
    indices    : list of sample indices to visualise; if None, first n are used
    n          : number of samples to show (ignored when indices is provided)
    save_path  : optional path to save the figure
    """
    if indices is None:
        indices = list(range(min(n, len(X))))

    n_show = len(indices)
    fig, axes = plt.subplots(n_show, 3, figsize=(12, 4 * n_show))

    # Ensure axes is always 2-D
    if n_show == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Original Image", "Ground Truth", "Prediction"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=13, fontweight="bold")

    for row, idx in enumerate(indices):
        img   = np.clip(X[idx], 0, 1)
        gt    = Y_true[idx].squeeze()
        pred  = Y_pred_bin[idx].squeeze()

        axes[row, 0].imshow(img)
        axes[row, 0].set_ylabel(f"Sample {idx}", fontsize=9)

        axes[row, 1].imshow(gt, cmap="gray", vmin=0, vmax=1)
        axes[row, 2].imshow(pred, cmap="gray", vmin=0, vmax=1)

        for ax in axes[row]:
            ax.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig


# ---------------------------------------------------------------------------
# Mask overlay
# ---------------------------------------------------------------------------

def overlay_mask(image_rgb, mask_bin, color=(255, 0, 0), alpha=0.5):
    """Blend a binary mask over an RGB image with the given colour and opacity.

    Parameters
    ----------
    image_rgb : uint8 ndarray (H, W, 3) or float32 in [0, 1]
    mask_bin  : binary ndarray (H, W) or (H, W, 1) with values 0 / 1 / 255
    color     : RGB tuple for the mask overlay, default red
    alpha     : opacity of the overlay in [0, 1]

    Returns
    -------
    blended   : uint8 ndarray (H, W, 3)
    """
    # Normalise image to uint8
    if image_rgb.dtype != np.uint8:
        img = np.clip(image_rgb * 255, 0, 255).astype(np.uint8)
    else:
        img = image_rgb.copy()

    # Normalise mask to binary uint8 (H, W)
    mask = mask_bin.squeeze()
    if mask.max() > 1:
        mask = (mask > 127).astype(np.uint8)
    else:
        mask = (mask > 0.5).astype(np.uint8)

    # Build coloured overlay
    overlay        = np.zeros_like(img, dtype=np.uint8)
    overlay[..., 0] = color[0]
    overlay[..., 1] = color[1]
    overlay[..., 2] = color[2]

    # Blend only where mask == 1
    mask3   = mask[..., np.newaxis]                            # (H, W, 1)
    blended = np.where(
        mask3,
        np.clip(img * (1 - alpha) + overlay * alpha, 0, 255).astype(np.uint8),
        img,
    )
    return blended


# ---------------------------------------------------------------------------
# Training curves
# ---------------------------------------------------------------------------

def plot_training_curves(history, save_path=None):
    """Plot loss, Dice coefficient, and accuracy curves from a Keras History.

    Parameters
    ----------
    history   : keras.callbacks.History object, or a plain dict of metric lists
    save_path : optional path to save the figure
    """
    hist = history.history if hasattr(history, "history") else history

    # Collect available metrics
    metrics_to_plot = []
    for key in ("loss", "dice_coefficient_tf", "dice_coefficient", "accuracy"):
        if key in hist:
            metrics_to_plot.append(key)

    n_plots = max(len(metrics_to_plot), 1)
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics_to_plot):
        train_vals = hist[metric]
        ax.plot(train_vals, label=f"Train {metric}", linewidth=2)
        val_key = f"val_{metric}"
        if val_key in hist:
            ax.plot(hist[val_key], label=f"Val {metric}", linewidth=2, linestyle="--")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(metric.replace("_", " ").title())
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle("Training Curves", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig


# ---------------------------------------------------------------------------
# Model comparison bar chart
# ---------------------------------------------------------------------------

def plot_metrics_comparison(results_dict, save_path=None):
    """Grouped bar chart comparing multiple models across standard metrics.

    Parameters
    ----------
    results_dict : dict  {model_name: {metric_name: value, ...}, ...}
                   Example: {"UNet": {"dice": 0.82, "iou": 0.75}, ...}
    save_path    : optional path to save the figure

    Returns
    -------
    fig : matplotlib Figure
    """
    model_names = list(results_dict.keys())
    if not model_names:
        raise ValueError("results_dict is empty.")

    # Collect union of all metric names in insertion order
    metric_names = []
    for m_dict in results_dict.values():
        for k in m_dict:
            if k not in metric_names:
                metric_names.append(k)

    n_models  = len(model_names)
    n_metrics = len(metric_names)
    x         = np.arange(n_metrics)
    bar_width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(max(8, 2 * n_metrics), 6))

    cmap   = plt.get_cmap("tab10")
    colors = [cmap(i) for i in range(n_models)]

    for i, (model_name, color) in enumerate(zip(model_names, colors)):
        values = [results_dict[model_name].get(m, 0.0) for m in metric_names]
        offset = (i - n_models / 2 + 0.5) * bar_width
        bars   = ax.bar(x + offset, values, width=bar_width, label=model_name, color=color)

        # Value labels on bars
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", " ").title() for m in metric_names], fontsize=10)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.12)
    ax.set_title("Model Performance Comparison", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig


# ---------------------------------------------------------------------------
# Connected-component tree counting
# ---------------------------------------------------------------------------

def count_dead_trees(mask_bin):
    """Count dead-tree instances in a binary segmentation mask using connected components.

    Parameters
    ----------
    mask_bin : ndarray (H, W) or (H, W, 1), binary values {0, 1} or {0, 255}

    Returns
    -------
    count      : int — number of connected foreground components (dead trees)
    labels     : uint32 ndarray (H, W) — label map (0 = background)
    stats      : ndarray (N+1, 5) — cv2 stats for each component
    centroids  : ndarray (N+1, 2) — (x, y) centroids for each component
    """
    mask = mask_bin.squeeze()
    if mask.dtype != np.uint8:
        if mask.max() <= 1.0:
            mask = (mask * 255).astype(np.uint8)
        else:
            mask = mask.astype(np.uint8)

    # Binarise: anything > 0 is foreground
    _, binary = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8, ltype=cv2.CV_32S
    )

    # Label 0 is background; foreground components start at 1
    count = num_labels - 1
    return count, labels, stats, centroids
