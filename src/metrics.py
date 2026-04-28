"""Metrics and loss functions for dead tree segmentation."""
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K


# ---------------------------------------------------------------------------
# NumPy / evaluation metrics
# ---------------------------------------------------------------------------

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """Dice coefficient (numpy).

    Accepts float predictions and thresholds them at 0.5.
    y_true, y_pred: arbitrary-shape numpy arrays with values in [0, 1].
    Returns scalar float in [0, 1].
    """
    y_true = y_true.flatten().astype(np.float32)
    y_pred = (y_pred.flatten() >= 0.5).astype(np.float32)
    intersection = np.sum(y_true * y_pred)
    return (2.0 * intersection + smooth) / (np.sum(y_true) + np.sum(y_pred) + smooth)


def iou_score(y_true, y_pred, smooth=1e-6):
    """Intersection-over-Union / Jaccard index (numpy).

    Thresholds float predictions at 0.5.
    Returns scalar float in [0, 1].
    """
    y_true = y_true.flatten().astype(np.float32)
    y_pred = (y_pred.flatten() >= 0.5).astype(np.float32)
    intersection = np.sum(y_true * y_pred)
    union        = np.sum(y_true) + np.sum(y_pred) - intersection
    return (intersection + smooth) / (union + smooth)


def precision_recall_f1(y_true, y_pred, smooth=1e-6):
    """Pixel-level precision, recall, and F1 score (numpy).

    Thresholds float predictions at 0.5.
    Returns dict with keys 'precision', 'recall', 'f1'.
    """
    y_true = y_true.flatten().astype(np.float32)
    y_pred = (y_pred.flatten() >= 0.5).astype(np.float32)

    tp = np.sum(y_true * y_pred)
    fp = np.sum((1 - y_true) * y_pred)
    fn = np.sum(y_true * (1 - y_pred))

    precision = (tp + smooth) / (tp + fp + smooth)
    recall    = (tp + smooth) / (tp + fn + smooth)
    f1        = 2.0 * precision * recall / (precision + recall + smooth)

    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


def compute_all_metrics(y_true, y_pred, threshold=0.5):
    """Compute all evaluation metrics and return as a single dict.

    y_true: ground-truth numpy array (float or int, values in {0, 1}).
    y_pred: predicted probability numpy array (float, values in [0, 1]).
    threshold: binarisation threshold applied to y_pred.
    Returns dict with keys: dice, iou, precision, recall, f1, accuracy.
    """
    y_true_flat = y_true.flatten().astype(np.float32)
    y_pred_bin  = (y_pred.flatten() >= threshold).astype(np.float32)

    # dice
    smooth       = 1e-6
    intersection = np.sum(y_true_flat * y_pred_bin)
    dice         = (2.0 * intersection + smooth) / (
        np.sum(y_true_flat) + np.sum(y_pred_bin) + smooth
    )

    # iou
    union = np.sum(y_true_flat) + np.sum(y_pred_bin) - intersection
    iou   = (intersection + smooth) / (union + smooth)

    # precision / recall / f1
    tp        = intersection
    fp        = np.sum((1 - y_true_flat) * y_pred_bin)
    fn        = np.sum(y_true_flat * (1 - y_pred_bin))
    precision = (tp + smooth) / (tp + fp + smooth)
    recall    = (tp + smooth) / (tp + fn + smooth)
    f1        = 2.0 * precision * recall / (precision + recall + smooth)

    # pixel accuracy
    accuracy = float(np.mean(y_true_flat == y_pred_bin))

    return {
        "dice":      float(dice),
        "iou":       float(iou),
        "precision": float(precision),
        "recall":    float(recall),
        "f1":        float(f1),
        "accuracy":  accuracy,
    }


# ---------------------------------------------------------------------------
# TensorFlow / Keras losses and metrics
# ---------------------------------------------------------------------------

def dice_coefficient_tf(y_true, y_pred, smooth=1.0):
    """Soft Dice coefficient for use as a Keras metric.

    Works on raw sigmoid probabilities (no thresholding).
    y_true, y_pred: tensors of any shape, values in [0, 1].
    Returns scalar tensor.
    """
    y_true_f     = K.flatten(K.cast(y_true, tf.float32))
    y_pred_f     = K.flatten(K.cast(y_pred, tf.float32))
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (
        K.sum(y_true_f) + K.sum(y_pred_f) + smooth
    )


def dice_loss_tf(y_true, y_pred, smooth=1.0):
    """1 - soft Dice coefficient, suitable as a Keras loss."""
    return 1.0 - dice_coefficient_tf(y_true, y_pred, smooth=smooth)


def _binary_focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0, eps=1e-7):
    """Element-wise binary focal loss (internal helper).

    focal = -alpha * y_true * log(y_pred + eps)
            -(1-alpha) * (1-y_true) * log(1 - y_pred + eps)
    then weighted by (1 - p_t)^gamma for hard-example mining.

    Returns mean scalar tensor.
    """
    y_true  = K.cast(y_true, tf.float32)
    y_pred  = K.cast(y_pred, tf.float32)
    y_pred  = K.clip(y_pred, eps, 1.0 - eps)

    focal_ce = (
        -alpha         * y_true       * K.log(y_pred)
        -(1.0 - alpha) * (1.0 - y_true) * K.log(1.0 - y_pred)
    )
    # Modulating factor
    p_t            = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
    modulating     = K.pow(1.0 - p_t, gamma)
    focal_loss_val = modulating * focal_ce
    return K.mean(focal_loss_val)


def combined_loss(y_true, y_pred, dice_weight=0.5, focal_weight=0.5):
    """Weighted combination of Dice loss and binary focal loss.

    Suitable as a Keras loss function.
    combined = dice_weight * dice_loss + focal_weight * focal_loss
    """
    d_loss = dice_loss_tf(y_true, y_pred)
    f_loss = _binary_focal_loss(y_true, y_pred)
    return dice_weight * d_loss + focal_weight * f_loss
