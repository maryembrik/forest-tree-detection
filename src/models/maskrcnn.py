"""
PyTorch Mask R-CNN wrapper for fine-tuning on binary dead-tree detection.

Provides:
    build_maskrcnn          – model factory (ResNet-50 + FPN backbone)
    get_maskrcnn_optimizer  – differential LR for backbone vs. heads
    train_maskrcnn          – full training loop with cosine LR scheduler
    predict_maskrcnn        – inference → combined binary mask
    TreeMaskRCNNDataset     – Dataset converting numpy images/masks to
                              the COCO-style format Mask R-CNN expects
"""

import os
import math
import numpy as np

import torch
import torch.utils.data as data
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torchvision.ops as ops

from scipy import ndimage as ndi


# ── Dataset ───────────────────────────────────────────────────────────────────

class TreeMaskRCNNDataset(data.Dataset):
    """
    Dataset adapter for Mask R-CNN.

    Accepts lists/arrays of float32 RGB images (H×W×3, values in [0,1])
    and corresponding binary segmentation masks (H×W, dtype uint8 or bool),
    then converts each mask into per-instance bounding boxes + binary masks
    by running connected-component labelling.

    Args:
        images (list[np.ndarray]): Float32 RGB images, shape (H, W, 3).
        masks  (list[np.ndarray]): Binary masks, shape (H, W), values {0,1}.
        transforms: Optional torchvision transform applied to the image tensor.
        min_area (int): Connected components smaller than this (pixels) are
                        discarded to avoid degenerate boxes.
    """

    def __init__(self, images, masks, transforms=None, min_area=16):
        self.images     = images
        self.masks      = masks
        self.transforms = transforms
        self.min_area   = min_area

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = np.asarray(self.images[idx], dtype=np.float32)
        mask  = np.asarray(self.masks[idx],  dtype=np.uint8)

        # HWC → CHW tensor
        img_tensor = torch.from_numpy(image.transpose(2, 0, 1))  # (3, H, W)

        # Connected-component labelling to get instances
        labeled, num_objs = ndi.label(mask)

        boxes        = []
        instance_masks = []
        labels       = []

        for inst_id in range(1, num_objs + 1):
            inst_mask = (labeled == inst_id).astype(np.uint8)
            area = inst_mask.sum()
            if area < self.min_area:
                continue

            rows = np.any(inst_mask, axis=1)
            cols = np.any(inst_mask, axis=0)
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]

            # Degenerate box guard
            if rmax <= rmin or cmax <= cmin:
                continue

            boxes.append([cmin, rmin, cmax, rmax])
            instance_masks.append(inst_mask)
            labels.append(1)  # class 1 = dead tree

        if len(boxes) == 0:
            # Return a dummy annotation for images with no valid instances
            boxes_t  = torch.zeros((0, 4), dtype=torch.float32)
            masks_t  = torch.zeros((0, mask.shape[0], mask.shape[1]), dtype=torch.uint8)
            labels_t = torch.zeros(0, dtype=torch.int64)
        else:
            boxes_t  = torch.tensor(boxes,          dtype=torch.float32)
            masks_t  = torch.tensor(
                np.stack(instance_masks, axis=0), dtype=torch.uint8
            )
            labels_t = torch.tensor(labels, dtype=torch.int64)

        target = {
            'boxes':   boxes_t,
            'masks':   masks_t,
            'labels':  labels_t,
            'image_id': torch.tensor([idx]),
            'area':    (boxes_t[:, 3] - boxes_t[:, 1]) * (boxes_t[:, 2] - boxes_t[:, 0])
                       if len(boxes) > 0 else torch.zeros(0),
            'iscrowd': torch.zeros(len(labels), dtype=torch.int64),
        }

        if self.transforms is not None:
            img_tensor = self.transforms(img_tensor)

        return img_tensor, target


# ── Model factory ─────────────────────────────────────────────────────────────

def build_maskrcnn(num_classes=2, pretrained=True):
    """
    Build a Mask R-CNN model with ResNet-50 + FPN backbone.

    The classification and mask heads are replaced to match `num_classes`
    (background + N foreground classes).  For binary dead-tree detection
    use the default num_classes=2.

    Args:
        num_classes (int): Total number of classes including background.
        pretrained  (bool): Use COCO-pretrained weights for the backbone.

    Returns:
        torchvision MaskRCNN model ready for fine-tuning.
    """
    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    model = maskrcnn_resnet50_fpn(weights=weights)

    # Replace box classifier head
    in_features_box = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features_box, num_classes)

    # Replace mask head
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer     = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )

    return model


# ── Optimizer ─────────────────────────────────────────────────────────────────

def get_maskrcnn_optimizer(model, lr_backbone=0.005, lr_head=0.01):
    """
    Create an SGD optimizer with differential learning rates:
    lower LR for the frozen backbone, higher LR for the detection heads.

    Args:
        model:       Mask R-CNN model from build_maskrcnn().
        lr_backbone: Learning rate for backbone parameters.
        lr_head:     Learning rate for RPN + RoI head parameters.

    Returns:
        torch.optim.SGD optimizer.
    """
    backbone_params = []
    head_params     = []

    backbone_names = {'backbone', 'fpn'}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        top_module = name.split('.')[0]
        if top_module in backbone_names:
            backbone_params.append(param)
        else:
            head_params.append(param)

    param_groups = [
        {'params': backbone_params, 'lr': lr_backbone},
        {'params': head_params,     'lr': lr_head},
    ]

    optimizer = torch.optim.SGD(
        param_groups,
        momentum=0.9,
        weight_decay=5e-4,
    )
    return optimizer


# ── Training loop ─────────────────────────────────────────────────────────────

def train_maskrcnn(
    model,
    data_loader,
    optimizer,
    device,
    epochs=20,
    val_loader=None,
    save_path="outputs/best_maskrcnn.pth",
    warmup_iters=100,
):
    """
    Full training loop for Mask R-CNN with a cosine LR scheduler and
    optional linear warmup.

    The combined loss (classification + box regression + mask) is logged
    each epoch.  The best checkpoint (lowest val loss, or lowest train loss
    when no val_loader is given) is saved to `save_path`.

    Args:
        model:       Mask R-CNN model.
        data_loader: Training DataLoader yielding (images, targets) batches.
        optimizer:   Optimizer from get_maskrcnn_optimizer() or any torch optimizer.
        device:      torch.device('cuda') or torch.device('cpu').
        epochs:      Number of training epochs.
        val_loader:  Optional validation DataLoader (same format as data_loader).
        save_path:   Path to save the best model weights.
        warmup_iters:Number of iterations for linear LR warmup.

    Returns:
        history (dict): Keys 'train_loss' and optionally 'val_loss',
                        each a list of per-epoch average losses.
    """
    model.to(device)

    # Cosine annealing over the full training run
    total_steps  = len(data_loader) * epochs
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=1e-6
    )

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)

    best_loss  = float('inf')
    history    = {'train_loss': []}
    if val_loader:
        history['val_loss'] = []

    global_step = 0

    for epoch in range(1, epochs + 1):
        # ── Training ─────────────────────────────────────────────────────────
        model.train()
        epoch_loss = 0.0

        for images, targets in data_loader:
            images  = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Linear warmup override
            if global_step < warmup_iters:
                warmup_factor = (global_step + 1) / warmup_iters
                for pg in optimizer.param_groups:
                    pg['lr'] = pg.get('initial_lr', pg['lr']) * warmup_factor

            loss_dict = model(images, targets)
            losses    = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            if global_step >= warmup_iters:
                lr_scheduler.step()

            epoch_loss  += losses.item()
            global_step += 1

        avg_train = epoch_loss / max(len(data_loader), 1)
        history['train_loss'].append(avg_train)

        # ── Validation ───────────────────────────────────────────────────────
        if val_loader is not None:
            avg_val = _evaluate_loss(model, val_loader, device)
            history['val_loss'].append(avg_val)
            monitor = avg_val
            print(f"Epoch {epoch}/{epochs}  train_loss={avg_train:.4f}  val_loss={avg_val:.4f}")
        else:
            monitor = avg_train
            print(f"Epoch {epoch}/{epochs}  train_loss={avg_train:.4f}")

        # ── Checkpoint ───────────────────────────────────────────────────────
        if monitor < best_loss:
            best_loss = monitor
            torch.save(model.state_dict(), save_path)
            print(f"  --> Saved best model to {save_path} (loss={best_loss:.4f})")

    return history


def _evaluate_loss(model, data_loader, device):
    """Compute average combined loss over a data loader (train mode, no grad)."""
    model.train()  # Mask R-CNN needs train mode to return loss dicts
    total = 0.0
    with torch.no_grad():
        for images, targets in data_loader:
            images  = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            total += sum(l.item() for l in loss_dict.values())
    return total / max(len(data_loader), 1)


# ── Inference ─────────────────────────────────────────────────────────────────

def predict_maskrcnn(
    model,
    image_rgb,
    score_threshold=0.5,
    nms_threshold=0.4,
    device=None,
):
    """
    Run inference on a single RGB image and return a combined binary mask.

    Detections with score < score_threshold are discarded.  Remaining masks
    are combined with logical OR to produce a single (H, W) binary array.
    NMS is applied on predicted boxes before combining masks.

    Args:
        model:           Mask R-CNN model (eval mode will be set internally).
        image_rgb:       np.ndarray, float32, shape (H, W, 3), values in [0, 1].
        score_threshold: Minimum detection confidence to keep.
        nms_threshold:   IoU threshold for NMS.
        device:          torch.device (auto-detected if None).

    Returns:
        combined_mask (np.ndarray): Binary mask (H, W), dtype uint8, values {0,1}.
        scores        (np.ndarray): Confidence scores of kept detections.
        boxes         (np.ndarray): Bounding boxes (N, 4) in [x1,y1,x2,y2] format.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.to(device)
    model.eval()

    # Prepare input
    img_t = torch.from_numpy(
        np.asarray(image_rgb, dtype=np.float32).transpose(2, 0, 1)
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        preds = model(img_t)

    pred    = preds[0]
    boxes_t = pred['boxes']
    scores_t = pred['scores']
    masks_t  = pred['masks']   # (N, 1, H, W) float in [0,1]

    # Score threshold
    keep = scores_t >= score_threshold
    boxes_t  = boxes_t[keep]
    scores_t = scores_t[keep]
    masks_t  = masks_t[keep]

    # NMS
    if len(boxes_t) > 0:
        nms_keep = ops.nms(boxes_t, scores_t, iou_threshold=nms_threshold)
        boxes_t  = boxes_t[nms_keep]
        scores_t = scores_t[nms_keep]
        masks_t  = masks_t[nms_keep]

    H, W = image_rgb.shape[:2]
    combined = np.zeros((H, W), dtype=np.uint8)

    if len(masks_t) > 0:
        binary_masks = (masks_t[:, 0, :, :].cpu().numpy() > 0.5).astype(np.uint8)
        combined     = np.any(binary_masks, axis=0).astype(np.uint8)

    return (
        combined,
        scores_t.cpu().numpy(),
        boxes_t.cpu().numpy(),
    )


# ── Convenience: collate function for DataLoader ──────────────────────────────

def maskrcnn_collate_fn(batch):
    """
    Custom collate function required by Mask R-CNN's DataLoader
    (images and targets must remain as lists, not stacked tensors).
    """
    images  = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets


# ── Quick smoke-test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = build_maskrcnn(num_classes=2, pretrained=True)
    model.to(device)

    # Dummy forward pass
    dummy_img = [torch.rand(3, 224, 224).to(device)]
    model.eval()
    with torch.no_grad():
        out = model(dummy_img)
    print("Inference keys:", list(out[0].keys()))
    print("boxes shape:  ", out[0]['boxes'].shape)
    print("masks shape:  ", out[0]['masks'].shape)
    print("Mask R-CNN smoke-test passed.")
