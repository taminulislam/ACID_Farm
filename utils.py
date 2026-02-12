"""
utils.py — Shared utilities for the I-JEPA segmentation pipeline.

Includes: metrics, loss functions, checkpoint helpers, logging setup.
"""

import os
import time
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ============================================================
#  Logging
# ============================================================

def setup_logger(name, log_dir=None, level=logging.INFO):
    """Create a logger that writes to console and optionally to file."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(log_dir, "train.log"))
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# ============================================================
#  Segmentation Loss
# ============================================================

class DiceBCELoss(nn.Module):
    """Combined Dice + Binary Cross-Entropy loss for binary segmentation."""

    def __init__(self, dice_weight=1.0, bce_weight=1.0, smooth=1.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        """
        Args:
            logits:  (B, 1, H, W) raw predictions
            targets: (B, 1, H, W) binary ground truth
        """
        # BCE loss
        bce_loss = self.bce(logits, targets)

        # Dice loss
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (probs_flat * targets_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (
            probs_flat.sum() + targets_flat.sum() + self.smooth
        )
        dice_loss = 1.0 - dice

        return self.dice_weight * dice_loss + self.bce_weight * bce_loss


# ============================================================
#  Metrics
# ============================================================

def compute_iou(logits, targets, threshold=0.5):
    """Compute Intersection over Union (IoU / Jaccard Index)."""
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    targets = (targets > threshold).float()

    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum() - intersection

    if union == 0:
        return 1.0  # both empty

    return (intersection / union).item()


def compute_dice(logits, targets, threshold=0.5):
    """Compute Dice coefficient (F1 for segmentation)."""
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    targets = (targets > threshold).float()

    intersection = (preds * targets).sum()
    total = preds.sum() + targets.sum()

    if total == 0:
        return 1.0

    return (2.0 * intersection / total).item()


def compute_pixel_accuracy(logits, targets, threshold=0.5):
    """Compute pixel-wise accuracy."""
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    targets = (targets > threshold).float()
    correct = (preds == targets).float().sum()
    total = targets.numel()
    return (correct / total).item()


# ============================================================
#  Checkpoint helpers
# ============================================================

def save_checkpoint(state, save_dir, filename="checkpoint.pth", is_best=False):
    """Save a training checkpoint."""
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    torch.save(state, path)
    if is_best:
        best_path = os.path.join(save_dir, "best_" + filename)
        torch.save(state, best_path)
    return path


def load_checkpoint(path, model=None, optimizer=None, device="cpu"):
    """Load a training checkpoint."""
    checkpoint = torch.load(path, map_location=device)

    if model is not None:
        if "model" in checkpoint:
            model.load_state_dict(checkpoint["model"], strict=False)
        elif "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"], strict=False)

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    epoch = checkpoint.get("epoch", 0)
    best_metric = checkpoint.get("best_metric", 0.0)

    return epoch, best_metric


# ============================================================
#  Learning Rate Scheduling
# ============================================================

def cosine_scheduler(base_value, final_value, epochs, warmup_epochs=0):
    """Generate a cosine annealing schedule with linear warmup."""
    warmup_schedule = np.linspace(0, base_value, warmup_epochs)
    cos_epochs = epochs - warmup_epochs
    cos_schedule = final_value + 0.5 * (base_value - final_value) * (
        1 + np.cos(np.pi * np.arange(cos_epochs) / cos_epochs)
    )
    schedule = np.concatenate([warmup_schedule, cos_schedule])
    return schedule


def ema_scheduler(start, end, epochs):
    """Generate an EMA momentum schedule (linear interpolation)."""
    return np.linspace(start, end, epochs)


# ============================================================
#  Early Stopping
# ============================================================

class EarlyStopping:
    """Early stopping to stop training if validation metric stops improving."""

    def __init__(self, patience=20, min_delta=0.001, mode="max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best = None
        self.should_stop = False

    def __call__(self, metric):
        if self.best is None:
            self.best = metric
            return False

        if self.mode == "max":
            improved = metric > self.best + self.min_delta
        else:
            improved = metric < self.best - self.min_delta

        if improved:
            self.best = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


# ============================================================
#  Timer
# ============================================================

class Timer:
    """Simple timer for tracking epoch duration."""

    def __init__(self):
        self.start_time = None

    def start(self):
        self.start_time = time.time()

    def elapsed(self):
        return time.time() - self.start_time

    def elapsed_str(self):
        s = self.elapsed()
        if s < 60:
            return f"{s:.1f}s"
        m = int(s // 60)
        s = s % 60
        return f"{m}m {s:.1f}s"
