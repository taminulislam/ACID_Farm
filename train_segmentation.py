"""
train_segmentation.py — Fine-tune segmentation model on labeled thermal cow frames.

Uses pretrained I-JEPA encoder + a lightweight decoder head.
Trains on your manually labeled frames from the processed/ folder.

Usage:
    python train_segmentation.py --config configs/segmentation.yaml
    python train_segmentation.py --config configs/segmentation.yaml --resume checkpoints/seg_checkpoint.pth
    python train_segmentation.py --config configs/segmentation.yaml --encoder_weights checkpoints/ijepa_encoder_best.pth
"""

import os
import argparse
import random

import yaml
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import get_segmentation_loaders
from models import SegmentationModel
from utils import (
    setup_logger,
    DiceBCELoss,
    compute_iou,
    compute_dice,
    compute_pixel_accuracy,
    save_checkpoint,
    load_checkpoint,
    cosine_scheduler,
    EarlyStopping,
    Timer,
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train one epoch."""
    model.train()
    total_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0
    n = 0

    for images, masks in tqdm(loader, desc="  Train", leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_iou += compute_iou(logits, masks)
        total_dice += compute_dice(logits, masks)
        n += 1

    return {
        "loss": total_loss / n,
        "iou": total_iou / n,
        "dice": total_dice / n,
    }


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Validate on the validation set."""
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0
    total_acc = 0.0
    n = 0

    for images, masks in tqdm(loader, desc="  Val  ", leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, masks)

        total_loss += loss.item()
        total_iou += compute_iou(logits, masks)
        total_dice += compute_dice(logits, masks)
        total_acc += compute_pixel_accuracy(logits, masks)
        n += 1

    return {
        "loss": total_loss / n,
        "iou": total_iou / n,
        "dice": total_dice / n,
        "accuracy": total_acc / n,
    }


def main():
    parser = argparse.ArgumentParser(description="Segmentation Training")
    parser.add_argument("--config", type=str, default="configs/segmentation.yaml")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from a segmentation checkpoint")
    parser.add_argument("--encoder_weights", type=str, default=None,
                        help="Override encoder weights path from config")
    parser.add_argument("--no_pretrain", action="store_true",
                        help="Train from scratch (no I-JEPA pretraining)")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("seed", 42))
    device = torch.device(cfg.get("device", "cuda"))

    # Setup logging
    log_dir = cfg["checkpoint"]["log_dir"]
    logger = setup_logger("segmentation", log_dir)
    writer = SummaryWriter(log_dir)

    logger.info("=" * 60)
    logger.info("Segmentation Fine-tuning")
    logger.info("=" * 60)

    # ---- Data ----
    train_loader, val_loader = get_segmentation_loaders(
        processed_dir=cfg["data"]["processed_dir"],
        crop_size=cfg["data"].get("crop_size", 224),
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["data"].get("num_workers", 4),
        train_split=cfg["data"].get("train_split", 0.85),
        cfg=cfg,
        seed=cfg.get("seed", 42),
    )
    logger.info(
        f"Data: train={len(train_loader.dataset)}, "
        f"val={len(val_loader.dataset)}"
    )

    # ---- Model ----
    encoder_path = args.encoder_weights or cfg["model"].get("encoder_weights", None)

    if args.no_pretrain:
        logger.info("Training from SCRATCH (no pretrained encoder)")
        encoder_path = None
    elif encoder_path and os.path.exists(encoder_path):
        logger.info(f"Loading pretrained encoder: {encoder_path}")
    else:
        logger.warning(
            f"Encoder weights not found at '{encoder_path}'. "
            "Training from scratch. Run pretrain_ijepa.py first for best results!"
        )
        encoder_path = None

    model = SegmentationModel.from_pretrained(encoder_path, cfg)
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"Total params: {n_params:.1f}M")

    # ---- Freeze encoder initially ----
    freeze_epochs = cfg["training"].get("freeze_encoder_epochs", 10)
    if freeze_epochs > 0 and encoder_path:
        model.freeze_encoder()
        logger.info(f"Encoder FROZEN for first {freeze_epochs} epochs")

    # ---- Optimizer with differential learning rates ----
    encoder_params = list(model.encoder.parameters())
    decoder_params = list(model.decoder.parameters())

    optimizer = torch.optim.AdamW([
        {"params": encoder_params, "lr": cfg["training"]["encoder_lr"]},
        {"params": decoder_params, "lr": cfg["training"]["decoder_lr"]},
    ], weight_decay=cfg["training"].get("weight_decay", 0.01))

    # ---- Loss ----
    criterion = DiceBCELoss(
        dice_weight=cfg["training"].get("dice_weight", 1.0),
        bce_weight=cfg["training"].get("bce_weight", 1.0),
    )

    # ---- LR Schedule ----
    epochs = cfg["training"]["epochs"]
    lr_schedule = cosine_scheduler(
        cfg["training"]["decoder_lr"],
        1e-6,
        epochs,
        cfg["training"].get("warmup_epochs", 5),
    )

    # ---- Early Stopping ----
    es_cfg = cfg.get("early_stopping", {})
    early_stopper = EarlyStopping(
        patience=es_cfg.get("patience", 20),
        min_delta=es_cfg.get("min_delta", 0.001),
        mode="max",
    )

    # ---- Resume ----
    start_epoch = 0
    best_iou = 0.0
    if args.resume:
        start_epoch, best_iou = load_checkpoint(
            args.resume, model=model, optimizer=optimizer, device=device,
        )
        logger.info(f"Resumed from epoch {start_epoch}, best IoU: {best_iou:.4f}")

    # ---- Training loop ----
    timer = Timer()
    save_dir = cfg["checkpoint"]["save_dir"]

    for epoch in range(start_epoch, epochs):
        timer.start()

        # Unfreeze encoder after warmup
        if epoch == freeze_epochs and encoder_path:
            model.unfreeze_encoder()
            logger.info(f"Epoch {epoch+1}: Encoder UNFROZEN — full fine-tuning")

        # Update learning rate
        lr = lr_schedule[min(epoch, len(lr_schedule) - 1)]
        for pg in optimizer.param_groups:
            if pg is optimizer.param_groups[0]:  # encoder
                pg["lr"] = lr * (cfg["training"]["encoder_lr"] /
                                  cfg["training"]["decoder_lr"])
            else:  # decoder
                pg["lr"] = lr

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
        )

        # Validate
        val_metrics = validate(model, val_loader, criterion, device)

        elapsed = timer.elapsed_str()

        logger.info(
            f"Epoch {epoch+1:3d}/{epochs} | "
            f"Train loss: {train_metrics['loss']:.4f} IoU: {train_metrics['iou']:.4f} | "
            f"Val loss: {val_metrics['loss']:.4f} IoU: {val_metrics['iou']:.4f} "
            f"Dice: {val_metrics['dice']:.4f} Acc: {val_metrics['accuracy']:.4f} | "
            f"{elapsed}"
        )

        # TensorBoard
        writer.add_scalars("loss", {
            "train": train_metrics["loss"], "val": val_metrics["loss"],
        }, epoch)
        writer.add_scalars("iou", {
            "train": train_metrics["iou"], "val": val_metrics["iou"],
        }, epoch)
        writer.add_scalar("val/dice", val_metrics["dice"], epoch)
        writer.add_scalar("val/accuracy", val_metrics["accuracy"], epoch)
        writer.add_scalar("lr", lr, epoch)

        # Save checkpoint
        is_best = val_metrics["iou"] > best_iou
        if is_best:
            best_iou = val_metrics["iou"]
            logger.info(f"  ★ New best IoU: {best_iou:.4f}")

        state = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_metric": best_iou,
            "config": cfg,
        }
        
        # Always save latest checkpoint (overwrites)
        save_checkpoint(state, save_dir, "seg_checkpoint.pth", is_best=is_best)

        # Save numbered checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            fname = f"seg_checkpoint_epoch{epoch+1}.pth"
            save_checkpoint(state, save_dir, fname, is_best=False)
            logger.info(f"  Saved checkpoint: {fname}")

        if is_best:
            # Also save just the model for easy loading
            torch.save(model.state_dict(),
                        os.path.join(save_dir, "segmentation_best.pth"))

        # Early stopping
        if early_stopper(val_metrics["iou"]):
            logger.info(f"Early stopping at epoch {epoch+1}")
            break

    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info(f"Best validation IoU: {best_iou:.4f}")
    logger.info(f"Model saved to: {save_dir}/segmentation_best.pth")
    logger.info("=" * 60)
    writer.close()


if __name__ == "__main__":
    main()
