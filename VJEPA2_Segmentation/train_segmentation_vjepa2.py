"""
train_segmentation_vjepa2.py — Supervised segmentation fine-tuning with V-JEPA 2 backbone.

Strategy:
  1. Load pretrained V-JEPA 2 encoder (frozen)
  2. Attach a multi-scale segmentation decoder
  3. Train on labeled image-mask pairs (269 samples)
  4. After N epochs, optionally unfreeze encoder with low LR
  5. Save best model based on validation IoU

Usage:
    python train_segmentation_vjepa2.py --config configs/train.yaml
"""

import os
import sys
import argparse
import random
import time
import logging

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset_vjepa2 import get_vjepa2_seg_loaders


# ============================================================
#  Segmentation Decoder
# ============================================================

class VJEPA2SegmentationDecoder(nn.Module):
    """Multi-scale decoder for V-JEPA 2 features → segmentation mask.
    
    Takes ViT patch features (B, N, D), reshapes to (B, D, H_p, W_p),
    then upsamples through transposed convolutions to produce a mask.
    """
    
    def __init__(self, embed_dim=1024, patch_size=16, img_size=224,
                 decoder_channels=None, num_classes=1):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.num_patches_h = img_size // patch_size  # 14
        self.num_patches_w = img_size // patch_size
        
        if decoder_channels is None:
            decoder_channels = [512, 256, 128, 64]
        
        # Build upsampling layers
        layers = []
        in_ch = embed_dim
        for out_ch in decoder_channels:
            layers.append(nn.ConvTranspose2d(
                in_ch, out_ch, kernel_size=2, stride=2,
            ))
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.GELU())
            in_ch = out_ch
        
        self.decoder = nn.Sequential(*layers)
        
        # Final prediction head
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, in_ch // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_ch // 2),
            nn.GELU(),
            nn.Conv2d(in_ch // 2, num_classes, kernel_size=1),
        )
    
    def forward(self, features):
        """
        Args:
            features: (B, N, D) patch features from V-JEPA 2 encoder
                      May include CLS token — we skip it if N > H_p * W_p
        Returns:
            logits: (B, num_classes, H, W) segmentation logits
        """
        B, N, D = features.shape
        num_spatial = self.num_patches_h * self.num_patches_w
        
        # Skip CLS/extra tokens if present
        if N > num_spatial:
            features = features[:, -num_spatial:]
        
        # Reshape to spatial: (B, D, H_p, W_p)
        x = features.permute(0, 2, 1).reshape(
            B, D, self.num_patches_h, self.num_patches_w,
        )
        
        # Upsample through decoder
        x = self.decoder(x)
        
        # Final prediction
        logits = self.head(x)
        
        return logits


class VJEPA2SegmentationModel(nn.Module):
    """Full segmentation model: V-JEPA 2 encoder + decoder."""
    
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, images):
        """
        Args:
            images: (B, C, H, W) single images
        Returns:
            logits: (B, 1, H', W') segmentation logits
        """
        # V-JEPA 2 expects video input: (B, C, T, H, W)
        # For single frames, repeat to create a mini-clip
        B, C, H, W = images.shape
        # Use a single-frame clip (T=2 for tubelet compatibility)
        video = images.unsqueeze(2).repeat(1, 1, 2, 1, 1)  # (B, C, 2, H, W)
        
        # Get features from encoder
        features = self.encoder(video)  # (B, N, D)
        
        # If features include temporal dimension, pool over time
        # V-JEPA 2 with T=2, tubelet_size=2 → 1 temporal patch
        # So features should already be spatial-only
        
        # Decode to segmentation mask
        logits = self.decoder(features)
        
        return logits


# ============================================================
#  Loss Functions
# ============================================================

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target):
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        p_t = torch.sigmoid(pred) * target + (1 - torch.sigmoid(pred)) * (1 - target)
        focal_weight = (1 - p_t) ** self.gamma
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        return (alpha_t * focal_weight * bce).mean()


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = torch.sigmoid(pred).view(-1)
        target = target.view(-1)
        intersection = (pred * target).sum()
        return 1 - (2 * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)


class CombinedLoss(nn.Module):
    def __init__(self, focal_weight=20.0, dice_weight=1.0):
        super().__init__()
        self.focal = FocalLoss()
        self.dice = DiceLoss()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
    
    def forward(self, pred_masks, gt_masks):
        if pred_masks.shape[-2:] != gt_masks.shape[-2:]:
            pred_masks = F.interpolate(
                pred_masks, gt_masks.shape[-2:],
                mode="bilinear", align_corners=False,
            )
        focal_loss = self.focal(pred_masks, gt_masks)
        dice_loss = self.dice(pred_masks, gt_masks)
        total = self.focal_weight * focal_loss + self.dice_weight * dice_loss
        return total, {
            "focal": focal_loss.item(),
            "dice": dice_loss.item(),
            "total": total.item(),
        }


# ============================================================
#  Metrics
# ============================================================

def compute_metrics(pred_masks, gt_masks, threshold=0.5):
    pred = (torch.sigmoid(pred_masks) > threshold).float()
    if pred.shape[-2:] != gt_masks.shape[-2:]:
        pred = F.interpolate(pred, gt_masks.shape[-2:], mode="nearest")
    
    intersection = (pred * gt_masks).sum(dim=(-2, -1))
    union = pred.sum(dim=(-2, -1)) + gt_masks.sum(dim=(-2, -1)) - intersection
    
    iou = (intersection / (union + 1e-6)).mean().item()
    dice = (2 * intersection / (pred.sum(dim=(-2, -1)) + gt_masks.sum(dim=(-2, -1)) + 1e-6)).mean().item()
    accuracy = (pred == gt_masks).float().mean().item()
    
    return {"iou": iou, "dice": dice, "accuracy": accuracy}


# ============================================================
#  Utilities
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logger(name, log_dir=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(log_dir, "train.log"))
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger


def save_checkpoint(state, save_dir, filename, is_best=False):
    os.makedirs(save_dir, exist_ok=True)
    torch.save(state, os.path.join(save_dir, filename))
    if is_best:
        torch.save(state, os.path.join(save_dir, "best_" + filename))


# ============================================================
#  Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="V-JEPA 2 Segmentation Fine-tuning")
    parser.add_argument("--config", type=str, default="configs/train.yaml")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--pretrained_encoder", type=str, default=None,
                        help="Path to pretrained V-JEPA 2 encoder (from pretrain_vjepa2.py)")
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    
    set_seed(cfg.get("seed", 42))
    device = torch.device(cfg.get("device", "cuda"))
    
    seg_cfg = cfg["segmentation"]
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    ckpt_cfg = cfg["checkpoint"]
    
    log_dir = ckpt_cfg["log_dir_seg"]
    logger = setup_logger("vjepa2_seg", log_dir)
    writer = SummaryWriter(log_dir)
    
    logger.info("=" * 60)
    logger.info("V-JEPA 2 Segmentation Fine-tuning")
    logger.info("=" * 60)
    
    # ---- Data ----
    train_loader, val_loader = get_vjepa2_seg_loaders(
        processed_dir=data_cfg["processed_dir"],
        crop_size=data_cfg.get("crop_size", 224),
        batch_size=seg_cfg["batch_size"],
        num_workers=data_cfg.get("num_workers", 4),
        seed=cfg.get("seed", 42),
    )
    logger.info(f"Data: train={len(train_loader.dataset)}, val={len(val_loader.dataset)}")
    
    # ---- Load V-JEPA 2 Encoder ----
    logger.info("Loading V-JEPA 2 ViT-Large via PyTorch Hub...")
    
    # torch.hub.load returns (encoder, predictor) tuple
    hub_result = torch.hub.load(
        "facebookresearch/vjepa2", "vjepa2_vit_large",
        force_reload=False,
    )
    encoder = hub_result[0]  # VisionTransformer (303.9M params)
    
    # Load our pretrained encoder weights if available
    pretrained_path = args.pretrained_encoder or os.path.join(
        ckpt_cfg["save_dir"], "vjepa2_encoder_final.pth"
    )
    if os.path.exists(pretrained_path):
        logger.info(f"Loading pretrained encoder from: {pretrained_path}")
        state_dict = torch.load(pretrained_path, map_location=device)
        encoder.load_state_dict(state_dict, strict=False)
    else:
        logger.info("Using default V-JEPA 2 pretrained weights (no domain pretraining)")
    
    encoder = encoder.to(device)
    
    total_params = sum(p.numel() for p in encoder.parameters()) / 1e6
    logger.info(f"Encoder params: {total_params:.1f}M")
    
    # ---- Freeze/Unfreeze Strategy ----
    freeze_encoder = seg_cfg.get("freeze_encoder", True)
    
    if freeze_encoder:
        for param in encoder.parameters():
            param.requires_grad = False
        logger.info("Encoder: FROZEN")
    
    # ---- Segmentation Decoder ----
    embed_dim = model_cfg.get("embed_dim", 1024)
    crop_size = data_cfg.get("crop_size", 224)
    patch_size = model_cfg.get("patch_size", 16)
    decoder_channels = model_cfg.get("decoder_channels", [512, 256, 128, 64])
    
    decoder = VJEPA2SegmentationDecoder(
        embed_dim=embed_dim,
        patch_size=patch_size,
        img_size=crop_size,
        decoder_channels=decoder_channels,
        num_classes=model_cfg.get("num_classes", 1),
    ).to(device)
    
    dec_params = sum(p.numel() for p in decoder.parameters()) / 1e6
    logger.info(f"Decoder params: {dec_params:.1f}M")
    
    # ---- Full Model ----
    model = VJEPA2SegmentationModel(encoder, decoder)
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    logger.info(f"Trainable params: {trainable:.1f}M")
    
    # ---- Optimizer ----
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=seg_cfg["learning_rate"],
        weight_decay=seg_cfg.get("weight_decay", 0.01),
    )
    
    # ---- Loss ----
    criterion = CombinedLoss(
        focal_weight=seg_cfg.get("focal_weight", 20.0),
        dice_weight=seg_cfg.get("dice_weight", 1.0),
    )
    
    # ---- LR Scheduler ----
    epochs = seg_cfg["epochs"]
    warmup_epochs = seg_cfg.get("warmup_epochs", 5)
    
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # ---- Resume ----
    start_epoch = 0
    best_iou = 0.0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        decoder.load_state_dict(ckpt["decoder"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"]
        best_iou = ckpt.get("best_iou", 0.0)
        logger.info(f"Resumed from epoch {start_epoch}, best IoU: {best_iou:.4f}")
    
    # ---- Training Config ----
    save_dir = ckpt_cfg["save_dir"]
    save_every = ckpt_cfg.get("save_every", 10)
    unfreeze_epoch = seg_cfg.get("unfreeze_encoder_epoch", 999)
    grad_clip = seg_cfg.get("grad_clip", 1.0)
    encoder_unfrozen = False
    
    # Early stopping
    es_patience = cfg.get("early_stopping", {}).get("patience", 30)
    patience_counter = 0
    
    # ---- Training Loop ----
    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        
        # --- Unfreeze encoder ---
        if epoch >= unfreeze_epoch and not encoder_unfrozen and freeze_encoder:
            for param in model.encoder.parameters():
                param.requires_grad = True
            encoder_unfrozen = True
            
            enc_lr = seg_cfg.get("encoder_unfreeze_lr", 5e-6)
            optimizer.add_param_group({
                "params": list(model.encoder.parameters()),
                "lr": enc_lr,
            })
            
            new_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
            logger.info(f"Epoch {epoch+1}: Encoder UNFROZEN (lr={enc_lr})")
            logger.info(f"Trainable params now: {new_trainable:.1f}M")
        
        # ---- Train ----
        model.train()
        if freeze_encoder and not encoder_unfrozen:
            model.encoder.eval()
        
        train_loss = 0.0
        train_iou = 0.0
        n_train = 0
        
        for images, masks in tqdm(train_loader, desc=f"  Train E{epoch+1}", leave=False):
            images = images.to(device)
            masks = masks.to(device)
            
            pred_masks = model(images)
            
            # Resize predictions to match masks if needed
            if pred_masks.shape[-2:] != masks.shape[-2:]:
                pred_masks = F.interpolate(
                    pred_masks, masks.shape[-2:],
                    mode="bilinear", align_corners=False,
                )
            
            loss, loss_dict = criterion(pred_masks, masks)
            
            optimizer.zero_grad()
            loss.backward()
            
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    grad_clip,
                )
            
            optimizer.step()
            
            metrics = compute_metrics(pred_masks.detach(), masks)
            train_loss += loss_dict["total"]
            train_iou += metrics["iou"]
            n_train += 1
        
        train_loss /= max(n_train, 1)
        train_iou /= max(n_train, 1)
        
        # ---- Validate ----
        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        val_dice = 0.0
        val_acc = 0.0
        n_val = 0
        
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"  Val   E{epoch+1}", leave=False):
                images = images.to(device)
                masks = masks.to(device)
                
                pred_masks = model(images)
                if pred_masks.shape[-2:] != masks.shape[-2:]:
                    pred_masks = F.interpolate(
                        pred_masks, masks.shape[-2:],
                        mode="bilinear", align_corners=False,
                    )
                
                loss, loss_dict = criterion(pred_masks, masks)
                metrics = compute_metrics(pred_masks, masks)
                
                val_loss += loss_dict["total"]
                val_iou += metrics["iou"]
                val_dice += metrics["dice"]
                val_acc += metrics["accuracy"]
                n_val += 1
        
        val_loss /= max(n_val, 1)
        val_iou /= max(n_val, 1)
        val_dice /= max(n_val, 1)
        val_acc /= max(n_val, 1)
        
        scheduler.step()
        elapsed = time.time() - t0
        
        # ---- Logging ----
        lr = optimizer.param_groups[0]["lr"]
        logger.info(
            f"Epoch {epoch+1:3d}/{epochs} | "
            f"Train loss: {train_loss:.4f} IoU: {train_iou:.4f} | "
            f"Val loss: {val_loss:.4f} IoU: {val_iou:.4f} "
            f"Dice: {val_dice:.4f} Acc: {val_acc:.4f} | "
            f"LR: {lr:.2e} | {elapsed:.1f}s"
        )
        
        writer.add_scalars("loss", {"train": train_loss, "val": val_loss}, epoch)
        writer.add_scalars("iou", {"train": train_iou, "val": val_iou}, epoch)
        writer.add_scalar("val/dice", val_dice, epoch)
        writer.add_scalar("val/accuracy", val_acc, epoch)
        writer.add_scalar("lr", lr, epoch)
        
        # ---- Checkpointing ----
        is_best = val_iou > best_iou
        if is_best:
            best_iou = val_iou
            patience_counter = 0
            logger.info(f"  ★ New best IoU: {best_iou:.4f}")
        else:
            patience_counter += 1
        
        state = {
            "epoch": epoch + 1,
            "decoder": decoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_iou": best_iou,
            "config": cfg,
        }
        
        save_checkpoint(state, save_dir, "vjepa2_seg_checkpoint.pth", is_best=is_best)
        
        if (epoch + 1) % save_every == 0:
            fname = f"vjepa2_seg_epoch{epoch+1}.pth"
            save_checkpoint(state, save_dir, fname)
            logger.info(f"  Saved checkpoint: {fname}")
        
        if is_best:
            torch.save(
                decoder.state_dict(),
                os.path.join(save_dir, "vjepa2_decoder_best.pth"),
            )
        
        # Early stopping
        if patience_counter >= es_patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
    
    logger.info("=" * 60)
    logger.info("Segmentation training complete!")
    logger.info(f"Best validation IoU: {best_iou:.4f}")
    logger.info(f"Decoder saved to: {save_dir}/vjepa2_decoder_best.pth")
    logger.info("=" * 60)
    writer.close()


if __name__ == "__main__":
    main()
