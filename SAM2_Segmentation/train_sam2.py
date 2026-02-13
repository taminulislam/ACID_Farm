"""
train_sam2.py — Fine-tune SAM 2.1 mask decoder on labeled thermal cow frames.

Strategy:
1. Load pretrained SAM 2.1 (Hiera Small)
2. Freeze image encoder + prompt encoder
3. Fine-tune ONLY the mask decoder
4. After N epochs, optionally unfreeze the image encoder with low LR
5. Use point prompts + bounding box prompts from GT masks

Usage:
    python train_sam2.py --config configs/train.yaml
    python train_sam2.py --config configs/train.yaml --resume checkpoints/sam2_checkpoint.pth
"""

import os
import sys
import argparse
import random

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset_sam2 import get_sam2_dataloaders


# ---- Loss Functions ----

class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance."""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target):
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        p = torch.sigmoid(pred)
        p_t = p * target + (1 - p) * (1 - target)
        focal_weight = (1 - p_t) ** self.gamma
        
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        loss = alpha_t * focal_weight * bce
        return loss.mean()


class DiceLoss(nn.Module):
    """Dice loss for segmentation."""
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice


class SAM2Loss(nn.Module):
    """Combined loss for SAM 2 fine-tuning: Focal + Dice + IoU prediction."""
    def __init__(self, focal_weight=20.0, dice_weight=1.0, iou_weight=1.0):
        super().__init__()
        self.focal = FocalLoss()
        self.dice = DiceLoss()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.iou_weight = iou_weight
    
    def forward(self, pred_masks, pred_iou, gt_masks):
        """
        pred_masks: (B, 1, H, W) logits
        pred_iou: (B, 1) predicted IoU scores
        gt_masks: (B, 1, H, W) binary
        """
        # Resize predictions to match GT if needed
        if pred_masks.shape[-2:] != gt_masks.shape[-2:]:
            pred_masks = F.interpolate(
                pred_masks, gt_masks.shape[-2:],
                mode="bilinear", align_corners=False,
            )
        
        # Mask losses
        focal_loss = self.focal(pred_masks, gt_masks)
        dice_loss = self.dice(pred_masks, gt_masks)
        
        # IoU prediction loss
        with torch.no_grad():
            pred_binary = (torch.sigmoid(pred_masks) > 0.5).float()
            intersection = (pred_binary * gt_masks).sum(dim=(-2, -1))
            union = pred_binary.sum(dim=(-2, -1)) + gt_masks.sum(dim=(-2, -1)) - intersection
            gt_iou = (intersection / (union + 1e-6)).squeeze(1)  # (B,)
        
        iou_loss = F.mse_loss(pred_iou.squeeze(1), gt_iou)
        
        total = (self.focal_weight * focal_loss +
                 self.dice_weight * dice_loss +
                 self.iou_weight * iou_loss)
        
        return total, {
            "focal": focal_loss.item(),
            "dice": dice_loss.item(),
            "iou_pred": iou_loss.item(),
            "total": total.item(),
        }


# ---- Metrics ----

def compute_metrics(pred_masks, gt_masks, threshold=0.5):
    """Compute IoU, Dice, and accuracy."""
    pred = (torch.sigmoid(pred_masks) > threshold).float()
    
    # Resize if needed
    if pred.shape[-2:] != gt_masks.shape[-2:]:
        pred = F.interpolate(pred, gt_masks.shape[-2:], mode="nearest")
    
    intersection = (pred * gt_masks).sum(dim=(-2, -1))
    union = pred.sum(dim=(-2, -1)) + gt_masks.sum(dim=(-2, -1)) - intersection
    
    iou = (intersection / (union + 1e-6)).mean().item()
    dice = (2.0 * intersection / (pred.sum(dim=(-2, -1)) + gt_masks.sum(dim=(-2, -1)) + 1e-6)).mean().item()
    accuracy = ((pred == gt_masks).float().mean()).item()
    
    return {"iou": iou, "dice": dice, "accuracy": accuracy}


# ---- Checkpoint Utilities ----

def save_checkpoint(state, save_dir, filename, is_best=False):
    """Save a checkpoint."""
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    torch.save(state, filepath)
    if is_best:
        best_path = os.path.join(save_dir, "best_" + filename)
        torch.save(state, best_path)


def setup_logger(name, log_dir=None):
    """Simple logger setup."""
    import logging
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # File handler
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(log_dir, "train.log"))
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
    return logger


# ---- Main Training ----

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="SAM 2 Fine-tuning")
    parser.add_argument("--config", type=str, default="configs/train.yaml")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint")
    args = parser.parse_args()
    
    # Load config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    
    set_seed(cfg.get("seed", 42))
    device = torch.device(cfg.get("device", "cuda"))
    
    # Setup logging
    log_dir = cfg["checkpoint"]["log_dir"]
    logger = setup_logger("sam2_train", log_dir)
    writer = SummaryWriter(log_dir)
    
    logger.info("=" * 60)
    logger.info("SAM 2.1 Fine-tuning for Thermal Cow Segmentation")
    logger.info("=" * 60)
    
    # ---- Data ----
    train_loader, val_loader = get_sam2_dataloaders(
        processed_dir=cfg["data"]["processed_dir"],
        img_size=cfg["data"]["crop_size"],
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["data"].get("num_workers", 4),
        train_split=cfg["data"].get("train_split", 0.85),
        cfg=cfg,
        seed=cfg.get("seed", 42),
    )
    logger.info(f"Data: train={len(train_loader.dataset)}, val={len(val_loader.dataset)}")
    
    # ---- Load SAM 2 Model ----
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    
    model_cfg = cfg["model"]["model_cfg"]
    checkpoint_path = cfg["model"]["checkpoint"]
    
    logger.info(f"Loading SAM 2.1 from: {checkpoint_path}")
    logger.info(f"Model config: {model_cfg}")
    
    sam2_model = build_sam2(model_cfg, checkpoint_path, device=device)
    
    # Count parameters
    total_params = sum(p.numel() for p in sam2_model.parameters()) / 1e6
    logger.info(f"Total SAM 2 params: {total_params:.1f}M")
    
    # ---- Freeze/Unfreeze Strategy ----
    freeze_encoder = cfg["model"].get("freeze_image_encoder", True)
    freeze_prompt = cfg["model"].get("freeze_prompt_encoder", True)
    
    if freeze_encoder:
        for param in sam2_model.image_encoder.parameters():
            param.requires_grad = False
        logger.info("Image encoder: FROZEN")
    
    if freeze_prompt:
        for param in sam2_model.sam_prompt_encoder.parameters():
            param.requires_grad = False
        logger.info("Prompt encoder: FROZEN")
    
    # Mask decoder is always trainable
    for param in sam2_model.sam_mask_decoder.parameters():
        param.requires_grad = True
    
    trainable = sum(p.numel() for p in sam2_model.parameters() if p.requires_grad) / 1e6
    logger.info(f"Trainable params: {trainable:.1f}M")
    
    # ---- Optimizer ----
    decoder_params = list(sam2_model.sam_mask_decoder.parameters())
    optimizer = torch.optim.AdamW(
        decoder_params,
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"].get("weight_decay", 0.01),
    )
    
    # ---- Loss ----
    criterion = SAM2Loss(
        focal_weight=cfg["training"].get("focal_weight", 20.0),
        dice_weight=cfg["training"].get("dice_weight", 1.0),
        iou_weight=cfg["training"].get("iou_weight", 1.0),
    )
    
    # ---- LR Scheduler ----
    epochs = cfg["training"]["epochs"]
    warmup_epochs = cfg["training"].get("warmup_epochs", 5)
    
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # ---- Resume ----
    start_epoch = 0
    best_iou = 0.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        sam2_model.sam_mask_decoder.load_state_dict(ckpt["mask_decoder"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"]
        best_iou = ckpt.get("best_iou", 0.0)
        logger.info(f"Resumed from epoch {start_epoch}, best IoU: {best_iou:.4f}")
    
    # ---- Early Stopping ----
    es_patience = cfg.get("early_stopping", {}).get("patience", 30)
    es_min_delta = cfg.get("early_stopping", {}).get("min_delta", 0.001)
    patience_counter = 0
    
    # ---- Helper: forward pass through SAM 2 ----
    # Spatial sizes for backbone feature pyramid (matching SAM2ImagePredictor)
    _bb_feat_sizes = [
        (256, 256),
        (128, 128),
        (64, 64),
    ]
    
    def forward_sam2(batch, training=True):
        """Run SAM 2 forward pass with prompts.
        
        Follows the exact same feature extraction as SAM2ImagePredictor.set_image()
        and SAM2ImagePredictor._predict().
        """
        images = batch["image"].to(device)  # (B, 3, 1024, 1024)
        gt_masks = batch["mask"].to(device)  # (B, 1, 1024, 1024)
        points = batch["points"].to(device)  # (B, N, 2)
        point_labels = batch["point_labels"].to(device)  # (B, N)
        
        B = images.shape[0]
        
        all_pred_masks = []
        all_pred_ious = []
        
        for i in range(B):
            # ---- Step 1: Encode image (same as SAM2ImagePredictor.set_image) ----
            with torch.set_grad_enabled(
                training and (not freeze_encoder or encoder_unfrozen)
            ):
                backbone_out = sam2_model.forward_image(images[i:i+1])
                _, vision_feats, _, _ = sam2_model._prepare_backbone_features(
                    backbone_out
                )
                # Add no_mem_embed to lowest-res feature map
                if sam2_model.directly_add_no_mem_embed:
                    vision_feats[-1] = vision_feats[-1] + sam2_model.no_mem_embed
                
                # Reshape (HW, B, C) -> (B, C, H, W) for each feature level
                feats = [
                    feat.permute(1, 2, 0).view(1, -1, *feat_size)
                    for feat, feat_size in zip(
                        vision_feats[::-1], _bb_feat_sizes[::-1]
                    )
                ][::-1]
                
                # feats[0] = (1, 32, 256, 256) high-res
                # feats[1] = (1, 64, 128, 128) mid-res
                # feats[2] = (1, 256,  64,  64) image embedding
                image_embed = feats[-1]           # (1, 256, 64, 64)
                high_res_features = feats[:-1]    # [(1, 32, 256, 256), (1, 64, 128, 128)]
            
            # ---- Step 2: Build prompts (same as SAM2ImagePredictor._predict) ----
            point_coords_i = points[i:i+1]        # (1, N, 2)
            point_labs_i = point_labels[i:i+1]     # (1, N)
            
            # Merge box prompt as 2 corner points with labels [2, 3]
            if "box" in batch:
                box = batch["box"][i:i+1].to(device)  # (1, 4) xyxy
                box_coords = box.reshape(-1, 2, 2)     # (1, 2, 2)
                box_labs = torch.tensor([[2, 3]], dtype=torch.int32, device=device)
                # Boxes go first, then points (SAM convention)
                point_coords_i = torch.cat([box_coords, point_coords_i], dim=1)
                point_labs_i = torch.cat([box_labs, point_labs_i], dim=1)
            
            # Run prompt encoder
            sparse_embeddings, dense_embeddings = sam2_model.sam_prompt_encoder(
                points=(point_coords_i, point_labs_i),
                boxes=None,
                masks=None,
            )
            
            # ---- Step 3: Run mask decoder ----
            low_res_masks, iou_predictions, _, _ = sam2_model.sam_mask_decoder(
                image_embeddings=image_embed,
                image_pe=sam2_model.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
                repeat_image=False,
                high_res_features=high_res_features,
            )
            
            # Upscale masks to input size (1024x1024)
            pred_mask = F.interpolate(
                low_res_masks.float(), (1024, 1024),
                mode="bilinear", align_corners=False,
            )
            
            all_pred_masks.append(pred_mask)
            all_pred_ious.append(iou_predictions)
        
        pred_masks = torch.cat(all_pred_masks, dim=0)  # (B, 1, 1024, 1024)
        pred_ious = torch.cat(all_pred_ious, dim=0)    # (B, 1)
        
        return pred_masks, pred_ious, gt_masks
    
    # ---- Training Loop ----
    save_dir = cfg["checkpoint"]["save_dir"]
    save_every = cfg["checkpoint"].get("save_every", 5)
    unfreeze_epoch = cfg["training"].get("unfreeze_encoder_epoch", 999)
    grad_clip = cfg["training"].get("grad_clip", 1.0)
    encoder_unfrozen = False
    
    import time
    
    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        
        # --- Unfreeze encoder after specified epoch ---
        if epoch >= unfreeze_epoch and not encoder_unfrozen and freeze_encoder:
            for param in sam2_model.image_encoder.parameters():
                param.requires_grad = True
            encoder_unfrozen = True
            
            # Add encoder params to optimizer with lower LR
            enc_lr = cfg["training"].get("encoder_unfreeze_lr", 5e-6)
            optimizer.add_param_group({
                "params": list(sam2_model.image_encoder.parameters()),
                "lr": enc_lr,
            })
            
            new_trainable = sum(p.numel() for p in sam2_model.parameters() if p.requires_grad) / 1e6
            logger.info(f"Epoch {epoch+1}: Image encoder UNFROZEN (lr={enc_lr})")
            logger.info(f"Trainable params now: {new_trainable:.1f}M")
        
        # ---- Train ----
        sam2_model.train()
        # Keep frozen parts in eval
        if freeze_encoder and not encoder_unfrozen:
            sam2_model.image_encoder.eval()
        if freeze_prompt:
            sam2_model.sam_prompt_encoder.eval()
        
        train_loss = 0.0
        train_iou = 0.0
        n_train = 0
        
        for batch in tqdm(train_loader, desc=f"  Train E{epoch+1}", leave=False):
            pred_masks, pred_ious, gt_masks = forward_sam2(batch, training=True)
            
            loss, loss_dict = criterion(pred_masks, pred_ious, gt_masks)
            
            optimizer.zero_grad()
            loss.backward()
            
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    sam2_model.sam_mask_decoder.parameters(), grad_clip)
            
            optimizer.step()
            
            metrics = compute_metrics(pred_masks.detach(), gt_masks)
            train_loss += loss_dict["total"]
            train_iou += metrics["iou"]
            n_train += 1
        
        train_loss /= max(n_train, 1)
        train_iou /= max(n_train, 1)
        
        # ---- Validate ----
        sam2_model.eval()
        val_loss = 0.0
        val_iou = 0.0
        val_dice = 0.0
        val_acc = 0.0
        n_val = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"  Val   E{epoch+1}", leave=False):
                pred_masks, pred_ious, gt_masks = forward_sam2(batch, training=False)
                loss, loss_dict = criterion(pred_masks, pred_ious, gt_masks)
                metrics = compute_metrics(pred_masks, gt_masks)
                
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
            "mask_decoder": sam2_model.sam_mask_decoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_iou": best_iou,
            "config": cfg,
        }
        
        # Save latest
        save_checkpoint(state, save_dir, "sam2_checkpoint.pth", is_best=is_best)
        
        # Save every N epochs
        if (epoch + 1) % save_every == 0:
            fname = f"sam2_checkpoint_epoch{epoch+1}.pth"
            save_checkpoint(state, save_dir, fname, is_best=False)
            logger.info(f"  Saved checkpoint: {fname}")
        
        if is_best:
            # Save just the mask decoder for easy loading
            torch.save(
                sam2_model.sam_mask_decoder.state_dict(),
                os.path.join(save_dir, "sam2_mask_decoder_best.pth"),
            )
        
        # Early stopping
        if patience_counter >= es_patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
    
    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info(f"Best validation IoU: {best_iou:.4f}")
    logger.info(f"Mask decoder saved to: {save_dir}/sam2_mask_decoder_best.pth")
    logger.info("=" * 60)
    writer.close()


if __name__ == "__main__":
    main()
