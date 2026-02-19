"""
pretrain_vjepa2.py — V-JEPA 2 self-supervised pretraining on thermal cow video clips.

Learns domain-specific spatio-temporal features from ALL thermal sequences
(labeled + unlabeled) without requiring any annotations.

Strategy:
  1. Load pretrained V-JEPA 2 ViT-Large as the context encoder
  2. Create EMA target encoder (exponential moving average copy)
  3. Mask random spatial patches (encoder handles temporal internally)
  4. Train predictor to predict target representations from context
  5. Loss: smooth L1 in latent space (no pixel reconstruction)

Usage:
    python pretrain_vjepa2.py --config configs/train.yaml
"""

import os
import sys
import argparse
import random
import math
import logging
import time
import copy

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset_vjepa2 import get_vjepa2_pretrain_loader


# ============================================================
#  Masking Utilities
# ============================================================

def generate_spatial_masks(batch_size, num_patches_h, num_patches_w,
                           num_targets=4, context_scale=(0.85, 1.0),
                           target_scale=(0.15, 0.2), device="cuda"):
    """Generate context and target masks for pretraining.
    
    Works on the SPATIAL patch grid that the encoder actually outputs.
    V-JEPA 2 handles temporal internally — we only mask spatial positions.
    
    Args:
        batch_size: B
        num_patches_h: spatial grid height (e.g., 14)
        num_patches_w: spatial grid width (e.g., 14)
        num_targets: number of target blocks
        context_scale: fraction of non-target patches to keep as context
        target_scale: fraction of patches per target block
        device: torch device
    
    Returns:
        context_indices: list of (N_ctx,) tensors per sample
        target_indices: list of (N_tgt,) tensors per sample
    """
    N = num_patches_h * num_patches_w  # 196
    
    context_indices_list = []
    target_indices_list = []
    
    for b in range(batch_size):
        is_target = torch.zeros(N, dtype=torch.bool, device=device)
        
        for _ in range(num_targets):
            tgt_frac = random.uniform(*target_scale)
            n_tgt = max(1, int(tgt_frac * N))
            
            block_h = max(1, int(math.sqrt(n_tgt)))
            block_w = max(1, n_tgt // block_h)
            
            start_h = random.randint(0, max(0, num_patches_h - block_h))
            start_w = random.randint(0, max(0, num_patches_w - block_w))
            
            for h in range(start_h, min(start_h + block_h, num_patches_h)):
                for w in range(start_w, min(start_w + block_w, num_patches_w)):
                    is_target[h * num_patches_w + w] = True
        
        # Target indices
        tgt_idx = torch.where(is_target)[0]
        if len(tgt_idx) == 0:
            tgt_idx = torch.tensor([0], device=device)
        
        # Context = everything that's NOT a target
        ctx_candidates = torch.where(~is_target)[0]
        
        # Optionally subsample context
        ctx_frac = random.uniform(*context_scale)
        n_ctx = max(1, int(ctx_frac * len(ctx_candidates)))
        if len(ctx_candidates) > n_ctx:
            perm = torch.randperm(len(ctx_candidates), device=device)[:n_ctx]
            ctx_idx = ctx_candidates[perm].sort()[0]
        else:
            ctx_idx = ctx_candidates
        
        context_indices_list.append(ctx_idx)
        target_indices_list.append(tgt_idx)
    
    return context_indices_list, target_indices_list


# ============================================================
#  Predictor Module
# ============================================================

class VJEPA2Predictor(nn.Module):
    """Lightweight transformer predictor for V-JEPA 2 pretraining.
    
    Predicts target patch representations from context patch representations
    and target position embeddings.
    """
    
    def __init__(self, num_patches, context_dim=1024, predictor_dim=384,
                 depth=6, num_heads=6):
        super().__init__()
        self.num_patches = num_patches
        self.predictor_dim = predictor_dim
        
        # Project context features to predictor dimension
        self.context_proj = nn.Linear(context_dim, predictor_dim)
        
        # Position embeddings for all spatial patches
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, predictor_dim)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # Mask token for target positions
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        
        # Transformer blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=predictor_dim,
            nhead=num_heads,
            dim_feedforward=predictor_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=depth,
        )
        
        # Project back to context dim
        self.output_proj = nn.Linear(predictor_dim, context_dim)
        self.norm = nn.LayerNorm(context_dim)
    
    def forward(self, context_features_list, context_indices_list, target_indices_list):
        """
        Args:
            context_features_list: list of (N_ctx, D) tensors per sample
            context_indices_list:  list of (N_ctx,) index tensors per sample
            target_indices_list:   list of (N_tgt,) index tensors per sample
        Returns:
            predictions: list of (N_tgt, D) predicted features for target patches
        """
        predictions = []
        
        for ctx_feat, ctx_idx, tgt_idx in zip(
            context_features_list, context_indices_list, target_indices_list
        ):
            N_ctx = ctx_feat.shape[0]
            N_tgt = tgt_idx.shape[0]
            
            # Clamp indices to valid range
            ctx_idx = ctx_idx.clamp(0, self.num_patches - 1)
            tgt_idx = tgt_idx.clamp(0, self.num_patches - 1)
            
            # Project context features and add positional embeddings
            ctx_proj = self.context_proj(ctx_feat)        # (N_ctx, pred_dim)
            ctx_proj = ctx_proj + self.pos_embed[0, ctx_idx]  # + pos embed
            
            # Create mask tokens with positional embeddings for targets
            tgt_tokens = self.mask_token.squeeze(0).expand(N_tgt, -1)    # (N_tgt, pred_dim)
            tgt_tokens = tgt_tokens + self.pos_embed[0, tgt_idx]  # + pos embed
            
            # Concatenate and process through transformer
            tokens = torch.cat([ctx_proj, tgt_tokens], dim=0).unsqueeze(0)  # (1, N_ctx+N_tgt, D')
            out = self.transformer(tokens)  # (1, N_ctx+N_tgt, D')
            
            # Extract target predictions
            tgt_preds = out[0, N_ctx:]  # (N_tgt, pred_dim)
            tgt_preds = self.norm(self.output_proj(tgt_preds))  # (N_tgt, D)
            
            predictions.append(tgt_preds)
        
        return predictions


# ============================================================
#  Utility functions
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
        fh = logging.FileHandler(os.path.join(log_dir, "pretrain.log"))
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger


def cosine_schedule(base_value, final_value, epochs, warmup_epochs=0):
    """Generate a cosine schedule array."""
    schedule = []
    for epoch in range(epochs):
        if epoch < warmup_epochs:
            val = base_value * (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
            val = final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * progress))
        schedule.append(val)
    return schedule


def ema_schedule(start, end, epochs):
    """Generate an EMA momentum schedule."""
    return [start + (end - start) * (epoch / max(1, epochs - 1)) for epoch in range(epochs)]


@torch.no_grad()
def update_ema(student, teacher, momentum):
    """Exponential moving average update."""
    for s_param, t_param in zip(student.parameters(), teacher.parameters()):
        t_param.data.mul_(momentum).add_(s_param.data, alpha=1.0 - momentum)


# ============================================================
#  Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="V-JEPA 2 Pretraining")
    parser.add_argument("--config", type=str, default="configs/train.yaml")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    
    set_seed(cfg.get("seed", 42))
    device = torch.device(cfg.get("device", "cuda"))
    
    pretrain_cfg = cfg["pretraining"]
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    ckpt_cfg = cfg["checkpoint"]
    
    log_dir = ckpt_cfg["log_dir_pretrain"]
    logger = setup_logger("vjepa2_pretrain", log_dir)
    writer = SummaryWriter(log_dir)
    
    logger.info("=" * 60)
    logger.info("V-JEPA 2 Self-Supervised Pretraining")
    logger.info("=" * 60)
    
    # ---- Data ----
    train_loader = get_vjepa2_pretrain_loader(
        processed_dir=data_cfg["processed_dir"],
        unprocessed_dir=data_cfg["unprocessed_dir"],
        clip_length=data_cfg.get("clip_length", 16),
        clip_stride=data_cfg.get("clip_stride", 4),
        crop_size=data_cfg.get("crop_size", 224),
        batch_size=pretrain_cfg["batch_size"],
        num_workers=data_cfg.get("num_workers", 4),
    )
    logger.info(f"Training clips: {len(train_loader.dataset)}")
    
    # ---- Load V-JEPA 2 Encoder ----
    logger.info("Loading V-JEPA 2 ViT-Large via PyTorch Hub...")
    
    hub_result = torch.hub.load(
        "facebookresearch/vjepa2", "vjepa2_vit_large",
        force_reload=False,
    )
    context_encoder = hub_result[0]  # VisionTransformer (303.9M params)
    context_encoder = context_encoder.to(device)
    
    # Create EMA target encoder (frozen copy)
    target_encoder = copy.deepcopy(context_encoder)
    for param in target_encoder.parameters():
        param.requires_grad = False
    target_encoder.eval()
    
    # Get model info
    embed_dim = model_cfg.get("embed_dim", 1024)
    crop_size = data_cfg.get("crop_size", 224)
    patch_size = model_cfg.get("patch_size", 16)
    
    num_patches_h = crop_size // patch_size  # 14
    num_patches_w = crop_size // patch_size  # 14
    num_patches = num_patches_h * num_patches_w  # 196
    
    total_params = sum(p.numel() for p in context_encoder.parameters()) / 1e6
    logger.info(f"V-JEPA 2 encoder params: {total_params:.1f}M")
    logger.info(f"Spatial patches: {num_patches_h}H × {num_patches_w}W = {num_patches}")
    
    # ---- Predictor ----
    predictor = VJEPA2Predictor(
        num_patches=num_patches,  # 196 spatial patches
        context_dim=embed_dim,
        predictor_dim=384,
        depth=6,
        num_heads=6,
    ).to(device)
    
    pred_params = sum(p.numel() for p in predictor.parameters()) / 1e6
    logger.info(f"Predictor params: {pred_params:.1f}M")
    
    # ---- Optimizer ----
    param_groups = [
        {"params": context_encoder.parameters(), "lr": pretrain_cfg["learning_rate"]},
        {"params": predictor.parameters(), "lr": pretrain_cfg["learning_rate"]},
    ]
    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=pretrain_cfg.get("weight_decay", 0.05),
    )
    
    # ---- Schedules ----
    epochs = pretrain_cfg["epochs"]
    warmup = pretrain_cfg.get("warmup_epochs", 10)
    
    lr_sched = cosine_schedule(
        pretrain_cfg["learning_rate"],
        pretrain_cfg.get("min_lr", 1e-6),
        epochs, warmup,
    )
    ema_sched = ema_schedule(
        pretrain_cfg["ema"]["start"],
        pretrain_cfg["ema"]["end"],
        epochs,
    )
    
    # ---- Resume ----
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        context_encoder.load_state_dict(ckpt["context_encoder"])
        target_encoder.load_state_dict(ckpt["target_encoder"])
        predictor.load_state_dict(ckpt["predictor"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"]
        logger.info(f"Resumed from epoch {start_epoch}")
    
    # ---- Masking config ----
    mask_cfg = pretrain_cfg.get("masking", {})
    num_targets = mask_cfg.get("num_targets", 4)
    target_scale = mask_cfg.get("target_scale", [0.15, 0.2])
    context_scale = mask_cfg.get("context_scale", [0.85, 1.0])
    
    # ---- Training Loop ----
    save_dir = ckpt_cfg["save_dir"]
    save_every = ckpt_cfg.get("save_every", 10)
    grad_clip = pretrain_cfg.get("clip_grad", 3.0)
    
    logger.info(f"Training for {epochs} epochs...")
    
    scaler = torch.amp.GradScaler("cuda")
    
    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        context_encoder.train()
        predictor.train()
        
        # Update learning rate
        lr = lr_sched[epoch]
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        
        epoch_loss = 0.0
        n_batches = 0
        
        for batch_idx, clips in enumerate(tqdm(
            train_loader, desc=f"  Pretrain E{epoch+1}", leave=False
        )):
            # clips shape: (B, T, C, H, W)
            clips = clips.to(device)
            B, T, C, H, W = clips.shape
            
            # Generate spatial masks (encoder outputs 196 spatial patches)
            ctx_indices_list, tgt_indices_list = generate_spatial_masks(
                B, num_patches_h, num_patches_w,
                num_targets=num_targets,
                context_scale=context_scale,
                target_scale=target_scale,
                device=device,
            )
            
            # V-JEPA 2 expects (B, C, T, H, W)
            video_input = clips.permute(0, 2, 1, 3, 4)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast("cuda"):
                # Forward through context encoder → (B, 196, 1024)
                all_features = context_encoder(video_input)
                
                # Extract context features per sample
                ctx_feats_list = []
                for b in range(B):
                    ctx_idx = ctx_indices_list[b]
                    ctx_feats_list.append(all_features[b, ctx_idx])
                
                # Forward through target encoder (no grad) → (B, 196, 1024)
                with torch.no_grad():
                    target_features = target_encoder(video_input)
                    
                    tgt_feats_list = []
                    for b in range(B):
                        tgt_idx = tgt_indices_list[b]
                        tgt_feats_list.append(target_features[b, tgt_idx])
                
                # Predictor: predict target features from context
                predictions = predictor(ctx_feats_list, ctx_indices_list, tgt_indices_list)
                
                # Loss: smooth L1 between predictions and target features
                loss = torch.tensor(0.0, device=device)
                for b in range(B):
                    pred = predictions[b]
                    tgt = tgt_feats_list[b]
                    n = min(pred.shape[0], tgt.shape[0])
                    loss = loss + F.smooth_l1_loss(pred[:n], tgt[:n])
                loss = loss / B
            
            scaler.scale(loss).backward()
            
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(context_encoder.parameters()) + list(predictor.parameters()),
                    grad_clip,
                )
            
            scaler.step(optimizer)
            scaler.update()
            
            # Update EMA target encoder
            update_ema(context_encoder, target_encoder, ema_sched[epoch])
            
            epoch_loss += loss.item()
            n_batches += 1
        
        # ---- Logging ----
        avg_loss = epoch_loss / max(n_batches, 1)
        elapsed = time.time() - t0
        ema_m = ema_sched[epoch]
        
        logger.info(
            f"Epoch {epoch+1:3d}/{epochs} | "
            f"Loss: {avg_loss:.4f} | "
            f"LR: {lr:.2e} | EMA: {ema_m:.4f} | "
            f"{elapsed:.1f}s"
        )
        
        writer.add_scalar("pretrain/loss", avg_loss, epoch)
        writer.add_scalar("pretrain/lr", lr, epoch)
        writer.add_scalar("pretrain/ema_momentum", ema_m, epoch)
        
        # ---- Save Checkpoint ----
        if (epoch + 1) % save_every == 0 or epoch == epochs - 1:
            state = {
                "epoch": epoch + 1,
                "context_encoder": context_encoder.state_dict(),
                "target_encoder": target_encoder.state_dict(),
                "predictor": predictor.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": cfg,
            }
            os.makedirs(save_dir, exist_ok=True)
            
            ckpt_path = os.path.join(save_dir, f"vjepa2_pretrain_epoch{epoch+1}.pth")
            torch.save(state, ckpt_path)
            logger.info(f"  Saved checkpoint: {ckpt_path}")
            
            # Also save latest
            latest_path = os.path.join(save_dir, "vjepa2_pretrained.pth")
            torch.save(state, latest_path)
    
    # Save final encoder state dict for segmentation
    encoder_path = os.path.join(save_dir, "vjepa2_encoder_final.pth")
    torch.save(context_encoder.state_dict(), encoder_path)
    
    logger.info("=" * 60)
    logger.info("Pretraining complete!")
    logger.info(f"Encoder saved to: {encoder_path}")
    logger.info("=" * 60)
    writer.close()


if __name__ == "__main__":
    main()

