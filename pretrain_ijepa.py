"""
pretrain_ijepa.py — I-JEPA self-supervised pretraining on thermal cow frames.

Learns domain-specific visual features from ALL frames (labeled + unlabeled)
without requiring any annotations.

Usage:
    python pretrain_ijepa.py --config configs/pretrain.yaml
    python pretrain_ijepa.py --config configs/pretrain.yaml --resume checkpoints/ijepa_checkpoint.pth
"""

import os
import argparse
import random

import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import get_pretrain_loader
from models import IJEPA, generate_masks
from utils import (
    setup_logger,
    save_checkpoint,
    load_checkpoint,
    cosine_scheduler,
    ema_scheduler,
    Timer,
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, optimizer, lr_schedule, ema_schedule,
                    epoch, cfg, device, logger, writer):
    """Train one epoch of I-JEPA pretraining."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    grid_size = model.context_encoder.patch_embed.grid_size
    num_patches = model.context_encoder.patch_embed.num_patches
    masking_cfg = cfg["masking"]

    pbar = tqdm(loader, desc=f"Epoch {epoch+1}", leave=True)
    for step, images in enumerate(pbar):
        global_step = epoch * len(loader) + step

        # Update learning rate
        lr = lr_schedule[min(global_step, len(lr_schedule) - 1)]
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        images = images.to(device, non_blocking=True)
        B = images.shape[0]

        # Generate masks
        context_mask, target_masks = generate_masks(
            B, num_patches, grid_size, masking_cfg
        )
        context_mask = context_mask.to(device)
        target_masks = [t.to(device) for t in target_masks]

        # Forward pass
        predictions, targets = model(images, context_mask, target_masks)

        # Compute loss: smooth L1 in representation space
        loss = 0.0
        for pred, tgt in zip(predictions, targets):
            loss += nn.functional.smooth_l1_loss(pred, tgt.detach())
        loss = loss / len(predictions)

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        clip_grad = cfg["training"].get("clip_grad", 3.0)
        if clip_grad > 0:
            nn.utils.clip_grad_norm_(
                list(model.context_encoder.parameters()) +
                list(model.predictor.parameters()),
                clip_grad,
            )

        optimizer.step()

        # EMA update of target encoder
        ema_m = ema_schedule[min(global_step, len(ema_schedule) - 1)]
        model.update_target_encoder(ema_m)

        total_loss += loss.item()
        num_batches += 1

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "lr": f"{lr:.2e}",
            "ema": f"{ema_m:.4f}",
        })

        # TensorBoard logging
        if global_step % 10 == 0:
            writer.add_scalar("pretrain/loss", loss.item(), global_step)
            writer.add_scalar("pretrain/lr", lr, global_step)
            writer.add_scalar("pretrain/ema_momentum", ema_m, global_step)

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss


def main():
    parser = argparse.ArgumentParser(description="I-JEPA Pretraining")
    parser.add_argument("--config", type=str, default="configs/pretrain.yaml",
                        help="Path to config file")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("seed", 42))
    device = torch.device(cfg.get("device", "cuda"))

    # Setup logging
    log_dir = cfg["checkpoint"]["log_dir"]
    logger = setup_logger("ijepa_pretrain", log_dir)
    writer = SummaryWriter(log_dir)

    logger.info("=" * 60)
    logger.info("I-JEPA Self-Supervised Pretraining")
    logger.info("=" * 60)

    # ---- Data ----
    loader = get_pretrain_loader(
        processed_dir=cfg["data"]["processed_dir"],
        unprocessed_dir=cfg["data"]["unprocessed_dir"],
        crop_size=cfg["data"].get("crop_size", 224),
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["data"].get("num_workers", 4),
    )
    logger.info(f"Data loaded: {len(loader.dataset)} images")

    # ---- Model ----
    model = IJEPA(cfg).to(device)
    n_params_ctx = sum(p.numel() for p in model.context_encoder.parameters()) / 1e6
    n_params_pred = sum(p.numel() for p in model.predictor.parameters()) / 1e6
    logger.info(f"Context encoder: {n_params_ctx:.1f}M params")
    logger.info(f"Predictor:       {n_params_pred:.1f}M params")

    # ---- Optimizer (only context encoder + predictor) ----
    optimizer = torch.optim.AdamW(
        list(model.context_encoder.parameters()) +
        list(model.predictor.parameters()),
        lr=cfg["training"]["base_lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    # ---- Schedules ----
    total_steps = cfg["training"]["epochs"] * len(loader)
    warmup_steps = cfg["training"]["warmup_epochs"] * len(loader)

    lr_schedule = cosine_scheduler(
        cfg["training"]["base_lr"],
        cfg["training"].get("min_lr", 1e-6),
        total_steps,
        warmup_steps,
    )

    ema_cfg = cfg["training"]["ema"]
    ema_schedule_vals = ema_scheduler(
        ema_cfg["start"], ema_cfg["end"], total_steps
    )

    # ---- Resume ----
    start_epoch = 0
    if args.resume:
        start_epoch, _ = load_checkpoint(
            args.resume, model=None, optimizer=optimizer, device=device
        )
        # Load model state manually
        ckpt = torch.load(args.resume, map_location=device)
        if "context_encoder" in ckpt:
            model.context_encoder.load_state_dict(ckpt["context_encoder"])
        if "target_encoder" in ckpt:
            model.target_encoder.load_state_dict(ckpt["target_encoder"])
        if "predictor" in ckpt:
            model.predictor.load_state_dict(ckpt["predictor"])
        logger.info(f"Resumed from epoch {start_epoch}")

    # ---- Training loop ----
    timer = Timer()
    save_dir = cfg["checkpoint"]["save_dir"]
    save_every = cfg["checkpoint"].get("save_every", 50)

    logger.info(f"Training for {cfg['training']['epochs']} epochs")
    logger.info(f"Batch size: {cfg['training']['batch_size']}")

    best_loss = float("inf")

    for epoch in range(start_epoch, cfg["training"]["epochs"]):
        timer.start()

        avg_loss = train_one_epoch(
            model, loader, optimizer, lr_schedule, ema_schedule_vals,
            epoch, cfg, device, logger, writer,
        )

        elapsed = timer.elapsed_str()
        logger.info(
            f"Epoch {epoch+1}/{cfg['training']['epochs']} — "
            f"loss: {avg_loss:.4f} — time: {elapsed}"
        )
        writer.add_scalar("pretrain/epoch_loss", avg_loss, epoch)

        # Save checkpoint
        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss

        if (epoch + 1) % save_every == 0 or is_best or epoch == 0:
            state = {
                "epoch": epoch + 1,
                "context_encoder": model.context_encoder.state_dict(),
                "target_encoder": model.target_encoder.state_dict(),
                "predictor": model.predictor.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_loss": best_loss,
                "config": cfg,
            }
            fname = f"ijepa_checkpoint_epoch{epoch+1}.pth"
            save_checkpoint(state, save_dir, fname, is_best=is_best)

            # Also save just the encoder for downstream use
            if is_best:
                torch.save(
                    model.context_encoder.state_dict(),
                    os.path.join(save_dir, "ijepa_encoder_best.pth"),
                )
                logger.info(f"  ★ New best loss: {best_loss:.4f} — saved encoder")

    # Final save
    torch.save(
        model.context_encoder.state_dict(),
        os.path.join(save_dir, "ijepa_encoder_final.pth"),
    )
    logger.info("=" * 60)
    logger.info("Pretraining complete!")
    logger.info(f"Best loss: {best_loss:.4f}")
    logger.info(f"Encoder saved to: {save_dir}/ijepa_encoder_best.pth")
    logger.info("=" * 60)
    writer.close()


if __name__ == "__main__":
    main()
