"""
predict.py — Inference script: generate segmentation masks for unlabeled frames.

Loads the trained segmentation model and processes all unprocessed sequences,
saving predicted masks and overlay visualizations.

Usage:
    python predict.py --config configs/segmentation.yaml
    python predict.py --weights checkpoints/segmentation_best.pth --input unprocessed --output predictions
"""

import os
import argparse
import glob

import yaml
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from models import SegmentationModel
from utils import setup_logger


def load_model(weights_path, cfg, device):
    """Load the trained segmentation model."""
    model = SegmentationModel(
        img_size=cfg.get("data", {}).get("crop_size", 224),
        patch_size=cfg["model"]["patch_size"],
        embed_dim=cfg["model"]["embed_dim"],
        depth=cfg["model"].get("depth", 12),
        num_heads=cfg["model"].get("num_heads", 6),
        num_classes=cfg["model"].get("num_classes", 1),
        decoder_channels=cfg["model"].get("decoder_channels", None),
    )

    state_dict = torch.load(weights_path, map_location=device)
    # Handle different checkpoint formats
    if "model" in state_dict:
        state_dict = state_dict["model"]
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    return model


def predict_single(model, image_path, transform, device, original_size,
                   threshold=0.5):
    """Predict segmentation mask for a single image."""
    img = Image.open(image_path).convert("RGB")

    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(img_tensor)  # (1, 1, H, W)
        probs = torch.sigmoid(logits)

    # Resize back to original dimensions
    probs = torch.nn.functional.interpolate(
        probs, size=original_size, mode="bilinear", align_corners=False,
    )

    mask = (probs.squeeze().cpu().numpy() > threshold).astype(np.uint8) * 255
    return mask


def create_overlay(image_path, mask, alpha=0.4, color=(0, 255, 0)):
    """Create an overlay visualization of mask on image."""
    img = np.array(Image.open(image_path).convert("RGB"))
    overlay = img.copy()

    # Apply color where mask is active
    mask_bool = mask > 127
    for c in range(3):
        overlay[:, :, c] = np.where(
            mask_bool,
            (1 - alpha) * img[:, :, c] + alpha * color[c],
            img[:, :, c],
        )

    return overlay.astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description="Predict segmentation masks")
    parser.add_argument("--config", type=str, default="configs/segmentation.yaml")
    parser.add_argument("--weights", type=str, default=None,
                        help="Path to trained model weights")
    parser.add_argument("--input", type=str, default="unprocessed",
                        help="Input directory with unlabeled sequences")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (default: save inside input sequences)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Binarization threshold")
    parser.add_argument("--save_overlay", action="store_true", default=True,
                        help="Save overlay visualizations")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg.get("device", "cuda"))
    logger = setup_logger("predict")

    # Resolve weights path
    weights_path = args.weights or os.path.join(
        cfg["checkpoint"]["save_dir"], "segmentation_best.pth"
    )
    if not os.path.exists(weights_path):
        logger.error(f"Model weights not found: {weights_path}")
        logger.error("Run train_segmentation.py first!")
        return

    logger.info("=" * 60)
    logger.info("Segmentation Inference")
    logger.info("=" * 60)
    logger.info(f"Model: {weights_path}")
    logger.info(f"Input: {args.input}")

    # Load model
    model = load_model(weights_path, cfg, device)
    logger.info("Model loaded successfully")

    # Image transform (no augmentation)
    crop_size = cfg["data"].get("crop_size", 224)
    transform = transforms.Compose([
        transforms.Resize((crop_size, crop_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    # Original image size for resizing masks back
    original_size = (240, 320)  # H, W

    # Find all sequences
    seq_dirs = sorted(glob.glob(os.path.join(args.input, "SEQ_*")))
    if not seq_dirs:
        logger.error(f"No sequences found in {args.input}")
        return

    total_processed = 0

    for seq_dir in seq_dirs:
        seq_name = os.path.basename(seq_dir)

        # Find images directory
        img_dir = os.path.join(seq_dir, "images")
        if not os.path.isdir(img_dir):
            img_dir = os.path.join(seq_dir, "frames")
        if not os.path.isdir(img_dir):
            logger.warning(f"No images found in {seq_dir}, skipping")
            continue

        image_paths = sorted(glob.glob(os.path.join(img_dir, "*.png")))
        if not image_paths:
            continue

        # Output directories
        if args.output:
            mask_dir = os.path.join(args.output, seq_name, "masks")
            overlay_dir = os.path.join(args.output, seq_name, "overlays")
        else:
            mask_dir = os.path.join(seq_dir, "masks")
            overlay_dir = os.path.join(seq_dir, "overlays")

        os.makedirs(mask_dir, exist_ok=True)
        if args.save_overlay:
            os.makedirs(overlay_dir, exist_ok=True)

        logger.info(f"Processing {seq_name}: {len(image_paths)} images")

        for img_path in tqdm(image_paths, desc=f"  {seq_name}", leave=True):
            fname = os.path.basename(img_path)

            # Predict mask
            mask = predict_single(
                model, img_path, transform, device,
                original_size, args.threshold,
            )

            # Save mask
            mask_img = Image.fromarray(mask, mode="L")
            mask_img.save(os.path.join(mask_dir, fname))

            # Save overlay
            if args.save_overlay:
                overlay = create_overlay(img_path, mask)
                overlay_img = Image.fromarray(overlay)
                overlay_img.save(os.path.join(overlay_dir, fname))

            total_processed += 1

    logger.info("=" * 60)
    logger.info(f"Done! Processed {total_processed} images from {len(seq_dirs)} sequences")
    if args.output:
        logger.info(f"Results saved to: {args.output}/")
    else:
        logger.info("Results saved alongside input images (masks/ and overlays/)")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
