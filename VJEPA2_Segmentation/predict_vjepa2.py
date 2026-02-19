"""
predict_vjepa2.py — Generate segmentation masks for unlabeled thermal frames.

Loads the fine-tuned V-JEPA 2 segmentation model and processes all
unprocessed sequences. Saves masks and overlays to a SEPARATE
predictions/ directory (does not overwrite I-JEPA or SAM 2 results).

Usage:
    python predict_vjepa2.py --config configs/train.yaml
    python predict_vjepa2.py --config configs/train.yaml --output predictions
"""

import os
import argparse
import glob
import logging

import yaml
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm


# ============================================================
#  Utilities
# ============================================================

def setup_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def create_overlay(image_np, mask, alpha=0.4, color=(0, 255, 0)):
    """Create an overlay visualization of mask on image."""
    overlay = image_np.copy()
    mask_bool = mask > 127
    
    for c in range(3):
        overlay[:, :, c] = np.where(
            mask_bool,
            (1 - alpha) * image_np[:, :, c] + alpha * color[c],
            image_np[:, :, c],
        )
    
    return overlay.astype(np.uint8)


# ============================================================
#  Model Loading
# ============================================================

def load_segmentation_model(cfg, decoder_weights_path, device):
    """Load V-JEPA 2 encoder + trained segmentation decoder."""
    from train_segmentation_vjepa2 import (
        VJEPA2SegmentationDecoder,
        VJEPA2SegmentationModel,
    )
    
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    
    # Load V-JEPA 2 encoder — hub returns (encoder, predictor) tuple
    hub_result = torch.hub.load(
        "facebookresearch/vjepa2", "vjepa2_vit_large",
        force_reload=False,
    )
    encoder = hub_result[0]  # VisionTransformer
    
    # Load pretrained encoder if available
    encoder_path = os.path.join(
        cfg["checkpoint"]["save_dir"], "vjepa2_encoder_final.pth"
    )
    if os.path.exists(encoder_path):
        encoder.load_state_dict(
            torch.load(encoder_path, map_location=device), strict=False,
        )
    
    encoder = encoder.to(device)
    encoder.eval()
    
    # Load decoder
    decoder = VJEPA2SegmentationDecoder(
        embed_dim=model_cfg.get("embed_dim", 1024),
        patch_size=model_cfg.get("patch_size", 16),
        img_size=data_cfg.get("crop_size", 224),
        decoder_channels=model_cfg.get("decoder_channels", [512, 256, 128, 64]),
        num_classes=model_cfg.get("num_classes", 1),
    ).to(device)
    
    # Load trained decoder weights
    state_dict = torch.load(decoder_weights_path, map_location=device)
    if isinstance(state_dict, dict) and "decoder" in state_dict:
        state_dict = state_dict["decoder"]
    decoder.load_state_dict(state_dict)
    decoder.eval()
    
    model = VJEPA2SegmentationModel(encoder, decoder)
    model.eval()
    
    return model


def predict_single(model, image_path, transform, device, original_size,
                   threshold=0.5):
    """Predict segmentation mask for a single image."""
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(img_tensor)  # (1, 1, H', W')
        probs = torch.sigmoid(logits)
    
    # Resize back to original dimensions
    probs = F.interpolate(
        probs, size=original_size, mode="bilinear", align_corners=False,
    )
    
    mask = (probs.squeeze().cpu().numpy() > threshold).astype(np.uint8) * 255
    return mask


# ============================================================
#  Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="V-JEPA 2 Segmentation Prediction")
    parser.add_argument("--config", type=str, default="configs/train.yaml")
    parser.add_argument("--weights", type=str, default=None,
                        help="Path to decoder weights")
    parser.add_argument("--input", type=str, default=None,
                        help="Input directory with unlabeled sequences")
    parser.add_argument("--output", type=str, default="predictions",
                        help="Output directory (default: predictions/)")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--save_overlay", action="store_true", default=True)
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    
    device = torch.device(cfg.get("device", "cuda"))
    logger = setup_logger("vjepa2_predict")
    
    # Resolve paths
    weights_path = args.weights or os.path.join(
        cfg["checkpoint"]["save_dir"], "vjepa2_decoder_best.pth",
    )
    input_dir = args.input or cfg["data"].get("unprocessed_dir", "../unprocessed")
    output_dir = args.output or cfg["data"].get("output_dir", "predictions")
    
    if not os.path.exists(weights_path):
        logger.error(f"Decoder weights not found: {weights_path}")
        logger.error("Run train_segmentation_vjepa2.py first!")
        return
    
    logger.info("=" * 60)
    logger.info("V-JEPA 2 Segmentation — Prediction")
    logger.info("=" * 60)
    logger.info(f"Weights: {weights_path}")
    logger.info(f"Input:   {input_dir}")
    logger.info(f"Output:  {output_dir}")
    
    # Load model
    model = load_segmentation_model(cfg, weights_path, device)
    logger.info("Model loaded successfully")
    
    # Transform
    crop_size = cfg["data"].get("crop_size", 224)
    transform = transforms.Compose([
        transforms.Resize((crop_size, crop_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    
    original_size = (240, 320)  # H, W
    
    # Find sequences
    seq_dirs = sorted(glob.glob(os.path.join(input_dir, "SEQ_*")))
    if not seq_dirs:
        logger.error(f"No sequences found in {input_dir}")
        return
    
    total_processed = 0
    
    for seq_dir in seq_dirs:
        seq_name = os.path.basename(seq_dir)
        
        img_dir = os.path.join(seq_dir, "images")
        if not os.path.isdir(img_dir):
            img_dir = os.path.join(seq_dir, "frames")
        if not os.path.isdir(img_dir):
            logger.warning(f"No images in {seq_dir}, skipping")
            continue
        
        image_paths = sorted(glob.glob(os.path.join(img_dir, "*.png")))
        if not image_paths:
            continue
        
        # Output dirs — ALWAYS save to predictions/ (separate from I-JEPA/SAM2)
        mask_dir = os.path.join(output_dir, seq_name, "masks")
        overlay_dir = os.path.join(output_dir, seq_name, "overlays")
        os.makedirs(mask_dir, exist_ok=True)
        if args.save_overlay:
            os.makedirs(overlay_dir, exist_ok=True)
        
        logger.info(f"Processing {seq_name}: {len(image_paths)} images")
        
        for img_path in tqdm(image_paths, desc=f"  {seq_name}", leave=True):
            fname = os.path.basename(img_path)
            
            mask = predict_single(
                model, img_path, transform, device,
                original_size, args.threshold,
            )
            
            # Save mask
            mask_img = Image.fromarray(mask, mode="L")
            mask_img.save(os.path.join(mask_dir, fname))
            
            # Save overlay
            if args.save_overlay:
                img_np = np.array(Image.open(img_path).convert("RGB"))
                overlay = create_overlay(img_np, mask)
                Image.fromarray(overlay).save(os.path.join(overlay_dir, fname))
            
            total_processed += 1
    
    logger.info("=" * 60)
    logger.info(f"Done! Processed {total_processed} images from {len(seq_dirs)} sequences")
    logger.info(f"Results saved to: {output_dir}/")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
