"""
predict_sam2.py — Inference: generate segmentation masks using fine-tuned SAM 2.

Loads the fine-tuned SAM 2 model and processes all unprocessed sequences,
saving predicted masks and overlay visualizations.

Usage:
    python predict_sam2.py --config configs/train.yaml
    python predict_sam2.py --config configs/train.yaml --input ../unprocessed --output predictions
"""

import os
import argparse
import glob

import yaml
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


def setup_logger(name, log_dir=None):
    """Simple logger setup."""
    import logging
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


def load_sam2_model(cfg, device):
    """Load SAM 2 model with fine-tuned mask decoder."""
    from sam2.build_sam import build_sam2
    
    model_cfg = cfg["model"]["model_cfg"]
    base_checkpoint = cfg["model"]["checkpoint"]
    
    # Build base model
    sam2_model = build_sam2(model_cfg, base_checkpoint, device=device)
    
    # Load fine-tuned mask decoder
    decoder_path = os.path.join(cfg["checkpoint"]["save_dir"], "sam2_mask_decoder_best.pth")
    if os.path.exists(decoder_path):
        decoder_state = torch.load(decoder_path, map_location=device)
        sam2_model.sam_mask_decoder.load_state_dict(decoder_state)
        print(f"Loaded fine-tuned mask decoder from: {decoder_path}")
    else:
        print(f"WARNING: Fine-tuned decoder not found at {decoder_path}")
        print("Using base SAM 2 model (not fine-tuned)")
    
    sam2_model.eval()
    return sam2_model


def predict_with_auto_prompt(sam2_model, image_np, device, img_size=1024):
    """Predict segmentation using automatic center-grid prompts.
    
    Uses a grid of point prompts to get comprehensive coverage,
    then merges the results.
    """
    h, w = image_np.shape[:2]
    
    # Resize to SAM input size
    image_resized = np.array(
        Image.fromarray(image_np).resize((img_size, img_size), Image.BILINEAR)
    )
    
    # Convert to tensor
    image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float() / 255.0
    image_tensor = image_tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        # Encode image
        backbone_out = sam2_model.forward_image(image_tensor)
        _, vision_feats, _, _ = sam2_model._prepare_backbone_features(backbone_out)
        
        if sam2_model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + sam2_model.no_mem_embed
        
        feats = [
            feat.permute(1, 2, 0).view(1, -1, *feat_size)
            for feat, feat_size in zip(
                vision_feats[::-1],
                sam2_model._bb_feat_sizes[::-1],
            )
        ][::-1]
        
        high_res_features = [feats[-2], feats[-1]]
        
        # Use grid of points as prompts
        grid_points = []
        grid_labels = []
        
        # 3x3 grid of foreground points (center-biased)
        for gy in [0.3, 0.5, 0.7]:
            for gx in [0.3, 0.5, 0.7]:
                grid_points.append([gx * img_size, gy * img_size])
                grid_labels.append(1)
        
        # 4 corner background points
        margin = 0.05
        for cy, cx in [(margin, margin), (margin, 1-margin),
                       (1-margin, margin), (1-margin, 1-margin)]:
            grid_points.append([cx * img_size, cy * img_size])
            grid_labels.append(0)
        
        points = torch.tensor([grid_points], dtype=torch.float32, device=device)
        labels = torch.tensor([grid_labels], dtype=torch.int64, device=device)
        
        # Encode prompts
        sparse_emb, dense_emb = sam2_model.sam_prompt_encoder(
            points=(points, labels),
            boxes=None,
            masks=None,
        )
        
        # Decode mask
        low_res_masks, iou_pred, _, _ = sam2_model.sam_mask_decoder(
            image_embeddings=feats[0],
            image_pe=sam2_model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=False,
            repeat_image=False,
            high_res_features=high_res_features,
        )
        
        # Upscale to original size
        mask = F.interpolate(
            low_res_masks, (h, w),
            mode="bilinear", align_corners=False,
        )
        
        mask = (torch.sigmoid(mask) > 0.5).squeeze().cpu().numpy().astype(np.uint8) * 255
    
    return mask


def predict_with_sam2_predictor(image_np, predictor, device):
    """Alternative: use the SAM2ImagePredictor API for cleaner inference."""
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    
    h, w = image_np.shape[:2]
    
    predictor.set_image(image_np)
    
    # Use center point as prompt
    cx, cy = w // 2, h // 2
    point_coords = np.array([[cx, cy]], dtype=np.float32)
    point_labels = np.array([1], dtype=np.int64)
    
    masks, scores, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=False,
    )
    
    mask = (masks[0] * 255).astype(np.uint8)
    return mask


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


def main():
    parser = argparse.ArgumentParser(description="SAM 2 Prediction")
    parser.add_argument("--config", type=str, default="configs/train.yaml")
    parser.add_argument("--input", type=str, default=None,
                        help="Input directory (default: from config)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (default: save inside input)")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--no_overlay", action="store_true")
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    
    device = torch.device(cfg.get("device", "cuda"))
    logger = setup_logger("sam2_predict")
    
    input_dir = args.input or cfg["data"]["unprocessed_dir"]
    
    # Check for fine-tuned decoder
    decoder_path = os.path.join(cfg["checkpoint"]["save_dir"], "sam2_mask_decoder_best.pth")
    if not os.path.exists(decoder_path):
        logger.error(f"Fine-tuned mask decoder not found: {decoder_path}")
        logger.error("Run train_sam2.py first!")
        return
    
    logger.info("=" * 60)
    logger.info("SAM 2 Segmentation Inference")
    logger.info("=" * 60)
    logger.info(f"Input: {input_dir}")
    
    # Load model
    sam2_model = load_sam2_model(cfg, device)
    logger.info("Model loaded successfully")
    
    img_size = cfg["data"]["crop_size"]
    
    # Find all sequences
    seq_dirs = sorted(glob.glob(os.path.join(input_dir, "SEQ_*")))
    if not seq_dirs:
        logger.error(f"No sequences found in {input_dir}")
        return
    
    total_processed = 0
    
    for seq_dir in seq_dirs:
        seq_name = os.path.basename(seq_dir)
        
        # Find images
        img_dir = os.path.join(seq_dir, "images")
        if not os.path.isdir(img_dir):
            img_dir = os.path.join(seq_dir, "frames")
        if not os.path.isdir(img_dir):
            logger.warning(f"No images in {seq_dir}, skipping")
            continue
        
        image_paths = sorted(glob.glob(os.path.join(img_dir, "*.png")))
        if not image_paths:
            continue
        
        # Output dirs
        if args.output:
            mask_dir = os.path.join(args.output, seq_name, "masks")
            overlay_dir = os.path.join(args.output, seq_name, "overlays")
        else:
            mask_dir = os.path.join(seq_dir, "masks")
            overlay_dir = os.path.join(seq_dir, "overlays")
        
        os.makedirs(mask_dir, exist_ok=True)
        if not args.no_overlay:
            os.makedirs(overlay_dir, exist_ok=True)
        
        logger.info(f"Processing {seq_name}: {len(image_paths)} images")
        
        for img_path in tqdm(image_paths, desc=f"  {seq_name}", leave=True):
            fname = os.path.basename(img_path)
            image_np = np.array(Image.open(img_path).convert("RGB"))
            
            # Predict
            mask = predict_with_auto_prompt(
                sam2_model, image_np, device, img_size,
            )
            
            # Save mask
            Image.fromarray(mask, mode="L").save(os.path.join(mask_dir, fname))
            
            # Save overlay
            if not args.no_overlay:
                overlay = create_overlay(image_np, mask)
                Image.fromarray(overlay).save(os.path.join(overlay_dir, fname))
            
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
