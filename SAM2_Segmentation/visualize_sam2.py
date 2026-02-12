"""
visualize_sam2.py — Visualization utilities for SAM 2 segmentation results.

Usage:
    python visualize_sam2.py --mode samples --processed ../processed --num 6
    python visualize_sam2.py --mode predictions --input ../unprocessed/SEQ_0495 --num 8
    python visualize_sam2.py --mode compare --processed ../processed --predictions ../unprocessed/SEQ_0495/masks --num 6
"""

import os
import glob
import argparse
import random

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def visualize_samples(processed_dir, num=6, save=None):
    """Visualize random labeled samples with masks."""
    pairs = []
    for seq_dir in sorted(glob.glob(os.path.join(processed_dir, "SEQ_*"))):
        img_dir = os.path.join(seq_dir, "images")
        if not os.path.isdir(img_dir):
            img_dir = os.path.join(seq_dir, "frames")
        if not os.path.isdir(img_dir):
            continue
        mask_dir = os.path.join(seq_dir, "masks")
        if not os.path.isdir(mask_dir):
            continue
        
        for mask_path in glob.glob(os.path.join(mask_dir, "*.png")):
            fname = os.path.basename(mask_path)
            img_path = os.path.join(img_dir, fname)
            if os.path.exists(img_path):
                pairs.append((img_path, mask_path))
    
    if not pairs:
        print("No labeled pairs found")
        return
    
    samples = random.sample(pairs, min(num, len(pairs)))
    
    fig, axes = plt.subplots(2, len(samples), figsize=(4 * len(samples), 8))
    if len(samples) == 1:
        axes = axes.reshape(2, 1)
    
    for i, (img_path, mask_path) in enumerate(samples):
        img = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))
        
        axes[0, i].imshow(img)
        axes[0, i].set_title(os.path.basename(img_path), fontsize=8)
        axes[0, i].axis("off")
        
        # Overlay
        overlay = img.copy()
        mask_bool = mask > 127
        overlay[mask_bool] = (0.6 * overlay[mask_bool] + 0.4 * np.array([0, 255, 0])).astype(np.uint8)
        axes[1, i].imshow(overlay)
        axes[1, i].set_title("GT Overlay", fontsize=8)
        axes[1, i].axis("off")
    
    plt.suptitle("Labeled Samples", fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    if save:
        plt.savefig(save, dpi=150, bbox_inches="tight")
        print(f"Saved to {save}")
    plt.show()


def visualize_predictions(input_dir, num=8, save=None):
    """Visualize prediction results (image + mask + overlay)."""
    img_dir = os.path.join(input_dir, "images")
    if not os.path.isdir(img_dir):
        img_dir = os.path.join(input_dir, "frames")
    mask_dir = os.path.join(input_dir, "masks")
    overlay_dir = os.path.join(input_dir, "overlays")
    
    if not os.path.isdir(mask_dir):
        print(f"No masks found in {input_dir}")
        return
    
    masks = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
    samples = random.sample(masks, min(num, len(masks)))
    
    n_rows = 3 if os.path.isdir(overlay_dir) else 2
    fig, axes = plt.subplots(n_rows, len(samples), figsize=(3 * len(samples), 3 * n_rows))
    if len(samples) == 1:
        axes = axes.reshape(n_rows, 1)
    
    for i, mask_path in enumerate(samples):
        fname = os.path.basename(mask_path)
        
        # Image
        img_path = os.path.join(img_dir, fname)
        if os.path.exists(img_path):
            img = np.array(Image.open(img_path).convert("RGB"))
            axes[0, i].imshow(img)
        axes[0, i].set_title(fname[:25], fontsize=7)
        axes[0, i].axis("off")
        
        # Mask
        mask = np.array(Image.open(mask_path).convert("L"))
        axes[1, i].imshow(mask, cmap="gray")
        axes[1, i].set_title("Predicted Mask", fontsize=8)
        axes[1, i].axis("off")
        
        # Overlay
        if n_rows == 3:
            ovl_path = os.path.join(overlay_dir, fname)
            if os.path.exists(ovl_path):
                ovl = np.array(Image.open(ovl_path).convert("RGB"))
                axes[2, i].imshow(ovl)
            axes[2, i].set_title("Overlay", fontsize=8)
            axes[2, i].axis("off")
    
    plt.suptitle("SAM 2 Predictions", fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    if save:
        plt.savefig(save, dpi=150, bbox_inches="tight")
        print(f"Saved to {save}")
    plt.show()


def compare_predictions(processed_dir, predictions_dir, num=6, save=None):
    """Compare ground truth vs predicted masks."""
    # Find GT pairs
    gt_pairs = {}
    for seq_dir in sorted(glob.glob(os.path.join(processed_dir, "SEQ_*"))):
        img_dir = os.path.join(seq_dir, "images")
        if not os.path.isdir(img_dir):
            img_dir = os.path.join(seq_dir, "frames")
        mask_dir = os.path.join(seq_dir, "masks")
        if not os.path.isdir(mask_dir):
            continue
        
        for mask_path in glob.glob(os.path.join(mask_dir, "*.png")):
            fname = os.path.basename(mask_path)
            img_path = os.path.join(img_dir, fname)
            if os.path.exists(img_path):
                gt_pairs[fname] = (img_path, mask_path)
    
    # Find predicted masks
    pred_masks = {os.path.basename(f): f 
                  for f in glob.glob(os.path.join(predictions_dir, "*.png"))}
    
    # Find overlapping
    common = set(gt_pairs.keys()) & set(pred_masks.keys())
    if not common:
        print("No overlapping images between GT and predictions")
        return
    
    samples = random.sample(sorted(common), min(num, len(common)))
    
    fig, axes = plt.subplots(3, len(samples), figsize=(3 * len(samples), 9))
    if len(samples) == 1:
        axes = axes.reshape(3, 1)
    
    for i, fname in enumerate(samples):
        img = np.array(Image.open(gt_pairs[fname][0]).convert("RGB"))
        gt_mask = np.array(Image.open(gt_pairs[fname][1]).convert("L"))
        pred_mask = np.array(Image.open(pred_masks[fname]).convert("L"))
        
        # Resize pred if different
        if pred_mask.shape != gt_mask.shape:
            pred_mask = np.array(
                Image.fromarray(pred_mask).resize(
                    (gt_mask.shape[1], gt_mask.shape[0]), Image.NEAREST)
            )
        
        axes[0, i].imshow(img)
        axes[0, i].set_title(fname[:25], fontsize=7)
        axes[0, i].axis("off")
        
        # GT overlay
        gt_overlay = img.copy()
        gt_overlay[gt_mask > 127] = (0.6 * gt_overlay[gt_mask > 127] + 0.4 * np.array([0, 255, 0])).astype(np.uint8)
        axes[1, i].imshow(gt_overlay)
        axes[1, i].set_title("Ground Truth", fontsize=8)
        axes[1, i].axis("off")
        
        # Pred overlay
        pred_overlay = img.copy()
        pred_overlay[pred_mask > 127] = (0.6 * pred_overlay[pred_mask > 127] + 0.4 * np.array([255, 0, 0])).astype(np.uint8)
        axes[2, i].imshow(pred_overlay)
        
        # Compute IoU
        gt_b = gt_mask > 127
        pred_b = pred_mask > 127
        inter = (gt_b & pred_b).sum()
        union = (gt_b | pred_b).sum()
        iou = inter / (union + 1e-6)
        axes[2, i].set_title(f"Pred (IoU: {iou:.3f})", fontsize=8)
        axes[2, i].axis("off")
    
    plt.suptitle("SAM 2: Ground Truth (green) vs Prediction (red)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    
    if save:
        plt.savefig(save, dpi=150, bbox_inches="tight")
        print(f"Saved to {save}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="SAM 2 Visualization")
    parser.add_argument("--mode", required=True,
                        choices=["samples", "predictions", "compare"])
    parser.add_argument("--processed", default="../processed")
    parser.add_argument("--input", default=None,
                        help="Input dir for predictions mode")
    parser.add_argument("--predictions", default=None,
                        help="Predictions mask dir for compare mode")
    parser.add_argument("--num", type=int, default=6)
    parser.add_argument("--save", type=str, default=None)
    args = parser.parse_args()
    
    if args.mode == "samples":
        visualize_samples(args.processed, args.num, args.save)
    elif args.mode == "predictions":
        if not args.input:
            print("Please specify --input directory")
            return
        visualize_predictions(args.input, args.num, args.save)
    elif args.mode == "compare":
        if not args.predictions:
            print("Please specify --predictions directory")
            return
        compare_predictions(args.processed, args.predictions, args.num, args.save)


if __name__ == "__main__":
    main()
