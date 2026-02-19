"""
visualize_vjepa2.py — Visualization utilities for V-JEPA 2 segmentation results.

Generates comparison grids, side-by-side views, and cross-pipeline comparisons
between V-JEPA 2, I-JEPA, and SAM 2 predictions.

Usage:
    python visualize_vjepa2.py --predictions predictions/ --output visualizations/
    python visualize_vjepa2.py --compare  # Compare all three pipelines
"""

import os
import argparse
import glob
import random

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def create_overlay(image_np, mask, alpha=0.4, color=(0, 255, 0)):
    """Create an overlay visualization."""
    overlay = image_np.copy()
    mask_bool = mask > 127
    for c in range(3):
        overlay[:, :, c] = np.where(
            mask_bool,
            (1 - alpha) * image_np[:, :, c] + alpha * color[c],
            image_np[:, :, c],
        )
    return overlay.astype(np.uint8)


def visualize_predictions(predictions_dir, output_dir, num_samples=16):
    """Generate grid visualization of V-JEPA 2 predictions."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect all mask files
    mask_files = sorted(glob.glob(
        os.path.join(predictions_dir, "SEQ_*", "masks", "*.png")
    ))
    
    if not mask_files:
        print(f"No masks found in {predictions_dir}")
        return
    
    # Sample randomly
    samples = random.sample(mask_files, min(num_samples, len(mask_files)))
    
    cols = 4
    rows = (len(samples) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols * 3, figsize=(cols * 12, rows * 4))
    
    if rows == 1:
        axes = axes[np.newaxis, :]
    
    for i, mask_path in enumerate(samples):
        row = i // cols
        col_base = (i % cols) * 3
        
        # Load mask
        mask = np.array(Image.open(mask_path))
        
        # Find corresponding original image
        parts = mask_path.split(os.sep)
        seq_idx = next(j for j, p in enumerate(parts) if p.startswith("SEQ_"))
        seq_name = parts[seq_idx]
        fname = os.path.basename(mask_path)
        
        # Try to find original image
        img_path = os.path.join(
            os.path.dirname(predictions_dir), "unprocessed",
            seq_name, "images", fname,
        )
        if not os.path.exists(img_path):
            img_path = os.path.join(
                os.path.dirname(predictions_dir), "unprocessed",
                seq_name, "frames", fname,
            )
        
        if os.path.exists(img_path):
            img = np.array(Image.open(img_path).convert("RGB"))
            overlay = create_overlay(img, mask)
        else:
            img = np.zeros_like(mask)
            img = np.stack([img] * 3, axis=-1)
            overlay = img
        
        # Plot
        axes[row, col_base].imshow(img)
        axes[row, col_base].set_title(f"{seq_name}/{fname}", fontsize=8)
        axes[row, col_base].axis("off")
        
        axes[row, col_base + 1].imshow(mask, cmap="gray")
        axes[row, col_base + 1].set_title("V-JEPA 2 Mask", fontsize=8)
        axes[row, col_base + 1].axis("off")
        
        axes[row, col_base + 2].imshow(overlay)
        axes[row, col_base + 2].set_title("Overlay", fontsize=8)
        axes[row, col_base + 2].axis("off")
    
    # Hide unused axes
    for i in range(len(samples), rows * cols):
        row = i // cols
        col_base = (i % cols) * 3
        for j in range(3):
            if col_base + j < axes.shape[1]:
                axes[row, col_base + j].axis("off")
    
    plt.suptitle("V-JEPA 2 Segmentation Predictions", fontsize=16, fontweight="bold")
    plt.tight_layout()
    save_path = os.path.join(output_dir, "vjepa2_predictions_grid.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved prediction grid to {save_path}")


def compare_pipelines(
    ijepa_dir="../unprocessed",
    sam2_dir="../SAM2_Segmentation/predictions",
    vjepa2_dir="predictions",
    output_dir="visualizations",
    num_samples=8,
):
    """Compare predictions across I-JEPA, SAM 2, and V-JEPA 2 pipelines."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Find common sequences
    vjepa2_seqs = set()
    for d in glob.glob(os.path.join(vjepa2_dir, "SEQ_*")):
        vjepa2_seqs.add(os.path.basename(d))
    
    if not vjepa2_seqs:
        print("No V-JEPA 2 predictions found")
        return
    
    # Collect sample frames
    all_samples = []
    for seq_name in sorted(vjepa2_seqs):
        vjepa2_masks = sorted(glob.glob(
            os.path.join(vjepa2_dir, seq_name, "masks", "*.png")
        ))
        for mask_path in vjepa2_masks[:3]:  # max 3 per sequence
            fname = os.path.basename(mask_path)
            all_samples.append((seq_name, fname))
    
    samples = random.sample(all_samples, min(num_samples, len(all_samples)))
    
    fig, axes = plt.subplots(len(samples), 5, figsize=(25, len(samples) * 4))
    if len(samples) == 1:
        axes = axes[np.newaxis, :]
    
    for i, (seq_name, fname) in enumerate(samples):
        # Original image
        img_path = os.path.join(ijepa_dir, seq_name, "images", fname)
        if not os.path.exists(img_path):
            img_path = os.path.join(ijepa_dir, seq_name, "frames", fname)
        
        if os.path.exists(img_path):
            img = np.array(Image.open(img_path).convert("RGB"))
        else:
            img = np.zeros((240, 320, 3), dtype=np.uint8)
        
        # Masks from each pipeline
        ijepa_mask_path = os.path.join(ijepa_dir, seq_name, "masks", fname)
        sam2_mask_path = os.path.join(sam2_dir, seq_name, "masks", fname)
        vjepa2_mask_path = os.path.join(vjepa2_dir, seq_name, "masks", fname)
        
        # Plot original
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"Original\n{seq_name}/{fname}", fontsize=9)
        axes[i, 0].axis("off")
        
        # I-JEPA
        if os.path.exists(ijepa_mask_path):
            mask = np.array(Image.open(ijepa_mask_path))
            overlay = create_overlay(img, mask, color=(255, 0, 0))  # Red
            axes[i, 1].imshow(overlay)
            axes[i, 1].set_title("I-JEPA", fontsize=9)
        else:
            axes[i, 1].text(0.5, 0.5, "N/A", ha="center", va="center")
            axes[i, 1].set_title("I-JEPA", fontsize=9)
        axes[i, 1].axis("off")
        
        # SAM 2
        if os.path.exists(sam2_mask_path):
            mask = np.array(Image.open(sam2_mask_path))
            overlay = create_overlay(img, mask, color=(0, 0, 255))  # Blue
            axes[i, 2].imshow(overlay)
            axes[i, 2].set_title("SAM 2.1", fontsize=9)
        else:
            axes[i, 2].text(0.5, 0.5, "N/A", ha="center", va="center")
            axes[i, 2].set_title("SAM 2.1", fontsize=9)
        axes[i, 2].axis("off")
        
        # V-JEPA 2
        if os.path.exists(vjepa2_mask_path):
            mask = np.array(Image.open(vjepa2_mask_path))
            overlay = create_overlay(img, mask, color=(0, 255, 0))  # Green
            axes[i, 3].imshow(overlay)
            axes[i, 3].set_title("V-JEPA 2", fontsize=9)
        else:
            axes[i, 3].text(0.5, 0.5, "N/A", ha="center", va="center")
            axes[i, 3].set_title("V-JEPA 2", fontsize=9)
        axes[i, 3].axis("off")
        
        # All overlays combined
        combined = img.copy()
        for path, clr in [
            (ijepa_mask_path, (255, 0, 0)),
            (sam2_mask_path, (0, 0, 255)),
            (vjepa2_mask_path, (0, 255, 0)),
        ]:
            if os.path.exists(path):
                m = np.array(Image.open(path))
                combined = create_overlay(combined, m, alpha=0.2, color=clr)
        
        axes[i, 4].imshow(combined)
        axes[i, 4].set_title("Combined\n(R=IJEPA, B=SAM2, G=VJEPA2)", fontsize=8)
        axes[i, 4].axis("off")
    
    plt.suptitle(
        "Pipeline Comparison: I-JEPA vs SAM 2.1 vs V-JEPA 2",
        fontsize=16, fontweight="bold",
    )
    plt.tight_layout()
    save_path = os.path.join(output_dir, "pipeline_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved comparison to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="V-JEPA 2 Visualization")
    parser.add_argument("--predictions", type=str, default="predictions",
                        help="V-JEPA 2 predictions directory")
    parser.add_argument("--output", type=str, default="visualizations",
                        help="Output directory for visualizations")
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--compare", action="store_true",
                        help="Compare all three pipelines")
    parser.add_argument("--ijepa_dir", type=str, default="../unprocessed")
    parser.add_argument("--sam2_dir", type=str, default="../SAM2_Segmentation/predictions")
    args = parser.parse_args()
    
    # Visualize V-JEPA 2 predictions
    visualize_predictions(
        args.predictions, args.output, args.num_samples,
    )
    
    # Compare pipelines if requested
    if args.compare:
        compare_pipelines(
            ijepa_dir=args.ijepa_dir,
            sam2_dir=args.sam2_dir,
            vjepa2_dir=args.predictions,
            output_dir=args.output,
            num_samples=min(8, args.num_samples),
        )
    
    print("Done!")


if __name__ == "__main__":
    main()
