"""
visualize.py — Visualization utilities for inspection and debugging.

Features:
  - Overlay masks on images (side-by-side comparison)
  - Plot training curves from TensorBoard logs
  - Create comparison grids: ground truth vs predicted
  - Random sample visualization

Usage:
    python visualize.py --mode samples --processed processed --num 8
    python visualize.py --mode compare --processed processed --predictions predictions/SEQ_0495
    python visualize.py --mode overlay --image path/to/image.png --mask path/to/mask.png
"""

import os
import argparse
import glob
import random

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

from dataset import find_labeled_pairs


def overlay_mask_on_image(image, mask, alpha=0.4, color=(0, 255, 0)):
    """Create a colored overlay of mask on image."""
    if isinstance(image, str):
        image = np.array(Image.open(image).convert("RGB"))
    if isinstance(mask, str):
        mask = np.array(Image.open(mask).convert("L"))

    overlay = image.copy()
    mask_bool = mask > 127

    for c in range(3):
        overlay[:, :, c] = np.where(
            mask_bool,
            (1 - alpha) * image[:, :, c] + alpha * color[c],
            image[:, :, c],
        )

    return overlay.astype(np.uint8)


def visualize_samples(processed_dir, num_samples=8, save_path=None):
    """Show random labeled samples with their masks."""
    pairs = find_labeled_pairs(processed_dir)
    if len(pairs) == 0:
        print("No labeled pairs found!")
        return

    samples = random.sample(pairs, min(num_samples, len(pairs)))
    n = len(samples)
    cols = min(4, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols * 3, figsize=(cols * 9, rows * 3))
    if rows == 1:
        axes = axes.reshape(1, -1)

    for i, (img_path, mask_path) in enumerate(samples):
        r = i // cols
        c = (i % cols) * 3

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))
        overlay = overlay_mask_on_image(image, mask)

        axes[r, c].imshow(image)
        axes[r, c].set_title("Image", fontsize=8)
        axes[r, c].axis("off")

        axes[r, c + 1].imshow(mask, cmap="gray")
        axes[r, c + 1].set_title("Mask", fontsize=8)
        axes[r, c + 1].axis("off")

        axes[r, c + 2].imshow(overlay)
        axes[r, c + 2].set_title("Overlay", fontsize=8)
        axes[r, c + 2].axis("off")

    # Hide empty axes
    for r in range(rows):
        for c in range(cols * 3):
            if not axes[r, c].has_data():
                axes[r, c].axis("off")

    plt.suptitle(f"Labeled Samples ({n} shown)", fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    else:
        plt.show()


def visualize_predictions(input_dir, num_samples=8, save_path=None):
    """Show predictions: image + predicted mask + overlay."""
    img_dir = os.path.join(input_dir, "images")
    if not os.path.isdir(img_dir):
        img_dir = os.path.join(input_dir, "frames")

    mask_dir = os.path.join(input_dir, "masks")

    if not os.path.isdir(img_dir) or not os.path.isdir(mask_dir):
        print(f"Need both images/ and masks/ in {input_dir}")
        return

    image_files = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    samples = random.sample(image_files, min(num_samples, len(image_files)))

    n = len(samples)
    cols = min(4, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols * 3, figsize=(cols * 9, rows * 3))
    if rows == 1:
        axes = axes.reshape(1, -1)

    for i, img_path in enumerate(samples):
        r = i // cols
        c = (i % cols) * 3
        fname = os.path.basename(img_path)
        mask_path = os.path.join(mask_dir, fname)

        image = np.array(Image.open(img_path).convert("RGB"))

        if os.path.exists(mask_path):
            mask = np.array(Image.open(mask_path).convert("L"))
            overlay = overlay_mask_on_image(image, mask)
        else:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            overlay = image

        axes[r, c].imshow(image)
        axes[r, c].set_title("Image", fontsize=8)
        axes[r, c].axis("off")

        axes[r, c + 1].imshow(mask, cmap="gray")
        axes[r, c + 1].set_title("Predicted Mask", fontsize=8)
        axes[r, c + 1].axis("off")

        axes[r, c + 2].imshow(overlay)
        axes[r, c + 2].set_title("Overlay", fontsize=8)
        axes[r, c + 2].axis("off")

    plt.suptitle(f"Predictions ({n} shown)", fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    else:
        plt.show()


def visualize_single_overlay(image_path, mask_path, save_path=None):
    """Show a single image + mask + overlay."""
    image = np.array(Image.open(image_path).convert("RGB"))
    mask = np.array(Image.open(mask_path).convert("L"))
    overlay = overlay_mask_on_image(image, mask)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image)
    axes[0].set_title("Image")
    axes[0].axis("off")
    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Mask")
    axes[1].axis("off")
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    else:
        plt.show()


def compare_gt_vs_pred(processed_dir, predictions_dir, num_samples=6,
                        save_path=None):
    """Compare ground truth masks vs predicted masks for validation."""
    pairs = find_labeled_pairs(processed_dir)
    if not pairs:
        print("No labeled pairs found!")
        return

    samples = random.sample(pairs, min(num_samples, len(pairs)))
    n = len(samples)

    fig, axes = plt.subplots(n, 4, figsize=(16, n * 3))
    if n == 1:
        axes = axes.reshape(1, -1)

    for i, (img_path, gt_mask_path) in enumerate(samples):
        image = np.array(Image.open(img_path).convert("RGB"))
        gt_mask = np.array(Image.open(gt_mask_path).convert("L"))
        gt_overlay = overlay_mask_on_image(image, gt_mask, color=(0, 255, 0))

        # Try to find corresponding prediction
        fname = os.path.basename(img_path)
        pred_mask_path = os.path.join(predictions_dir, fname)

        axes[i, 0].imshow(image)
        axes[i, 0].set_title("Image", fontsize=9)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(gt_overlay)
        axes[i, 1].set_title("Ground Truth", fontsize=9)
        axes[i, 1].axis("off")

        if os.path.exists(pred_mask_path):
            pred_mask = np.array(Image.open(pred_mask_path).convert("L"))
            pred_overlay = overlay_mask_on_image(image, pred_mask, color=(255, 0, 0))

            axes[i, 2].imshow(pred_overlay)
            axes[i, 2].set_title("Predicted", fontsize=9)
            axes[i, 2].axis("off")

            # Difference map
            diff = np.abs(gt_mask.astype(float) - pred_mask.astype(float))
            axes[i, 3].imshow(diff, cmap="hot")
            axes[i, 3].set_title("Difference", fontsize=9)
            axes[i, 3].axis("off")
        else:
            axes[i, 2].text(0.5, 0.5, "No prediction", ha="center", va="center")
            axes[i, 2].axis("off")
            axes[i, 3].axis("off")

    plt.suptitle("Ground Truth vs Predicted", fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualization utilities")
    parser.add_argument("--mode", choices=["samples", "predictions", "overlay",
                                            "compare"],
                        required=True)
    parser.add_argument("--processed", default="processed")
    parser.add_argument("--predictions", default=None)
    parser.add_argument("--image", default=None)
    parser.add_argument("--mask", default=None)
    parser.add_argument("--num", type=int, default=8)
    parser.add_argument("--save", default=None, help="Save path for figure")
    args = parser.parse_args()

    if args.mode == "samples":
        visualize_samples(args.processed, args.num, args.save)
    elif args.mode == "predictions":
        if args.predictions is None:
            print("--predictions directory required!")
            return
        visualize_predictions(args.predictions, args.num, args.save)
    elif args.mode == "overlay":
        if args.image is None or args.mask is None:
            print("--image and --mask required!")
            return
        visualize_single_overlay(args.image, args.mask, args.save)
    elif args.mode == "compare":
        if args.predictions is None:
            print("--predictions directory required!")
            return
        compare_gt_vs_pred(args.processed, args.predictions, args.num, args.save)


if __name__ == "__main__":
    main()
