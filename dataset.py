"""
dataset.py — Data loading for I-JEPA pretraining and segmentation fine-tuning.

Two dataset classes:
  1. IJEPAPretrainDataset  — all frames (no labels) for self-supervised learning
  2. SegmentationDataset   — labeled image-mask pairs for supervised fine-tuning
"""

import os
import glob
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False


# ============================================================
#  Helper: discover all image paths
# ============================================================

def find_all_images(root_dir):
    """Find all .png images under a root directory.
    Handles both 'images/' and 'frames/' sub-folder naming.
    """
    patterns = [
        os.path.join(root_dir, "**", "images", "*.png"),
        os.path.join(root_dir, "**", "frames", "*.png"),
    ]
    paths = []
    for pat in patterns:
        paths.extend(glob.glob(pat, recursive=True))
    # Deduplicate and sort
    paths = sorted(set(paths))
    return paths


def find_labeled_pairs(processed_dir):
    """Find (image_path, mask_path) pairs from the processed directory.
    Matches images ↔ masks by filename within each sequence folder.
    """
    pairs = []
    seq_dirs = sorted(glob.glob(os.path.join(processed_dir, "SEQ_*")))

    for seq_dir in seq_dirs:
        # Find image folder (could be 'images/' or 'frames/')
        img_dir = os.path.join(seq_dir, "images")
        if not os.path.isdir(img_dir):
            img_dir = os.path.join(seq_dir, "frames")
        if not os.path.isdir(img_dir):
            continue

        mask_dir = os.path.join(seq_dir, "masks")
        if not os.path.isdir(mask_dir):
            continue

        # Match by filename
        mask_files = {os.path.basename(f): f
                      for f in glob.glob(os.path.join(mask_dir, "*.png"))}

        for img_path in sorted(glob.glob(os.path.join(img_dir, "*.png"))):
            fname = os.path.basename(img_path)
            if fname in mask_files:
                pairs.append((img_path, mask_files[fname]))

    return pairs


# ============================================================
#  Dataset 1: I-JEPA Pretraining (unlabeled)
# ============================================================

class IJEPAPretrainDataset(Dataset):
    """
    Loads ALL thermal frames (processed + unprocessed) for I-JEPA
    self-supervised pretraining.  Returns image tensors only.
    """

    def __init__(self, processed_dir, unprocessed_dir, crop_size=224):
        super().__init__()
        # Gather images from both directories
        self.image_paths = []
        if processed_dir and os.path.isdir(processed_dir):
            self.image_paths.extend(find_all_images(processed_dir))
        if unprocessed_dir and os.path.isdir(unprocessed_dir):
            self.image_paths.extend(find_all_images(unprocessed_dir))
        self.image_paths = sorted(set(self.image_paths))

        assert len(self.image_paths) > 0, (
            f"No images found in {processed_dir} or {unprocessed_dir}"
        )

        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(crop_size, scale=(0.4, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.2, 0.1),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        img = self.transform(img)
        return img


# ============================================================
#  Dataset 2: Segmentation (labeled)
# ============================================================

class SegmentationDataset(Dataset):
    """
    Loads labeled image-mask pairs from the processed directory
    for supervised segmentation training.
    """

    def __init__(self, processed_dir, crop_size=224, augment=True, cfg=None):
        super().__init__()
        self.pairs = find_labeled_pairs(processed_dir)
        assert len(self.pairs) > 0, f"No labeled pairs found in {processed_dir}"

        self.crop_size = crop_size
        self.augment = augment

        # Build augmentation pipeline
        if augment and HAS_ALBUMENTATIONS:
            aug_cfg = cfg.get("augmentation", {}) if cfg else {}
            self.transform = A.Compose([
                A.RandomResizedCrop(
                    height=crop_size, width=crop_size,
                    scale=(0.5, 1.0), ratio=(0.75, 1.33),
                ),
                A.HorizontalFlip(
                    p=0.5 if aug_cfg.get("horizontal_flip", True) else 0.0
                ),
                A.Rotate(
                    limit=aug_cfg.get("rotation_limit", 15),
                    p=0.5,
                    border_mode=0,
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=aug_cfg.get("brightness_limit", 0.2),
                    contrast_limit=aug_cfg.get("contrast_limit", 0.2),
                    p=0.5,
                ),
                A.GaussNoise(p=0.3 if aug_cfg.get("gaussian_noise", True) else 0.0),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ])
        else:
            # Simple transform without augmentation
            self.transform = None

        # Fallback torchvision transforms (used when albumentations unavailable)
        self.img_transform = transforms.Compose([
            transforms.Resize((crop_size, crop_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize(
                (crop_size, crop_size),
                interpolation=transforms.InterpolationMode.NEAREST,
            ),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform is not None and self.augment:
            # Albumentations expects numpy arrays
            img_np = np.array(img)
            mask_np = np.array(mask)
            # Binarize mask: 0 or 1
            mask_np = (mask_np > 127).astype(np.float32)

            transformed = self.transform(image=img_np, mask=mask_np)
            img_t = transformed["image"]        # (C, H, W) float
            mask_t = transformed["mask"]         # (H, W) float
            mask_t = mask_t.unsqueeze(0)         # (1, H, W)
        else:
            img_t = self.img_transform(img)
            mask_t = self.mask_transform(mask)
            mask_t = (mask_t > 0.5).float()      # binarize

        return img_t, mask_t


# ============================================================
#  Utility: create data loaders
# ============================================================

def get_pretrain_loader(processed_dir, unprocessed_dir, crop_size=224,
                        batch_size=64, num_workers=4):
    """Create DataLoader for I-JEPA pretraining."""
    dataset = IJEPAPretrainDataset(processed_dir, unprocessed_dir, crop_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    print(f"[Pretrain] {len(dataset)} images, {len(loader)} batches "
          f"(batch_size={batch_size})")
    return loader


def get_segmentation_loaders(processed_dir, crop_size=224, batch_size=16,
                             num_workers=4, train_split=0.85, cfg=None, seed=42):
    """Create train and validation DataLoaders for segmentation."""
    full_dataset = SegmentationDataset(processed_dir, crop_size, augment=True, cfg=cfg)

    # Split into train / val
    n_total = len(full_dataset)
    n_train = int(n_total * train_split)
    n_val = n_total - n_train

    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(full_dataset, [n_train, n_val],
                                       generator=generator)

    # Disable augmentation for val set
    val_dataset = SegmentationDataset(processed_dir, crop_size, augment=False)
    val_indices = val_set.indices
    val_set = torch.utils.data.Subset(val_dataset, val_indices)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"[Segmentation] train={n_train}, val={n_val} "
          f"(total={n_total}, split={train_split})")
    return train_loader, val_loader


# ============================================================
#  Quick test
# ============================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed", default="processed")
    parser.add_argument("--unprocessed", default="unprocessed")
    args = parser.parse_args()

    print("=" * 60)
    print("Testing IJEPAPretrainDataset")
    print("=" * 60)
    ds = IJEPAPretrainDataset(args.processed, args.unprocessed)
    print(f"  Total images: {len(ds)}")
    sample = ds[0]
    print(f"  Sample shape: {sample.shape}")
    print(f"  Sample range: [{sample.min():.3f}, {sample.max():.3f}]")

    print()
    print("=" * 60)
    print("Testing SegmentationDataset")
    print("=" * 60)
    ds2 = SegmentationDataset(args.processed, augment=False)
    print(f"  Total labeled pairs: {len(ds2)}")
    img, mask = ds2[0]
    print(f"  Image shape:  {img.shape}")
    print(f"  Mask shape:   {mask.shape}")
    print(f"  Mask unique:  {mask.unique().tolist()}")

    print()
    print("=" * 60)
    print("Testing DataLoaders")
    print("=" * 60)
    train_loader, val_loader = get_segmentation_loaders(
        args.processed, batch_size=4, num_workers=0
    )
    for imgs, masks in train_loader:
        print(f"  Train batch — images: {imgs.shape}, masks: {masks.shape}")
        break
    for imgs, masks in val_loader:
        print(f"  Val batch   — images: {imgs.shape}, masks: {masks.shape}")
        break

    print("\n✓ All dataset tests passed!")
