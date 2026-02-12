"""
dataset_sam2.py — Data loading utilities for SAM 2 fine-tuning.

Handles loading labeled image-mask pairs from the processed/ folder,
resizing to 1024x1024 (SAM's expected input), and generating prompts
(points + bounding boxes) from ground truth masks.
"""

import os
import glob
import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_bounding_box(mask, noise_pixels=0):
    """Extract bounding box from a binary mask with optional noise."""
    ys, xs = np.where(mask > 0.5)
    if len(ys) == 0:
        # No foreground — return full-image box
        h, w = mask.shape
        return np.array([0, 0, w, h], dtype=np.float32)
    
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    
    # Add noise for robustness during training
    if noise_pixels > 0:
        h, w = mask.shape
        x_min = max(0, x_min - random.randint(0, noise_pixels))
        y_min = max(0, y_min - random.randint(0, noise_pixels))
        x_max = min(w, x_max + random.randint(0, noise_pixels))
        y_max = min(h, y_max + random.randint(0, noise_pixels))
    
    return np.array([x_min, y_min, x_max, y_max], dtype=np.float32)


def sample_points_from_mask(mask, num_points=3, num_negative=1):
    """Sample random points from foreground and background of a mask.
    
    Returns:
        points: (N, 2) array of (x, y) coordinates
        labels: (N,) array of labels (1=foreground, 0=background)
    """
    h, w = mask.shape
    fg_ys, fg_xs = np.where(mask > 0.5)
    bg_ys, bg_xs = np.where(mask <= 0.5)
    
    points = []
    labels = []
    
    # Sample foreground points
    if len(fg_ys) > 0:
        n_fg = min(num_points, len(fg_ys))
        indices = np.random.choice(len(fg_ys), n_fg, replace=False)
        for idx in indices:
            points.append([fg_xs[idx], fg_ys[idx]])
            labels.append(1)
    
    # Sample background points
    if len(bg_ys) > 0 and num_negative > 0:
        n_bg = min(num_negative, len(bg_ys))
        indices = np.random.choice(len(bg_ys), n_bg, replace=False)
        for idx in indices:
            points.append([bg_xs[idx], bg_ys[idx]])
            labels.append(0)
    
    if len(points) == 0:
        # Fallback: center point
        points = [[w // 2, h // 2]]
        labels = [1]
    
    return np.array(points, dtype=np.float32), np.array(labels, dtype=np.int64)


class SAM2SegmentationDataset(Dataset):
    """Dataset for SAM 2 fine-tuning with point and box prompts."""
    
    def __init__(self, image_paths, mask_paths, img_size=1024,
                 num_points=3, num_negative=1,
                 use_box=True, box_noise=10,
                 augment=True):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.img_size = img_size
        self.num_points = num_points
        self.num_negative = num_negative
        self.use_box = use_box
        self.box_noise = box_noise
        self.augment = augment
        
        # Augmentation for training
        if augment:
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.3),
                A.ShiftScaleRotate(
                    shift_limit=0.05, scale_limit=0.1,
                    rotate_limit=15, p=0.5,
                    border_mode=0,
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0.2, p=0.5,
                ),
                A.GaussNoise(p=0.2),
            ])
        else:
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
            ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image and mask
        image = np.array(Image.open(self.image_paths[idx]).convert("RGB"))
        mask = np.array(Image.open(self.mask_paths[idx]).convert("L"))
        
        # Binarize mask
        mask = (mask > 127).astype(np.float32)
        
        # Apply augmentation
        transformed = self.transform(image=image, mask=mask)
        image = transformed["image"]
        mask = transformed["mask"]
        
        # Generate prompts from the (augmented) mask
        points, point_labels = sample_points_from_mask(
            mask, self.num_points, self.num_negative,
        )
        
        box = None
        if self.use_box:
            box = get_bounding_box(mask, self.box_noise if self.augment else 0)
        
        # Normalize image to [0, 1] and convert to tensor
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(mask).unsqueeze(0).float()  # (1, H, W)
        points = torch.from_numpy(points)
        point_labels = torch.from_numpy(point_labels)
        
        result = {
            "image": image,                # (3, 1024, 1024)
            "mask": mask,                  # (1, 1024, 1024)
            "points": points,              # (N, 2) in (x, y) format
            "point_labels": point_labels,  # (N,)
            "original_size": torch.tensor([self.img_size, self.img_size]),
        }
        
        if box is not None:
            result["box"] = torch.from_numpy(box)  # (4,) in xyxy format
        
        return result


def collect_labeled_pairs(processed_dir):
    """Collect all labeled image-mask pairs from the processed directory."""
    pairs = []
    
    seq_dirs = sorted(glob.glob(os.path.join(processed_dir, "SEQ_*")))
    
    for seq_dir in seq_dirs:
        # Find images directory
        img_dir = os.path.join(seq_dir, "images")
        if not os.path.isdir(img_dir):
            img_dir = os.path.join(seq_dir, "frames")
        if not os.path.isdir(img_dir):
            continue
        
        mask_dir = os.path.join(seq_dir, "masks")
        if not os.path.isdir(mask_dir):
            continue
        
        # Match images to masks
        mask_files = {os.path.basename(f): f 
                      for f in glob.glob(os.path.join(mask_dir, "*.png"))}
        
        for img_path in sorted(glob.glob(os.path.join(img_dir, "*.png"))):
            fname = os.path.basename(img_path)
            if fname in mask_files:
                pairs.append((img_path, mask_files[fname]))
    
    return pairs


def get_sam2_dataloaders(processed_dir, img_size=1024, batch_size=4,
                         num_workers=4, train_split=0.85, cfg=None, seed=42):
    """Create train and validation dataloaders for SAM 2 fine-tuning."""
    
    pairs = collect_labeled_pairs(processed_dir)
    
    if len(pairs) == 0:
        raise ValueError(f"No labeled pairs found in {processed_dir}")
    
    # Split into train / val
    random.seed(seed)
    indices = list(range(len(pairs)))
    random.shuffle(indices)
    
    n_train = int(len(indices) * train_split)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    train_pairs = [pairs[i] for i in train_indices]
    val_pairs = [pairs[i] for i in val_indices]
    
    # Prompt settings from config
    num_points = 3
    num_negative = 1
    use_box = True
    box_noise = 10
    if cfg:
        num_points = cfg.get("training", {}).get("num_points_per_mask", 3)
        use_box = cfg.get("training", {}).get("use_box_prompt", True)
        box_noise = cfg.get("training", {}).get("box_noise_pixels", 10)
    
    train_dataset = SAM2SegmentationDataset(
        [p[0] for p in train_pairs],
        [p[1] for p in train_pairs],
        img_size=img_size,
        num_points=num_points,
        num_negative=num_negative,
        use_box=use_box,
        box_noise=box_noise,
        augment=True,
    )
    
    val_dataset = SAM2SegmentationDataset(
        [p[0] for p in val_pairs],
        [p[1] for p in val_pairs],
        img_size=img_size,
        num_points=num_points,
        num_negative=num_negative,
        use_box=use_box,
        box_noise=0,  # No noise for validation
        augment=False,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader


# ---- Self-test ----
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed", default="../processed")
    args = parser.parse_args()
    
    pairs = collect_labeled_pairs(args.processed)
    print(f"Found {len(pairs)} labeled image-mask pairs")
    
    if len(pairs) > 0:
        ds = SAM2SegmentationDataset(
            [p[0] for p in pairs[:5]],
            [p[1] for p in pairs[:5]],
            img_size=1024,
            augment=True,
        )
        sample = ds[0]
        print(f"Image shape: {sample['image'].shape}")
        print(f"Mask shape:  {sample['mask'].shape}")
        print(f"Points:      {sample['points'].shape}")
        print(f"Labels:      {sample['point_labels']}")
        if "box" in sample:
            print(f"Box:         {sample['box']}")
        print("✓ Dataset test passed!")
