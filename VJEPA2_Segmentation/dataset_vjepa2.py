"""
dataset_vjepa2.py — Data loading for V-JEPA 2 pretraining and segmentation.

Two dataset classes:
  1. VJEPA2PretrainDataset    — video clips (consecutive frames) for self-supervised learning
  2. VJEPA2SegmentationDataset — labeled image-mask pairs for supervised segmentation

Usage:
    from dataset_vjepa2 import get_vjepa2_pretrain_loader, get_vjepa2_seg_loaders
"""

import os
import glob
import random

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
#  Helper: discover sequences and frames
# ============================================================

def find_all_sequences(root_dir):
    """Find all SEQ_* directories and their image paths."""
    sequences = {}
    seq_dirs = sorted(glob.glob(os.path.join(root_dir, "SEQ_*")))
    
    for seq_dir in seq_dirs:
        seq_name = os.path.basename(seq_dir)
        img_dir = os.path.join(seq_dir, "images")
        if not os.path.isdir(img_dir):
            img_dir = os.path.join(seq_dir, "frames")
        if not os.path.isdir(img_dir):
            continue
        
        images = sorted(glob.glob(os.path.join(img_dir, "*.png")))
        if images:
            sequences[seq_name] = images
    
    return sequences


def find_labeled_pairs(processed_dir):
    """Find (image_path, mask_path) pairs from the processed directory."""
    pairs = []
    seq_dirs = sorted(glob.glob(os.path.join(processed_dir, "SEQ_*")))
    
    for seq_dir in seq_dirs:
        img_dir = os.path.join(seq_dir, "images")
        mask_dir = os.path.join(seq_dir, "masks")
        
        if not os.path.isdir(img_dir) or not os.path.isdir(mask_dir):
            continue
        
        for img_path in sorted(glob.glob(os.path.join(img_dir, "*.png"))):
            fname = os.path.basename(img_path)
            mask_path = os.path.join(mask_dir, fname)
            if os.path.exists(mask_path):
                pairs.append((img_path, mask_path))
    
    return pairs


# ============================================================
#  Dataset 1: V-JEPA 2 Pretraining (unlabeled video clips)
# ============================================================

class VJEPA2PretrainDataset(Dataset):
    """Loads video clips (T consecutive frames) from all thermal sequences
    for V-JEPA 2 self-supervised pretraining.
    
    Each clip is T consecutive frames from a single sequence.
    Returns a tensor of shape (T, C, H, W).
    """
    
    def __init__(self, processed_dir, unprocessed_dir, clip_length=16,
                 clip_stride=4, crop_size=224):
        super().__init__()
        self.clip_length = clip_length
        self.crop_size = crop_size
        
        # Collect all sequences from both processed and unprocessed dirs
        all_sequences = {}
        if os.path.exists(processed_dir):
            all_sequences.update(find_all_sequences(processed_dir))
        if os.path.exists(unprocessed_dir):
            all_sequences.update(find_all_sequences(unprocessed_dir))
        
        # Build clip index: (seq_name, start_frame_idx)
        self.clips = []
        self.sequences = all_sequences
        
        for seq_name, frame_paths in all_sequences.items():
            n_frames = len(frame_paths)
            if n_frames < clip_length:
                # If sequence is shorter than clip, still use it (will repeat)
                self.clips.append((seq_name, 0))
            else:
                for start in range(0, n_frames - clip_length + 1, clip_stride):
                    self.clips.append((seq_name, start))
        
        # Transforms (consistent across clip)
        self.transform = transforms.Compose([
            transforms.Resize((crop_size, crop_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
    
    def __len__(self):
        return len(self.clips)
    
    def __getitem__(self, idx):
        seq_name, start = self.clips[idx]
        frame_paths = self.sequences[seq_name]
        n_frames = len(frame_paths)
        
        frames = []
        for t in range(self.clip_length):
            # Handle sequences shorter than clip_length by wrapping
            frame_idx = min(start + t, n_frames - 1)
            img = Image.open(frame_paths[frame_idx]).convert("RGB")
            img = self.transform(img)
            frames.append(img)
        
        # Stack to (T, C, H, W)
        clip = torch.stack(frames, dim=0)
        return clip


# ============================================================
#  Dataset 2: Segmentation (labeled, single-frame)
# ============================================================

class VJEPA2SegmentationDataset(Dataset):
    """Loads labeled image-mask pairs for supervised segmentation.
    Returns single frames (not video clips).
    """
    
    def __init__(self, processed_dir, crop_size=224, augment=True):
        super().__init__()
        self.pairs = find_labeled_pairs(processed_dir)
        self.crop_size = crop_size
        self.augment = augment
        
        if augment and HAS_ALBUMENTATIONS:
            self.aug = A.Compose([
                A.Resize(crop_size, crop_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.RandomRotate90(p=0.3),
                A.RandomBrightnessContrast(p=0.3),
                A.GaussNoise(p=0.2),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ])
        else:
            self.aug = None
        
        # Fallback transforms (no augmentation)
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
        
        if self.aug is not None:
            img_np = np.array(img)
            mask_np = np.array(mask)
            # Binarize mask
            mask_np = (mask_np > 127).astype(np.float32)
            
            augmented = self.aug(image=img_np, mask=mask_np)
            img_tensor = augmented["image"]           # (C, H, W)
            mask_tensor = augmented["mask"].unsqueeze(0)  # (1, H, W)
        else:
            img_tensor = self.img_transform(img)
            mask_tensor = self.mask_transform(mask)
            mask_tensor = (mask_tensor > 0.5).float()
        
        return img_tensor, mask_tensor


# ============================================================
#  Data loader factory functions
# ============================================================

def get_vjepa2_pretrain_loader(processed_dir, unprocessed_dir,
                                clip_length=16, clip_stride=4,
                                crop_size=224, batch_size=8,
                                num_workers=4):
    """Create DataLoader for V-JEPA 2 pretraining."""
    dataset = VJEPA2PretrainDataset(
        processed_dir=processed_dir,
        unprocessed_dir=unprocessed_dir,
        clip_length=clip_length,
        clip_stride=clip_stride,
        crop_size=crop_size,
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return loader


def get_vjepa2_seg_loaders(processed_dir, crop_size=224, batch_size=16,
                            num_workers=4, train_split=0.85, seed=42):
    """Create train and validation DataLoaders for segmentation."""
    full_dataset = VJEPA2SegmentationDataset(
        processed_dir=processed_dir,
        crop_size=crop_size,
        augment=True,
    )
    
    n_total = len(full_dataset)
    n_train = int(n_total * train_split)
    n_val = n_total - n_train
    
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        full_dataset, [n_train, n_val], generator=generator,
    )
    
    # Disable augmentation for validation
    val_dataset_no_aug = VJEPA2SegmentationDataset(
        processed_dir=processed_dir,
        crop_size=crop_size,
        augment=False,
    )
    val_indices = val_dataset.indices
    val_subset = torch.utils.data.Subset(val_dataset_no_aug, val_indices)
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    
    return train_loader, val_loader
