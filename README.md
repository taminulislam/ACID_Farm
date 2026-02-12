# I-JEPA Segmentation Pipeline for Thermal Cow Imagery 🐄🔥

Self-supervised pretraining + supervised fine-tuning pipeline for generating segmentation masks on thermal cow frames from the ACID Farm dataset.

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────┐
│  STEP 1: I-JEPA Pretraining (Self-Supervised)               │
│  • Input:  ALL ~3,059 thermal frames (no labels needed)     │
│  • Output: Pretrained ViT encoder weights                   │
│  • What:   Learns "what thermal cows look like"             │
├─────────────────────────────────────────────────────────────┤
│  STEP 2: Segmentation Fine-tuning (Supervised)              │
│  • Input:  269 labeled image-mask pairs                     │
│  • Output: Trained segmentation model                       │
│  • What:   Learns to generate masks matching your style     │
├─────────────────────────────────────────────────────────────┤
│  STEP 3: Inference (Prediction)                             │
│  • Input:  2,790 unlabeled frames                           │
│  • Output: Predicted masks + overlays                       │
│  • What:   Generates masks for remaining frames             │
└─────────────────────────────────────────────────────────────┘
```

## Project Structure

```
ACID_Farm/
├── configs/
│   ├── pretrain.yaml          # I-JEPA pretraining hyperparameters
│   └── segmentation.yaml      # Segmentation training hyperparameters
├── processed/                  # Labeled data (269 image-mask pairs)
│   ├── SEQ_0483/
│   │   ├── images/ (or frames/)
│   │   ├── masks/
│   │   └── overlays/
│   └── ...
├── unprocessed/                # Unlabeled data (2,790 images)
│   ├── SEQ_0495/
│   │   └── images/
│   └── ...
├── checkpoints/                # Saved model weights (git-ignored)
├── logs/                       # TensorBoard logs (git-ignored)
├── dataset.py                  # Data loading utilities
├── models.py                   # ViT encoder, I-JEPA, segmentation model
├── utils.py                    # Losses, metrics, schedulers
├── pretrain_ijepa.py           # STEP 1: Self-supervised pretraining
├── train_segmentation.py       # STEP 2: Supervised fine-tuning
├── predict.py                  # STEP 3: Generate masks
├── visualize.py                # Visualization utilities
├── setup_env.sh                # Environment setup script
├── .gitignore                  # Git ignore rules
└── README.md                   # This file
```

## Quick Start

### 0. Setup Environment

```bash
# Create and activate the conda environment
bash setup_env.sh

# Activate the environment
conda activate ijepa_seg

# Verify installation
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### 1. Verify Data Loading

```bash
# Test that datasets load correctly
python dataset.py --processed processed --unprocessed unprocessed
```

Expected output:
```
Testing IJEPAPretrainDataset
  Total images: ~3059
  Sample shape: torch.Size([3, 224, 224])

Testing SegmentationDataset
  Total labeled pairs: 269
  Image shape:  torch.Size([3, 224, 224])
  Mask shape:   torch.Size([1, 224, 224])

✓ All dataset tests passed!
```

### 2. Verify Models

```bash
# Test model architectures
python models.py
```

### 3. Visualize Labeled Data

```bash
# See what your labeled data looks like
python visualize.py --mode samples --processed processed --num 8 --save labeled_samples.png
```

---

## Step-by-Step Training Guide

### STEP 1: I-JEPA Pretraining (Self-Supervised)

This step learns thermal cow features from ALL your frames without any labels.

```bash
python pretrain_ijepa.py --config configs/pretrain.yaml
```

**Expected duration:** ~4-6 hours on RTX 6000 Ada (300 epochs × ~3,059 images)

**Monitor training:**
```bash
# In a separate terminal
tensorboard --logdir logs/pretrain --port 6006
```

**What to look for:**
- Loss should steadily decrease
- After ~100 epochs, features become meaningful
- Model saves checkpoints every 50 epochs

**Output:**
- `checkpoints/ijepa_encoder_best.pth` — best encoder weights
- `checkpoints/ijepa_encoder_final.pth` — final encoder weights

### STEP 2: Segmentation Fine-tuning

Train the segmentation head using your 269 labeled frames.

```bash
python train_segmentation.py --config configs/segmentation.yaml
```

**Training strategy:**
1. **Epochs 1-10:** Encoder frozen, only decoder trains
2. **Epochs 11+:** Full model fine-tuning with differential LR
3. **Early stopping** after 20 epochs without improvement

**Monitor:**
```bash
tensorboard --logdir logs/segmentation --port 6007
```

**What to look for:**
- Val IoU should reach 0.7-0.9
- Val Dice should reach 0.8-0.95
- Loss should decrease steadily

**Output:**
- `checkpoints/segmentation_best.pth` — best model

### STEP 3: Predict on Unlabeled Frames

Generate masks for all 2,790 unlabeled frames.

```bash
python predict.py --config configs/segmentation.yaml
```

**Output:**
- Masks saved to `unprocessed/SEQ_XXXX/masks/`
- Overlays saved to `unprocessed/SEQ_XXXX/overlays/`

### STEP 4: Inspect Results

```bash
# View predictions for a specific sequence
python visualize.py --mode predictions --predictions unprocessed/SEQ_0495 --num 8

# Compare against ground truth (on processed sequences)
python visualize.py --mode compare --processed processed --predictions predictions --num 6
```

---

## Comparison: With vs Without I-JEPA

| Metric | Without I-JEPA | With I-JEPA |
|--------|---------------|-------------|
| **Training data** | 269 labeled only | 3,059 pretrain + 269 labeled |
| **Domain knowledge** | None (random init) | Thermal cow features |
| **Expected IoU** | 0.60-0.75 | 0.75-0.90 |
| **Generalization** | Limited | Better |

To train without I-JEPA (for comparison):
```bash
python train_segmentation.py --config configs/segmentation.yaml --no_pretrain
```

---

## Advanced Options

### Resume Training
```bash
# Resume pretraining
python pretrain_ijepa.py --config configs/pretrain.yaml --resume checkpoints/ijepa_checkpoint_epoch100.pth

# Resume segmentation training
python train_segmentation.py --config configs/segmentation.yaml --resume checkpoints/seg_checkpoint.pth
```

### Custom Encoder Weights
```bash
python train_segmentation.py --config configs/segmentation.yaml --encoder_weights path/to/custom_encoder.pth
```

### Adjust Prediction Threshold
```bash
python predict.py --threshold 0.3  # more aggressive (detects more cow pixels)
python predict.py --threshold 0.7  # more conservative (fewer false positives)
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| CUDA out of memory | Reduce `batch_size` in config |
| No images found | Check `processed/` and `unprocessed/` paths |
| Low IoU | Train pretraining longer, reduce `freeze_encoder_epochs` |
| Noisy predictions | Increase threshold, add post-processing |

## Hardware Requirements

- **GPU:** NVIDIA GPU with ≥ 8GB VRAM (tested on RTX 6000 Ada 49GB)
- **RAM:** 16GB+
- **Disk:** ~5GB for checkpoints
