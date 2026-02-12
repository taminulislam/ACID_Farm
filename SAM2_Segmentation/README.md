# SAM 2 Segmentation Pipeline for Thermal Cow Imagery

Fine-tune **SAM 2.1** (Segment Anything Model 2) on labeled thermal cow frames for high-quality instance segmentation.

## Why SAM 2?

| Feature | I-JEPA Pipeline | SAM 2 Pipeline |
|---------|----------------|----------------|
| **Encoder** | ViT-Small (22M, trained on 3K frames) | Hiera (trained on 11M images, 1B masks) |
| **Decoder** | Simple 3-layer ConvTranspose | SAM mask decoder (prompt-based) |
| **Expected IoU** | ~0.50 | **0.80-0.92** |
| **Boundary quality** | Rough edges | Sharp, precise |
| **Prompt support** | None | Points, boxes, masks |

## Project Structure

```
SAM2_Segmentation/
├── setup_env.sh              # Environment setup (Python 3.11 + SAM 2)
├── configs/
│   └── train.yaml            # Training configuration
├── dataset_sam2.py           # Data loading with prompt generation
├── train_sam2.py             # Fine-tune SAM 2.1 mask decoder
├── predict_sam2.py           # Generate masks for unlabeled frames
├── visualize_sam2.py         # Visualization utilities
├── checkpoints/              # Model weights (git-ignored)
│   ├── sam2.1_hiera_small.pt       # Base SAM 2.1 checkpoint
│   ├── sam2_mask_decoder_best.pth  # Fine-tuned decoder
│   └── sam2_checkpoint_epoch*.pth  # Training checkpoints
├── logs/                     # TensorBoard logs (git-ignored)
└── README.md                 # This file
```

## Quick Start

### 1. Setup Environment

```bash
cd ACID_Farm/SAM2_Segmentation
bash setup_env.sh
conda activate sam2_seg
```

### 2. Fine-tune SAM 2

```bash
python train_sam2.py --config configs/train.yaml
```

### 3. Generate Predictions

```bash
python predict_sam2.py --config configs/train.yaml
```

### 4. Visualize Results

```bash
python visualize_sam2.py --mode predictions --input ../unprocessed/SEQ_0495 --num 8
```

## Pipeline Details

### Training Strategy

1. **Phase 1 (Epochs 1-50):** Only train the **mask decoder** (~4M params)
   - Image encoder frozen (saves memory)
   - Prompt encoder frozen
   - Learning rate: 1e-4

2. **Phase 2 (Epochs 51+):** Unfreeze image encoder for full fine-tuning
   - Encoder LR: 5e-6 (100x lower than decoder)
   - Fine-grained adaptation to thermal domain

### Prompt Strategy

During training, prompts are **automatically generated** from ground truth masks:
- **Point prompts:** 3 random foreground + 1 background point per mask
- **Box prompts:** Bounding box derived from mask with ±10px noise
- This teaches SAM 2 to respond correctly to various prompts

### Loss Function

Combined loss (following SAM's original training):
- **Focal Loss** (weight=20): Handles class imbalance
- **Dice Loss** (weight=1): Optimizes region overlap
- **IoU Prediction Loss** (weight=1): Improves predicted confidence

### Resume Training

```bash
# Resume from checkpoint
python train_sam2.py --config configs/train.yaml --resume checkpoints/sam2_checkpoint_epoch50.pth
```

## Hardware Requirements

| Resource | Training | Inference |
|----------|----------|-----------|
| **GPU Memory** | ~16 GB | ~8 GB |
| **Time (RTX 6000)** | ~2-4 hours | ~10 min |
| **Python** | ≥ 3.10 | ≥ 3.10 |
| **PyTorch** | ≥ 2.5.1 | ≥ 2.5.1 |

## Data Requirements

Uses the same data as the I-JEPA pipeline:
- **Labeled:** `../processed/` (269 image-mask pairs)
- **Unlabeled:** `../unprocessed/` (2,790 frames for prediction)
