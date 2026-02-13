# Thermal Cow Segmentation: Experimental Comparison

**Project:** ACID Farm - Automated Cow Instance Detection  
**Date:** February 2026  
**Objective:** Develop robust segmentation models for thermal imagery of dairy cows

---

## Executive Summary

This document compares two deep learning approaches for segmenting cows in thermal imagery: (1) a semi-supervised I-JEPA-based pipeline and (2) a supervised SAM 2.1 fine-tuning approach. The SAM 2.1 pipeline achieved **76% IoU**, representing a **52% improvement** over the I-JEPA baseline (50% IoU), demonstrating the effectiveness of leveraging large-scale pretrained foundation models for domain-specific segmentation tasks.

---

## Dataset Overview

### Data Composition
- **Total Sequences:** 36 thermal video sequences
- **Total Frames:** ~8,000+ unlabeled thermal images
- **Labeled Data:** 269 image-mask pairs from 5 sequences
  - Training set: 228 pairs (85%)
  - Validation set: 41 pairs (15%)
- **Image Specifications:**
  - Original resolution: 240×320 pixels
  - Format: Grayscale thermal imagery
  - Subject: Dairy cows in barn environment

### Data Challenges
- Limited labeled data (269 samples)
- Low resolution thermal imagery
- Variable cow poses and occlusions
- Cluttered barn backgrounds
- Domain gap from natural RGB images

---

## Experiment 1: I-JEPA Semi-Supervised Pipeline

### Methodology

**Architecture:**
- **Backbone:** Vision Transformer (ViT-Base/16)
- **Pretraining:** I-JEPA (Image-based Joint-Embedding Predictive Architecture)
- **Decoder:** Lightweight 3-layer transposed convolution network

**Training Strategy:**
1. **Phase 1 - Unsupervised Pretraining (I-JEPA):**
   - Used all 8,000+ unlabeled thermal frames
   - Self-supervised learning via masked patch prediction
   - Trained for 150 epochs with resume capability
   - Objective: Learn general visual representations from thermal imagery

2. **Phase 2 - Supervised Fine-tuning:**
   - Froze pretrained ViT encoder
   - Trained only the segmentation decoder
   - Used 228 labeled image-mask pairs
   - Binary cross-entropy + Dice loss
   - 100 epochs with early stopping

**Implementation Details:**
- Environment: Python 3.9, PyTorch 2.0.1
- Optimizer: AdamW (lr=1e-4, weight decay=0.05)
- Batch size: 16 (pretraining), 8 (fine-tuning)
- Image size: 224×224 (resized from 240×320)
- Data augmentation: Random flip, rotation, color jitter

### Results

| Metric | Value |
|--------|-------|
| **Validation IoU** | 0.50 |
| **Validation Dice** | 0.67 |
| **Validation Accuracy** | 0.96 |
| **Training Time** | ~12 hours (pretraining) + 2 hours (fine-tuning) |

**Key Observations:**
- ✅ Successfully leveraged unlabeled data for representation learning
- ✅ Reasonable baseline performance with limited supervision
- ❌ Moderate IoU indicates room for improvement
- ❌ Simple decoder may be insufficient for complex segmentation
- ❌ Domain gap: I-JEPA pretrained on thermal data vs. SAM on natural images

**Strengths:**
- Utilizes all available unlabeled data
- Learns domain-specific features from thermal imagery
- Computationally efficient decoder

**Limitations:**
- Pretraining on limited thermal data (~8K images) vs. foundation models (millions)
- Simple decoder architecture
- No multi-scale feature fusion
- Limited to binary segmentation

---

## Experiment 2: SAM 2.1 Fine-Tuning Pipeline

### Methodology

**Architecture:**
- **Backbone:** SAM 2.1 Hiera-Small (46.1M parameters)
- **Pretrained on:** SA-1B dataset (11M images, 1.1B masks)
- **Fine-tuning:** Mask decoder only (initially), then full model

**Training Strategy:**
1. **Phase 1 - Decoder Fine-tuning (Epochs 1-50):**
   - Froze image encoder (34.4M params)
   - Froze prompt encoder (0.04M params)
   - Trained mask decoder only (11.7M params)
   - Learning rate: 1e-4 with cosine decay + warmup

2. **Phase 2 - Full Model Fine-tuning (Epoch 51+):**
   - Unfroze image encoder with low learning rate (5e-6)
   - Continued training mask decoder (lr=1e-4)
   - Total trainable: 46.1M parameters

**Prompt Engineering:**
- **Point prompts:** 3 random foreground points + 1 background point per mask
- **Bounding box prompts:** Derived from ground truth masks with noise augmentation
- **Inference:** Grid-based automatic prompting (3×3 foreground + 4 corner background)

**Implementation Details:**
- Environment: Python 3.11, PyTorch 2.5.1, CUDA 12.4
- Optimizer: AdamW (decoder lr=1e-4, encoder lr=5e-6, weight decay=0.01)
- Batch size: 4 (limited by 1024×1024 image size)
- Image size: 1024×1024 (SAM 2 native resolution)
- Loss: Focal (20×) + Dice (1×) + IoU prediction (1×)
- Data augmentation: Flip, rotation, brightness/contrast, Gaussian noise
- Gradient clipping: 1.0
- Early stopping: 30 epochs patience

### Results

| Metric | I-JEPA Baseline | SAM 2.1 (Best) | Improvement |
|--------|-----------------|----------------|-------------|
| **Validation IoU** | 0.50 | **0.76** | **+52%** |
| **Validation Dice** | 0.67 | **0.86** | **+29%** |
| **Validation Accuracy** | 0.96 | **0.98** | **+2%** |
| **Training Time** | 14 hours | 2 hours | **-86%** |

**Training Progression:**
- Epoch 1: Val IoU 0.68 (already exceeds I-JEPA!)
- Epoch 8: Val IoU 0.74 (first peak)
- Epoch 34: Val IoU **0.76** (best model) ⭐
- Epoch 50: Training stopped (encoder unfrozen)

**Key Observations:**
- ✅ Immediate strong performance from pretrained weights
- ✅ Significant improvement over I-JEPA baseline
- ✅ Faster convergence (no pretraining phase needed)
- ✅ Superior multi-scale feature extraction
- ✅ Robust to thermal domain despite RGB pretraining

**Strengths:**
- Leverages massive-scale pretraining (1.1B masks)
- Advanced decoder with high-resolution feature fusion
- Prompt-based inference enables flexible deployment
- Strong generalization despite domain shift

**Limitations:**
- Requires more GPU memory (1024×1024 images, batch size 4)
- Larger model size (176MB checkpoint)
- No utilization of unlabeled thermal data

---

## Comparative Analysis

### Performance Comparison

**Quantitative Metrics:**

| Aspect | I-JEPA | SAM 2.1 | Winner |
|--------|--------|---------|--------|
| IoU | 0.50 | **0.76** | SAM 2.1 (+52%) |
| Dice | 0.67 | **0.86** | SAM 2.1 (+29%) |
| Accuracy | 0.96 | **0.98** | SAM 2.1 (+2%) |
| Training Time | 14h | **2h** | SAM 2.1 (-86%) |
| GPU Memory | Low | High | I-JEPA |
| Model Size | 87MB | 176MB | I-JEPA |

**Qualitative Observations:**
- **SAM 2.1** produces sharper, more accurate boundaries
- **SAM 2.1** better handles occlusions and partial views
- **I-JEPA** occasionally misses small or distant cows
- **SAM 2.1** more robust to background clutter

### Architectural Differences

| Component | I-JEPA Pipeline | SAM 2.1 Pipeline |
|-----------|-----------------|------------------|
| **Encoder** | ViT-Base/16 (86M) | Hiera-Small (34M) |
| **Decoder** | 3-layer TransConv | Multi-scale transformer decoder |
| **Pretraining** | I-JEPA on 8K thermal images | SA-1B (11M images, 1.1B masks) |
| **Input Size** | 224×224 | 1024×1024 |
| **Prompts** | None | Points + boxes |
| **Multi-scale** | No | Yes (3-level FPN) |

### Data Utilization

**I-JEPA:**
- ✅ Uses all 8,000+ unlabeled frames (pretraining)
- ✅ Learns domain-specific thermal features
- ❌ Limited scale compared to foundation models

**SAM 2.1:**
- ❌ Only uses 269 labeled pairs (fine-tuning)
- ❌ Does not leverage unlabeled thermal data
- ✅ Benefits from massive-scale RGB pretraining

### Computational Requirements

**I-JEPA:**
- Pretraining: ~12 hours on single GPU
- Fine-tuning: ~2 hours
- Total: ~14 hours
- Memory: Moderate (batch size 16)

**SAM 2.1:**
- Fine-tuning only: ~2 hours
- Memory: High (batch size 4, 1024×1024 images)
- Inference: ~2-3 seconds per image

---

## Conclusions and Recommendations

### Key Findings

1. **Foundation Models Excel:** SAM 2.1's pretraining on 1.1B masks provides superior features compared to I-JEPA's 8K thermal images, despite the domain gap.

2. **Quality Over Quantity:** 269 labeled samples with SAM 2.1 outperform 8,000 unlabeled + 269 labeled with I-JEPA, highlighting the importance of model architecture and pretraining scale.

3. **Efficiency Gains:** SAM 2.1 achieves better results in 1/7th the training time by eliminating the pretraining phase.

4. **Prompt Engineering Matters:** SAM 2.1's prompt-based approach enables flexible inference strategies (automatic grid, interactive, etc.).

### Recommendations

**For Production Deployment:**
- ✅ **Use SAM 2.1 pipeline** for best segmentation quality
- ✅ Implement automatic grid-based prompting for batch processing
- ✅ Consider model quantization to reduce memory footprint
- ⚠️ Monitor performance on edge cases (heavy occlusion, extreme poses)

**For Future Research:**
1. **Hybrid Approach:** Combine SAM 2.1 with I-JEPA pretraining on thermal data
2. **Active Learning:** Use SAM 2.1 to pseudo-label unlabeled frames, retrain
3. **Multi-Task Learning:** Joint training on segmentation + detection + tracking
4. **Domain Adaptation:** Fine-tune SAM 2.1 encoder on unlabeled thermal data
5. **Ensemble Methods:** Combine I-JEPA and SAM 2.1 predictions

**For Data Collection:**
- Prioritize labeling diverse scenarios (occlusions, multiple cows, edge cases)
- Consider semi-automated labeling with SAM 2.1 + human verification
- Expand to other thermal imaging conditions (outdoor, different times)

### Impact

The **52% improvement in IoU** demonstrates that modern foundation models like SAM 2.1 can effectively transfer to specialized domains (thermal imagery) with minimal fine-tuning. This has significant implications for agricultural AI applications where labeled data is scarce but pretrained models are abundant.

**Practical Applications:**
- Automated cow monitoring and health assessment
- Precision livestock farming
- Behavior analysis and anomaly detection
- Integration with tracking systems for individual cow identification

---

## Appendix: Technical Details

### File Structure

```
ACID_Farm/
├── I-JEPA Pipeline
│   ├── pretrain_ijepa.py          # Unsupervised pretraining
│   ├── train_segmentation.py     # Supervised fine-tuning
│   ├── predict.py                 # Inference
│   ├── dataset.py                 # Data loading
│   ├── models.py                  # Model definitions
│   └── configs/
│       ├── pretrain.yaml
│       └── train.yaml
│
└── SAM2_Segmentation/
    ├── train_sam2.py              # Fine-tuning script
    ├── predict_sam2.py            # Inference script
    ├── dataset_sam2.py            # SAM 2 data loading
    ├── visualize_sam2.py          # Visualization utilities
    ├── setup_env.sh               # Environment setup
    └── configs/
        └── train.yaml
```

### Reproducibility

**I-JEPA Pipeline:**
```bash
conda activate ijepa_seg
python pretrain_ijepa.py --config configs/pretrain.yaml
python train_segmentation.py --config configs/train.yaml
python predict.py --config configs/train.yaml
```

**SAM 2.1 Pipeline:**
```bash
bash setup_env.sh
conda activate sam2_seg
python train_sam2.py --config configs/train.yaml
python predict_sam2.py --config configs/train.yaml
```

### Hardware Specifications
- GPU: NVIDIA GPU with CUDA 12.4 support
- RAM: 32GB+ recommended
- Storage: ~5GB for datasets, ~500MB for checkpoints

---

**Document Version:** 1.0  
**Last Updated:** February 12, 2026  
**Contact:** ACID Farm Research Team
