#!/bin/bash
# ============================================================
# I-JEPA Segmentation Pipeline — Environment Setup
# ============================================================
# Usage:  bash setup_env.sh
# After:  conda activate ijepa_seg
# ============================================================

set -e

ENV_NAME="ijepa_seg"

echo "============================================"
echo "  Creating conda environment: $ENV_NAME"
echo "============================================"

# Create conda environment
conda create -n $ENV_NAME python=3.9 -y

# Activate
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

echo "============================================"
echo "  Installing PyTorch (CUDA 11.8)"
echo "============================================"
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118

echo "============================================"
echo "  Installing dependencies"
echo "============================================"
pip install \
    timm==0.9.12 \
    einops==0.7.0 \
    tensorboard==2.15.1 \
    scikit-learn==1.3.2 \
    opencv-python-headless==4.9.0.80 \
    matplotlib==3.8.2 \
    tqdm==4.66.1 \
    pyyaml==6.0.1 \
    Pillow==10.2.0 \
    numpy==1.24.4 \
    albumentations==1.3.1

echo "============================================"
echo "  Environment '$ENV_NAME' is ready!"
echo "  Activate with: conda activate $ENV_NAME"
echo "============================================"
