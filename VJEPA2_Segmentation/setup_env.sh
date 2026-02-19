#!/bin/bash
# ============================================================
# V-JEPA 2 Segmentation Pipeline — Environment Setup
# ============================================================
# Installs V-JEPA 2 dependencies into the EXISTING sam2_seg env.
#
# Usage:
#     bash setup_env.sh
# ============================================================

set -e

ENV_NAME="sam2_seg"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================================"
echo "  V-JEPA 2 Segmentation — Environment Setup"
echo "  (Installing into existing '${ENV_NAME}' environment)"
echo "============================================================"

# ---- Step 1: Install V-JEPA 2 from source ----
VJEPA2_DIR="/tmp/vjepa2"
if [ -d "${VJEPA2_DIR}" ]; then
    echo "[INFO] V-JEPA 2 repo already cloned at ${VJEPA2_DIR}"
    cd "${VJEPA2_DIR}" && git pull
else
    echo "[INFO] Cloning V-JEPA 2 repository..."
    git clone https://github.com/facebookresearch/vjepa2.git "${VJEPA2_DIR}"
fi

echo "[INFO] Installing V-JEPA 2 package..."
cd "${VJEPA2_DIR}"
conda run -n "${ENV_NAME}" pip install -e .

# ---- Step 2: Install additional dependencies ----
echo "[INFO] Installing extra dependencies (timm, einops, decord)..."
conda run -n "${ENV_NAME}" pip install \
    timm \
    einops \
    decord

# ---- Step 3: Create directory structure ----
echo "[INFO] Creating pipeline directories..."
mkdir -p "${SCRIPT_DIR}/checkpoints"
mkdir -p "${SCRIPT_DIR}/logs/pretrain"
mkdir -p "${SCRIPT_DIR}/logs/segmentation"
mkdir -p "${SCRIPT_DIR}/predictions"
mkdir -p "${SCRIPT_DIR}/configs"

# ---- Step 4: Download V-JEPA 2 ViT-Large checkpoint ----
CKPT_PATH="${SCRIPT_DIR}/checkpoints/vitl.pt"
if [ -f "${CKPT_PATH}" ]; then
    echo "[INFO] V-JEPA 2 ViT-Large checkpoint already exists."
else
    echo "[INFO] Downloading V-JEPA 2 ViT-Large checkpoint (~1.2GB)..."
    wget -q --show-progress \
        https://dl.fbaipublicfiles.com/vjepa2/vitl.pt \
        -O "${CKPT_PATH}"
fi

echo ""
echo "============================================================"
echo "  Setup complete!"
echo "============================================================"
echo "  Environment:    ${ENV_NAME} (existing)"
echo "  Checkpoint:     ${CKPT_PATH}"
echo "  Pipeline:       ${SCRIPT_DIR}/"
echo "============================================================"
