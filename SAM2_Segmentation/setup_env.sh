#!/bin/bash
# ============================================================
# SAM 2 Segmentation Pipeline — Environment Setup
# ============================================================
# Creates a conda environment with Python 3.11, PyTorch 2.5.1,
# and installs SAM 2 from the official Facebook Research repo.
# ============================================================

set -e

ENV_NAME="sam2_seg"
PYTHON_VERSION="3.11"

echo "============================================"
echo "  Setting up SAM 2 Segmentation Environment"
echo "============================================"

# Check if environment exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Environment '${ENV_NAME}' already exists."
    echo "To recreate: conda env remove -n ${ENV_NAME}"
    echo "Activating existing environment..."
    eval "$(conda shell.bash hook)"
    conda activate ${ENV_NAME}
else
    echo "Creating conda environment: ${ENV_NAME} (Python ${PYTHON_VERSION})"
    conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y

    eval "$(conda shell.bash hook)"
    conda activate ${ENV_NAME}

    echo ""
    echo "Installing PyTorch 2.5.1 with CUDA 12.4..."
    pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124

    echo ""
    echo "Installing SAM 2 from source..."
    cd /tmp
    if [ -d "sam2" ]; then
        rm -rf sam2
    fi
    git clone https://github.com/facebookresearch/sam2.git
    cd sam2
    SAM2_BUILD_ALLOW_ERRORS=1 pip install -e .
    cd -

    echo ""
    echo "Installing additional dependencies..."
    pip install \
        albumentations==1.3.1 \
        opencv-python-headless==4.9.0.80 \
        matplotlib==3.8.2 \
        tensorboard==2.15.1 \
        scikit-learn==1.3.2 \
        tqdm \
        pyyaml \
        einops

    echo ""
    echo "Downloading SAM 2.1 checkpoint (sam2.1_hiera_small)..."
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    CKPT_DIR="${SCRIPT_DIR}/checkpoints"
    mkdir -p "${CKPT_DIR}"
    
    # Download SAM 2.1 Hiera Small checkpoint
    if [ ! -f "${CKPT_DIR}/sam2.1_hiera_small.pt" ]; then
        wget -q --show-progress -O "${CKPT_DIR}/sam2.1_hiera_small.pt" \
            https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt
        echo "Downloaded sam2.1_hiera_small.pt → ${CKPT_DIR}/"
    else
        echo "Checkpoint already exists: ${CKPT_DIR}/sam2.1_hiera_small.pt"
    fi
fi

echo ""
echo "============================================"
echo "  Environment '${ENV_NAME}' is ready!"
echo "  Activate with: conda activate ${ENV_NAME}"
echo "============================================"
echo ""
echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo ""
echo "Next steps:"
echo "  1. conda activate ${ENV_NAME}"
echo "  2. python train_sam2.py --config configs/train.yaml"
echo "  3. python predict_sam2.py --config configs/train.yaml"
