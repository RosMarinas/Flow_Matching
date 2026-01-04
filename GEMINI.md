# Flow Matching Reproduction Project

## Project Overview

This project is a reproduction of the paper **"Flow Matching for Generative Modeling" (ICLR 2023)**, developed as part of a graduate Machine Learning course (Project 3). It implements **Optimal Transport Conditional Flow Matching (OT-CFM)** and other variants to generate data by regressing a vector field that transports a noise distribution to a data distribution.

The implementation includes both 2D toy experiments (8 Gaussians) and image generation support (CIFAR-10, ImageNet, CelebA) using an improved U-Net architecture.

### Key Technologies
- **Language**: Python >= 3.9
- **Framework**: PyTorch
- **Dependency Management**: `uv`
- **Math/Physics**: `torchdiffeq` (ODE Solvers), Optimal Transport Theory

## Directory Structure

### `Flow_Matching/src/` (Source Code)
*   **`train.py`**: The entry point for training. Supports CLI arguments for dataset, model type, flow matching objective, and resolution.
*   **`models.py`**: Defines the `ConditionalFlowMatching` abstract base class and its implementations (OT, VP, Target).
*   **`unet.py`**: A deep U-Net architecture with Time Embeddings and Attention, capable of handling 64x64 or 128x128 images.
*   **`datasets.py`**: Handles data loading for '8gaussians', 'cifar10', 'imagenet', and 'celeba'.
*   **`todolist.md`**: Project roadmap.

### Root Directory
*   **`Flow_Matching/`**: Project root containing `pyproject.toml` and `README.md`.
*   **`Flow_Matching_Detailed_Analysis.md`**: A detailed theoretical analysis of the Flow Matching paper.
*   **`Project3.pdf`**: Original assignment description.
*   **`Thesis/`**: Contains the source PDFs and related literature.

## Setup and Usage

### Prerequisites
This project uses **uv** for fast package management.

1.  **Install Dependencies**:
    ```bash
    cd Flow_Matching
    uv sync
    ```

### Training

The `train.py` script is the central hub for running experiments. **Run from the `src` directory.**

```bash
cd Flow_Matching
uv run python src/train.py [ARGS]
```

#### Examples

**1. 2D Toy Dataset (8 Gaussians)**
Fast training to visualize vector fields.
```bash
uv run python src/train.py \
    --dataset 8gaussians \
    --fm_type ot \
    --model mlp \
    --iterations 5000
```

**2. CIFAR-10 Image Generation (32x32)**
```bash
uv run python src/train.py \
    --dataset cifar10 \
    --fm_type ot \
    --model unet \
    --batch_size 128 \
    --image_size 32 \
    --iterations 50000
```

**3. ImageNet / CelebA (High Resolution)**
To train on ImageNet (requires downloaded data) or CelebA (auto-download):
```bash
# CelebA 64x64
uv run python src/train.py \
    --dataset celeba \
    --model unet \
    --image_size 64 \
    --batch_size 64 \
    --data_root ./data

# ImageNet 64x64 (Custom Path)
uv run python src/train.py \
    --dataset imagenet \
    --model unet \
    --image_size 64 \
    --data_root /path/to/imagenet \
    --batch_size 64
```

### Key Arguments
*   `--dataset`: `8gaussians`, `cifar10`, `imagenet`, `celeba`.
*   `--fm_type`: `ot` (Optimal Transport), `vp` (Variance Preserving), `target`.
*   `--model`: `mlp` (toy), `unet` (images).
*   `--image_size`: Resolution (e.g., 32, 64, 128). Defaults to 32 (CIFAR) or 64 (others).
*   `--data_root`: Path to store/load datasets.

## Development Status

*   [x] Basic OT-CFM implementation.
*   [x] U-Net architecture for images (Enhanced for higher res).
*   [x] Unified training script with CLI.
*   [x] ImageNet/CelebA support.
*   [ ] Full CIFAR-10/ImageNet training and FID evaluation.