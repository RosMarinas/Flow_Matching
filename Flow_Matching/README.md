# Flow Matching Reproduction

This repository contains a reproduction of the core algorithm from the paper "Flow Matching for Generative Modeling" (ICLR 2023).

## Algorithm

We implemented the **Optimal Transport Conditional Flow Matching (OT-CFM)**.
- **Model**: A 4-layer MLP with Gaussian Fourier time embeddings.
- **Loss**: Regression objective matching the conditional vector field $u_t(x|x_1)$.
- **Path**: Optimal Transport displacement interpolation (linear interpolation between noise and data).
- **Sampling**: Euler method ODE solver.

## Structure

- `models.py`: Contains the `VectorFieldNet` (MLP) and `OTFlowMatching` (Loss & Sampling logic).
- `train.py`: Training script for 2D Toy Datasets (8 Gaussians).
- `pyproject.toml`: Dependencies management using `uv`.

## Usage

1. **Install Dependencies**:
   Ensure you have `uv` installed.
   ```bash
   uv sync
   ```

2. **Train**:
   Run the training script. It will train on the "8 Gaussians" dataset by default.
   ```bash
   uv run python train.py
   ```

3. **Results**:
   - Check the `results/` folder for generated plots (`step_XXXXX.png`) showing the flow evolution.
   - The final model is saved as `results/model.pth`.

## Hyperparameters

- Iterations: 5000
- Batch Size: 512
- Learning Rate: 1e-3
- Sigma Min: 1e-4

## References

- [Flow Matching for Generative Modeling (arXiv)](https://arxiv.org/abs/2210.02747)
- [Official Implementation](https://github.com/facebookresearch/flow_matching)
