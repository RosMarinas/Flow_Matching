# GEMINI.md


## Core Philosophy

1. 你的修改应该使得项目更易于理解和维护，而不是更复杂。需要进行改进时请直接在源代码中修改，而不是新建文件.
2. 你可以编写脚本临时测试功能，但是请新建一个temp/目录，将这些脚本放在那里，并在完成后删除它们。
3. 所有需要运行的脚本都必须交由用户执行。 不要在代码中自动运行任何脚本。
4. 在完成某个任务后请及时更新plan.md与本文件，确保它们与当前代码状态一致。

## Project Overview

This is a Flow Matching implementation for a graduate machine learning course (Project 3). The project implements the paper "Flow Matching for Generative Modeling" (ICLR 2023), focusing on reproducing key experiments on 2D toy data and CIFAR-10.

**Current Status**: Phase 3 complete. **CIFAR-10 training infrastructure is ready.** Implemented U-Net architecture, data loader, and training script (`src/train_cifar.py`). Verified correct execution on CUDA device.

## Environment Management

This project uses **`uv`** as the package manager (not pip/venv).

### Essential Commands

```bash
# Install/update dependencies
uv sync

# Run Python scripts (ALWAYS use uv run)
uv run python src/train_toy.py --path OT --epochs 5000
# OR simply
uv run src/train_toy.py --path OT --epochs 5000

# Install new package
# add to pyproject.toml first, then:
uv sync

```

**⚠️ CRITICAL**: Never use `python` directly. Always use `uv run` or scripts will fail to find installed packages.

## Code Architecture

The codebase follows a **flat structure** (max 2 levels deep) in `src/`:

### Core Components

1. **`src/cfm.py`** - Conditional Flow Matching algorithm
   - `ConditionalFlowMatching` class with `compute_loss(x1)` method
   - Supports OT (Optimal Transport), VP (Variance Preserving), VE paths
   - OT path uses straight-line trajectories: `ψ_t(x0) = (1-(1-σ_min)t)x0 + t*x1`
   - Fixed: Explicit formulas in `ConditionalFlowMatchingV2` now match `paths.py`.

2. **`src/paths.py`** - Probability path implementations
   - `get_conditional_path(path_type, sigma_min)` returns path function
   - Computes both intermediate state `ψ_t` and target vector field `u_t`
   - **Fixed VP**: `ψ_t = sin(t*π/2)x1 + cos(t*π/2)x0`, `u_t = π/2(cos(t*π/2)x1 - sin(t*π/2)x0)`
   - **Fixed VE**: Corrected interpolation and log-derivative target.

3. **`src/models.py`** - Neural network architectures
   - `MLPVectorField` - 5-layer MLP (512 hidden units) for 2D data
   - Includes sinusoidal time embedding
   - Input: (x, y) + time(t), Output: (vx, vy)

4. **`src/solver.py`** - ODE solvers for sampling
   - `euler_solver(model, num_samples, num_steps)` - Fast sampling
   - `rk4_solver` - Higher quality
   - Forward ODE: `dx/dt = v_t(x)`, where `x(0) ~ N(0,I)` → `x(1) ~ p_data`

5. **`src/data.py`** - Data loading
   - `get_toy_dataloader()` - 2D toy datasets (checkerboard, etc.)
   - Note: CIFAR-10 loader NOT yet implemented (Phase 3)

6. **`src/visualize.py`** - Visualization utilities
   - `plot_vector_field()` - Quiver plot of vector field at different times
   - `plot_trajectories()` - Generation paths from noise to data
   - `compare_paths()` - Side-by-side OT vs VP comparison

7. **`src/train_toy.py`** - 2D training script
   - Usage: `uv run src/train_toy.py --path OT --epochs 5000`
   - Supports hyperparameter configuration via CLI args
   - Generates visualizations automatically

### What's Implemented (Phase 1-3: ✅ Complete)

- CFM core algorithm with OT/VP/VE paths (Mathematically Verified)
- MLP vector field network for 2D data
- 2D toy data generator (checkerboard)
- ODE solvers (Euler, RK4)
- Training script with visualization for 2D data
- Vector field and trajectory plotting
- **U-Net architecture for CIFAR-10**
- **CIFAR-10 data loader**
- **CIFAR-10 training script (`src/train_cifar.py`)**

### What's NOT Yet Implemented (Phase 4-7: ⏳ Pending)

- CIFAR-10 Full Training (Phase 4)
- FID/NLL evaluation metrics
- NFE sweep/ablation study

## Running Experiments

### 2D Toy Data (Phase 2 - Working)

```bash
# Train OT path model
uv run src/train_toy.py --path OT --epochs 1000 --batch_size 256

# Train VP path model (for comparison)
uv run src/train_toy.py --path VP --epochs 1000

# Custom hyperparameters
uv run src/train_toy.py --path OT --hidden_dim 1024 --num_layers 7 --lr 5e-4
```

Outputs saved to:
- Checkpoints: `results/toy/checkpoints/{OT,VP}/model_final.pt`
- Figures: `report/figures/toy/` (vector fields, trajectories, training curves)

### CIFAR-10 (Phase 4 - Not Ready)

Not yet implemented. U-Net and CIFAR-10 training script coming in Phase 3.

## Key Implementation Details

### OT Path vs VP Path

**OT Path** (Optimal Transport - default):
- Straight-line trajectories from noise to data
- Constant target vector field: `u_t = x1 - (1-σ_min)x0`
- More efficient: requires fewer NFE (Neural Function Evaluations)
- This is the key innovation of Flow Matching paper

**VP Path** (Variance Preserving):
- Curved trajectories like DDPM
- Time-varying target based on exact time derivative of the path
- Used in paper for ablation studies
- Requires more NFE to achieve same quality

### Loss Function

CFM loss (Equation 23 from paper):
```python
L_CFM = E[t, x0, x1] ||v_t(ψ_t(x0)) - u_t||^2
```

where:
- `t ~ Uniform[0,1]`
- `x0 ~ N(0,I)` (noise)
- `x1` is data sample
- `ψ_t(x0)` is the conditional path from x0 to x1
- `u_t` is the target vector field (Time derivative of ψ_t)
- `v_t` is the predicted vector field from neural network

### Model Input/Output Shapes

For 2D data:
- Input `x`: shape `(B, 2)` where B is batch size
- Input `t`: shape `(B, 1)` or `(B,)` (will be reshaped)
- Output `v`: shape `(B, 2)` - 2D vector field

For images (when U-Net implemented):
- Input `x`: shape `(B, C, H, W)` - e.g., `(B, 3, 32, 32)` for CIFAR-10
- Input `t`: shape `(B, 1, 1, 1)` (broadcast over spatial dims)
- Output `v`: shape `(B, C, H, W)` - RGB vector field

## Common Workflows

### Training a 2D Model

```bash
# Quick test (100 epochs)
uv run src/train_toy.py --path OT --epochs 100 --log_interval 10

# Full training (1000 epochs)
uv run src/train_toy.py --path OT --epochs 1000

# Train both paths for comparison
uv run src/train_toy.py --path OT --epochs 1000
uv run src/train_toy.py --path VP --epochs 1000
```

### Debugging Training Issues

If loss doesn't converge:
1. Check learning rate (try 5e-4 instead of 1e-3)
2. Verify data normalization (should be reasonable range)
3. Ensure model is in train mode: `model.train()`
4. Check gradient flow: `print(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))`

### Generating New Visualizations

```python
from src.visualize import plot_vector_field, plot_trajectories
from src.solver import euler_solver

# Load trained model
model = MLPVectorField()
model.load_state_dict(torch.load("results/toy/checkpoints/OT/model_final.pt"))
model.eval()

# Plot vector field
plot_vector_field(model, t_values=[0.0, 0.25, 0.5, 0.75, 1.0])

# Plot generation trajectories
plot_trajectories(model, euler_solver, n_samples=20, num_steps=100)
```

## Project Phases (from plan.md)

- ✅ **Phase 1**: Environment & Core Algorithm (Complete)
- ✅ **Phase 2**: 2D Toy Experiments (Complete - Fixed & Verified)
- ✅ **Phase 3**: CIFAR-10 U-Net Architecture (Complete - Verified)
- ⏳ **Phase 4**: CIFAR-10 Full Training (Pending)
- ⏳ **Phase 5**: NFE Efficiency Analysis (Pending)
- ⏳ **Phase 6**: Report Figures (Pending)
- ⏳ **Phase 7**: Report Writing (Pending)

## Success Criteria

Based on Project3.pdf requirements:

**Minimum** (passing):
- Implement OT path CFM ✅
- 2D checkerboard visualization ✅
- CIFAR-10 basic training (pending)
- Compute FID (pending)

**Target** (good quality):
- OT + VP dual path ✅
- FID < 8.0, NLL < 3.2 (pending)
- NFE analysis (pending)

**Excellent** (top tier):
- FID < 7.0, NLL < 3.1
- Complete NFE sweep
- Clear FID vs NFE curves
- Results aligned with paper

## Diagnostic Notes

- Pylance may show "unresolved import" warnings - this is normal
- The environment is managed by uv, and Pylance may not detect it immediately
- Code will run correctly with `uv run` despite these warnings
- All imports work when scripts are executed via `uv run`

## Reference Files

- `plan.md` - Detailed implementation plan (in Chinese)
- `Flow_Matching_Detailed_Analysis.md` - Paper analysis (in Chinese)
- `Project3.pdf` - Course assignment requirements
- `GEMINI.md` - This file