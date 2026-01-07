# CLAUDE.md


## Core Philosophy

1. 你的修改应该使得项目更易于理解和维护，而不是更复杂。需要进行改进时请直接在源代码中修改，而不是新建文件.
2. 你可以编写脚本临时测试功能，但是请新建一个temp/目录，将这些脚本放在那里，并在完成后删除它们。
3. 请尽量复用已有代码和函数，避免重复造轮子。
1. 在完成某个任务后请及时更新plan.md与本文件，确保它们与当前代码状态一致，不要新建总结文件。

## Project Overview

This is a Flow Matching implementation for a graduate machine learning course (Project 3). The project implements the paper "Flow Matching for Generative Modeling" (ICLR 2023), focusing on reproducing key experiments on 2D toy data and CIFAR-10.

**Current Status**: ⏳ **Phase 8 IN PROGRESS** (2026-01-07)
- ✅ 2D Toy: OT vs VP完整对比可视化
- ✅ CIFAR-10: OT和VP模型训练完成（1000 epochs）
- ✅ CIFAR-10: NFE评估完成（8个NFE值，2000样本）
- ✅ 初步报告完成
- ⏳ **正在进行**: DDPM完整实现和三方法对比

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
   - `plot_flow_evolution()` - Scatter plots of particle distribution over time
   - `compare_paths()` - Side-by-side OT vs VP comparison

7. **`src/train_toy.py`** - 2D training script
   - Usage: `uv run src/train_toy.py --path OT --epochs 5000`
   - Supports hyperparameter configuration via CLI args
   - Generates visualizations automatically

### What's Implemented (Phases 1-7: ✅ Complete)

**核心算法**：
- ✅ CFM core algorithm with OT/VP/VE paths
- ✅ DDPM core algorithm (`src/ddpm.py`)
- ✅ DDIM sampling with variable NFE (`src/ddpm_solver.py`)
- ✅ MLP and U-Net architectures (support both continuous and discrete time)
- ✅ ODE solvers (Euler, RK4)
- ✅ NLL/FID metrics (`src/metrics.py`)

**训练脚本**：
- ✅ 2D toy CFM training (`src/train_toy.py`)
- ✅ 2D toy DDPM training (`src/train_toy_ddpm.py`)
- ✅ CIFAR-10 CFM training (`src/train_cifar.py`) - 1000 epochs完成
- ⏳ CIFAR-10 DDPM training (`src/train_cifar_ddpm.py`) - 需创建

**评估与可视化**：
- ✅ NFE efficiency analysis (`src/nfe_sweep.py`) - OT vs VP对比
- ✅ Visualization utilities (`src/visualize.py`)
- ✅ Report figure generator (`src/generate_report_figures.py`)

**已完成实验**：
- ✅ 2D Toy: OT vs VP完整对比
- ✅ CIFAR-10: OT和VP模型训练（1000 epochs）
- ✅ CIFAR-10: NFE sweep评估（8个NFE值）

### What's Being Implemented (Phase 8: ⏳ In Progress)

- ⏳ CIFAR-10 DDPM training script creation
- ⏳ DDPM model training (2D + CIFAR-10)
- ⏳ Extend NFE sweep for three-way comparison (OT + VP + DDPM)
- ⏳ Three-way visualization generation
- ⏳ Documentation updates

## Running Experiments

### 2D Toy Data (✅ CFM Complete, ⏳ DDPM Pending)

**CFM Models**:
```bash
# Train CFM models
uv run src/train_toy.py --path OT --epochs 2000 --hidden_dim 512 --num_layers 5
uv run src/train_toy.py --path VP --epochs 2000 --hidden_dim 512 --num_layers 5
```

**DDPM Model** (Phase 8):
```bash
# Train DDPM model (30-60 min)
uv run src/train_toy_ddpm.py \
    --hidden_dim 128 \
    --num_layers 3 \
    --epochs 2000 \
    --dataset checkerboard \
    --output_dir results/toy/DDPM
```

**Generate figures**:
```bash
PYTHONIOENCODING=utf-8 uv run src/generate_report_figures.py --generate_2d
```

### CIFAR-10 (✅ CFM Complete, ⏳ DDPM In Progress)

**CFM Models** (已训练完成):
```bash
# OT: results/cifar10/OT/model_final_1000epoch.pt
# VP: results/cifar10/VP/model_final_1000epoch.pt
```

**DDPM Model** (Phase 8):
```bash
# Quick validation (100 epochs, ~45 min)
uv run src/train_cifar_ddpm.py \
    --epochs 100 \
    --output_dir results/cifar10/DDPM

# Full training (1000 epochs, ~6-8 hours)
uv run src/train_cifar_ddpm.py \
    --epochs 1000 \
    --beta_schedule linear \
    --output_dir results/cifar10/DDPM \
    --device cuda
```

### Three-Way Comparison (Phase 8)

**Run NFE sweep**:
```bash
# Two-way comparison (OT vs VP) - Already done ✅
uv run src/evaluate_models.py \
    --checkpoint_ot results/cifar10/OT/model_final_1000epoch.pt \
    --checkpoint_vp results/cifar10/VP/model_final_1000epoch.pt \
    --num_samples 2000

# Three-way comparison (OT + VP + DDPM) - Phase 8 ⏳
uv run src/nfe_sweep.py \
    --checkpoint_ot results/cifar10/OT/model_final_1000epoch.pt \
    --checkpoint_vp results/cifar10/VP/model_final_1000epoch.pt \
    --checkpoint_ddpm results/cifar10/DDPM/model_final_1000epoch.pt \
    --num_samples 2000 \
    --model_channels 32 \
    --output_dir results/cifar10/nfe_sweep_three_way
```

**Generate comparison figures**:
```bash
# Two-way figures
PYTHONIOENCODING=utf-8 uv run src/generate_report_figures.py --generate_cifar10

# Three-way figures (Phase 8)
PYTHONIOENCODING=utf-8 uv run src/generate_report_figures.py --generate_three_way
```

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

For images (CIFAR-10):
- Input `x`: shape `(B, 3, 32, 32)`
- Input `t`: shape `(B, 1, 1, 1)` (broadcast over spatial dims)
- Output `v`: shape `(B, 3, 32, 32)` - RGB vector field

### ⚠️ Important Model Configurations

**从checkpoint推断的实际配置**（与默认参数不同）：

**CIFAR-10 U-Net**:
```python
UNet(
    in_channels=3,
    out_channels=3,
    model_channels=32,  # ⚠️ 不是128!
    num_res_blocks=2,
    channel_mult=(1, 2, 2, 2),
    attention_resolutions=(2,),  # 在16x16分辨率
    dropout=0.1,
    num_heads=4
)
```

**2D Toy MLP**:
```python
MLPVectorField(
    hidden_dim=512,  # ⚠️ 不是128!
    num_layers=5      # ⚠️ 不是3!
)
```

**加载模型时必须使用这些配置，否则会导致参数不匹配错误！**

## Common Workflows

### 快速生成所有报告图表

```bash
# 1. 生成2D Toy对比图
PYTHONIOENCODING=utf-8 uv run src/generate_report_figures.py --generate_2d

# 2. 运行CIFAR-10评估（如未运行）
uv run src/evaluate_models.py \
    --checkpoint_ot results/cifar10/OT/model_final_1000epoch.pt \
    --checkpoint_vp results/cifar10/VP/model_final_1000epoch.pt \
    --num_samples 2000

# 3. 生成CIFAR-10对比图（评估完成后）
PYTHONIOENCODING=utf-8 uv run src/generate_report_figures.py --generate_cifar10

# 或一次性生成所有
PYTHONIOENCODING=utf-8 uv run src/generate_report_figures.py
```

### 训练新模型

**2D Toy**:
```bash
uv run src/train_toy.py --path OT --epochs 2000 --hidden_dim 512 --num_layers 5
uv run src/train_toy.py --path VP --epochs 2000 --hidden_dim 512 --num_layers 5
```

**CIFAR-10**:
```bash
uv run src/train_cifar.py --path OT --epochs 1000 --model_channels 32
uv run src/train_cifar.py --path VP --epochs 1000 --model_channels 32
```

### Debugging Training Issues

If loss doesn't converge:
1. Check learning rate (try 5e-4 instead of 1e-3)
2. Verify data normalization (should be reasonable range)
3. Ensure model is in train mode: `model.train()`
4. Check gradient flow: `print(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))`
5. **Check model config matches checkpoint** (see Important Model Configurations above)

## Project Phases

- ✅ **Phase 1-2**: Core Algorithm & 2D Toy (Complete)
- ✅ **Phase 3**: CIFAR-10 U-Net Architecture (Complete)
- ✅ **Phase 4**: CIFAR-10 Full Training (Complete - 1000 epochs)
- ✅ **Phase 5**: NFE Efficiency Analysis (Complete - 8 NFE values)
- ✅ **Phase 6**: Comparison Experiments (Complete - OT vs VP)
- ✅ **Phase 7**: Initial Report (Complete)
- ⏳ **Phase 8**: DDPM vs Flow Matching (In Progress)
  - ⏳ 8.1: Train 2D DDPM model
  - ⏳ 8.2: Train CIFAR-10 DDPM model
  - ⏳ 8.3: Extend NFE sweep for DDPM
  - ⏳ 8.4: Run three-way NFE sweep
  - ⏳ 8.5: Generate three-way visualizations
  - ⏳ 8.6: Update documentation

## Success Criteria

Based on Project3.pdf requirements:

**Minimum** (passing, 60-70分):
- ✅ Implement OT path CFM
- ✅ 2D checkerboard visualization
- ✅ CIFAR-10 basic training
- ✅ Compute FID

**Target** (good quality, 70-85分):
- ✅ OT + VP dual path
- ✅ FID < 8.0, NLL < 3.2 (CFM models)
- ✅ NFE analysis (OT vs VP)
- ✅ Detailed visualizations

**Excellent** (top tier, 85-100分):
- ✅ All Target requirements
- ⏳ **DDPM baseline implementation**
- ⏳ **Three-way comparison (OT + VP + DDPM)**
- ✅ Complete NFE sweep
- ✅ Clear FID vs NFE curves
- ⏳ **Results aligned with paper Table 1**

## Diagnostic Notes

- Pylance may show "unresolved import" warnings - this is normal
- The environment is managed by uv, and Pylance may not detect it immediately
- Code will run correctly with `uv run` despite these warnings
- All imports work when scripts are executed via `uv run`

## Reference Files

- `plan.md` - Detailed implementation plan (in Chinese)
- `Flow_Matching_Detailed_Analysis.md` - Paper analysis (in Chinese)
- `Project3.pdf` - Course assignment requirements
- `CLAUDE.md` - This file