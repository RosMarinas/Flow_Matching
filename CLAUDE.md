# CLAUDE.md


## Core Philosophy

1. 你的修改应该使得项目更易于理解和维护，而不是更复杂。需要进行改进时请直接在源代码中修改，而不是新建文件.
2. 你可以编写脚本临时测试功能，但是请新建一个temp/目录，将这些脚本放在那里，并在完成后删除它们。
3. 请尽量复用已有代码和函数，避免重复造轮子。
1. 在完成某个任务后请及时更新plan.md与本文件，确保它们与当前代码状态一致，不要新建总结文件。

## Project Overview

This is a Flow Matching implementation for a graduate machine learning course (Project 3). The project implements the paper "Flow Matching for Generative Modeling" (ICLR 2023), focusing on reproducing key experiments on 2D toy data and CIFAR-10.

**Current Status**: Phase 6 进行中 - 对比实验实施。
- ✅ 2D Toy完整对比可视化已完成
- ✅ CIFAR-10训练完成（1000 epochs）
- ⏳ CIFAR-10 NFE评估运行中
- ⏳ 报告图表生成中

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

### What's Implemented (Phase 1-6: ✅ Complete)

**核心算法**：
- CFM core algorithm with OT/VP/VE paths (Mathematically Verified)
- MLP vector field network for 2D data
- U-Net architecture for CIFAR-10
- ODE solvers (Euler, RK4)
- NLL/FID metrics (`src/metrics.py`) - Memory efficient

**训练脚本**：
- 2D toy training (`src/train_toy.py`) - 自动生成可视化
- CIFAR-10 training (`src/train_cifar.py`) - 1000 epochs完成

**评估与可视化**：
- NFE efficiency analysis (`src/nfe_sweep.py`) - FID vs NFE曲线
- Model evaluation (`src/evaluate_models.py`) - 调用nfe_sweep
- Visualization utilities (`src/visualize.py`) - 包含所有对比函数
- Report figure generator (`src/generate_report_figures.py`) - 汇总所有图表

**已完成实验**：
- ✅ 2D Toy: OT vs VP完整对比（向量场、轨迹、流演化）
- ✅ CIFAR-10: OT和VP模型训练完成（1000 epochs）
- ⏳ CIFAR-10: NFE sweep评估运行中

### What's NOT Yet Implemented (Phase 7: ⏳ Pending)

- DDPM baseline implementation（可选，用于完整对比）
- Final report writing

## Running Experiments

### 2D Toy Data (✅ Complete)

```bash
# Train models
uv run src/train_toy.py --path OT --epochs 2000 --hidden_dim 512 --num_layers 5
uv run src/train_toy.py --path VP --epochs 2000 --hidden_dim 512 --num_layers 5

# Generate comparison figures
PYTHONIOENCODING=utf-8 uv run src/generate_report_figures.py --generate_2d
```

Outputs:
- Checkpoints: `results/toy/checkpoints/model_ot.pt`, `model_vp.pt`
- Figures: `report/figures/toy/` (OT vs VP对比图)

### CIFAR-10 (⏳ 评估中)

**训练已完成**：
```bash
# Models already trained (1000 epochs)
# OT: results/cifar10/OT/model_final_1000epoch.pt
# VP: results/cifar10/VP/model_final_1000epoch.pt
```

**运行评估**：
```bash
# Quick evaluation (2000 samples, ~30 min)
uv run src/evaluate_models.py \
    --checkpoint_ot results/cifar10/OT/model_final_1000epoch.pt \
    --checkpoint_vp results/cifar10/VP/model_final_1000epoch.pt \
    --num_samples 2000

# Full evaluation (10000 samples, ~2-3 hours)
uv run src/evaluate_models.py \
    --checkpoint_ot results/cifar10/OT/model_final_1000epoch.pt \
    --checkpoint_vp results/cifar10/VP/model_final_1000epoch.pt \
    --num_samples 10000
```

**生成对比图表**：
```bash
# After evaluation completes
PYTHONIOENCODING=utf-8 uv run src/generate_report_figures.py --generate_cifar10

# Or generate all figures at once
PYTHONIOENCODING=utf-8 uv run src/generate_report_figures.py
```

Outputs:
- NFE results: `results/cifar10/nfe_sweep/nfe_comparison_results.json`
- Figures: `report/figures/cifar10/` (FID vs NFE, sample comparison)
- Tables: `report/tables/table1_comparison.md`

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

- ✅ **Phase 1**: Environment & Core Algorithm (Complete)
- ✅ **Phase 2**: 2D Toy Experiments (Complete)
- ✅ **Phase 3**: CIFAR-10 U-Net Architecture (Complete)
- ✅ **Phase 4**: CIFAR-10 Full Training (Complete - 1000 epochs)
- ✅ **Phase 5**: NFE Efficiency Analysis Script (Complete)
- 🔄 **Phase 6**: 对比实验实施 (进行中)
  - ✅ 2D Toy完整对比可视化
  - ⏳ CIFAR-10 NFE评估（运行中）
  - ⏳ 报告图表生成（待评估完成）
- ⏳ **Phase 7**: 报告撰写 (Pending)

## Success Criteria

Based on Project3.pdf requirements:

**Minimum** (passing):
- ✅ Implement OT path CFM
- ✅ 2D checkerboard visualization
- ✅ CIFAR-10 basic training
- ⏳ Compute FID (运行中)

**Target** (good quality):
- ✅ OT + VP dual path
- ⏳ FID < 8.0, NLL < 3.2 (待评估结果)
- ✅ NFE analysis (脚本就绪)

**Excellent** (top tier):
- ⏳ FID < 7.0, NLL < 3.1
- ✅ Complete NFE sweep
- ✅ Clear FID vs NFE curves
- ⏳ Results aligned with paper

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