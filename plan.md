# Flow Matching 论文复现计划 - Project 3

**目标**: 完整复现 Flow Matching 论文核心实验，满足 Project3.pdf 所有要求并追求高质量

**时间框架**: 3-4 周

**环境策略**:
- 本地 (MacBook Air): 2D toy 实验、代码开发、调试、可视化
- 服务器: CIFAR-10 完整训练、FID/NLL 评估、NFE sweep

---

## 当前进度

- [x] Phase 0: 项目初始化
- [x] Phase 1: 环境搭建与核心算法 ✅ **完成** (2026-01-04)
- [x] Phase 2: 2D 玩具数据实验 ✅ **完成** (2026-01-04)
- [x] Phase 3: CIFAR-10 U-Net 架构 ✅ **完成** (2026-01-04)
- [x] Phase 4: CIFAR-10 完整训练 (5-7 天) - **脚本已验证（含 Loss/NLL 可视化与记录），显存泄漏已修复**
- [x] Phase 5: NFE 效率分析 (2-3 天) - **脚本已实现并验证**
- [ ] Phase 6: 报告撰写 (2-3 天)

---

## 📁 项目结构（极简版，最多2层）

```
Flow_Matching/
├── 📄 CLAUDE.md                          # AI 助手使用指南
├── 📄 plan.md                            # 本文件：实施计划
├── 📄 pyproject.toml                     # 项目依赖
├── 📄 Project3.pdf                       # 大作业要求
├── 📄 Flow_Matching_Detailed_Analysis.md  # 论文分析
├── 📄 README.md                          # 项目说明
│
├── 📂 Thesis/                            # 📚 论文参考（保留）
│   ├── Flow Matching for Generative Modeling.pdf
│   └── ...（其他论文）
│
├── 📂 src/                               # 💻 所有运行代码（最多2层）
│   ├── cfm.py                            # CFM 核心算法
│   ├── paths.py                          # OT/VP 路径
│   ├── solver.py                         # ODE 求解器
│   ├── models.py                         # 神经网络（MLP + U-Net）
│   ├── data.py                           # 数据加载（2D + CIFAR-10）
│   ├── visualize.py                      # 可视化工具
│   ├── metrics.py                        # FID/NLL 计算
│   ├── train_toy.py                      # 2D 训练脚本
│   ├── train_cifar.py                    # CIFAR-10 训练脚本
│   ├── evaluate_cifar.py                 # 评估脚本
│   └── nfe_sweep.py                      # NFE 效率分析
│
├── 📂 models/                            # 🎯 预训练/下载的模型
│   └── （留空，用于存放下载的权重）
│
├── 📂 results/                           # 📊 所有实验结果 + 权重
│   ├── toy/                              # 2D 实验结果
│   │   ├── checkpoints/                  # 模型权重
│   │   └── figures/                      # 可视化图表
│   └── cifar10/                          # CIFAR-10 实验结果
│       ├── checkpoints/                  # 模型权重
│       └── figures/                      # 可视化图表
│
└── 📂 report/                            # 📝 报告编写
    ├── figures/                          # 报告用图表
    ├── diagrams/                         # 报告用示意图
    └── report.md                         # 报告正文
```

---s

## 📊 结构说明

### 根目录（项目文件）
- **CLAUDE.md**: AI 助手使用说明
- **plan.md**: 实施计划（本文档）
- **pyproject.toml**: 依赖管理
- **Project3.pdf**: 大作业要求
- **Flow_Matching_Detailed_Analysis.md**: 论文详细分析

### 核心目录（4个）

1. **`src/`** - 所有运行代码（扁平结构，最多2层）
2. **`models/`** - 下载的模型权重
3. **`results/`** - 实验结果（checkpoints + figures）
4. **`report/`** - 报告材料

### ✅ 已完成（Phase 1-3）
- `src/cfm.py` - CFM 核心算法（已修复 VP/VE 路径逻辑，经数值验证正确）
- `src/paths.py` - OT/VP/VE 路径（已通过数值微分验证，确保目标向量场为路径的真实时间导数）
- `src/solver.py` - ODE 求解器（Euler, RK4, dopri5）
- `src/models.py` - MLP（2D 网络）
- `src/data.py` - 2D 数据集
- `src/visualize.py` - 可视化工具
- `src/train_toy.py` - 2D 训练脚本（使用 `uv run` 执行）
- `src/models.py` - U-Net（CIFAR-10 网络）
- `src/data.py` - CIFAR-10 数据加载
- `src/train_cifar.py` - CIFAR-10 训练脚本（验证通过）

### ⏳ 待实现（Phase 4-6）
- `src/metrics.py` - FID/NLL 计算
- `src/evaluate_cifar.py` - 评估脚本
- `src/nfe_sweep.py` - NFE 效率分析

---

## 🎯 文件清单（按优先级）

| 文件路径 | 优先级 | 状态 | 说明 |
|---------|--------|------|------|
| `src/cfm.py` | 🔴 最高 | ✅ | CFM 核心 |
| `src/paths.py` | 🔴 最高 | ✅ | OT/VP 路径 |
| `src/solver.py` | 🔴 最高 | ✅ | ODE 求解器 |
| `src/models.py` | 🔴 最高 | 🟡 | MLP ✅, U-Net ⏳ |
| `src/data.py` | 🔴 最高 | 🟡 | 2D ✅, CIFAR-10 ⏳ |
| `src/visualize.py` | 🔴 最高 | ✅ | 可视化工具 |
| `src/train_toy.py` | 🔴 最高 | ✅ | 2D 训练 |
| `src/compare.py` | 🔴 最高 | ✅ | OT vs VP 对比训练 |
| `src/train_cifar.py` | 🟡 中 | ✅ | CIFAR-10 训练 |
| `src/metrics.py` | 🟡 中 | ✅ | FID/NLL |
| `src/nfe_sweep.py` | 🟢 低 | ✅ | NFE 分析 |

---
- [ ] 实现 CFM 核心算法（OT 路径 + VP 扩散路径）
- [ ] 模块化架构：U-Net (CIFAR-10) + MLP (2D toy)
- [ ] 训练稳定性保证（学习率调度、warmup、检查点）
- [ ] 代码可读性和可复现性

### 实验报告 (60分)
- [ ] 算法原理和数学推导（含 OT 路径构造）
- [ ] 2D 棋盘格可视化（向量场、生成轨迹）
- [ ] CIFAR-10 定量结果（**目标: FID < 7.0, NLL < 3.1**）
- [ ] FM-OT vs Diffusion 对比（NFE 分析）
- [ ] 消融实验和讨论

---

## Phase 1: 环境搭建与核心算法 (本地, 2-3 天)

### 任务清单
- [ ] 更新 `pyproject.toml` 添加依赖
  ```toml
  torch, torchvision, torchdiffeq, torch-fid,
  matplotlib, seaborn, tqdm, tensorboard,
  pyyaml, einops, scipy, jupyter
  ```

- [ ] 创建项目目录结构（所有 `__init__.py`）

- [ ] **实现核心 CFM 算法** (`src/flow_matching/cfm.py`)
  ```python
  class ConditionalFlowMatching:
      def compute_loss(self, x1):
          # 1. Sample t ~ U[0,1]
          # 2. Sample x0 ~ N(0,I)
          # 3. Compute ψ_t(x0) = (1-(1-σ_min)t)x0 + t*x1  (OT path)
          # 4. Target: x1 - (1-σ_min)x0
          # 5. Predict v_t(ψ_t)
          # 6. MSE loss
  ```

- [ ] **实现概率路径** (`src/flow_matching/paths.py`)
  - OT 路径（Optimal Transport，直线路径）
  - VP 路径（Variance Preserving 扩散，弯曲路径）

- [ ] 实现 MLP 向量场网络 (`src/models/mlp.py`)
  - 5 层，每层 512 维
  - 输入: (x, y, t)，输出: (vx, vy)

### 关键文件
1. `src/flow_matching/cfm.py` - 核心算法
2. `src/flow_matching/paths.py` - 路径实现
3. `src/models/mlp.py` - 2D 网络
4. `pyproject.toml` - 依赖配置

### 验收标准
- ✅ 所有依赖安装成功 (`uv sync`)
- ✅ CFM loss 计算正确（形状检查）
- ✅ MLP 可以训练

---

## Phase 2: 2D 玩具数据实验 (本地, 2-3 天) ✅ **已完成** (2026-01-04)

### 任务清单
- [x] 实现 2D 棋盘格数据生成 (`src/data.py`)
  - 棋盘格、高斯混合、螺旋数据集

- [x] 实现 2D 训练脚本 (`src/train_toy.py`)
  - 配置: batch_size=256, lr=1e-3, epochs=1000
  - 保存检查点和训练曲线
  - 支持自定义超参数（hidden_dim, num_layers, lr）

- [x] **实现可视化函数** (`src/visualize.py`)
  - `plot_vector_field(model, t_values)` - 向量场箭头图 ✅
  - `plot_trajectories(model, n_samples)` - 生成路径轨迹 ✅
  - `compare_paths(model_ot, model_vp, n_samples)` - OT vs VP 对比 ✅
  - 支持自定义轨迹数量（默认30条）
  - 添加背景数据分布可视化

- [x] **实现对比训练脚本** (`src/compare.py`)
  - 同时训练 OT 和 VP 两个模型
  - 生成训练曲线对比
  - 生成路径对比可视化

- [x] 修复关键 Bug
  - ODE 求解器时间范围：从 [0, 0.99] 修复为 [0.05, 1.0]
  - 修复相对导入问题（cfm.py, visualize.py, compare.py）

- [x] 运行训练并生成可视化
  - 训练 OT 和 VP 模型
  - 验证 OT 路径轨迹更直
  - 验证 VP 路径轨迹更弯曲

### 关键文件
1. `src/train_toy.py` - 单个路径训练脚本
2. `src/compare.py` - OT vs VP 对比训练脚本
3. `src/visualize.py` - 可视化核心（向量场、轨迹、对比）
4. `src/data.py` - 2D 数据生成
5. `src/solver.py` - ODE 求解器（Euler, RK4）

### 验收标准
- ✅ 模型收敛（loss < 0.01）
- ✅ 向量场图显示路径指向目标
- ✅ 轨迹图清晰展示路径差异
- ✅ 支持对比 OT vs VP 两种路径

### 实验结果
- Checkpoints: `results/toy/checkpoints/` (model_ot.pt, model_vp.pt)
- Figures: `report/figures/toy/` (训练曲线、向量场、轨迹对比)

---

## Phase 3: CIFAR-10 U-Net 架构 (本地开发, 2-3 天) ✅ **已完成** (2026-01-04)

### 任务清单
- [x] **实现 U-Net** (`src/models/unet.py`)
  - 基于 Dhariwal & Nichol (2021)
  - 配置: channels=256, depth=2, attention_res=[16]
  - GroupNorm + SiLU activation
  - Time embedding (sinusoidal)

- [x] 实现 CIFAR-10 数据加载 (`src/data/datasets.py`)
  - 标准化到 [-1, 1]
  - Data augmentation（可选）

- [x] 实现 ODE Solver (`src/flow_matching/solver.py`)
  - Euler method（快速采样）
  - RK4（中等质量）
  - dopri5（高质量，自适应步长）

- [x] 本地小规模测试
  - 训练 100 iterations 验证代码
  - 生成样本检查输出形状

### 关键文件
1. `src/models/unet.py` - U-Net 架构
2. `src/flow_matching/solver.py` - ODE 求解器
3. `src/data/datasets.py` - CIFAR-10 加载

### 验收标准
- ✅ U-Net 前向传播成功
- ✅ ODE solver 可以生成样本
- ✅ 代码可以在服务器运行

---

## Phase 4: CIFAR-10 完整训练 (服务器, 5-7 天)

### 任务清单
- [ ] 实现训练器 (`src/training/trainer.py`)
  - 支持 Adam optimizer + warmup + polynomial decay
  - TensorBoard logging
  - 自动检查点保存

- [ ] **实现 FID 计算** (`src/evaluation/metrics.py`)
  - 使用 `torch-fid` 库
  - 生成 50,000 样本
  - 与 CIFAR-10 测试集计算距离

- [ ] **实现 NLL 计算** (`src/evaluation/metrics.py`)
  - 反向 ODE: x1 → x0
  - Hutchinson trace estimator (K=5)
  - Uniform dequantization
  - 输出: bits/dim

- [ ] **服务器训练脚本** (`experiments/cifar10/train_cifar.py`)
  - 训练 2 个模型: FM-OT + FM-VP
  - 配置: batch_size=256, lr=5e-4, epochs=1000
  - 每 10 epochs 保存检查点

- [ ] 推送代码到服务器并运行训练
  - 使用 `screen` 或 `tmux` 管理长时间训练
  - 监控 TensorBoard

### 关键文件
1. `experiments/cifar10/train_cifar.py` - 主训练脚本
2. `src/evaluation/metrics.py` - FID/NLL 计算
3. `src/training/trainer.py` - 训练器

### 目标指标
- **FM-OT**: FID < 7.0, NLL < 3.1
- **FM-VP**: FID < 8.5, NLL < 3.2

### 验收标准
- ✅ 训练曲线收敛
- ✅ FID < 8.0
- ✅ NLL < 3.2

---

## Phase 5: NFE 效率分析 (服务器, 2-3 天)

### 任务清单
- [ ] **实现 NFE sweep** (`experiments/ablations/nfe_sweep.py`)
  ```python
  nfe_values = [10, 20, 50, 100, 200, 500]
  for nfe in nfe_values:
      samples = sample(model, num_steps=nfe, method='euler')
      fid = compute_fid(samples, ...)
      # Record (nfe, fid)
  ```

- [ ] 对比 OT vs VP 在不同 NFE 下的表现
  - 固定步长求解器 (Euler)
  - 记录每个 NFE 的 FID

- [ ] 生成 FID vs NFE 曲线图
  - **关键图表**: 横轴 NFE（对数尺度），纵轴 FID
  - OT 路径应在低 NFE 下保持较好性能

### 关键文件
1. `experiments/ablations/nfe_sweep.py` - NFE sweep 脚本
2. `src/flow_matching/solver.py` - 固定步长求解器

### 验收标准
- ✅ OT 路径在 NFE=10-50 时 FID 仍可接受
- ✅ VP 路径在低 NFE 下性能显著下降
- ✅ 生成清晰的对比图

---

## Phase 6: 报告图表生成 (本地, 2-3 天)

### 必需图表清单

**1. 算法原理 (10分)**
- [ ] CFM Loss 数学推导
- [ ] OT 路径构造示意图
- [ ] 架构流程图

**2. 2D Toy 实验 (15分)**
- [ ] 向量场可视化 (t=0, 0.5, 1.0，quiver plot)
- [ ] 生成轨迹图（10-20 条路径，颜色编码时间）
- [ ] OT vs VP 路径对比（并排展示）
- [ ] 概率密度演化（直方图，不同时间步）

**3. CIFAR-10 结果 (15分)**
- [ ] 生成样本网格 (16x16，256 张图片)
- [ ] 训练曲线（Loss vs Epoch）
- [ ] FID 和 NLL 数值表格
- [ ] 与论文结果对比

**4. NFE 分析 (10分)**
- [ ] **关键图表**: FID vs NFE 曲线（OT vs VP）
- [ ] 不同 NFE 下的样本质量对比（NFE=10, 50, 100, 500）
- [ ] 收敛速度对比（FID vs Training Epoch）

**5. 消融实验 (5分)**
- [ ] 路径类型对比
- [ ] 超参数敏感性（学习率、batch size）
- [ ] 训练稳定性分析

### 实现脚本
- [ ] 创建 `notebooks/generate_report_figures.ipynb`
  - 所有图表生成的统一脚本
  - 保存到 `reports/figures/`

### 关键文件
1. `notebooks/generate_report_figures.ipynb` - 图表生成
2. `src/evaluation/visualization.py` - 可视化函数库

---

## Phase 7: 报告撰写 (本地, 2-3 天)

### 报告结构（基于 Project3.pdf 要求）

**1. 算法原理和实现细节 (15分)**
- [ ] Flow Matching 数学推导
- [ ] OT 路径 vs VP 扩散路径
- [ ] 网络架构描述（MLP + U-Net）
- [ ] 训练策略（optimizer、schedule）

**2. 实验设置和可视化 (10分)**
- [ ] 超参数配置
- [ ] 2D toy 实验设置
- [ ] CIFAR-10 训练流程
- [ ] 评估指标说明（FID、NLL）

**3. CIFAR-10 结果分析 (20分)**
- [ ] 定量结果（FID、NLL 表格）
- [ ] 生成样本展示
- [ ] 与论文对比分析
- [ ] 训练过程分析

**4. 消融实验 (10分)**
- [ ] **核心**: OT vs VP NFE 对比
- [ ] 路径类型影响
- [ ] 采样步数影响
- [ ] 训练效率对比

**5. 讨论和改进 (5分)**
- [ ] FM 的优势（直线路径、低 NFE）
- [ ] 局限性和未来工作
- [ ] 对生成模型领域的启示

### 关键文件
1. `reports/Project3_Report.pdf` - 最终报告
2. `reports/figures/` - 所有图表

---

## 依赖管理

### `pyproject.toml` 配置
```toml
[project]
name = "flow-matching"
version = "0.1.0"
description = "Flow Matching for Generative Modeling - Project 3"
requires-python = ">=3.11"
dependencies = [
    "torch>=2.0.0",
    "torchvision>=0.15.0",
    "numpy>=1.24.0",
    "matplotlib>=3.7.0",
    "seaborn>=0.12.0",
    "tqdm>=4.65.0",
    "tensorboard>=2.13.0",
    "pillow>=9.5.0",
    "scipy>=1.10.0",
    "pyyaml>=6.0",
    "torchdiffeq>=0.2.3",
    "torch-fid>=0.3.0",
    "einops>=0.6.0",
    "click>=8.1.0",
    "wandb>=0.15.0",
    "jupyter>=1.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.3.0",
    "black>=23.0.0",
    "flake8>=6.0.0",
]
```

---

## 本地 vs 服务器工作流

### 本地开发（MacBook Air）
**适用场景**:
- 代码开发和调试
- 2D toy 实验（快速验证）
- 小规模 CIFAR-10 测试（100 iterations）
- 可视化和报告撰写

**操作**:
```bash
# 安装依赖
uv sync

# 运行本地实验
python experiments/toy_2d/train_toy.py --config config/toy_2d.yaml

# Jupyter 可视化
jupyter notebook notebooks/visualize_vector_fields.ipynb
```

### 服务器训练（远程 GPU）
**适用场景**:
- CIFAR-10 完整训练（1000 epochs）
- FID/NLL 计算（需要大量内存）
- NFE sweep（多次采样）

**工作流**:
```bash
# 1. 本地开发完成后推送
git add .
git commit -m "Implement CFM core and U-Net"
git push

# 2. 服务器拉取代码
ssh server
cd Flow_Matching
git pull
uv sync

# 3. 使用 screen 训练
screen -S fm_ot
python experiments/cifar10/train_cifar.py --path OT --gpu 0

# 4. 分离 screen: Ctrl+A, D
# 5. 重新连接: screen -r fm_ot

# 6. 训练完成后下载结果
rsync -avz server:~/Flow_Matching/results/ ./results/
```

---

## 成功标准（基于 Project3.pdf）

### 最低要求（及格，60-70分）
- [ ] 实现 OT 路径 CFM
- [ ] 2D 棋盘格可视化
- [ ] CIFAR-10 基础训练
- [ ] 计算 FID（可能不理想）
- [ ] 基础报告

### 完整要求（良好，70-85分）
- [ ] 所有最低要求
- [ ] OT + VP 双路径实现
- [ ] FID < 8.0, NLL < 3.2
- [ ] NFE 分析（至少 3 个点）
- [ ] 详细可视化

### 优秀要求（优秀，85-100分）
- [ ] 所有完整要求
- [ ] **FID < 7.0, NLL < 3.1**
- [ ] 完整 NFE sweep（6+ 个点）
- [ ] 清晰的 FID vs NFE 曲线
- [ ] 与论文结果对齐
- [ ] 深入的分析和讨论

### 评分对照表（论文 Table 1）
| 模型 | FID (越低越好) | NLL (越低越好) | NFE |
|------|---------------|----------------|-----|
| FM-OT (论文) | 6.35 | 2.99 | 142 |
| FM-VP (论文) | 8.06 | 3.10 | 183 |
| DDPM (论文)  | 7.48 | 3.12 | 274 |

**你的目标**: FM-OT FID < 7.0, 接近或优于论文

---

## 核心算法实现细节

### CFM Loss (论文 Equation 2/23)
```python
# src/flow_matching/cfm.py
class ConditionalFlowMatching:
    def __init__(self, model, path_type='OT', sigma_min=1e-4):
        self.model = model
        self.sigma_min = sigma_min

    def compute_loss(self, x1):
        """
        Compute CFM loss for batch x1.

        OT Path:
        - ψ_t(x0) = (1 - (1-σ_min)t)x0 + t*x1
        - Target: u_t = x1 - (1-σ_min)x0
        """
        batch_size = x1.shape[0]

        # 1. Sample time t ~ U[0,1]
        t = torch.rand(batch_size, 1, 1, 1)  # Shape for images

        # 2. Sample noise x0 ~ N(0,I)
        x0 = torch.randn_like(x1)

        # 3. Compute conditional path ψ_t(x0)
        sigma_t = 1 - (1 - self.sigma_min) * t
        psi_t = sigma_t * x0 + t * x1

        # 4. Target vector field
        target = x1 - (1 - self.sigma_min) * x0

        # 5. Predict vector field
        v_pred = self.model(psi_t, t)

        # 6. MSE loss
        loss = F.mse_loss(v_pred, target)
        return loss
```

### 采样流程
```python
# src/flow_matching/solver.py
def sample(model, num_samples, num_steps=100, method='euler'):
    """
    Generate samples from trained model.

    Forward ODE: dx/dt = v_t(x), x(0)=x0 ~ N(0,I), x(1) ~ p_data
    """
    x = torch.randn(num_samples, 3, 32, 32)
    dt = 1.0 / num_steps

    for i in range(num_steps):
        t = torch.ones(num_samples, 1, 1, 1) * (i / num_steps)

        if method == 'euler':
            v = model(x, t)
            x = x + dt * v
        elif method == 'rk4':
            k1 = model(x, t)
            k2 = model(x + 0.5*dt*k1, t + 0.5*dt)
            k3 = model(x + 0.5*dt*k2, t + 0.5*dt)
            k4 = model(x + dt*k3, t + dt)
            x = x + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

    return x
```

---

## 常见问题和解决方案

### Q1: 训练不稳定
**现象**: Loss 发散或 NaN
**解决**:
- 降低学习率（5e-4 → 1e-4）
- 增加 warmup steps
- 检查梯度裁剪（clip_grad_norm_）

### Q2: FID 计算内存不足
**解决**:
- 减少生成样本数（50k → 10k）
- 分批计算 FID
- 使用服务器更大内存

### Q3: NLL 计算太慢
**解决**:
- 减少重要性采样数（K=5 → K=1）
- 只在测试集子集计算
- 使用 Euler solver（更快但稍不精确）

### Q4: 采样质量差
**检查**:
- 模型是否充分训练（loss 是否收敛）
- 采样步数是否足够（至少 50 步）
- 数据归一化是否正确（[-1, 1] vs [0, 1]）

---

## 时间线（3-4 周）

| 周次 | 阶段 | 主要任务 | 预期输出 | 状态 |
|------|------|---------|---------|------|
| Week 1 | Phase 1-2 | 环境搭建、核心算法、2D 实验 | 2D 可视化图表 | ✅ 已完成 |
| Week 2 | Phase 3-4 | U-Net 实现、CIFAR-10 训练 | 训练好的模型 | ⏳ 待开始 |
| Week 3 | Phase 5-6 | NFE 分析、FID/NLL 计算 | 定量结果、对比图表 | ⏳ 待开始 |
| Week 4 | Phase 7 | 报告撰写、图表完善 | 完整报告 | ⏳ 待开始 |

---

## 关键文件优先级

### 🔴 高优先级（必须首先实现）
1. `src/flow_matching/cfm.py` - CFM 核心算法
2. `src/flow_matching/paths.py` - OT/VP 路径
3. `src/models/unet.py` - U-Net 架构
4. `src/evaluation/metrics.py` - FID/NLL 计算
5. `experiments/toy_2d/train_toy.py` - 2D 验证脚本

### 🟡 中优先级（核心功能）
6. `experiments/cifar10/train_cifar.py` - CIFAR-10 训练
7. `src/training/trainer.py` - 训练器
8. `src/evaluation/visualization.py` - 可视化
9. `experiments/ablations/nfe_sweep.py` - NFE 分析

### 🟢 低优先级（辅助功能）
10. `src/utils/logger.py` - 日志记录
11. `src/utils/checkpoint.py` - 检查点管理
12. `config/*.yaml` - 配置文件

---



**最后更新**: 2026-01-04
**状态**: 🟢 准备开始实施
