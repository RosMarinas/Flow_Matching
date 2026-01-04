# Flow Matching Reproduction To-Do List

为了更深入、全面地复现 "Flow Matching for Generative Modeling" (ICLR 2023)，我们将项目扩展到真实图像数据集 (CIFAR-10)，并对比不同的 Flow Matching 变体。

## 1. 基础架构升级 (Architecture & Infrastructure)
- [ ] **依赖更新**: 添加 `torchvision`, `torchmetrics` (用于 FID/IS), `clean-fid` 等库到 `pyproject.toml`。
- [ ] **U-Net 实现**: 实现一个适用于 CIFAR-10 (32x32) 的 U-Net 架构。
    - 需包含 Time Embedding (Gaussian Fourier Projection)。
    - (可选) Class Embedding 用于 Conditional Generation。
- [ ] **Dataset Loader**: 实现 CIFAR-10 的数据加载与预处理 (Normalize, Flip 等)。

## 2. 核心算法扩展 (Core Algorithm Expansion)
- [ ] **Flow Matching 变体**:
    - **OT-FM (Optimal Transport)**: 现有的，需适配图像数据。
    - **VP-FM (Variance Preserving)**: 复现类似扩散模型的路径 ($\mu_t = \cos(t)x_1, \sigma_t = \sin(t)$ 或类似形式)，用于对比。
    - **Target-FM**: 简单的线性插值路径 $\mu_t = tx_1 + (1-t)x_0, \sigma_t = \sigma_{min}$。
- [ ] **ODE Solver**:
    - 集成 `torchdiffeq` 的 `dopri5` (自适应步长) 与 `euler` (固定步长) 进行对比。

## 3. 训练与评估 (Training & Evaluation)
- [ ] **训练脚本重构 (`train.py`)**:
    - 支持命令行参数 (argparse): 选择模型、数据集、FM 类型、Batch Size 等。
    - 支持 Checkpoint 保存与恢复。
    - 记录 Loss 曲线。
- [ ] **评估脚本 (`eval.py`)**:
    - 计算 FID (Fréchet Inception Distance) 分数。
    - 生成 NFE (Number of Function Evaluations) vs FID 的分析图表。
    - 可视化生成的图片网格。

## 4. 实验计划 (Experiments)
1.  **Toy Data 对比**: 在 8-Gaussians 上对比 OT-FM, VP-FM, Target-FM 的收敛速度和轨迹形状 (复现论文 Figure 1/2)。
2.  **CIFAR-10 训练**: 在 CIFAR-10 上训练 OT-FM (预计需要较长时间，需优化代码效率)。
3.  **Solver 分析**: 对比 Euler 和 Dopri5 在不同 NFE 下的生成质量。

## 5. 报告撰写 (Reporting)
- [ ] 整理实验结果，绘制 Loss 曲线和 FID 对比图。
- [ ] 分析不同路径 (OT vs VP) 对训练效率的影响。
