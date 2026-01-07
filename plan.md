# Flow Matching 论文复现计划 - Project 3

**目标**: 完整复现 Flow Matching 论文核心实验，完成 DDPM vs Flow Matching 三方法对比

**当前状态**: ⏳ **正在进行 Phase 8: DDPM完整实现和对比**

---

## 当前进度

### 已完成阶段 ✅
- [x] Phase 1-2: 核心算法与2D玩具数据实验
- [x] Phase 3: CIFAR-10 U-Net架构
- [x] Phase 4: CIFAR-10完整训练（OT + VP, 1000 epochs）
- [x] Phase 5: NFE效率分析（OT vs VP，8个NFE值）
- [x] Phase 6: 对比实验（2D Toy + CIFAR-10可视化）
- [x] Phase 7: 初步报告撰写

### 当前阶段 ⏳
- [ ] **Phase 8: DDPM完整实现和三方法对比**
  - [ ] 8.1: 训练2D DDPM模型（2000 epochs）
  - [ ] 8.2: 训练CIFAR-10 DDPM模型（1000 epochs）
  - [ ] 8.3: 扩展NFE pipeline支持DDPM
  - [ ] 8.4: 运行三方法NFE sweep
  - [ ] 8.5: 生成综合对比可视化
  - [ ] 8.6: 更新文档

---

## Phase 8: DDPM完整实现和三方法对比

**目的**: 完整实现DDPM基线并与Flow Matching（OT + VP）进行全面NFE对比

### 已有代码
- ✅ `src/ddpm.py` - DDPM核心算法
- ✅ `src/ddpm_solver.py` - DDIM采样（支持variable NFE）
- ✅ `src/train_toy_ddpm.py` - 2D DDPM训练脚本
- ✅ `src/models.py` - 支持discrete time embedding

### 需要创建
- [ ] `src/train_cifar_ddpm.py` - CIFAR-10 DDPM训练脚本
- [ ] 扩展 `src/nfe_sweep.py` - 添加DDPM sweep支持
- [ ] 扩展 `src/visualize.py` - 添加三方法对比函数

### 实验步骤

#### 8.1 训练2D DDPM模型
```bash
uv run src/train_toy_ddpm.py \
    --hidden_dim 128 \
    --num_layers 3 \
    --epochs 2000 \
    --dataset checkerboard \
    --num_timesteps 1000 \
    --beta_schedule linear \
    --output_dir results/toy/DDPM
```

**输出**: `results/toy/DDPM/model_final.pt`

#### 8.2 训练CIFAR-10 DDPM模型
```bash
# 快速验证（100 epochs，~45分钟）
uv run src/train_cifar_ddpm.py --epochs 100 --output_dir results/cifar10/DDPM

# 完整训练（1000 epochs，~6-8小时）
uv run src/train_cifar_ddpm.py \
    --epochs 1000 \
    --beta_schedule linear \
    --output_dir results/cifar10/DDPM \
    --device cuda
```

**输出**: `results/cifar10/DDPM/model_final_1000epoch.pt`

#### 8.3 扩展NFE sweep
修改 `src/nfe_sweep.py`:
- 添加 `--checkpoint_ddpm` 参数
- 创建 `load_ddpm_model()` 函数
- 创建 `run_ddpm_sweep()` 函数（使用DDIM采样）
- 更新主函数支持三方法sweep

#### 8.4 运行三方法NFE sweep
```bash
uv run src/nfe_sweep.py \
    --checkpoint_ot results/cifar10/OT/model_final_1000epoch.pt \
    --checkpoint_vp results/cifar10/VP/model_final_1000epoch.pt \
    --checkpoint_ddpm results/cifar10/DDPM/model_final_1000epoch.pt \
    --num_samples 2000 \
    --model_channels 32 \
    --output_dir results/cifar10/nfe_sweep_three_way
```

**NFE值**: [10, 20, 30, 40, 50, 60, 80, 100]

**输出**:
- `results/cifar10/nfe_sweep_three_way/nfe_comparison_three_way.json`
- `results/cifar10/nfe_sweep_three_way/nfe_curve_three_way.png`

#### 8.5 生成综合对比可视化
修改 `src/generate_report_figures.py`:
- 添加 `generate_three_way_comparison()` 函数
- 生成FID vs NFE三曲线
- 生成样本对比网格（NFE=50, 100）
- 生成2D轨迹三方法对比
- 生成训练曲线对比

```bash
PYTHONIOENCODING=utf-8 uv run src/generate_report_figures.py --generate_three_way
```

### 预期结果（基于Flow Matching论文Table 1）

| Method | FID @ NFE=100 | NLL | Full NFE |
|--------|---------------|-----|----------|
| FM-OT  | ~6.35         | ~2.99 | 142      |
| FM-VP  | ~8.06         | ~3.10 | 183      |
| DDPM   | ~7-8          | ~3.12 | 274      |

**关键验证点**:
- FM-OT在低NFE下最优
- DDPM需要更多NFE达到相同质量
- 三方法清晰对比展示Flow Matching优势

---

## 文件清单

### 新建文件
- [ ] `src/train_cifar_ddpm.py` - CIFAR-10 DDPM训练脚本

### 修改文件
- [ ] `src/nfe_sweep.py` - 添加DDPM sweep支持
- [ ] `src/visualize.py` - 添加三方法对比函数
- [ ] `src/generate_report_figures.py` - 添加三方法对比生成
- [ ] `plan.md` - 本文档，更新进度
- [ ] `CLAUDE.md` - 更新使用说明

### 输出文件
- `results/toy/DDPM/model_final.pt`
- `results/cifar10/DDPM/model_final_1000epoch.pt`
- `results/cifar10/nfe_sweep_three_way/nfe_comparison_three_way.json`
- `report/figures/cifar10/fid_vs_nfe_three_way.png`
- `report/figures/toy/trajectories_three_way.png`
- `report/figures/cifar10/training_curves_three_way.png`

---

## 成功标准

### 优秀（85-100分）
- ✅ OT + VP双路径实现
- ✅ 2D Toy完整对比可视化
- ✅ CIFAR-10定量评估（FID, NLL）
- ⏳ **DDPM完整实现和对比**
- ✅ NFE效率分析（OT vs VP）
- ⏳ **三方法NFE对比（OT + VP + DDPM）**
- ✅ 清晰的FID vs NFE曲线
- ⏳ **结果与论文Table 1对齐**

### 当前目标
完成DDPM实现和三方法对比，达到优秀水平（85-90分）

---

## 时间估计

- **今天**: Phase 8.1-8.2（训练2D DDPM，创建CIFAR-10 DDPM脚本）
- **明天-后天**: Phase 8.2（CIFAR-10 DDPM训练，6-8小时）
- **第3-4天**: Phase 8.3-8.4（扩展pipeline + 运行NFE sweep）
- **第5天**: Phase 8.5-8.6（生成可视化 + 更新文档）

**总计**: ~4-5天（含训练等待时间）

---

## 关键配置

### 模型架构（必须匹配）

**CIFAR-10 U-Net**（CFM和DDPM）:
```python
UNet(
    model_channels=32,
    num_res_blocks=2,
    channel_mult=(1, 2, 2, 2),
    attention_resolutions=(2,),
    dropout=0.1,
    num_heads=4,
    use_discrete_time=True  # 仅DDPM需要
)
```

**2D MLP**:
- CFM: `hidden_dim=512, num_layers=5`
- DDPM: `hidden_dim=128, num_layers=3` (较小，训练更快)

### DDPM配置
- `num_timesteps=1000`
- `beta_schedule="linear"` (或"cosine")
- `beta_start=1e-4, beta_end=2e-2`
- `use_discrete_time=True` (critical!)

---

## 参考文件

- `CLAUDE.md` - 详细使用说明
- `Flow_Matching_Detailed_Analysis.md` - 论文分析
- `Project3.pdf` - 课程要求
- `Thesis/Flow_Matching_for_Generative_Modeling.md` - DDPM参考

---

**最后更新**: 2026-01-07
**状态**: 🟡 正在实施DDPM vs Flow Matching对比
