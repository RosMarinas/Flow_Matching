# Project3 报告：Flow Matching 实现与对比分析

## 📁 目录结构

```
paper_draft/
├── main.tex                 # 主LaTeX文件
├── references.bib           # 参考文献数据库
├── README.md               # 本文件
└── figures/
    ├── toy/                # 2D Toy实验图片
    │   ├── ddpm_evolution.png
    │   ├── flow_evolution_OT.png
    │   ├── flow_evolution_VP.png
    │   ├── ot_vs_vp_trajectories.png
    │   ├── vector_field_OT.png
    │   └── vector_field_VP.png
    └── cifar10/            # CIFAR-10实验图片
        ├── nfe_comparison_three_way.png  # 核心结果图
        ├── OT_epoch_1000.png
        ├── VP_epoch_1000.png
        └── DDPM_epoch_1000_nfe100.png
```

## 🚀 快速开始

### 1. 安装LaTeX

**macOS**:
```bash
# 安装MacTeX（推荐）
brew install --cask mactex

# 或下载安装包
# https://www.tug.org/mactex/
```

**Linux (Ubuntu/Debian)**:
```bash
sudo apt-get update
sudo apt-get install texlive-full
```

**Windows**:
```bash
# 下载MiKTeX或TeX Live
# https://miktex.org/download
# 或
# https://www.tug.org/texlive/
```

### 2. 编译报告

进入`paper_draft/`目录，运行以下命令：

```bash
cd paper_draft

# 方法1：使用pdflatex（推荐）
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex

# 方法2：使用xelatex（如果需要中文支持）
xelatex main.tex
bibtex main
xelatex main.tex
xelatex main.tex

# 方法3：使用latexmk（自动化）
latexmk -pdf main.tex
```

编译成功后，会生成`main.pdf`文件。

### 3. 查看报告

```bash
# macOS
open main.pdf

# Linux
xdg-open main.pdf

# Windows
start main.pdf
```

## 📊 报告内容

### 页数分配（总计8页）

| 章节 | 页数 | 内容 |
|------|------|------|
| 1. 引言 | 0.7页 | 论文背景、核心问题、复现工作内容 |
| 2. 相关工作 | 0.8页 | DDPM、Score模型、Flow Matching |
| 3. 方法 | 1.5页 | 算法模型、实现细节 |
| 4. 实验 | 4.0页 | **核心章节**，2D Toy + CIFAR-10 |
| 5. 结论 | 0.5页 | 总结、展望 |
| 参考文献 | 0.5页 | 核心论文引用 |

### 核心图表

1. **图1**：三种方法的粒子演化对比（三列并排）
2. **图2**：OT vs VP轨迹对比
3. **图3**：向量场对比（两列并排）
4. **图4**：CIFAR-10生成样本对比
5. **图5**：FID vs NFE曲线（核心结果图）

### 定量结果表

- **表1**：三种方法FID对比（不同NFE）
- **表2**：采样效率对比（加速比）
- **表3**：与论文结果对比

## 📝 关键发现

1. **OT路径在低NFE下显著优于VP和DDPM**（2.9x-6.7x加速）
2. **Flow Matching训练更简单**（无需复杂的beta schedule设计）
3. **相对趋势与论文一致**（OT < VP, FM优于DDPM在低NFE）
4. **向量场可视化验证了理论**（OT的直线路径vs VP的曲线路径）

## 🔧 常见问题

### 编译错误

**问题1：找不到图片文件**
```
Error: File `figures/toy/xxx.png` not found
```
**解决**：确保所有图片文件已复制到`figures/`目录，路径正确。

**问题2：参考文献引用错误**
```
Warning: Citation `xxx' undefined
```
**解决**：运行`bibtex main`，然后重新编译2次。

**问题3：表格格式错误**
```
Error: Misplaced alignment tab
```
**解决**：检查表格中的`\\`和`&`符号是否正确使用。

### 篇幅控制

**问题：报告超过8页**
**解决**：
1. 减少图表之间的空白：使用`\vspace{-2mm}`
2. 缩小图片：调整`\includegraphics[width=0.XX\textwidth]`
3. 使用双栏布局：`\begin{figure*}[H]`
4. 精简文字描述

## 📧 联系方式

如有问题，请联系：Your Name <your.email@example.com>

## 📄 许可证

本项目代码和报告遵循MIT License。

---

**最后更新**：2026-01-08
