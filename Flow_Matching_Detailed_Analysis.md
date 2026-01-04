# Flow Matching for Generative Modeling: 详细数学推导报告

> **作者**: Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, Matt Le
> **会议**: ICLR 2023
> **机构**: Meta AI (FAIR), Weizmann Institute of Science
> **报告生成日期**: 2026-01-03

---

## 📋 目录

1. [核心思想与动机](#1-核心思想与动机)
2. [预备知识：连续归一化流(CNF)](#2-预备知识连续归一化流cnf)
3. [Flow Matching目标函数](#3-flow-matching目标函数)
4. [条件Flow Matching (关键推导)](#4-条件flow-matching-关键推导)
5. [高斯条件概率路径族](#5-高斯条件概率路径族)
6. [特殊实例：扩散路径与OT路径](#6-特殊实例扩散路径与ot路径)
7. [与Diffusion Model的关系](#7-与diffusion-model的关系)
8. [算法实现与实验](#8-算法实现与实验)
9. [总结与展望](#9-总结与展望)

---

## 1. 核心思想与动机

### 1.1 研究背景

**Diffusion Models的局限性**:
- 训练稳定,但采样路径受限于扩散过程
- 需要大量采样步骤才能获得高质量样本
- 训练时间长(需要百万级迭代)

**连续归一化流(CNF)的潜力**:
- 可以建模任意概率路径
- 理论上更通用,但训练困难
- 最大似然训练需要昂贵的ODE模拟

### 1.2 Flow Matching的核心创新

> **Flow Matching (FM)**: 一种**仿真自由**的训练CNF方法,通过回归向量场来匹配固定的条件概率路径

**三大优势**:
1. **通用性**: 兼容高斯条件概率路径族,包含扩散路径作为特例
2. **高效性**: 无需ODE模拟即可训练,直接优化向量场
3. **灵活性**: 可以设计非扩散的概率路径(如OT路径),提升性能

---

## 2. 预备知识：连续归一化流(CNF)

### 2.1 基本定义

**概率密度路径** (Probability Density Path):
$$p: [0,1] \times \mathbb{R}^d \to \mathbb{R}_{>0}$$
- 时间相关的概率密度函数
- 满足归一化条件: $\int p_t(x) dx = 1$

**向量场** (Vector Field):
$$v: [0,1] \times \mathbb{R}^d \to \mathbb{R}^d$$
- 时间相关的向量场,用于定义流

**流** (Flow): $\phi: [0,1] \times \mathbb{R}^d \to \mathbb{R}^d$

通过ODE定义:
$$\frac{d}{dt}\phi_t(x) = v_t(\phi_t(x)) \tag{1}$$
$$\phi_0(x) = x \tag{2}$$

### 2.2 Push-Forward操作

**核心**: 用流将简单分布 $p_0$ 转换为复杂分布 $p_1$

$$p_t = [\phi_t]_* p_0 \tag{3}$$

其中 push-forward 算子定义为:
$$[\phi_t]_* p_0(x) = p_0(\phi_t^{-1}(x)) \det \left[ \frac{\partial \phi_t^{-1}}{\partial x}(x) \right] \tag{4}$$

**直观解释**:
- $\phi_t^{-1}(x)$: 找到位置 $x$ 在 $t=0$ 时的源头
- $p_0(\phi_t^{-1}(x))$: 该源头的概率密度
- $\det[\frac{\partial \phi_t^{-1}}{\partial x}]$: 体积变换因子(Jacobian行列式)

### 2.3 向量场生成概率路径

**定义**: 向量场 $v_t$ **生成**概率路径 $p_t$ 当且仅当流 $\phi_t$ 满足方程(3)

**检验工具 - 连续性方程**:
$$\frac{d}{dt}p_t(x) + \text{div}(p_t(x)v_t(x)) = 0 \tag{26}$$

其中散度算子:
$$\text{div} = \sum_{i=1}^d \frac{\partial}{\partial x_i}$$

**物理意义**: 概率质量守恒 - 概率的变化等于概率流的净流出

---

## 3. Flow Matching目标函数

### 3.1 目标设定

**假设**:
- 数据分布: $x_1 \sim q(x_1)$ (未知)
- 先验分布: $p_0 = \mathcal{N}(0, I)$ (标准高斯)
- 目标分布: $p_1 \approx q(x_1)$

**目标概率路径** $p_t$: 从 $p_0$ 到 $p_1$ 的连续变换

**目标向量场** $u_t$: 生成 $p_t$ 的向量场

### 3.2 Flow Matching损失

$$\mathcal{L}_{\text{FM}}(\theta) = \mathbb{E}_{t, p_t(x)} \| v_t(x) - u_t(x) \|^2 \tag{5}$$

**其中**:
- $t \sim \mathcal{U}[0,1]$: 时间均匀分布
- $x \sim p_t(x)$: 从边际路径采样
- $v_t(x; \theta)$: 神经网络参数化的向量场
- $u_t(x)$: 目标向量场

**直观理解**:
- 这是一个**回归问题**,回归目标向量场 $u_t$
- 当损失为0时,神经网络 $v_t$ 学会生成 $p_t$

### 3.3 核心挑战

**问题**: 无法直接计算 $\mathcal{L}_{\text{FM}}$

**原因**:
1. 边际概率路径 $p_t(x)$ 未知(包含难处理的积分)
2. 目标向量场 $u_t(x)$ 未知(同样包含难处理的积分)

**解决方案**: 使用**条件** (conditional) 构造来绕过这些困难!

---

## 4. 条件Flow Matching (关键推导)

### 4.1 条件概率路径

**定义**: 对每个数据样本 $x_1$, 定义条件路径 $p_t(x|x_1)$

**边界条件**:
- $t=0$: $p_0(x|x_1) = p(x) = \mathcal{N}(0, I)$ (噪声)
- $t=1$: $p_1(x|x_1) = \mathcal{N}(x_1, \sigma^2 I)$ (集中在数据)

### 4.2 边际化构造

**边际概率路径**:
$$p_t(x) = \int p_t(x|x_1)q(x_1)dx_1 \tag{6}$$

**直观解释**:
- $p_t(x)$ 是所有条件路径 $p_t(x|x_1)$ 的混合
- 权重由数据分布 $q(x_1)$ 决定

**在 $t=1$ 时**:
$$p_1(x) = \int p_1(x|x_1)q(x_1)dx_1 \approx q(x_1) \tag{7}$$

当 $\sigma \to 0$, $p_1(x)$ 无限接近数据分布 $q(x_1)$

### 4.3 边际向量场

**关键定义**:
$$u_t(x) = \int u_t(x|x_1) \frac{p_t(x|x_1)q(x_1)}{p_t(x)} dx_1 \tag{8}$$

**直观理解**:
- $u_t(x)$ 是所有条件向量场 $u_t(x|x_1)$ 的加权平均
- 权重正比于该条件路径在位置 $x$ 的概率贡献

### 4.4 定理1: 边际向量场生成边际路径

**定理陈述**:
若条件向量场 $u_t(x|x_1)$ 生成条件路径 $p_t(x|x_1)$, 则边际向量场 $u_t(x)$ (方程8) 生成边际路径 $p_t(x)$ (方程6)

**证明** (使用连续性方程):

$$\begin{aligned}
\frac{d}{dt}p_t(x) &= \int \left(\frac{d}{dt}p_t(x|x_1)\right)q(x_1)dx_1 \\
&= -\int \text{div}\left(u_t(x|x_1)p_t(x|x_1)\right)q(x_1)dx_1 \\
&= -\text{div}\left(\int u_t(x|x_1)p_t(x|x_1)q(x_1)dx_1\right) \\
&= -\text{div}\left(u_t(x)p_t(x)\right)
\end{aligned}$$

**关键步骤解释**:
1. 交换导数和积分 (Leibniz规则)
2. 对每个条件路径应用连续性方程
3. 将散度提到积分外面
4. 使用方程(8)的定义

**结论**: $u_t$ 和 $p_t$ 满足连续性方程,因此 $u_t$ 生成 $p_t$

### 4.5 条件Flow Matching损失

$$\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t,q(x_1),p_t(x|x_1)} \| v_t(x) - u_t(x|x_1) \|^2 \tag{9}$$

**关键优势**:
- 只需要条件路径 $p_t(x|x_1)$ (容易设计!)
- 只需要条件向量场 $u_t(x|x_1)$ (容易计算!)
- 无需边际量 $p_t(x)$ 和 $u_t(x)$

### 4.6 定理2: FM和CFM的梯度等价性

**定理陈述**:
$$\nabla_\theta \mathcal{L}_{\text{FM}}(\theta) = \nabla_\theta \mathcal{L}_{\text{CFM}}(\theta)$$

**证明核心思路**:

**Step 1**: 展开范数平方
$$\begin{aligned}
\|v_t(x) - u_t(x)\|^2 &= \|v_t(x)\|^2 - 2\langle v_t(x), u_t(x)\rangle + \|u_t(x)\|^2 \\
\|v_t(x) - u_t(x|x_1)\|^2 &= \|v_t(x)\|^2 - 2\langle v_t(x), u_t(x|x_1)\rangle + \|u_t(x|x_1)\|^2
\end{aligned}$$

**Step 2**: 利用边际化和条件期望的等价性

$$\begin{aligned}
\mathbb{E}_{p_t(x)} \|v_t(x)\|^2 &= \int \|v_t(x)\|^2 p_t(x) dx \\
&= \int \|v_t(x)\|^2 p_t(x|x_1) q(x_1) dx_1 dx \\
&= \mathbb{E}_{q(x_1), p_t(x|x_1)} \|v_t(x)\|^2
\end{aligned}$$

**Step 3**: 交叉项的等价性

$$\begin{aligned}
\mathbb{E}_{p_t(x)} \langle v_t(x), u_t(x) \rangle &= \int \left\langle v_t(x), \frac{\int u_t(x|x_1) p_t(x|x_1) q(x_1) dx_1}{p_t(x)} \right\rangle p_t(x) dx \\
&= \int \left\langle v_t(x), u_t(x|x_1) \right\rangle p_t(x|x_1) q(x_1) dx_1 dx \\
&= \mathbb{E}_{q(x_1), p_t(x|x_1)} \left\langle v_t(x), u_t(x|x_1) \right\rangle
\end{aligned}$$

**结论**: 除独立于 $\theta$ 的常数项外, $\mathcal{L}_{\text{CFM}} = \mathcal{L}_{\text{FM}}$

---

## 5. 高斯条件概率路径族

### 5.1 一般形式

$$p_t(x|x_1) = \mathcal{N}(x \mid \mu_t(x_1), \sigma_t(x_1)^2 I) \tag{10}$$

**边界条件**:
- $t=0$: $\mu_0(x_1) = 0$, $\sigma_0(x_1) = 1$
- $t=1$: $\mu_1(x_1) = x_1$, $\sigma_1(x_1) = \sigma_{\min} \approx 0$

### 5.2 条件流

**定义**:
$$\psi_t(x) = \sigma_t(x_1)x + \mu_t(x_1) \tag{11}$$

**直观解释**: 简单的**仿射变换**
- 先缩放噪声 $x$: $\sigma_t(x_1)x$
- 再平移: $+ \mu_t(x_1)$

**验证**:
$$[\psi_t]_* p_0(x) = p_t(x|x_1)$$

因为如果 $x \sim \mathcal{N}(0, I)$, 则 $\psi_t(x) \sim \mathcal{N}(\mu_t(x_1), \sigma_t(x_1)^2 I)$

### 5.3 定理3: 高斯路径的向量场

**定理陈述**:
给定条件流(11),生成它的唯一向量场为:

$$u_t(x|x_1) = \frac{\sigma_t'(x_1)}{\sigma_t(x_1)} \left( x - \mu_t(x_1) \right) + \mu_t'(x_1) \tag{15}$$

**证明**:

**Step 1**: 从流方程出发
$$\frac{d}{dt}\psi_t(x) = u_t(\psi_t(x)|x_1)$$

**Step 2**: 对 $\psi_t$ 求导
$$\psi_t'(x) = \sigma_t'(x_1)x + \mu_t'(x_1)$$

**Step 3**: 反演流得到逆变换
$$\psi_t^{-1}(y) = \frac{y - \mu_t(x_1)}{\sigma_t(x_1)}$$

**Step 4**: 代入流方程
$$\begin{aligned}
\psi_t'(\psi_t^{-1}(y)) &= u_t(y|x_1) \\
\sigma_t'(x_1)\frac{y - \mu_t(x_1)}{\sigma_t(x_1)} + \mu_t'(x_1) &= u_t(y|x_1) \\
\frac{\sigma_t'(x_1)}{\sigma_t(x_1)} (y - \mu_t(x_1)) + \mu_t'(x_1) &= u_t(y|x_1)
\end{aligned}$$

**物理意义**:
- $\frac{\sigma_t'(x_1)}{\sigma_t(x_1)} (x - \mu_t(x_1))$: 缩放变化引起的收缩/膨胀
- $\mu_t'(x_1)$: 平移变化引起的漂移

### 5.4 重参数化后的CFM损失

从 $x_0 \sim p_0 = \mathcal{N}(0, I)$ 采样:
$$x = \psi_t(x_0) = \sigma_t(x_1)x_0 + \mu_t(x_1)$$

CFM损失变为:
$$\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t,q(x_1),p(x_0)} \left\| v_t(\psi_t(x_0)) - \frac{d}{dt} \psi_t(x_0) \right\|^2 \tag{14}$$

其中:
$$\frac{d}{dt} \psi_t(x_0) = x_1 - (1-\sigma_{\min})x_0$$

---

## 6. 特殊实例：扩散路径与OT路径

### 6.1 扩散路径 (Diffusion Paths)

#### 类型I: Variance Exploding (VE) 路径

**条件路径**:
$$p_t(x|x_1) = \mathcal{N}(x|x_1, \sigma_{1-t}^2 I) \tag{16}$$

**参数**:
- $\mu_t(x_1) = x_1$
- $\sigma_t(x_1) = \sigma_{1-t}$

**条件向量场** (代入定理3):
$$u_t(x|x_1) = -\frac{\sigma'_{1-t}}{\sigma_{1-t}}(x - x_1) \tag{17}$$

**特点**:
- 均值保持不变,方差爆炸增长
- 对应DDPM中的噪声注入过程

#### 类型II: Variance Preserving (VP) 路径

**条件路径**:
$$p_t(x|x_1) = \mathcal{N}(x \mid \alpha_{1-t}x_1, (1 - \alpha_{1-t}^2)I) \tag{18}$$

其中:
$$\alpha_t = e^{-\frac{1}{2}T(t)}, \quad T(t) = \int_0^t \beta(s)ds$$

**参数**:
- $\mu_t(x_1) = \alpha_{1-t}x_1$
- $\sigma_t(x_1) = \sqrt{1 - \alpha_{1-t}^2}$

**条件向量场**:
$$u_t(x|x_1) = -\frac{T'(1-t)}{2} \left[ \frac{e^{-T(1-t)}x - e^{-\frac{1}{2}T(1-t)}x_1}{1 - e^{-T(1-t)}} \right] \tag{19}$$

**特点**:
- 均值和方差同时变化
- 对应DDPM中的标准扩散过程

### 6.2 最优传输路径 (Optimal Transport Paths)

#### 理论背景：位移插值 (Displacement Interpolation)

该路径不仅仅是简单的线性插值，它对应于两个高斯分布（$p_0$和$p_1$）之间的 **Wasserstein-2 最优传输 (Optimal Transport)** 解。
根据 McCann (1997) 的理论，这种线性插值被称为 **位移插值 (Displacement Interpolant)**：
$$p_t = [(1-t)\mathrm{id} + t\psi]_{\#} p_0$$
其中 $\psi$ 是将 $p_0$ 推向 $p_1$ 的最优传输映射。

#### 路径设计

**关键思想**: 让均值和标准差随时间 $t$ **线性**变化。

$$\mu_t(x_1) = tx_1, \quad \sigma_t(x_1) = 1 - (1 - \sigma_{\min})t \tag{20}$$

**直观解释**:
- **均值**: 从 $0$ 匀速移动到 $x_1$。
- **标准差**: 从 $1$ 匀速收缩到 $\sigma_{\min} \approx 0$。

#### 条件流与几何直观

$$\psi_t(x) = (1 - (1 - \sigma_{\min})t)x + tx_1 \tag{22}$$

**几何意义**: **直线轨迹 (Straight Line Trajectories)**
- 粒子在流场中沿直线运动，且速度恒定。
- **无回溯 (No backtracking)**: 相比于扩散路径可能产生的"过冲" (overshoot) 现象，OT路径始终指向目标，效率最高。

#### 条件向量场

$$u_t(x|x_1) = \frac{x_1 - (1 - \sigma_{\min})x}{1 - (1 - \sigma_{\min})t} \tag{21}$$

**特点**:
1. **方向恒定**: 对于给定的 $x$ 和 $x_1$，向量场的方向在时间上保持一致（虽然大小会变），这使得神经网络更容易拟合。
2. **简单性**: 形式简单，计算开销极低。
3. **ODE求解友好**: 由于轨迹是直线，ODE求解器可以用很大的步长（很少的步数 NFE）就能获得高精度的解。

#### OT路径的CFM损失

$$\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t,q(x_1),p(x_0)} \left\| v_t(\psi_t(x_0)) - \left( x_1 - (1 - \sigma_{\min}) x_0 \right) \right\|^2 \tag{23}$$

**实现关键**:
```python
# 采样时间t
t = torch.rand(batch_size, 1, 1, 1)

# 采样噪声和数据
x0 = torch.randn_like(x1)
x1 = data[indices]

# 计算目标点
sigma_min = 1e-4
target = x1 - (1 - sigma_min) * x0

# 前向传播
psi_t = (1 - (1 - sigma_min) * t) * x0 + t * x1
v_pred = model(psi_t, t)

# 损失
loss = F.mse_loss(v_pred, target)
```

### 6.3 扩散路径 vs OT路径对比

| 特性 | 扩散路径 (Diffusion) | OT路径 (Optimal Transport) |
|------|-------------------|-------------------------|
| **路径形状** | **曲线 (Curved)** | **直线 (Straight)** |
| **速度变化** | 非恒定，末端变化剧烈 | **恒定速度** |
| **几何直观** | 可能会"过冲"或迂回 | 最短路径，直达目标 |
| **向量场复杂性** | 较复杂，随时间变化大 | 简单，方向较稳定 |
| **ODE求解难度** | 难 (需较小步长/自适应步长) | 易 (大步长，甚至单步) |
| **NFE (采样步数)** | 高 (通常 > 100) | **低 (可 < 20)** |
| **训练收敛** | 较慢 | **更快** (简单回归任务) |

**论文核心发现**:
扩散路径的采样轨迹往往是弯曲的，导致ODE求解器需要更多步数来追踪曲线。而OT路径是直线，理论上甚至可以用 Euler 方法单步求解（虽然多步精度更高）。实验表明，在相同 NFE 下，OT 路径生成的样本质量显著优于扩散路径。

---

## 7. 与Diffusion Model的关系

### 7.1 Score Matching回顾

**Denoising Score Matching目标**:
$$\mathcal{L}_{\text{SM}}(\theta) = \mathbb{E}_{t,q(x_1),p_t(x|x_1)} \lambda(t) \|s_t(x) - \nabla \log p_t(x|x_1)\|^2$$

对于高斯路径 $p_t(x|x_1) = \mathcal{N}(x|\mu_t, \sigma_t^2 I)$:
$$\nabla \log p_t(x|x_1) = \frac{x - \mu_t(x_1)}{\sigma_t^2(x_1)}$$

### 7.2 从Score到Vector Field

**关系**:
$$v_t(x) = -\frac{T'(1-t)}{2} \left[ s_t(x) - x \right] \tag{46}$$

其中 $s_t(x) = \nabla \log p_t(x)$ 是score函数

**关键差异**:
- **Score Matching**: 回归**梯度场** $\nabla \log p_t(x)$
- **Flow Matching**: 回归**向量场** $u_t(x)$

### 7.3 Flow Matching的优势

1. **更稳定**: 直接回归向量场,避免score的数值不稳定
2. **更通用**: 可以使用任意概率路径,不限于扩散
3. **更高效**: OT路径提供更快训练和采样

---

## 8. 算法实现与实验

### 8.1 训练算法

```python
# 伪代码
def train_flow_matching(model, data_loader, num_epochs, path_type='OT'):
    for epoch in range(num_epochs):
        for x1 in data_loader:
            # Step 1: 采样时间
            t = torch.rand(batch_size, 1, 1, 1).uniform_(0, 1)

            # Step 2: 采样噪声
            x0 = torch.randn_like(x1)

            # Step 3: 构造条件路径
            if path_type == 'OT':
                sigma_min = 1e-4
                psi_t = (1 - (1 - sigma_min) * t) * x0 + t * x1
                target = x1 - (1 - sigma_min) * x0
            elif path_type == 'VP':
                alpha_t = torch.exp(-0.5 * T(t))
                sigma_t = torch.sqrt(1 - alpha_t**2)
                psi_t = alpha_t * x1 + sigma_t * x0
                target = (x1 - alpha_t * x1) / sigma_t

            # Step 4: 预测向量场
            v_pred = model(psi_t, t)

            # Step 5: 计算损失
            loss = F.mse_loss(v_pred, target)

            # Step 6: 反向传播
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
```

### 8.2 采样算法

```python
def sample(model, num_samples, steps=100):
    # 初始化噪声
    x = torch.randn(num_samples, channels, height, width)

    # ODE求解
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.tensor([i / steps] * num_samples)

        # 预测向量场
        v = model(x, t)

        # Euler步进
        x = x + dt * v

    return x
```

### 8.3 实验结果

#### CIFAR-10

| 方法 | NLL↓ | FID↓ | NFE↓ |
|------|------|------|------|
| DDPM | 3.12 | 7.48 | 274 |
| Score Matching | 3.16 | 19.94 | 242 |
| **FM w/ Diffusion** | 3.10 | 8.06 | 183 |
| **FM w/ OT** | **2.99** | **6.35** | **142** |

#### ImageNet-64×64

| 方法 | NLL↓ | FID↓ | NFE↓ |
|------|------|------|------|
| Score Matching | 3.56 | 5.68 | 178 |
| **FM w/ Diffusion** | 3.54 | 6.37 | 193 |
| **FM w/ OT** | **3.53** | **5.02** | **122** |

**关键发现**:
1. **训练效率**: 在 ImageNet-128 上，FM 仅需 500k 次迭代即可达到甚至超越 Dhariwal & Nichol (2021) 训练 4.36m 次迭代的效果（尽管 FM 模型稍大），整体图像吞吐量减少了约 33%。
2. **采样效率**: FM-OT 在极低的 NFE (Number of Function Evaluations) 下仍能保持较好的样本质量。例如在 NFE=100 左右即可获得高质量样本，而扩散模型通常需要更多步数。
3. **性能更优**: FM-OT 在 CIFAR-10 和 ImageNet 上 consistently 取得了比 Score Matching 和 DDPM 更低的 FID 和 NLL 分数。
4. **训练稳定性**: 相比于 Score Matching 中 score 函数数值的不稳定性，直接回归向量场更加稳定。

---

## 9. 总结与展望

### 9.1 核心贡献

1. **理论贡献**:
   - Flow Matching框架: 仿真自由的CNF训练
   - 条件路径构造: 避免难处理的边际量
   - 梯度等价性定理: CFM = FM (in gradient)

2. **方法贡献**:
   - OT路径: 线性插值,简单高效
   - 兼容扩散路径: 更稳定的训练替代方案

3. **实验贡献**:
   - ImageNet上SOTA性能
   - 训练速度+采样效率的双重提升

### 9.2 数学推导核心要点

1. **连续性方程**: 检验向量场生成概率路径的核心工具
2. **边际化技巧**: 从条件到边际的桥梁
3. **期望交换**: $\mathbb{E}_{p_t} = \mathbb{E}_{q(x_1)}\mathbb{E}_{p_t(\cdot|x_1)}$
4. **高斯路径**: 仿射变换 $\psi_t(x) = \sigma_t x + \mu_t$ 的简洁性

### 9.3 对你的Project的启发

**推荐改进方向**:

1. **新概率路径设计**:
   - Riemannian几何路径
   - 自适应路径(根据数据难度调整)
   - 分段路径(不同阶段用不同路径)

2. **条件生成扩展**:
   - Class-conditional FM
   - Text-to-image with FM
   - Super-resolution with FM

3. **理论分析**:
   - 为什么OT路径泛化更好?
   - 路径曲率与性能的关系
   - 收敛性分析

4. **效率优化**:
   - 自适应时间步长
   - 多分辨率训练
   - 混合精度训练

### 9.4 复现建议

**Week 1-2**: 基础复现
- [ ] 实现FM-OT在CIFAR-10
- [ ] 复现FID < 7.0
- [ ] 对比扩散路径

**Week 3-4**: 创新实验
- [ ] 设计新概率路径
- [ ] 条件生成实验
- [ ] 消融研究

**Week 5**: 论文撰写
- [ ] 理论推导
- [ ] 实验结果整理
- [ ] 可视化与对比

---

## 附录A: 关键公式速查

### A.1 基本定义

| 符号 | 含义 |
|------|------|
| $p_t(x)$ | 时间相关的概率密度 |
| $v_t(x)$ | 时间相关的向量场 |
| $\phi_t(x)$ | 流(由向量场生成) |
| $[\phi_t]_* p_0$ | Push-forward操作 |
| $u_t(x)$ | 目标向量场 |

### A.2 核心公式

1. **流方程**: $\frac{d}{dt}\phi_t(x) = v_t(\phi_t(x))$
2. **FM损失**: $\mathbb{E}_{t,p_t(x)} \|v_t(x) - u_t(x)\|^2$
3. **CFM损失**: $\mathbb{E}_{t,q(x_1),p_t(x|x_1)} \|v_t(x) - u_t(x|x_1)\|^2$
4. **连续性方程**: $\frac{d}{dt}p_t + \text{div}(p_t v_t) = 0$
5. **OT向量场**: $u_t(x|x_1) = \frac{x_1 - (1-\sigma_{\min})x}{1 - (1-\sigma_{\min})t}$

### A.3 高斯路径参数
---
| 路径类型 | $\mu_t(x_1)$ | $\sigma_t(x_1)$ | $u_t(x|x_1)$ |
|----|-------------------|---------------------------|---------------------------------------------------------------------------|
| OT | $tx_1$ | $1-(1-\sigma_{\min})t$ | $\frac{x_1-(1-\sigma_{\min})x}{1-(1-\sigma_{\min})t}$ |
| VE | $x_1$ | $\sigma_{1-t}$ | $-\frac{\sigma'_{1-t}}{\sigma_{1-t}}(x-x_1)$ |
| VP | $\alpha_{1-t}x_1$ | $\sqrt{1-\alpha_{1-t}^2}$ | $-\frac{T'(1-t)}{2}\frac{e^{-T(1-t)}x - e^{-T(1-t)/2}x_1}{1-e^{-T(1-t)}}$ |

---

**报告完成** 🎉

祝你的Project 3顺利! 如果有任何数学细节不清楚,随时问我!
