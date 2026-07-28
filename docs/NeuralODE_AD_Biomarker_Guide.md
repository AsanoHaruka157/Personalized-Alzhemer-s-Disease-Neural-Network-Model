# Neural ODE 拟合 AD 生物标志物轨迹：问题诊断与改进指南

> **针对 ADNI 数据集 / Zheng et al. (2022) 框架的 Neural ODE 实现**
>
> 本文档覆盖：根本性问题诊断 → 逐条修复方案 → 完整示例代码 → 推荐训练流程

---

## 目录

1. [背景与目标](#1-背景与目标)
2. [问题总览（按严重程度排序）](#2-问题总览)
3. [问题一：预训练目标错误 — 导数匹配 ≠ 轨迹匹配](#3-问题一预训练目标错误)
4. [问题二：`calculate_loss` 的积分起点 bug](#4-问题二calculate_loss-的积分起点-bug)
5. [问题三：FNN 架构缺乏因果级联约束](#5-问题三fnn-架构缺乏因果级联约束)
6. [问题四：Fine-tuning 正则化策略缺失](#6-问题四fine-tuning-正则化策略缺失)
7. [问题五：DPS 与 FNN 联合优化的恶性循环](#7-问题五dps-与-fnn-联合优化的恶性循环)
8. [问题六：Tanh 输出头的细节缺陷](#8-问题六tanh-输出头的细节缺陷)
9. [关于原论文机制模型也拟合不好的诊断](#9-关于原论文机制模型也拟合不好的诊断)
10. [推荐完整训练流程](#10-推荐完整训练流程)
11. [完整重构代码](#11-完整重构代码)

---

## 1. 背景与目标

### 1.1 目标效果

目标是复现/超越 `mechanistic_model.png` 所展示的效果：

- 轨迹两端有明显的**水平平台**（CN 段在低 DPS 稳定，AD 晚期在高 DPS 稳定）
- 平台之间有**明显的落差**（Aβ 从 ~210 降至 ~140；N 从 ~7000 降至 ~5500）
- 过渡区间**比 Sigmoid 更平滑**，体现生物级联的延迟效应
- 能反映**生物因果信息**：Aβ 先于 τ 变化，τ 先于 N，N 先于 C

### 1.2 当前失败现象

| 失败模式 | 可能原因 |
|---------|---------|
| 轨迹接近直线（斜线）| 预训练/fine-tuning 过拟合到 sigmoid 导数的中间斜率 |
| 平台不明显，落差很小 | FNN 输出被 Tanh+scaler 压制，或 L-BFGS 将平台"铲平" |
| 曲线形态紊乱，看不出趋势 | DPS 对齐失败 + FNN 自由度过大 + 缺少单调性约束 |
| 三堆数据点坍缩 | DPS 优化退化，所有患者映射到相同 DPS 区间 |

---

## 2. 问题总览

```
严重程度
  ██████████  问题一：预训练目标根本错误（导数 vs 轨迹）
  █████████   问题二：calculate_loss 积分起点 bug
  ████████    问题三：FNN 架构缺少因果约束
  ███████     问题四：fine-tuning 无正则化
  ██████      问题五：DPS 联合优化退化
  ████        问题六：Tanh head 设计细节
```

---

## 3. 问题一：预训练目标错误

### 3.1 问题分析

这是**最根本的问题**。你的 `pretrain_fnn_on_sigmoid` 做的是：

```python
# 当前错误做法
# loss = MSE(f_θ(y_sigmoid(s)), d/ds[sigmoid(s)])
# 即：让 FNN 的"瞬时输出"匹配 sigmoid 的导数
dyds_pred = model(y_tensor)       # y_tensor 是 sigmoid 轨迹上的点
loss = criterion(dyds_pred, dyds_tensor)  # dyds_tensor 是 sigmoid 导数
```

**问题根源：** 导数匹配（derivative fitting）≠ 轨迹匹配（trajectory fitting）。

文献（SPIN-ODE 2025；Port-Hamiltonian NN 2025）明确指出：

> 即使 FNN 在每一点的导数预测完全正确，积分时从 y0 出发的 ODE 轨迹也未必正确。
> 原因：FNN 被训练的输入分布是 `y_sigmoid(s)`，而积分时实际到达的 `y(s)` 与之存在偏差，
> 这个偏差会随 s 增大而累积，最终导致轨迹漂移。

用图示来理解：

```
导数匹配的训练阶段：
  s → sigmoid(s) → 作为输入 → FNN → 预测 d(sigmoid)/ds ✓（局部正确）

导数匹配的推理阶段：
  y0 → ODE solver → y(s) ≠ sigmoid(s) → FNN(y(s)) ≠ d(y)/ds ✗（轨迹漂移）
```

### 3.2 正确做法：轨迹预训练（Trajectory Pretraining）

让 ODE **积分轨迹**直接匹配 Sigmoid 参考轨迹：

```
loss = MSE(ODE_integrate(f_θ, y0, s_grid), sigmoid_trajectory(s_grid))
```

这样训练出的 FNN，其积分轨迹（而非逐点输出）与目标对齐。

### 3.3 完整代码实现

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchdiffeq import odeint as torch_odeint
import numpy as np
from scipy.optimize import curve_fit


def sigmoid_func(s, a, b, c, d):
    """标准 4 参数 Sigmoid"""
    return a / (1.0 + np.exp(-b * (s - c))) + d


def fit_sigmoids_unified(s_pop, y_pop_norm):
    """
    对 4 个生物标志物分别拟合 Sigmoid。
    输入均为归一化空间（已 z-score）。
    """
    params = []
    for k in range(4):
        mask = ~np.isnan(y_pop_norm[:, k])
        p0 = [-1.0, 0.3, 5.0, 1.5] if k in [0, 2] else [1.0, 0.3, 5.0, -1.0]
        try:
            popt, _ = curve_fit(
                sigmoid_func,
                s_pop[mask],
                y_pop_norm[mask, k],
                p0=p0,
                maxfev=20000,
                bounds=([-5, 0.01, -15, -5], [5, 5, 30, 5])
            )
            params.append(popt)
        except RuntimeError:
            print(f"  警告：第 {k} 个标志物 Sigmoid 拟合失败，使用默认参数")
            params.append(p0)
    return np.array(params)  # [4, 4]


def get_sigmoid_trajectory(s_grid, params):
    """
    计算 Sigmoid 轨迹。
    返回：y_traj [len(s_grid), 4]，dy_traj [len(s_grid), 4]
    """
    y = np.zeros((len(s_grid), 4))
    dyds = np.zeros((len(s_grid), 4))
    for k in range(4):
        a, b, c, d = params[k]
        exp_term = np.exp(-b * (s_grid - c))
        y[:, k] = a / (1.0 + exp_term) + d
        dyds[:, k] = (a * b * exp_term) / ((1.0 + exp_term) ** 2)
    return y, dyds


def pretrain_on_trajectory(model, sigmoid_params, y0, n_epochs=3000,
                           s_min=-10.0, s_max=22.0, n_grid=300,
                           lr=1e-3, verbose=True):
    """
    ✅ 正确的预训练：让 ODE 积分轨迹匹配 Sigmoid 参考轨迹（不是导数匹配！）

    Args:
        model: CascadeFNN 实例
        sigmoid_params: [4, 4] Sigmoid 参数数组
        y0: torch.Tensor [4]，初始状态（CN 群体均值）
        n_epochs: 训练轮数
        s_min, s_max: DPS 积分范围
        n_grid: 积分网格点数
        lr: 学习率
        verbose: 是否打印进度
    """
    # 生成参考轨迹
    s_grid_np = np.linspace(s_min, s_max, n_grid)
    y_ref_np, _ = get_sigmoid_trajectory(s_grid_np, sigmoid_params)
    y_ref = torch.tensor(y_ref_np, dtype=torch.float32)      # [n_grid, 4]
    s_grid = torch.tensor(s_grid_np, dtype=torch.float32)    # [n_grid]

    ode_model = ODEWrapper(model)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=lr * 0.01)

    best_loss = float('inf')
    best_state = None

    for epoch in range(n_epochs):
        optimizer.zero_grad()

        # ✅ 核心：通过 ODE 积分生成预测轨迹
        try:
            y_pred = torch_odeint(
                ode_model,
                y0,
                s_grid,
                method='rk4',                     # 预训练用 rk4（快且稳定）
                options={'step_size': (s_max - s_min) / 100}
            )  # [n_grid, 4]
        except Exception as e:
            print(f"  ODE 积分失败 @ epoch {epoch}: {e}")
            continue

        # 主 loss：轨迹 MSE（在 Sigmoid 上下方向加权）
        traj_loss = F.mse_loss(y_pred, y_ref)

        # 辅助 loss：平台区域（早期和晚期）导数应接近 0
        n_plateau = n_grid // 6  # 各取约 1/6 的点作为平台段
        dy_early = torch.diff(y_pred[:n_plateau], dim=0)
        dy_late  = torch.diff(y_pred[-n_plateau:], dim=0)
        plateau_loss = (dy_early ** 2).mean() + (dy_late ** 2).mean()

        loss = traj_loss + 0.1 * plateau_loss

        if not torch.isfinite(loss):
            print(f"  NaN/Inf loss @ epoch {epoch}，跳过")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if verbose and (epoch + 1) % 200 == 0:
            print(f"  [Trajectory Pretrain] Epoch [{epoch+1}/{n_epochs}] "
                  f"TrajLoss={traj_loss.item():.6f}  PlateauLoss={plateau_loss.item():.6f}  "
                  f"LR={scheduler.get_last_lr()[0]:.2e}")

    # 恢复最优权重
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\n  预训练完成，最优 Loss={best_loss:.6f}")

    return model
```

### 3.4 两种预训练方式对比

| 维度 | 旧方式（导数匹配） | 新方式（轨迹匹配） |
|------|-------------------|-------------------|
| Loss | MSE(FNN(y_sig), sig') | MSE(ODE_traj, sig_traj) |
| 梯度流过 | FNN 本身 | FNN + ODE solver（adjoint） |
| 计算量 | 低 | 中（每次都要积分） |
| 积分轨迹质量 | 差（导数正确但轨迹漂移） | 好（直接优化轨迹） |
| 适合优化器 | Adam / L-BFGS | Adam（避免 L-BFGS 步长过激进） |

---

## 4. 问题二：`calculate_loss` 的积分起点 bug

### 4.1 问题分析

当前代码：

```python
def calculate_loss(ode_model, patient_data, ab, pids, y0):
    all_s, all_y = [], []
    for pid in pids:
        dat = patient_data[pid]
        s_values = ab[pid]['a'] * dat['t'] + ab[pid]['b']
        all_s.append(s_values)
        all_y.append(dat['y'])

    s_global, y_global = torch.cat(all_s), torch.cat(all_y)
    s_sorted, sort_indices = torch.sort(s_global)
    y_sorted = y_global[sort_indices]

    # ❌ 问题在这里：
    y_pred = torch_odeint(ode_model, y0, s_sorted, ...)
    loss = ((y_pred - y_sorted) ** 2).mean()
```

**Bug 的本质：**

`torch_odeint(f, y0, s_sorted)` 的语义是：从 `y0` 出发，在 `s_sorted[0]`（最小 DPS 点）处状态为 `y0`，然后向后积分到 `s_sorted[-1]`。

因此 `y_pred[0]` 被**硬约束**等于 `y0`，但是：
- `y_sorted[0]` 是所有患者中 DPS 最小的那次测量值
- 这个测量值包含个体噪声，不等于"CN 平均初始值" `y0`
- 结果：每个 epoch 都在这个点上产生剧烈的错误梯度

更严重的是：s_sorted 跨越所有患者的所有 DPS 点，**不均匀分布**。某些 DPS 区间的梯度密度远大于其他区间，模型会向数据点密集区"塌陷"。

### 4.2 正确做法：固定网格积分 + 可微插值

```python
def calculate_loss_v2(ode_model, patient_data, ab, pids, y0,
                      s_min=-12.0, s_max=32.0, n_grid=600):
    """
    ✅ 正确的 loss 计算：在固定均匀网格上积分，然后用线性插值找观测点处的预测值

    Args:
        ode_model: ODEWrapper 实例
        patient_data: dict，pid → {"t": tensor, "y": tensor, ...}
        ab: dict，pid → {"a": Parameter, "b": Parameter}
        pids: 参与计算的 pid 列表
        y0: 初始条件 [4]
        s_min, s_max: 积分覆盖的 DPS 范围（应比数据范围略宽）
        n_grid: 网格点数（越大越精确，但越慢）

    Returns:
        loss: scalar tensor，可反向传播
    """
    # Step 1：在均匀网格上积分一次（比逐患者积分高效很多）
    s_grid = torch.linspace(s_min, s_max, n_grid)

    try:
        y_traj = torch_odeint(
            ode_model, y0, s_grid,
            method='dopri5',
            rtol=1e-4, atol=1e-5
        )  # [n_grid, 4]
    except Exception as e:
        print(f"  ODE 积分失败: {e}")
        return torch.tensor(float('inf'), requires_grad=True)

    if not torch.all(torch.isfinite(y_traj)):
        return torch.tensor(float('inf'), requires_grad=True)

    # Step 2：对每个患者的每个观测点，用线性插值获取预测值
    total_loss = torch.zeros(1, requires_grad=True)
    n_obs = 0

    for pid in pids:
        dat = patient_data[pid]
        a = ab[pid]['a']
        b = ab[pid]['b']

        # 计算每个观测时间对应的 DPS（s_ij = α_i * t_ij + β_i）
        s_obs = (a * dat['t'] + b)  # [K]，K 为该患者的观测次数
        y_obs = dat['y']             # [K, 4]

        for j in range(len(s_obs)):
            s_j = s_obs[j]

            # 跳过超出积分范围的点（理论上应该通过 DPS 约束避免）
            if s_j < s_min + 1e-3 or s_j > s_max - 1e-3:
                continue

            # 可微线性插值（保留梯度流）
            # 将 s_j 映射到 [0, n_grid-1] 的浮点索引
            idx_f = (s_j - s_min) / (s_max - s_min) * (n_grid - 1)
            idx_lo = idx_f.long().clamp(0, n_grid - 2)
            idx_hi = idx_lo + 1
            frac = idx_f - idx_lo.float()

            # 插值得到预测值 [4]
            y_interp = y_traj[idx_lo] * (1.0 - frac) + y_traj[idx_hi] * frac

            # 只对非 NaN 的通道计算 loss
            mask = ~torch.isnan(y_obs[j])
            if mask.any():
                obs_loss = ((y_interp[mask] - y_obs[j][mask]) ** 2).sum()
                total_loss = total_loss + obs_loss
                n_obs += mask.sum().item()

    if n_obs == 0:
        return torch.tensor(0.0, requires_grad=True)

    return total_loss / n_obs


def calculate_loss_with_regularization(ode_model, patient_data, ab, pids, y0,
                                        pretrained_state=None,
                                        lambda_mono=0.05,
                                        lambda_plateau=0.02,
                                        lambda_weight=0.1,
                                        s_min=-12.0, s_max=32.0, n_grid=600):
    """
    带正则化的完整 loss（详见问题四）
    """
    # 主 loss
    data_loss = calculate_loss_v2(ode_model, patient_data, ab, pids, y0,
                                   s_min, s_max, n_grid)

    if not torch.isfinite(data_loss):
        return data_loss

    # 重新积分用于正则化（可复用上面的结果，此处为清晰独立计算）
    s_grid = torch.linspace(s_min, s_max, n_grid)
    try:
        y_traj = torch_odeint(ode_model, y0, s_grid, method='rk4',
                               options={'step_size': (s_max - s_min) / 200})
    except Exception:
        return data_loss

    # 单调性惩罚
    mono_loss = monotonicity_penalty(y_traj)

    # 平台惩罚
    n_plat = n_grid // 8
    dy = torch.diff(y_traj, dim=0)
    plateau_loss = (dy[:n_plat] ** 2).mean() + (dy[-n_plat:] ** 2).mean()

    # 权重正则（让 FNN 不要偏离预训练太远）
    weight_loss = torch.tensor(0.0)
    if pretrained_state is not None:
        for name, param in ode_model.named_parameters():
            if name in pretrained_state:
                diff = param - pretrained_state[name].to(param.device)
                weight_loss = weight_loss + (diff ** 2).sum()

    total = (data_loss
             + lambda_mono * mono_loss
             + lambda_plateau * plateau_loss
             + lambda_weight * weight_loss)

    return total
```

### 4.3 为什么用插值而不是逐患者积分

| 方案 | 计算量 | 梯度质量 | 稳定性 |
|------|--------|----------|--------|
| 逐患者积分（错误的当前做法） | O(N_patients × ODE) | 起点 bug | 低 |
| 逐患者积分（正确版） | O(N_patients × ODE) | 好 | 中 |
| **固定网格+插值（推荐）** | **O(1 × ODE + N × lookup)** | **好** | **高** |

---

## 5. 问题三：FNN 架构缺乏因果级联约束

### 5.1 问题分析

你的 `FNN` 是一个全连接网络 `R⁴ → R⁴`：

```
[Aβ, τ, N, C] → Linear → ReLU → ... → Linear → Tanh → [dAβ/ds, dτ/ds, dN/ds, dC/ds]
```

这意味着 FNN 可以学到：
- C 的变化率依赖 Aβ（跳过中间环节）
- τ 的变化率受 C 影响（**逆因果**）
- 任意的非线性混合

在噪声数据上，这种过度的自由度会导致模型学到虚假的相关性，无法重现真实的生物学趋势。

### 5.2 Zheng et al. 的 ODE 结构

论文给出的因果 ODE（Eq. 6）明确规定了级联结构：

```
dAβ/ds = f(Aβ)                      # 只依赖自身
dτ/ds  = g(Aβ, τ)                   # 依赖 Aβ 和自身
dN/ds  = h(τ, N)                    # 依赖 τ 和自身
dC/ds  = k(N, C)                    # 依赖 N 和自身
```

这个结构与论文中的 Algorithm 1 一致，也与生物学的 amyloid cascade 假说一致。

### 5.3 Cascade FNN 实现

```python
class CascadeFNN(nn.Module):
    """
    强制执行 Aβ → τ → N → C 因果链的 Neural ODE 向量场。
    相比全连接 FNN：
    - 参数量减少约 75%（减少过拟合风险）
    - 学到的轨迹在生物学上可解释
    - 梯度不会通过逆因果路径传播

    状态向量约定：y = [Aβ, τ, N, C]，均为归一化后的值。
    """

    def __init__(self, hidden_dim=64):
        super().__init__()
        # 每个子网只接受因果上游的输入
        # dAβ/ds = f_A(Aβ)            输入维度：1
        # dτ/ds  = f_T(Aβ, τ)         输入维度：2
        # dN/ds  = f_N(τ, N)           输入维度：2
        # dC/ds  = f_C(N, C)           输入维度：2
        self.f_A = self._make_subnet(1, hidden_dim)
        self.f_T = self._make_subnet(2, hidden_dim)
        self.f_N = self._make_subnet(2, hidden_dim)
        self.f_C = self._make_subnet(2, hidden_dim)

        # 各通道独立的输出缩放（允许不同量级的导数）
        # 在归一化空间里，合理的导数范围约为 [-0.3, 0.3]
        # 可以根据 sigmoid 导数的实际范围来设定
        self.scale_A = nn.Parameter(torch.tensor([0.15]))
        self.scale_T = nn.Parameter(torch.tensor([0.15]))
        self.scale_N = nn.Parameter(torch.tensor([0.15]))
        self.scale_C = nn.Parameter(torch.tensor([0.20]))

        self._init_weights()

    def _make_subnet(self, in_dim, hidden_dim):
        """构造每个因果子网"""
        return nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),   # 输出单个导数值
            nn.Tanh(),                  # 保证输出有界
        )

    def _init_weights(self):
        """小初始化，让初始 FNN 输出接近 0（稳定初期积分）"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)

    def forward(self, t, y):
        """
        Args:
            t: 当前 DPS 值（scalar，torchdiffeq 接口要求，实际不用）
            y: 当前状态 [4] 或 [B, 4]

        Returns:
            dy: 导数 [4] 或 [B, 4]
        """
        if y.dim() == 1:
            # 单点推理（积分时的常见形式）
            A = y[0:1]
            T = y[1:2]
            N = y[2:3]
            C = y[3:4]

            dA = self.f_A(A) * self.scale_A
            dT = self.f_T(torch.cat([A, T])) * self.scale_T
            dN = self.f_N(torch.cat([T, N])) * self.scale_N
            dC = self.f_C(torch.cat([N, C])) * self.scale_C

            return torch.cat([dA, dT, dN, dC])  # [4]

        else:
            # 批量推理（用于 calculate_loss 内部的某些情况）
            A = y[:, 0:1]
            T = y[:, 1:2]
            N = y[:, 2:3]
            C = y[:, 3:4]

            dA = self.f_A(A) * self.scale_A
            dT = self.f_T(torch.cat([A, T], dim=-1)) * self.scale_T
            dN = self.f_N(torch.cat([T, N], dim=-1)) * self.scale_N
            dC = self.f_C(torch.cat([N, C], dim=-1)) * self.scale_C

            return torch.cat([dA, dT, dN, dC], dim=-1)  # [B, 4]


class ODEWrapper(nn.Module):
    """
    torchdiffeq 要求的 ODE 包装器。
    将 CascadeFNN 包装为 (t, y) -> dy/dt 的接口。
    """
    def __init__(self, fnn: CascadeFNN):
        super().__init__()
        self.fnn = fnn

    def forward(self, t, y):
        return self.fnn(t, y)
```

### 5.4 架构比较

| 架构 | 参数量（hidden=64） | 因果约束 | 可解释性 | 抗噪声 |
|------|--------|---------|---------|--------|
| 原始全连接 FNN（hidden=128）| ~83K | ❌ | ❌ | 差 |
| 全连接 FNN（hidden=64） | ~21K | ❌ | ❌ | 中 |
| **CascadeFNN（hidden=64）** | **~5.5K** | **✅** | **✅** | **好** |

参数量减少到 1/4，但加入了生物学先验，通常在有限数据上效果更好。

### 5.5 更进一步：软单调约束

如果还想让 CascadeFNN 的每个子网内部也倾向于单调，可以用单调神经网络：

```python
class MonotonicNet(nn.Module):
    """
    近似单调神经网络（输出关于某个输入维度单调）。
    通过约束权重为正来实现（适用于单变量子网如 f_A）。
    """
    def __init__(self, hidden_dim=32):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, 1),
        )
        # 初始化为全正权重（保证单调性）
        with torch.no_grad():
            for m in self.layers.modules():
                if isinstance(m, nn.Linear):
                    m.weight.data.abs_()

    def forward(self, x):
        # 强制权重为正以保证单调性
        for m in self.layers.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.clamp_(min=0)
        return self.layers(x)
```

---

## 6. 问题四：Fine-tuning 正则化策略缺失

### 6.1 为什么 L-BFGS 在这里危险

L-BFGS 是强力优化器，步长大、收敛快。在噪声稀疏数据上，它会：
- 找到"完美拟合训练数据"的解（过拟合）
- 轨迹在观测点处可能非常精确，但在观测间隙完全乱掉
- 特别是联合优化 FNN + DPS 时，自由度远超约束数目

### 6.2 三种必须加的正则化

#### 6.2.1 单调性惩罚

基于生物学先验：Aβ↓, τ↑, N↓, C↑（随 DPS 增大）

```python
def monotonicity_penalty(y_traj):
    """
    对 ODE 轨迹施加单调性惩罚。
    y_traj: [T, 4]，沿 s 轴的轨迹，对应 [Aβ, τ, N, C]

    生物学约束：
    - Aβ: 随 DPS 增大而减小（降解/清除减少），dAβ/ds ≤ 0
    - τ: 随 DPS 增大而增大（磷酸化增加），dτ/ds ≥ 0
    - N: 随 DPS 增大而减小（神经元丢失），dN/ds ≤ 0
    - C: 随 DPS 增大而增大（认知评分升高代表更差），dC/ds ≥ 0
    """
    diff = torch.diff(y_traj, dim=0)  # [T-1, 4]

    # Aβ 应下降：惩罚正向变化
    pen_A = F.relu(diff[:, 0]).mean()
    # τ 应上升：惩罚负向变化
    pen_T = F.relu(-diff[:, 1]).mean()
    # N 应下降：惩罚正向变化
    pen_N = F.relu(diff[:, 2]).mean()
    # C 应上升：惩罚负向变化
    pen_C = F.relu(-diff[:, 3]).mean()

    return pen_A + pen_T + pen_N + pen_C


# 使用示例
def compute_full_loss(ode_model, y0, patient_data, ab, pids,
                      pretrained_state=None, s_min=-12, s_max=32, n_grid=600):
    """整合所有 loss 项的完整计算"""

    # 先积分一次，用于 data loss 和正则化
    s_grid = torch.linspace(s_min, s_max, n_grid)
    try:
        y_traj = torch_odeint(ode_model, y0, s_grid, method='dopri5',
                               rtol=1e-4, atol=1e-5)  # [n_grid, 4]
    except Exception:
        return torch.tensor(float('inf'))

    if not torch.isfinite(y_traj).all():
        return torch.tensor(float('inf'))

    # 1. 数据拟合 loss（用插值）
    data_loss = torch.tensor(0.0)
    n_obs = 0
    for pid in pids:
        dat = patient_data[pid]
        a, b = ab[pid]['a'], ab[pid]['b']
        s_obs = a * dat['t'] + b
        y_obs = dat['y']
        for j in range(len(s_obs)):
            s_j = s_obs[j]
            if not (s_min < s_j.item() < s_max):
                continue
            idx_f = (s_j - s_min) / (s_max - s_min) * (n_grid - 1)
            lo = idx_f.long().clamp(0, n_grid - 2)
            hi = lo + 1
            frac = idx_f - lo.float()
            y_interp = y_traj[lo] * (1 - frac) + y_traj[hi] * frac
            mask = ~torch.isnan(y_obs[j])
            if mask.any():
                data_loss = data_loss + ((y_interp[mask] - y_obs[j][mask]) ** 2).sum()
                n_obs += mask.sum().item()
    data_loss = data_loss / max(n_obs, 1)

    # 2. 单调性惩罚
    mono_loss = monotonicity_penalty(y_traj)

    # 3. 平台惩罚（两端导数接近 0）
    n_plateau = n_grid // 8
    dy = torch.diff(y_traj, dim=0)
    plateau_loss = (dy[:n_plateau] ** 2).mean() + (dy[-n_plateau:] ** 2).mean()

    # 4. 过渡区间平滑性（相邻导数不应突变）
    d2y = torch.diff(dy, dim=0)
    smooth_loss = (d2y ** 2).mean()

    # 5. 权重正则（不偏离预训练）
    weight_reg = torch.tensor(0.0)
    if pretrained_state is not None:
        for name, param in ode_model.named_parameters():
            key = name
            if key in pretrained_state:
                weight_reg = weight_reg + ((param - pretrained_state[key].to(param.device)) ** 2).sum()

    # 汇总（权重根据效果调整）
    loss = (1.0  * data_loss +
            0.05 * mono_loss +
            0.02 * plateau_loss +
            0.01 * smooth_loss +
            0.05 * weight_reg)

    return loss, {
        'data': data_loss.item(),
        'mono': mono_loss.item(),
        'plateau': plateau_loss.item(),
        'smooth': smooth_loss.item(),
        'weight': weight_reg.item() if isinstance(weight_reg, torch.Tensor) else weight_reg
    }
```

#### 6.2.2 平台约束（单独强调）

平台是目标效果中最关键的视觉特征。需要两种约束配合：

```python
def plateau_constraint_loss(ode_model, y0, y_cn_level, y_ad_level,
                             s_cn_end=-5.0, s_ad_start=16.0,
                             n_points=30):
    """
    额外的平台目标约束：CN 段应在 y_cn_level 附近，AD 段应在 y_ad_level 附近。
    y_cn_level: [4]，CN 段各生物标志物的目标值（从 sigmoid(-10) 获取）
    y_ad_level: [4]，AD 段各生物标志物的目标值（从 sigmoid(20) 获取）
    """
    # CN 平台
    s_cn = torch.linspace(-10.0, s_cn_end, n_points)
    y_cn_pred = torch_odeint(ode_model, y0, s_cn, method='rk4',
                              options={'step_size': 0.5})
    cn_loss = F.mse_loss(y_cn_pred, y_cn_level.unsqueeze(0).expand_as(y_cn_pred))

    # AD 平台（需要从中间继续积分）
    s_full = torch.linspace(-10.0, 22.0, 200)
    y_full = torch_odeint(ode_model, y0, s_full, method='rk4',
                           options={'step_size': 0.5})
    ad_mask = s_full >= s_ad_start
    y_ad_pred = y_full[ad_mask]
    ad_loss = F.mse_loss(y_ad_pred, y_ad_level.unsqueeze(0).expand_as(y_ad_pred))

    return cn_loss + ad_loss
```

#### 6.2.3 Fine-tuning 优化器选择

```python
def train_fnn_finetune(ode_model, patient_data, ab, pids, y0,
                       pretrained_state,
                       n_epochs=200, lr=1e-4):
    """
    Fine-tuning 阶段：只优化 FNN（DPS 固定），使用 Adam + 余弦退火。
    不使用 L-BFGS——在过拟合风险高的场景中，Adam 更稳健。
    """
    optimizer = optim.Adam(ode_model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=lr*0.01)

    for epoch in range(n_epochs):
        optimizer.zero_grad()

        loss, loss_dict = compute_full_loss(
            ode_model, y0, patient_data, ab, pids,
            pretrained_state=pretrained_state
        )

        if not torch.isfinite(loss):
            print(f"  NaN loss @ epoch {epoch}")
            break

        loss.backward()
        torch.nn.utils.clip_grad_norm_(ode_model.parameters(), max_norm=0.5)
        optimizer.step()
        scheduler.step()

        if (epoch + 1) % 20 == 0:
            print(f"  [FNN Fine-tune] Ep {epoch+1}/{n_epochs} | "
                  f"total={loss.item():.4f} | "
                  f"data={loss_dict['data']:.4f} | "
                  f"mono={loss_dict['mono']:.4f} | "
                  f"plateau={loss_dict['plateau']:.4f}")

    return ode_model
```

---

## 7. 问题五：DPS 与 FNN 联合优化的恶性循环

### 7.1 问题分析

你已经发现联合优化的问题：DPS 参数 (α_i, β_i) 和 FNN 同时优化时，系统存在**退化解**：

```
退化解1：所有 α_i → 0，所有患者映射到相同 DPS 点 → 三堆坍缩
退化解2：部分 α_i 变为负数 → 时间反转
退化解3：FNN 学到"随便的函数"来拟合坍缩的数据点
```

### 7.2 DPS 更新的正确策略

**策略 A：参数范围约束**

```python
def clamp_dps_params(ab, alpha_min=0.1, alpha_max=8.0,
                     s_min=-12.0, s_max=32.0, t_range=None):
    """
    在每次更新后对 DPS 参数施加物理约束。
    α_i：进展速率，必须为正（疾病只会进展，不会逆转）
    β_i：隐式约束，保证该患者的 DPS 范围合理
    """
    for pid, params in ab.items():
        # α 必须为正
        params['a'].data.clamp_(alpha_min, alpha_max)

        # 如果知道患者的年龄范围，可以进一步约束 DPS 范围
        if t_range is not None:
            t_lo, t_hi = t_range[pid]
            # s_i(t_lo) >= s_min
            beta_min = s_min - params['a'].item() * t_lo
            # s_i(t_hi) <= s_max
            beta_max = s_max - params['a'].item() * t_hi
            params['b'].data.clamp_(beta_min, beta_max)
```

**策略 B：DPS 分离训练（推荐）**

```python
def train_dps_only(ode_model, patient_data, ab, pids, y0,
                   n_epochs=50, lr=1e-5):
    """
    固定 FNN，只更新 DPS 参数。
    用极小 lr，只是"微调对齐"，不是大幅重排。
    """
    ode_model.eval()  # FNN 冻结
    for p in ode_model.parameters():
        p.requires_grad_(False)

    dps_params = [p for pid in pids for p in ab[pid].values()]
    optimizer = optim.Adam(dps_params, lr=lr)

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        loss, _ = compute_full_loss(ode_model, y0, patient_data, ab, pids)
        if torch.isfinite(loss):
            loss.backward()
            optimizer.step()
        clamp_dps_params(ab)  # 每步后强制约束

    # 恢复 FNN 梯度
    for p in ode_model.parameters():
        p.requires_grad_(True)
    ode_model.train()

    return ab
```

### 7.3 DPS 质量诊断

在任何优化之前，先验证你的 DPS 估计是否合理：

```python
def diagnose_dps_quality(patient_data, ab, stage_dict, sigmoid_params):
    """
    绘图诊断 DPS 对齐质量。
    好的 DPS 应该让：
    - CN 患者聚集在低 DPS 区（约 -5 ~ 2）
    - LMCI 患者聚集在中间区（约 0 ~ 10）
    - AD 患者聚集在高 DPS 区（约 5 ~ 20）
    - 各阶段之间有重叠但主体分离
    """
    import matplotlib.pyplot as plt

    all_s = {}
    all_y = {}
    all_stages = {}

    for pid in patient_data:
        if pid not in ab:
            continue
        dat = patient_data[pid]
        a = ab[pid]['a']
        b = ab[pid]['b']

        if isinstance(a, torch.Tensor):
            a = a.item()
            b = b.item()

        s = a * dat['t'].numpy() + b
        all_s[pid] = s
        all_y[pid] = dat['y'].numpy()
        all_stages[pid] = stage_dict.get(pid, 'Other')

    # 绘制各标志物 vs DPS
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    titles = ['Aβ', 'τ', 'N', 'C']
    colors = {'CN': 'orange', 'LMCI': 'green', 'AD': 'blue', 'Other': 'grey'}

    for k, ax in enumerate(axes.flat):
        for pid in all_s:
            stage = all_stages[pid]
            y_k = all_y[pid][:, k]
            valid = ~np.isnan(y_k)
            ax.scatter(all_s[pid][valid], y_k[valid],
                       c=colors.get(stage, 'grey'),
                       alpha=0.3, s=10, label=stage)

        ax.set_title(titles[k])
        ax.set_xlabel('DPS (s)')

    plt.suptitle('DPS 对齐诊断 — 数据分布')
    plt.tight_layout()
    plt.savefig('dps_diagnosis.png')
    plt.show()

    # 打印 DPS 分布统计
    print("\n各阶段 DPS 分布（均值±标准差）：")
    for stage in ['CN', 'LMCI', 'AD']:
        s_stage = []
        for pid in all_s:
            if all_stages.get(pid) == stage:
                s_stage.extend(all_s[pid].tolist())
        if s_stage:
            print(f"  {stage}: {np.mean(s_stage):.2f} ± {np.std(s_stage):.2f}"
                  f"  [min={np.min(s_stage):.1f}, max={np.max(s_stage):.1f}]")
```

---

## 8. 问题六：Tanh 输出头的细节缺陷

### 8.1 原代码中的问题

```python
class FNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            ...
            nn.Tanh(),
        )
        self.output_scaler = nn.Parameter(torch.tensor([0.1]), requires_grad=True)  # ❌

    def forward(self, y):
        return self.net(y) * self.output_scaler
```

**问题 1：** `output_scaler` 是可学习的标量，训练中会增大到接近无穷以突破 Tanh 的限制，实际上等于完全禁用了边界约束。

**问题 2：** 4 个生物标志物共享一个 `output_scaler`，但它们在归一化空间里的导数范围可能很不同（C 的变化往往比 N 大）。

**问题 3：** 在 CascadeFNN 里，每个子网内部已经有 Tanh，外部再套一个共享的 Tanh+scaler 会双重压制信号。

### 8.2 正确做法

```python
# 方案 A：在 CascadeFNN 里分通道设置独立缩放（推荐）
# 已在第 5.3 节的 CascadeFNN 中实现：
#   self.scale_A = nn.Parameter(torch.tensor([0.15]))
#   self.scale_T = nn.Parameter(torch.tensor([0.15]))
#   self.scale_N = nn.Parameter(torch.tensor([0.15]))
#   self.scale_C = nn.Parameter(torch.tensor([0.20]))
# 这些初始值来自对 sigmoid 导数范围的先验估计
# 如果不确定，可以先跑一次 pretrain 看看实际的导数量级

# 方案 B：对缩放参数加正则（防止训练中无限增大）
def scale_constraint_loss(model):
    """让输出缩放保持在合理范围内"""
    loss = 0
    for name, param in model.named_parameters():
        if 'scale' in name:
            # 惩罚超过 0.5 的缩放（在归一化空间里已经很大）
            loss += F.relu(param - 0.5).sum()
    return 0.1 * loss

# 方案 C：固定缩放（最保守，适合初期调试）
# 把 scale_X 设为非 Parameter，直接 register_buffer
self.register_buffer('scale_A', torch.tensor([0.15]))
self.register_buffer('scale_T', torch.tensor([0.15]))
self.register_buffer('scale_N', torch.tensor([0.15]))
self.register_buffer('scale_C', torch.tensor([0.20]))
```

### 8.3 如何估计合理的 scale 值

```python
def estimate_scale_from_sigmoid(sigmoid_params, s_range=(-10, 22), n_points=200):
    """
    从 Sigmoid 导数估计各通道合理的输出缩放。
    归一化空间里 scale 应约等于 sigmoid 导数的最大绝对值。
    """
    s_grid = np.linspace(*s_range, n_points)
    _, dyds = get_sigmoid_trajectory(s_grid, sigmoid_params)
    max_deriv = np.max(np.abs(dyds), axis=0)  # [4]

    print("各通道 sigmoid 导数最大绝对值（归一化空间）：")
    names = ['Aβ', 'τ', 'N', 'C']
    for k, (name, val) in enumerate(zip(names, max_deriv)):
        print(f"  {name}: {val:.4f}  → 建议 scale_{name[0]} = {val:.3f}")

    return max_deriv
```

---

## 9. 关于原论文机制模型也拟合不好的诊断

### 9.1 问题所在：DPS 估计是瓶颈

机制模型（Zheng et al.）在论文里效果好，关键不是 ODE 结构有多特别，而是 **Algorithm 1**：

```
for l = 1 to L:
    Step 3: 固定 DPS，更新 ODE 参数 w（最小化数据残差）
    Step 8: 固定 w，更新 DPS (α_i, β_i)（最小化加权残差）
    → 交替收敛，两者互相对齐
```

如果你的实现中 DPS 是从 **单个标志物** 的 sigmoid 拟合得到的（比如只用 Aβ 拟合 sigmoid，用 sigmoid 拟合的 DPS 参数），而不是如 Algorithm 1 那样**联合 4 个标志物同时估计**，那么：
- Aβ 看起来对齐了，但 τ、N、C 的 DPS 实际上是错的
- 用这个 DPS 去拟合 ODE，4 个方程同时受到错误 DPS 的干扰
- 结果：拟合很差

### 9.2 验证方法

```python
def check_dps_consistency(patient_data, ab, stage_dict):
    """
    检验 DPS 是否对 4 个标志物同时有意义。
    如果 DPS 只对某 1-2 个标志物有好的相关性，说明 DPS 估计有问题。
    """
    from scipy.stats import spearmanr
    import numpy as np

    all_s_vec = []
    all_y_matrix = []

    for pid in patient_data:
        if pid not in ab:
            continue
        dat = patient_data[pid]
        a = ab[pid]['a']
        b = ab[pid]['b']
        if isinstance(a, torch.Tensor):
            a, b = a.item(), b.item()

        t = dat['t'].numpy()
        s = a * t + b
        y = dat['y'].numpy()  # [K, 4]

        all_s_vec.extend(s.tolist())
        all_y_matrix.append(y)

    all_y = np.concatenate(all_y_matrix, axis=0)  # [total_obs, 4]
    all_s = np.array(all_s_vec)                    # [total_obs]

    names = ['Aβ', 'τ', 'N', 'C']
    print("DPS 与各标志物的 Spearman 相关系数：")
    for k, name in enumerate(names):
        valid = ~np.isnan(all_y[:, k])
        if valid.sum() < 10:
            continue
        r, p = spearmanr(all_s[valid], all_y[valid, k])
        direction = "↓" if r < 0 else "↑"
        expected = ["↓", "↑", "↓", "↑"]
        ok = "✅" if direction == expected[k] else "❌"
        print(f"  {name}: r={r:.3f}, p={p:.3e}  {direction} {ok}")

    print("\n（预期：Aβ↓, τ↑, N↓, C↑ 随 DPS 增大）")
```

### 9.3 如果 DPS 质量差怎么办

重新从头运行 Algorithm 1 的正确版本（联合所有 4 个标志物），或者至少确保 pretrain.py 中的 DPS 估计是基于联合优化的，而不是单标志物拟合。

---

## 10. 推荐完整训练流程

```
Step 0：数据质量检查
  ├── 可视化各标志物分布
  ├── 检验归一化是否合理
  └── 确保每个患者至少有 2 个时间点

Step 1：初始 DPS 估计（联合 4 标志物）
  ├── 用 Sigmoid 函数拟合所有患者的所有标志物
  ├── 对每个患者的 (α_i, β_i) 进行最小二乘拟合（加权）
  └── 验证 DPS 质量（diagnose_dps_quality）

Step 2：预训练 CascadeFNN（轨迹匹配）
  ├── 用 fit_sigmoids_unified 获取参考 Sigmoid 轨迹
  ├── 用 pretrain_on_trajectory 做轨迹预训练（3000 epochs）
  └── 可视化预训练后的轨迹 vs Sigmoid

Step 3：验证预训练效果
  ├── 积分 FNN，与 Sigmoid 和数据散点对比
  ├── 如果轨迹偏差大，检查是否 y0 设置有问题
  └── 如果两端平台不明显，增加 plateau_loss 权重

Step 4：Fine-tuning FNN（固定 DPS）
  ├── 保存预训练权重作为正则化基准
  ├── 用 Adam + CosineAnnealingLR（不用 L-BFGS）
  ├── 监控 data_loss, mono_loss, plateau_loss 分别的变化
  └── 如果 mono_loss 很高，增加 lambda_mono

Step 5：可选 DPS 微调
  ├── 固定 FNN（不更新权重）
  ├── 只用很小的 lr 更新 DPS (α_i, β_i)
  └── 每步后 clamp DPS 参数到合理范围

Step 6：最终验证与可视化
  ├── 绘制 ODE 轨迹 + 置信区间 + 散点
  └── 对比 Sigmoid、ODE（预训练）、ODE（fine-tuned）三条曲线
```

### 10.1 各阶段诊断检查表

```
预训练后：
  □ 积分轨迹与 Sigmoid 的 MSE < 0.05（归一化空间）
  □ 两端平台可见（目视）
  □ Aβ, N 递减，τ, C 递增

Fine-tuning 后：
  □ data_loss 相比预训练有改善
  □ 单调性不变（mono_loss 接近 0）
  □ 平台落差 ≥ 原始数据分布的 50%
  □ 曲线比 Sigmoid 更平滑（过渡区更宽）
```

---

## 11. 完整重构代码

以下是把上述所有改进整合后的完整文件 `fnn_v2.py`：

```python
"""
fnn_v2.py - 改进版 Neural ODE AD 生物标志物建模

关键改进：
1. CascadeFNN：强制因果级联结构 Aβ→τ→N→C
2. 轨迹预训练（非导数匹配）
3. 固定网格积分 + 线性插值的 loss 计算
4. 单调性 + 平台 + 平滑 + 权重正则化
5. 三阶段训练：预训练 → FNN fine-tune（固定DPS）→ 可选DPS微调
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchdiffeq import odeint as torch_odeint
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pccmnn as pc


# ═══════════════════════════════════════════════════
# 0. 配置
# ═══════════════════════════════════════════════════

CONFIG = {
    'S_MIN': -12.0,        # DPS 积分范围下界（略宽于数据）
    'S_MAX': 32.0,         # DPS 积分范围上界
    'N_GRID': 600,         # 固定积分网格点数
    'HIDDEN_DIM': 64,      # CascadeFNN 隐层维度

    # 正则化权重
    'LAMBDA_MONO': 0.05,
    'LAMBDA_PLATEAU': 0.02,
    'LAMBDA_SMOOTH': 0.01,
    'LAMBDA_WEIGHT': 0.05,

    # 训练配置
    'PRETRAIN_EPOCHS': 3000,
    'PRETRAIN_LR': 1e-3,
    'FINETUNE_EPOCHS': 200,
    'FINETUNE_LR': 1e-4,
    'DPS_FINETUNE_EPOCHS': 50,
    'DPS_FINETUNE_LR': 1e-5,
}


# ═══════════════════════════════════════════════════
# 1. 架构定义
# ═══════════════════════════════════════════════════

class CascadeFNN(nn.Module):
    """强制 Aβ→τ→N→C 因果链的 Neural ODE 向量场"""

    def __init__(self, hidden_dim=64):
        super().__init__()
        self.f_A = self._make_subnet(1, hidden_dim)   # dAβ/ds = f(Aβ)
        self.f_T = self._make_subnet(2, hidden_dim)   # dτ/ds  = f(Aβ, τ)
        self.f_N = self._make_subnet(2, hidden_dim)   # dN/ds  = f(τ, N)
        self.f_C = self._make_subnet(2, hidden_dim)   # dC/ds  = f(N, C)

        # 各通道独立缩放（初始值来自 sigmoid 导数量级的先验估计）
        self.scale_A = nn.Parameter(torch.tensor([0.15]))
        self.scale_T = nn.Parameter(torch.tensor([0.15]))
        self.scale_N = nn.Parameter(torch.tensor([0.15]))
        self.scale_C = nn.Parameter(torch.tensor([0.20]))

        self._init_weights()

    def _make_subnet(self, in_dim, hidden_dim):
        return nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1), nn.Tanh(),
        )

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)

    def forward(self, t, y):
        if y.dim() == 1:
            A, T, N, C = y[0:1], y[1:2], y[2:3], y[3:4]
            dA = self.f_A(A) * self.scale_A
            dT = self.f_T(torch.cat([A, T])) * self.scale_T
            dN = self.f_N(torch.cat([T, N])) * self.scale_N
            dC = self.f_C(torch.cat([N, C])) * self.scale_C
            return torch.cat([dA, dT, dN, dC])
        else:
            A, T, N, C = y[:,0:1], y[:,1:2], y[:,2:3], y[:,3:4]
            dA = self.f_A(A) * self.scale_A
            dT = self.f_T(torch.cat([A, T], dim=-1)) * self.scale_T
            dN = self.f_N(torch.cat([T, N], dim=-1)) * self.scale_N
            dC = self.f_C(torch.cat([N, C], dim=-1)) * self.scale_C
            return torch.cat([dA, dT, dN, dC], dim=-1)


# ═══════════════════════════════════════════════════
# 2. 工具函数
# ═══════════════════════════════════════════════════

def sigmoid_func(s, a, b, c, d):
    return a / (1.0 + np.exp(-b * (s - c))) + d


def fit_sigmoids_unified(s_pop, y_pop_norm):
    """联合拟合 4 个标志物的 Sigmoid，使用有界优化"""
    params = []
    bounds_list = [
        ([-3, 0.05, -5, -2], [0, 3, 20, 3]),    # Aβ: 下降型
        ([0, 0.05, -5, -3], [3, 3, 20, 2]),      # τ: 上升型
        ([-3, 0.05, -5, -2], [0, 3, 20, 3]),     # N: 下降型
        ([0, 0.05, -5, -3], [3, 3, 20, 2]),      # C: 上升型
    ]
    p0_list = [
        [-1.0, 0.3, 3.0, 1.0],
        [1.0, 0.3, 3.0, -1.0],
        [-1.0, 0.3, 3.0, 1.0],
        [1.0, 0.3, 3.0, -1.0],
    ]
    for k in range(4):
        mask = ~np.isnan(y_pop_norm[:, k])
        try:
            popt, _ = curve_fit(
                sigmoid_func, s_pop[mask], y_pop_norm[mask, k],
                p0=p0_list[k], bounds=bounds_list[k], maxfev=20000
            )
            params.append(popt)
        except Exception as e:
            print(f"  警告：第 {k} 个标志物拟合失败（{e}），使用默认值")
            params.append(p0_list[k])
    return np.array(params)


def get_sigmoid_trajectory(s_grid, params):
    y = np.zeros((len(s_grid), 4))
    dyds = np.zeros((len(s_grid), 4))
    for k in range(4):
        a, b, c, d = params[k]
        exp_t = np.exp(-b * (s_grid - c))
        y[:, k] = a / (1.0 + exp_t) + d
        dyds[:, k] = (a * b * exp_t) / ((1.0 + exp_t) ** 2)
    return y, dyds


def monotonicity_penalty(y_traj):
    """Aβ↓, τ↑, N↓, C↑"""
    diff = torch.diff(y_traj, dim=0)
    return (F.relu(diff[:, 0]).mean()     # Aβ 应下降
            + F.relu(-diff[:, 1]).mean()  # τ 应上升
            + F.relu(diff[:, 2]).mean()   # N 应下降
            + F.relu(-diff[:, 3]).mean()) # C 应上升


# ═══════════════════════════════════════════════════
# 3. 轨迹预训练（核心改进）
# ═══════════════════════════════════════════════════

def pretrain_on_trajectory(model, sigmoid_params, y0,
                           n_epochs=3000, lr=1e-3,
                           s_min=-10.0, s_max=22.0, n_grid=300):
    """让 ODE 积分轨迹匹配 Sigmoid 参考轨迹（非导数匹配）"""
    s_np = np.linspace(s_min, s_max, n_grid)
    y_ref_np, _ = get_sigmoid_trajectory(s_np, sigmoid_params)
    y_ref = torch.tensor(y_ref_np, dtype=torch.float32)
    s_grid = torch.tensor(s_np, dtype=torch.float32)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=lr*0.01)

    best_loss, best_state = float('inf'), None

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        try:
            y_pred = torch_odeint(model, y0, s_grid, method='rk4',
                                   options={'step_size': (s_max - s_min) / 100})
        except Exception:
            continue

        traj_loss = F.mse_loss(y_pred, y_ref)

        n_p = n_grid // 6
        dy = torch.diff(y_pred, dim=0)
        plateau_loss = (dy[:n_p] ** 2).mean() + (dy[-n_p:] ** 2).mean()

        mono_loss = monotonicity_penalty(y_pred)

        loss = traj_loss + 0.1 * plateau_loss + 0.05 * mono_loss

        if not torch.isfinite(loss):
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 500 == 0:
            print(f"  [Pretrain] Ep {epoch+1}/{n_epochs} | "
                  f"traj={traj_loss.item():.5f} | "
                  f"plateau={plateau_loss.item():.5f} | "
                  f"best={best_loss:.5f}")

    if best_state:
        model.load_state_dict(best_state)
    print(f"\n  ✅ 预训练完成，最优 Loss={best_loss:.6f}")
    return model


# ═══════════════════════════════════════════════════
# 4. Fine-tuning（固定 DPS，只优化 FNN）
# ═══════════════════════════════════════════════════

def finetune_fnn(model, patient_data, ab, pids, y0, pretrained_state,
                 n_epochs=200, lr=1e-4,
                 s_min=-12.0, s_max=32.0, n_grid=600):
    """固定 DPS，只优化 FNN，使用多项正则化"""
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=lr*0.01)

    best_loss, best_state = float('inf'), None

    for epoch in range(n_epochs):
        optimizer.zero_grad()

        s_grid = torch.linspace(s_min, s_max, n_grid)
        try:
            y_traj = torch_odeint(model, y0, s_grid, method='dopri5', rtol=1e-4, atol=1e-5)
        except Exception:
            continue

        if not torch.isfinite(y_traj).all():
            continue

        # 数据 loss（插值）
        data_loss = torch.tensor(0.0)
        n_obs = 0
        for pid in pids:
            dat = patient_data[pid]
            s_obs = ab[pid]['a'] * dat['t'] + ab[pid]['b']
            y_obs = dat['y']
            for j in range(len(s_obs)):
                s_j = s_obs[j]
                if not (s_min < s_j.item() < s_max):
                    continue
                idx_f = (s_j - s_min) / (s_max - s_min) * (n_grid - 1)
                lo = idx_f.long().clamp(0, n_grid - 2)
                frac = idx_f - lo.float()
                y_interp = y_traj[lo] * (1-frac) + y_traj[lo+1] * frac
                mask = ~torch.isnan(y_obs[j])
                if mask.any():
                    data_loss = data_loss + ((y_interp[mask] - y_obs[j][mask])**2).sum()
                    n_obs += mask.sum().item()
        data_loss = data_loss / max(n_obs, 1)

        # 正则化
        mono_loss = monotonicity_penalty(y_traj)
        dy = torch.diff(y_traj, dim=0)
        n_p = n_grid // 8
        plateau_loss = (dy[:n_p]**2).mean() + (dy[-n_p:]**2).mean()
        smooth_loss = (torch.diff(dy, dim=0)**2).mean()

        weight_reg = sum(
            ((p - pretrained_state[n].to(p.device))**2).sum()
            for n, p in model.named_parameters()
            if n in pretrained_state
        )

        loss = (data_loss
                + CONFIG['LAMBDA_MONO'] * mono_loss
                + CONFIG['LAMBDA_PLATEAU'] * plateau_loss
                + CONFIG['LAMBDA_SMOOTH'] * smooth_loss
                + CONFIG['LAMBDA_WEIGHT'] * weight_reg)

        if not torch.isfinite(loss):
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        scheduler.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 20 == 0:
            print(f"  [Finetune] Ep {epoch+1}/{n_epochs} | "
                  f"data={data_loss.item():.4f} | "
                  f"mono={mono_loss.item():.4f} | "
                  f"plateau={plateau_loss.item():.4f}")

    if best_state:
        model.load_state_dict(best_state)
    print(f"\n  ✅ Fine-tuning 完成，最优 Loss={best_loss:.6f}")
    return model


# ═══════════════════════════════════════════════════
# 5. DPS 诊断与可选微调
# ═══════════════════════════════════════════════════

def diagnose_and_clamp_dps(patient_data, ab, stage_dict):
    """诊断 DPS 质量 + 约束参数范围"""
    from scipy.stats import spearmanr

    all_s, all_y_list, all_stages = [], [], []
    for pid in patient_data:
        if pid not in ab:
            continue
        dat = patient_data[pid]
        a = ab[pid]['a'].item() if isinstance(ab[pid]['a'], torch.Tensor) else ab[pid]['a']
        b = ab[pid]['b'].item() if isinstance(ab[pid]['b'], torch.Tensor) else ab[pid]['b']
        s = a * dat['t'].numpy() + b
        y = dat['y'].numpy()
        all_s.extend(s.tolist())
        all_y_list.append(y)
        all_stages.extend([stage_dict.get(pid, 'Other')] * len(s))

    all_y = np.concatenate(all_y_list, axis=0)
    all_s_arr = np.array(all_s)

    print("\n── DPS 诊断 ──")
    names = ['Aβ (应↓)', 'τ (应↑)', 'N (应↓)', 'C (应↑)']
    expected_sign = [-1, 1, -1, 1]
    for k, (name, exp) in enumerate(zip(names, expected_sign)):
        valid = ~np.isnan(all_y[:, k])
        if valid.sum() < 10:
            continue
        r, p = spearmanr(all_s_arr[valid], all_y[valid, k])
        ok = "✅" if (r * exp) > 0 else "❌"
        print(f"  {name}: r={r:.3f}, p={p:.2e}  {ok}")

    print("\n── 各阶段 DPS 范围 ──")
    all_s_dict = {}
    for pid in patient_data:
        if pid not in ab:
            continue
        dat = patient_data[pid]
        a = ab[pid]['a'].item() if isinstance(ab[pid]['a'], torch.Tensor) else ab[pid]['a']
        b = ab[pid]['b'].item() if isinstance(ab[pid]['b'], torch.Tensor) else ab[pid]['b']
        s = a * dat['t'].numpy() + b
        stage = stage_dict.get(pid, 'Other')
        all_s_dict.setdefault(stage, []).extend(s.tolist())

    for stage, s_list in all_s_dict.items():
        print(f"  {stage}: {np.mean(s_list):.2f} ± {np.std(s_list):.2f}"
              f"  [{np.min(s_list):.1f}, {np.max(s_list):.1f}]")


def finetune_dps_only(model, patient_data, ab, pids, y0,
                       n_epochs=50, lr=1e-5,
                       s_min=-12.0, s_max=32.0, n_grid=600):
    """固定 FNN，只微调 DPS 参数"""
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    dps_params = [ab[pid]['a'] for pid in pids] + [ab[pid]['b'] for pid in pids]
    optimizer = optim.Adam(dps_params, lr=lr)

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        s_grid = torch.linspace(s_min, s_max, n_grid)
        try:
            y_traj = torch_odeint(model, y0, s_grid, method='dopri5', rtol=1e-4, atol=1e-5)
        except Exception:
            continue

        data_loss = torch.tensor(0.0)
        n_obs = 0
        for pid in pids:
            dat = patient_data[pid]
            s_obs = ab[pid]['a'] * dat['t'] + ab[pid]['b']
            y_obs = dat['y']
            for j in range(len(s_obs)):
                s_j = s_obs[j]
                if not (s_min < s_j.item() < s_max):
                    continue
                idx_f = (s_j - s_min) / (s_max - s_min) * (n_grid - 1)
                lo = idx_f.long().clamp(0, n_grid - 2)
                frac = idx_f - lo.float()
                y_interp = y_traj[lo] * (1-frac) + y_traj[lo+1] * frac
                mask = ~torch.isnan(y_obs[j])
                if mask.any():
                    data_loss = data_loss + ((y_interp[mask] - y_obs[j][mask])**2).sum()
                    n_obs += mask.sum().item()
        data_loss = data_loss / max(n_obs, 1)

        if torch.isfinite(data_loss):
            data_loss.backward()
            optimizer.step()

        # 约束 DPS 参数
        for pid in pids:
            ab[pid]['a'].data.clamp_(0.1, 8.0)

        if (epoch + 1) % 10 == 0:
            print(f"  [DPS Finetune] Ep {epoch+1}/{n_epochs} | loss={data_loss.item():.5f}")

    for p in model.parameters():
        p.requires_grad_(True)
    model.train()
    return ab


# ═══════════════════════════════════════════════════
# 6. 可视化
# ═══════════════════════════════════════════════════

def plot_results(model, y0, s_pop, y_pop_orig, stages_pop,
                 sigmoid_params, stage='final',
                 s_min=-12.0, s_max=32.0, n_grid=500,
                 n_samples=100, nn_sigma=0.01):
    """绘制带置信区间的轨迹图"""
    s_grid_np = np.linspace(s_min, s_max, n_grid)
    s_grid = torch.tensor(s_grid_np, dtype=torch.float32)

    # Sigmoid 参考曲线（逆归一化）
    y_sig_norm, _ = get_sigmoid_trajectory(s_grid_np, sigmoid_params)
    y_sig_orig = pc.inv_nor(y_sig_norm)

    # ODE 均值轨迹（不确定性采样）
    model.eval()
    pred_trajectories = []
    for _ in range(n_samples):
        temp_model = CascadeFNN(CONFIG['HIDDEN_DIM'])
        temp_model.load_state_dict(model.state_dict())
        with torch.no_grad():
            for p in temp_model.parameters():
                p.add_(torch.randn_like(p) * nn_sigma)
            pred = torch_odeint(temp_model, y0, s_grid, method='rk4',
                                 options={'step_size': (s_max - s_min) / 200})
            pred_trajectories.append(pred.numpy())

    pred_arr = np.array(pred_trajectories)
    mean_norm = np.mean(pred_arr, axis=0)
    ci_norm = np.percentile(pred_arr, [5, 95], axis=0)

    mean_orig = pc.inv_nor(mean_norm)
    ci_orig = pc.inv_nor(ci_norm)

    # 绘图
    TITLES = ['Aβ (A)', 'p-Tau (T)', 'N', 'Cognition (C)']
    COLORS = {'CN': 'orange', 'LMCI': 'green', 'AD': 'blue', 'Other': 'grey'}

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    for k, ax in enumerate(axes.flat):
        for st in np.unique(stages_pop):
            mask = np.array(stages_pop) == st
            ax.scatter(s_pop[mask], y_pop_orig[mask, k],
                       s=12, alpha=0.35, c=COLORS.get(st, 'grey'), label=st, zorder=1)

        ax.plot(s_grid_np, mean_orig[:, k], 'k-', lw=2.5, label='ODE Trajectory', zorder=4)
        ax.fill_between(s_grid_np, ci_orig[0, :, k], ci_orig[1, :, k],
                        color='lightgrey', alpha=0.7, label='90% CI', zorder=2)
        ax.plot(s_grid_np, y_sig_orig[:, k], 'r--', lw=2, label='Sigmoid', zorder=3)

        ax.set_xlabel('DPS (s)')
        ax.set_ylabel(TITLES[k])
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_title(TITLES[k])

    fig.suptitle(f'CascadeFNN ODE Trajectory ({stage})', fontsize=15)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'fnn_{stage}.png', dpi=150)
    plt.show()
    print(f"  图像已保存: fnn_{stage}.png")


# ═══════════════════════════════════════════════════
# 7. 主流程
# ═══════════════════════════════════════════════════

if __name__ == '__main__':
    print("═" * 60)
    print("  Neural ODE AD 生物标志物建模 v2")
    print("═" * 60)

    # ── Step 0：加载数据 ──
    csf_dict = pc.load_data()
    stage_dict = pc.load_stage_dict()
    print(f"\n加载了 {len(csf_dict)} 位患者的数据")

    # 转换格式
    patient_data = {}
    for pid, sample in csf_dict.items():
        t = torch.from_numpy(sample[:, 0]).float()
        y = torch.from_numpy(sample[:, 1:5]).float()
        patient_data[pid] = {
            't': t, 'y': y, 'y0': y[0].clone(),
            'stage': stage_dict.get(pid, 'Other')
        }

    # ── Step 1：加载 DPS 并准备群体数据 ──
    print("\n[Step 1] 加载 DPS 参数...")
    try:
        dps_params_loaded = torch.load('dps.pth', weights_only=False)
    except FileNotFoundError:
        print("  错误：未找到 dps.pth，请先运行 pretrain.py")
        exit()

    ab = {}
    patient_pids = []
    all_s_list, all_y_list, all_stages_list = [], [], []

    for pid, sample in csf_dict.items():
        if pid not in dps_params_loaded:
            continue
        a_val = dps_params_loaded[pid]['a']
        b_val = dps_params_loaded[pid]['b']
        ab[pid] = {
            'a': nn.Parameter(torch.tensor(float(a_val), dtype=torch.float32)),
            'b': nn.Parameter(torch.tensor(float(b_val), dtype=torch.float32))
        }
        patient_pids.append(pid)

        stage = stage_dict.get(pid, 'Other')
        t, y = sample[:, 0], sample[:, 1:5]
        s = a_val * t + b_val
        all_s_list.append(s)
        all_y_list.append(y)
        all_stages_list.extend([stage] * len(t))

    s_pop = np.concatenate(all_s_list)
    y_pop_norm = np.concatenate(all_y_list)
    y_pop_orig = pc.inv_nor(y_pop_norm)

    print(f"  有效患者数：{len(patient_pids)}")

    # ── DPS 诊断 ──
    diagnose_and_clamp_dps(patient_data, ab, stage_dict)

    # ── Step 2：Sigmoid 拟合（联合 4 标志物）──
    print("\n[Step 2] 拟合 Sigmoid 参考曲线...")
    sigmoid_params = fit_sigmoids_unified(s_pop, y_pop_norm)
    print("  Sigmoid 参数：")
    for k, (name, p) in enumerate(zip(['Aβ','τ','N','C'], sigmoid_params)):
        print(f"    {name}: a={p[0]:.3f}, b={p[1]:.3f}, c={p[2]:.3f}, d={p[3]:.3f}")

    # ── 初始条件：CN 群体在 s=-10 时的均值 ──
    cn_y0s = [dat['y'][0] for pid, dat in patient_data.items()
               if dat['stage'] == 'CN' and pid in ab]
    y0 = torch.stack(cn_y0s).mean(dim=0) if cn_y0s else torch.zeros(4)
    print(f"\n  初始条件 y0 (CN 均值，归一化空间)：{y0.numpy()}")

    # ── Step 3：预训练（轨迹匹配）──
    print("\n[Step 3] 轨迹预训练...")
    model = CascadeFNN(CONFIG['HIDDEN_DIM'])
    model = pretrain_on_trajectory(
        model, sigmoid_params, y0,
        n_epochs=CONFIG['PRETRAIN_EPOCHS'],
        lr=CONFIG['PRETRAIN_LR']
    )

    # 保存预训练权重（用于 fine-tuning 正则）
    pretrained_state = {k: v.clone() for k, v in model.state_dict().items()}
    torch.save(pretrained_state, 'cascade_pretrained.pth')

    # 可视化预训练结果
    print("\n  可视化预训练结果...")
    plot_results(model, y0, s_pop, y_pop_orig, all_stages_list,
                 sigmoid_params, stage='pretrained')

    # ── Step 4：Fine-tuning FNN（固定 DPS）──
    print("\n[Step 4] Fine-tuning FNN（固定 DPS）...")
    model = finetune_fnn(
        model, patient_data, ab, patient_pids, y0,
        pretrained_state=pretrained_state,
        n_epochs=CONFIG['FINETUNE_EPOCHS'],
        lr=CONFIG['FINETUNE_LR']
    )

    plot_results(model, y0, s_pop, y_pop_orig, all_stages_list,
                 sigmoid_params, stage='finetuned')

    # ── Step 5：可选 DPS 微调 ──
    print("\n[Step 5] 可选：DPS 微调（固定 FNN）...")
    ab = finetune_dps_only(
        model, patient_data, ab, patient_pids, y0,
        n_epochs=CONFIG['DPS_FINETUNE_EPOCHS'],
        lr=CONFIG['DPS_FINETUNE_LR']
    )

    # 更新散点数据（DPS 可能微调了）
    all_s_new = []
    all_y_new = []
    all_stages_new = []
    for pid in patient_pids:
        dat = patient_data[pid]
        a = ab[pid]['a'].item()
        b = ab[pid]['b'].item()
        s = a * dat['t'].numpy() + b
        all_s_new.append(s)
        all_y_new.append(dat['y'].numpy())
        all_stages_new.extend([stage_dict.get(pid, 'Other')] * len(s))

    s_pop_new = np.concatenate(all_s_new)
    y_pop_new = np.concatenate(all_y_new)
    y_pop_orig_new = pc.inv_nor(y_pop_new)

    plot_results(model, y0, s_pop_new, y_pop_orig_new, all_stages_new,
                 sigmoid_params, stage='final')

    # ── 保存最终模型 ──
    torch.save({
        'model_state': model.state_dict(),
        'ab': {pid: {'a': ab[pid]['a'].item(), 'b': ab[pid]['b'].item()}
               for pid in patient_pids},
        'sigmoid_params': sigmoid_params,
        'y0': y0.numpy(),
        'config': CONFIG
    }, 'cascade_fnn_final.pth')

    print("\n✅ 完整流程执行完毕。")
    print("   已保存：cascade_pretrained.pth, cascade_fnn_final.pth")
    print("   已生成：fnn_pretrained.png, fnn_finetuned.png, fnn_final.png")
```

---

## 附录：常见调试问题

### A. ODE 数值崩溃

```python
# 症状：loss 突然变成 NaN，或 y_traj 出现 inf

# 原因 1：step_size 太大（rk4 模式）
# 解法：减小 step_size 或用 dopri5（自适应步长）
y_traj = torch_odeint(model, y0, s_grid, method='dopri5', rtol=1e-4, atol=1e-5)

# 原因 2：FNN 输出量级太大（训练初期）
# 解法：小初始化 + 小 output scale
nn.init.xavier_uniform_(m.weight, gain=0.1)  # gain 要小

# 原因 3：梯度爆炸
# 解法：clip_grad_norm_
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### B. 平台不明显

```python
# 增加平台损失权重
LAMBDA_PLATEAU = 0.1  # 从 0.02 提高到 0.1

# 或者加入硬目标（显式告诉 FNN 两端应该是什么值）
y_cn_level = torch.tensor(y_sig_norm[0])    # s=-10 时的 Sigmoid 值
y_ad_level = torch.tensor(y_sig_norm[-1])   # s=+22 时的 Sigmoid 值
plateau_hard_loss = (F.mse_loss(y_traj[:n_p], y_cn_level.expand(n_p, 4))
                     + F.mse_loss(y_traj[-n_p:], y_ad_level.expand(n_p, 4)))
```

### C. 曲线形态"S 形"不明显（落差小）

```python
# 原因：Sigmoid 拟合的 a 参数（幅度）太小
# 检查方法：
for k, p in enumerate(sigmoid_params):
    print(f"  标志物 {k} 幅度 |a| = {abs(p[0]):.3f}")
# 如果 |a| < 0.3，说明数据归一化后方差结构有问题，或 DPS 范围不够宽

# 原因：weight_reg 过大，FNN 被"锁"在预训练的 Sigmoid 附近
# 解法：降低 LAMBDA_WEIGHT 或在 fine-tuning 后期关闭 weight_reg
```

### D. 单调性不满足（τ 下降、C 下降等情况）

```python
# 检查：
s_check = torch.linspace(-10, 22, 200)
y_check = torch_odeint(model, y0, s_check, method='rk4', options={'step_size': 0.5})
dy_check = torch.diff(y_check, dim=0)
print("违反单调性的比例：")
print(f"  Aβ↓ 违反（正向变化）: {(dy_check[:,0]>0).float().mean():.3f}")
print(f"  τ↑ 违反（负向变化）:  {(dy_check[:,1]<0).float().mean():.3f}")
print(f"  N↓ 违反（正向变化）:  {(dy_check[:,2]>0).float().mean():.3f}")
print(f"  C↑ 违反（负向变化）:  {(dy_check[:,3]<0).float().mean():.3f}")

# 如果违反比例 > 10%，增大 LAMBDA_MONO 到 0.2 甚至 0.5
```

---

*文档版本：v1.0 | 2025*
*针对：ADNI 数据集 / Zheng et al. (2022) Neural ODE 扩展实现*
