import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import pccmnn as pc
from torch.utils.data import Dataset, DataLoader
import pandas as pd

# ------------------ 加载数据 ------------------
def load_data():
    """仿照main.py中的数据加载方法"""
    csf_dict = pc.load_csf_data()
    print("Number of valid patients:", len(csf_dict))
    
    patient_data = {}
    for pid, sample in csf_dict.items():
        t = torch.from_numpy(sample[:, 0]).float()            # 年龄
        y = torch.from_numpy(sample[:, 1:5]).float()          # biomarker A/T/N/C
        patient_data[pid] = {"t": t, "y": y, "y0": y[0].clone()}
    
    return patient_data

class ODEModel(torch.nn.Module):
    """
    Quadratic ODE with constrained dependencies:
      fA(A) uses {1, A, A^2}
      fT(A,T) uses {1, A, T, A^2, T^2, A*T}
      fN(T,N) uses {1, T, N, T^2, N^2, T*N}
      fC(N,C) uses {1, N, C, N^2, C^2, N*C}
    dy/ds = f(y)
    y = [A, T, N, C]
    """
    def __init__(self):
        super().__init__()
        # fA: wa0 + wa1*A + wa2*A^2
        self.wa0 = torch.nn.Parameter(torch.tensor(0.0))
        self.wa1 = torch.nn.Parameter(torch.tensor(0.917))  # A linear negative
        self.wa2 = torch.nn.Parameter(torch.tensor(-0.873))

        # fT: wt0 + wt1*A + wt2*A^2 + wt3*T + wt4*T^2 + wt5*A*T
        self.wt0  = torch.nn.Parameter(torch.tensor(0.0))
        self.wt1 = torch.nn.Parameter(torch.tensor(0.788))  # A term negative
        self.wt2  = torch.nn.Parameter(torch.tensor(-0.246))  # T term positive
        self.wt3 = torch.nn.Parameter(torch.tensor(0.002))
        self.wt4 = torch.nn.Parameter(torch.tensor(3.066))
        self.wt5 = torch.nn.Parameter(torch.tensor(-3.650))

        # fN: wn0 + wn1*T + wn2*T^2 + wn3*N + wn4*N^2 + wn5*T*N
        self.wn0  = torch.nn.Parameter(torch.tensor(0.0))
        self.wn1 = torch.nn.Parameter(torch.tensor(1.267))  # T term positive
        self.wn2  = torch.nn.Parameter(torch.tensor(-1.253))  # N term negative
        self.wn3 = torch.nn.Parameter(torch.tensor(0.018))
        self.wn4 = torch.nn.Parameter(torch.tensor(2.342))
        self.wn5 = torch.nn.Parameter(torch.tensor(-4.015))

        # fC: wc0 + wc1*C + wc2*C^2 + wc3*N + wc4*N^2 + wc5*N*C
        self.wc0  = torch.nn.Parameter(torch.tensor(0.0))
        self.wc1 = torch.nn.Parameter(torch.tensor(0.159))  # N term negative
        self.wc2  = torch.nn.Parameter(torch.tensor(0.202))  # C term positive
        self.wc3 = torch.nn.Parameter(torch.tensor(0.010))
        self.wc4 = torch.nn.Parameter(torch.tensor(0.019))
        self.wc5 = torch.nn.Parameter(torch.tensor(-0.176))

    def f(self, y: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        A, T, N, C = y[..., 0], y[..., 1], y[..., 2], y[..., 3]
        fA = self.wa0 + self.wa1 * A + self.wa2 * (A * A)
        fT = (
            self.wt0 + self.wt1 * A + self.wt2 * A**2 +
            self.wt3 * T + self.wt4 * T**2 + self.wt5 * (A * T)
        )
        fN = (
            self.wn0 + self.wn1 * N + self.wn2 * N**2 +
            self.wn3 * T + self.wn4 * T**2 + self.wn5 * (T * N)
        )
        fC = (
            self.wc0 + self.wc1 * C + self.wc2 * C**2 +
            self.wc3 * N + self.wc4 * N**2 + self.wc5 * (N * C)
        )
        return torch.stack([fA, fT, fN, fC], dim=-1)

    def forward(self, s_grid: torch.Tensor, y0: torch.Tensor) -> torch.Tensor:
        ys = [y0]
        for i in range(1, len(s_grid)):
            h = s_grid[i] - s_grid[i-1]
            y_i = ys[-1]
            k1 = self.f(y_i, s_grid[i-1])
            k2 = self.f(y_i + 0.5 * h * k1, s_grid[i-1])
            k3 = self.f(y_i + 0.5 * h * k2, s_grid[i-1])
            k4 = self.f(y_i + h * k3, s_grid[i-1])
            y_next = y_i + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            ys.append(y_next)
        return torch.stack(ys)

import copy
from torch.utils.data import Dataset, DataLoader

class PatientDataset(Dataset):
    def __init__(self, pids):
        self.pids = pids
    def __len__(self):
        return len(self.pids)
    def __getitem__(self, idx):
        return self.pids[idx]

def calculate_global_loss(model, s_global, y_global, y0_global=None, s_penalty_weight=0.0):
    # 从最小 s 开始，用统一初值 y0 生成整条轨迹，与按 s 排序后的 y_global 对齐计算 MSE
    if y0_global is None:
        y0_global = torch.tensor([0.1, 0.0, 0.0, 0.0])
    y_pred_global = model(s_global, y0_global)
    loss = (y_pred_global - y_global) ** 2
    loss = loss.mean()
    # s 越界惩罚（可选）
    if s_penalty_weight > 0:
        s_min, s_max = -10.0, 20.0
        below = torch.clamp(s_min - s_global, min=0.0)
        above = torch.clamp(s_global - s_max, min=0.0)
        s_penalty = (below.pow(2) + above.pow(2)).mean()
        loss = loss + s_penalty_weight * s_penalty
    return loss

def fit_population_poly_ode(
    patient_data,
    stage_dict,
    n_adam=100,
    n_lbfgs=1,
    adam_lr_w=1e-3,
    lbfgs_lr_ab=5e-2,
    batch_size=128,
    delta=1,
    max_lbfgs_it=10,
    s_penalty_weight=1e-2,
    pretrained_ab: dict = None,
):
    device = torch.device('cpu')
    model = ODEModel().to(device)

    # 初始化每个患者的 θ（alpha,beta）：若提供了预训练结果则直接使用
    ab = {}
    if pretrained_ab is not None:
        for pid, entry in pretrained_ab.items():
            theta_src = entry['theta'] if isinstance(entry, dict) and 'theta' in entry else None
            if theta_src is None:
                # 兼容 (a,b) 形式：将其近似映射回 theta（theta0 ~ log(a), theta1=b）
                a, b = entry
                theta0_val = float(torch.log(torch.tensor(a - 1e-4))) if a > 1e-4 else -5.0
                theta1_val = float(b)
            else:
                theta0_val = float(theta_src[0].detach())
                theta1_val = float(theta_src[1].detach())
            ab[pid] = {"theta": torch.tensor([theta0_val, theta1_val], requires_grad=True)}
    else:
        for pid, dat in patient_data.items():
            stage = stage_dict.get(pid, 'Other')
            if stage == 'CN':
                s_range = (-10.0, 0.0)
            elif stage == 'LMCI':
                s_range = (0.0, 10.0)
            elif stage == 'AD':
                s_range = (10.0, 20.0)
            else:
                s_range = (-5.0, 5.0)
            theta0 = torch.empty(1).uniform_(-5.0, 1.3862944)
            theta1 = torch.empty(1).uniform_(s_range[0], s_range[1])
            ab[pid] = {"theta": torch.tensor([theta0.item(), theta1.item()], requires_grad=True)}

    opt_w_adam = torch.optim.Adam(model.parameters(), lr=adam_lr_w)
    opt_ab_lbfgs = {
        pid: torch.optim.LBFGS([ab[pid]['theta']], lr=lbfgs_lr_ab, max_iter=max_lbfgs_it)
        for pid in ab
    }

    # 准备数据加载（患者级 minibatch）
    patient_pids = list(patient_data.keys())
    use_minibatch = batch_size < len(patient_pids)
    dataloader = None
    if use_minibatch and n_adam > 0:
        dataset = PatientDataset(patient_pids)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        batch_iterator = iter(dataloader)
    else:
        batch_iterator = None

    for it in range(n_adam):
        # 选择批次
        if use_minibatch:
            try:
                batch_pids = next(batch_iterator)
            except StopIteration:
                batch_iterator = iter(dataloader)
                batch_pids = next(batch_iterator)
            batch_pids_list = [pid.item() for pid in batch_pids]
        else:
            batch_pids_list = patient_pids

        # 构建本批次的 s,y 全局序列
        all_s_values, all_y_values = [], []
        for pid in batch_pids_list:
            dat = patient_data[pid]
            t = dat['t']
            theta = ab[pid]['theta']
            alpha = torch.nn.functional.softplus(theta[0]) + 1e-4
            beta = theta[1]
            s_values = alpha * t + beta
            all_s_values.append(s_values)
            all_y_values.append(dat['y'])

        s_global = torch.cat(all_s_values)
        y_global = torch.cat(all_y_values)
        s_sorted, sort_idx = torch.sort(s_global)
        y_sorted = y_global[sort_idx]

        # 更新模型参数（Adam）
        opt_w_adam.zero_grad()
        loss_w = calculate_global_loss(model, s_sorted, y_sorted, s_penalty_weight=s_penalty_weight)
        loss_w.backward()
        opt_w_adam.step()

        if (it + 1) % delta == 0:
            # 使用 L-BFGS 单独优化每个患者的 θ，使全局损失下降
            for pid in patient_pids:
                dat = patient_data[pid]
                if dat['t'].shape[0] < 2:
                    continue
                def closure_ab():
                    opt_ab_lbfgs[pid].zero_grad()
                    theta = ab[pid]['theta']
                    a = torch.nn.functional.softplus(theta[0]) + 1e-4
                    b = theta[1]
                    s_vals = a * dat['t'] + b
                    s_s, idx = torch.sort(s_vals)
                    y_s = dat['y'][idx]
                    l = calculate_global_loss(model, s_s, y_s, s_penalty_weight=s_penalty_weight)
                    l.backward()
                    return l
                try:
                    opt_ab_lbfgs[pid].step(closure_ab)
                except Exception as e:
                    print(f"LBFGS for pid {pid} failed: {e}")

        if (it + 1) % 10 == 0:
            print(f"Iter {it+1:03d}/{n_adam} | MSE={float(loss_w):.6f}")

    # 导出 α,β
    alpha_beta = {pid: (float(torch.nn.functional.softplus(v['theta'][0]).detach() + 1e-4),
                        float(v['theta'][1].detach())) for pid, v in ab.items()}
    model.eval()
    return model, alpha_beta

def plot_population(model, patient_data, ab_dict, stage_dict, name='poly_ode'):
    # 统计 s 的 10%-90% 分位
    all_s_values = []
    for p in patient_data:
        a, b = ab_dict[p]
        s_values = a * patient_data[p]['t'] + b
        all_s_values.append(s_values)
    all_s_flat = torch.cat(all_s_values)
    s_10 = torch.quantile(all_s_flat, 0.10)
    s_90 = torch.quantile(all_s_flat, 0.90)
    print(f"S value range: 10th percentile = {float(s_10):.2f}, 90th percentile = {float(s_90):.2f}")

    s_curve = torch.linspace(s_10, s_90, 200)
    with torch.no_grad():
        y_curve_full = model(s_curve, torch.tensor([0.1, 0, 0, 0]))
        y_curve_full = y_curve_full.detach().numpy()
    y_curve_full = pc.inv_nor(y_curve_full)

    fig2, axes = plt.subplots(2, 2, figsize=(9, 6))
    for k, ax in enumerate(axes.flat):
        s_by_stage = {'CN': [], 'LMCI': [], 'AD': [], 'Other': []}
        y_by_stage = {'CN': [], 'LMCI': [], 'AD': [], 'Other': []}
        for p in patient_data:
            a, b = ab_dict[p]
            stage = stage_dict.get(p, 'Other')
            if stage not in s_by_stage:
                stage = 'Other'
            s_values = a * patient_data[p]['t'] + b
            y_values = patient_data[p]['y'][:, k]
            mask = (s_values >= s_10) & (s_values <= s_90)
            if torch.any(mask):
                s_by_stage[stage].append(s_values[mask])
                y_by_stage[stage].append(y_values[mask])
        for stage, s_points in s_by_stage.items():
            if s_points:
                s_all = torch.cat(s_points).numpy()
                y_all = torch.cat(y_by_stage[stage]).numpy()
                y_all = pc.inv_nor(y_all, k)
                color = colors.get(stage, 'grey')
                ax.scatter(s_all, y_all, s=15, alpha=0.6, c=color, label=stage)
        ax.plot(s_curve, y_curve_full[:, k], lw=1.5, c='red', linestyle='--', label='Trajectory')
        ax.set_xlabel('Disease progression score  s')
        ax.set_ylabel(TITLES[k])
        ax.legend(fontsize=8)
    fig2.suptitle(f'Population Poly-ODE (s in 10-90 percentile: [{float(s_10):.2f}, {float(s_90):.2f}])')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f'{name}.png')
    plt.show()