import torch
import torch.nn as nn
import torch.optim as optim
import copy
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import pccmnn as pc
import os
from torch.utils.data import Dataset, DataLoader

name = 'original'
discription = 'This is original code simulating algorithm 1 in the paper'

# ------------------ 加载数据 ------------------
csf_dict = pc.load_data()

keys_to_delete = []
for key in csf_dict:
    sample = csf_dict[key]
    sample = sample[~np.isnan(sample).any(axis=1)]   # 删缺失
    csf_dict[key] = sample
    if sample.shape[0] < 2 :
        keys_to_delete.append(key)
for key in keys_to_delete:
    del csf_dict[key]
# ---------------------------------------------------

# ★ 把 numpy 数组转成 patient_data ★
patient_data = {}
for pid, sample in csf_dict.items():
    t = torch.from_numpy(sample[:, 0]).float()            # 年龄
    y = torch.from_numpy(sample[:, 1:5]).float()          # biomarker A/T/N/C
    patient_data[pid] = {"t": t, "y": y, "y0": y[0].clone()}

print("Valid patients: ", len(patient_data))

import torch.nn as nn

class ODEModel(nn.Module):
    def __init__(self, hidden_dim=32, num_layers=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),
        )

    def f(self, y: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        return self.net(y)

    def forward(self, s_grid: torch.Tensor, y0: torch.Tensor) -> torch.Tensor:
        # 4th-order Runge-Kutta integration to solve the ODE
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
            
        return torch.stack(ys)          # (len_s, 4)

def residual(model, dat, ab_pid_theta, sigma=None):
    """
    Calculates the residual: dy/dt - f(y),
    where dy/dt is computed by finite difference (centered for interior, one-sided for endpoints).
    Returns mean squared error of the residual.
    """
    num_steps = dat['t'].shape[0]
    if num_steps < 2:
        device = next(model.parameters()).device
        return torch.tensor(0.0, device=device)

    alpha = torch.exp(ab_pid_theta[0]) + 1e-4
    beta  = ab_pid_theta[1]
    s = alpha * dat['t'] + beta  # (N,)
    y = dat['y']                 # (N, D)

    # Compute dy/dt using finite differences
    t = dat['t']
    dy_dt = torch.zeros_like(y)
    # Forward difference for the first point
    dy_dt[0] = (y[1] - y[0]) / (t[1] - t[0])
    # Centered difference for interior points
    if num_steps > 2:
        dy_dt[1:-1] = (y[2:] - y[:-2]) / (t[2:].unsqueeze(1) - t[:-2].unsqueeze(1))
    # Backward difference for the last point
    dy_dt[-1] = (y[-1] - y[-2]) / (t[-1] - t[-2])

    # Model prediction f(y) at each s, y
    fy = torch.zeros_like(y)
    for i in range(len(s)):
        fy[i] = model.f(y[i], s[i])

    residual = dy_dt - fy
    loss = residual ** 2
    if sigma is not None:
        loss = loss / sigma
    return loss.mean()


def traj_loss(model, dat, alpha, beta, sigma=None):
    """
    Calculates the loss based on the global trajectory from y0.
    """
    num_steps = dat['t'].shape[0]
    if num_steps < 2:
        device = next(model.parameters()).device
        return torch.tensor(0.0, device=device)

    s = alpha * dat['t'] + beta
    
    y0 = dat['y'][0]
    
    y_pred_trajectory = model(s, y0)
    y_actual_trajectory = dat['y']
    
    loss = (y_pred_trajectory - y_actual_trajectory)**2
    if sigma is not None:
        loss = loss / sigma
    return loss.sum()

def calculate_combined_loss(model, dat, ab_pid_theta, sigma=None, lambda_=0.5):
    """
    Calculates a combined loss: lambda % sequential and (1-lambda)% global.
    """
    res = residual(model, dat, ab_pid_theta, sigma)
    loss_global = calculate_global_loss(model, dat, ab_pid_theta, sigma)
    combined_loss = (1-lambda_) * loss_global + lambda_ * res
        
    return combined_loss


import torch, torch.nn as nn, torch.optim as optim
import torch.nn.functional as F
from math import ceil

class PatientDataset(Dataset):
    def __init__(self, pids):
        self.pids = pids

    def __len__(self):
        return len(self.pids)

    def __getitem__(self, idx):
        return self.pids[idx]

def fit_population(
        patient_data,
        n_epochs      = 100,
        param_update_steps = 10,
        adam_lr_w    = 1e-2,
        adam_lr_ab   = 1e-3,
        ):
    
    # ---------- 计算每个生物标记物的全局方差以调整权重 ----------
    all_biomarkers = torch.cat([dat['y'] for dat in patient_data.values()], dim=0)
    sigma = all_biomarkers.var(dim=0)
    # 加上一个很小的数防止除以零
    sigma = sigma.clamp_min(1e-8)

    # ---------- 初始化 ----------
    model = ODEModel(hidden_dim=32, num_layers=2)
    def weights_init(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    model.apply(weights_init)
    
    # Accelerate model with JIT compilation
    try:
        model = torch.jit.script(model)
        print("Model successfully compiled with JIT.")
    except Exception as e:
        print(f"JIT compilation failed: {e}. Running in eager mode.")

    a = {}
    b = {}
    for pid, dat in patient_data.items():
        max_, min_ = dat['t'].max(), dat['t'].min()
        a[pid] = torch.rand(1) * 4.0
        low = -10 - a[pid] * min_
        high = 20 - a[pid] * max_
        b[pid] = low + torch.rand(1) * (high - low)

    opt_w  = optim.Adam(model.parameters(), lr=adam_lr_w, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(opt_w, milestones=list(range(60, 151)), gamma=0.99, last_epoch=-1)

    for it in range(n_epochs):
        loss = torch.zeros(4)
        sigma = torch.zeros(4)
        # update model parameters
        for k in range(4):
            for l in range(param_update_steps):
                loss_k = torch.tensor(0.0,requires_grad=True)
                for pid, dat in patient_data.items():
                    loss_k = loss_k + traj_loss(model, dat, a[pid], b[pid])
                
                loss_k.backward()
                opt_w.step()
            sigma_k = loss_k / (len(patient_data)+2*it-4)
            loss[k] = loss_k.item()
            sigma[k] = sigma_k.item()

        # update ab
        for i in range(it):
            for pid, dat in patient_data.items():
                opt_ab = optim.Adam([a[pid], b[pid]], lr=adam_lr_ab)
                loss_j = torch.tensor(0.0,requires_grad=True)
                for l in range(param_update_steps):
                    loss_j = loss_j + traj_loss(model, dat, a[pid], b[pid], sigma)
                loss_j.backward()
                opt_ab.step()

        if it%20 == 0:
            print(f"Epoch {it} loss: {loss.mean()}")

    return model, a, b

model, a, b = fit_population(patient_data)
