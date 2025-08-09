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

# ------------------ 你已有的部分 ------------------
csf_dict = pc.load_csf_data()

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

import torch.nn as nn

# 下面的层曾经在调试中用过，目前看可能没什么用
class NegativeTanh(nn.Module):
    def forward(self, x):
        return -torch.tanh(x)

class Gaussian(nn.Module):
    """
    Gaussian activation function.
    Applies the element-wise function:
    Gaussian(x) = exp(-x^2)
    
    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.exp(-torch.pow(x, 2))

# ---------- 计算y的分位数并保存为全局变量B, C, D ----------
# 收集所有患者的y数据
all_y_data = torch.cat([dat['y'] for dat in patient_data.values()], dim=0)

# 计算5分位数、50分位数（中位数）和95分位数
B = torch.quantile(all_y_data, 0.05, dim=0)  # 5分位数
C = torch.quantile(all_y_data, 0.50, dim=0)  # 50分位数（中位数）
D = torch.quantile(all_y_data, 0.95, dim=0)  # 95分位数

class ResidualBlock(nn.Module):
    """残差块"""
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(dim, dim),
        )
        
    def forward(self, x):
        return x + self.block(x)  # 残差连接

class PopulationODE(nn.Module):
    def __init__(self, hidden_dim=32, num_residual_blocks=2):
        super().__init__()
        
        # 1. 输入层：5个输入（4个生物标记物 + s值）-> hidden_dim
        self.input_layer = nn.Linear(5, hidden_dim)
        self.input_activation = nn.Tanh()
        self.input_dropout = nn.Dropout(0.1)
        
        # 2. 残差块序列
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(num_residual_blocks)
        ])
        
        # 3. 输出层：hidden_dim -> 4个生物标记物
        self.output_layer = nn.Linear(hidden_dim, 4)
        self.output_activation = nn.Sigmoid()
        self.output_dropout = nn.Dropout(0.1)
        
        # 4. A为可学习参数
        self.A = nn.Parameter(torch.zeros(4, dtype=torch.float32))

    def f(self, y, s):
        y = y.float()
        s = s.float()
        # 确保s是标量，将其扩展为与y兼容的形状
        if s.dim() == 0:  # s是标量
            s_expanded = s.unsqueeze(0)  # 变成[1]
        else:
            s_expanded = s
        
        # 将y和s连接起来作为网络输入
        z = torch.cat([y, s_expanded], dim=0)  # 连接为[5]的向量
        
        # 前向传播通过残差网络
        x = z.unsqueeze(0)  # 添加batch维度 -> [1, 5]
        
        # 输入层
        x = self.input_layer(x)  # [1, hidden_dim]
        x = self.input_activation(x)
        x = self.input_dropout(x)
        
        # 残差块序列
        for residual_block in self.residual_blocks:
            x = residual_block(x)
        
        # 输出层
        x = self.output_layer(x)  # [1, 4]
        x = self.output_activation(x)
        x = self.output_dropout(x)
        
        net_output = x.squeeze(0)  # 移除batch维度 -> [4]
        
        # 应用ODE动力学
        result = 1e-2*torch.sigmoid(y)*(self.A-torch.sigmoid(y))*net_output
        return result

    def forward(self, s_grid, y0):
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

def calculate_sequential_loss(model, dat, ab_pid_theta, sigma=None):
    """
    Calculates the loss by sequentially predicting the next step.
    For each time point y_i, it predicts y_{i+1} and computes the loss.
    """
    num_steps = dat['t'].shape[0]
    if num_steps < 2:
        # Cannot compute loss for a single data point.
        # Ensure tensor is on the correct device.
        device = next(model.parameters()).device
        return torch.tensor(0.0, device=device)

    alpha = torch.exp(ab_pid_theta[0]) + 1e-4
    beta  = ab_pid_theta[1]
    s = alpha * dat['t'] + beta

    y_preds = []
    y_actuals = []

    for i in range(num_steps - 1):
        s_grid_step = s[i:i+2]  # Slice [s[i], s[i+1]]
        y0_step = dat['y'][i]
        
        # model.forward returns predictions for the entire s_grid_step
        y_pred_step = model(s_grid_step, y0_step)
        
        # The prediction for s[i+1] is the second element
        y_preds.append(y_pred_step[1])
        y_actuals.append(dat['y'][i+1])

    y_preds = torch.stack(y_preds)
    y_actuals = torch.stack(y_actuals)
    
    # Calculate Mean Squared Error for this patient
    loss = (y_preds - y_actuals)**2
    if sigma is not None:
        loss = loss / sigma
    
    return loss.mean()

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
    # 需要为每个时间点调用f函数
    fy = torch.zeros_like(y)
    for i in range(len(s)):
        fy[i] = model.f(y[i], s[i])

    residual = dy_dt - fy
    loss = residual ** 2
    if sigma is not None:
        loss = loss / sigma
    return loss.mean()


def calculate_global_loss(model, dat, ab_pid_theta, sigma=None):
    """
    Calculates the loss based on the global trajectory from y0.
    """
    num_steps = dat['t'].shape[0]
    if num_steps < 2:
        device = next(model.parameters()).device
        return torch.tensor(0.0, device=device)

    alpha = torch.exp(ab_pid_theta[0]) + 1e-4
    beta  = ab_pid_theta[1]
    s = alpha * dat['t'] + beta
    
    y0 = dat['y'][0]
    
    y_pred_trajectory = model(s, y0)
    y_actual_trajectory = dat['y']
    
    loss = (y_pred_trajectory - y_actual_trajectory)**2
    if sigma is not None:
        loss = loss / sigma
        
    return loss.mean()

def calculate_smoothness_loss(model, s_grid, y0):
    """
    Calculates a smoothness penalty based on the second derivative of the trajectory.
    """
    y_pred = model(s_grid, y0)
    
    # Approximate second derivative using central differences
    # y'' ≈ (y_{i+1} - 2y_i + y_{i-1}) / h^2
    # We penalize the numerator, as h is constant
    if len(y_pred) < 3:
        return 0
    second_deriv = y_pred[2:] - 2 * y_pred[1:-1] + y_pred[:-2]
    
    return torch.mean(second_deriv**2)


def calculate_combined_loss(model, dat, ab_pid_theta, sigma=None, reg_lambda_smooth=0.0):
    """
    Calculates a combined loss: 50% sequential and 50% global.
    Includes an optional regularization term for smoothness.
    """
    #loss_seq = calculate_sequential_loss(model, dat, ab_pid_theta, sigma)
    res = residual(model, dat, ab_pid_theta, sigma)
    loss_global = calculate_global_loss(model, dat, ab_pid_theta, sigma)
    zero = torch.zeros(4)
    one = torch.ones(4)
    # 为f函数调用提供s值，这里使用0作为默认值
    s_default = torch.tensor(0.0)
    combined_loss = ( loss_global + res ) * (torch.norm(model.f(B, s_default)) + torch.norm(model.f(D, s_default))) / (torch.norm(model.f(C, s_default)) + 1e-2)
        
    if reg_lambda_smooth > 0:
        alpha = torch.exp(ab_pid_theta[0]) + 1e-4
        beta  = ab_pid_theta[1]
        s = alpha * dat['t'] + beta
        if len(s) > 2:
            smooth_loss = calculate_smoothness_loss(model, s, dat['y0'])
            combined_loss += reg_lambda_smooth * smooth_loss

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
        n_adam      = 40,      # adam 阶段迭代次数
        n_lbfgs    = 20,     # lbfgs 阶段迭代次数
        adam_lr_w    = 1e-3,
        adam_lr_ab   = 1e-3,
        lbfgs_lr_w   = 1e-3,
        lbfgs_lr_ab  = 1e-3,
        batch_size=128,
        weighted_sampling=True,
        max_lbfgs_it = 10,
        tolerance_grad = 0,
        tolerance_change = 0,
        reg_lambda_alpha=1e-2,
        reg_lambda_smooth=0.,
        early_stop_patience=100,
        early_stop_threshold=0.001):
    
    # ---------- 加载分期字典 ----------
    stage_dict = pc.load_stage_dict()

    # ---------- 筛选 LMCI 患者数据 ----------
    lmci_patient_data = {
        pid: dat for pid, dat in patient_data.items() if stage_dict.get(pid) == 'LMCI'
    }

    if not lmci_patient_data:
        print("没有 LMCI 阶段的患者数据，无法训练模型。")
        return None, None  # 或者抛出异常，根据你的需求

    # ---------- 计算每个生物标记物的全局方差以调整权重 ----------
    all_biomarkers = torch.cat([dat['y'] for dat in lmci_patient_data.values()], dim=0)
    sigma = all_biomarkers.var(dim=0)
    # 加上一个很小的数防止除以零
    sigma = sigma.clamp_min(1e-8)

    # ---------- 初始化 ----------
    model = PopulationODE(hidden_dim=16)
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

    # --------- Minibatch Dataloader Setup -----------
    patient_pids = list(lmci_patient_data.keys())
    use_minibatch = batch_size < len(patient_pids)

    batch_iterator = None
    if use_minibatch and n_adam > 0:
        print(f"Using mini-batching with batch size {batch_size}.")
        dataset = PatientDataset(patient_pids)
        
        sampler = None
        if weighted_sampling:
            print("Using weighted sampling based on the number of time points per patient.")
            weights = [float(lmci_patient_data[pid]['t'].shape[0]) for pid in patient_pids]
            sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=n_adam * batch_size, replacement=True)
        
        dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, shuffle=(sampler is None))
        batch_iterator = iter(dataloader)

    lbfgs_dataloader = None
    if use_minibatch and n_lbfgs > 0:
        print("Preparing mini-batching for L-BFGS phase.")
        if 'dataset' not in locals():
            dataset = PatientDataset(patient_pids)
        # For L-BFGS, we typically iterate over epochs. A simple shuffled dataloader is better.
        lbfgs_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    lbfgs_batch_iterator = None # Will be initialized inside loop

    ab = {}
    for pid, dat in lmci_patient_data.items():
        stage = stage_dict.get(pid, 'Other')
        t_mean = dat['t'].mean()

        if stage == 'CN':
            s_range = (-10.0, 0.0)
        elif stage == 'LMCI':
            s_range = (0.0, 10.0)
        elif stage == 'AD':
            s_range = (10.0, 20.0)
        else:  # Default for 'Other' or missing stages
            s_range = (-5.0, 5.0)
        
        # Initialize alpha to a random value around 1.0, ensuring it's positive
        alpha_init = torch.rand(1) * 0.4 + 0.8  # Range [0.8, 1.2]
        
        # Target an initial s randomly within the stage-specific range
        s_target = torch.rand(1) * (s_range[1] - s_range[0]) + s_range[0]
        
        # Calculate beta based on s = alpha*t + beta, using mean age
        beta_init = s_target - alpha_init * t_mean
        
        # Convert to theta space where alpha = exp(theta0)+eps and beta = theta1
        theta0 = torch.log(alpha_init - 1e-4)
        theta1 = beta_init

        ab[pid] = {'theta': torch.tensor([theta0.item(), theta1.item()], requires_grad=True)}

    # --------- Adam 优化器池 -----------
    opt_w_adam  = optim.Adam(model.parameters(), lr=adam_lr_w, weight_decay=1e-4)
    opt_ab_adam = {pid: optim.Adam([ab[pid]['theta']], lr=adam_lr_ab)
                   for pid in ab}
    scheduler = optim.lr_scheduler.MultiStepLR(opt_w_adam, milestones=[200,300,400,500,600,700,800,900,1000], gamma=0.5, last_epoch=-1)
    opt_w_lbfgs = optim.LBFGS(model.parameters(), 
                                lr=lbfgs_lr_w,
                                max_iter=max_lbfgs_it, 
                                tolerance_grad=tolerance_grad, 
                                tolerance_change=tolerance_change)
    opt_ab_lbfgs = {pid: optim.LBFGS([ab[pid]['theta']],
                                   lr=lbfgs_lr_ab,
                                   max_iter=max_lbfgs_it,
                                   tolerance_grad=tolerance_grad,
                                   tolerance_change=tolerance_change)
                    for pid in ab}

    # --------- 训练循环 -----------
    training_stopped = False
    # Early stopping state
    early_stop_counter = 0
    last_adam_loss = float('inf')
    adam_loss_history = []

    for it in range(n_adam + n_lbfgs):
        use_adam = it < n_adam

        # ======================== 更新 w =========================
        if use_adam:
            if use_minibatch:
                try:
                    batch_pids = next(batch_iterator)
                except StopIteration:
                    print("Batch iterator exhausted. This should not happen with the configured sampler.")
                    continue
            else:
                batch_pids = patient_pids

            opt_w_adam.zero_grad()
            loss_w = 0.
            
            # Convert tensor PIDs to integer for dict lookup
            batch_pids_list = [pid.item() for pid in batch_pids] if use_minibatch else batch_pids
            valid_pids_in_batch = [pid for pid in batch_pids_list if lmci_patient_data[pid]['t'].shape[0] >= 2]
            
            if not valid_pids_in_batch:
                continue

            for pid in valid_pids_in_batch:
                dat = lmci_patient_data[pid]
                loss_w += calculate_combined_loss(model, dat, ab[pid]['theta'], sigma=sigma, reg_lambda_smooth=reg_lambda_smooth)
            
            if len(valid_pids_in_batch) > 0:
                loss_w /= len(valid_pids_in_batch)

            # --- Early stopping and logging with moving average ---
            adam_loss_history.append(loss_w.item())
            if len(adam_loss_history) > 30: # Moving average window
                adam_loss_history.pop(0)
            current_avg_loss = sum(adam_loss_history) / len(adam_loss_history)

            # --- Early stopping logic ---
            if last_adam_loss != float('inf'):
                relative_change = (last_adam_loss - current_avg_loss) / abs(last_adam_loss) if last_adam_loss != 0 else float('inf')
                if relative_change < early_stop_threshold:
                    early_stop_counter += 1
                else:
                    early_stop_counter = 0
            last_adam_loss = current_avg_loss

            if early_stop_counter >= early_stop_patience:
                print(f"Iter {it+1:02d}: Early stopping. Loss improvement < {early_stop_threshold*100:.1f}% for {early_stop_patience} steps.")
                break
            # --- End early stopping logic ---

            if torch.isnan(loss_w):
                print(f"Iter {it+1:02d}: Adam loss for w is NaN. Stopping training.")
                training_stopped = True
            else:
                loss_w.backward()
                opt_w_adam.step()
                scheduler.step()
        else:  # L-BFGS
            # Reset early stopping state when not using Adam
            early_stop_counter = 0
            last_adam_loss = float('inf')

            if use_minibatch:
                if lbfgs_batch_iterator is None:
                    lbfgs_batch_iterator = iter(lbfgs_dataloader)
                try:
                    batch_pids = next(lbfgs_batch_iterator)
                except StopIteration:
                    # Epoch finished for L-BFGS, reset iterator
                    lbfgs_batch_iterator = iter(lbfgs_dataloader)
                    batch_pids = next(lbfgs_batch_iterator)
                
                batch_pids_list = [pid.item() for pid in batch_pids]
                valid_pids_in_batch = [p for p in batch_pids_list if lmci_patient_data[p]['t'].shape[0] >= 2]
                
                if not valid_pids_in_batch:
                    continue

                def closure_w():
                    opt_w_lbfgs.zero_grad()
                    loss = 0.
                    for pid in valid_pids_in_batch:
                        dat = lmci_patient_data[pid]
                        loss += calculate_combined_loss(model, dat, ab[pid]['theta'], sigma=sigma, reg_lambda_smooth=reg_lambda_smooth)
                    
                    if len(valid_pids_in_batch) > 0:
                        loss /= len(valid_pids_in_batch)

                    if not torch.isnan(loss):
                        loss.backward()
                    return loss
                loss_w = opt_w_lbfgs.step(closure_w)
            else:
                def closure_w():
                    opt_w_lbfgs.zero_grad()
                    loss = 0.
                    for pid, dat in lmci_patient_data.items():
                        loss += calculate_combined_loss(model, dat, ab[pid]['theta'], sigma=sigma, reg_lambda_smooth=reg_lambda_smooth)
                    if not torch.isnan(loss):
                        loss.backward()
                    return loss
                loss_w = opt_w_lbfgs.step(closure_w)

            if torch.isnan(loss_w):
                print(f"Iter {it+1:02d}: L-BFGS loss for w is NaN. Stopping training.")
                training_stopped = True

        if training_stopped:
            break

        # ---------- 可选：估计 σk ----------
        with torch.no_grad():
            all_residuals = []
            for pid, dat in lmci_patient_data.items():
                num_steps = dat['t'].shape[0]
                if num_steps < 2:
                    continue
                
                a = torch.exp(ab[pid]['theta'][0]) + 1e-4
                b = ab[pid]['theta'][1]
                s = a * dat['t'] + b
                
                for i in range(num_steps - 1):
                    s_grid_step = s[i:i+2]
                    y0_step = dat['y'][i]
                    y_pred_step = model(s_grid_step, y0_step)
                    residual = y_pred_step[1] - dat['y'][i+1]
                    all_residuals.append(residual)

            if all_residuals:
                all_residuals = torch.stack(all_residuals)
                # Variance of residuals for each biomarker
                sigma = all_residuals.var(dim=0)
            else:
                sigma = torch.ones(4) # Fallback if no residuals calculated


        # ====================== 更新 α,β ==========================
        if use_adam:
            current_pids_for_ab = valid_pids_in_batch
        elif use_minibatch: # LBFGS with minibatch
            current_pids_for_ab = valid_pids_in_batch
        else: # LBFGS full batch
            current_pids_for_ab = lmci_patient_data.keys()
            
        for pid in current_pids_for_ab:
            dat = lmci_patient_data[pid]
            if dat['t'].shape[0] < 2:
                continue

            if use_adam:
                opt_ab_adam[pid].zero_grad()
                loss_ab = calculate_combined_loss(model, dat, ab[pid]['theta'], sigma, reg_lambda_smooth)

                if torch.isnan(loss_ab):
                    print(f"Iter {it+1:02d}: Adam loss for α,β for pid {pid} is NaN. Stopping training.")
                    training_stopped = True
                    break
                loss_ab.backward()
                opt_ab_adam[pid].step()
            else:  # L-BFGS
                def closure_ab():
                    opt_ab_lbfgs[pid].zero_grad()
                    loss = calculate_combined_loss(model, dat, ab[pid]['theta'], sigma, reg_lambda_smooth=reg_lambda_smooth)
                    if not torch.isnan(loss):
                        loss.backward()
                    return loss
                loss_ab = opt_ab_lbfgs[pid].step(closure_ab)

                if torch.isnan(loss_ab):
                    print(f"Iter {it+1:02d}: L-BFGS loss for α,β for pid {pid} is NaN. Stopping training.")
                    training_stopped = True
                    break
        
        if training_stopped:
            break

        # ----------- 监控 ----------
        if (it+1) % 1 == 0:
            if use_adam:
                print(f"iter {it+1:02d}/{n_adam+n_lbfgs} | "
                      f"Adam | "
                      f"Batch MSE={loss_w.item():.4f} | "
                      f"Avg MSE={current_avg_loss:.4f}")
            else: # L-BFGS logging
                if use_minibatch:
                    print(f"iter {it+1:02d}/{n_adam+n_lbfgs} | "
                          f"LBFGS | "
                          f"Batch MSE={loss_w.item():.4f}")
                else:
                    with torch.no_grad():
                        total_loss = 0.
                        num_patients = 0
                        for pid, dat in lmci_patient_data.items():
                            if dat['t'].shape[0] >= 2:
                                total_loss += calculate_combined_loss(model, dat, ab[pid]['theta'], sigma=sigma)
                                num_patients += 1
                        
                        mean_loss = total_loss / num_patients if num_patients > 0 else 0.
                    print(f"iter {it+1:02d}/{n_adam+n_lbfgs} | "
                        f"LBFGS | "
                        f"MSE={mean_loss.item():.4f}")

    # --------- 输出 ----------
    model.eval() # Switch to evaluation mode before returning
    alpha_beta = {pid: (float(torch.exp(v['theta'][0])+1e-4),
                        float(v['theta'][1]))
                  for pid, v in ab.items()}
    return model, alpha_beta, lmci_patient_data

model_fix, ab_dict, lmci_patient_data = fit_population(
    patient_data,)

current_pid = os.getpid()
# Use the .save() method for JIT-scripted models
model_fix.save(f'model_{current_pid}.pt')

# 保存 alpha 和 beta 参数
import pickle
alpha_beta_filename = f'ab_{current_pid}.pkl'
with open(alpha_beta_filename, 'wb') as f:
    pickle.dump(ab_dict, f)
print(f"Alpha和Beta参数已保存到: {alpha_beta_filename}")

# 同时保存为可读的文本格式
alpha_beta_txt_filename = f'alpha_beta_{current_pid}.txt'
with open(alpha_beta_txt_filename, 'w') as f:
    f.write("Patient_ID\tAlpha\tBeta\n")
    for pid, (alpha, beta) in ab_dict.items():
        f.write(f"{pid}\t{alpha:.6f}\t{beta:.6f}\n")
print(f"Alpha和Beta参数(文本格式)已保存到: {alpha_beta_txt_filename}")

# ---------- 2. 使用评估接口进行模型评估 -----------------
# 导入评估模块
import eval

# 加载阶段字典
stage_dict = pc.load_stage_dict()

print("\n=== 开始模型评估 ===")

# 使用初值模式评估LMCI数据集
print("\n--- 评估LMCI数据集 (初值模式) ---")
eval_result_lmci = eval.evaluate_model_with_data(
    model=model_fix,
    patient_data=patient_data,  # 使用完整的patient_data，函数内部会过滤
    ab_dict=ab_dict,
    stage_dict=stage_dict,
    dataset_filter='LMCI',  # 只评估LMCI阶段
    mode='global',  # 使用初值模式
    output_prefix=f'lmci_{current_pid}',
    show_plot=False  # 不显示图片，只保存
)

if eval_result_lmci:
    print(f"LMCI评估结果已保存: {eval_result_lmci}")


print("\n=== 模型评估完成 ===")

# ---------- 计算CN、LMCI、AD三个数据集上的标准MSE ----------
print("\n计算各阶段的标准MSE...")

# 按阶段分组患者
stage_patients = {'CN': [], 'LMCI': [], 'AD': []}
# 修改：只考虑 LMCI 患者
for pid in lmci_patient_data.keys():
    stage = stage_dict.get(pid, 'Other')
    if stage in stage_patients:
        stage_patients[stage].append(pid)

# 计算每个阶段的标准MSE - 使用网格拟合方法
stage_mses = {}
with torch.no_grad():
    for stage, pids in stage_patients.items():
        if not pids:
            stage_mses[stage] = float('nan')
            print(f"{stage}: 无患者数据")
            continue
        
        # 收集该阶段所有患者的s和y数据点
        stage_s_points = []
        stage_y_points = []
        
        for pid in pids:
            dat = lmci_patient_data[pid]
            if dat['t'].shape[0] < 2:
                continue
                
            alpha, beta = ab_dict[pid]
            s_patient = alpha * dat['t'] + beta
            y_patient = dat['y']
            stage_s_points.append(s_patient)
            stage_y_points.append(y_patient)
        
        if not stage_s_points:
            stage_mses[stage] = float('nan')
            print(f"{stage}: 无有效患者数据")
            continue
            
        # 连接该阶段所有数据点
        stage_s_points = torch.cat(stage_s_points)
        stage_y_points = torch.cat(stage_y_points, dim=0)
        
        # 确定该阶段的s范围
        stage_s_min = stage_s_points.min()
        stage_s_max = stage_s_points.max()
        
        # 生成5个网格点
        s_grid = torch.linspace(stage_s_min.item(), stage_s_max.item(), 5)
        window_size = 0.2 * (stage_s_max - stage_s_min)  # 窗口大小为s范围的20%
        
        # 使用真实均值拟合轨迹
        y_fitted = []
        
        for s_point in s_grid:
            # 在当前s点附近寻找数据点并计算真实均值
            mask = torch.abs(stage_s_points - s_point) <= window_size
            if mask.sum() > 0:
                y_local_mean = stage_y_points[mask].mean(dim=0)
                y_fitted.append(y_local_mean)
            else:
                # 如果没有附近的数据点，使用最近的数据点
                distances = torch.abs(stage_s_points - s_point)
                closest_idx = torch.argmin(distances)
                y_fitted.append(stage_y_points[closest_idx])
        
        y_fitted = torch.stack(y_fitted)  # (5, 4)
        
        # 使用模型预测相同的网格点
        # 使用该阶段所有患者y0的均值作为初始点
        y0_stage = torch.stack([lmci_patient_data[pid]['y0'] for pid in pids if lmci_patient_data[pid]['t'].shape[0] >= 2]).mean(0)
        y_predicted = model_fix(s_grid, y0_stage)  # (5, 4)
        
        # 计算标准MSE
        mse = torch.mean((y_predicted - y_fitted) ** 2).item()
        stage_mses[stage] = mse
        
        print(f"{stage}: {len(pids)}个患者, 5个网格点, MSE = {mse:.6f}")
        print(f"  s范围: [{stage_s_min.item():.3f}, {stage_s_max.item():.3f}], 窗口大小: {window_size:.3f}")

# 获取进程PID
current_pid = os.getpid()

# 追加到输出文件
output_filename = f'results.out'
with open(output_filename, 'a') as f:  # 使用追加模式 'a'
    f.write(f"Process PID: {current_pid}\n")
    f.write(f"CN MSE: {stage_mses.get('CN', 'N/A')}\n")
    f.write(f"LMCI MSE: {stage_mses.get('LMCI', 'N/A')}\n")
    f.write(f"AD MSE: {stage_mses.get('AD', 'N/A')}\n")


print(f"\n结果已保存到文件: {output_filename}")
print(f"进程PID: {current_pid}")
print(f"CN MSE: {stage_mses.get('CN', 'N/A')}")
print(f"LMCI MSE: {stage_mses.get('LMCI', 'N/A')}")
print(f"AD MSE: {stage_mses.get('AD', 'N/A')}")

# ---------- 从分布中随机抽取初值画多条轨迹 ----------
print("\n=== 绘制多条随机初值轨迹 ===")

# 收集所有LMCI患者的初值
all_y0_lmci = torch.stack([dat['y0'] for dat in lmci_patient_data.values()])  # (N, 4)

# 计算初值的均值和标准差
y0_mean = all_y0_lmci.mean(dim=0)  # (4,)
y0_std = all_y0_lmci.std(dim=0)    # (4,)

print(f"LMCI患者初值统计:")
print(f"均值: {y0_mean.numpy()}")
print(f"标准差: {y0_std.numpy()}")

# 设置s范围（使用之前计算的范围）
all_s_values_lmci = torch.cat([ab_dict[p][0] * dat['t'] + ab_dict[p][1] for p, dat in lmci_patient_data.items()])
s_min_multi = torch.quantile(all_s_values_lmci, 0.05)
s_max_multi = torch.quantile(all_s_values_lmci, 0.95)
s_curve_multi = torch.linspace(s_min_multi.item(), s_max_multi.item(), 100)

print(f"s范围: [{s_min_multi.item():.3f}, {s_max_multi.item():.3f}]")

# 随机抽取多个初值
num_trajectories = 8  # 画8条轨迹
torch.manual_seed(42)  # 设置随机种子以便复现

# 方法1: 从正态分布采样
random_y0_normal = []
for i in range(num_trajectories):
    # 从正态分布采样，使用实际数据的均值和标准差
    y0_sample = torch.normal(y0_mean, y0_std * 0.5)  # 使用0.5倍标准差避免过于极端的值
    # 确保值在合理范围内
    y0_sample = torch.clamp(y0_sample, 0.0, 3.0)
    random_y0_normal.append(y0_sample)

# 方法2: 从实际数据中随机选择
random_indices = torch.randperm(len(all_y0_lmci))[:num_trajectories//2]
random_y0_real = [all_y0_lmci[i] for i in random_indices]

# 合并两种方法的初值
all_random_y0 = random_y0_normal + random_y0_real

print(f"生成了 {len(all_random_y0)} 个随机初值")

# 使用模型预测所有轨迹
trajectories = []
with torch.no_grad():
    for i, y0 in enumerate(all_random_y0):
        traj = model_fix(s_curve_multi, y0)
        trajectories.append(traj.numpy())
        print(f"轨迹 {i+1}: 初值 = {y0.numpy()}")

trajectories = np.array(trajectories)  # (num_trajectories, len_s, 4)

# 绘制多轨迹图
TITLES = ['Aβ (A)', 'p-Tau (T)', 'Neurodeg. (N)', 'Cognition (C)']
colors_traj = plt.cm.tab10(np.linspace(0, 1, len(all_random_y0)))

fig_multi, axes_multi = plt.subplots(2, 2, figsize=(14, 10))

for k, ax in enumerate(axes_multi.flat):
    # 绘制真实数据点（背景）
    for pid, dat in lmci_patient_data.items():
        if pid in ab_dict:
            alpha, beta = ab_dict[pid]
            s_patient = alpha * dat['t'] + beta
            y_patient = dat['y'][:, k]
            ax.scatter(s_patient.numpy(), y_patient.numpy(), 
                      s=8, alpha=0.3, c='lightgray', label='Real Data' if pid == list(lmci_patient_data.keys())[0] else "")
    
    # 绘制多条预测轨迹
    for i, traj in enumerate(trajectories):
        label = f'Trajectory {i+1}' if i < 3 else ""  # 只为前3条轨迹添加标签，避免图例过于拥挤
        ax.plot(s_curve_multi.numpy(), traj[:, k], 
               color=colors_traj[i], linewidth=2, alpha=0.8, label=label)
    
    # 绘制均值轨迹（参考）
    mean_traj = model_fix(s_curve_multi, y0_mean).detach().numpy()
    ax.plot(s_curve_multi.numpy(), mean_traj[:, k], 
           'k--', linewidth=3, alpha=0.9, label='Mean Trajectory')
    
    ax.set_xlabel('Disease progression score s')
    ax.set_ylabel(TITLES[k])
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title(f'{TITLES[k]} - Multiple Random Initial Values')

fig_multi.suptitle(f'Multiple Trajectories from Random Initial Values (n={len(all_random_y0)})', fontsize=14)
plt.tight_layout()

# 保存图片
multi_traj_filename = f'multi_traj_{current_pid}.png'
plt.savefig(multi_traj_filename, dpi=300, bbox_inches='tight')
print(f"\n多轨迹图已保存: {multi_traj_filename}")

# 显示图片
plt.show()

# ---------- 分析轨迹的变异性 ----------
print("\n=== 轨迹变异性分析 ===")

# 计算每个时间点的标准差
traj_std = np.std(trajectories, axis=0)  # (len_s, 4)
traj_mean = np.mean(trajectories, axis=0)  # (len_s, 4)

# 绘制变异性图
fig_var, axes_var = plt.subplots(2, 2, figsize=(14, 10))

for k, ax in enumerate(axes_var.flat):
    # 绘制均值轨迹
    ax.plot(s_curve_multi.numpy(), traj_mean[:, k], 'b-', linewidth=3, label='Mean')
    
    # 绘制置信区间 (均值 ± 标准差)
    ax.fill_between(s_curve_multi.numpy(), 
                   traj_mean[:, k] - traj_std[:, k], 
                   traj_mean[:, k] + traj_std[:, k], 
                   alpha=0.3, color='blue', label='±1 STD')
    
    # 绘制置信区间 (均值 ± 2倍标准差)
    ax.fill_between(s_curve_multi.numpy(), 
                   traj_mean[:, k] - 2*traj_std[:, k], 
                   traj_mean[:, k] + 2*traj_std[:, k], 
                   alpha=0.2, color='blue', label='±2 STD')
    
    ax.set_xlabel('Disease progression score s')
    ax.set_ylabel(TITLES[k])
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title(f'{TITLES[k]} - Trajectory Variability')
    
    # 打印变异性统计
    mean_std = np.mean(traj_std[:, k])
    max_std = np.max(traj_std[:, k])
    print(f"{TITLES[k]}: 平均标准差={mean_std:.4f}, 最大标准差={max_std:.4f}")

fig_var.suptitle('Trajectory Variability Analysis', fontsize=14)
plt.tight_layout()

# 保存变异性图
var_filename = f'traj_var_{current_pid}.png'
plt.savefig(var_filename, dpi=300, bbox_inches='tight')
print(f"变异性分析图已保存: {var_filename}")

plt.show()

print(f"\n=== 多轨迹分析完成 ===")
