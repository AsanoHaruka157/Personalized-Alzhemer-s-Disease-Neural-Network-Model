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
from datetime import datetime

# ------------------ 加载数据 ------------------
csf_dict = pc.load_csf_data()
print("Number of valid patients:", len(csf_dict))

patient_data = {}
for pid, sample in csf_dict.items():
    t = torch.from_numpy(sample[:, 0]).float()            # 年龄
    y = torch.from_numpy(sample[:, 1:5]).float()          # biomarker A/T/N/C
    patient_data[pid] = {"t": t, "y": y, "y0": y[0].clone()}

import torch.nn as nn

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

Message = f"This is a hybrid model with polynomial terms and FNN with activation Tanh"
name = 'fnn'

class ODEModel(nn.Module):
    def __init__(self, hidden_dim=32, num_layers=2):
        super().__init__()

        # 1. 输入层：5个输入（4个生物标记物 + s值）-> hidden_dim
        self.net = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 4),
            nn.Sigmoid()
        )
        
        # 2. 多项式系数：15个可学习参数
        # g(y0) = a0*y0^2 + a1*y0 + a2
        # g(y1) = b0*y1^2 + b1*y0*y1 + b2*y1 + b3
        # g(y2) = c0*y2^2 + c1*y1*y2 + c2*y2 + c3
        # g(y3) = d0*y3^2 + d1*y2*y3 + d2*y3 + d3
        self.poly_coeffs = nn.Parameter(torch.zeros(15))  # 初始化15个系数


    def f(self, y: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        y = y.float()
        s = s.float()
        # 确保s是标量，将其扩展为与y兼容的形状
        if s.dim() == 0:  # s是标量
            s_expanded = s.unsqueeze(0)  # 变成[1]
        else:
            s_expanded = s
        
        # 将y和s连接起来作为网络输入
        z = torch.cat([y, s_expanded], dim=0)  # 连接为[5]的向量
        
        # 神经网络部分
        net_output = self.net(z)
        
        # 多项式部分 g(y)
        y0, y1, y2, y3 = y[0], y[1], y[2], y[3]
        
        # 提取多项式系数
        a0, a1, a2 = self.poly_coeffs[0], self.poly_coeffs[1], self.poly_coeffs[2]
        b0, b1, b2, b3 = self.poly_coeffs[3], self.poly_coeffs[4], self.poly_coeffs[5], self.poly_coeffs[6]
        c0, c1, c2, c3 = self.poly_coeffs[7], self.poly_coeffs[8], self.poly_coeffs[9], self.poly_coeffs[10]
        d0, d1, d2, d3 = self.poly_coeffs[11], self.poly_coeffs[12], self.poly_coeffs[13], self.poly_coeffs[14]
        
        # 计算多项式项
        g0 = a0 * y0**2 + a1 * y0 + a2
        g1 = b0 * y1**2 + b1 * y0 * y1 + b2 * y1 + b3
        g2 = c0 * y2**2 + c1 * y1 * y2 + c2 * y2 + c3
        g3 = d0 * y3**2 + d1 * y2 * y3 + d2 * y3 + d3
        
        g_y = torch.stack([g0, g1, g2, g3])
        
        # 最终输出：f(y,s) = g(y) + net(y,s)
        return g_y + net_output

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
    # 需要为每个时间点调用f函数
    fy = torch.zeros_like(y)
    for i in range(len(s)):
        fy[i] = model.f(y[i], s[i])

    residual = dy_dt - fy
    loss = residual ** 2
    if sigma is not None:
        loss = loss * sigma
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
        loss = loss * sigma
        
    return loss.mean()

def calculate_combined_loss(model, dat, ab_pid_theta, sigma=None, r=0.5):
    """
    Calculates a combined loss: r% residual and (1-r)% global.
    """
    res = residual(model, dat, ab_pid_theta, sigma)
    loss_global = calculate_global_loss(model, dat, ab_pid_theta, sigma)
    '''
    zero = torch.zeros(4)
    one = torch.ones(4)
    s_default = torch.tensor(0.0)
    combined_loss = ( loss_global + res ) * torch.exp(torch.norm(model.f(B, s_default)) + torch.norm(model.f(D, s_default))) / (torch.norm(model.f(C, s_default)) + 1e-5)
    '''
    combined_loss = r * res + (1-r) * loss_global
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
        n_adam      = 120,      # adam 阶段迭代次数
        n_lbfgs    = 0,     # lbfgs 阶段迭代次数
        adam_lr_w    = 5e-3,
        adam_lr_ab   = 1e-4,
        lbfgs_lr_w   = 1e-2,
        lbfgs_lr_ab  = 1e-2,
        batch_size=128,
        weighted_sampling=True,
        delta = 30,
        max_lbfgs_it = 10,
        tolerance_grad = 0,
        tolerance_change = 0,
        reg_lambda_alpha=1e-2,
        reg_lambda_smooth=0.,
        early_stop_patience=80,
        early_stop_threshold=0.001):
    sigma = torch.ones(4)

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

    # --------- Minibatch Dataloader Setup -----------
    patient_pids = list(patient_data.keys())
    use_minibatch = batch_size < len(patient_pids)

    batch_iterator = None
    if use_minibatch and n_adam > 0:
        print(f"Using mini-batching with batch size {batch_size}.")
        dataset = PatientDataset(patient_pids)
        
        sampler = None
        if weighted_sampling:
            print("Using weighted sampling based on the number of time points per patient.")
            weights = [float(patient_data[pid]['t'].shape[0]) for pid in patient_pids]
            sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=n_adam * batch_size, replacement=True)
        
        dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, shuffle=(sampler is None))
        batch_iterator = iter(dataloader)

    stage_dict = pc.load_stage_dict()
    ab = {}
    for pid, dat in patient_data.items():
        stage = stage_dict.get(pid, 'Other')
        t = dat['t']

        if stage == 'CN':
            s_range = (-10.0, 0.0)
        elif stage == 'LMCI':
            s_range = (0.0, 10.0)
        elif stage == 'AD':
            s_range = (10.0, 20.0)
        else:  # Default for 'Other' or missing stages
            s_range = (-5.0, 5.0)
        
        alpha_init = torch.empty(1).uniform_(1, torch.exp(torch.tensor(4.0)))
        alpha_init = torch.min( alpha_init, 10/(t.max()-t.min()))
        # Convert to theta space where alpha = exp(theta0)+eps and beta = theta1
        theta0 = torch.log(alpha_init - 1e-4)
        beta_init = torch.empty(1).uniform_(s_range[0]-theta0.item()*t.min().item(), s_range[1]-theta0.item()*t.max().item())
        theta1 = beta_init

        ab[pid] = {'theta': torch.tensor([theta0.item(), theta1.item()], requires_grad=True)}

    # --------- Adam 优化器池 -----------
    opt_w_adam  = optim.Adam(model.parameters(), lr=adam_lr_w, weight_decay=1e-4)
    opt_ab_adam = {pid: optim.Adam([ab[pid]['theta']], lr=adam_lr_ab)
                   for pid in ab}
    scheduler = optim.lr_scheduler.MultiStepLR(opt_w_adam, milestones=list(range(60, 151)), gamma=0.99, last_epoch=-1)
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
    
    # 模型备份相关变量
    model_backup = None
    ab_backup = None
    backup_iter = 0

    # 交替更新：Adam更新w 10轮，然后L-BFGS更新a,b一轮
    for it in range(n_adam):
        # ======================== 更新 w (Adam) =========================
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
        valid_pids_in_batch = [pid for pid in batch_pids_list if patient_data[pid]['t'].shape[0] >= 2]
        
        if not valid_pids_in_batch:
            continue

        for pid in valid_pids_in_batch:
            dat = patient_data[pid]
            loss_w += calculate_combined_loss(model, dat, ab[pid]['theta'], sigma=sigma)
        
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

        # ====================== 每delta轮更新一次 α,β (L-BFGS) ==========================
        if (it + 1) % delta == 0:
            print(f"Iter {it+1:02d}: Updating α,β with L-BFGS...")
            
            # 保存模型和ab的备份
            model_backup = copy.deepcopy(model.state_dict())
            ab_backup = {pid: ab[pid]['theta'].clone() for pid in ab}
            backup_iter = it + 1
            print(f"Iter {it+1:02d}: Model backup saved.")
            
            # 使用所有患者数据更新a,b
            nan_detected = False
            for pid in patient_data.keys():
                dat = patient_data[pid]
                if dat['t'].shape[0] < 2:
                    continue

                def closure_ab():
                    opt_ab_lbfgs[pid].zero_grad()
                    loss = calculate_combined_loss(model, dat, ab[pid]['theta'], sigma)
                    if not torch.isnan(loss):
                        loss.backward()
                    return loss
                
                try:
                    loss_ab = opt_ab_lbfgs[pid].step(closure_ab)
                    if torch.isnan(loss_ab):
                        print(f"Iter {it+1:02d}: L-BFGS loss for α,β for pid {pid} is NaN.")
                        nan_detected = True
                        break
                except Exception as e:
                    print(f"Iter {it+1:02d}: L-BFGS failed for pid {pid}: {e}")
                    nan_detected = True
                    break
            
            # 如果检测到NaN，恢复备份
            if nan_detected:
                print(f"Iter {it+1:02d}: NaN detected, restoring model from backup (iter {backup_iter}).")
                model.load_state_dict(model_backup)
                for pid in ab:
                    ab[pid]['theta'] = ab_backup[pid].clone()
                print(f"Iter {it+1:02d}: Model restored successfully.")
            else:
                print(f"Iter {it+1:02d}: L-BFGS update completed successfully.")

        if training_stopped:
            break

        # ---------- 可选：估计 σk ----------
        with torch.no_grad():
            all_residuals = []
            for pid, dat in patient_data.items():
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

        # ----------- 监控 ----------
        if (it+1) % 1 == 0:
            print(f"iter {it+1:02d}/{n_adam} | "
                  f"Adam | "
                  f"Batch MSE={loss_w.item():.4f} | "
                  f"Avg MSE={current_avg_loss:.4f}")

    # --------- 输出 ----------
    model.eval() # Switch to evaluation mode before returning
    
    # 检查是否使用了备份模型
    if model_backup is not None:
        print(f"Training completed. Last backup was saved at iteration {backup_iter}.")
    
    alpha_beta = {pid: (float(torch.exp(v['theta'][0])+1e-4),
                        float(v['theta'][1]))
                  for pid, v in ab.items()}
    return model, alpha_beta

model, ab_dict = fit_population(
    patient_data,)

try:
    torch.save(model.state_dict(), f'{name}.pth')
except Exception as e:
    print(f"Error saving model: {e} 喵！   _(┐ ◟;ﾟдﾟ)ノ")

# ---------- 2. 绘制人群四联图 (根据s的10%和90%分位数) -----------------
# 收集所有患者的s值
all_s_values = []
for p in patient_data:
    s_values = ab_dict[p][0] * patient_data[p]['t'] + ab_dict[p][1]
    all_s_values.append(s_values)

# 计算所有s值的10分位数和90分位数
all_s_flat = torch.cat(all_s_values)
s_10_percentile = torch.quantile(all_s_flat, 0.10)
s_90_percentile = torch.quantile(all_s_flat, 0.90)

print(f"S value range: 10th percentile = {s_10_percentile:.2f}, 90th percentile = {s_90_percentile:.2f}")

# 使用10分位数到90分位数的范围
s_min = s_10_percentile
s_max = s_90_percentile
s_curve = torch.linspace(s_min, s_max, 100)

# 过滤在10-90分位数范围内的数据点
keep = []
for p in patient_data:
    s_values = ab_dict[p][0] * patient_data[p]['t'] + ab_dict[p][1]
    # 检查是否有任何s值在范围内
    if torch.any((s_values >= s_min) & (s_values <= s_max)):
        keep.append(p)

stage_dict = pc.load_stage_dict()

# 准备绘图数据 - 使用10分位数对应的y值作为起始点
y0_pop = torch.tensor([0.1,0,0,0])

# ---------- 添加新轨迹：从10分位数开始的完整轨迹 ----------
with torch.no_grad():
    y_curve_full = model(s_curve, y0_pop)
    y_curve_full = y_curve_full.detach().numpy()
y_curve_full = pc.inv_nor(y_curve_full)

TITLES = ['Aβ (A)', 'p-Tau (T)', 'N', 'Cognition (C)']

fig2, axes = plt.subplots(2, 2, figsize=(9, 6))
for k, ax in enumerate(axes.flat):
    # --- 分阶段准备散点数据 ---
    s_by_stage = {'CN': [], 'LMCI': [], 'AD': [], 'Other': []}
    y_by_stage = {'CN': [], 'LMCI': [], 'AD': [], 'Other': []}

    for p in keep:
        a, b = ab_dict[p]
        stage = stage_dict.get(p, 'Other')
        if stage not in s_by_stage:
            stage = 'Other'
        
        s_values = a * patient_data[p]['t'] + b
        y_values = patient_data[p]['y'][:, k]
        
        # 只保留在10-90分位数范围内的数据点
        mask = (s_values >= s_min) & (s_values <= s_max)
        if torch.any(mask):
            s_by_stage[stage].append(s_values[mask])
            y_by_stage[stage].append(y_values[mask])

    # --- 绘制散点 (分期颜色) ---
    colors = {'CN': 'orange', 'LMCI': 'green', 'AD': 'blue', 'Other': 'grey'}
    for stage, s_points_list in s_by_stage.items():
        if s_points_list:
            s_all = torch.cat(s_points_list).numpy()
            y_all = torch.cat(y_by_stage[stage]).numpy()
            y_all = pc.inv_nor(y_all, k)
            ax.scatter(s_all, y_all, s=15, alpha=0.6, c=colors[stage], label=stage)
    
    # --- 从平均初值开始的完整轨迹 ---
    y_curve = y_curve_full[:,k]
    ax.plot(s_curve, y_curve, lw=1.5, c='red', linestyle='--', label='Trajectory from Mean Initial')
    
    ax.set_xlabel('Disease progression score  s')
    ax.set_ylabel(TITLES[k])
    ax.legend(fontsize=8)

fig2.suptitle(f'Population Model (s in 10-90 percentile: [{float(s_min):.2f}, {float(s_max):.2f}])')
plt.tight_layout(rect=[0,0,1,0.96])
plt.savefig(f'{name}.png')

plt.show()

def eval_global_loss(y_pred, y_true):
    y_pred_t = torch.as_tensor(y_pred)
    return torch.mean((y_pred_t - y_true) ** 2)
loss = 0
with torch.no_grad():
    for pid in patient_data:
        a, b = ab_dict[pid]
        s = a * patient_data[pid]['t'] + b
        y_pred = model(s, patient_data[pid]['y0'])
        y_pred = y_pred.numpy()
        loss += eval_global_loss(y_pred, patient_data[pid]['y'])/len(y_pred)
    loss /= len(patient_data)

current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# 追加到输出文件
output_filename = f'experiments.out'
with open(output_filename, 'a') as f:  # 使用追加模式 'a'
    f.write(name)
    f.write(f"Time: {current_time}\n")
    f.write(Message)
    f.write("Model structure:\n")
    f.write(str(model))
    f.write(f"MSE: {loss:.4f}")