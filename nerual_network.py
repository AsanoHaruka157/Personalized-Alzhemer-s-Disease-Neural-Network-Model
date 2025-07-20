
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import pccmnn as pc

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

class PopulationODE(nn.Module):
    def __init__(self, hidden_dim=32):
        super().__init__()
        # A 1-hidden-layer neural network to replace the polynomial model
        self.net = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 4)
        )

    def f(self, y):
        """ The ODE function, now defined by a neural network. """
        return self.net(y)

    def forward(self, s_grid, y0):
        # Simple Euler integration to solve the ODE
        ys = [y0]
        for i in range(1, len(s_grid)):
            h = s_grid[i] - s_grid[i-1]
            ys.append(ys[-1] + h * self.f(ys[-1]))
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

def calculate_combined_loss(model, dat, ab_pid_theta, sigma=None):
    """
    Calculates a combined loss: 50% sequential and 50% global.
    """
    loss_seq = calculate_sequential_loss(model, dat, ab_pid_theta, sigma)
    loss_global = calculate_global_loss(model, dat, ab_pid_theta, sigma)
    return 0.5 * loss_seq + 0.5 * loss_global


import torch, torch.nn as nn, torch.optim as optim
import torch.nn.functional as F
from math import ceil

def fit_population(
        patient_data,
        n_adam      = 220,      # adam 阶段迭代次数
        n_lbfgs    = 80,     # lbfgs 阶段迭代次数
        adam_lr_w    = 5e-3,    # Adjusted learning rate for the neural network
        adam_lr_ab   = 5e-3,
        lbfgs_lr_w   = 1e-2,
        lbfgs_lr_ab  = 1e-2,
        max_lbfgs_it = 10,
        tolerance_grad = 0,
        tolerance_change = 0):
    
    # ---------- 计算每个生物标记物的全局方差以调整权重 ----------
    all_biomarkers = torch.cat([dat['y'] for dat in patient_data.values()], dim=0)
    sigma = all_biomarkers.var(dim=0)
    # 加上一个很小的数防止除以零
    sigma = sigma.clamp_min(1e-8)

    # ---------- 初始化 ----------
    model = PopulationODE()
    def weights_init(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    model.apply(weights_init)
    
    stage_dict = pc.load_stage_dict()
    ab = {}
    for pid, dat in patient_data.items():
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
    opt_w_adam  = optim.Adam(model.parameters(), lr=adam_lr_w)
    opt_ab_adam = {pid: optim.Adam([ab[pid]['theta']], lr=adam_lr_ab)
                   for pid in ab}
    scheduler = optim.lr_scheduler.MultiStepLR(opt_w_adam, milestones=[100], gamma=0.5, last_epoch=-1)

    # --------- 训练循环 -----------
    training_stopped = False
    for it in range(n_adam + n_lbfgs):
        use_adam = it < n_adam

        # ======================== 更新 w =========================
        if use_adam:
            opt_w_adam.zero_grad()
            loss_w = 0.
            for pid, dat in patient_data.items():
                loss_w += calculate_combined_loss(model, dat, ab[pid]['theta'], sigma=sigma)
            
            if torch.isnan(loss_w):
                print(f"Iter {it+1:02d}: Adam loss for w is NaN. Stopping training.")
                training_stopped = True
            else:
                loss_w.backward()
                opt_w_adam.step()
                scheduler.step()
        else:  # L-BFGS
            opt_w = optim.LBFGS(model.parameters(), 
                                lr=lbfgs_lr_w,
                                max_iter=max_lbfgs_it, 
                                tolerance_grad=tolerance_grad, 
                                tolerance_change=tolerance_change)

            def closure_w():
                opt_w.zero_grad()
                loss = 0.
                for pid, dat in patient_data.items():
                    loss += calculate_combined_loss(model, dat, ab[pid]['theta'], sigma=sigma)
                if not torch.isnan(loss):
                    loss.backward()
                return loss
            loss_w = opt_w.step(closure_w)

            if torch.isnan(loss_w):
                print(f"Iter {it+1:02d}: L-BFGS loss for w is NaN. Stopping training.")
                training_stopped = True

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


        # ====================== 更新 α,β ==========================
        for pid, dat in patient_data.items():
            if use_adam:
                opt_ab_adam[pid].zero_grad()
                loss_ab = calculate_combined_loss(model, dat, ab[pid]['theta'], sigma)

                if torch.isnan(loss_ab):
                    print(f"Iter {it+1:02d}: Adam loss for α,β for pid {pid} is NaN. Stopping training.")
                    training_stopped = True
                    break
                loss_ab.backward()
                opt_ab_adam[pid].step()
            else:  # L-BFGS
                opt_ab = optim.LBFGS([ab[pid]['theta']],
                                     lr=lbfgs_lr_ab,
                                     max_iter=max_lbfgs_it, 
                                     tolerance_grad=tolerance_grad, 
                                     tolerance_change=tolerance_change)

                def closure_ab():
                    opt_ab.zero_grad()
                    loss = calculate_combined_loss(model, dat, ab[pid]['theta'], sigma)
                    if not torch.isnan(loss):
                        loss.backward()
                    return loss
                loss_ab = opt_ab.step(closure_ab)

                if torch.isnan(loss_ab):
                    print(f"Iter {it+1:02d}: L-BFGS loss for α,β for pid {pid} is NaN. Stopping training.")
                    training_stopped = True
                    break
        
        if training_stopped:
            break

        # ----------- 监控 ----------
        if (it+1) % 1 == 0:
            with torch.no_grad():
                total_loss = 0.
                num_patients = 0
                for pid, dat in patient_data.items():
                    if dat['t'].shape[0] >= 2:
                         total_loss += calculate_combined_loss(model, dat, ab[pid]['theta'])
                         num_patients += 1
                
                mean_loss = total_loss / num_patients if num_patients > 0 else 0.

            print(f"iter {it+1:02d}/{n_adam+n_lbfgs} | "
                  f"{'Adam' if use_adam else 'LBFGS'} | "
                  f"MSE={mean_loss.item():.4f}")

    # --------- 输出 ----------
    model.eval() # Switch to evaluation mode before returning
    alpha_beta = {pid: (float(torch.exp(v['theta'][0])+1e-4),
                        float(v['theta'][1]))
                  for pid, v in ab.items()}
    return model, alpha_beta

model_fix, ab_dict = fit_population(
    patient_data,)

# ---------- 2. 绘制人群四联图 （仅绘制 |s|≤bound 的病例） -----------------
# -------- 过滤病例 ---------
# s的限界
bound = 20
keep = [p for p in patient_data
        if (torch.abs(ab_dict[p][0] * patient_data[p]['t']
                      + ab_dict[p][1]).max() < bound)]

stage_dict = pc.load_stage_dict()

# The trained model is now directly returned from the fit_population function.
# No need to instantiate a new model and load parameters manually.

# 准备绘图数据
y0_pop = torch.stack([patient_data[p]['y0'] for p in keep]).mean(0)
s_curve = torch.linspace(-10, 20, 400)
y_curve = model_fix(s_curve, y0_pop).detach().numpy()

TITLES = ['Aβ (A)', 'p-Tau (T)', 'Neurodeg. (N)', 'Cognition (C)']

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
        
        s_by_stage[stage].append(a * patient_data[p]['t'] + b)
        y_by_stage[stage].append(patient_data[p]['y'][:, k])

    # --- 绘制散点 (分期颜色) ---
    colors = {'CN': 'orange', 'LMCI': 'green', 'AD': 'blue', 'Other': 'grey'}
    for stage, s_points_list in s_by_stage.items():
        if s_points_list:
            s_all = torch.cat(s_points_list).numpy()
            y_all = torch.cat(y_by_stage[stage]).numpy()
            ax.scatter(s_all, y_all, s=15, alpha=0.6, c=colors[stage], label=stage)

    # --- 曲线 (平均轨迹) ---
    ax.plot(s_curve, y_curve[:, k], lw=1.5, c='black', label='Mean Trajectory')
    ax.set_xlabel('Disease progression score  s')
    ax.set_ylabel(TITLES[k])
    ax.legend()

fig2.suptitle(f'Population biomarker trajectories (|s| ≤ {bound})')
plt.tight_layout(rect=[0,0,1,0.96])
plt.savefig('nn.png')
plt.show()