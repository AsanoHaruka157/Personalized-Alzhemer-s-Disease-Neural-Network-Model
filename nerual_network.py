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

class PopulationODE(nn.Module):
    def __init__(self, hidden_dim=32):
        super().__init__()
        
        # 1. 量级网络 (Magnitude Network)
        #    决定变化的最大速度和方向 (+/-)。
        #    最后一层是线性层，以便输出任意实数。
        self.magnitude_net = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 4),
        )

        # 2. 门控网络 (Gating Network)
        #    它的任务是学习一个从 y 到 z 的映射。
        #    这个 z 将作为高斯函数的输入，从而控制门控的开关。
        self.gating_net = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.Tanh(),
            # 使用LayerNorm可以稳定中间层的激活值分布，有助于训练
            nn.LayerNorm(hidden_dim), 
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.LayerNorm(hidden_dim),
            # 最终输出一个控制信号 z，这个z的中心可以由该层的偏置项学习
            nn.Linear(hidden_dim, 4) 
        )
        
        # 3. 高斯激活层
        self.gaussian_activation = Gaussian()

    def f(self, y):
        """
        ODE 函数: dy/ds = f(y)
        f(y) = magnitude * gate
        """
        
        # 计算变化的潜在“量级”和“方向”
        # 输出范围是(-inf, +inf)
        magnitude = self.magnitude_net(y)
        
        # 通过门控网络计算高斯函数的输入 z
        # 网络会学习如何调整 z 的值，使得当 y 处于某个关键点时，z 接近 0
        gate_input = self.gating_net(y)
        
        # 计算高斯门控的输出值，范围是 (0, 1]
        # 当 gate_input 接近 0 时，gate_output 接近 1 (阀门打开)
        # 当 gate_input 远离 0 时，gate_output 接近 0 (阀门关闭)
        gate_output = self.gaussian_activation(gate_input)
        
        # 最终导数 = 量级 * 门控
        # 实现了在特定区间(gate_output ≈ 1)发生剧烈变化，在其他区间(gate_output ≈ 0)保持平稳
        return magnitude * gate_output

    def forward(self, s_grid, y0):
        # 4th-order Runge-Kutta integration to solve the ODE
        ys = [y0]
        for i in range(1, len(s_grid)):
            h = s_grid[i] - s_grid[i-1]
            y_i = ys[-1]
            
            k1 = self.f(y_i)
            k2 = self.f(y_i + 0.5 * h * k1)
            k3 = self.f(y_i + 0.5 * h * k2)
            k4 = self.f(y_i + h * k3)
            
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
    loss_seq = calculate_sequential_loss(model, dat, ab_pid_theta, sigma)
    loss_global = calculate_global_loss(model, dat, ab_pid_theta, sigma)
    
    combined_loss = 0.5 * loss_seq + 0.5 * loss_global
        
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
        n_adam      = 250,      # adam 阶段迭代次数
        n_lbfgs    = 50,     # lbfgs 阶段迭代次数
        adam_lr_w    = 1e-2,
        adam_lr_ab   = 1e-3,
        lbfgs_lr_w   = 1e-2,
        lbfgs_lr_ab  = 1e-2,
        batch_size=64,
        weighted_sampling=True,
        max_lbfgs_it = 10,
        tolerance_grad = 0,
        tolerance_change = 0,
        reg_lambda_alpha=1e-2,
        reg_lambda_smooth=1e-2,
        early_stop_patience=100,
        early_stop_threshold=0.005):
    
    # ---------- 计算每个生物标记物的全局方差以调整权重 ----------
    all_biomarkers = torch.cat([dat['y'] for dat in patient_data.values()], dim=0)
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
    scheduler = optim.lr_scheduler.MultiStepLR(opt_w_adam, milestones=[100,200,300], gamma=0.5, last_epoch=-1)
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
            valid_pids_in_batch = [pid for pid in batch_pids_list if patient_data[pid]['t'].shape[0] >= 2]
            
            if not valid_pids_in_batch:
                continue

            for pid in valid_pids_in_batch:
                dat = patient_data[pid]
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

            def closure_w():
                opt_w_lbfgs.zero_grad()
                loss = 0.
                for pid, dat in patient_data.items():
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
        current_pids_for_ab = valid_pids_in_batch if use_adam else patient_data.keys()
        for pid in current_pids_for_ab:
            dat = patient_data[pid]
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
                with torch.no_grad():
                    total_loss = 0.
                    num_patients = 0
                    for pid, dat in patient_data.items():
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
    return model, alpha_beta

model_fix, ab_dict = fit_population(
    patient_data,)

current_pid = os.getpid()
# Use the .save() method for JIT-scripted models
model_fix.save(f'model_{current_pid}.pt')

# ---------- 2. 绘制人群四联图 (根据s的5%和95%分位数) -----------------
# -------- 计算所有s值并确定分位数 ---------
all_s_values = torch.cat([ab_dict[p][0] * dat['t'] + ab_dict[p][1] for p, dat in patient_data.items()])
s_min = torch.quantile(all_s_values, 0.05)
s_max = torch.quantile(all_s_values, 0.95)

# -------- 根据s的分位数范围过滤病例 ---------
keep = []
for p in patient_data:
    s_values = ab_dict[p][0] * patient_data[p]['t'] + ab_dict[p][1]
    if s_values.min() >= s_min and s_values.max() <= s_max:
        keep.append(p)

stage_dict = pc.load_stage_dict()

# The trained model is now directly returned from the fit_population function.
# No need to instantiate a new model and load parameters manually.

# 准备绘图数据
if not keep:
    print("Warning: No patients left after filtering by s-quantiles. Plot will only show the mean trajectory based on all patients.")
    # Fallback to avoid crashing: use all patients for y0_pop
    y0_pop = torch.stack([dat['y0'] for dat in patient_data.values()]).mean(0)
else:
    y0_pop = torch.stack([patient_data[p]['y0'] for p in keep]).mean(0)

s_curve = torch.linspace(s_min.item(), s_max.item(), 400)
y_curve = model_fix(s_curve, y0_pop).detach().numpy()

# --- 计算置信区间 ---
with torch.no_grad():
    all_trajectories = []
    if keep:
        for p in keep:
            a, b = ab_dict[p]
            s_patient = a * patient_data[p]['t'] + b
            # We need to predict on the common s_curve for comparable intervals
            y_traj_p = model_fix(s_curve, patient_data[p]['y0'])
            all_trajectories.append(y_traj_p)
        
        # Stack trajectories and calculate percentiles
        all_trajectories = torch.stack(all_trajectories) # (num_patients, len_s_curve, 4)
        q25 = torch.quantile(all_trajectories, 0.25, dim=0).numpy()
        q75 = torch.quantile(all_trajectories, 0.75, dim=0).numpy()
    else:
        q25, q75 = None, None
# --- 结束计算 ---


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
    
    # --- 置信区间 ---
    if q25 is not None and q75 is not None:
        ax.fill_between(s_curve, q25[:, k], q75[:, k], color='gray', alpha=0.3, label='Confidence Interval')

    ax.set_xlabel('Disease progression score  s')
    ax.set_ylabel(TITLES[k])
    ax.legend()

fig2.suptitle(f'Population biomarker trajectories (s in [{s_min.item():.2f}, {s_max.item():.2f}])')
plt.tight_layout(rect=[0,0,1,0.96])
plt.savefig(f'result_{current_pid}.png')

# ---------- 为 alpha vs. MSE 散点图计算最终 MSE ----------
final_mses = {}
with torch.no_grad():
    all_biomarkers = torch.cat([dat['y'] for dat in patient_data.values()], dim=0)
    sigma = all_biomarkers.var(dim=0).clamp_min(1e-8)

    # 为计算损失，将 alpha, beta 转回 theta
    ab_theta_final = {}
    for pid, (alpha, beta) in ab_dict.items():
        theta0 = torch.log(torch.tensor(alpha) - 1e-4)
        theta1 = torch.tensor(beta)
        ab_theta_final[pid] = torch.tensor([theta0.item(), theta1.item()])

    for pid, dat in patient_data.items():
        if dat['t'].shape[0] >= 2:
            mse = calculate_combined_loss(model_fix, dat, ab_theta_final[pid], sigma)
            final_mses[pid] = mse.item()

# 提取 alpha 值
alphas = {pid: val[0] for pid, val in ab_dict.items()}

# 准备绘图数据
pids_list = list(final_mses.keys())
mse_values = [final_mses[p] for p in pids_list]
alpha_values = [alphas[p] for p in pids_list]

# 绘制 alpha vs. MSE 散点图
fig_alpha_mse, ax_alpha_mse = plt.subplots(figsize=(8, 6))
ax_alpha_mse.scatter(alpha_values, mse_values, alpha=0.6)
ax_alpha_mse.set_xlabel('Alpha')
ax_alpha_mse.set_ylabel('Final MSE')
ax_alpha_mse.set_title('Alpha vs. Final MSE for each patient')
ax_alpha_mse.grid(True)
plt.tight_layout()
plt.savefig(f'alpha_mse_{current_pid}.png')
plt.show()