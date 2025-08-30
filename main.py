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
csf_dict = pc.load_data()
print("Number of valid patients:", len(csf_dict))

patient_data = {}
for pid, sample in csf_dict.items():
    t = torch.from_numpy(sample[:, 0]).float()            # 年龄
    y = torch.from_numpy(sample[:, 1:5]).float()          # biomarker A/T/N/C
    patient_data[pid] = {"t": t, "y": y, "y0": y[0].clone()}

print("Valid patients: ", len(patient_data))

Message = f"This is a simple FNN model plus polynomial model with fixed pretrained DPS parameters."
name = 'fpp'

class ODEModel(nn.Module):
    def __init__(self, hidden_dim=32, num_layers=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 4096),
            nn.ReLU(),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Linear(2048, 4),
            nn.Tanh()
        )
                # fA: wa0 + wa1*A + wa2*A^2
        self.wa0 = torch.nn.Parameter(torch.tensor(0.01))
        self.wa1 = torch.nn.Parameter(torch.tensor(0.01))  # A linear negative
        self.wa2 = torch.nn.Parameter(torch.tensor(-0.01))

        # fT: wt0 + wt1*A + wt2*A^2 + wt3*T + wt4*T^2 + wt5*A*T
        self.wt0  = torch.nn.Parameter(torch.tensor(0.01))
        self.wt1 = torch.nn.Parameter(torch.tensor(-0.01))  # A term negative
        self.wt2  = torch.nn.Parameter(torch.tensor(-0.01))  # T term positive
        self.wt3 = torch.nn.Parameter(torch.tensor(0.01))
        self.wt4 = torch.nn.Parameter(torch.tensor(0.01))
        self.wt5 = torch.nn.Parameter(torch.tensor(-0.01))

        # fN: wn0 + wn1*T + wn2*T^2 + wn3*N + wn4*N^2 + wn5*T*N
        self.wn0  = torch.nn.Parameter(torch.tensor(0.01))
        self.wn1 = torch.nn.Parameter(torch.tensor(-0.01))  # T term positive
        self.wn2  = torch.nn.Parameter(torch.tensor(-0.01))  # N term negative
        self.wn3 = torch.nn.Parameter(torch.tensor(0.01))
        self.wn4 = torch.nn.Parameter(torch.tensor(0.01))
        self.wn5 = torch.nn.Parameter(torch.tensor(-0.01))

        # fC: wc0 + wc1*C + wc2*C^2 + wc3*N + wc4*N^2 + wc5*N*C
        self.wc0  = torch.nn.Parameter(torch.tensor(0.01))
        self.wc1 = torch.nn.Parameter(torch.tensor(-0.01))  # N term negative
        self.wc2  = torch.nn.Parameter(torch.tensor(-0.01))  # C term positive
        self.wc3 = torch.nn.Parameter(torch.tensor(0.01))
        self.wc4 = torch.nn.Parameter(torch.tensor(0.01))
        self.wc5 = torch.nn.Parameter(torch.tensor(-0.01))


    def poly(self, y: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
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
        p = torch.stack([fA, fT, fN, fC], dim=-1)
        return p

    def f(self, y: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        return self.net(y)+self.poly(y, s)

    def _rk4_integrate(self, s_grid: torch.Tensor, y0: torch.Tensor, f_fn) -> torch.Tensor:
        ys = [y0]
        for i in range(1, len(s_grid)):
            h = s_grid[i] - s_grid[i - 1]
            y_i = ys[-1]

            k1 = f_fn(y_i, s_grid[i - 1])
            k2 = f_fn(y_i + 0.5 * h * k1, s_grid[i - 1])
            k3 = f_fn(y_i + 0.5 * h * k2, s_grid[i - 1])
            k4 = f_fn(y_i + h * k3, s_grid[i - 1])

            y_next = y_i + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            ys.append(y_next)

        return torch.stack(ys)

    def forward(self, s_grid: torch.Tensor, y0: torch.Tensor) -> torch.Tensor:
        # Full model: poly(y)*net(y)
        return self._rk4_integrate(s_grid, y0, self.f)

    def forward_net_only(self, s_grid: torch.Tensor, y0: torch.Tensor) -> torch.Tensor:
        # Neural-only dynamics: dy/ds = net(y)
        return self._rk4_integrate(s_grid, y0, lambda y, s: self.net(y))

    def forward_poly_only(self, s_grid: torch.Tensor, y0: torch.Tensor) -> torch.Tensor:
        # Polynomial-only dynamics: dy/ds = poly(y)
        return self._rk4_integrate(s_grid, y0, lambda y, s: self.poly(y, s))

def residual(model, s_global, y_global, sigma=None):
    """
    Calculates the global residual: dy/dt - f(y) for all data points.
    where dy/dt is computed by finite difference (centered for interior, one-sided for endpoints).
    Returns mean squared error of the residual across all data points.
    """
    # Compute dy/dt using finite differences
    dy_dt = torch.zeros_like(y_global)
    
    # Forward difference for the first point
    dy_dt[0] = (y_global[1] - y_global[0]) / (s_global[1] - s_global[0])
    # Centered difference for interior points
    if len(s_global) > 2:
        dy_dt[1:-1] = (y_global[2:] - y_global[:-2]) / (s_global[2:].unsqueeze(1) - s_global[:-2].unsqueeze(1))
    # Backward difference for the last point
    dy_dt[-1] = (y_global[-1] - y_global[-2]) / (s_global[-1] - s_global[-2])

    # Model prediction f(y) at each s, y
    fy = torch.zeros_like(y_global)
    for i in range(len(s_global)):
        fy[i] = model.f(y_global[i], s_global[i])

    residual = dy_dt - fy
    loss = residual ** 2
    if sigma is not None:
        loss = loss * sigma

    return loss.mean()


def calculate_global_loss(model, s_global, y_global, sigma=None):
    """
    Calculates the global loss based on prediction from s=-10, y0=[0.1,0,0,0].
    All s values are concatenated and sorted as s_global, then predict for all data points.
    """
    # 从s=-10, y0=[0.1,0,0,0]开始预测
    s_start = torch.tensor(-10.0)
    y0_global = torch.tensor([0.1, 0, 0, 0])
    
    # 预测整个轨迹
    y_pred_global = model(s_global, y0_global)
    
    # 计算MSE
    loss = (y_pred_global - y_global)**2
    if sigma is not None:
        loss = loss * sigma

    return loss.mean()

def calculate_combined_loss(model, s_global, y_global, sigma=None, r=0.5, s_penalty_weight=0.1):
    """
    Calculates a combined loss: r% residual and (1-r)% global.
    Adds exponential penalty for s values outside [-10, 20] range.
    """
    #res = residual(model, s_global, y_global, sigma)
    loss_global = calculate_global_loss(model, s_global, y_global, sigma)
    '''
    zero = torch.zeros(4)
    one = torch.ones(4)
    s_default = torch.tensor(0.0)
    combined_loss = ( loss_global + res ) * torch.exp(torch.norm(model.f(B, s_default)) + torch.norm(model.f(D, s_default))) / (torch.norm(model.f(C, s_default)) + 1e-5)
    '''
    #combined_loss = r * res + (1-r) * loss_global
    return loss_global


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
        hidden_dim=16,
        n_adam      = 300,      # adam 阶段迭代次数
        batch_size=128,
        opt_w_lr=1e-3,
        weighted_sampling=True,
        early_stop_patience=80,
        early_stop_threshold=0.001):
    sigma = torch.ones(4)

    # ---------- 初始化 ----------
    model = ODEModel(hidden_dim=hidden_dim)
    def weights_init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0, std=0.005)  # 将权重初始化为很小的随机值
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)  # 将偏置初始化为0
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
    
    # 加载预训练的DPS参数
    try:
        ab = torch.load('dps.pth')
        print("Successfully loaded pretrained DPS parameters from dps.pth")
        
        # 将theta参数设置为不需要梯度，因为我们不再训练a,b
        for pid in ab:
            if 'theta' in ab[pid]:
                ab[pid]['theta'] = ab[pid]['theta'].detach().requires_grad_(False)
            else:
                # 如果加载的是直接的alpha, beta值，转换为theta格式
                alpha, beta = ab[pid]
                ab[pid] = {'theta': alpha, 'beta': beta}
                
    except FileNotFoundError:
        print("Warning: dps.pth not found. Computing a,b from age to stage mapping.")
        ab = {}
        for pid, dat in patient_data.items():
            stage = stage_dict.get(pid, 'Other')
            t = dat['t']  # 年龄时间序列

            # 根据stage确定s的范围
            if stage == 'CN':
                s_range = (-10.0, 0.0)
            elif stage == 'LMCI':
                s_range = (0.0, 10.0)
            elif stage == 'AD':
                s_range = (10.0, 20.0)
            else:  # Default for 'Other' or missing stages
                s_range = (-5.0, 5.0)
            
            # 线性变换：将年龄范围 [t_min, t_max] 映射到s范围 [s_min, s_max]
            # s = a * t + b
            # 求解：s_min = a * t_min + b, s_max = a * t_max + b
            t_min, t_max = t.min().item(), t.max().item()
            s_min, s_max = s_range[0], s_range[1]
            
            # 如果t_min == t_max，设置默认值
            if abs(t_max - t_min) < 1e-6:
                a = 1.0
                b = s_min - a * t_min
            else:
                # 计算线性变换参数
                a = (s_max - s_min) / (t_max - t_min)
                b = s_min - a * t_min
            
            # 确保a > 0，并转换为theta格式
            a = max(a, 1e-4)  # 确保a > 0
            theta0 = torch.log(torch.tensor(a - 1e-4))
            theta1 = torch.tensor(b)
            
            ab[pid] = {'theta': torch.tensor([theta0.item(), theta1.item()], requires_grad=False)}

    # --------- Adam 优化器池 -----------
    opt_w  = optim.Adam(model.parameters(), lr=opt_w_lr, weight_decay=1e-4)
    # 不再需要a,b的优化器，因为我们不再训练这些参数
    scheduler = optim.lr_scheduler.MultiStepLR(opt_w, milestones=list(range(60, 151)), gamma=0.99, last_epoch=-1)

    # --------- 训练循环 -----------
    training_stopped = False
    # Early stopping state
    early_stop_counter = 0
    last_adam_loss = float('inf')
    adam_loss_history = []

    # 只更新模型参数w，a,b使用预训练参数
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

        opt_w.zero_grad()
        loss_w = 0.
        
        # Convert tensor PIDs to integer for dict lookup
        batch_pids_list = [pid.item() for pid in batch_pids] if use_minibatch else batch_pids
        valid_pids_in_batch = [pid for pid in batch_pids_list if patient_data[pid]['t'].shape[0] >= 2]
        
        if not valid_pids_in_batch:
            continue

        # 收集所有患者的s值和y值
        all_s_values = []
        all_y_values = []
        
        for pid in valid_pids_in_batch:
            dat = patient_data[pid]
            a = F.softplus(ab[pid]['theta'][0]).item() + 1e-4
            b = ab[pid]['theta'][1]
            s_values = a * dat['t'] + b
            y_values = dat['y']
            
            all_s_values.append(s_values)
            all_y_values.append(y_values)
        
        # 拼接所有s值和y值
        s_global = torch.cat(all_s_values)
        y_global = torch.cat(all_y_values)
        
        # 对s_global排序，并相应重排y_global
        s_global_sorted, sort_indices = torch.sort(s_global)
        y_global_sorted = y_global[sort_indices]

        # 过滤掉s_global_sorted的5%和95%分位数以外的数据
        s_5_percentile = torch.quantile(s_global_sorted, 0.05)
        s_95_percentile = torch.quantile(s_global_sorted, 0.95)
        
        # 创建过滤掩码
        mask = (s_global_sorted >= s_5_percentile) & (s_global_sorted <= s_95_percentile)
        
        # 应用过滤
        s_global_filtered = s_global_sorted[mask]
        y_global_filtered = y_global_sorted[mask]
        
        # 计算全局损失（使用过滤后的数据）
        loss_w = calculate_combined_loss(model, s_global_filtered, y_global_filtered, sigma=sigma)

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
            opt_w.step()
            scheduler.step()

        if training_stopped:
            break

        # ---------- 可选：估计 σk ----------
        with torch.no_grad():
            y0 = torch.tensor([0.1, 0, 0, 0])
            
            # 预测整个轨迹
            y_pred = model(s_global_filtered, y0)
            sigma = (y_pred - y_global_filtered)**2
            if torch.any(sigma):
                sigma = sigma.mean(dim=0)
            else:
                sigma = torch.ones(4) # Fallback if no residuals calculated

        # ----------- 监控 ----------
        if (it+1) % 50 == 0:
            print(f"iter {it+1:02d}/{n_adam} | "
                  f"Adam | "
                  f"Batch MSE={loss_w.item():.4f} | "
                  f"Avg MSE={current_avg_loss:.4f}")

    # --------- 输出 ----------
    model.eval() # Switch to evaluation mode before returning
    
    return model

model = fit_population(
    patient_data)

try:
    torch.save(model.state_dict(), f'{name}.pth')
except Exception as e:
    print(f"Error saving model: {e} 喵！   _(┐ ◟;ﾟдﾟ)ノ")

# ---------- 2. 绘制人群四联图 (根据s的10%和90%分位数) -----------------
with torch.no_grad():
    # 收集所有患者的s值（一致使用 softplus(θ0)+1e-4 作为 a）
    ab = torch.load('dps.pth')
    all_s_values = []
    for p in patient_data:
        a = F.softplus(ab[p]['theta'][0]).item() + 1e-4
        b = ab[p]['theta'][1]
        s_values = a * patient_data[p]['t'] + b
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
        a = F.softplus(ab[p]['theta'][0]).item() + 1e-4
        b = ab[p]['theta'][1]
        s_values = a * patient_data[p]['t'] + b
        # 检查是否有任何s值在范围内
        if torch.any((s_values >= s_min) & (s_values <= s_max)):
            keep.append(p)

    stage_dict = pc.load_stage_dict()

    # 准备绘图数据 - 使用10分位数对应的y值作为起始点
    y0_pop = torch.tensor([0.1,0,0,0])

    # ---------- 添加新轨迹：从10分位数开始，分别计算完整/仅NN/仅多项式轨迹 ----------
    y_curve_full = model(s_curve, y0_pop)
    y_curve_net  = model.forward_net_only(s_curve, y0_pop)
    y_curve_poly = model.forward_poly_only(s_curve, y0_pop)
    y_curve_full = y_curve_full.detach().numpy()
    y_curve_net  = y_curve_net.detach().numpy()
    y_curve_poly = y_curve_poly.detach().numpy()
    y_curve_full = pc.inv_nor(y_curve_full)
    y_curve_net  = pc.inv_nor(y_curve_net)
    y_curve_poly = pc.inv_nor(y_curve_poly)

    TITLES = ['Aβ (A)', 'p-Tau (T)', 'N', 'Cognition (C)']

    fig2, axes = plt.subplots(2, 2, figsize=(9, 6))
    for k, ax in enumerate(axes.flat):
        # --- 分阶段准备散点数据 ---
        s_by_stage = {'CN': [], 'LMCI': [], 'AD': [], 'Other': []}
        y_by_stage = {'CN': [], 'LMCI': [], 'AD': [], 'Other': []}

        for p in keep:
            a = F.softplus(ab[p]['theta'][0]).item() + 1e-4
            b = ab[p]['theta'][1]
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
        
        # --- 轨迹曲线：完整/仅NN/仅多项式 ---
        ax.plot(s_curve, y_curve_full[:,k], lw=1.6, c='red', linestyle='--', label='Full (poly + net)')
        ax.plot(s_curve, y_curve_net[:,k],  lw=1.2, c='purple', linestyle='-.', label='Net only')
        ax.plot(s_curve, y_curve_poly[:,k], lw=1.2, c='teal', linestyle=':', label='Poly only')
        
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
            a = F.softplus(ab[pid]['theta'][0]).item() + 1e-4
            b = ab[pid]['theta'][1]
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
