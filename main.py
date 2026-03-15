import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pccmnn as pc
from torchdiffeq import odeint as torch_odeint

# --- 0. 資料載入和準備 ---
csf_dict = pc.load_data()
stage_dict = pc.load_stage_dict()
print(f"成功載入 {len(csf_dict)} 位患者的資料。")

# 转换数据格式以适应 PyTorch
patient_data = {}
for pid, sample in csf_dict.items():
    t = torch.from_numpy(sample[:, 0]).float()
    y = torch.from_numpy(sample[:, 1:5]).float()
    patient_data[pid] = {"t": t, "y": y, "y0": y[0].clone(), "stage": stage_dict.get(pid, 'Other')}

def get_cn_average_y0(patient_data, stage_dict):
    cn_y0s = []
    for pid, data in patient_data.items():
        if stage_dict.get(pid) == 'CN':
            cn_y0s.append(data['y0'])
    if not cn_y0s:
        print("警告: 未找到CN患者, 使用預設y0。")
        return torch.tensor([0.1, 0, 0, 0])
    
    # 将所有CN患者的y0堆叠成矩阵
    cn_y0s_tensor = torch.stack(cn_y0s)  # shape: (num_cn_patients, 4)
    
    # 对每个生物标志物分别计算非NaN值的平均
    avg_y0 = torch.zeros(4)
    for k in range(4):
        y0_k = cn_y0s_tensor[:, k]
        valid_mask = ~torch.isnan(y0_k)
        if valid_mask.sum() > 0:
            avg_y0[k] = y0_k[valid_mask].mean()
        else:
            # 如果所有CN患者在该标志物上都是NaN，使用0
            avg_y0[k] = 0.0
            print(f"警告: 所有CN患者在生物标志物{k}上的初始值都是NaN，使用0。")
    
    print(f"使用CN群體的平均初始值（非NaN）: {avg_y0.numpy()}")
    return avg_y0

y0_cn_avg = get_cn_average_y0(patient_data, stage_dict)
name = 'fnn'

def compute_y_minmax_01(patient_data_dict):
    """
    统计每个 biomarker 的 min/max（忽略 NaN），用于“仅输入到神经网络时”归一化到[0,1]。
    注意：ODE 的状态 y 仍然在原空间演化，这里只是构造网络输入 y01。
    """
    ys = []
    for _, dat in patient_data_dict.items():
        ys.append(dat["y"])
    y_all = torch.cat(ys, dim=0)  # (N,4)
    y_min = torch.zeros(4)
    y_max = torch.ones(4)
    for k in range(4):
        col = y_all[:, k]
        mask = ~torch.isnan(col)
        if mask.any():
            y_min[k] = col[mask].min()
            y_max[k] = col[mask].max()
        else:
            y_min[k] = 0.0
            y_max[k] = 1.0
    # 防止除零
    y_max = torch.where((y_max - y_min) < 1e-6, y_min + 1e-6, y_max)
    return y_min, y_max

y_min01, y_max01 = compute_y_minmax_01(patient_data)


# === Neural ODE 模型定义（级联结构）===
class NeuralODE(nn.Module):
    """
    级联结构的 Neural ODE，预测“速率”(rate)，由 ODEModel 乘以饱和余量实现物理约束。
    变量顺序约定为: [Abeta, pTau, N, C]

    级联依赖:
      dA/ds  由 A 驱动
      dT/ds  由 (A,T) 驱动
      dN/ds  由 (T,N) 驱动
      dC/ds  由 (N,C) 驱动

    输出层 bias 初始化为 -5，使 softplus(bias)≈0，模型从“近静止”开始学习。
    """
    def __init__(self, hidden_dim=16):
        super().__init__()
        def make_net(in_dim: int):
            fc1 = nn.Linear(in_dim, hidden_dim)
            fc2 = nn.Linear(hidden_dim, 1)
            nn.init.constant_(fc2.bias, -5.0)
            return nn.Sequential(fc1, nn.ReLU(), fc2)

        self.net_A = make_net(1)  # A
        self.net_T = make_net(2)  # [A,T]
        self.net_N = make_net(2)  # [T,N]
        self.net_C = make_net(2)  # [N,C]


class ODEModel(nn.Module):
    """
    torchdiffeq 需要的 ODE 函数封装：func(t, y)
    - 仅对输入网络的 y 做[0,1]归一化（用全体数据 min/max）
    - 用 softplus(rate) * 饱和余量 实现边界/符号约束：
        Abeta, N: 非正且接近0时导数->0
        Tau, C : 非负且接近1时导数->0
    """
    def __init__(self, fnn_model: NeuralODE, y_min: torch.Tensor, y_max: torch.Tensor):
        super().__init__()
        self.fnn = fnn_model
        self.register_buffer("y_min", y_min.float())
        self.register_buffer("y_max", y_max.float())

    def _norm01(self, y: torch.Tensor) -> torch.Tensor:
        y01 = (y - self.y_min) / (self.y_max - self.y_min)
        return torch.clamp(y01, 0.0, 1.0)

    def forward(self, t, y):
        squeeze_back = False
        if y.dim() == 1:
            y = y.unsqueeze(0)
            squeeze_back = True

        y01 = self._norm01(y)
        A = y01[:, 0:1]
        T = y01[:, 1:2]
        N = y01[:, 2:3]
        C = y01[:, 3:4]

        rate_A = self.fnn.net_A(A)
        dA = -F.softplus(rate_A) * (A + 1e-6)

        rate_T = self.fnn.net_T(torch.cat([A, T], dim=1))
        dT = F.softplus(rate_T) * (1.0 - T + 1e-6)

        rate_N = self.fnn.net_N(torch.cat([T, N], dim=1))
        dN = -F.softplus(rate_N) * (N + 1e-6)

        rate_C = self.fnn.net_C(torch.cat([N, C], dim=1))
        dC = F.softplus(rate_C) * (1.0 - C + 1e-6)

        out = torch.cat([dA, dT, dN, dC], dim=1)
        return out.squeeze(0) if squeeze_back else out


# === Sigmoid辅助函数 ===
def get_sigmoid_values(sigmoid_params, s_grid_np):
    """
    根据sigmoid参数计算在s_grid上的函数值
    sigmoid_params: shape (4,4) -> [a,b,c,d] for each biomarker
    s_grid_np: numpy array
    return: numpy array (len(s_grid), 4)
    """
    y_on_grid = np.zeros((len(s_grid_np), 4))
    for k in range(4):
        a, b, c, d = sigmoid_params[k]
        y_on_grid[:, k] = a / (1.0 + np.exp(-b * (s_grid_np - c))) + d
    return y_on_grid


def get_sigmoid_y_dyds_tensor(s_tensor: torch.Tensor, sigmoid_params, device=None):
    """
    torch版：给定 s (Ns,) 返回 sigmoid 的 y(s) 与 dy/ds(s)
    sigmoid: y = a/(1+exp(-b(s-c))) + d
    dy/ds = a*b*exp(-b(s-c)) / (1+exp(-b(s-c)))^2

    s_tensor: (Ns,) float tensor
    sigmoid_params: (4,4) array/list -> [a,b,c,d] for each biomarker
    return:
      y: (Ns,4)
      dyds: (Ns,4)
    """
    if device is None:
        device = s_tensor.device
    sig = torch.tensor(sigmoid_params, dtype=torch.float32, device=device)  # (4,4)
    a = sig[:, 0].view(1, 4)
    b = sig[:, 1].view(1, 4)
    c = sig[:, 2].view(1, 4)
    d = sig[:, 3].view(1, 4)

    s = s_tensor.view(-1, 1)  # (Ns,1)
    exp_term = torch.exp(-b * (s - c))
    denom = (1.0 + exp_term)
    y = a / denom + d
    dyds = (a * b * exp_term) / (denom ** 2)
    return y, dyds


def get_stage_mean_y(patient_data, stage, fallback_tensor):
    """
    计算指定stage的平均y（忽略NaN），若无数据则返回fallback。
    """
    ys = []
    for _, dat in patient_data.items():
        if dat['stage'] == stage:
            ys.append(dat['y'])
    if not ys:
        return fallback_tensor.clone()
    y_all = torch.cat(ys, dim=0)  # (N,4)
    mean_y = torch.zeros(4)
    for k in range(4):
        col = y_all[:, k]
        mask = ~torch.isnan(col)
        if mask.any():
            mean_y[k] = col[mask].mean()
        else:
            mean_y[k] = fallback_tensor[k]
    return mean_y


def adjust_sigmoid_params(sigmoid_params, y_cn_avg, y_ad_avg, s0=-20.0):
    """
    调整sigmoid参数，使：
    - 上平台接近 CN 平均初值
    - 下平台接近 AD 平均值
    - 在 s=s0 处穿过 y_cn_avg
    """
    sig = np.array(sigmoid_params, dtype=np.float32)  # (4,4)
    adjusted = np.zeros_like(sig)
    eps = 1e-6
    for k in range(4):
        b_orig = float(sig[k, 1])
        b = max(abs(b_orig), 0.05)

        upper = float(y_cn_avg[k])
        lower = float(y_ad_avg[k])
        a = upper - lower
        d = lower

        # 确保 y0(=CN初值) 在平台区间内
        y0 = float(y_cn_avg[k])
        lo = min(lower, upper) + eps
        hi = max(lower, upper) - eps
        y0 = float(np.clip(y0, lo, hi))

        # 计算 c，使 y(s0)=y0
        denom = (a / (y0 - d)) - 1.0
        denom = max(denom, eps)
        c = s0 + (1.0 / b) * np.log(denom)

        adjusted[k] = np.array([a, b, c, d], dtype=np.float32)
    return adjusted


def build_s_grid_from_dps(dps_params_loaded, patient_data, margin_ratio=0.1, num_points=300):
    """
    依据预训练的DPS参数和病人时间点，构建s_grid
    """
    s_all = []
    for pid, dat in patient_data.items():
        if pid in dps_params_loaded:
            a = dps_params_loaded[pid]['a']
            b = dps_params_loaded[pid]['b']
            s_vals = a * dat['t'].numpy() + b
            s_all.extend(s_vals)
    s_all = np.array(s_all)
    s_min, s_max = s_all.min(), s_all.max()
    s_margin = (s_max - s_min) * margin_ratio
    s_grid_np = np.linspace(s_min - s_margin, s_max + s_margin, num_points)
    return s_grid_np


def train_neural_ode_to_sigmoid_with_dyds(
    fnn_model,
    sigmoid_params,
    s_grid_np,
    epochs=2000,
    lr=1e-3,
    lambda_traj=1.0,
    lambda_dyds=1.0,
):
    """
    训练 Neural ODE：
    - 轨线损失：ODE预测轨线 vs sigmoid曲线的L2
    - 导数损失：rhs vs A(=sigmoid导数矩阵) 的L2
    """
    ode_model = ODEModel(fnn_model, y_min01, y_max01).train()
    optimizer = optim.Adam(ode_model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    s_tensor = torch.tensor(s_grid_np, dtype=torch.float32)
    y_sigmoid, dyds_sigmoid = get_sigmoid_y_dyds_tensor(s_tensor, sigmoid_params)

    # 初值选用sigmoid起点
    y0 = y_sigmoid[0]

    # 记录loss历史
    loss_history = {
        'epoch': [],
        'traj_loss': [],
        'dyds_loss': []
    }

    # 每10%打印一次进度
    print_interval = max(1, epochs // 10)

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        try:
            # 轨线预测
            y_pred = torch_odeint(ode_model, y0, s_tensor, method='dopri5', rtol=1e-4, atol=1e-5)
            loss_traj = criterion(y_pred, y_sigmoid)

            # rhs 与 A 的导数匹配
            rhs = ode_model(torch.tensor(0.0, device=y_pred.device), y_sigmoid)
            loss_dyds = criterion(rhs, dyds_sigmoid)

            loss = lambda_traj * loss_traj + lambda_dyds * loss_dyds
            loss.backward()
            optimizer.step()

            loss_history['epoch'].append(epoch)
            loss_history['traj_loss'].append(loss_traj.item())
            loss_history['dyds_loss'].append(loss_dyds.item())

            if epoch % print_interval == 0 or epoch == 1 or epoch == epochs:
                progress = int(epoch / epochs * 100)
                print(f"[Train {progress:3d}%] traj={loss_traj.item():.6f}, dyds={loss_dyds.item():.6f}")
        except Exception as e:
            print(f"ODE求解失败: {e}")
            break

    ode_model.eval()
    return ode_model.fnn, loss_history

# --- 1. 定義訓練流程 ---
# ===== 优化后的 FNN 损失 =====
def calculate_loss_fnn(
    ode_model,
    patient_data,
    ab,
    pids,
    y0,
    sigmoid_params=None,
    lambda_sigmoid=1.0,
    lambda_dyds=1.0,
    n_dyds_samples=200,
):
    """
    高效版 FNN loss：
    - 所有病人、所有 biomarker 的 s 值一次合并；
    - 去重、排序，确保 torchdiffeq 时间轴严格递增；
    - 在去重时间轴上解一次 ODE，再映射回原索引。
    - 添加正则化项：在[-5:20:1]的向量上约束ODE右侧函数
    """
    try:
        s_all_list, y_true_list, k_list = [], [], []
        for k in range(4):
            for pid in pids:
                dat = patient_data[pid]
                s_values = ab[pid]['a'] * dat['t'] + ab[pid]['b']
                y_values = dat['y'][:, k]
                valid_mask = ~torch.isnan(y_values)
                if valid_mask.any():
                    s_all_list.append(s_values[valid_mask])
                    y_true_list.append(y_values[valid_mask])
                    k_list.extend([k] * valid_mask.sum().item())

        if not s_all_list:
            return torch.tensor(0.0, requires_grad=True)

        s_all = torch.cat(s_all_list)
        y_true_all = torch.cat(y_true_list)
        k_all = torch.tensor(k_list, device=y_true_all.device, dtype=torch.long)

        # 排序和去重操作不进计算图
        with torch.no_grad():
            s_sorted, sort_idx = torch.sort(s_all)
            y_sorted = y_true_all[sort_idx]
            k_sorted = k_all[sort_idx]
            s_unique, inv = torch.unique_consecutive(s_sorted, return_inverse=True)

        # 一次 ODE 求解
        y_unique = torch_odeint(
            ode_model, y0, s_unique, method='dopri5', rtol=1e-4, atol=1e-5
        )  # (Nu, 4)

        # 映射回原索引
        y_all = y_unique[inv]  # (N, 4)
        y_pred_selected = y_all[torch.arange(y_all.size(0)), k_sorted]

        # 数据拟合损失 - 使用SmoothL1Loss
        smooth_l1_loss = nn.SmoothL1Loss(reduction='sum')
        data_loss = smooth_l1_loss(y_pred_selected, y_sorted)

        # Sigmoid形态约束（L2到预训练Sigmoid），仅当提供了sigmoid_params
        sigmoid_loss = torch.tensor(0.0, device=y_unique.device)
        dyds_reg_loss = torch.tensor(0.0, device=y_unique.device)
        if sigmoid_params is not None:
            # 将sigmoid参数转为tensor并放到设备上
            sig = torch.tensor(sigmoid_params, dtype=torch.float32, device=y_unique.device)  # (4,4)
            a = sig[:, 0].view(1, 4)  # (1,4)
            b = sig[:, 1].view(1, 4)
            c = sig[:, 2].view(1, 4)
            d = sig[:, 3].view(1, 4)
            s_expanded = s_unique.view(-1, 1)  # (Nu,1)
            exp_term = torch.exp(-b * (s_expanded - c))
            y_sig = a / (1.0 + exp_term) + d  # (Nu,4)
            sigmoid_loss = torch.mean((y_unique - y_sig) ** 2)

            # === 新增：导数场正则化（Vector Field Regularization）===
            # 在 s 的范围内随机采样一些点，要求 fnn(y_sigmoid(s)) ≈ dy/ds_sigmoid(s)
            with torch.no_grad():
                s_lo = s_unique.min().item()
                s_hi = s_unique.max().item()
            s_sample = (torch.rand(n_dyds_samples, device=y_unique.device) * (s_hi - s_lo) + s_lo).float()
            y_sig_s, dyds_sig_s = get_sigmoid_y_dyds_tensor(s_sample, sigmoid_params, device=y_unique.device)
            # 用 ode_model(t,y) 计算导数，确保同样走“输入归一化 + 饱和余量”结构
            dyds_pred = ode_model(torch.tensor(0.0, device=y_unique.device), y_sig_s)
            dyds_reg_loss = torch.mean((dyds_pred - dyds_sig_s) ** 2)
        
        # === 正则化项 ===
        # FNN模型参数的L1正则化
        l1_reg = 0.0
        for param in ode_model.parameters():
            l1_reg += torch.sum(torch.abs(param))
        
        # 总损失：数据损失 + L1正则化 + Sigmoid形态约束 + 导数场正则化
        lambda_l1 = 0.0001  # L1正则化权重
        
        total_loss = (
            data_loss
            + lambda_l1 * l1_reg
            + lambda_sigmoid * sigmoid_loss
            + lambda_dyds * dyds_reg_loss
        )
        
        # 返回总损失和各分量
        loss_dict = {
            'total': total_loss.item() if torch.isfinite(total_loss) else float('inf'),
            'data': data_loss.item(),
            'l1': l1_reg.item(),
            'lambda_l1': lambda_l1,
            'sigmoid': sigmoid_loss.item() if torch.isfinite(sigmoid_loss) else float('inf'),
            'lambda_sigmoid': lambda_sigmoid
            ,
            'dyds_reg': dyds_reg_loss.item() if torch.isfinite(dyds_reg_loss) else float('inf'),
            'lambda_dyds': lambda_dyds,
            'n_dyds_samples': n_dyds_samples
        }
        
        if torch.isfinite(total_loss):
            return total_loss, loss_dict
        else:
            return torch.tensor(float('inf'), requires_grad=True), loss_dict

    except Exception as e:
        print(f"FNN loss 计算出错: {e}")
        import traceback
        traceback.print_exc()
        loss_dict = {'total': float('inf'), 'data': 0, 'l1': 0, 'lambda': 0.0001}
        return torch.tensor(float('inf'), requires_grad=True), loss_dict


# ===== 优化后的 DPS 损失 =====
def calculate_loss_dps(ode_model, patient_data, ab, pids, y0):
    """
    高效版 DPS loss：
    - 所有病人、所有时间点与 biomarker 合并；
    - 去重、排序，确保 torchdiffeq 时间轴严格递增；
    - 在去重时间轴上解一次 ODE，再映射回原索引。
    - 添加a,b参数的L2正则化
    """
    try:
        s_all_list, y_true_list, k_list = [], [], []
        for pid in pids:
            dat = patient_data[pid]
            s_values = ab[pid]['a'] * dat['t'] + ab[pid]['b']
            y_true = dat['y']
            for i in range(y_true.shape[0]):
                for k in range(4):
                    if not torch.isnan(y_true[i, k]):
                        s_all_list.append(s_values[i])
                        y_true_list.append(y_true[i, k])
                        k_list.append(k)

        if not s_all_list:
            return torch.tensor(0.0, requires_grad=True), {'total': 0.0, 'data': 0.0, 'l2': 0.0}

        s_all = torch.stack(s_all_list)
        y_true_all = torch.stack(y_true_list)
        k_all = torch.tensor(k_list, device=y_true_all.device, dtype=torch.long)

        with torch.no_grad():
            s_sorted, sort_idx = torch.sort(s_all)
            y_sorted = y_true_all[sort_idx]
            k_sorted = k_all[sort_idx]
            s_unique, inv = torch.unique_consecutive(s_sorted, return_inverse=True)

        y_unique = torch_odeint(
            ode_model, y0, s_unique, method='dopri5', rtol=1e-4, atol=1e-5
        )

        y_all = y_unique[inv]
        y_pred_selected = y_all[torch.arange(y_all.size(0)), k_sorted]

        # 数据拟合损失
        data_loss = ((y_pred_selected - y_sorted) ** 2).sum()
        
        # a,b参数的L2正则化
        l2_reg = 0.0
        for pid in pids:
            l2_reg += ab[pid]['a'] ** 2 + ab[pid]['b'] ** 2
        
        # 总损失
        lambda_l2 = 0.01  # L2正则化权重
        total_loss = data_loss + lambda_l2 * l2_reg
        
        # 返回损失和分量
        loss_dict = {
            'total': total_loss.item() if torch.isfinite(total_loss) else float('inf'),
            'data': data_loss.item(),
            'l2': l2_reg.item()
        }
        
        if torch.isfinite(total_loss):
            return total_loss, loss_dict
        else:
            return torch.tensor(float('inf'), requires_grad=True), loss_dict

    except Exception as e:
        print(f"DPS loss 计算出错: {e}")
        loss_dict = {'total': float('inf'), 'data': 0.0, 'l2': 0.0}
        return torch.tensor(float('inf'), requires_grad=True), loss_dict

def train_fnn_with_fixed_dps(
    fnn_pretrained,
    patient_data,
    y0,
    dps_path='dps.pth',
    n_epochs=2000,
    lr_fnn=1e-3,
    sigmoid_params=None,
    lambda_traj=1.0,
    lambda_dyds=1.0,
):
    """
    仅训练FNN，DPS(a,b)从头到尾固定不训练。
    损失：
    - 轨线损失：ODE预测轨线 vs sigmoid曲线的L2
    - 导数损失：rhs vs A(=sigmoid导数矩阵) 的L2
    """
    print(f"\n--- 开始训练 FNN (DPS固定) ---")

    # 加载预训练的FNN模型
    ode_model = ODEModel(fnn_pretrained, y_min01, y_max01).train()

    # 加载DPS参数（固定）
    try:
        dps_params_loaded = torch.load(dps_path, weights_only=False)
        ab = {}
        for pid in patient_data.keys():
            if pid in dps_params_loaded:
                ab[pid] = {
                    'a': torch.tensor(dps_params_loaded[pid]['a'], dtype=torch.float32),
                    'b': torch.tensor(dps_params_loaded[pid]['b'], dtype=torch.float32)
                }
        print(f"成功从 {dps_path} 加载DPS参数（固定不训练）。")
    except FileNotFoundError:
        print(f"错误: 未找到 {dps_path}。")
        return None, None, None

    patient_pids = list(ab.keys())

    opt_fnn = optim.Adam(ode_model.parameters(), lr=lr_fnn)

    # 记录loss历史
    loss_history = {
        'epoch': [],
        'traj_loss': [],
        'dyds_loss': []
    }

    print_interval = max(1, n_epochs // 10)

    for epoch in range(1, n_epochs + 1):
        opt_fnn.zero_grad()
        loss_fnn, loss_fnn_dict = calculate_loss_fnn(
            ode_model, patient_data, ab, patient_pids, y0,
            sigmoid_params=sigmoid_params,
            lambda_sigmoid=lambda_traj,
            lambda_dyds=lambda_dyds,
            n_dyds_samples=0,
        )
        if torch.isfinite(loss_fnn):
            loss_fnn.backward()
            opt_fnn.step()

        # 记录loss
        loss_history['epoch'].append(epoch)
        loss_history['traj_loss'].append(loss_fnn_dict.get('sigmoid', 0.0))
        loss_history['dyds_loss'].append(loss_fnn_dict.get('dyds_reg', 0.0))

        if epoch % print_interval == 0 or epoch == 1 or epoch == n_epochs:
            progress = int(epoch / n_epochs * 100)
            print(
                f"[Train {progress:3d}%] traj={loss_fnn_dict.get('sigmoid', 0.0):.6f}, "
                f"dyds={loss_fnn_dict.get('dyds_reg', 0.0):.6f}"
            )

    ode_model.eval()
    print("\n训练完成！")
    return ode_model, ab, loss_history


# --- 3. 主程序 ---
if __name__ == '__main__':
    # 新建 NeuralODE 模型（级联结构）
    print("\n--- 初始化 NeuralODE 模型 ---")
    fnn_pretrained = NeuralODE(hidden_dim=16)

    # 加载sigmoid参数与DPS参数
    try:
        sigmoid_params = torch.load('sigmoid.pth', weights_only=False)
        print("成功加载 sigmoid.pth")
    except FileNotFoundError:
        print("错误: 未找到 sigmoid.pth，请先运行 pretrain.py")
        exit()

    try:
        dps_params_loaded = torch.load('dps.pth', weights_only=False)
        print("成功加载 dps.pth")
    except FileNotFoundError:
        print("错误: 未找到 dps.pth，请先运行 pretrain.py")
        exit()

    # 计算CN平均初值与AD平均值，用于调整sigmoid
    y_cn_avg = y0_cn_avg
    y_ad_avg = get_stage_mean_y(patient_data, 'AD', y_cn_avg)

    # 调整sigmoid：上平台~CN初值，下平台~AD均值，且在s=-20穿过CN初值
    sigmoid_params = adjust_sigmoid_params(sigmoid_params, y_cn_avg, y_ad_avg, s0=-20.0)

    # 构建s_grid用于阶段1
    s_grid_np = build_s_grid_from_dps(dps_params_loaded, patient_data)

    # 训练：Sigmoid轨线 + 导数约束
    print("\n--- 训练：Sigmoid轨线 + 导数约束 ---")
    fnn_trained, loss_history = train_neural_ode_to_sigmoid_with_dyds(
        fnn_pretrained,
        sigmoid_params,
        s_grid_np,
        epochs=2000,
        lr=1e-3,
        lambda_traj=1.0,
        lambda_dyds=1.0,
    )

    if fnn_trained is None:
        exit()

    final_model = ODEModel(fnn_trained, y_min01, y_max01).eval()

    # 保存训练后的模型
    torch.save(fnn_trained.state_dict(), f'{name}.pth')
    print(f"\n模型已保存到 {name}.pth")
    
    # --- 绘制损失曲线 ---
    print("\n--- 绘制损失曲线 ---")
    fig_loss, ax = plt.subplots(1, 1, figsize=(8, 5))
    
    steps = list(range(len(loss_history['traj_loss'])))
    
    ax.plot(steps, loss_history['traj_loss'], 'b-', linewidth=2, alpha=0.7, label='Trajectory L2')
    ax.plot(steps, loss_history['dyds_loss'], 'r-', linewidth=2, alpha=0.7, label='RHS L2')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Loss Curve', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    ax.legend()
    
    plt.tight_layout()
    loss_curve_filename = f'{name}_loss_curve.png'
    plt.savefig(loss_curve_filename)
    print(f"损失曲线已保存到 {loss_curve_filename}")
    plt.show()
    
    # --- 绘图 ---
    print("\n--- 生成可视化结果 ---")
    
    # 计算实际数据的s范围
    all_s_values = []
    for pid, dat in patient_data.items():
        if pid in dps_params_loaded:
            a = float(dps_params_loaded[pid]['a'])
            b = float(dps_params_loaded[pid]['b'])
            s_values = a * dat['t'].numpy() + b
            all_s_values.extend(s_values)
    
    s_min, s_max = np.min(all_s_values), np.max(all_s_values)
    s_margin = (s_max - s_min) * 0.1  # 扩展10%的边距
    s_grid = torch.linspace(s_min - s_margin, s_max + s_margin, 300)
    print(f"s_grid范围: [{s_grid.min():.2f}, {s_grid.max():.2f}]")
    
    with torch.no_grad():
        try:
            y_pred = torch_odeint(final_model, y0_cn_avg, s_grid, method='dopri5', rtol=1e-4, atol=1e-5)
            y_pred_orig = pc.inv_nor(y_pred.numpy())
        except Exception as e:
            print(f"绘图时ODE求解失败: {e}")
            exit()

        # 计算sigmoid曲线（原空间）
        y_sigmoid_np = get_sigmoid_values(sigmoid_params, s_grid.numpy())
        y_sigmoid_orig = pc.inv_nor(y_sigmoid_np)
        
        TITLES = ['Aβ (A)', 'p-Tau (T)', 'N', 'Cognition (C)']
        colors = {'CN': 'orange', 'LMCI': 'green', 'AD': 'blue', 'Other': 'grey'}
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 11))
        axes = axes.flat
        
        for k in range(4):
            ax = axes[k]
            
            # 绘制数据点
            for pid, dat in patient_data.items():
                if pid in dps_params_loaded:
                    stage = dat['stage']
                    a = float(dps_params_loaded[pid]['a'])
                    b = float(dps_params_loaded[pid]['b'])
                    s = a * dat['t'].numpy() + b
                    y_orig = pc.inv_nor(dat['y'][:, k].numpy(), k)
                    ax.scatter(s, y_orig, s=15, alpha=0.5, c=colors[stage])
            
            # 绘制FNN轨迹
            ax.plot(s_grid.numpy(), y_pred_orig[:, k], 'r-', lw=2.5, label='FNN Trajectory', zorder=3)

            # 绘制Sigmoid曲线
            ax.plot(s_grid.numpy(), y_sigmoid_orig[:, k], 'k--', lw=2.0, label='Sigmoid', zorder=2)
            
            ax.set_xlabel('Disease Progression Score (s)')
            ax.set_ylabel(TITLES[k])
            
            # 设置横轴范围与s_grid一致
            ax.set_xlim(s_grid.min().item(), s_grid.max().item())
            
            ax.legend()
            ax.grid(True, alpha=0.4)
            ax.set_title(TITLES[k])
        
        fig.suptitle('FNN Model (Fixed DPS) with Sigmoid Constraints', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f'{name}.png')
        print(f"结果图已保存到 {name}.png")
        plt.show()

    print("\n完整流程执行完毕。")
    
    print("\n完整流程执行完毕。")