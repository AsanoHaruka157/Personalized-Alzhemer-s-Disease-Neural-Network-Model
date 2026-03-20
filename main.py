import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pccmnn as pc
from torchdiffeq import odeint as torch_odeint

torch.set_default_dtype(torch.float64)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- 0. 資料載入和準備 ---
csf_dict = pc.load_data()
stage_dict = pc.load_stage_dict()
print(f"成功載入 {len(csf_dict)} 位患者的資料。")

# 转换数据格式以适应 PyTorch
patient_data = {}
for pid, sample in csf_dict.items():
    t = torch.from_numpy(sample[:, 0]).to(dtype=torch.get_default_dtype(), device=device)
    y = torch.from_numpy(sample[:, 1:5]).to(dtype=torch.get_default_dtype(), device=device)
    patient_data[pid] = {"t": t, "y": y, "y0": y[0].clone(), "stage": stage_dict.get(pid, 'Other')}

def get_cn_average_y0(patient_data, stage_dict):
    cn_y0s = []
    for pid, data in patient_data.items():
        if stage_dict.get(pid) == 'CN':
            cn_y0s.append(data['y0'])
    if not cn_y0s:
        print("警告: 未找到CN患者, 使用預設y0。")
        return torch.tensor([0.1, 0, 0, 0], device=device, dtype=torch.get_default_dtype())
    
    # 将所有CN患者的y0堆叠成矩阵
    cn_y0s_tensor = torch.stack(cn_y0s)  # shape: (num_cn_patients, 4)
    
    # 对每个生物标志物分别计算非NaN值的平均
    avg_y0 = torch.zeros(4, device=device, dtype=torch.get_default_dtype())
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

# ===== 全局超参数（主程序训练） =====
PRETRAIN_EPOCHS = 5000
PRETRAIN_LR = 1e-2
PRETRAIN_GAMMA = 0.999
LAMBDA_TRAJ = 1.0
LAMBDA_DYDS = 1.0           # 保留梯度场loss超参数（可选）
LAMBDA_DATA_L2 = 1e-2        # 误差方差倒数加权的 data L2 loss 权重
INV_VAR_EPS = 1e-8
USE_TRAJ_LOSS = True        # 主程序是否启用 trajectory loss
USE_GRADIENT_LOSS = False   # 主程序是否启用 gradient loss
USE_DATA_L2_LOSS = True    # 主程序是否启用 data l2 loss
FIGURE_PATH = 'main.png'
LOSS_PATH = 'main_loss.png'
MODEL_PATH = 'main.pt'


def compute_y_minmax_01(patient_data_dict):
    """
    统计每个 biomarker 的 min/max（忽略 NaN），用于“仅输入到神经网络时”归一化到[0,1]。
    注意：ODE 的状态 y 仍然在原空间演化，这里只是构造网络输入 y01。
    """
    ys = []
    for _, dat in patient_data_dict.items():
        ys.append(dat["y"])
    y_all = torch.cat(ys, dim=0)  # (N,4)
    y_min = torch.zeros(4, device=device, dtype=torch.get_default_dtype())
    y_max = torch.ones(4, device=device, dtype=torch.get_default_dtype())
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


class FNN(nn.Module):
    """
    单一前向神经网络：接收 4 维输入 y=[A, T, N, C]，输出 4 维导数 dy/ds
    """
    def __init__(self, input_dim=4, hidden_dim=128, output_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh(),
        )

    def forward(self, y):
        return self.net(y)


class ODEModel(nn.Module):
    """
    torchdiffeq 需要的 ODE 函数封装：func(t, y)
    - 仅对输入网络的 y 做[0,1]归一化（用全体数据 min/max）
    """
    def __init__(self, fnn_model: FNN, y_min: torch.Tensor, y_max: torch.Tensor):
        super().__init__()
        self.fnn = fnn_model
        self.register_buffer("y_min", y_min.to(dtype=torch.get_default_dtype(), device=device))
        self.register_buffer("y_max", y_max.to(dtype=torch.get_default_dtype(), device=device))

    def _norm01(self, y: torch.Tensor) -> torch.Tensor:
        y01 = (y - self.y_min) / (self.y_max - self.y_min)
        return torch.clamp(y01, 0.0, 1.0)

    def forward(self, t, y):
        squeeze_back = False
        if y.dim() == 1:
            y = y.unsqueeze(0)
            squeeze_back = True

        y01 = self._norm01(y)
        # 使用单一网络直接输出导数
        out = self.fnn(y01)
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
        exp_arg = np.clip(-b * (s_grid_np - c), -50.0, 50.0)
        y_on_grid[:, k] = a / (1.0 + np.exp(exp_arg)) + d
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
    sig = torch.tensor(sigmoid_params, dtype=torch.get_default_dtype(), device=device)  # (4,4)
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
    sig = np.array(sigmoid_params, dtype=np.float64)  # (4,4)
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

        adjusted[k] = np.array([a, b, c, d], dtype=np.float64)
    return adjusted


def build_s_grid_from_dps(dps_params_loaded, patient_data, margin_ratio=0.1, num_points=500):
    """
    固定 s_grid 范围：从 -20 到 30，间隔 0.01
    """
    s_grid_np = np.arange(-20.0, 30.0, 1.0)
    return s_grid_np


def train_neural_ode_to_sigmoid_with_dyds(
    fnn_model,
    sigmoid_params,
    s_grid_np,
    epochs=2000,
    lr=1e-3,
    gamma=0.997,
    lambda_traj=1.0,
    lambda_dyds=1e-3,
    lambda_data_l2=1.0,
    inv_var_eps=1e-8,
    use_traj_loss=True,
    use_gradient_loss=True,
    use_data_l2_loss=True,
):
    """
    训练 Neural ODE：
    - 轨线损失：ODE预测轨线 vs sigmoid曲线的L2
    - （可选）数据L2损失：误差方差倒数加权的L2
    - （可选）导数损失：rhs vs A(=sigmoid导数矩阵) 的L2
    """
    ode_model = ODEModel(fnn_model, y_min01, y_max01).train()
    criterion = nn.MSELoss()

    s_tensor = torch.tensor(s_grid_np, dtype=torch.get_default_dtype(), device=device)
    fnn_model = fnn_model.to(device)
    ode_model = ode_model.to(device)

    # 对s进行排序和去重，确保ODE求解器收到有效的输入
    s_sorted, sort_idx = torch.sort(s_tensor)
    s_unique, inverse_idx = torch.unique_consecutive(s_sorted, return_inverse=True)

    y_sigmoid_all, dyds_sigmoid_all = get_sigmoid_y_dyds_tensor(s_tensor, sigmoid_params)
    y_sigmoid, dyds_sigmoid = get_sigmoid_y_dyds_tensor(s_unique, sigmoid_params)

    # 如果有去重或排序，需要映射回来
    y_sigmoid_sorted = y_sigmoid_all[sort_idx]
    dyds_sigmoid_sorted = dyds_sigmoid_all[sort_idx]
    y_sigmoid_mapped = y_sigmoid_sorted[inverse_idx]
    dyds_sigmoid_mapped = dyds_sigmoid_sorted[inverse_idx]

    # 梯度场约束的输入/目标全局固定（由sigmoid唯一决定，与训练轮次无关）
    y_rhs_input = y_sigmoid_mapped.detach()
    A_target = dyds_sigmoid_mapped.detach()

    # 初值选用sigmoid起点
    y0 = y_sigmoid[0]

    # 记录loss历史
    loss_history = {
        'epoch': [],
        'traj_loss': [],
        'data_l2_loss': [],
        'dyds_loss': []
    }

    # 每10%打印一次进度
    print_interval = max(1, epochs // 10)

    optimizer = optim.Adam(ode_model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    for epoch in range(1, epochs + 1):
        loss_traj = torch.tensor(0.0, device=device, dtype=torch.get_default_dtype())
        loss_data_l2 = torch.tensor(0.0, device=device, dtype=torch.get_default_dtype())
        loss_dyds = torch.tensor(0.0, device=device, dtype=torch.get_default_dtype())

        optimizer.zero_grad()
        try:
            # 轨线预测（使用去重后的s）
            y_pred_unique = torch_odeint(ode_model, y0, s_unique, method='dopri5', rtol=1e-4, atol=1e-5)
            # 映射回原始s的索引
            y_pred = y_pred_unique[inverse_idx]

            if use_traj_loss:
                loss_traj = criterion(y_pred, y_sigmoid_mapped)

            # 误差方差倒数加权 data L2 loss（代码保留，可选启用）
            if use_data_l2_loss:
                err = y_pred - y_sigmoid_mapped  # (N,4)
                err_var = torch.var(err, dim=0, unbiased=False)  # (4,)
                inv_var = 1.0 / (err_var + inv_var_eps)          # (4,)
                loss_data_l2 = torch.mean((err ** 2) * inv_var.view(1, -1))

            # 梯度场约束（代码保留，可选启用）
            if use_gradient_loss:
                rhs = ode_model(torch.tensor(0.0, device=y_pred.device), y_rhs_input)
                loss_dyds = torch.norm((rhs - A_target).reshape(-1), p=2)

            loss = lambda_traj * loss_traj + lambda_data_l2 * loss_data_l2 + lambda_dyds * loss_dyds
            if torch.isfinite(loss):
                loss.backward()
                optimizer.step()
        except Exception as e:
            print(f"ODE求解失败: {e}")
            loss = torch.tensor(float('inf'), device=device, dtype=torch.get_default_dtype())

        scheduler.step()

        loss_history['epoch'].append(epoch)
        loss_history['traj_loss'].append(float(loss_traj))
        loss_history['data_l2_loss'].append(float(loss_data_l2))
        loss_history['dyds_loss'].append(float(loss_dyds))

        if epoch % print_interval == 0 or epoch == 1 or epoch == epochs:
            progress = int(epoch / epochs * 100)
            print(
                f"[Train {progress:3d}%] traj={float(loss_traj):.6f}, "
                f"data_l2={float(loss_data_l2):.6f}, dyds={float(loss_dyds):.6f}"
            )

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
            return torch.tensor(0.0, device=device, dtype=torch.get_default_dtype(), requires_grad=True)

        s_all = torch.cat(s_all_list)
        y_true_all = torch.cat(y_true_list)
        k_all = torch.tensor(k_list, device=y_true_all.device, dtype=torch.long)
        y_true_all = y_true_all.to(dtype=torch.get_default_dtype())

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
        sigmoid_loss = torch.tensor(0.0, device=y_unique.device, dtype=torch.get_default_dtype())
        dyds_reg_loss = torch.tensor(0.0, device=y_unique.device, dtype=torch.get_default_dtype())
        if sigmoid_params is not None:
            # 将sigmoid参数转为tensor并放到设备上
            sig = torch.tensor(sigmoid_params, dtype=torch.get_default_dtype(), device=y_unique.device)  # (4,4)
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
            s_sample = (torch.rand(n_dyds_samples, device=y_unique.device, dtype=torch.get_default_dtype()) * (s_hi - s_lo) + s_lo)
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
            return torch.tensor(float('inf'), device=device, dtype=torch.get_default_dtype(), requires_grad=True), loss_dict

    except Exception as e:
        print(f"FNN loss 计算出错: {e}")
        import traceback
        traceback.print_exc()
        loss_dict = {'total': float('inf'), 'data': 0, 'l1': 0, 'lambda': 0.0001}
        return torch.tensor(float('inf'), device=device, dtype=torch.get_default_dtype(), requires_grad=True), loss_dict


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
            return torch.tensor(0.0, device=device, dtype=torch.get_default_dtype(), requires_grad=True), {'total': 0.0, 'data': 0.0, 'l2': 0.0}

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
            return torch.tensor(float('inf'), device=device, dtype=torch.get_default_dtype(), requires_grad=True), loss_dict

    except Exception as e:
        print(f"DPS loss 计算出错: {e}")
        loss_dict = {'total': float('inf'), 'data': 0.0, 'l2': 0.0}
        return torch.tensor(float('inf'), device=device, dtype=torch.get_default_dtype(), requires_grad=True), loss_dict

def train_fnn_with_fixed_dps(
    fnn_pretrained,
    patient_data,
    y0,
    dps_path='dps.pth',
    n_epochs=2000,
    lr_fnn=1e-2,
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
                    'a': torch.tensor(dps_params_loaded[pid]['a'], dtype=torch.get_default_dtype(), device=device),
                    'b': torch.tensor(dps_params_loaded[pid]['b'], dtype=torch.get_default_dtype(), device=device)
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
    # 新建 FNN 模型（单一网络结构）
    print("\n--- 初始化 FNN 模型 ---")
    fnn_pretrained = FNN(input_dim=4, hidden_dim=128, output_dim=4).to(device)

    # 加载预训练模型参数
    try:
        fnn_pretrained.load_state_dict(torch.load('pretrain.pth', map_location=device, weights_only=False))
        print("成功加载 pretrain.pth")
    except FileNotFoundError:
        print("错误: 未找到 pretrain.pth，请先运行 pretrain.py")
        exit()

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
    y_cn_avg = y0_cn_avg.to(device)
    y_ad_avg = get_stage_mean_y(patient_data, 'AD', y_cn_avg).to(device)


    # 构建s_grid用于阶段1
    s_grid_np = build_s_grid_from_dps(dps_params_loaded, patient_data)

    # 训练：Sigmoid轨线（data L2 / gradient loss 可按超参数开关）
    print("\n--- 训练：Sigmoid轨线 ---")
    fnn_trained, loss_history = train_neural_ode_to_sigmoid_with_dyds(
        fnn_pretrained,
        sigmoid_params,
        s_grid_np,
        epochs=PRETRAIN_EPOCHS,
        lr=PRETRAIN_LR,
        gamma=PRETRAIN_GAMMA,
        lambda_traj=LAMBDA_TRAJ,
        lambda_dyds=LAMBDA_DYDS,
        lambda_data_l2=LAMBDA_DATA_L2,
        inv_var_eps=INV_VAR_EPS,
        use_traj_loss=USE_TRAJ_LOSS,
        use_gradient_loss=USE_GRADIENT_LOSS,
        use_data_l2_loss=USE_DATA_L2_LOSS,
    )

    if fnn_trained is None:
        exit()

    final_model = ODEModel(fnn_trained, y_min01, y_max01).to(device).eval()

    # 保存训练后的模型
    torch.save(fnn_trained.state_dict(), MODEL_PATH)
    print(f"\n模型已保存到 {MODEL_PATH}")
    
    # --- 绘制损失曲线 ---
    print("\n--- 绘制损失曲线 ---")
    steps = list(range(len(loss_history['traj_loss'])))

    plot_items = []
    if USE_TRAJ_LOSS:
        plot_items.append(('traj_loss', 'Trajectory L2', 'b-'))
    if USE_DATA_L2_LOSS:
        plot_items.append(('data_l2_loss', 'Inv-Var Weighted Data L2', 'g-'))
    if USE_GRADIENT_LOSS:
        plot_items.append(('dyds_loss', 'RHS L2 (optional)', 'r-'))

    if len(plot_items) > 0:
        fig_loss, axes = plt.subplots(len(plot_items), 1, figsize=(8, 4 * len(plot_items)), squeeze=False)
        axes = axes.flatten()

        for i, (key, title, style) in enumerate(plot_items):
            ax = axes[i]
            ax.plot(steps, loss_history[key], style, linewidth=2, alpha=0.8)
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('Loss', fontsize=12)
            ax.set_title(title, fontsize=13)
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')

        plt.tight_layout()
        loss_curve_filename = f'{LOSS_PATH}'
        plt.savefig(loss_curve_filename)
        print(f"损失曲线已保存到 {LOSS_PATH}.png")
        plt.show()
    else:
        print("所有loss绘图开关均为False，跳过损失曲线绘制。")
    
    # --- 绘图 ---
    print("\n--- 生成可视化结果 ---")
    
    # 固定可视化范围
    s_grid = torch.arange(-20.0, 30.0 + 0.1, 0.1, device=device, dtype=torch.get_default_dtype())
    print(f"s_grid范围: [{s_grid.min():.2f}, {s_grid.max():.2f}]")
    
    # 对s进行排序和去重，确保ODE求解器收到有效的输入
    s_sorted, sort_idx = torch.sort(s_grid)
    s_unique, inverse_idx = torch.unique_consecutive(s_sorted, return_inverse=True)
    
    with torch.no_grad():
        try:
            # 使用 sigmoid 起点作为初值（与 pretrain 一致）
            y_sigmoid_plot, _ = get_sigmoid_y_dyds_tensor(s_unique, sigmoid_params, device=device)
            y0_plot = y_sigmoid_plot[0]

            y_pred_unique = torch_odeint(final_model, y0_plot, s_unique, method='dopri5', rtol=1e-4, atol=1e-5)
            # 映射回原始s的索引
            y_pred = y_pred_unique[inverse_idx]
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
                    ax.scatter(s, y_orig, s=22, alpha=0.8, c=colors[stage], edgecolors='none', zorder=1)
            
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
        plt.savefig(f'{FIGURE_PATH}')
        print(f"结果图已保存到 {FIGURE_PATH}")
        plt.show()

    print("\n完整流程执行完毕。")
    
    print("\n完整流程执行完毕。")