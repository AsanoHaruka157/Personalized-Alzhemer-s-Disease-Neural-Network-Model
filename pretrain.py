import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import pccmnn as pc  # 假设您有这个文件来加载和反归一化数据

# --- 0. 数据加载和准备 ---
csf_dict = pc.load_data()
stage_dict = pc.load_stage_dict()
print(f"成功加载 {len(csf_dict)} 位患者的数据。")

# --- 1. 为每位患者分配DPS变换参数（可优化版本）---
def assign_dps_params(csf_dict, stage_dict):
    """
    为每位患者创建可优化的DPS变换参数 a 和 b。
    a: CN=1, LMCI=2, AD=4 的初始值，但可优化
    b: 随机初始化，但可优化
    """
    s_ranges = {
        'CN': (-10, 0),
        'LMCI': (-2, 8),
        'AD': (5, 20),
        'Other': (-10, 20)
    }
    a_init_values = {'CN': 1.0, 'LMCI': 2.0, 'AD': 4.0, 'Other': 1.0}

    dps_params = {}
    for pid, sample in csf_dict.items():
        stage = stage_dict.get(pid, 'Other')
        t = sample[:, 0]

        a_init = a_init_values[stage]
        a_param = nn.Parameter(torch.tensor(a_init, dtype=torch.float32))

        s_min, s_max = s_ranges[stage]
        t_initial = t[0]
        s_initial_target = np.random.uniform(s_min, s_max)
        b_init = s_initial_target - a_init * t_initial
        b_param = nn.Parameter(torch.tensor(b_init, dtype=torch.float32))

        dps_params[pid] = {'a': a_param, 'b': b_param, 'stage': stage}

    return dps_params


def compute_s_values(csf_dict, dps_params):
    """
    根据当前的DPS参数计算所有患者的s值
    """
    patient_data = {}
    all_s_points = []
    all_y_points = []
    all_stages = []

    for pid, sample in csf_dict.items():
        t = sample[:, 0]
        y = sample[:, 1:5]

        params = dps_params[pid]
        a = params['a']
        b = params['b']
        stage = params['stage']

        s = a * torch.tensor(t, dtype=torch.float32) + b

        patient_data[pid] = {
            't': t,
            'y': y,
            's': s.detach().numpy(),
            'stage': stage,
            'a': a.item(),
            'b': b.item()
        }

        all_s_points.append(s.detach().numpy())
        all_y_points.append(y)
        all_stages.extend([stage] * len(t))

    s_population = np.concatenate(all_s_points)
    y_population = np.concatenate(all_y_points)

    return patient_data, s_population, y_population, all_stages


def get_cn_average_y0(patient_data):
    """
    计算CN（认知正常）群体在第一次访问时的平均生物标记物值（忽略NaN）。
    """
    cn_y0s = []
    for pid, data in patient_data.items():
        if data['stage'] == 'CN':
            cn_y0s.append(data['y'][0])

    if not cn_y0s:
        print("警告：未找到CN患者数据，将使用默认初始值 [0.1, 0, 0, 0]。")
        return np.array([0.1, 0.0, 0.0, 0.0])

    cn_y0s_array = np.array(cn_y0s)
    avg_y0 = np.nanmean(cn_y0s_array, axis=0)
    avg_y0 = np.nan_to_num(avg_y0, nan=0.0)

    print(f"计算出的CN群体平均初始值（非NaN，归一化后）: {avg_y0}")
    return avg_y0


def get_cn_average_s0(patient_data):
    """
    计算CN群体第一次访问对应的平均s值。
    """
    cn_s0s = []
    for pid, data in patient_data.items():
        if data['stage'] == 'CN':
            cn_s0s.append(data['s'][0])

    if not cn_s0s:
        print("警告：未找到CN患者s0，将使用默认s0=0.0。")
        return 0.0

    avg_s0 = float(np.nanmean(np.array(cn_s0s)))
    if np.isnan(avg_s0):
        print("警告：CN s0均为NaN，将使用默认s0=0.0。")
        return 0.0

    print(f"计算出的CN群体平均s0: {avg_s0}")
    return avg_s0


def get_ad_average_y(patient_data):
    """
    计算AD群体所有时间点的平均生物标记物值（忽略NaN）。
    """
    ad_ys = []
    for pid, data in patient_data.items():
        if data['stage'] == 'AD':
            ad_ys.append(data['y'])

    if not ad_ys:
        print("警告：未找到AD患者数据，将使用默认值 [0.0, 0.0, 0.0, 0.0]。")
        return np.array([0.0, 0.0, 0.0, 0.0])

    ad_ys_array = np.concatenate(ad_ys, axis=0)
    avg_y = np.nanmean(ad_ys_array, axis=0)
    avg_y = np.nan_to_num(avg_y, nan=0.0)

    print(f"计算出的AD群体平均值（非NaN，归一化后）: {avg_y}")
    return avg_y


# --- 2. Sigmoid模型（使用scipy优化 + 正则约束）---

def sigmoid(s, a, b, c, d):
    exp_arg = np.clip(-b * (s - c), -50.0, 50.0)
    exp_term = np.exp(exp_arg)
    y = a / (1.0 + exp_term) + d
    return y


def _sigmoid_regularized_residuals(params, s_valid, y_valid, reg_cfg):
    a, b, c, d = params

    y_pred = sigmoid(s_valid, a, b, c, d)
    residuals = y_pred - y_valid

    w_turn = reg_cfg['w_turn']
    w_center = reg_cfg['w_center']
    w_curv = reg_cfg['w_curv']
    w_cn = reg_cfg['w_cn']
    w_plat = reg_cfg['w_plat']
    target_b_abs = reg_cfg['target_b_abs']
    target_b_max = reg_cfg['target_b_max']
    target_amp_max = reg_cfg['target_amp_max']

    # 让曲线经过指定点 (s_target, y_target)
    s_target = reg_cfg.get('s_target', 0.0)
    y_target = reg_cfg.get('y_target', 0.0)
    pass_pen = (sigmoid(s_target, a, b, c, d) - y_target) * np.sqrt(w_cn)

    # 上/下平台接近 CN 与 AD 平均值
    y_cn = reg_cfg.get('y_cn', 0.0)
    y_ad = reg_cfg.get('y_ad', 0.0)
    upper = np.maximum(d, d + a)
    lower = np.minimum(d, d + a)
    plat_pen_upper = (upper - y_cn) * np.sqrt(w_plat)
    plat_pen_lower = (lower - y_ad) * np.sqrt(w_plat)

    # 中心点在[0,10]内
    center_low = max(0.0, -c) * np.sqrt(w_center)
    center_high = max(0.0, c - 10.0) * np.sqrt(w_center)

    # 控制曲率：|b| 处于 [target_b_abs, target_b_max]
    curv_pen_low = max(0.0, target_b_abs - np.abs(b)) * np.sqrt(w_curv)
    curv_pen_high = max(0.0, np.abs(b) - target_b_max) * np.sqrt(w_curv)

    # 幅度不要过大：|a| <= target_amp_max
    amp_pen = max(0.0, np.abs(a) - target_amp_max) * np.sqrt(w_curv)

    # 增强"两个拐点"约束：不仅要接近目标值，还要在指定范围内
    b_safe = b if np.abs(b) > 1e-6 else 1e-6
    turn_left = c - (np.log(2.0) / b_safe)
    turn_right = c + (np.log(2.0) / b_safe)
    
    # 左拐点约束：接近0且在[-2, 2]范围内
    turn_pen_left = (turn_left - 0.0) ** 2 * np.sqrt(w_turn)
    if turn_left < -2.0:
        turn_pen_left += ((-2.0 - turn_left) ** 2) * np.sqrt(w_turn)
    elif turn_left > 2.0:
        turn_pen_left += ((turn_left - 2.0) ** 2) * np.sqrt(w_turn)
    
    # 右拐点约束：接近10且在[8, 12]范围内
    turn_pen_right = (turn_right - 10.0) ** 2 * np.sqrt(w_turn)
    if turn_right < 8.0:
        turn_pen_right += ((8.0 - turn_right) ** 2) * np.sqrt(w_turn)
    elif turn_right > 12.0:
        turn_pen_right += ((turn_right - 12.0) ** 2) * np.sqrt(w_turn)

    reg_terms = np.array([
        turn_pen_left,
        turn_pen_right,
        center_low,
        center_high,
        curv_pen_low,
        curv_pen_high,
        amp_pen,
        pass_pen,
        plat_pen_upper,
        plat_pen_lower
    ])

    return np.concatenate([residuals, reg_terms])


def _sigmoid_fit_loss(params, s_valid, y_valid, reg_cfg):
    a, b, c, d = params
    y_pred = sigmoid(s_valid, a, b, c, d)
    residuals = y_pred - y_valid

    w_turn = reg_cfg['w_turn']
    w_center = reg_cfg['w_center']
    w_curv = reg_cfg['w_curv']
    w_cn = reg_cfg['w_cn']
    w_plat = reg_cfg['w_plat']
    target_b_abs = reg_cfg['target_b_abs']
    target_b_max = reg_cfg.get('target_b_max', 0.5)

    # 让曲线经过指定点 (s_target, y_target)
    s_target = reg_cfg.get('s_target', 0.0)
    y_target = reg_cfg.get('y_target', 0.0)
    pass_pen = (sigmoid(s_target, a, b, c, d) - y_target) * np.sqrt(w_cn)

    # 上/下平台接近 CN 与 AD 平均值
    y_cn = reg_cfg.get('y_cn', 0.0)
    y_ad = reg_cfg.get('y_ad', 0.0)
    upper = np.maximum(d, d + a)
    lower = np.minimum(d, d + a)
    plat_pen_upper = (upper - y_cn) * np.sqrt(w_plat)
    plat_pen_lower = (lower - y_ad) * np.sqrt(w_plat)

    # 中心点在[0,10]内
    center_low = max(0.0, -c) * np.sqrt(w_center)
    center_high = max(0.0, c - 10.0) * np.sqrt(w_center)

    # 控制曲率：|b| 处于 [target_b_abs, target_b_max]
    curv_pen_low = max(0.0, target_b_abs - np.abs(b)) * np.sqrt(w_curv)
    curv_pen_high = max(0.0, np.abs(b) - target_b_max) * np.sqrt(w_curv)

    # 增强"两个拐点"约束：不仅要接近目标值，还要在指定范围内
    b_safe = b if np.abs(b) > 1e-6 else 1e-6
    turn_left = c - (np.log(2.0) / b_safe)
    turn_right = c + (np.log(2.0) / b_safe)
    
    # 左拐点约束：接近0且在[-2, 2]范围内
    turn_pen_left = (turn_left - 0.0) ** 2 * np.sqrt(w_turn)
    if turn_left < -2.0:
        turn_pen_left += ((-2.0 - turn_left) ** 2) * np.sqrt(w_turn)
    elif turn_left > 2.0:
        turn_pen_left += ((turn_left - 2.0) ** 2) * np.sqrt(w_turn)
    
    # 右拐点约束：接近10且在[8, 12]范围内
    turn_pen_right = (turn_right - 10.0) ** 2 * np.sqrt(w_turn)
    if turn_right < 8.0:
        turn_pen_right += ((8.0 - turn_right) ** 2) * np.sqrt(w_turn)
    elif turn_right > 12.0:
        turn_pen_right += ((turn_right - 12.0) ** 2) * np.sqrt(w_turn)

    reg_terms = np.array([
        turn_pen_left,
        turn_pen_right,
        center_low,
        center_high,
        curv_pen_low,
        curv_pen_high,
        pass_pen,
        plat_pen_upper,
        plat_pen_lower
    ])
    all_res = np.concatenate([residuals, reg_terms])
    return float(np.mean(all_res ** 2))


def fit_sigmoids_regularized(s_data, y_data, reg_cfg=None):
    """
    为4个biomarker拟合sigmoid函数，正则约束：
    1) 两个拐点在0、10附近
    2) 中心点在0~10之间
    3) 曲率更大（|b|更大）
    4) 幅度不要过大
    5) 通过指定点 (s_target, y_target)
    6) 上/下平台接近 CN/AD 平均值
    """
    if reg_cfg is None:
        reg_cfg = {
            'w_turn': 12.0,
            'w_center': 4.0,
            'w_curv': 3.0,
            'w_cn': 8.0,
            'w_plat': 6.0,
            'target_b_abs': 0.2,
            'target_b_max': 0.4,
            'target_amp_max': 2.0
        }

    sigmoid_params = []
    total_loss = 0.0
    for k in range(4):
        y_k = y_data[:, k]
        valid_mask = ~np.isnan(y_k)
        s_k_valid = s_data[valid_mask]
        y_k_valid = y_k[valid_mask]

        reg_cfg_k = dict(reg_cfg)
        y_cn = reg_cfg.get('y_cn', 0.0)
        y_ad = reg_cfg.get('y_ad', 0.0)
        reg_cfg_k['y_cn'] = y_cn[k] if isinstance(y_cn, (list, np.ndarray)) else y_cn
        reg_cfg_k['y_ad'] = y_ad[k] if isinstance(y_ad, (list, np.ndarray)) else y_ad
        y_target = reg_cfg.get('y_target', 0.0)
        reg_cfg_k['y_target'] = y_target[k] if isinstance(y_target, (list, np.ndarray)) else y_target
        reg_cfg_k['s_target'] = reg_cfg.get('s_target', -20.0)

        if len(y_k_valid) < 5:
            default_params = np.array([1.0, 1.0, 5.0, 0.0])
            sigmoid_params.append(default_params)
            total_loss += _sigmoid_fit_loss(default_params, s_k_valid, y_k_valid, reg_cfg_k)
            continue

        amp_init = np.max(y_k_valid) - np.min(y_k_valid)
        center_init = np.median(s_k_valid)
        slope_sign = -1.0 if np.corrcoef(s_k_valid, y_k_valid)[0, 1] < 0 else 1.0
        p0 = [amp_init, 0.3 * slope_sign, center_init, np.min(y_k_valid)]

        result = least_squares(
            _sigmoid_regularized_residuals,
            x0=p0,
            args=(s_k_valid, y_k_valid, reg_cfg_k),
            max_nfev=10000
        )
        sigmoid_params.append(result.x)
        total_loss += _sigmoid_fit_loss(result.x, s_k_valid, y_k_valid, reg_cfg_k)

    return np.array(sigmoid_params), float(total_loss)


# --- 3. 训练DPS参数（逻辑不变，仅使用sigmoid作为拟合目标）---
# --- 3. 仅Sigmoid训练 ---
def train_sigmoid_only(csf_dict, stage_dict):
    print("\n开始仅Sigmoid训练（不拟合DPS参数）...")
    dps_params = assign_dps_params(csf_dict, stage_dict)

    patient_data, s_pop, y_pop_norm, _ = compute_s_values(csf_dict, dps_params)
    y0_cn = get_cn_average_y0(patient_data)
    y_ad = get_ad_average_y(patient_data)

    sigmoid_params, sigmoid_loss = fit_sigmoids_regularized(
        s_pop,
        y_pop_norm,
        reg_cfg={
            'w_turn': 12.0,
            'w_center': 4.0,
            'w_curv': 3.0,
            'w_cn': 10.0,
            'w_plat': 8.0,
            'target_b_abs': 0.2,
            'target_b_max': 0.5,
            'target_amp_max': 2.0,
            's_target': -20.0,
            'y_target': y0_cn,
            'y_cn': y0_cn,
            'y_ad': y_ad
        }
    )

    torch.save(sigmoid_params, 'sigmoid.pth')
    print(f"Sigmoid训练完成，Loss: {sigmoid_loss:.6f}")

    return dps_params, sigmoid_params


# --- 5. 绘图 ---
def get_sigmoid_derivatives(s_grid, params):
    y_on_grid = np.zeros((len(s_grid), 4))
    dyds_on_grid = np.zeros((len(s_grid), 4))

    for k in range(4):
        a, b, c, d = params[k]
        exp_arg = np.clip(-b * (s_grid - c), -50.0, 50.0)
        exp_term = np.exp(exp_arg)
        denom = (1.0 + exp_term) ** 2
        y_on_grid[:, k] = a / (1.0 + exp_term) + d
        dyds_on_grid[:, k] = np.divide(a * b * exp_term, denom, out=np.zeros_like(exp_term), where=denom != 0)

    return y_on_grid, dyds_on_grid


def plot_results(s_pop, y_pop, stages_pop, s_grid, sigmoid_params):
    print("正在生成最终结果图...")

    y_pop_orig = pc.inv_nor(y_pop)
    y_sigmoid_grid_norm, _ = get_sigmoid_derivatives(s_grid, sigmoid_params)
    y_sigmoid_grid_orig = pc.inv_nor(y_sigmoid_grid_norm)

    TITLES = ['Aβ (A)', 'p-Tau (T)', 'N', 'Cognition (C)']
    colors = {'CN': 'orange', 'LMCI': 'green', 'AD': 'blue', 'Other': 'grey'}

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flat

    unique_stages = np.unique(stages_pop)
    scatter_data = {}
    for stage in unique_stages:
        mask = np.array(stages_pop) == stage
        scatter_data[stage] = (s_pop[mask], y_pop_orig[mask])

    for k in range(4):
        ax = axes[k]
        for stage in unique_stages:
            s_vals, y_vals = scatter_data[stage]
            ax.scatter(s_vals, y_vals[:, k], s=15, alpha=0.5, c=colors[stage], label=stage)

        ax.plot(s_grid, y_sigmoid_grid_orig[:, k], 'r-', lw=2.5, label='Sigmoid Fit', zorder=3)

        ax.set_xlabel('Disease Progression Score (s)')
        ax.set_ylabel(TITLES[k])
        ax.set_xlim(s_grid.min(), s_grid.max())
        ax.legend()
        ax.grid(True, alpha=0.4)
        ax.set_title(f'Trajectory for {TITLES[k]}')

    fig.suptitle('AD Biomarker Trajectories: Sigmoid Fit', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('pretrain.png')
    plt.show()


if __name__ == '__main__':
    print("开始预训练流程...")

    # 1. 仅训练sigmoid（不拟合DPS参数）
    dps_params, _ = train_sigmoid_only(
        csf_dict,
        stage_dict
    )

    # 2. 读取最优sigmoid参数
    print("读取最优sigmoid参数...")
    sigmoid_params = torch.load('sigmoid.pth', weights_only=False)

    # 3. 计算最终s值与患者数据
    print("计算最终s值...")
    patient_data, s_pop, y_pop_norm, stages_pop = compute_s_values(csf_dict, dps_params)

    # 4. 计算CN群体平均初始值（如需后续使用）
    print("计算CN群体平均初始值...")
    _ = get_cn_average_y0(patient_data)

    # 5. 绘图
    print("生成结果图表...")
    s_min, s_max = s_pop.min(), s_pop.max()
    s_margin = (s_max - s_min) * 0.1
    s_grid = np.linspace(s_min - s_margin, s_max + s_margin, 300)

    plot_results(s_pop, y_pop_norm, stages_pop, s_grid, sigmoid_params)
