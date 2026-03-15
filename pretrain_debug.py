import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import pccmnn as pc


def assign_dps_params(csf_dict, stage_dict):
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


def sigmoid(s, a, b, c, d):
    return a / (1.0 + np.exp(-b * (s - c))) + d


def _sigmoid_regularized_residuals(params, s_valid, y_valid, reg_cfg):
    a, b, c, d = params

    y_pred = sigmoid(s_valid, a, b, c, d)
    residuals = y_pred - y_valid

    b_safe = b if np.abs(b) > 1e-6 else 1e-6
    turn_left = c - (np.log(2.0) / b_safe)
    turn_right = c + (np.log(2.0) / b_safe)

    w_turn = reg_cfg['w_turn']
    w_center = reg_cfg['w_center']
    w_curv = reg_cfg['w_curv']
    target_b_abs = reg_cfg['target_b_abs']

    turn_pen_left = (turn_left - 0.0) * np.sqrt(w_turn)
    turn_pen_right = (turn_right - 10.0) * np.sqrt(w_turn)

    center_low = max(0.0, -c) * np.sqrt(w_center)
    center_high = max(0.0, c - 10.0) * np.sqrt(w_center)

    curv_pen = max(0.0, target_b_abs - np.abs(b)) * np.sqrt(w_curv)

    reg_terms = np.array([turn_pen_left, turn_pen_right, center_low, center_high, curv_pen])
    return np.concatenate([residuals, reg_terms])


def fit_sigmoids_regularized(s_data, y_data, reg_cfg=None):
    if reg_cfg is None:
        reg_cfg = {
            'w_turn': 5.0,
            'w_center': 2.0,
            'w_curv': 3.0,
            'target_b_abs': 0.6
        }

    sigmoid_params = []
    for k in range(4):
        y_k = y_data[:, k]
        valid_mask = ~np.isnan(y_k)
        s_k_valid = s_data[valid_mask]
        y_k_valid = y_k[valid_mask]

        if len(y_k_valid) < 5:
            sigmoid_params.append([1.0, 1.0, 5.0, 0.0])
            continue

        amp_init = np.max(y_k_valid) - np.min(y_k_valid)
        center_init = np.median(s_k_valid)
        corr = np.corrcoef(s_k_valid, y_k_valid)[0, 1]
        slope_sign = -1.0 if corr < 0 else 1.0
        p0 = [amp_init, 1.0 * slope_sign, center_init, np.min(y_k_valid)]

        result = least_squares(
            _sigmoid_regularized_residuals,
            x0=p0,
            args=(s_k_valid, y_k_valid, reg_cfg),
            max_nfev=10000
        )
        sigmoid_params.append(result.x)

    return np.array(sigmoid_params)


def sigmoid_torch(s_tensor, params_np):
    params = torch.tensor(params_np, dtype=torch.float32, device=s_tensor.device)
    a = params[:, 0].view(1, 4)
    b = params[:, 1].view(1, 4)
    c = params[:, 2].view(1, 4)
    d = params[:, 3].view(1, 4)

    s = s_tensor.view(-1, 1)
    exp_term = torch.exp(-b * (s - c))
    y = a / (1.0 + exp_term) + d
    return y


def build_population_tensors(csf_dict, dps_params):
    s_list = []
    y_list = []
    for pid, sample in csf_dict.items():
        t_np = sample[:, 0]
        y_np = sample[:, 1:5]

        params = dps_params[pid]
        a = params['a']
        b = params['b']

        t = torch.tensor(t_np, dtype=torch.float32)
        y = torch.tensor(y_np, dtype=torch.float32)
        s = a * t + b

        s_list.append(s)
        y_list.append(y)

    s_tensor = torch.cat(s_list, dim=0)
    y_tensor = torch.cat(y_list, dim=0)
    return s_tensor, y_tensor


def train_dps_params_lbfgs(csf_dict, dps_params, sigmoid_params, epochs=200, lr=1e-2):
    params_to_opt = []
    for pid, params in dps_params.items():
        params_to_opt.append(params['a'])
        params_to_opt.append(params['b'])

    optimizer = optim.LBFGS(params_to_opt, lr=lr, max_iter=20, history_size=10, line_search_fn="strong_wolfe")
    mse_criterion = nn.MSELoss()

    for _ in range(epochs):
        def closure():
            optimizer.zero_grad()
            s_tensor, y_tensor = build_population_tensors(csf_dict, dps_params)
            y_pred = sigmoid_torch(s_tensor, sigmoid_params)
            valid_mask = ~torch.isnan(y_tensor)
            loss = mse_criterion(y_pred[valid_mask], y_tensor[valid_mask])
            loss.backward()
            return loss

        optimizer.step(closure)


def get_sigmoid_values(sigmoid_params, s_grid_np):
    y_on_grid = np.zeros((len(s_grid_np), 4))
    for k in range(4):
        a, b, c, d = sigmoid_params[k]
        y_on_grid[:, k] = a / (1.0 + np.exp(-b * (s_grid_np - c))) + d
    return y_on_grid


def plot_panel(ax, s_pop, y_pop_norm, stages_pop, s_grid, sigmoid_params, biomarker_idx, title):
    y_pop_orig = pc.inv_nor(y_pop_norm)

    y_sigmoid_grid_norm = get_sigmoid_values(sigmoid_params, s_grid)
    y_sigmoid_grid_orig = pc.inv_nor(y_sigmoid_grid_norm)

    titles = ['Aβ (A)', 'p-Tau (T)', 'N', 'Cognition (C)']
    colors = {'CN': 'orange', 'LMCI': 'green', 'AD': 'blue', 'Other': 'grey'}

    ax.set_title(f"{title} - {titles[biomarker_idx]}")
    ax.set_xlabel('Disease Progression Score (s)')
    ax.set_ylabel('Biomarker value')
    ax.grid(True, alpha=0.4)

    unique_stages = np.unique(stages_pop)
    for stage in unique_stages:
        mask = np.array(stages_pop) == stage
        ax.scatter(
            s_pop[mask],
            y_pop_orig[mask][:, biomarker_idx],
            s=10,
            alpha=0.4,
            c=colors[stage]
        )

    ax.plot(s_grid, y_sigmoid_grid_orig[:, biomarker_idx], lw=2.0)


def main():
    csf_dict = pc.load_data()
    stage_dict = pc.load_stage_dict()

    dps_params = assign_dps_params(csf_dict, stage_dict)

    # fig1: 初始dps
    patient_data, s_pop, y_pop_norm, stages_pop = compute_s_values(csf_dict, dps_params)
    s_min, s_max = s_pop.min(), s_pop.max()
    s_margin = (s_max - s_min) * 0.1
    s_grid = np.linspace(s_min - s_margin, s_max + s_margin, 300)

    sigmoid_params = fit_sigmoids_regularized(s_pop, y_pop_norm)

    # fig2: 初次sigmoid拟合
    patient_data2, s_pop2, y_pop_norm2, stages_pop2 = compute_s_values(csf_dict, dps_params)

    # fig3: LBFGS优化dps
    train_dps_params_lbfgs(csf_dict, dps_params, sigmoid_params, epochs=200, lr=1e-2)
    patient_data3, s_pop3, y_pop_norm3, stages_pop3 = compute_s_values(csf_dict, dps_params)

    s_min3, s_max3 = s_pop3.min(), s_pop3.max()
    s_margin3 = (s_max3 - s_min3) * 0.1
    s_grid3 = np.linspace(s_min3 - s_margin3, s_max3 + s_margin3, 300)

    fig, axes = plt.subplots(4, 3, figsize=(18, 20), sharex='col')

    panel_titles = ['Fig1: Init DPS', 'Fig2: Fit Sigmoid', 'Fig3: DPS after LBFGS']
    panel_data = [
        (s_pop, y_pop_norm, stages_pop, s_grid),
        (s_pop2, y_pop_norm2, stages_pop2, s_grid),
        (s_pop3, y_pop_norm3, stages_pop3, s_grid3)
    ]

    for row in range(4):
        for col in range(3):
            s_p, y_p, st_p, s_g = panel_data[col]
            plot_panel(
                axes[row, col],
                s_p,
                y_p,
                st_p,
                s_g,
                sigmoid_params,
                row,
                panel_titles[col]
            )

    plt.tight_layout()
    plt.savefig('debug.png')
    plt.show()


if __name__ == "__main__":
    main()
