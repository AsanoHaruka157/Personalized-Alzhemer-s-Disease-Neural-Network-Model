"""
Sigmoid curve fitting for AD biomarker trajectories.

This script fits regularized sigmoid curves to the four key Alzheimer's disease
biomarkers (Aβ, p-Tau, N, Cognition) across the disease progression score (s).
It also assigns initial DPS (Disease Progression Score) transformation parameters
to each patient based on their diagnostic stage.

This is the first stage of the experimental pipeline: sigmoid fitting provides
the target trajectories that the FNN-based Neural ODE will later learn to reproduce.

Outputs:
  - sigmoid.pth: fitted sigmoid parameters (4 biomarkers x 4 params each)
  - dps.pth: per-patient DPS transformation parameters (a, b)
  - sigmoid.png: visualization of fitted sigmoid curves against patient data
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import pccmnn as pc


# ==============================================================================
# 0. Data loading
# ==============================================================================
csf_dict = pc.load_data()
stage_dict = pc.load_stage_dict()
print(f"Successfully loaded data for {len(csf_dict)} patients.")


# ==============================================================================
# 1. DPS parameter assignment
# ==============================================================================
def assign_dps_params(csf_dict, stage_dict):
    """Assign initial (optimizable) DPS transformation parameters per patient.

    The DPS transformation maps calendar time t to disease progression score s:
        s = a * t + b

    Initial values are set based on diagnostic stage:
      - CN:    a=1.0, s in [-10, 0]
      - LMCI:  a=2.0, s in [-2, 8]
      - AD:    a=4.0, s in [5, 20]
      - Other: a=1.0, s in [-10, 20]

    Args:
        csf_dict: Patient biomarker data dictionary.
        stage_dict: Patient diagnostic stage dictionary.

    Returns:
        dict: {pid: {'a': nn.Parameter, 'b': nn.Parameter, 'stage': str}}
    """
    s_ranges = {
        'CN': (-10, 0),
        'LMCI': (-2, 8),
        'AD': (5, 20),
        'Other': (-10, 20),
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
    """Compute disease progression scores (s) for all patients using current DPS params.

    Args:
        csf_dict: Patient biomarker data dictionary.
        dps_params: DPS parameters per patient.

    Returns:
        tuple: (patient_data, s_population, y_population, all_stages)
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
            'b': b.item(),
        }

        all_s_points.append(s.detach().numpy())
        all_y_points.append(y)
        all_stages.extend([stage] * len(t))

    s_population = np.concatenate(all_s_points)
    y_population = np.concatenate(all_y_points)

    return patient_data, s_population, y_population, all_stages


# ==============================================================================
# 2. Population statistics
# ==============================================================================
def get_cn_average_y0(patient_data):
    """Compute the average baseline biomarker values for cognitively normal (CN) subjects.

    Args:
        patient_data: Dictionary of per-patient data with 'stage' and 'y' keys.

    Returns:
        np.ndarray: Average initial biomarker values [Aβ, Tau, N, C] (normalized).
    """
    cn_y0s = []
    for pid, data in patient_data.items():
        if data['stage'] == 'CN':
            cn_y0s.append(data['y'][0])

    if not cn_y0s:
        print("Warning: No CN patients found. Using default y0 = [0.1, 0, 0, 0].")
        return np.array([0.1, 0.0, 0.0, 0.0])

    cn_y0s_array = np.array(cn_y0s)
    avg_y0 = np.nanmean(cn_y0s_array, axis=0)
    avg_y0 = np.nan_to_num(avg_y0, nan=0.0)

    print(f"CN population average initial values (normalized): {avg_y0}")
    return avg_y0


def get_cn_average_s0(patient_data):
    """Compute the average initial disease progression score for CN subjects.

    Args:
        patient_data: Dictionary of per-patient data.

    Returns:
        float: Average initial s value for CN patients.
    """
    cn_s0s = []
    for pid, data in patient_data.items():
        if data['stage'] == 'CN':
            cn_s0s.append(data['s'][0])

    if not cn_s0s:
        print("Warning: No CN patients found. Using default s0 = 0.0.")
        return 0.0

    avg_s0 = float(np.nanmean(np.array(cn_s0s)))
    if np.isnan(avg_s0):
        print("Warning: All CN s0 values are NaN. Using default s0 = 0.0.")
        return 0.0

    print(f"CN population average s0: {avg_s0}")
    return avg_s0


def get_ad_average_y(patient_data):
    """Compute the average biomarker values across all visits for AD subjects.

    Args:
        patient_data: Dictionary of per-patient data.

    Returns:
        np.ndarray: Average biomarker values [Aβ, Tau, N, C] (normalized).
    """
    ad_ys = []
    for pid, data in patient_data.items():
        if data['stage'] == 'AD':
            ad_ys.append(data['y'])

    if not ad_ys:
        print("Warning: No AD patients found. Using default values [0, 0, 0, 0].")
        return np.array([0.0, 0.0, 0.0, 0.0])

    ad_ys_array = np.concatenate(ad_ys, axis=0)
    avg_y = np.nanmean(ad_ys_array, axis=0)
    avg_y = np.nan_to_num(avg_y, nan=0.0)

    print(f"AD population average values (normalized): {avg_y}")
    return avg_y


def get_ad_average_y_final(patient_data):
    """Compute the average *final visit* biomarker values for AD subjects.

    Args:
        patient_data: Dictionary of per-patient data.

    Returns:
        np.ndarray: Average final-visit biomarker values [Aβ, Tau, N, C] (normalized).
    """
    ad_y_final = []
    for _, data in patient_data.items():
        if data['stage'] == 'AD' and len(data['y']) > 0:
            ad_y_final.append(data['y'][-1])

    if not ad_y_final:
        print("Warning: No AD patients found. Using default values [0, 0, 0, 0].")
        return np.array([0.0, 0.0, 0.0, 0.0])

    ad_y_final = np.array(ad_y_final)
    avg_y_final = np.nanmean(ad_y_final, axis=0)
    avg_y_final = np.nan_to_num(avg_y_final, nan=0.0)

    print(f"AD population average final values (normalized): {avg_y_final}")
    return avg_y_final


# ==============================================================================
# 3. Sigmoid model and regularized fitting
# ==============================================================================
def sigmoid(s, a, b, c, d):
    """Generalized sigmoid (logistic) function.

    y(s) = a / (1 + exp(-b * (s - c))) + d

    Args:
        s: Input disease progression score(s).
        a: Amplitude (vertical scale).
        b: Slope / steepness (sign determines direction).
        c: Center / inflection point.
        d: Vertical offset (lower asymptote).

    Returns:
        Function value(s) at s.
    """
    exp_arg = np.clip(-b * (s - c), -50.0, 50.0)
    exp_term = np.exp(exp_arg)
    return a / (1.0 + exp_term) + d


def _sigmoid_regularized_residuals(params, s_valid, y_valid, reg_cfg):
    """Compute regularized residuals for scipy least_squares optimization.

    Combines data-fitting residuals with penalty terms that enforce:
      - Platform values near CN (upper) and AD (lower) population averages
      - Inflection point (c) in [0, 10]
      - Curvature (|b|) in [target_b_abs, target_b_max]
      - Amplitude (|a|) <= target_amp_max
      - Curve passes through two target anchor points
      - Turning points within desired range

    Args:
        params: [a, b, c, d] sigmoid parameters.
        s_valid: Valid s values (1D array).
        y_valid: Valid y values (1D array).
        reg_cfg: Regularization configuration dictionary.

    Returns:
        np.ndarray: Concatenated [data_residuals, regularization_terms].
    """
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

    # Anchor points: curve should pass through (s_target, y_target)
    s_target_1 = reg_cfg.get('s_target_1', reg_cfg.get('s_target', 0.0))
    y_target_1 = reg_cfg.get('y_target_1', reg_cfg.get('y_target', 0.0))
    s_target_2 = reg_cfg.get('s_target_2', 30.0)
    y_target_2 = reg_cfg.get('y_target_2', y_target_1)
    pass_pen_1 = (sigmoid(s_target_1, a, b, c, d) - y_target_1) * np.sqrt(w_cn)
    pass_pen_2 = (sigmoid(s_target_2, a, b, c, d) - y_target_2) * np.sqrt(w_cn)

    # Platform penalties: upper ~ CN average, lower ~ AD average
    y_cn = reg_cfg.get('y_cn', 0.0)
    y_ad = reg_cfg.get('y_ad', 0.0)
    upper = np.maximum(d, d + a)
    lower = np.minimum(d, d + a)
    use_plat_upper = reg_cfg.get('use_plat_upper', True)
    plat_pen_upper = (upper - y_cn) * np.sqrt(w_plat) if use_plat_upper else 0.0
    plat_pen_lower = (lower - y_ad) * np.sqrt(w_plat)

    # Center in [0, 10]
    center_low = max(0.0, -c) * np.sqrt(w_center)
    center_high = max(0.0, c - 10.0) * np.sqrt(w_center)

    # Curvature bounds: target_b_abs <= |b| <= target_b_max
    curv_pen_low = max(0.0, target_b_abs - np.abs(b)) * np.sqrt(w_curv)
    curv_pen_high = max(0.0, np.abs(b) - target_b_max) * np.sqrt(w_curv)

    # Amplitude cap
    amp_pen = max(0.0, np.abs(a) - target_amp_max) * np.sqrt(w_curv)

    # Turning point constraints
    b_safe = b if np.abs(b) > 1e-6 else 1e-6
    turn_left = c - (np.log(2.0) / b_safe)
    turn_right = c + (np.log(2.0) / b_safe)
    use_turn_right = reg_cfg.get('use_turn_right', True)
    turn_pen_left = (turn_left - 0.0) * np.sqrt(w_turn)
    turn_pen_right = (turn_right - 10.0) * np.sqrt(w_turn) if use_turn_right else 0.0

    reg_terms = np.array([
        turn_pen_left, turn_pen_right, center_low, center_high,
        curv_pen_low, curv_pen_high, amp_pen,
        pass_pen_1, pass_pen_2, plat_pen_upper, plat_pen_lower,
    ])

    return np.concatenate([residuals, reg_terms])


def _sigmoid_fit_loss(params, s_valid, y_valid, reg_cfg):
    """Compute scalar loss for sigmoid fitting (MSE of residuals + regularization).

    Args:
        params: [a, b, c, d].
        s_valid, y_valid: Valid data points.
        reg_cfg: Regularization config.

    Returns:
        float: Scalar loss value.
    """
    a, b, c, d = params
    y_pred = sigmoid(s_valid, a, b, c, d)
    residuals = y_pred - y_valid

    w_turn = reg_cfg['w_turn']
    w_center = reg_cfg['w_center']
    w_curv = reg_cfg['w_curv']
    w_cn = reg_cfg['w_cn']
    w_plat = reg_cfg['w_plat']
    target_b_abs = reg_cfg['target_b_abs']
    target_b_max = reg_cfg.get('target_b_max', 0.9)

    s_target_1 = reg_cfg.get('s_target_1', reg_cfg.get('s_target', 0.0))
    y_target_1 = reg_cfg.get('y_target_1', reg_cfg.get('y_target', 0.0))
    s_target_2 = reg_cfg.get('s_target_2', 20.0)
    y_target_2 = reg_cfg.get('y_target_2', y_target_1)
    pass_pen_1 = (sigmoid(s_target_1, a, b, c, d) - y_target_1) * np.sqrt(w_cn)
    pass_pen_2 = (sigmoid(s_target_2, a, b, c, d) - y_target_2) * np.sqrt(w_cn)

    y_cn = reg_cfg.get('y_cn', 0.0)
    y_ad = reg_cfg.get('y_ad', 0.0)
    upper = np.maximum(d, d + a)
    lower = np.minimum(d, d + a)
    use_plat_upper = reg_cfg.get('use_plat_upper', True)
    plat_pen_upper = (upper - y_cn) * np.sqrt(w_plat) if use_plat_upper else 0.0
    plat_pen_lower = (lower - y_ad) * np.sqrt(w_plat)

    center_low = max(0.0, -c) * np.sqrt(w_center)
    center_high = max(0.0, c - 10.0) * np.sqrt(w_center)

    curv_pen_low = max(0.0, target_b_abs - np.abs(b)) * np.sqrt(w_curv)
    curv_pen_high = max(0.0, np.abs(b) - target_b_max) * np.sqrt(w_curv)

    b_safe = b if np.abs(b) > 1e-6 else 1e-6
    turn_left = c - (np.log(2.0) / b_safe)
    turn_right = c + (np.log(2.0) / b_safe)
    use_turn_right = reg_cfg.get('use_turn_right', True)
    turn_pen_left = (turn_left - 0.0) * np.sqrt(w_turn)
    turn_pen_right = (turn_right - 10.0) * np.sqrt(w_turn) if use_turn_right else 0.0

    reg_terms = np.array([
        turn_pen_left, turn_pen_right, center_low, center_high,
        curv_pen_low, curv_pen_high, pass_pen_1, pass_pen_2,
        plat_pen_upper, plat_pen_lower,
    ])
    all_res = np.concatenate([residuals, reg_terms])
    return float(np.mean(all_res ** 2))


def fit_sigmoids_regularized(s_data, y_data, reg_cfg=None):
    """Fit regularized sigmoid curves to each of the 4 biomarkers.

    Biomarker-specific adjustments:
      - k=1 (Tau): fixed upper/lower platforms (a+d=130, d=70 in original space),
        stronger curvature constraints.
      - k=3 (Cognition): no upper platform or right turning point constraints.

    Args:
        s_data: Disease progression scores (N,).
        y_data: Biomarker values (N, 4), normalized.
        reg_cfg: Regularization configuration dictionary.

    Returns:
        tuple: (sigmoid_params (4, 4), total_loss (float))
    """
    def _tau_bc_residuals(x_bc, s_valid, y_valid, reg_cfg_k, fixed_a, fixed_d):
        """Residuals for Tau with fixed amplitude and offset."""
        b, c = x_bc
        params = np.array([fixed_a, b, c, fixed_d], dtype=np.float64)
        return _sigmoid_regularized_residuals(params, s_valid, y_valid, reg_cfg_k)

    if reg_cfg is None:
        reg_cfg = {
            'w_turn': 5.0, 'w_center': 2.0, 'w_curv': 5.0,
            'w_cn': 8.0, 'w_plat': 6.0,
            'target_b_abs': 0.9, 'target_b_max': 1.2,
            'target_amp_max': 4.0,
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

        y_target_1 = reg_cfg.get('y_target_1', reg_cfg.get('y_target', 0.0))
        reg_cfg_k['y_target_1'] = (y_target_1[k]
                                   if isinstance(y_target_1, (list, np.ndarray))
                                   else y_target_1)
        reg_cfg_k['s_target_1'] = reg_cfg.get('s_target_1', reg_cfg.get('s_target', -20.0))

        y_target_2 = reg_cfg.get('y_target_2', y_ad)
        reg_cfg_k['y_target_2'] = (y_target_2[k]
                                   if isinstance(y_target_2, (list, np.ndarray))
                                   else y_target_2)
        reg_cfg_k['s_target_2'] = reg_cfg.get('s_target_2', 20.0)

        # Tau (k=1): fix platform, stronger curvature
        if k == 1:
            reg_cfg_k['target_b_abs'] = 2.5
            reg_cfg_k['target_b_max'] = 3.0
            reg_cfg_k['w_curv'] = 100

        # Cognition (k=3): no upper platform or right-turning-point constraints
        if k == 3:
            reg_cfg_k['use_plat_upper'] = False
            reg_cfg_k['use_turn_right'] = False

        if len(y_k_valid) < 5:
            default_params = np.array([1.0, 1.0, 5.0, 0.0])
            sigmoid_params.append(default_params)
            total_loss += _sigmoid_fit_loss(default_params, s_k_valid, y_k_valid, reg_cfg_k)
            continue

        amp_init = np.max(y_k_valid) - np.min(y_k_valid)
        center_init = np.median(s_k_valid)
        slope_sign = -1.0 if np.corrcoef(s_k_valid, y_k_valid)[0, 1] < 0 else 1.0

        if k == 1 and reg_cfg.get('tau_fix_platform', False):
            tau_upper = float(reg_cfg.get('tau_upper_norm', 1.0))
            tau_lower = float(reg_cfg.get('tau_lower_norm', 0.0))
            fixed_d = tau_lower
            fixed_a = tau_upper - tau_lower

            x0_bc = [1.0 * slope_sign, center_init]
            result = least_squares(
                _tau_bc_residuals,
                x0=x0_bc,
                args=(s_k_valid, y_k_valid, reg_cfg_k, fixed_a, fixed_d),
                max_nfev=10000,
            )
            b_opt, c_opt = result.x
            params_opt = np.array([fixed_a, b_opt, c_opt, fixed_d], dtype=np.float64)
        else:
            p0 = [amp_init, 1.0 * slope_sign, center_init, np.min(y_k_valid)]
            result = least_squares(
                _sigmoid_regularized_residuals,
                x0=p0,
                args=(s_k_valid, y_k_valid, reg_cfg_k),
                max_nfev=10000,
            )
            params_opt = result.x

        sigmoid_params.append(params_opt)
        total_loss += _sigmoid_fit_loss(params_opt, s_k_valid, y_k_valid, reg_cfg_k)

    return np.array(sigmoid_params), float(total_loss)


# ==============================================================================
# 4. Sigmoid-only training pipeline
# ==============================================================================
def train_sigmoid_only(csf_dict, stage_dict):
    """Train sigmoid curves without fitting DPS parameters.

    Uses initial DPS parameter assignment and fits regularized sigmoid curves
    to all four biomarkers. Tau is constrained with fixed platforms
    (original space: d=70, a+d=130).

    Args:
        csf_dict: Patient biomarker data.
        stage_dict: Patient diagnostic stages.

    Returns:
        tuple: (dps_params, sigmoid_params)
    """
    print("\nStarting sigmoid-only training (no DPS optimization)...")
    dps_params = assign_dps_params(csf_dict, stage_dict)

    patient_data, s_pop, y_pop_norm, _ = compute_s_values(csf_dict, dps_params)
    y0_cn = get_cn_average_y0(patient_data)
    y_ad = get_ad_average_y(patient_data)
    y_ad_final = get_ad_average_y_final(patient_data)

    # Tau fixed-platform constraint: original-space targets
    mean_std = np.load('../data/mean_std.npy')
    tau_mean = float(mean_std[0, 1])
    tau_std = float(mean_std[1, 1])
    tau_lower_norm = (70.0 - tau_mean) / tau_std
    tau_upper_norm = (130.0 - tau_mean) / tau_std

    y_target_1 = np.array(y0_cn, copy=True)
    y_target_2 = np.array(y_ad_final, copy=True)
    y_target_1[1] = tau_lower_norm
    y_target_2[1] = tau_upper_norm

    print("Tau fixed platforms (original space): d=70, a+d=130")
    print(f"Tau fixed platforms (normalized): d={tau_lower_norm:.6f}, a+d={tau_upper_norm:.6f}")

    sigmoid_params, sigmoid_loss = fit_sigmoids_regularized(
        s_pop, y_pop_norm,
        reg_cfg={
            'w_turn': 5.0, 'w_center': 2.0, 'w_curv': 3.0,
            'w_cn': 10.0, 'w_plat': 8.0,
            'target_b_abs': 0.6, 'target_b_max': 0.9,
            'target_amp_max': 2.0,
            's_target_1': -20.0, 'y_target_1': y_target_1,
            's_target_2': 30.0, 'y_target_2': y_target_2,
            'tau_fix_platform': True,
            'tau_lower_norm': tau_lower_norm,
            'tau_upper_norm': tau_upper_norm,
            'y_cn': y0_cn, 'y_ad': y_ad,
        },
    )

    torch.save(sigmoid_params, '../models/sigmoid.pth')
    print(f"Sigmoid training complete. Loss: {sigmoid_loss:.6f}")

    return dps_params, sigmoid_params


# ==============================================================================
# 5. Visualization
# ==============================================================================
def get_sigmoid_derivatives(s_grid, params):
    """Evaluate sigmoid curves and their derivatives on a grid.

    Args:
        s_grid: 1D array of s values.
        params: (4, 4) array of sigmoid parameters per biomarker.

    Returns:
        tuple: (y_on_grid (N, 4), dyds_on_grid (N, 4))
    """
    y_on_grid = np.zeros((len(s_grid), 4))
    dyds_on_grid = np.zeros((len(s_grid), 4))

    for k in range(4):
        a, b, c, d = params[k]
        exp_arg = np.clip(-b * (s_grid - c), -50.0, 50.0)
        exp_term = np.exp(exp_arg)
        denom = (1.0 + exp_term) ** 2
        y_on_grid[:, k] = a / (1.0 + exp_term) + d
        dyds_on_grid[:, k] = np.divide(
            a * b * exp_term, denom,
            out=np.zeros_like(exp_term), where=denom != 0,
        )

    return y_on_grid, dyds_on_grid


def plot_results(s_pop, y_pop, stages_pop, s_grid, sigmoid_params):
    """Plot sigmoid-fitted curves against patient data for all 4 biomarkers.

    Args:
        s_pop: Population s values.
        y_pop: Population y values (normalized).
        stages_pop: Diagnostic stages for each data point.
        s_grid: s-axis grid for plotting curves.
        sigmoid_params: Fitted sigmoid parameters.
    """
    print("Generating final result figure...")

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
            ax.scatter(s_vals, y_vals[:, k], s=15, alpha=0.5,
                       c=colors[stage], label=stage)

        ax.plot(s_grid, y_sigmoid_grid_orig[:, k], 'r-', lw=2.5,
                label='Sigmoid Fit', zorder=3)

        ax.set_xlabel('Disease Progression Score (s)')
        ax.set_ylabel(TITLES[k])
        ax.set_xlim(s_grid.min(), s_grid.max())
        ax.legend()
        ax.grid(True, alpha=0.4)
        ax.set_title(f'Trajectory for {TITLES[k]}')

    fig.suptitle('AD Biomarker Trajectories: Sigmoid Fit', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('../figures/sigmoid.png')
    plt.show()


# ==============================================================================
# Main
# ==============================================================================
if __name__ == '__main__':
    print("Starting pretraining pipeline...")

    # 1. Train sigmoid curves (no DPS optimization)
    dps_params, _ = train_sigmoid_only(csf_dict, stage_dict)

    # Save initial DPS parameters
    dps_save = {
        pid: {'a': params['a'].item(), 'b': params['b'].item()}
        for pid, params in dps_params.items()
    }
    torch.save(dps_save, '../models/dps.pth')
    print("DPS parameters saved to dps.pth")

    # 2. Load optimal sigmoid parameters
    print("Loading optimal sigmoid parameters...")
    sigmoid_params = torch.load('../models/sigmoid.pth', weights_only=False)

    # 3. Compute final s values and patient data
    print("Computing final s values...")
    patient_data, s_pop, y_pop_norm, stages_pop = compute_s_values(
        csf_dict, dps_params,
    )

    # 4. Compute CN population average initial values
    print("Computing CN population average initial values...")
    _ = get_cn_average_y0(patient_data)

    # 5. Plot results
    print("Generating result figure...")
    s_min, s_max = s_pop.min(), s_pop.max()
    s_margin = (s_max - s_min) * 0.1
    s_grid = np.linspace(s_min - s_margin, s_max + s_margin, 300)

    plot_results(s_pop, y_pop_norm, stages_pop, s_grid, sigmoid_params)
