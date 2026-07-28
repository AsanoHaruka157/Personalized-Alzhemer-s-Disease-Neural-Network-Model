"""
FNN-based Neural ODE training pipeline for AD biomarker trajectory modeling.

This script implements a two-stage training approach:
  1. Pretrain a Feedforward Neural Network (FNN) to match sigmoid-fitted
     biomarker trajectories (trajectory-matching objective).
  2. Fine-tune the FNN and per-patient DPS (Disease Progression Score)
     parameters on real patient data using alternating L-BFGS optimization,
     with biological constraints (monotonicity, plateau).

Note: This pipeline and main.py are currently independent experimental routes
with different model architectures and training strategies. They will be
unified in a future refactoring.

Outputs:
  - fnn.pth: trained FNN model weights
  - pretrain_result.png: pretraining visualization
  - fnn.png: final model trajectory visualization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, least_squares

try:
    from torchdiffeq import odeint as torch_odeint
except ImportError:
    raise ImportError(
        "torchdiffeq is required. Install with: pip install torchdiffeq"
    )

import pccmnn as pc


# ==============================================================================
# 0. Data loading and preparation
# ==============================================================================
try:
    csf_dict = pc.load_data()
    stage_dict = pc.load_stage_dict()
    print(f"Successfully loaded data for {len(csf_dict)} patients.")
except Exception as e:
    print(f"Error: Unable to load data. Ensure pccmnn.py and data files exist. "
          f"Details: {e}")
    exit()

patient_data = {}
for pid, sample in csf_dict.items():
    t = torch.from_numpy(sample[:, 0]).float()
    y = torch.from_numpy(sample[:, 1:5]).float()
    patient_data[pid] = {
        "t": t, "y": y, "y0": y[0].clone(),
        "stage": stage_dict.get(pid, 'Other'),
    }


# ==============================================================================
# 1. Load DPS parameters and create scatter data
# ==============================================================================
def load_dps_and_create_scatter(csf_dict, dps_path='../models/dps.pth'):
    """Load DPS parameters and construct (s, y) scatter points for all patients.

    Args:
        csf_dict: Patient biomarker data.
        dps_path: Path to saved DPS parameter file.

    Returns:
        tuple: (s_all, y_all, all_stages) — concatenated s, y, and stage labels.
    """
    print(f"Loading DPS parameters from {dps_path}...")
    try:
        dps_params = torch.load(dps_path, weights_only=False)
    except FileNotFoundError:
        print(f"Error: {dps_path} not found. Run sigmoid.py first to generate it.")
        exit()

    all_s, all_y, all_stages = [], [], []
    for pid, sample in csf_dict.items():
        if pid in dps_params:
            stage = stage_dict.get(pid, 'Other')
            t, y = sample[:, 0], sample[:, 1:5]

            a = dps_params[pid]['a']
            b = dps_params[pid]['b']
            s = a * t + b

            all_s.append(s)
            all_y.append(y)
            all_stages.extend([stage] * len(t))

    s_all = np.concatenate(all_s)
    y_all = np.concatenate(all_y)

    # Filter NaN rows
    valid_mask = ~np.isnan(y_all).any(axis=1)
    s_all, y_all = s_all[valid_mask], y_all[valid_mask]
    all_stages = [s for s, m in zip(all_stages, valid_mask) if m]
    print(f"Scatter data generated from DPS params (after NaN filtering: "
          f"{len(s_all)} points).")
    return s_all, y_all, all_stages


# ==============================================================================
# 2. Sigmoid fitting (regularized, shared with sigmoid.py)
# ==============================================================================
def sigmoid(s, a, b, c, d):
    """Generalized sigmoid (logistic) function: y = a/(1+exp(-b*(s-c))) + d."""
    exp_arg = np.clip(-b * (s - c), -50.0, 50.0)
    exp_term = np.exp(exp_arg)
    return a / (1.0 + exp_term) + d


def _sigmoid_regularized_residuals(params, s_valid, y_valid, reg_cfg):
    """Compute regularized residuals for scipy least_squares sigmoid fitting.

    See sigmoid.py for full documentation of regularization terms.
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

    s_target_1 = reg_cfg.get('s_target_1', reg_cfg.get('s_target', 0.0))
    y_target_1 = reg_cfg.get('y_target_1', reg_cfg.get('y_target', 0.0))
    s_target_2 = reg_cfg.get('s_target_2', 30.0)
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

    amp_pen = max(0.0, np.abs(a) - target_amp_max) * np.sqrt(w_curv)

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


def fit_sigmoids(s_data, y_data, reg_cfg=None):
    """Fit regularized sigmoid curves to each of the 4 biomarkers.

    Args:
        s_data: s values (N,).
        y_data: Biomarker values (N, 4), normalized.
        reg_cfg: Regularization configuration.

    Returns:
        np.ndarray: Fitted parameters (4, 4).
    """
    if reg_cfg is None:
        reg_cfg = {
            'w_turn': 5.0, 'w_center': 2.0, 'w_curv': 5.0,
            'w_cn': 8.0, 'w_plat': 6.0,
            'target_b_abs': 0.6, 'target_b_max': 1.2,
            'target_amp_max': 3.0,
        }

    sigmoid_params = []
    for k in range(4):
        y_k = y_data[:, k]
        valid_mask = ~np.isnan(y_k)
        s_k_valid = s_data[valid_mask]
        y_k_valid = y_k[valid_mask]

        reg_cfg_k = dict(reg_cfg)
        for key in ['y_cn', 'y_ad', 'y_target_1', 'y_target_2']:
            val = reg_cfg.get(key, 0.0)
            reg_cfg_k[key] = val[k] if isinstance(val, (list, np.ndarray)) else val

        # Tau (k=1): stronger curvature constraint
        if k == 1:
            reg_cfg_k['target_b_abs'] = 2.5
            reg_cfg_k['target_b_max'] = 3.0
            reg_cfg_k['w_curv'] = 100

        # Cognition (k=3): no upper platform or right turning point constraint
        if k == 3:
            reg_cfg_k['use_plat_upper'] = False
            reg_cfg_k['use_turn_right'] = False

        if len(y_k_valid) < 5:
            sigmoid_params.append(np.array([1.0, 1.0, 5.0, 0.0]))
            continue

        amp_init = y_k_valid.max() - y_k_valid.min()
        center_init = np.median(s_k_valid)
        slope_sign = -1.0 if np.corrcoef(s_k_valid, y_k_valid)[0, 1] < 0 else 1.0
        p0 = [amp_init, 1.0 * slope_sign, center_init, y_k_valid.min()]

        try:
            result = least_squares(
                _sigmoid_regularized_residuals,
                x0=p0, args=(s_k_valid, y_k_valid, reg_cfg_k),
                max_nfev=10000,
            )
            sigmoid_params.append(result.x)
        except Exception:
            # Fallback: simple curve_fit
            try:
                popt, _ = curve_fit(
                    sigmoid, s_k_valid, y_k_valid, p0=p0, maxfev=10000,
                )
                sigmoid_params.append(popt)
            except Exception:
                sigmoid_params.append(np.array([1.0, 1.0, 5.0, 0.0]))

    return np.array(sigmoid_params)


def get_sigmoid_derivatives(s_grid, params):
    """Evaluate sigmoid curves and their derivatives on a grid.

    Args:
        s_grid: 1D array of s values.
        params: (4, 4) sigmoid parameters.

    Returns:
        tuple: (y (N, 4), dyds (N, 4))
    """
    y = np.zeros((len(s_grid), 4))
    dyds = np.zeros((len(s_grid), 4))
    for k in range(4):
        a, b, c, d = params[k]
        exp_term = np.exp(-b * (s_grid - c))
        y[:, k] = a / (1.0 + exp_term) + d
        dyds[:, k] = (a * b * exp_term) / ((1.0 + exp_term) ** 2)
    return y, dyds


# ==============================================================================
# 3. FNN model definition and sigmoid-trajectory pretraining
# ==============================================================================
class FNN(nn.Module):
    """Feedforward Neural Network mapping y (4 biomarkers) → dy/ds (4 derivatives).

    Architecture: 4 → 128 → 128 → 128 → 4 (3 hidden layers, ReLU activations).
    Note: This architecture differs from main.py's FNN (which uses 2 hidden
    layers + Tanh output). These will be unified in a future refactoring.
    """

    def __init__(self, input_dim=4, hidden_dim=128, output_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, y):
        return self.net(y)


def pretrain_fnn_on_sigmoid(sigmoid_params, y0, s_grid=None, n_epochs=1000, lr=1e-3):
    """Pretrain FNN so that ODE-integrated trajectory matches sigmoid target.

    Args:
        sigmoid_params: (4, 4) sigmoid parameters.
        y0: Initial biomarker values (4,) — CN population average.
        s_grid: s-axis grid for trajectory matching. Defaults to [-10, 20] with 300 points.
        n_epochs: Number of training epochs.
        lr: Learning rate for Adam optimizer.

    Returns:
        FNN: Pretrained FNN model (in eval mode).
    """
    print("\n--- Stage 1: Pretraining FNN on sigmoid trajectory (Adam, trajectory matching) ---")

    if s_grid is None:
        s_grid = np.linspace(-10, 20, 300)

    # Compute sigmoid target trajectory (normalized space)
    y_target_norm, _ = get_sigmoid_derivatives(s_grid, sigmoid_params)
    y_target = torch.tensor(y_target_norm, dtype=torch.float32)  # (N, 4)
    s_tensor = torch.tensor(s_grid, dtype=torch.float32)

    model = FNN()
    ode_model = ODEModel(model)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        y_pred = torch_odeint(
            ode_model, y0, s_tensor,
            method='rk4', options={'step_size': 0.5},
        )
        loss = F.mse_loss(y_pred, y_target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f"  Traj pretrain epoch {epoch + 1}, loss={loss.item():.6f}")

    model.eval()
    return model


# ==============================================================================
# 4. ODE model wrapper and biological constraints
# ==============================================================================
class ODEModel(nn.Module):
    """ODE function wrapper for torchdiffeq: func(t, y) → dy/dt.

    Note: This is a simple wrapper without input normalization. The ODEModel in
    main.py includes [0,1] normalization via y_min/y_max. These will be unified
    in a future refactoring.
    """

    def __init__(self, fnn_model):
        super().__init__()
        self.fnn = fnn_model

    def forward(self, t, y):
        return self.fnn(y)


def monotonicity_penalty(y_traj):
    """Penalize violations of expected biomarker monotonicity along s.

    Constraints: dAβ/ds ≤ 0, dN/ds ≤ 0, dτ/ds ≥ 0, dC/ds ≥ 0.

    Args:
        y_traj: Trajectory tensor of shape (T, 4).

    Returns:
        Scalar penalty tensor.
    """
    diff = torch.diff(y_traj, dim=0)  # (T-1, 4)
    pen_A = F.relu(diff[:, 0]).mean()    # Penalize Aβ increase
    pen_T = F.relu(-diff[:, 1]).mean()   # Penalize τ decrease
    pen_N = F.relu(diff[:, 2]).mean()    # Penalize N increase
    pen_C = F.relu(-diff[:, 3]).mean()   # Penalize C decrease
    return pen_A + pen_T + pen_N + pen_C


def plateau_penalty(ode_model, y0, s_early_end=-6.0, s_late_start=18.0):
    """Penalize non-zero derivatives in CN (early) and late-AD (late) plateaus.

    Args:
        ode_model: ODE model.
        y0: Initial condition (4,).
        s_early_end: End of the early (CN) plateau.
        s_late_start: Start of the late (AD) plateau.

    Returns:
        Scalar penalty tensor.
    """
    s_early = torch.linspace(-10, s_early_end, 30)
    s_late = torch.linspace(s_late_start, 22, 30)

    with torch.enable_grad():
        # Early plateau derivatives
        y_e = torch_odeint(
            ode_model, y0, s_early,
            method='rk4', options={'step_size': 0.5},
        )
        dy_early = torch.stack(
            [ode_model(0, y_e[i]) for i in range(len(s_early))]
        )

        # Late plateau derivatives
        y_l = torch_odeint(
            ode_model, y0, torch.cat([s_early[-1:], s_late]),
            method='rk4', options={'step_size': 0.5},
        )[1:]
        dy_late = torch.stack(
            [ode_model(0, y_l[i]) for i in range(len(s_late))]
        )

    return (dy_early ** 2).mean() + (dy_late ** 2).mean()


# ==============================================================================
# 5. Loss functions
# ==============================================================================
def calculate_loss(ode_model, patient_data, ab, pids, y0,
                   w_mono=0.01, w_plateau=0.001, debug=False):
    """Compute total loss = data MSE + monotonicity + plateau penalties.

    Uses sorted, deduplicated s-values for efficient ODE integration.

    Args:
        ode_model: ODE model.
        patient_data: Per-patient data dict.
        ab: Per-patient DPS parameters {pid: {'a': ..., 'b': ...}}.
        pids: List of patient IDs to include.
        y0: Initial condition (4,).
        w_mono: Monotonicity penalty weight.
        w_plateau: Plateau penalty weight.
        debug: If True, print warnings.

    Returns:
        tuple: (total_loss, data_loss, mono_loss, plateau_loss)
    """
    all_s, all_y = [], []
    for pid in pids:
        dat = patient_data[pid]
        s_values = ab[pid]['a'] * dat['t'] + ab[pid]['b']
        all_s.append(s_values)
        all_y.append(dat['y'])

    s_global, y_global = torch.cat(all_s), torch.cat(all_y)

    # Filter NaN data points
    valid = ~torch.isnan(y_global).any(dim=1)
    if not valid.any():
        if debug:
            print("  [WARNING] All data points are NaN!")
        inf = torch.tensor(float('inf'))
        return inf, inf, inf, inf

    s_global, y_global = s_global[valid], y_global[valid]
    s_sorted, sort_indices = torch.sort(s_global)
    y_sorted = y_global[sort_indices]

    # Try multiple ODE solvers with fallback
    for method, opts in [
        ('dopri5', {'rtol': 1e-4, 'atol': 1e-5}),
        ('rk4', {'step_size': 0.5}),
        ('euler', {'step_size': 0.5}),
    ]:
        try:
            y_pred = torch_odeint(ode_model, y0, s_sorted, method=method, **opts)
            if torch.isnan(y_pred).any() or torch.isinf(y_pred).any():
                continue
            data_loss = ((y_pred - y_sorted) ** 2).mean()
            if not torch.isfinite(data_loss):
                continue

            mono_loss = monotonicity_penalty(y_pred)
            plat_loss = plateau_penalty(ode_model, y0)

            total_loss = data_loss + w_mono * mono_loss + w_plateau * plat_loss
            if torch.isfinite(total_loss):
                return total_loss, data_loss, mono_loss, plat_loss
        except Exception:
            continue

    if debug:
        print(f"  [WARNING] All ODE solvers failed! y0={y0}")
    inf = torch.tensor(float('inf'))
    return inf, inf, inf, inf


# ==============================================================================
# 6. Alternating FNN + DPS optimization (Algorithm 1)
# ==============================================================================
def train_fnn_on_data(initial_fnn, patient_data, y0, dps_path='../models/dps.pth',
                      n_outer=10, n_inner=5, lr_fnn=1e-6, lr_dps=1e-2):
    """Alternating optimization: outer loop alternates between FNN (population)
    and DPS (individual) parameter updates.

    This implements Algorithm 1 from the manuscript:
      - Step A: Fix DPS, optimize FNN (population parameter calibration)
      - Step B: Fix FNN, optimize per-patient DPS (individual parameter update)

    Args:
        initial_fnn: Pretrained FNN model.
        patient_data: Per-patient data.
        y0: CN average initial condition (4,).
        dps_path: Path to DPS parameter file.
        n_outer: Number of outer (alternating) iterations.
        n_inner: Number of L-BFGS iterations per step.
        lr_fnn: Learning rate for FNN optimization.
        lr_dps: Learning rate for DPS optimization.

    Returns:
        tuple: (trained_ode_model, ab_params)
    """
    print("\n--- Alternating FNN (population) and DPS (individual) optimization "
          "(Algorithm 1) ---")
    ode_model = ODEModel(initial_fnn).train()

    dps_params_loaded = torch.load(dps_path, weights_only=False)
    ab = {}
    for pid, data in patient_data.items():
        if pid in dps_params_loaded:
            ab[pid] = {
                'a': nn.Parameter(torch.tensor(
                    dps_params_loaded[pid]['a'], dtype=torch.float32)),
                'b': nn.Parameter(torch.tensor(
                    dps_params_loaded[pid]['b'], dtype=torch.float32)),
            }
    patient_pids = list(ab.keys())
    print(f"  y0 (CN mean): {y0}, patients: {len(patient_pids)}")
    dps_params = [p for pid in patient_pids for p in ab[pid].values()]

    for outer in range(n_outer):
        # --- Step A: Fix DPS, optimize FNN ---
        for p in dps_params:
            p.requires_grad = False
        for p in ode_model.parameters():
            p.requires_grad = True

        comp_fnn = {}
        opt_fnn = optim.LBFGS(
            ode_model.parameters(), lr=lr_fnn,
            max_iter=n_inner, line_search_fn="strong_wolfe",
        )

        def closure_fnn():
            opt_fnn.zero_grad()
            total, data, mono, plat = calculate_loss(
                ode_model, patient_data, ab, patient_pids, y0,
                debug=(outer == 0),
            )
            comp_fnn['data'], comp_fnn['mono'], comp_fnn['plat'] = data, mono, plat
            if torch.isfinite(total):
                total.backward()
            return total

        loss_fnn = opt_fnn.step(closure_fnn)

        # --- Step B: Fix FNN, optimize DPS ---
        for p in ode_model.parameters():
            p.requires_grad = False
        for p in dps_params:
            p.requires_grad = True

        comp_dps = {}
        opt_dps = optim.LBFGS(
            dps_params, lr=lr_dps,
            max_iter=n_inner, line_search_fn="strong_wolfe",
        )

        def closure_dps():
            opt_dps.zero_grad()
            total, data, mono, plat = calculate_loss(
                ode_model, patient_data, ab, patient_pids, y0,
            )
            comp_dps['data'], comp_dps['mono'], comp_dps['plat'] = data, mono, plat
            if torch.isfinite(total):
                total.backward()
            return total

        loss_dps = opt_dps.step(closure_dps)

        print(f"  Outer [{outer + 1}/{n_outer}]")
        print(f"    [FNN] total={loss_fnn.item():.6f}, "
              f"data={comp_fnn['data'].item():.6f}, "
              f"mono={comp_fnn['mono'].item():.6f}, "
              f"plat={comp_fnn['plat'].item():.6f}")
        print(f"    [DPS] total={loss_dps.item():.6f}, "
              f"data={comp_dps['data'].item():.6f}, "
              f"mono={comp_dps['mono'].item():.6f}, "
              f"plat={comp_dps['plat'].item():.6f}")

    # Restore requires_grad for all parameters
    for p in dps_params:
        p.requires_grad = True
    for p in ode_model.parameters():
        p.requires_grad = True

    ode_model.eval()
    return ode_model, ab


# ==============================================================================
# 7. Visualization and saving
# ==============================================================================
def plot_pretrain_results(s_pop, y_pop_orig, stages_pop, s_grid,
                          sigmoid_params, pretrained_fnn, y0):
    """Plot pretraining results: sigmoid fit vs. pretrained FNN trajectory.

    Args:
        s_pop, y_pop_orig: Scatter data (original units).
        stages_pop: Stage labels.
        s_grid: s-axis grid.
        sigmoid_params: Fitted sigmoid parameters.
        pretrained_fnn: Pretrained FNN model.
        y0: Initial condition.
    """
    print("\nGenerating pretraining result figure...")

    y_sigmoid_grid_norm, _ = get_sigmoid_derivatives(s_grid, sigmoid_params)
    y_sigmoid_grid_orig = pc.inv_nor(y_sigmoid_grid_norm)

    ode_model = ODEModel(pretrained_fnn).eval()
    with torch.no_grad():
        y_fnn_traj_norm = torch_odeint(
            ode_model, y0, torch.from_numpy(s_grid).float(),
        ).numpy()
    y_fnn_traj_orig = pc.inv_nor(y_fnn_traj_norm)

    TITLES = ['Aβ (A)', 'p-Tau (T)', 'N', 'Cognition (C)']
    colors = {'CN': 'orange', 'LMCI': 'green', 'AD': 'blue', 'Other': 'grey'}
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    axes = axes.flat

    for k in range(4):
        ax = axes[k]
        for stage in np.unique(stages_pop):
            mask = np.array(stages_pop) == stage
            ax.scatter(s_pop[mask], y_pop_orig[mask, k], s=15, alpha=0.4,
                       c=colors[stage], label=f'{stage}')

        ax.plot(s_grid, y_sigmoid_grid_orig[:, k], 'r--', lw=2.5,
                label='Sigmoid Fit', zorder=3)
        ax.plot(s_grid, y_fnn_traj_orig[:, k], 'k-', lw=2.5,
                label='Pre-trained FNN Trajectory', zorder=4)

        ax.set_xlabel('Disease Progression Score (s)')
        ax.set_ylabel(TITLES[k])
        ax.legend()
        ax.grid(True, alpha=0.4)
        ax.set_title(TITLES[k])

    fig.suptitle('FNN Pre-training Result vs. Sigmoid Fit', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('../figures/pretrain_result.png', dpi=150)
    print("Pretraining result saved to pretrain_result.png")
    plt.show()


def plot_and_save(s_pop, y_pop_orig, stages_pop, s_grid, final_model, y0,
                  sigmoid_params, model_save_path='fnn.pth'):
    """Plot final model trajectory with uncertainty quantification and save model.

    Uncertainty is estimated by sampling NN weights from N(μ, σ²) with σ=1e-2.

    Args:
        s_pop, y_pop_orig: Scatter data (original units).
        stages_pop: Stage labels.
        s_grid: s-axis grid.
        final_model: Trained ODE model.
        y0: Initial condition.
        sigmoid_params: Sigmoid parameters for reference curve.
        model_save_path: Path to save trained model weights.
    """
    print("\nGenerating final comparison figure and saving model...")

    torch.save(final_model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    # Uncertainty quantification via weight perturbation
    n_samples = 100
    nn_sigma = 1e-2
    pred_trajectories = []
    s_grid_tensor = torch.from_numpy(s_grid).float()

    print(f"Sampling {n_samples} trajectories from N(μ, {nn_sigma**2}) "
          f"for uncertainty quantification...")
    for i in range(n_samples):
        temp_fnn = FNN()
        temp_fnn.load_state_dict(final_model.fnn.state_dict())
        temp_ode_model = ODEModel(temp_fnn).eval()

        with torch.no_grad():
            for param in temp_ode_model.fnn.net.parameters():
                noise = torch.randn_like(param) * nn_sigma
                param.add_(noise)

            pred = torch_odeint(
                temp_ode_model, y0, s_grid_tensor,
                method='dopri5', rtol=1e-4, atol=1e-5,
            )
            pred_trajectories.append(pred.numpy())

    pred_trajectories = np.array(pred_trajectories)
    mean_pred_norm = np.mean(pred_trajectories, axis=0)
    ci_norm = np.percentile(pred_trajectories, [5, 95], axis=0)

    mean_pred_orig = pc.inv_nor(mean_pred_norm)
    ci_orig = pc.inv_nor(ci_norm)

    # Sigmoid reference curve
    y_sigmoid_grid_norm, _ = get_sigmoid_derivatives(s_grid, sigmoid_params)
    y_sigmoid_grid_orig = pc.inv_nor(y_sigmoid_grid_norm)

    TITLES = ['Aβ (A)', 'p-Tau (T)', 'N', 'Cognition (C)']
    colors = {'CN': 'orange', 'LMCI': 'green', 'AD': 'blue', 'Other': 'grey'}
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    axes = axes.flat

    for k in range(4):
        ax = axes[k]
        for stage in np.unique(stages_pop):
            mask = np.array(stages_pop) == stage
            ax.scatter(s_pop[mask], y_pop_orig[mask, k], s=15, alpha=0.4,
                       c=colors[stage], label=f'{stage}')

        ax.plot(s_grid, mean_pred_orig[:, k], 'k-', lw=2.5,
                label='Mean Trajectory', zorder=4)
        ax.fill_between(
            s_grid, ci_orig[0, :, k], ci_orig[1, :, k],
            color='lightgrey', alpha=0.8,
            label='90% CI (NN Uncertainty)', zorder=1,
        )
        ax.plot(s_grid, y_sigmoid_grid_orig[:, k], 'r--', lw=2.5,
                label='Sigmoid Fit', zorder=3)

        ax.set_xlabel('Disease Progression Score (s)')
        ax.set_ylabel(TITLES[k])
        ax.legend()
        ax.grid(True, alpha=0.4)
        ax.set_title(TITLES[k])

    fig.suptitle('FNN Model Trajectory', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('../figures/fnn.png')
    plt.show()


# ==============================================================================
# Main pipeline
# ==============================================================================
if __name__ == '__main__':
    # 1. Prepare data
    s_pop_np, y_pop_norm_np, stages_pop_np = load_dps_and_create_scatter(
        csf_dict, dps_path='../models/dps.pth',
    )
    y_pop_orig = pc.inv_nor(y_pop_norm_np)

    # CN population average initial values
    cn_y0s = [dat['y0'] for pid, dat in patient_data.items()
              if dat['stage'] == 'CN']
    if cn_y0s:
        y0_cn_stack = torch.stack(cn_y0s)
        y0_cn_avg = torch.nanmean(y0_cn_stack, dim=0)
        y0_cn_avg = torch.nan_to_num(y0_cn_avg, nan=0.0)
    else:
        y0_cn_avg = torch.tensor([0.1, 0.0, 0.0, 0.0])
    print(f"CN population average initial values: {y0_cn_avg.numpy()}")

    # AD population averages (for regularization)
    ad_ys = []
    ad_y_final = []
    for pid, dat in patient_data.items():
        if dat['stage'] == 'AD':
            ad_ys.append(dat['y'].numpy())
            ad_y_final.append(dat['y'][-1].numpy())
    if ad_ys:
        y_ad_avg = np.nanmean(np.concatenate(ad_ys, axis=0), axis=0)
        y_ad_avg = np.nan_to_num(y_ad_avg, nan=0.0)
        y_ad_final_avg = np.nanmean(np.array(ad_y_final), axis=0)
        y_ad_final_avg = np.nan_to_num(y_ad_final_avg, nan=0.0)
    else:
        y_ad_avg = np.array([0.0, 0.0, 0.0, 0.0])
        y_ad_final_avg = np.array([0.0, 0.0, 0.0, 0.0])

    # Tau fixed-platform constraint (original space: d=70, a+d=130)
    try:
        mean_std = np.load('../data/mean_std.npy')
        tau_mean, tau_std = float(mean_std[0, 1]), float(mean_std[1, 1])
        tau_lower_norm = (70.0 - tau_mean) / tau_std
        tau_upper_norm = (130.0 - tau_mean) / tau_std
    except Exception:
        tau_lower_norm, tau_upper_norm = 0.0, 1.0

    y_target_1 = y0_cn_avg.numpy().copy()
    y_target_2 = y_ad_final_avg.copy()
    y_target_1[1] = tau_lower_norm   # Tau lower platform = 70
    y_target_2[1] = tau_upper_norm   # Tau upper platform = 130
    print(f"Tau fixed platforms (normalized): "
          f"d={tau_lower_norm:.4f}, a+d={tau_upper_norm:.4f}")

    # 2. Stage 1: Pretrain FNN via trajectory matching (regularized sigmoid)
    valid_mask = ~np.isnan(y_pop_norm_np).any(axis=1)
    s_clean = s_pop_np[valid_mask]
    y_clean = y_pop_norm_np[valid_mask]
    print(f"Valid data points for sigmoid fitting: {len(s_clean)}")
    s_grid_np_pretrain = np.linspace(-10, 20, 500)

    reg_cfg = {
        'w_turn': 5.0, 'w_center': 2.0, 'w_curv': 5.0,
        'w_cn': 10.0, 'w_plat': 8.0,
        'target_b_abs': 0.6, 'target_b_max': 1.2,
        'target_amp_max': 3.0,
        's_target_1': -20.0, 'y_target_1': y_target_1,
        's_target_2': 30.0, 'y_target_2': y_target_2,
        'y_cn': y0_cn_avg.numpy(), 'y_ad': y_ad_avg,
    }
    sigmoid_p = fit_sigmoids(s_clean, y_clean, reg_cfg=reg_cfg)
    print(f"Sigmoid parameters: {sigmoid_p}")
    fnn_pretrained = pretrain_fnn_on_sigmoid(
        sigmoid_p, y0_cn_avg, s_grid=s_grid_np_pretrain,
    )

    # 3. Plot pretraining results
    plot_pretrain_results(
        s_pop_np, y_pop_orig, stages_pop_np, s_grid_np_pretrain,
        sigmoid_p, fnn_pretrained, y0_cn_avg,
    )

    # 4. Stage 2: Fine-tune on real data (L-BFGS alternating optimization)
    print("\n--- Stage 2: Fine-tuning FNN and DPS on real data (L-BFGS) ---")
    final_ode_model, final_ab = train_fnn_on_data(
        fnn_pretrained, patient_data, y0_cn_avg, dps_path='../models/dps.pth',
    )

    # 5. Plot and save final results
    s_grid_np_final = np.linspace(-10, 20, 300)
    plot_and_save(
        s_pop_np, y_pop_orig, stages_pop_np, s_grid_np_final,
        final_ode_model, y0_cn_avg, sigmoid_p,
    )

    print("\nFull pipeline complete.")
