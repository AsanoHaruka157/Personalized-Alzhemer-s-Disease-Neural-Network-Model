"""
Main Neural ODE training pipeline for AD biomarker trajectory modeling.

This script trains a Neural ODE (with a Feedforward Neural Network as the
right-hand-side function) to reproduce sigmoid-fitted biomarker trajectories
for Alzheimer's disease progression.

Key features:
  - float64 precision throughout for numerical stability
  - [0,1] input normalization for the neural network (via population min/max)
  - Multiple optional loss components: trajectory L2, inverse-variance-weighted
    data L2, and gradient-field (RHS) matching
  - ODE integration via torchdiffeq (dopri5)

Note: This pipeline and fnn.py are currently independent experimental routes
with different model architectures and training strategies. They will be
unified in a future refactoring.

Outputs:
  - main.pt: trained FNN model weights
  - main.png: trajectory visualization with patient scatter data
  - main_loss.png: training loss curves
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import pccmnn as pc
from torchdiffeq import odeint as torch_odeint

torch.set_default_dtype(torch.float64)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ==============================================================================
# 0. Data loading and preparation
# ==============================================================================
csf_dict = pc.load_data()
stage_dict = pc.load_stage_dict()
print(f"Successfully loaded data for {len(csf_dict)} patients.")

# Convert data format for PyTorch
patient_data = {}
for pid, sample in csf_dict.items():
    t = torch.from_numpy(sample[:, 0]).to(
        dtype=torch.get_default_dtype(), device=device)
    y = torch.from_numpy(sample[:, 1:5]).to(
        dtype=torch.get_default_dtype(), device=device)
    patient_data[pid] = {
        "t": t, "y": y, "y0": y[0].clone(),
        "stage": stage_dict.get(pid, 'Other'),
    }


def get_cn_average_y0(patient_data, stage_dict):
    """Compute the average baseline biomarker values for CN subjects.

    Args:
        patient_data: Per-patient data dict.
        stage_dict: Patient diagnostic stage dict.

    Returns:
        torch.Tensor: Average initial biomarker values [Aβ, Tau, N, C].
    """
    cn_y0s = []
    for pid, data in patient_data.items():
        if stage_dict.get(pid) == 'CN':
            cn_y0s.append(data['y0'])

    if not cn_y0s:
        print("Warning: No CN patients found. Using default y0.")
        return torch.tensor([0.1, 0, 0, 0], device=device,
                            dtype=torch.get_default_dtype())

    cn_y0s_tensor = torch.stack(cn_y0s)  # (num_cn_patients, 4)

    avg_y0 = torch.zeros(4, device=device, dtype=torch.get_default_dtype())
    for k in range(4):
        y0_k = cn_y0s_tensor[:, k]
        valid_mask = ~torch.isnan(y0_k)
        if valid_mask.sum() > 0:
            avg_y0[k] = y0_k[valid_mask].mean()
        else:
            avg_y0[k] = 0.0
            print(f"Warning: All CN patients have NaN for biomarker {k}. Using 0.")

    print(f"CN population average initial values (non-NaN): {avg_y0.numpy()}")
    return avg_y0


y0_cn_avg = get_cn_average_y0(patient_data, stage_dict)

# ==============================================================================
# Global hyperparameters (main training)
# ==============================================================================
PRETRAIN_EPOCHS = 5000
PRETRAIN_LR = 1e-3
PRETRAIN_GAMMA = 0.999
LAMBDA_TRAJ = 1.0          # Trajectory L2 loss weight
LAMBDA_DYDS = 1.0          # Gradient-field loss weight (optional)
LAMBDA_DATA_L2 = 1e-2      # Inverse-variance-weighted data L2 loss weight
INV_VAR_EPS = 1e-8
USE_TRAJ_LOSS = True       # Enable trajectory loss
USE_GRADIENT_LOSS = False  # Enable gradient-field loss
USE_DATA_L2_LOSS = True    # Enable inverse-variance-weighted data L2 loss
FIGURE_PATH = '../figures/main.png'
LOSS_PATH = '../figures/main_loss.png'
MODEL_PATH = '../models/main.pt'


# ==============================================================================
# 1. Utility functions
# ==============================================================================
def compute_y_minmax_01(patient_data_dict):
    """Compute per-biomarker min/max (ignoring NaN) for [0,1] normalization.

    This normalization is applied ONLY to the neural network input; the ODE
    state y continues to evolve in the original (z-scored) space.

    Args:
        patient_data_dict: Per-patient data dict.

    Returns:
        tuple: (y_min, y_max) — each is a (4,) tensor.
    """
    ys = []
    for _, dat in patient_data_dict.items():
        ys.append(dat["y"])
    y_all = torch.cat(ys, dim=0)  # (N, 4)

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

    # Prevent division by zero
    y_max = torch.where((y_max - y_min) < 1e-6, y_min + 1e-6, y_max)
    return y_min, y_max


y_min01, y_max01 = compute_y_minmax_01(patient_data)


# ==============================================================================
# 2. Model definitions
# ==============================================================================
class FNN(nn.Module):
    """Feedforward Neural Network: maps y=[A, T, N, C] (4D) → dy/ds (4D).

    Architecture: 4 → 128 → 128 → 4 with ReLU hidden activations and Tanh output.
    Note: This differs from fnn.py's FNN (3 hidden layers, no Tanh). These will
    be unified in a future refactoring.
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
    """ODE function wrapper for torchdiffeq: func(t, y) → dy/dt.

    Applies [0,1] normalization to y before passing to the FNN, using
    population-wide min/max values. This differs from fnn.py's ODEModel
    which does not apply normalization. These will be unified in a future
    refactoring.
    """

    def __init__(self, fnn_model: FNN, y_min: torch.Tensor, y_max: torch.Tensor):
        super().__init__()
        self.fnn = fnn_model
        self.register_buffer(
            "y_min", y_min.to(dtype=torch.get_default_dtype(), device=device))
        self.register_buffer(
            "y_max", y_max.to(dtype=torch.get_default_dtype(), device=device))

    def _norm01(self, y: torch.Tensor) -> torch.Tensor:
        """Normalize y to [0, 1] range using population min/max."""
        y01 = (y - self.y_min) / (self.y_max - self.y_min)
        return torch.clamp(y01, 0.0, 1.0)

    def forward(self, t, y):
        squeeze_back = False
        if y.dim() == 1:
            y = y.unsqueeze(0)
            squeeze_back = True

        y01 = self._norm01(y)
        out = self.fnn(y01)
        return out.squeeze(0) if squeeze_back else out


# ==============================================================================
# 3. Sigmoid helper functions
# ==============================================================================
def get_sigmoid_values(sigmoid_params, s_grid_np, force_sigmoid_for_c=True):
    """Evaluate sigmoid curves on a numpy s_grid.

    For biomarker C (k=3), if d is non-finite and force_sigmoid_for_c is False,
    uses a quadratic alternative: y = a*(s - b)^2 + c.
    Otherwise, evaluates the standard sigmoid for all biomarkers.

    Args:
        sigmoid_params: (4, 4) array of sigmoid parameters.
        s_grid_np: 1D numpy array of s values.
        force_sigmoid_for_c: If True, always use sigmoid for C as well.

    Returns:
        np.ndarray: y values of shape (len(s_grid), 4).
    """
    y_on_grid = np.zeros((len(s_grid_np), 4))
    for k in range(4):
        a, b, c, d = sigmoid_params[k]
        if k == 3 and (not np.isfinite(d)) and (not force_sigmoid_for_c):
            # C non-sigmoid (e.g., quadratic) fallback
            y_on_grid[:, k] = a * ((s_grid_np - b) ** 2) + c
            continue

        exp_arg = np.clip(-b * (s_grid_np - c), -50.0, 50.0)
        y_on_grid[:, k] = a / (1.0 + np.exp(exp_arg)) + d
    return y_on_grid


def get_sigmoid_y_dyds_tensor(s_tensor: torch.Tensor, sigmoid_params,
                              device=None, force_sigmoid_for_c=True):
    """Evaluate sigmoid curves AND their derivatives on a torch s_grid.

    For biomarker C (k=3): if d is non-finite and force_sigmoid_for_c is False,
    uses quadratic: y = a*(s - b)^2 + c, dy/ds = 2a*(s - b).
    Otherwise uses standard sigmoid with analytical derivative.

    Args:
        s_tensor: 1D torch tensor of s values.
        sigmoid_params: (4, 4) array of sigmoid parameters.
        device: Torch device (auto-detected if None).
        force_sigmoid_for_c: If True, always use sigmoid for C.

    Returns:
        tuple: (y (Ns, 4), dyds (Ns, 4))
    """
    if device is None:
        device = s_tensor.device
    sig = torch.tensor(sigmoid_params, dtype=torch.get_default_dtype(),
                       device=device)  # (4, 4)

    s = s_tensor.view(-1, 1)  # (Ns, 1)
    y = torch.zeros((s.shape[0], 4), dtype=torch.get_default_dtype(),
                    device=device)
    dyds = torch.zeros_like(y)

    for k in range(4):
        a = sig[k, 0]
        b = sig[k, 1]
        c = sig[k, 2]
        d = sig[k, 3]

        use_quad_c = (k == 3) and (not torch.isfinite(d)) and (
            not force_sigmoid_for_c)
        if use_quad_c:
            y[:, k] = a * ((s[:, 0] - b) ** 2) + c
            dyds[:, k] = 2.0 * a * (s[:, 0] - b)
        else:
            exp_term = torch.exp(-b * (s[:, 0] - c))
            denom = (1.0 + exp_term)
            y[:, k] = a / denom + d
            dyds[:, k] = (a * b * exp_term) / (denom ** 2)

    return y, dyds


def get_stage_mean_y(patient_data, stage, fallback_tensor):
    """Compute the mean y (ignoring NaN) for a given diagnostic stage.

    Args:
        patient_data: Per-patient data dict.
        stage: Diagnostic stage label (e.g., 'AD').
        fallback_tensor: Fallback tensor if no data exists for the stage.

    Returns:
        torch.Tensor: Mean y values (4,).
    """
    ys = []
    for _, dat in patient_data.items():
        if dat['stage'] == stage:
            ys.append(dat['y'])
    if not ys:
        return fallback_tensor.clone()

    y_all = torch.cat(ys, dim=0)  # (N, 4)
    mean_y = torch.zeros(4)
    for k in range(4):
        col = y_all[:, k]
        mask = ~torch.isnan(col)
        if mask.any():
            mean_y[k] = col[mask].mean()
        else:
            mean_y[k] = fallback_tensor[k]
    return mean_y


def build_s_grid_from_dps(dps_params_loaded, patient_data,
                          margin_ratio=0.1, num_points=500):
    """Build a fixed s_grid from -10 to 30 with step 1.0.

    Args:
        dps_params_loaded: Loaded DPS parameters (unused, kept for API compatibility).
        patient_data: Patient data (unused, kept for API compatibility).
        margin_ratio: (Unused) margin ratio.
        num_points: (Unused) number of points.

    Returns:
        np.ndarray: s_grid from -10 to 30.
    """
    s_grid_np = np.arange(-10.0, 30.0, 1.0)
    return s_grid_np


# ==============================================================================
# 4. Training: Neural ODE to sigmoid trajectory
# ==============================================================================
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
    """Train Neural ODE to reproduce sigmoid trajectories.

    Loss components (all optional, controlled by flags):
      1. Trajectory L2: ODE-predicted trajectory vs. sigmoid curve
      2. Inverse-variance-weighted data L2 loss
      3. Gradient-field (RHS) L2: fnn(y_sigmoid(s)) vs. dy/ds_sigmoid(s)

    Uses sorted + deduplicated s values for efficient ODE integration,
    then maps predictions back to original indices.

    Args:
        fnn_model: FNN model to train.
        sigmoid_params: (4, 4) sigmoid parameters.
        s_grid_np: 1D numpy array of s values.
        epochs: Number of training epochs.
        lr: Initial learning rate (Adam).
        gamma: ExponentialLR decay factor.
        lambda_traj: Trajectory loss weight.
        lambda_dyds: Gradient-field loss weight.
        lambda_data_l2: Inverse-variance-weighted data L2 loss weight.
        inv_var_eps: Epsilon for inverse variance computation.
        use_traj_loss: Enable trajectory loss.
        use_gradient_loss: Enable gradient-field loss.
        use_data_l2_loss: Enable data L2 loss.

    Returns:
        tuple: (trained_fnn, loss_history_dict)
    """
    ode_model = ODEModel(fnn_model, y_min01, y_max01).train()
    criterion = nn.MSELoss()

    s_tensor = torch.tensor(s_grid_np, dtype=torch.get_default_dtype(),
                            device=device)
    fnn_model = fnn_model.to(device)
    ode_model = ode_model.to(device)

    # Sort and deduplicate s to ensure valid ODE solver input
    s_sorted, sort_idx = torch.sort(s_tensor)
    s_unique, inverse_idx = torch.unique_consecutive(
        s_sorted, return_inverse=True)

    y_sigmoid_all, dyds_sigmoid_all = get_sigmoid_y_dyds_tensor(
        s_tensor, sigmoid_params)
    y_sigmoid, dyds_sigmoid = get_sigmoid_y_dyds_tensor(
        s_unique, sigmoid_params)

    # Map sigmoid values back to original (unsorted) indices
    y_sigmoid_sorted = y_sigmoid_all[sort_idx]
    dyds_sigmoid_sorted = dyds_sigmoid_all[sort_idx]
    y_sigmoid_mapped = y_sigmoid_sorted[inverse_idx]
    dyds_sigmoid_mapped = dyds_sigmoid_sorted[inverse_idx]

    # Gradient-field constraint targets (fixed, epoch-independent)
    y_rhs_input = y_sigmoid_mapped.detach()
    A_target = dyds_sigmoid_mapped.detach()

    # Initial condition: sigmoid starting point
    y0 = y_sigmoid[0]

    loss_history = {
        'epoch': [], 'traj_loss': [], 'data_l2_loss': [], 'dyds_loss': [],
    }
    print_interval = max(1, epochs // 10)

    optimizer = optim.Adam(ode_model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    for epoch in range(1, epochs + 1):
        loss_traj = torch.tensor(0.0, device=device,
                                 dtype=torch.get_default_dtype())
        loss_data_l2 = torch.tensor(0.0, device=device,
                                    dtype=torch.get_default_dtype())
        loss_dyds = torch.tensor(0.0, device=device,
                                 dtype=torch.get_default_dtype())

        optimizer.zero_grad()
        try:
            # ODE integration on deduplicated s
            y_pred_unique = torch_odeint(
                ode_model, y0, s_unique,
                method='dopri5', rtol=1e-4, atol=1e-5,
            )
            y_pred = y_pred_unique[inverse_idx]  # Map back to original indices

            if use_traj_loss:
                loss_traj = criterion(y_pred, y_sigmoid_mapped)

            if use_data_l2_loss:
                err = y_pred - y_sigmoid_mapped  # (N, 4)
                err_var = torch.var(err, dim=0, unbiased=False)  # (4,)
                inv_var = 1.0 / (err_var + inv_var_eps)
                loss_data_l2 = torch.mean((err ** 2) * inv_var.view(1, -1))

            if use_gradient_loss:
                rhs = ode_model(
                    torch.tensor(0.0, device=y_pred.device), y_rhs_input)
                loss_dyds = torch.norm((rhs - A_target).reshape(-1), p=2)

            loss = (lambda_traj * loss_traj
                    + lambda_data_l2 * loss_data_l2
                    + lambda_dyds * loss_dyds)
            if torch.isfinite(loss):
                loss.backward()
                optimizer.step()
        except Exception as e:
            print(f"ODE solver failed: {e}")
            loss = torch.tensor(float('inf'), device=device,
                                dtype=torch.get_default_dtype())

        scheduler.step()

        loss_history['epoch'].append(epoch)
        loss_history['traj_loss'].append(float(loss_traj))
        loss_history['data_l2_loss'].append(float(loss_data_l2))
        loss_history['dyds_loss'].append(float(loss_dyds))

        if epoch % print_interval == 0 or epoch == 1 or epoch == epochs:
            progress = int(epoch / epochs * 100)
            print(f"[Train {progress:3d}%] traj={float(loss_traj):.6f}, "
                  f"data_l2={float(loss_data_l2):.6f}, "
                  f"dyds={float(loss_dyds):.6f}")

    ode_model.eval()
    return ode_model.fnn, loss_history


# ==============================================================================
# 5. FNN loss with DPS-fixed data fitting (optional second stage)
# ==============================================================================
def calculate_loss_fnn(
    ode_model, patient_data, ab, pids, y0,
    sigmoid_params=None, lambda_sigmoid=1.0, lambda_dyds=1.0,
    n_dyds_samples=200,
):
    """Efficient FNN loss with data fitting + sigmoid + gradient-field regularization.

    Merges all patients and biomarkers into a single batch, deduplicates and
    sorts s values for a single ODE solve, then maps predictions back to
    original indices.

    Args:
        ode_model: ODE model.
        patient_data: Per-patient data dict.
        ab: Per-patient DPS params {pid: {'a': tensor, 'b': tensor}}.
        pids: List of patient IDs.
        y0: Initial condition (4,).
        sigmoid_params: Optional (4, 4) sigmoid params for regularization.
        lambda_sigmoid: Sigmoid shape constraint weight.
        lambda_dyds: Gradient-field regularization weight.
        n_dyds_samples: Number of random s samples for gradient-field reg.

    Returns:
        tuple: (total_loss_tensor, loss_dict)
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
            return torch.tensor(0.0, device=device,
                                dtype=torch.get_default_dtype(),
                                requires_grad=True)

        s_all = torch.cat(s_all_list)
        y_true_all = torch.cat(y_true_list)
        k_all = torch.tensor(k_list, device=y_true_all.device, dtype=torch.long)
        y_true_all = y_true_all.to(dtype=torch.get_default_dtype())

        # Sort + deduplicate outside computation graph
        with torch.no_grad():
            s_sorted, sort_idx = torch.sort(s_all)
            y_sorted = y_true_all[sort_idx]
            k_sorted = k_all[sort_idx]
            s_unique, inv = torch.unique_consecutive(
                s_sorted, return_inverse=True)

        # Single ODE solve on deduplicated s
        y_unique = torch_odeint(
            ode_model, y0, s_unique,
            method='dopri5', rtol=1e-4, atol=1e-5,
        )  # (Nu, 4)

        y_all = y_unique[inv]  # (N, 4)
        y_pred_selected = y_all[torch.arange(y_all.size(0)), k_sorted]

        # Data fitting loss (SmoothL1)
        smooth_l1_loss = nn.SmoothL1Loss(reduction='sum')
        data_loss = smooth_l1_loss(y_pred_selected, y_sorted)

        # Sigmoid shape constraint + gradient-field regularization
        sigmoid_loss = torch.tensor(0.0, device=y_unique.device,
                                    dtype=torch.get_default_dtype())
        dyds_reg_loss = torch.tensor(0.0, device=y_unique.device,
                                     dtype=torch.get_default_dtype())
        if sigmoid_params is not None:
            sig = torch.tensor(sigmoid_params, dtype=torch.get_default_dtype(),
                               device=y_unique.device)  # (4, 4)
            a = sig[:, 0].view(1, 4)
            b = sig[:, 1].view(1, 4)
            c = sig[:, 2].view(1, 4)
            d = sig[:, 3].view(1, 4)
            s_expanded = s_unique.view(-1, 1)  # (Nu, 1)
            exp_term = torch.exp(-b * (s_expanded - c))
            y_sig = a / (1.0 + exp_term) + d  # (Nu, 4)
            sigmoid_loss = torch.mean((y_unique - y_sig) ** 2)

            # Gradient-field (vector field) regularization
            with torch.no_grad():
                s_lo = s_unique.min().item()
                s_hi = s_unique.max().item()
            s_sample = (torch.rand(n_dyds_samples, device=y_unique.device,
                                   dtype=torch.get_default_dtype())
                        * (s_hi - s_lo) + s_lo)
            y_sig_s, dyds_sig_s = get_sigmoid_y_dyds_tensor(
                s_sample, sigmoid_params, device=y_unique.device)
            dyds_pred = ode_model(
                torch.tensor(0.0, device=y_unique.device), y_sig_s)
            dyds_reg_loss = torch.mean((dyds_pred - dyds_sig_s) ** 2)

        # L1 regularization on FNN parameters
        l1_reg = 0.0
        for param in ode_model.parameters():
            l1_reg += torch.sum(torch.abs(param))

        lambda_l1 = 0.0001
        total_loss = (data_loss + lambda_l1 * l1_reg
                      + lambda_sigmoid * sigmoid_loss
                      + lambda_dyds * dyds_reg_loss)

        loss_dict = {
            'total': (total_loss.item()
                      if torch.isfinite(total_loss) else float('inf')),
            'data': data_loss.item(),
            'l1': l1_reg.item(),
            'lambda_l1': lambda_l1,
            'sigmoid': (sigmoid_loss.item()
                        if torch.isfinite(sigmoid_loss) else float('inf')),
            'lambda_sigmoid': lambda_sigmoid,
            'dyds_reg': (dyds_reg_loss.item()
                         if torch.isfinite(dyds_reg_loss) else float('inf')),
            'lambda_dyds': lambda_dyds,
            'n_dyds_samples': n_dyds_samples,
        }

        if torch.isfinite(total_loss):
            return total_loss, loss_dict
        else:
            return torch.tensor(float('inf'), device=device,
                                dtype=torch.get_default_dtype(),
                                requires_grad=True), loss_dict

    except Exception as e:
        print(f"FNN loss computation error: {e}")
        import traceback
        traceback.print_exc()
        loss_dict = {'total': float('inf'), 'data': 0, 'l1': 0,
                     'lambda': 0.0001}
        return torch.tensor(float('inf'), device=device,
                            dtype=torch.get_default_dtype(),
                            requires_grad=True), loss_dict


def calculate_loss_dps(ode_model, patient_data, ab, pids, y0):
    """Efficient DPS loss with data fitting + L2 regularization on a, b.

    Args:
        ode_model: ODE model.
        patient_data: Per-patient data dict.
        ab: Per-patient DPS params.
        pids: List of patient IDs.
        y0: Initial condition (4,).

    Returns:
        tuple: (total_loss_tensor, loss_dict)
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
            return (torch.tensor(0.0, device=device,
                                 dtype=torch.get_default_dtype(),
                                 requires_grad=True),
                    {'total': 0.0, 'data': 0.0, 'l2': 0.0})

        s_all = torch.stack(s_all_list)
        y_true_all = torch.stack(y_true_list)
        k_all = torch.tensor(k_list, device=y_true_all.device, dtype=torch.long)

        with torch.no_grad():
            s_sorted, sort_idx = torch.sort(s_all)
            y_sorted = y_true_all[sort_idx]
            k_sorted = k_all[sort_idx]
            s_unique, inv = torch.unique_consecutive(
                s_sorted, return_inverse=True)

        y_unique = torch_odeint(
            ode_model, y0, s_unique,
            method='dopri5', rtol=1e-4, atol=1e-5,
        )

        y_all = y_unique[inv]
        y_pred_selected = y_all[torch.arange(y_all.size(0)), k_sorted]

        data_loss = ((y_pred_selected - y_sorted) ** 2).sum()

        # L2 regularization on DPS parameters a, b
        l2_reg = 0.0
        for pid in pids:
            l2_reg += ab[pid]['a'] ** 2 + ab[pid]['b'] ** 2

        lambda_l2 = 0.01
        total_loss = data_loss + lambda_l2 * l2_reg

        loss_dict = {
            'total': (total_loss.item()
                      if torch.isfinite(total_loss) else float('inf')),
            'data': data_loss.item(),
            'l2': l2_reg.item(),
        }

        if torch.isfinite(total_loss):
            return total_loss, loss_dict
        else:
            return torch.tensor(float('inf'), device=device,
                                dtype=torch.get_default_dtype(),
                                requires_grad=True), loss_dict

    except Exception as e:
        print(f"DPS loss computation error: {e}")
        loss_dict = {'total': float('inf'), 'data': 0.0, 'l2': 0.0}
        return torch.tensor(float('inf'), device=device,
                            dtype=torch.get_default_dtype(),
                            requires_grad=True), loss_dict


def train_fnn_with_fixed_dps(
    fnn_pretrained, patient_data, y0, dps_path='dps.pth',
    n_epochs=2000, lr_fnn=1e-2, sigmoid_params=None,
    lambda_traj=1.0, lambda_dyds=1.0,
):
    """Train FNN only, with DPS parameters (a, b) fixed throughout.

    Args:
        fnn_pretrained: Pretrained FNN model.
        patient_data: Per-patient data.
        y0: Initial condition (4,).
        dps_path: Path to DPS parameter file.
        n_epochs: Number of training epochs.
        lr_fnn: Learning rate for FNN (Adam).
        sigmoid_params: Optional sigmoid params for regularization.
        lambda_traj: Trajectory regularization weight.
        lambda_dyds: Gradient-field regularization weight.

    Returns:
        tuple: (ode_model, ab_dict, loss_history) or (None, None, None) on error.
    """
    print(f"\n--- Training FNN (DPS fixed) ---")

    ode_model = ODEModel(fnn_pretrained, y_min01, y_max01).train()

    try:
        dps_params_loaded = torch.load(dps_path, weights_only=False)
        ab = {}
        for pid in patient_data.keys():
            if pid in dps_params_loaded:
                ab[pid] = {
                    'a': torch.tensor(dps_params_loaded[pid]['a'],
                                      dtype=torch.get_default_dtype(),
                                      device=device),
                    'b': torch.tensor(dps_params_loaded[pid]['b'],
                                      dtype=torch.get_default_dtype(),
                                      device=device),
                }
        print(f"Successfully loaded DPS parameters from {dps_path} (fixed).")
    except FileNotFoundError:
        print(f"Error: {dps_path} not found.")
        return None, None, None

    patient_pids = list(ab.keys())
    opt_fnn = optim.Adam(ode_model.parameters(), lr=lr_fnn)

    loss_history = {'epoch': [], 'traj_loss': [], 'dyds_loss': []}
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

        loss_history['epoch'].append(epoch)
        loss_history['traj_loss'].append(
            loss_fnn_dict.get('sigmoid', 0.0))
        loss_history['dyds_loss'].append(
            loss_fnn_dict.get('dyds_reg', 0.0))

        if epoch % print_interval == 0 or epoch == 1 or epoch == n_epochs:
            progress = int(epoch / n_epochs * 100)
            print(f"[Train {progress:3d}%] "
                  f"traj={loss_fnn_dict.get('sigmoid', 0.0):.6f}, "
                  f"dyds={loss_fnn_dict.get('dyds_reg', 0.0):.6f}")

    ode_model.eval()
    print("\nTraining complete!")
    return ode_model, ab, loss_history


# ==============================================================================
# 6. Main program
# ==============================================================================
if __name__ == '__main__':
    # Initialize FNN model
    print("\n--- Initializing FNN model ---")
    fnn_pretrained = FNN(input_dim=4, hidden_dim=128, output_dim=4).to(device)

    # Initialize weights from N(0, 0.01)
    with torch.no_grad():
        for p in fnn_pretrained.parameters():
            nn.init.normal_(p, mean=0.0, std=0.01)
    print("Model parameters initialized from N(0, 0.01).")

    # Load sigmoid and DPS parameters
    try:
        sigmoid_params = torch.load('../models/sigmoid.pth', weights_only=False)
        print("Successfully loaded sigmoid.pth")
    except FileNotFoundError:
        print("Error: sigmoid.pth not found. Run sigmoid.py first.")
        exit()

    try:
        dps_params_loaded = torch.load('../models/dps.pth', weights_only=False)
        print("Successfully loaded dps.pth")
    except FileNotFoundError:
        print("Error: dps.pth not found. Run sigmoid.py first.")
        exit()

    # Compute CN average initial and AD average values
    y_cn_avg = y0_cn_avg.to(device)
    y_ad_avg = get_stage_mean_y(patient_data, 'AD', y_cn_avg).to(device)

    # Build s_grid for stage 1
    s_grid_np = build_s_grid_from_dps(dps_params_loaded, patient_data)

    # Train: sigmoid trajectory matching
    print("\n--- Training: Sigmoid trajectory matching ---")
    fnn_trained, loss_history = train_neural_ode_to_sigmoid_with_dyds(
        fnn_pretrained, sigmoid_params, s_grid_np,
        epochs=PRETRAIN_EPOCHS, lr=PRETRAIN_LR, gamma=PRETRAIN_GAMMA,
        lambda_traj=LAMBDA_TRAJ, lambda_dyds=LAMBDA_DYDS,
        lambda_data_l2=LAMBDA_DATA_L2, inv_var_eps=INV_VAR_EPS,
        use_traj_loss=USE_TRAJ_LOSS,
        use_gradient_loss=USE_GRADIENT_LOSS,
        use_data_l2_loss=USE_DATA_L2_LOSS,
    )

    if fnn_trained is None:
        exit()

    final_model = ODEModel(fnn_trained, y_min01, y_max01).to(device).eval()

    # Save trained model
    torch.save(fnn_trained.state_dict(), MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")

    # --- Plot loss curves ---
    print("\n--- Generating loss curves ---")
    steps = list(range(len(loss_history['traj_loss'])))

    plot_items = []
    if USE_TRAJ_LOSS:
        plot_items.append(('traj_loss', 'Trajectory L2', 'b-'))
    if USE_DATA_L2_LOSS:
        plot_items.append(('data_l2_loss', 'Inv-Var Weighted Data L2', 'g-'))
    if USE_GRADIENT_LOSS:
        plot_items.append(('dyds_loss', 'RHS L2 (optional)', 'r-'))

    if len(plot_items) > 0:
        fig_loss, axes = plt.subplots(
            len(plot_items), 1, figsize=(8, 4 * len(plot_items)),
            squeeze=False)
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
        plt.savefig(LOSS_PATH)
        print(f"Loss curve saved to {LOSS_PATH}")
        plt.show()
    else:
        print("All loss plotting switches are False; skipping loss curves.")

    # --- Generate visualization ---
    print("\n--- Generating visualization ---")

    s_grid = torch.arange(-20.0, 30.0 + 0.1, 0.1, device=device,
                          dtype=torch.get_default_dtype())
    print(f"s_grid range: [{s_grid.min():.2f}, {s_grid.max():.2f}]")

    s_sorted, sort_idx = torch.sort(s_grid)
    s_unique, inverse_idx = torch.unique_consecutive(
        s_sorted, return_inverse=True)

    with torch.no_grad():
        try:
            # Use sigmoid starting point as initial condition
            y_sigmoid_plot, _ = get_sigmoid_y_dyds_tensor(
                s_unique, sigmoid_params, device=device)
            y0_plot = y_sigmoid_plot[0]

            y_pred_unique = torch_odeint(
                final_model, y0_plot, s_unique,
                method='dopri5', rtol=1e-4, atol=1e-5,
            )
            y_pred = y_pred_unique[inverse_idx]
            y_pred_orig = pc.inv_nor(y_pred.numpy())
        except Exception as e:
            print(f"ODE solver failed during plotting: {e}")
            exit()

        # Compute sigmoid curve (original space)
        y_sigmoid_np = get_sigmoid_values(sigmoid_params, s_grid.numpy())
        y_sigmoid_orig = pc.inv_nor(y_sigmoid_np)

        TITLES = ['Aβ (A)', 'Tau (T)', 'N', 'Cognition (C)']
        colors = {'CN': 'orange', 'LMCI': 'green', 'AD': 'blue', 'Other': 'grey'}

        fig, axes = plt.subplots(2, 2, figsize=(14, 11))
        axes = axes.flat

        for k in range(4):
            ax = axes[k]

            # Plot patient data points
            for pid, dat in patient_data.items():
                if pid in dps_params_loaded:
                    stage = dat['stage']
                    a = float(dps_params_loaded[pid]['a'])
                    b = float(dps_params_loaded[pid]['b'])
                    s = a * dat['t'].numpy() + b
                    y_orig = pc.inv_nor(dat['y'][:, k].numpy(), k)
                    ax.scatter(s, y_orig, s=22, alpha=0.8,
                               c=colors[stage], edgecolors='none', zorder=1)

            # Plot FNN trajectory
            ax.plot(s_grid.numpy(), y_pred_orig[:, k], 'r-', lw=2.5,
                    label='FNN Trajectory', zorder=3)

            # Plot sigmoid curve
            ax.plot(s_grid.numpy(), y_sigmoid_orig[:, k], 'k--', lw=2.0,
                    label='Sigmoid', zorder=2)

            ax.set_xlabel('Disease Progression Score (s)')
            ax.set_ylabel(TITLES[k])
            ax.set_xlim(s_grid.min().item(), s_grid.max().item())
            ax.legend()
            ax.grid(True, alpha=0.4)
            ax.set_title(TITLES[k])

        fig.suptitle('FNN Model (Fixed DPS) with Sigmoid Constraints',
                     fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(FIGURE_PATH)
        print(f"Result figure saved to {FIGURE_PATH}")
        plt.show()

    print("\nFull pipeline complete.")
