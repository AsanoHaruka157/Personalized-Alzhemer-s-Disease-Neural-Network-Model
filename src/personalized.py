"""
Personalized model fine-tuning for individual Alzheimer's disease patients.

This script loads a pre-trained population-level Neural ODE model and
fine-tunes only the most "sensitive" parameters (identified via sensitivity
analysis) to personalize predictions for individual patients.

Key features:
  - Uses only the first 3 data points for training, with remaining points
    held out for cross-validation
  - Parameters to fine-tune are selected from a precomputed sensitivity
    analysis (sensitive_params.json)
  - L-BFGS optimizer for efficient few-shot personalization
  - Visualizes population vs. personalized model predictions

Dependencies:
  - fpp.pth: pre-trained population model weights
  - dps_fpp.pth: DPS parameters from the population model training
  - sensitive_params.json: sensitivity analysis results

Outputs:
  - personalization.png: comparison of population vs. personalized predictions
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import random
import copy
import matplotlib.pyplot as plt
import pccmnn as pc
from torchdiffeq import odeint as torch_odeint


# ==============================================================================
# 1. Model definition (consistent with main.py architecture)
# ==============================================================================
class ODEModel(nn.Module):
    """Neural ODE model combining a polynomial dynamics term + neural network.

    Dynamics: dy/ds = poly(y) + net(y) * output_scaler

    The polynomial component captures known biomarker interactions:
      - dA/ds = wA · [1, A, A^2]
      - dT/ds = wT · [1, T, T^2, A, A^2, A*T]
      - dN/ds = wN · [1, N, N^2, T, T^2, T*N]
      - dC/ds = wC · [1, C, C^2, N, N^2, N*C]

    Note: This architecture differs from both main.py and fnn.py ODEModel
    definitions. These will be unified in a future refactoring.
    """

    def __init__(self, hidden_dim=1024):
        super().__init__()
        # Neural network component f(y)
        self.net = nn.Sequential(
            nn.Linear(4, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 4), nn.Tanh(),
        )
        self.output_scaler = nn.Parameter(torch.tensor([0.1]))

        # Polynomial component p(y)
        self.wA = nn.Parameter(torch.zeros(3))
        self.wT = nn.Parameter(torch.zeros(6))
        self.wN = nn.Parameter(torch.zeros(6))
        self.wC = nn.Parameter(torch.zeros(6))

    def combined_dynamics(self, s, y):
        """Compute combined polynomial + neural network dynamics.

        Args:
            s: Disease progression score (unused, kept for API compatibility).
            y: Biomarker state tensor (..., 4).

        Returns:
            dy/ds tensor of same shape as y.
        """
        poly_A = torch.stack(
            [torch.ones_like(y[..., 0]), y[..., 0], y[..., 0] ** 2],
            dim=-1) @ self.wA
        poly_T = torch.stack(
            [torch.ones_like(y[..., 1]), y[..., 1], y[..., 1] ** 2,
             y[..., 0], y[..., 0] ** 2, y[..., 0] * y[..., 1]],
            dim=-1) @ self.wT
        poly_N = torch.stack(
            [torch.ones_like(y[..., 2]), y[..., 2], y[..., 2] ** 2,
             y[..., 1], y[..., 1] ** 2, y[..., 1] * y[..., 2]],
            dim=-1) @ self.wN
        poly_C = torch.stack(
            [torch.ones_like(y[..., 3]), y[..., 3], y[..., 3] ** 2,
             y[..., 2], y[..., 2] ** 2, y[..., 2] * y[..., 3]],
            dim=-1) @ self.wC
        poly_dyds = torch.stack([poly_A, poly_T, poly_N, poly_C], dim=-1)

        net_dyds = self.net(y) * self.output_scaler
        return net_dyds + poly_dyds

    def forward(self, s_grid, y0):
        """Integrate ODE from y0 over s_grid.

        Uses rk4 solver with euler fallback for robustness.

        Args:
            s_grid: 1D tensor of s values (must be strictly increasing).
            y0: Initial condition (4,).

        Returns:
            Trajectory tensor (len(s_grid), 4), or NaN tensor on failure.
        """
        try:
            return torch_odeint(
                self.combined_dynamics, y0, s_grid,
                method='rk4', options={'step_size': 0.1},
            )
        except (RuntimeError, ValueError) as e:
            print(f"  [Warning] Solver 'rk4' failed: {e}. Retrying with 'euler'...")
            try:
                return torch_odeint(
                    self.combined_dynamics, y0, s_grid,
                    method='euler', options={'step_size': 0.1},
                )
            except (RuntimeError, ValueError) as e2:
                print(f"  [Error] Solver 'euler' also failed: {e2}. Returning NaNs.")
                return torch.full((len(s_grid), 4), float('nan'),
                                  dtype=y0.dtype)


# ==============================================================================
# 2. Personalization function
# ==============================================================================
def personalize_for_patient(pid, population_model, dps_params, patient_data,
                            sensitive_indices, epochs=10, n_iter=10):
    """Fine-tune sensitive parameters of the population model for one patient.

    Only the parameters whose flat indices are in `sensitive_indices` are
    unfrozen and optimized. Uses only the first 3 data points for training.

    Args:
        pid: Patient ID.
        population_model: Pre-trained population ODE model.
        dps_params: Per-patient DPS parameters.
        patient_data: Per-patient biomarker data.
        sensitive_indices: Set of flat parameter indices to fine-tune.
        epochs: Number of L-BFGS outer iterations.
        n_iter: Max iterations per L-BFGS step.

    Returns:
        tuple: (personalized_model, personal_dps_dict)
    """
    personal_model = copy.deepcopy(population_model)
    personal_model.train()

    # Freeze all parameters initially
    for param in personal_model.parameters():
        param.requires_grad = False

    # Unfreeze only sensitive parameters
    flat_params_list = list(personal_model.parameters())
    pointer = 0
    for i, param in enumerate(flat_params_list):
        num_elements = param.numel()
        is_sensitive = any(
            (pointer + j) in sensitive_indices
            for j in range(num_elements))
        if is_sensitive:
            param.requires_grad = True
        pointer += num_elements

    # Create personalized DPS parameters (not fine-tuned in this version)
    personal_dps_a = nn.Parameter(dps_params[pid]['a'].detach().clone())
    personal_dps_b = nn.Parameter(dps_params[pid]['b'].detach().clone())

    # Collect optimizable parameters
    params_to_optimize = [
        p for p in personal_model.parameters() if p.requires_grad]
    # DPS parameters are not fine-tuned in current version:
    # params_to_optimize.extend([personal_dps_a, personal_dps_b])

    if not params_to_optimize:
        print(f"  [Info] No sensitive parameters for patient {pid}. "
              f"Using population model as-is.")
        personal_model.eval()
        return personal_model, {
            'a': personal_dps_a.item(),
            'b': personal_dps_b.item(),
        }

    # L-BFGS optimizer
    optimizer = optim.LBFGS(
        params_to_optimize, max_iter=n_iter, lr=5e-7,
        line_search_fn="strong_wolfe",
    )

    # Use only the first 3 data points for training
    patient_t_train = patient_data[pid]['t'][:3]
    patient_y_train = patient_data[pid]['y'][:3]

    last_loss = torch.tensor(0.0)
    for i in range(epochs):
        def closure():
            nonlocal last_loss
            optimizer.zero_grad()
            s_personal_train = personal_dps_a * patient_t_train + personal_dps_b
            s_sorted, indices = torch.sort(s_personal_train)
            y_sorted = patient_y_train[indices]

            # ODE initial condition = first time point's biomarker values
            y_pred = personal_model(s_sorted, y_sorted[0])

            if torch.isnan(y_pred).any():
                return torch.tensor(float('inf'))

            loss = torch.mean((y_pred - y_sorted) ** 2)
            if torch.isfinite(loss):
                loss.backward()

            last_loss = loss
            return loss

        optimizer.zero_grad()
        optimizer.step(closure)

    print(f"  Final loss for patient {pid}: {last_loss.item()}")
    personal_model.eval()
    return personal_model, {
        'a': dps_params[pid]['a'].item(),
        'b': dps_params[pid]['b'].item(),
    }


# ==============================================================================
# 3. Main program
# ==============================================================================
if __name__ == '__main__':
    N_PATIENTS_TO_VISUALIZE = 1

    torch.set_default_dtype(torch.float64)

    print("Loading data and pre-trained model...")
    population_model = ODEModel()
    population_model.double()
    try:
        population_model.load_state_dict(torch.load('../models/fpp.pth'))
        population_model.eval()
        print("Successfully loaded population model 'fpp.pth'.")
    except FileNotFoundError:
        print("Error: 'fpp.pth' not found. Run main.py first to generate it.")
        exit()

    try:
        dps_params = torch.load('../models/dps_fpp.pth')
        for pid in dps_params:
            dps_params[pid]['a'] = dps_params[pid]['a'].double()
            dps_params[pid]['b'] = dps_params[pid]['b'].double()
        print("Successfully loaded DPS parameters 'dps_fpp.pth'.")
    except FileNotFoundError:
        print("Error: 'dps_fpp.pth' not found. Ensure main.py has been run "
              "to generate this file.")
        exit()

    try:
        with open('../models/sensitive_params.json', 'r') as f:
            sensitive_params_info = json.load(f)
        sensitive_indices = {item['index'] for item in sensitive_params_info}
        print(f"Successfully loaded {len(sensitive_indices)} "
              f"sensitive parameter indices.")
    except FileNotFoundError:
        print("Error: 'sensitive_params.json' not found. "
              "Run sensitivity analysis first.")
        exit()

    csf_dict = pc.load_data()
    stage_dict = pc.load_stage_dict()
    patient_data = {
        pid: {
            "t": torch.from_numpy(sample[:, 0]).double(),
            "y": torch.from_numpy(sample[:, 1:5]).double(),
            "y0": torch.from_numpy(sample[:1, 1:5]).double().squeeze(0),
        }
        for pid, sample in csf_dict.items()
    }

    # Filter for patients with at least 4 data points (for cross-validation)
    eligible_pids = [
        pid for pid, data in patient_data.items()
        if len(data['t']) >= 4
    ]

    if len(eligible_pids) < N_PATIENTS_TO_VISUALIZE:
        print(f"Error: Only {len(eligible_pids)} patients have >= 4 data points. "
              f"Need at least {N_PATIENTS_TO_VISUALIZE}.")
        exit()

    # Randomly select patients for visualization
    selected_pids = random.sample(eligible_pids, N_PATIENTS_TO_VISUALIZE)
    print(f"\nRandomly selected {N_PATIENTS_TO_VISUALIZE} patient(s) "
          f"for visualization: {selected_pids}")

    # Create figure
    fig, axes = plt.subplots(
        N_PATIENTS_TO_VISUALIZE, 4,
        figsize=(20, 5 * N_PATIENTS_TO_VISUALIZE), squeeze=False,
    )
    TITLES = ['Aβ (A)', 'p-Tau (T)', 'N', 'Cognition (C)']

    for i, pid in enumerate(selected_pids):
        print(f"\nPersonalizing model for patient {pid}...")

        personal_model, personal_dps = personalize_for_patient(
            pid, population_model, dps_params, patient_data,
            sensitive_indices,
        )
        print("Personalization complete.")

        t_patient = patient_data[pid]['t'].numpy()
        y_patient_orig = pc.inv_nor(patient_data[pid]['y'].numpy())

        s_pers = personal_dps['a'] * t_patient + personal_dps['b']

        # Define plotting grid based on patient data range
        s_pers_sorted, s_pers_indices = np.sort(s_pers), np.argsort(s_pers)
        y_patient_sorted = patient_data[pid]['y'][s_pers_indices]
        y0_pers = y_patient_sorted[0]

        padding = 2.0
        s_min_plot = s_pers_sorted[0]
        s_max_plot = s_pers_sorted[-1] + padding
        s_grid_plot = torch.linspace(s_min_plot, s_max_plot, 200).double()

        with torch.no_grad():
            # Generate population and personalized trajectories
            y_pop_pred_norm = population_model(s_grid_plot, y0_pers)
            y_pers_pred_norm = personal_model(s_grid_plot, y0_pers)

            y_pop_pred_orig = (
                pc.inv_nor(y_pop_pred_norm.numpy())
                if not torch.isnan(y_pop_pred_norm).any()
                else np.full((len(s_grid_plot), 4), np.nan))
            y_pers_pred_orig = (
                pc.inv_nor(y_pers_pred_norm.numpy())
                if not torch.isnan(y_pers_pred_norm).any()
                else np.full((len(s_grid_plot), 4), np.nan))

        for k in range(4):
            ax = axes[i, k]

            # Training data (first 3 points)
            s_pers_train = s_pers[:3]
            y_patient_orig_train = y_patient_orig[:3, k]

            # Test data (remaining points)
            s_pers_test = s_pers[3:]
            y_patient_orig_test = y_patient_orig[3:, k]

            ax.plot(s_pers_train, y_patient_orig_train, 'o',
                    color='blue', markersize=8, label='Training Data')
            if len(s_pers_test) > 0:
                ax.plot(s_pers_test, y_patient_orig_test, 'X',
                        color='red', markersize=10, mew=2,
                        label='Test Data')

            ax.plot(s_grid_plot.numpy(), y_pop_pred_orig[:, k], '--',
                    color='gray', linewidth=2, label='Population Model')
            ax.plot(s_grid_plot.numpy(), y_pers_pred_orig[:, k], '-',
                    color='green', linewidth=2.5,
                    label='Personalized Model')

            if i == 0:
                ax.set_title(TITLES[k], fontsize=14)
            if k == 0:
                ax.set_ylabel(f'Patient {pid}', fontsize=14)
            if i == 0 and k == 3:
                ax.legend(loc='best')

            ax.grid(True, linestyle=':', alpha=0.6)

    fig.supxlabel('Disease Progression Score (s)', fontsize=16)
    fig.suptitle('Personalized Model Prediction', fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('../figures/personalization.png')
    plt.show()
