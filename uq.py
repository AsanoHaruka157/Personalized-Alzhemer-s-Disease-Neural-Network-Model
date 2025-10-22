import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import pccmnn as pc
import json

try:
    from torchdiffeq import odeint as torch_odeint
except ImportError:
    raise ImportError("Please install torchdiffeq to run this script: pip install torchdiffeq")

# --- 0. Data Loading and Preparation ---
csf_dict = pc.load_data()
stage_dict = pc.load_stage_dict()
print(f"Successfully loaded data for {len(csf_dict)} patients.")

patient_data = {}
for pid, sample in csf_dict.items():
    t = torch.from_numpy(sample[:, 0]).float()
    y = torch.from_numpy(sample[:, 1:5]).float()
    patient_data[pid] = {"t": t, "y": y, "y0": y[0].clone()}

def get_cn_average_y0(patient_data, stage_dict):
    cn_y0s = [data['y0'] for pid, data in patient_data.items() if stage_dict.get(pid) == 'CN']
    if not cn_y0s:
        print("Warning: No CN patients found, using default y0.")
        return torch.tensor([0.1, 0, 0, 0])
    avg_y0 = torch.stack(cn_y0s).mean(dim=0)
    print(f"Using average initial values from CN group: {avg_y0.numpy()}")
    return avg_y0

y0_cn_avg = get_cn_average_y0(patient_data, stage_dict)
name = 'fpp'

# --- 1. Define Hybrid ODE Model ---
class ODEModel(nn.Module):
    def __init__(self, hidden_dim=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 4), nn.Tanh()
        )
        self.output_scaler = nn.Parameter(torch.tensor([0.1]), requires_grad=True)
        self.wA = nn.Parameter(torch.zeros(3))
        self.wT = nn.Parameter(torch.zeros(6))
        self.wN = nn.Parameter(torch.zeros(6))
        self.wC = nn.Parameter(torch.zeros(6))
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=1e-3)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def load_poly_params(self, path='poly.pth'):
        try:
            poly_coeffs = torch.load(path)
            self.wA.data, self.wT.data, self.wN.data, self.wC.data = \
                poly_coeffs['wA'], poly_coeffs['wT'], poly_coeffs['wN'], poly_coeffs['wC']
            print(f"Successfully loaded pre-trained polynomial coefficients from {path}.")
        except FileNotFoundError:
            print(f"Warning: {path} not found.")

    def poly(self, y: torch.Tensor) -> torch.Tensor:
        A, T, N, C = y[..., 0], y[..., 1], y[..., 2], y[..., 3]
        phi_A = torch.stack([torch.ones_like(A), A, A**2], dim=-1) @ self.wA
        phi_T = torch.stack([torch.ones_like(T), T, T**2, A, A**2, A*T], dim=-1) @ self.wT
        phi_N = torch.stack([torch.ones_like(N), N, N**2, T, T**2, T*N], dim=-1) @ self.wN
        phi_C = torch.stack([torch.ones_like(C), C, C**2, N, N**2, N*C], dim=-1) @ self.wC
        return torch.stack([phi_A, phi_T, phi_N, phi_C], dim=-1)

    def f(self, y: torch.Tensor) -> torch.Tensor:
        return self.net(y) * self.output_scaler

    def combined_dynamics(self, s: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.f(y) + self.poly(y)

    def forward(self, s_grid: torch.Tensor, y0: torch.Tensor) -> torch.Tensor:
        return torch_odeint(self.combined_dynamics, y0, s_grid, method='dopri5', rtol=1e-4, atol=1e-5)

# --- 2. Define Training Procedure ---

# --- NEW: SENSITIVITY ANALYSIS AND PLOTTING FUNCTION ---
def main(patient_data, stage_dict, y0_cn_avg, base_name='fpp'):
    print("\n--- Starting Sensitivity Analysis and Plotting ---")
    
    model_path = f'{base_name}.pth'
    print(f"Loading original model {model_path}...")
    original_model = ODEModel()
    try:
        original_model.load_state_dict(torch.load(model_path))
    except FileNotFoundError:
        print(f"Error: '{model_path}' not found. Please train the model first.")
        return
    original_model.eval()

    params_path = 'sensitive_params.json'
    print(f"Loading sensitive parameter list {params_path}...")
    try:
        with open(params_path, 'r') as f:
            sensitive_params_info = json.load(f)
        print(f"Found {len(sensitive_params_info)} sensitive parameters.")
    except FileNotFoundError:
        print(f"Error: '{params_path}' not found. Please run sa.py first.")
        return

    num_samples = 100
    variance = 0.01
    std_dev = np.sqrt(variance)
    all_trajectories = []
    s_grid = torch.linspace(-10, 20, 200)

    param_map = {idx: (name, i) for idx, (name, param) in enumerate(original_model.named_parameters()) for i in range(param.numel())}

    print(f"Starting Monte Carlo simulation with {num_samples} samples...")
    successful_samples = 0
    while successful_samples < num_samples:
        print(f"  Generating sample {successful_samples + 1}/{num_samples}...", end='\r')
        perturbed_model = ODEModel()
        perturbed_model.load_state_dict(original_model.state_dict())
        perturbed_model.eval()

        with torch.no_grad():
            flat_params = {name: p.data.flatten() for name, p in perturbed_model.named_parameters()}
            for param_info in sensitive_params_info:
                global_idx = param_info['index']
                if global_idx in param_map:
                    param_name, local_idx = param_map[global_idx]
                    original_value = flat_params[param_name][local_idx]
                    new_value = torch.normal(mean=original_value, std=torch.tensor(std_dev))
                    flat_params[param_name][local_idx] = new_value
            for name, param in perturbed_model.named_parameters():
                param.data = flat_params[name].reshape(param.data.shape)

        try:
            with torch.no_grad():
                y_pred = perturbed_model(s_grid, y0_cn_avg)
            if not torch.isnan(y_pred).any():
                all_trajectories.append(y_pred.numpy())
                successful_samples += 1
        except Exception:
            # If solver throws an exception, just continue to the next attempt
            continue

    print("\nMonte Carlo simulation complete.")

    if not all_trajectories:
        print("No trajectories were successfully generated. Cannot plot.")
        return

    trajectories_stack = np.stack(all_trajectories, axis=0)
    lower_bound = np.percentile(trajectories_stack, 2.5, axis=0)
    upper_bound = np.percentile(trajectories_stack, 97.5, axis=0)

    print("Generating plot with confidence intervals...")
    with torch.no_grad():
        y_combined_normalized = original_model(s_grid, y0_cn_avg).numpy()
    
    lower_bound_orig = pc.inv_nor(lower_bound)
    upper_bound_orig = pc.inv_nor(upper_bound)
    y_combined_orig = pc.inv_nor(y_combined_normalized)

    dps_path = f'dps_{base_name}.pth'
    try:
        trained_ab = torch.load(dps_path)
    except FileNotFoundError:
        print(f"Warning: '{dps_path}' not found. Scatter plot will not show patient data.")
        trained_ab = {}

    TITLES = ['Aβ (A)', 'p-Tau (T)', 'N', 'Cognition (C)']
    colors = {'CN': 'orange', 'LMCI': 'green', 'AD': 'blue', 'Other': 'grey'}
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    axes = axes.flat
    
    for k in range(4):
        ax = axes[k]
        for pid, dat in patient_data.items():
            if pid in trained_ab:
                stage = stage_dict.get(pid, 'Other')
                a = trained_ab[pid]['a'].item()
                b = trained_ab[pid]['b'].item()
                s = a * dat['t'].numpy() + b
                y_orig = pc.inv_nor(dat['y'][:, k].numpy(), k)
                ax.scatter(s, y_orig, s=10, alpha=0.4, c=colors.get(stage, 'grey'))
        
        ax.plot(s_grid.numpy(), y_combined_orig[:, k], 'r-', lw=2.5, label='Original Trajectory', zorder=4)
        ax.fill_between(s_grid.numpy(), lower_bound_orig[:, k], upper_bound_orig[:, k], color='gray', alpha=0.3, label='95% Confidence Interval')
        ax.set_xlabel('Disease Progression Score (s)')
        ax.set_ylabel(TITLES[k])
        ax.legend()
        ax.grid(True, alpha=0.4)
        ax.set_title(TITLES[k])
        
    output_name = f'{base_name}_uq.png'
    fig.suptitle('Hybrid Model Trajectories with Uncertainty', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_name)
    print(f"Sensitivity analysis plot saved to {output_name}")
    plt.show()

if __name__ == '__main__':
    
    main(patient_data, stage_dict, y0_cn_avg, base_name=name)