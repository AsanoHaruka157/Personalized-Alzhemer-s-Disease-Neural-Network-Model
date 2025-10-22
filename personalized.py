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


# --- 1. Define Model Structure (consistent with main.py) ---
class ODEModel(nn.Module):
    def __init__(self, hidden_dim=1024):
        super().__init__()
        # Neural Network f(y)
        self.net = nn.Sequential(
            nn.Linear(4, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 4), nn.Tanh()
        )
        self.output_scaler = nn.Parameter(torch.tensor([0.1]))
        # Polynomial p(y)
        self.wA = nn.Parameter(torch.zeros(3))
        self.wT = nn.Parameter(torch.zeros(6))
        self.wN = nn.Parameter(torch.zeros(6))
        self.wC = nn.Parameter(torch.zeros(6))

    def combined_dynamics(self, s, y):
        poly_A = torch.stack([torch.ones_like(y[..., 0]), y[..., 0], y[..., 0]**2], dim=-1) @ self.wA
        poly_T = torch.stack([torch.ones_like(y[..., 1]), y[..., 1], y[..., 1]**2, y[..., 0], y[..., 0]**2, y[..., 0]*y[..., 1]], dim=-1) @ self.wT
        poly_N = torch.stack([torch.ones_like(y[..., 2]), y[..., 2], y[..., 2]**2, y[..., 1], y[..., 1]**2, y[..., 1]*y[..., 2]], dim=-1) @ self.wN
        poly_C = torch.stack([torch.ones_like(y[..., 3]), y[..., 3], y[..., 3]**2, y[..., 2], y[..., 2]**2, y[..., 2]*y[..., 3]], dim=-1) @ self.wC
        poly_dyds = torch.stack([poly_A, poly_T, poly_N, poly_C], dim=-1)
        net_dyds = self.net(y) * self.output_scaler
        return net_dyds + poly_dyds

    def forward(self, s_grid, y0):
        try:
            # First, try a more robust rk4 solver
            return torch_odeint(self.combined_dynamics, y0, s_grid, method='rk4', options={'step_size': 0.1})
        except (RuntimeError, ValueError) as e:
            print(f"  [Warning] Solver 'rk4' failed: {e}. Retrying with 'euler'...")
            try:
                # If rk4 fails, switch to the most stable euler solver
                return torch_odeint(self.combined_dynamics, y0, s_grid, method='euler', options={'step_size': 0.1})
            except (RuntimeError, ValueError) as e2:
                print(f"  [Error] Solver 'euler' also failed: {e2}. Returning NaNs.")
                # If all solvers fail, return a NaN tensor for the upper layer to handle
                return torch.full((len(s_grid), 4), float('nan'), dtype=y0.dtype)


# --- 2. Core Function for Personalization ---
def personalize_for_patient(pid, population_model, dps_params, patient_data, sensitive_indices, epochs=10, n_iter=10):
    """
    Personalizes the model for a single patient using their training data.
    """
    personal_model = copy.deepcopy(population_model)
    personal_model.train()

    # Freeze all parameters initially
    for param in personal_model.parameters():
        param.requires_grad = False

    # Unfreeze sensitive parameters based on indices
    flat_params_list = list(personal_model.parameters())
    pointer = 0
    for i, param in enumerate(flat_params_list):
        num_elements = param.numel()
        is_sensitive = any((pointer + j) in sensitive_indices for j in range(num_elements))
        if is_sensitive:
            param.requires_grad = True
        pointer += num_elements
    
    # Create personalized DPS parameters
    personal_dps_a = nn.Parameter(dps_params[pid]['a'].detach().clone())
    personal_dps_b = nn.Parameter(dps_params[pid]['b'].detach().clone())

    # Collect all parameters that need to be optimized
    params_to_optimize = [p for p in personal_model.parameters() if p.requires_grad]
    # params_to_optimize.extend([personal_dps_a, personal_dps_b]) # DPS parameters are not fine-tuned

    if not params_to_optimize:
        print(f"  [Info] No sensitive parameters to optimize for patient {pid}. Using population model.")
        personal_model.eval()
        return personal_model, {'a': personal_dps_a.item(), 'b': personal_dps_b.item()}

    # Use L-BFGS for optimization
    optimizer = optim.LBFGS(params_to_optimize, max_iter=n_iter, lr=5e-7, line_search_fn="strong_wolfe")

    # *** Use only the first 3 data points for training ***
    patient_t_train = patient_data[pid]['t'][:3]
    patient_y_train = patient_data[pid]['y'][:3]
    patient_y0 = patient_data[pid]['y0'] # The initial condition is always the first point

    last_loss = torch.tensor(0.0) # Variable to store the loss from closure
    for i in range(epochs):
        def closure():
            nonlocal last_loss
            optimizer.zero_grad()
            s_personal_train = personal_dps_a * patient_t_train + personal_dps_b
            s_sorted, indices = torch.sort(s_personal_train)
            y_sorted = patient_y_train[indices]
            
            # The ODE initial condition must correspond to the first time point in s_sorted.
            y_pred = personal_model(s_sorted, y_sorted[0])
            
            # Check if the solver returned NaNs
            if torch.isnan(y_pred).any():
                return torch.tensor(float('inf')) # If solver fails, return infinite loss

            loss = torch.mean((y_pred - y_sorted)**2)
            if torch.isfinite(loss):
                loss.backward()
            
            last_loss = loss
            return loss
        optimizer.zero_grad()
        optimizer.step(closure)
    
    print(f"  Final loss for patient {pid}: {last_loss.item()}")
    personal_model.eval()
    return personal_model, {'a': dps_params[pid]['a'].item(), 'b': dps_params[pid]['b'].item()}

# --- 3. Main Program ---
if __name__ == '__main__':
    N_PATIENTS_TO_VISUALIZE = 1

    # Set global dtype to float64 for better numerical precision
    torch.set_default_dtype(torch.float64)

    print("Loading data and pre-trained model...")
    population_model = ODEModel()
    population_model.double() 
    try:
        population_model.load_state_dict(torch.load('fpp.pth'))
        population_model.eval()
        print("Successfully loaded population model 'fpp.pth'.")
    except FileNotFoundError:
        print("Error: 'fpp.pth' not found. Please run main.py first.")
        exit()

    try:
        dps_params = torch.load('dps_fpp.pth')
        for pid in dps_params:
            dps_params[pid]['a'] = dps_params[pid]['a'].double()
            dps_params[pid]['b'] = dps_params[pid]['b'].double()
        print("Successfully loaded DPS parameters 'dps_fpp.pth'.")
    except FileNotFoundError:
        print("Error: 'dps_fpp.pth' not found. Please ensure main.py has been run to generate this file.")
        exit()
        
    try:
        with open('sensitive_params.json', 'r') as f:
            sensitive_params_info = json.load(f)
        sensitive_indices = {item['index'] for item in sensitive_params_info}
        print(f"Successfully loaded {len(sensitive_indices)} sensitive parameter indices.")
    except FileNotFoundError:
        print("Error: 'sensitive_params.json' not found. Please run the sensitivity analysis script first.")
        exit()

    csf_dict = pc.load_data()
    stage_dict = pc.load_stage_dict() # Load the stage dictionary
    patient_data = {pid: {"t": torch.from_numpy(sample[:, 0]).double(),
                          "y": torch.from_numpy(sample[:, 1:5]).double(),
                          "y0": torch.from_numpy(sample[:1, 1:5]).double().squeeze(0)}
                     for pid, sample in csf_dict.items()}

    # *** Filter for patients with at least 4 data points for cross-validation ***
    eligible_pids = [pid for pid, data in patient_data.items() if len(data['t']) >= 4]
    
    if len(eligible_pids) < N_PATIENTS_TO_VISUALIZE:
        print(f"Error: Only {len(eligible_pids)} patients have at least 4 data points. Need at least {N_PATIENTS_TO_VISUALIZE} for visualization.")
        exit()
        
    # Randomly select N_PATIENTS_TO_VISUALIZE eligible patients for display
    selected_pids = random.sample(eligible_pids, N_PATIENTS_TO_VISUALIZE)
    print(f"\nRandomly selected {N_PATIENTS_TO_VISUALIZE} patient ID(s) for visualization: {selected_pids}")

    # Create a figure with a dynamic number of rows. squeeze=False ensures axes is always 2D.
    fig, axes = plt.subplots(N_PATIENTS_TO_VISUALIZE, 4, figsize=(20, 5 * N_PATIENTS_TO_VISUALIZE), squeeze=False)
    TITLES = ['Aβ (A)', 'p-Tau (T)', 'N', 'Cognition (C)']
    
    for i, pid in enumerate(selected_pids):
        print(f"\nPersonalizing model for patient {pid}...")
        
        personal_model, personal_dps = personalize_for_patient(
            pid, population_model, dps_params, patient_data, sensitive_indices
        )
        print("Personalization complete.")

        t_patient = patient_data[pid]['t'].numpy()
        y_patient_orig = pc.inv_nor(patient_data[pid]['y'].numpy())

        s_pers = personal_dps['a'] * t_patient + personal_dps['b']

        # --- Define plotting grid based on each patient's data ---
        s_pers_sorted, s_pers_indices = np.sort(s_pers), np.argsort(s_pers)
        y_patient_sorted = patient_data[pid]['y'][s_pers_indices]
        y0_pers = y_patient_sorted[0]

        padding = 2.0
        s_min_plot = s_pers_sorted[0]
        s_max_plot = s_pers_sorted[-1] + padding
        s_grid_plot = torch.linspace(s_min_plot, s_max_plot, 200).double()

        with torch.no_grad():
            # Generate trajectories from the patient's specific initial condition and grid
            y_pop_pred_norm = population_model(s_grid_plot, y0_pers)
            y_pers_pred_norm = personal_model(s_grid_plot, y0_pers)
            
            y_pop_pred_orig = pc.inv_nor(y_pop_pred_norm.numpy()) if not torch.isnan(y_pop_pred_norm).any() else np.full((len(s_grid_plot), 4), np.nan)
            y_pers_pred_orig = pc.inv_nor(y_pers_pred_norm.numpy()) if not torch.isnan(y_pers_pred_norm).any() else np.full((len(s_grid_plot), 4), np.nan)

        for k in range(4):
            ax = axes[i, k]
            
            s_pers_train, y_patient_orig_train = s_pers[:3], y_patient_orig[:3, k]
            s_pers_test, y_patient_orig_test = s_pers[3:], y_patient_orig[3:, k]

            ax.plot(s_pers_train, y_patient_orig_train, 'o', color='blue', markersize=8, label='Training Data')
            if len(s_pers_test) > 0:
                ax.plot(s_pers_test, y_patient_orig_test, 'X', color='red', markersize=10, mew=2, label='Test Data')

            ax.plot(s_grid_plot.numpy(), y_pop_pred_orig[:, k], '--', color='gray', linewidth=2, label='Population Model')
            ax.plot(s_grid_plot.numpy(), y_pers_pred_orig[:, k], '-', color='green', linewidth=2.5, label='Personalized Model')

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
    plt.savefig('personalization.png')
    plt.show()