import torch
import torch.nn as nn
import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
import json
import pccmnn as pc # Your custom data loading module
import pytensor.tensor as pt
from pytensor.graph import Apply, Op
import os
import pickle

# Suppress verbose logging from PyTensor
import logging
logger = logging.getLogger("pytensor.graph.rewriting.basic")
logger.setLevel(logging.ERROR)

from torchdiffeq import odeint as torch_odeint

# --- 1. Define the identical ODEModel class from main.py ---
class ODEModel(nn.Module):
    def __init__(self, hidden_dim=1024):
        super().__init__()
        # Neural network part f(y)
        self.net = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 4),
            nn.Tanh()
        )
        self.output_scaler = nn.Parameter(torch.tensor([0.1]))
        # Polynomial part p(y)
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
        return torch_odeint(self.combined_dynamics, y0, s_grid, method='dopri5', rtol=1e-4, atol=1e-5)

# --- 2. PyMC Wrapper for the PyTorch ODE Model ---
class PytorchODE(Op):
    """
    A PyTensor Op that wraps the PyTorch ODE forward pass.
    This acts as the bridge between PyMC's variables and the PyTorch model.
    """
    def __init__(self, torch_model, s_grid, y0, sensitive_param_info):
        self.torch_model = torch_model
        self.s_grid = s_grid
        self.y0 = y0
        self.sensitive_param_info = sensitive_param_info

        # Freeze all non-sensitive parameters
        for param in self.torch_model.parameters():
            param.requires_grad = False

    def make_node(self, sensitive_params_var):
        # Defines the inputs and outputs of the Op for the PyTensor graph
        inputs = [pt.as_tensor_variable(sensitive_params_var)]
        outputs = [pt.matrix('y_pred')]
        return Apply(self, inputs, outputs)

    def perform(self, node, inputs, output_storage):
        # The actual computation
        sensitive_params_values = inputs[0]
        
        # 1. Update the sensitive parameters in the torch model
        with torch.no_grad():
            for i, info in enumerate(self.sensitive_param_info):
                param = info['param']
                idx = info['flat_idx']
                # Update the specific element in the flattened parameter tensor
                param.view(-1)[idx] = torch.tensor(sensitive_params_values[i], dtype=param.dtype)
        
        # 2. Run the forward pass
        try:
            with torch.no_grad():
                y_pred_tensor = self.torch_model(self.s_grid, self.y0)
            # Convert to float64 for PyMC
            y_pred = y_pred_tensor.numpy().astype(np.float64)
        except Exception:
            # If ODE solver fails, return an array of NaNs with the correct shape and type
            y_pred = np.full((len(self.s_grid), self.y0.shape[0]), np.nan, dtype=np.float64)

        output_storage[0][0] = y_pred

# --- 3. Helper Functions ---
def get_sensitive_param_info(model, sensitive_params_list):
    """
    Maps sensitive parameter names from JSON to actual torch.nn.Parameter objects.
    """
    param_info = []
    param_dict = {name: p for name, p in model.named_parameters()}
    
    for sens_param in sensitive_params_list:
        name = sens_param['name']
        # The name from sa.py might be like 'net.0.weight_123'
        base_name, flat_idx_str = name.rsplit('_', 1)
        flat_idx = int(flat_idx_str)
        
        if base_name in param_dict:
            param = param_dict[base_name]
            original_value = param.view(-1)[flat_idx].item()
            param_info.append({
                'name': name,
                'param': param,
                'flat_idx': flat_idx,
                'original_value': original_value
            })
    print(f"Successfully identified and mapped {len(param_info)} sensitive parameters.")
    return param_info

# --- 4. Main Execution Block ---
if __name__ == '__main__':
    # --- Load Data and Models ---
    print("Loading data and pre-trained models...")
    # Data
    csf_dict = pc.load_data()
    stage_dict = pc.load_stage_dict()
    patient_data = {pid: {"t": torch.from_numpy(s[:, 0]).float(), "y": torch.from_numpy(s[:, 1:5]).float()}
                    for pid, s in csf_dict.items()}
    # DPS parameters
    dps_params = torch.load('dps_fpp.pth')
    
    # ODE Model
    model = ODEModel()
    model.load_state_dict(torch.load('fpp.pth'))
    model.eval() # Set to evaluation mode

    # Sensitive parameters list
    with open('sensitive_params.json', 'r') as f:
        sensitive_params_list = json.load(f)

    # Initial condition (average of CN patients)
    cn_y0_list = [torch.from_numpy(csf_dict[pid][0, 1:5]).float() for pid in dps_params if stage_dict.get(pid) == 'CN']
    y0_cn_avg = torch.stack(cn_y0_list).mean(dim=0)
    print(f"Using CN average initial condition: {y0_cn_avg.numpy()}")
    
    # --- Prepare Data for PyMC ---
    # Collate all patient data into a single timeline based on DPS
    all_s_values, all_y_values = [], []
    for pid, params in dps_params.items():
        if pid in patient_data:
            dat = patient_data[pid]
            a = params['a'].item()
            b = params['b'].item()
            s_values = a * dat['t'] + b
            all_s_values.append(s_values)
            all_y_values.append(dat['y'])
            
    s_global = torch.cat(all_s_values)
    y_global = torch.cat(all_y_values)
    
    s_sorted, sort_indices = torch.sort(s_global)
    # Convert observed data to float64 numpy array for PyMC
    y_sorted_observed = y_global[sort_indices].numpy().astype(np.float64)

    # --- Setup PyMC Model ---
    print("\nBuilding PyMC model...")
    sens_param_info = get_sensitive_param_info(model, sensitive_params_list)
    initial_param_values = [info['original_value'] for info in sens_param_info]
    
    ode_op = PytorchODE(model, s_sorted, y0_cn_avg, sens_param_info)

    with pm.Model() as pymc_model:
        # Priors for sensitive parameters
        # Centered around their trained values, with some uncertainty
        param_stds = [1e-3 * abs(val) + 1e-4 for val in initial_param_values]
        sensitive_params = pm.Normal('sensitive_params', mu=initial_param_values, sigma=param_stds, shape=len(initial_param_values))

        # Model prediction
        y_pred = ode_op(sensitive_params)
        
        # Likelihood
        # Define observation noise as a learnable parameter
        sigma = pm.HalfCauchy('sigma', beta=1, shape=4) 
        likelihood = pm.Normal('likelihood', mu=y_pred, sigma=sigma, observed=y_sorted_observed)
        
    # --- Run MCMC Sampling ---
    TRACE_FILE = 'mcmc_trace.pkl'
    if os.path.exists(TRACE_FILE):
        print(f"\nLoading saved MCMC trace from {TRACE_FILE}...")
        with open(TRACE_FILE, 'rb') as f:
            trace = pickle.load(f)
    else:
        print("\nStarting MCMC sampling (this may take a while)...")
        with pymc_model:
            # Using 1000 tuning steps and 2000 draws per chain.
            # For a real analysis, you might increase the draws.
            trace = pm.sample(draws=500, tune=100, chains=2, cores=1, target_accept=0.9)
        
        print(f"Saving MCMC trace to {TRACE_FILE}...")
        with open(TRACE_FILE, 'wb') as f:
            pickle.dump(trace, f)

    # --- Posterior Predictive Analysis to Generate Confidence Intervals ---
    print("\nGenerating posterior predictions for confidence intervals...")
    s_grid_plot = torch.linspace(-10, 20, 200)
    
    # Extract posterior samples for the sensitive parameters
    posterior_samples = trace.posterior['sensitive_params'].values
    # Flatten chains and draws
    n_chains, n_draws, n_params = posterior_samples.shape
    posterior_samples = posterior_samples.reshape(n_chains * n_draws, n_params)
    
    # Subsample for faster prediction (e.g., 500 trajectories)
    sample_indices = np.random.randint(0, len(posterior_samples), 50)
    
    predictions = []
    for i, sample in enumerate(posterior_samples[sample_indices]):
        if (i+1) % 10 == 0:
            print(f"  ...running prediction for sample {i+1}/50")
        
        # Update model with this sample's parameters
        with torch.no_grad():
            for j, info in enumerate(sens_param_info):
                info['param'].view(-1)[info['flat_idx']] = torch.tensor(sample[j], dtype=info['param'].dtype)
        
        # Get trajectory for this parameter set
        with torch.no_grad():
            try:
                pred_traj = model(s_grid_plot, y0_cn_avg).numpy()
                predictions.append(pred_traj)
            except Exception:
                # Skip if a parameter set causes the ODE solver to fail
                continue
    
    predictions = np.array(predictions) # Shape: (n_samples, n_timesteps, n_features)

    # Calculate mean and confidence intervals
    y_mean = np.mean(predictions, axis=0)
    y_lower = np.percentile(predictions, 2.5, axis=0)
    y_upper = np.percentile(predictions, 97.5, axis=0)

    # Inverse normalize for plotting
    y_mean_orig = pc.inv_nor(y_mean)
    y_lower_orig = pc.inv_nor(y_lower)
    y_upper_orig = pc.inv_nor(y_upper)
    
    # --- Plotting ---
    print("\nPlotting final results...")
    TITLES = ['Aβ (A)', 'p-Tau (T)', 'N', 'Cognition (C)']
    colors = {'CN': 'orange', 'LMCI': 'green', 'AD': 'blue', 'Other': 'grey'}
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flat
    
    for k in range(4):
        ax = axes[k]
        # Scatter plot of patient data
        for pid, dat in patient_data.items():
            if pid in dps_params:
                stage = stage_dict.get(pid, 'Other')
                a = dps_params[pid]['a'].item()
                b = dps_params[pid]['b'].item()
                s = a * dat['t'].numpy() + b
                y_orig = pc.inv_nor(dat['y'][:, k].numpy(), k)
                ax.scatter(s, y_orig, s=12, alpha=0.3, c=colors[stage], label=stage if k==0 and pid==list(dps_params.keys())[0] else "")

        # Plot mean trajectory and confidence interval
        ax.plot(s_grid_plot.numpy(), y_mean_orig[:, k], 'r-', lw=2.5, label='Mean Trajectory', zorder=4)
        ax.fill_between(s_grid_plot.numpy(), y_lower_orig[:, k], y_upper_orig[:, k], color='red', alpha=0.2, label='95% Confidence Interval', zorder=3)
        
        ax.set_xlabel('Disease Progression Score (s)')
        ax.set_ylabel(TITLES[k])
        ax.grid(True, alpha=0.4)
        ax.set_title(TITLES[k])
    
    # Create a single legend for the figure
    handles, labels = axes[0].get_legend_handles_labels()
    # Manually create proxy artists for scatter points
    from matplotlib.lines import Line2D
    stage_handles = [Line2D([0], [0], marker='o', color='w', label=stage,
                           markerfacecolor=color, markersize=10) for stage, color in colors.items() if stage != 'Other']
    fig.legend(handles=handles + stage_handles, loc='upper right', bbox_to_anchor=(0.98, 0.98))

    fig.suptitle('Hybrid Model Trajectories with 95% Confidence Intervals (PyMC)', fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('fpp_uq.png', dpi=300)
    print("\nPlot saved to fpp_uq.png")
    plt.show()