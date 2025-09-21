import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import pccmnn as pc
from torch.utils.data import Dataset, DataLoader
from datetime import datetime

# ------------------ 加载数据 ------------------
csf_dict = pc.load_data()
print("Number of valid patients:", len(csf_dict))

patient_data = {}
for pid, sample in csf_dict.items():
    t = torch.from_numpy(sample[:, 0]).float()            # 年龄
    y = torch.from_numpy(sample[:, 1:5]).float()          # biomarker A/T/N/C
    patient_data[pid] = {"t": t, "y": y, "y0": y[0].clone()}

stage_dict = pc.load_stage_dict()

def get_cn_average_y0(patient_data, stage_dict):
    cn_y0s = []
    for pid, data in patient_data.items():
        if stage_dict.get(pid) == 'CN':
            cn_y0s.append(data['y0'])
    if not cn_y0s:
        print("Warning: No CN patients found. Using default y0.")
        return torch.tensor([0.1, 0, 0, 0])
    avg_y0 = torch.stack(cn_y0s).mean(dim=0)
    print(f"Using average CN y0: {avg_y0.numpy()}")
    return avg_y0

y0_cn_avg = get_cn_average_y0(patient_data, stage_dict)


# ---------- 计算y的分位数并保存为全局变量B, C, D ----------
all_y_data = torch.cat([dat['y'] for dat in patient_data.values()], dim=0)
B = torch.quantile(all_y_data, 0.05, dim=0)
C = torch.quantile(all_y_data, 0.50, dim=0)
D = torch.quantile(all_y_data, 0.95, dim=0)

Message = f"Polynomial-only model with fixed pretrained DPS parameters."
name = 'sp'


def _pos(x: torch.Tensor) -> torch.Tensor:
    """Strictly positive parameter (stabilized)."""
    return F.softplus(x) + 1e-8


def _inv_softplus(y: float) -> float:
    """Inverse of softplus for initializing to a desired positive value."""
    # y = log(1 + exp(x))  ->  x = log(exp(y) - 1)
    return float(torch.log(torch.expm1(torch.tensor(y))).item())


class ODEModel(nn.Module):
    def __init__(self, init_small: float = 0.05):
        super().__init__()
        
        def p(v):  # raw parameter that softplus->v
            return nn.Parameter(torch.tensor(_inv_softplus(v)))

        # Logistic growth parameters: r*y*(1-y/K)
        # A
        self.rA = p(0.01)
        self.KA = p(2.0)
        # T
        self.rT = p(0.01)
        self.KT = p(2.0)
        # N
        self.rN = p(0.01)
        self.KN = p(2.0)
        # C
        self.rC = p(0.01)
        self.KC = p(2.0)

        # Coupling terms
        self.at2 = p(init_small)  # A^2 -> T ( + )
        self.at1 = p(init_small)  # A*T ( - )
        self.tt2 = p(init_small)  # T^2 -> N ( + )
        self.tn1 = p(init_small)  # T*N ( - )
        self.nc1 = p(init_small)  # N*C ( - )

    def poly(self, y: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        A, T, N, C = y[..., 0], y[..., 1], y[..., 2], y[..., 3]

        # Logistic self-dynamics
        dA_self = (_pos(self.rA) + 0.01) * A * (1 - A / _pos(self.KA))
        dT_self = (_pos(self.rT) + 0.01) * T * (1 - T / _pos(self.KT))
        dN_self = (_pos(self.rN) + 0.01) * N * (1 - N / _pos(self.KN))
        dC_self = (_pos(self.rC) + 0.01) * C * (1 - C / _pos(self.KC))

        # Full equations with coupling
        dA = dA_self
        dT = dT_self + _pos(self.at2) * (A * A) - _pos(self.at1) * (A * T)
        dN = dN_self + _pos(self.tt2) * (T * T) - _pos(self.tn1) * (T * N)
        dC = dC_self - _pos(self.nc1) * (N * C)
        
        p = torch.stack([dA, dT, dN, dC], dim=-1)
        return p

    def _rk4_integrate(self, s_grid: torch.Tensor, y0: torch.Tensor, f_fn) -> torch.Tensor:
        ys = [y0]
        for i in range(1, len(s_grid)):
            h = s_grid[i] - s_grid[i - 1]
            y_i = ys[-1]

            k1 = f_fn(y_i, s_grid[i - 1])
            k2 = f_fn(y_i + 0.5 * h * k1, s_grid[i - 1] + 0.5 * h)
            k3 = f_fn(y_i + 0.5 * h * k2, s_grid[i - 1] + 0.5 * h)
            k4 = f_fn(y_i + h * k3, s_grid[i])

            y_next = y_i + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            ys.append(y_next)

        return torch.stack(ys)

    def forward(self, s_grid: torch.Tensor, y0: torch.Tensor) -> torch.Tensor:
        # 仅多项式动力学: dy/ds = poly(y)
        return self._rk4_integrate(s_grid, y0, lambda y, s: self.poly(y, s))


def calculate_global_loss(model, s_global, y_global, y0_global, sigma=None, tail_penalty_factor=0.0):
    if s_global.numel() == 0:
        return torch.tensor(0.0, device=s_global.device, dtype=s_global.dtype, requires_grad=True)

    y_pred_global = model(s_global, y0_global)
    loss = (y_pred_global - y_global) ** 2

    if tail_penalty_factor > 0:
        s_detached = s_global.detach()
        if s_detached.numel() > 0:
            s_min = s_detached.min()
            s_max = s_detached.max()
            if s_max > s_min:
                # Linearly scale weights from 1 to 1+factor
                weights = 1.0 + tail_penalty_factor * (s_detached - s_min) / (s_max - s_min)
                loss = loss * weights.unsqueeze(-1)

    if sigma is not None:
        loss = loss * sigma
    return loss.mean()


class PatientDataset(Dataset):
    def __init__(self, pids):
        self.pids = pids

    def __len__(self):
        return len(self.pids)

    def __getitem__(self, idx):
        return self.pids[idx]


def pack_params(model):
    return np.concatenate([p.data.detach().numpy().flatten() for p in model.parameters()])

def unpack_params(model, params_flat):
    offset = 0
    for p in model.parameters():
        numel = p.numel()
        p.data = torch.from_numpy(params_flat[offset:offset + numel]).view_as(p.data)
        offset += numel

def pack_dps_params(ab):
    return np.concatenate([ab[pid]['theta'].detach().numpy().flatten() for pid in sorted(ab.keys())])

def unpack_dps_params(ab, params_flat):
    offset = 0
    for pid in sorted(ab.keys()):
        numel = ab[pid]['theta'].numel()
        ab[pid]['theta'] = torch.from_numpy(params_flat[offset:offset + numel]).view_as(ab[pid]['theta'])
        offset += numel
    return ab

def fit_population(
    patient_data,
    y0_global,
    n_epochs=10,
    max_iter_w=20,
    max_iter_dps=20,
    batch_size=128,
    lr_w=1e-3,
    lr_dps=1e-3,
    weighted_sampling=True,
    inducement_weight=1e-1,
    boundary_penalty_weight=50.0,
    variance_penalty_weight=0.05,
    dps_init_method='pretrain',
):
    sigma = torch.ones(4)
    model = ODEModel()

    # ---------- 初始化 ----------
    # --------- Minibatch Dataloader Setup -----------
    patient_pids = list(patient_data.keys())
    use_minibatch = batch_size < len(patient_pids)

    batch_iterator = None
    if use_minibatch and n_epochs > 0:
        print(f"Using mini-batching with batch size {batch_size}.")
        dataset = PatientDataset(patient_pids)

        sampler = None
        if weighted_sampling:
            print("Using weighted sampling based on the number of time points per patient.")
            weights = [float(patient_data[pid]['t'].shape[0]) for pid in patient_pids]
            sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=n_epochs * batch_size, replacement=True)

        dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, shuffle=(sampler is None))
        batch_iterator = iter(dataloader)

    # Pre-calculate scores for inducement term (now removed from loss)
    pid_to_score = {}
    for pid, dat in patient_data.items():
        y = dat['y']
        z = (-y[:, 0]) + (y[:, 1]) + (-y[:, 2]) + (y[:, 3])
        pid_to_score[pid] = float(z.mean())
    
    ab = {}
    if dps_init_method == 'new':
        # Initialize DPS parameters from scratch with randomization
        print("Initializing DPS parameters randomly to cover [-10, 20].")
        for pid, dat in patient_data.items():
            # Randomly initialize theta0, then compute 'a'
            theta0 = torch.randn(1).item()
            a_init = 4.0 * torch.sigmoid(torch.tensor(theta0))

            # Initialize 'b' such that mean(s) is random within [-10, 20]
            t_mean = dat['t'].mean()
            s_target_mean = torch.rand(1).item() * 30.0 - 10.0 # Random value in [-10, 20]
            b_init = s_target_mean - a_init * t_mean
            theta1 = torch.tensor(b_init)
            
            theta = torch.tensor([theta0, theta1.item()], dtype=torch.float32, requires_grad=True)
            ab[pid] = {"theta": theta}
            
    elif dps_init_method == 'pretrain':
        # 加载预训练的DPS参数
        try:
            ab = torch.load('dps.pth')
            print("Successfully loaded pretrained DPS parameters from dps.pth")
            for pid in list(ab.keys()):
                if pid not in patient_data:
                    del ab[pid]
                    continue
                if 'theta' in ab[pid]:
                    ab[pid]['theta'] = ab[pid]['theta'].detach().requires_grad_(True)
                else:
                    alpha, beta = ab[pid]
                    ab[pid] = {'theta': alpha, 'beta': beta} # This will fail later, but keep original logic
        except FileNotFoundError:
            print("Warning: dps.pth not found. Falling back to 'new' random initialization.")
            # Fallback to 'new' method
            for pid, dat in patient_data.items():
                # Randomly initialize theta0, then compute 'a'
                theta0 = torch.randn(1).item()
                a_init = 4.0 * torch.sigmoid(torch.tensor(theta0))

                # Initialize 'b' such that mean(s) is random within [-10, 20]
                t_mean = dat['t'].mean()
                s_target_mean = torch.rand(1).item() * 30.0 - 10.0 # Random value in [-10, 20]
                b_init = s_target_mean - a_init * t_mean
                theta1 = torch.tensor(b_init)
                
                theta = torch.tensor([theta0, theta1.item()], dtype=torch.float32, requires_grad=True)
                ab[pid] = {"theta": theta}
    else:
        raise ValueError(f"Unknown dps_init_method: '{dps_init_method}'. Choose 'new' or 'pretrain'.")

    # --------- LBFGS Training Loop -----------
    patient_pids = list(patient_data.keys())
    valid_pids_in_batch = patient_pids # Assuming no minibatching for simplicity of revert, can be added back

    for epoch in range(n_epochs):

        # --- Optimize polynomial parameters (w) using LBFGS ---
        opt_w = optim.LBFGS(model.parameters(), max_iter=max_iter_w, lr=lr_w)

        def closure_w():
            opt_w.zero_grad()
            all_s_values = []
            all_y_values = []
            for pid in valid_pids_in_batch:
                dat = patient_data[pid]
                a = 4.0 * torch.sigmoid(ab[pid]['theta'][0]).item()
                b = ab[pid]['theta'][1].item()
                s_values = a * dat['t'] + b
                all_s_values.append(s_values)
                all_y_values.append(dat['y'])
            
            s_global = torch.cat(all_s_values)
            y_global = torch.cat(all_y_values)
            
            s_global_sorted, sort_indices = torch.sort(s_global)
            y_global_sorted = y_global[sort_indices]

            # --- ROBUST FIX for duplicates ---
            if s_global_sorted.numel() > 0:
                unique_mask = torch.cat([
                    torch.tensor([True]), 
                    s_global_sorted[1:] != s_global_sorted[:-1]
                ])
                unique_s = s_global_sorted[unique_mask]
                unique_y = y_global_sorted[unique_mask]
            else:
                unique_s = s_global_sorted
                unique_y = y_global_sorted
            
            loss = calculate_global_loss(
                model, unique_s, unique_y, y0_global,
                sigma=sigma
            )
        
            loss.backward()
            return loss

        loss_w = opt_w.step(closure_w)

        # --- Optimize DPS parameters (theta) using LBFGS ---
        dps_params = [ab[pid]['theta'] for pid in valid_pids_in_batch]
        opt_dps = optim.LBFGS(dps_params, max_iter=max_iter_dps, lr=lr_dps)
        
        def closure_dps():
            opt_dps.zero_grad()
            
            all_s_for_epoch = []
            pid_to_s_vals = {}
            pid_to_alpha_beta = {}

            # First, compute all s_vals and alpha/beta for the current epoch state
            for pid in valid_pids_in_batch:
                dat = patient_data[pid]
                theta = ab[pid]['theta']
                alpha = 4.0 * torch.sigmoid(theta[0])
                beta = theta[1]
                s_vals = alpha * dat['t']
                all_s_for_epoch.append(s_vals)
                pid_to_s_vals[pid] = s_vals
                pid_to_alpha_beta[pid] = (alpha, beta)

            # Calculate variance penalty across all patients
            s_global = torch.cat(all_s_for_epoch)
            s_variance = torch.var(s_global)
            variance_loss = -variance_penalty_weight * s_variance

            # Calculate other loss components
            all_y_values = [patient_data[pid]['y'] for pid in valid_pids_in_batch]
            y_global = torch.cat(all_y_values)
            s_global_sorted, sort_indices = torch.sort(s_global)
            y_global_sorted = y_global[sort_indices]

            # --- ROBUST FIX for duplicates ---
            if s_global_sorted.numel() > 0:
                unique_mask = torch.cat([
                    torch.tensor([True]), 
                    s_global_sorted[1:] != s_global_sorted[:-1]
                ])
                unique_s = s_global_sorted[unique_mask]
                unique_y = y_global_sorted[unique_mask]
            else:
                unique_s = s_global_sorted
                unique_y = y_global_sorted

            mse_loss = calculate_global_loss(
                model, unique_s, unique_y, y0_global,
                sigma=sigma
            )
            
            inducement_loss = 0.0
            boundary_penalty = 0.0
            for pid in valid_pids_in_batch:
                s_vals = pid_to_s_vals[pid]
                alpha, beta = pid_to_alpha_beta[pid]
                
                # Inducement term
                z_pid = pid_to_score[pid]
                inducement_loss += -z_pid * (alpha + beta)

                # Boundary penalty
                below = torch.clamp(-10.0 - s_vals, min=0.0)
                above = torch.clamp(s_vals - 20.0, min=0.0)
                boundary_penalty += (below.pow(2) + above.pow(2)).mean()

            total_loss = (mse_loss + 
                          inducement_weight * inducement_loss + 
                          boundary_penalty * boundary_penalty_weight + 
                          variance_loss)

            total_loss.backward()
            return total_loss

        loss_dps = opt_dps.step(closure_dps)
        
        print(f"Epoch {epoch+1:02d}/{n_epochs} | Loss_w={loss_w.item():.4f} | Loss_dps={loss_dps.item():.4f}")

        with torch.no_grad():
            all_s_values = []
            all_y_values = []
            for pid in valid_pids_in_batch:
                dat = patient_data[pid]
                a = 4.0 * torch.sigmoid(ab[pid]['theta'][0]).item()
                b = ab[pid]['theta'][1].item()
                s_values = a * dat['t'] + b
                all_s_values.append(s_values)
                all_y_values.append(dat['y'])
            
            s_global = torch.cat(all_s_values)
            y_global = torch.cat(all_y_values)
            s_global_sorted, sort_indices = torch.sort(s_global)
            y_global_sorted = y_global[sort_indices]
            
            # --- ROBUST FIX for duplicates ---
            if s_global_sorted.numel() > 0:
                unique_mask = torch.cat([
                    torch.tensor([True]), 
                    s_global_sorted[1:] != s_global_sorted[:-1]
                ])
                unique_s = s_global_sorted[unique_mask]
                unique_y = y_global_sorted[unique_mask]
            else:
                unique_s = s_global_sorted
                unique_y = y_global_sorted

            if unique_s.numel() > 0:
                y_pred = model(unique_s, y0_global)
                new_sigma = (y_pred - unique_y) ** 2
                if torch.any(new_sigma):
                    sigma = new_sigma.mean(dim=0)
                else:
                    sigma = torch.ones(4)

    model.eval()
    return model, ab


model, trained_ab = fit_population(patient_data, y0_cn_avg, n_epochs=10, max_iter_w=20, max_iter_dps=20)

try:
    torch.save(model.state_dict(), f'{name}.pth')
    torch.save(trained_ab, f'dps_{name}.pth')
    print(f"Successfully saved model to {name}.pth and dps parameters to dps_{name}.pth")
except Exception as e:
    print(f"Error saving model: {e}")


# ---------- 绘制人群四联图 (根据s的5%和95%分位数) -----------------
with torch.no_grad():
    ab = trained_ab  # Use the trained DPS parameters

    all_s_values = []
    for p in patient_data:
        a = 4.0 * torch.sigmoid(ab[p]['theta'][0]).item()
        b = ab[p]['theta'][1]
        s_values = a * patient_data[p]['t'] + b
        all_s_values.append(s_values)
    
    all_s_flat = torch.cat(all_s_values)
    s_min = all_s_flat.min().item()
    s_max = all_s_flat.max().item()

    print(f"Plotting s value range: [{s_min:.2f}, {s_max:.2f}]")

    s_curve = torch.linspace(s_min, s_max, 100)

    keep = patient_data.keys()

    stage_dict = pc.load_stage_dict()

    y0_pop = y0_cn_avg

    # 仅多项式轨迹
    y_curve_poly = model(s_curve, y0_pop)
    y_curve_poly = y_curve_poly.detach().numpy()
    y_curve_poly = pc.inv_nor(y_curve_poly)

    TITLES = ['Aβ (A)', 'p-Tau (T)', 'N', 'Cognition (C)']

    fig2, axes = plt.subplots(2, 2, figsize=(9, 6))
    for k, ax in enumerate(axes.flat):
        # --- 分阶段准备散点数据 ---
        s_by_stage = {'CN': [], 'LMCI': [], 'AD': [], 'Other': []}
        y_by_stage = {'CN': [], 'LMCI': [], 'AD': [], 'Other': []}

        for p in keep:
            a = 4.0 * torch.sigmoid(ab[p]['theta'][0]).item()
            b = ab[p]['theta'][1]
            stage = stage_dict.get(p, 'Other')
            if stage not in s_by_stage:
                stage = 'Other'

            s_values = a * patient_data[p]['t'] + b
            y_values = patient_data[p]['y'][:, k]

            s_by_stage[stage].append(s_values)
            y_by_stage[stage].append(y_values)

        colors = {'CN': 'orange', 'LMCI': 'green', 'AD': 'blue', 'Other': 'grey'}
        for stage, s_points_list in s_by_stage.items():
            if s_points_list:
                s_all = torch.cat(s_points_list).numpy()
                y_all = torch.cat(y_by_stage[stage]).numpy()
                y_all = pc.inv_nor(y_all, k)
                ax.scatter(s_all, y_all, s=15, alpha=0.6, c=colors[stage], label=stage)

        ax.plot(s_curve, y_curve_poly[:, k], lw=1.6, c='black', label='Poly only')

        ax.set_xlabel('Disease progression score  s')
        ax.set_ylabel(TITLES[k])
        ax.legend(fontsize=8)

    fig2.suptitle(f'Population Model (Poly only) (s range: [{float(s_min):.2f}, {float(s_max):.2f}])')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f'{name}.png')
    plt.show()

    def eval_global_loss(y_pred, y_true):
        y_pred_t = torch.as_tensor(y_pred)
        return torch.mean((y_pred_t - y_true) ** 2)

    loss = 0
    with torch.no_grad():
        for pid in patient_data:
            a = 4.0 * torch.sigmoid(ab[pid]['theta'][0]).item()
            b = ab[pid]['theta'][1]
            s = a * patient_data[pid]['t'] + b
            y_pred = model(s, patient_data[pid]['y0'])
            y_pred = y_pred.numpy()
            loss += eval_global_loss(y_pred, patient_data[pid]['y']) / len(y_pred)
        loss /= len(patient_data)

print("\n--- Trained ODE Model Equations ---")
with torch.no_grad():
    # A
    rA = (_pos(model.rA) + 0.01).item()
    KA = _pos(model.KA).item()
    print(f"dA/ds = {rA:.4f}*A*(1 - A/{KA:.4f})")

    # T
    rT = (_pos(model.rT) + 0.01).item()
    KT = _pos(model.KT).item()
    at2 = _pos(model.at2).item()
    at1 = _pos(model.at1).item()
    print(f"dT/ds = {rT:.4f}*T*(1 - T/{KT:.4f}) + {at2:.4f}*A^2 - {at1:.4f}*A*T")

    # N
    rN = (_pos(model.rN) + 0.01).item()
    KN = _pos(model.KN).item()
    tt2 = _pos(model.tt2).item()
    tn1 = _pos(model.tn1).item()
    print(f"dN/ds = {rN:.4f}*N*(1 - N/{KN:.4f}) + {tt2:.4f}*T^2 - {tn1:.4f}*T*N")

    # C
    rC = (_pos(model.rC) + 0.01).item()
    KC = _pos(model.KC).item()
    nc1 = _pos(model.nc1).item()
    print(f"dC/ds = {rC:.4f}*C*(1 - C/{KC:.4f}) - {nc1:.4f}*N*C")
print("---------------------------------\n")