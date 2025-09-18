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

        # A self-dynamics: +a1*A - a2*A^2
        self.a1 = p(0.30)
        self.a2 = p(0.10)

        # T self-dynamics + coupling from A: +t1*T - t2*T^2 + at2*A^2 - at1*A*T
        self.t1 = p(0.25)
        self.t2 = p(0.08)
        self.at2 = p(init_small)  # A^2 -> T ( + )
        self.at1 = p(init_small)  # A*T ( - )

        # N self-dynamics + coupling from T: +n1*N - n2*N^2 + tt2*T^2 - tn1*T*N
        self.n1 = p(0.20)
        self.n2 = p(0.06)
        self.tt2 = p(init_small)  # T^2 -> N ( + )
        self.tn1 = p(init_small)  # T*N ( - )

        # C self-dynamics + impairment from N: +c1*C - c2*C^2 - nc1*N*C
        self.c1 = p(0.12)
        self.c2 = p(0.04)
        self.nc1 = p(init_small)  # N*C ( - )

    def poly(self, y: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        A, T, N, C = y[..., 0], y[..., 1], y[..., 2], y[..., 3]

        # dA/ds = +a1*A - a2*A^2
        dA = _pos(self.a1) * A - _pos(self.a2) * (A * A)

        # dT/ds = +t1*T - t2*T^2 + at2*A^2 - at1*A*T
        dT = _pos(self.t1) * T - _pos(self.t2) * (T * T) \
             + _pos(self.at2) * (A * A) - _pos(self.at1) * (A * T)

        # dN/ds = +n1*N - n2*N^2 + tt2*T^2 - tn1*T*N
        dN = _pos(self.n1) * N - _pos(self.n2) * (N * N) \
             + _pos(self.tt2) * (T * T) - _pos(self.tn1) * (T * N)

        # dC/ds = +c1*C - c2*C^2 - nc1*N*C
        dC = _pos(self.c1) * C - _pos(self.c2) * (C * C) \
             - _pos(self.nc1) * (N * C)
        
        p = torch.stack([dA, dT, dN, dC], dim=-1)
        return p

    def _rk4_integrate(self, s_grid: torch.Tensor, y0: torch.Tensor, f_fn) -> torch.Tensor:
        ys = [y0]
        for i in range(1, len(s_grid)):
            h = s_grid[i] - s_grid[i - 1]
            y_i = ys[-1]

            k1 = f_fn(y_i, s_grid[i - 1])
            k2 = f_fn(y_i + 0.5 * h * k1, s_grid[i - 1])
            k3 = f_fn(y_i + 0.5 * h * k2, s_grid[i - 1])
            k4 = f_fn(y_i + h * k3, s_grid[i - 1])

            y_next = y_i + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            ys.append(y_next)

        return torch.stack(ys)

    def forward(self, s_grid: torch.Tensor, y0: torch.Tensor) -> torch.Tensor:
        # 仅多项式动力学: dy/ds = poly(y)
        return self._rk4_integrate(s_grid, y0, lambda y, s: self.poly(y, s))


def calculate_global_loss(model, s_global, y_global, y0_global, sigma=None, tail_penalty_factor=0.0):
    # 预测整个轨迹（仅多项式）
    y_pred_global = model(s_global, y0_global)
    loss = (y_pred_global - y_global) ** 2

    if tail_penalty_factor > 0:
        s_detached = s_global.detach()
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


def fit_population(
    patient_data,
    y0_global,
    n_epochs=10,
    max_iter_w=20,
    max_iter_dps=20,
    batch_size=128,
    lr_w=5e-2,
    lr_dps=5e-2,
    weighted_sampling=True,
    inducement_weight=1e-3,
    tail_penalty_factor=50.0,
):
    sigma = torch.ones(4)

    # ---------- 初始化 ----------
    model = ODEModel()

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

    # Pre-calculate scores for inducement term
    pid_to_score = {}
    for pid, dat in patient_data.items():
        y = dat['y']
        # Score: higher means more "diseased"
        z = (-y[:, 0]) + (y[:, 1]) + (-y[:, 2]) + (y[:, 3])
        pid_to_score[pid] = float(z.mean())

    stage_dict = pc.load_stage_dict()

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
        print("Warning: dps.pth not found. Computing a,b from age to stage mapping.")
        ab = {}
        for pid, dat in patient_data.items():
            stage = stage_dict.get(pid, 'Other')
            t = dat['t']
            if stage == 'CN':
                s_range = (-10.0, 0.0)
            elif stage == 'LMCI':
                s_range = (0.0, 10.0)
            elif stage == 'AD':
                s_range = (10.0, 20.0)
            else:
                s_range = (-5.0, 5.0)
            t_min, t_max = t.min().item(), t.max().item()
            s_min, s_max = s_range[0], s_range[1]
            if abs(t_max - t_min) < 1e-6:
                a = 1.0
                b = s_min - a * t_min
            else:
                a = (s_max - s_min) / (t_max - t_min)
                b = s_min - a * t_min
            a = max(a, 1e-4)
            # theta0 = torch.log(torch.tensor(a - 1e-4)) # This is inverse of softplus: log(exp(y)-1) which is complicated.
            # a = softplus(theta0)+eps. Let's use an approximation, assuming a is small.
            # a is close to softplus(a).
            theta0 = torch.tensor(a)
            theta1 = torch.tensor(b)
            ab[pid] = {'theta': torch.tensor([theta0.item(), theta1.item()], requires_grad=True)}

    # --------- 训练循环 -----------
    for epoch in range(n_epochs):
        if use_minibatch:
            try:
                batch_pids = next(batch_iterator)
            except StopIteration:
                print("Batch iterator exhausted.")
                break
        else:
            batch_pids = patient_pids

        batch_pids_list = [pid.item() for pid in batch_pids] if use_minibatch else batch_pids
        valid_pids_in_batch = [pid for pid in batch_pids_list if patient_data[pid]['t'].shape[0] >= 2 and pid in ab]
        if not valid_pids_in_batch:
            continue
        
        # --- Optimize polynomial parameters (w) ---
        opt_w = optim.LBFGS(model.parameters(), max_iter=max_iter_w, lr=lr_w)

        def closure_w():
            opt_w.zero_grad()
            all_s_values = []
            all_y_values = []
            for pid in valid_pids_in_batch:
                dat = patient_data[pid]
                a = F.softplus(ab[pid]['theta'][0]).item() + 1e-4
                b = ab[pid]['theta'][1].item()
                s_values = a * dat['t'] + b
                y_values = dat['y']
                all_s_values.append(s_values)
                all_y_values.append(y_values)

            s_global = torch.cat(all_s_values)
            y_global = torch.cat(all_y_values)
            s_global_sorted, sort_indices = torch.sort(s_global)
            y_global_sorted = y_global[sort_indices]

            s_5_percentile = torch.quantile(s_global_sorted, 0.05)
            s_95_percentile = torch.quantile(s_global_sorted, 0.95)
            mask = (s_global_sorted >= s_5_percentile) & (s_global_sorted <= s_95_percentile)
            s_global_filtered = s_global_sorted[mask]
            y_global_filtered = y_global_sorted[mask]

            loss = calculate_global_loss(model, s_global_filtered, y_global_filtered, y0_global, sigma=sigma, tail_penalty_factor=tail_penalty_factor)
            if torch.isnan(loss):
                return loss
            loss.backward()
            return loss

        loss_w = opt_w.step(closure_w)

        # --- Optimize DPS parameters (theta) ---
        dps_params = [ab[pid]['theta'] for pid in valid_pids_in_batch]
        opt_dps = optim.LBFGS(dps_params, max_iter=max_iter_dps, lr=lr_dps)
        
        def closure_dps():
            opt_dps.zero_grad()
            all_s_values = []
            all_y_values = []
            induce_loss = 0.0
            for pid in valid_pids_in_batch:
                dat = patient_data[pid]
                a = F.softplus(ab[pid]['theta'][0]) + 1e-4
                b = ab[pid]['theta'][1]
                s_values = a * dat['t'] + b
                y_values = dat['y']
                all_s_values.append(s_values)
                all_y_values.append(y_values)

                z_pid = pid_to_score[pid]
                induce_loss += -z_pid * (a + b)


            s_global = torch.cat(all_s_values)
            y_global = torch.cat(all_y_values)
            s_global_sorted, sort_indices = torch.sort(s_global)
            y_global_sorted = y_global[sort_indices]
            
            s_5_percentile = torch.quantile(s_global_sorted, 0.05)
            s_95_percentile = torch.quantile(s_global_sorted, 0.95)
            mask = (s_global_sorted >= s_5_percentile) & (s_global_sorted <= s_95_percentile)
            s_global_filtered = s_global_sorted[mask]
            y_global_filtered = y_global_sorted[mask]

            mse_loss = calculate_global_loss(model, s_global_filtered, y_global_filtered, y0_global, sigma=sigma, tail_penalty_factor=tail_penalty_factor)
            
            total_loss = mse_loss + inducement_weight * induce_loss

            if torch.isnan(total_loss):
                return total_loss
            total_loss.backward()
            return total_loss

        loss_dps = opt_dps.step(closure_dps)
        
        print(f"Epoch {epoch+1:02d}/{n_epochs} | Loss_w={loss_w.item():.4f} | Loss_dps={loss_dps.item():.4f}")

        with torch.no_grad():
            # This logic needs to be inside the closure to be part of the graph, but let's update sigma once per epoch
            all_s_values = []
            all_y_values = []
            for pid in valid_pids_in_batch:
                dat = patient_data[pid]
                a = F.softplus(ab[pid]['theta'][0]).item() + 1e-4
                b = ab[pid]['theta'][1].item()
                s_values = a * dat['t'] + b
                all_s_values.append(s_values)
                all_y_values.append(dat['y'])
            s_global = torch.cat(all_s_values)
            y_global = torch.cat(all_y_values)
            s_global_sorted, sort_indices = torch.sort(s_global)
            y_global_sorted = y_global[sort_indices]
            s_5_percentile = torch.quantile(s_global_sorted, 0.05)
            s_95_percentile = torch.quantile(s_global_sorted, 0.95)
            mask = (s_global_sorted >= s_5_percentile) & (s_global_sorted <= s_95_percentile)
            s_global_filtered = s_global_sorted[mask]
            y_global_filtered = y_global_sorted[mask]
            
            y_pred = model(s_global_filtered, y0_global)
            new_sigma = (y_pred - y_global_filtered) ** 2
            if torch.any(new_sigma):
                sigma = new_sigma.mean(dim=0)
            else:
                sigma = torch.ones(4)


    model.eval()
    return model


model = fit_population(patient_data, y0_cn_avg)

try:
    torch.save(model.state_dict(), f'{name}.pth')
except Exception as e:
    print(f"Error saving model: {e}")


# ---------- 绘制人群四联图 (根据s的10%和90%分位数) -----------------
with torch.no_grad():
    # 收集所有患者的s值
    ab = torch.load('dps.pth')
    all_s_values = []
    for p in patient_data:
        a = F.softplus(ab[p]['theta'][0]).item() + 1e-4
        b = ab[p]['theta'][1]
        s_values = a * patient_data[p]['t'] + b
        all_s_values.append(s_values)

    all_s_flat = torch.cat(all_s_values)
    s_10_percentile = torch.quantile(all_s_flat, 0.10)
    s_90_percentile = torch.quantile(all_s_flat, 0.90)

    print(f"S value range: 10th percentile = {s_10_percentile:.2f}, 90th percentile = {s_90_percentile:.2f}")

    s_min = s_10_percentile
    s_max = s_90_percentile
    s_curve = torch.linspace(s_min, s_max, 100)

    keep = []
    for p in patient_data:
        a = F.softplus(ab[p]['theta'][0]).item() + 1e-4
        b = ab[p]['theta'][1]
        s_values = a * patient_data[p]['t'] + b
        if torch.any((s_values >= s_min) & (s_values <= s_max)):
            keep.append(p)

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
            a = F.softplus(ab[p]['theta'][0]).item() + 1e-4
            b = ab[p]['theta'][1]
            stage = stage_dict.get(p, 'Other')
            if stage not in s_by_stage:
                stage = 'Other'

            s_values = a * patient_data[p]['t'] + b
            y_values = patient_data[p]['y'][:, k]

            mask = (s_values >= s_min) & (s_values <= s_max)
            if torch.any(mask):
                s_by_stage[stage].append(s_values[mask])
                y_by_stage[stage].append(y_values[mask])

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

    fig2.suptitle(f'Population Model (Poly only) (s in 10-90 percentile: [{float(s_min):.2f}, {float(s_max):.2f}])')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f'{name}.png')
    plt.show()

    def eval_global_loss(y_pred, y_true):
        y_pred_t = torch.as_tensor(y_pred)
        return torch.mean((y_pred_t - y_true) ** 2)

    loss = 0
    with torch.no_grad():
        for pid in patient_data:
            a = F.softplus(ab[pid]['theta'][0]).item() + 1e-4
            b = ab[pid]['theta'][1]
            s = a * patient_data[pid]['t'] + b
            y_pred = model(s, patient_data[pid]['y0'])
            y_pred = y_pred.numpy()
            loss += eval_global_loss(y_pred, patient_data[pid]['y']) / len(y_pred)
        loss /= len(patient_data)

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    output_filename = 'experiments.out'
    with open(output_filename, 'a') as f:
        f.write(name)
        f.write(f"Time: {current_time}\n")
        f.write(Message)
        f.write("Model structure:\n")
        f.write(str(model))
        f.write(f"MSE: {loss:.4f}")


