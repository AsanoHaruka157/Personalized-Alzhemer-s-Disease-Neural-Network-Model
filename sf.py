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
csf_dict = pc.load_csf_data()
print("Number of valid patients:", len(csf_dict))

patient_data = {}
for pid, sample in csf_dict.items():
    t = torch.from_numpy(sample[:, 0]).float()            # 年龄
    y = torch.from_numpy(sample[:, 1:5]).float()          # biomarker A/T/N/C
    patient_data[pid] = {"t": t, "y": y, "y0": y[0].clone()}


class Gaussian(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.exp(-torch.pow(x, 2))


# ---------- 计算y的分位数并保存为全局变量B, C, D ----------
all_y_data = torch.cat([dat['y'] for dat in patient_data.values()], dim=0)
B = torch.quantile(all_y_data, 0.05, dim=0)
C = torch.quantile(all_y_data, 0.50, dim=0)
D = torch.quantile(all_y_data, 0.95, dim=0)

Message = f"FNN-only model with fixed pretrained DPS parameters."
name = 'sf'


class ODEModel(nn.Module):
    def __init__(self, hidden_dim=64, num_layers=3):
        super().__init__()
        layers = []
        input_dim = 4
        output_dim = 4
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def f(self, y: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        return self.net(y)

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
        # 仅FNN动力学: dy/ds = net(y)
        return self._rk4_integrate(s_grid, y0, self.f)


def calculate_global_loss(model, s_global, y_global, sigma=None):
    # 从s=-10, y0=[0.1,0,0,0]开始预测
    y0_global = torch.tensor([0.1, 0, 0, 0])
    # 预测整个轨迹（仅FNN）
    y_pred_global = model(s_global, y0_global)
    loss = (y_pred_global - y_global) ** 2
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
    n_adam=300,
    batch_size=128,
    opt_w_lr=1e-3,
    weighted_sampling=True,
    early_stop_patience=80,
    early_stop_threshold=0.001,
):
    sigma = torch.ones(4)

    # ---------- 初始化 ----------
    model = ODEModel(hidden_dim=64, num_layers=3)

    def weights_init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0, std=0.005)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    model.apply(weights_init)

    # --------- Minibatch Dataloader Setup -----------
    patient_pids = list(patient_data.keys())
    use_minibatch = batch_size < len(patient_pids)

    batch_iterator = None
    if use_minibatch and n_adam > 0:
        print(f"Using mini-batching with batch size {batch_size}.")
        dataset = PatientDataset(patient_pids)

        sampler = None
        if weighted_sampling:
            print("Using weighted sampling based on the number of time points per patient.")
            weights = [float(patient_data[pid]['t'].shape[0]) for pid in patient_pids]
            sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=n_adam * batch_size, replacement=True)

        dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, shuffle=(sampler is None))
        batch_iterator = iter(dataloader)

    stage_dict = pc.load_stage_dict()

    # 加载预训练的DPS参数
    try:
        ab = torch.load('dps.pth')
        print("Successfully loaded pretrained DPS parameters from dps.pth")
        for pid in ab:
            if 'theta' in ab[pid]:
                ab[pid]['theta'] = ab[pid]['theta'].detach().requires_grad_(False)
            else:
                alpha, beta = ab[pid]
                ab[pid] = {'theta': alpha, 'beta': beta}
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
            theta0 = torch.log(torch.tensor(a - 1e-4))
            theta1 = torch.tensor(b)
            ab[pid] = {'theta': torch.tensor([theta0.item(), theta1.item()], requires_grad=False)}

    # --------- 优化器 -----------
    opt_w = optim.Adam(model.parameters(), lr=opt_w_lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(opt_w, milestones=list(range(60, 151)), gamma=0.99, last_epoch=-1)

    # --------- 训练循环 -----------
    training_stopped = False
    early_stop_counter = 0
    last_adam_loss = float('inf')
    adam_loss_history = []

    for it in range(n_adam):
        if use_minibatch:
            try:
                batch_pids = next(batch_iterator)
            except StopIteration:
                print("Batch iterator exhausted. This should not happen with the configured sampler.")
                continue
        else:
            batch_pids = patient_pids

        opt_w.zero_grad()
        # Convert tensor PIDs to integer for dict lookup
        batch_pids_list = [pid.item() for pid in batch_pids] if use_minibatch else batch_pids
        valid_pids_in_batch = [pid for pid in batch_pids_list if patient_data[pid]['t'].shape[0] >= 2]
        if not valid_pids_in_batch:
            continue

        all_s_values = []
        all_y_values = []
        for pid in valid_pids_in_batch:
            dat = patient_data[pid]
            a = F.softplus(ab[pid]['theta'][0]).item() + 1e-4
            b = ab[pid]['theta'][1]
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

        loss_w = calculate_global_loss(model, s_global_filtered, y_global_filtered, sigma=sigma)

        adam_loss_history.append(loss_w.item())
        if len(adam_loss_history) > 30:
            adam_loss_history.pop(0)
        current_avg_loss = sum(adam_loss_history) / len(adam_loss_history)

        if last_adam_loss != float('inf'):
            relative_change = (last_adam_loss - current_avg_loss) / abs(last_adam_loss) if last_adam_loss != 0 else float('inf')
            if relative_change < early_stop_threshold:
                early_stop_counter += 1
            else:
                early_stop_counter = 0
        last_adam_loss = current_avg_loss

        if early_stop_counter >= early_stop_patience:
            print(f"Iter {it+1:02d}: Early stopping. Loss improvement < {early_stop_threshold*100:.1f}% for {early_stop_patience} steps.")
            break

        if torch.isnan(loss_w):
            print(f"Iter {it+1:02d}: Adam loss is NaN. Stopping training.")
            training_stopped = True
        else:
            loss_w.backward()
            opt_w.step()
            scheduler.step()

        if training_stopped:
            break

        with torch.no_grad():
            y0 = torch.tensor([0.1, 0, 0, 0])
            y_pred = model(s_global_filtered, y0)
            sigma = (y_pred - y_global_filtered) ** 2
            if torch.any(sigma):
                sigma = sigma.mean(dim=0)
            else:
                sigma = torch.ones(4)

        if (it + 1) % 50 == 0:
            print(f"iter {it+1:02d}/{n_adam} | Adam | Batch MSE={loss_w.item():.4f} | Avg MSE={current_avg_loss:.4f}")

    model.eval()
    return model


model = fit_population(patient_data)

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

    y0_pop = torch.tensor([0.1, 0, 0, 0])

    # 仅FNN轨迹
    y_curve_net = model(s_curve, y0_pop)
    y_curve_net = y_curve_net.detach().numpy()
    y_curve_net = pc.inv_nor(y_curve_net)

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

        ax.plot(s_curve, y_curve_net[:, k], lw=1.6, c='purple', label='Net only')

        ax.set_xlabel('Disease progression score  s')
        ax.set_ylabel(TITLES[k])
        ax.legend(fontsize=8)

    fig2.suptitle(f'Population Model (Net only) (s in 10-90 percentile: [{float(s_min):.2f}, {float(s_max):.2f}])')
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

