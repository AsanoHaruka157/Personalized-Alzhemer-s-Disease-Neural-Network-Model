import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

try:
    import torchbnn as bnn
except ImportError:
    raise ImportError("請安裝 torchbnn 以運行此腳本: pip install torchbnn")

try:
    from torchdiffeq import odeint as torch_odeint
except ImportError:
    raise ImportError("請安裝 torchdiffeq 以運行此腳本: pip install torchdiffeq")

import pccmnn as pc

# --- 0. 數據加載和準備 ---
try:
    csf_dict = pc.load_data()
    stage_dict = pc.load_stage_dict()
    print(f"成功加載 {len(csf_dict)} 位患者的數據。")
except Exception as e:
    print(f"錯誤：無法加載數據，請確保 `pccmnn.py` 和相關數據文件存在。錯誤訊息: {e}")
    exit()

# 轉換數據格式
patient_data = {
    pid: {
        "t": torch.from_numpy(sample[:, 0]).float(),
        "y": torch.from_numpy(sample[:, 1:5]).float(),
        "y0": torch.from_numpy(sample[0, 1:5]).float(),
        "stage": stage_dict.get(pid, 'Other')
    }
    for pid, sample in csf_dict.items()
}

# --- 1. 加載DPS並準備訓練數據 ---
def load_dps_and_create_scatter(csf_dict, dps_path='dps.pth'):
    print(f"正在從 {dps_path} 加載 DPS 參數...")
    try:
        dps_params = torch.load(dps_path, weights_only=False)
    except FileNotFoundError:
        print(f"錯誤: 未找到 {dps_path}。請先運行 pretrain.py 生成此文件。")
        exit()

    all_s, all_y, all_stages = [], [], []
    for pid, sample in csf_dict.items():
        if pid in dps_params:
            stage = stage_dict.get(pid, 'Other')
            t, y = sample[:, 0], sample[:, 1:5]
            a, b = dps_params[pid]['a'], dps_params[pid]['b']
            s = a * t + b
            all_s.append(s)
            all_y.append(y)
            all_stages.extend([stage] * len(t))
            
    print("成功使用加載的 DPS 參數生成散點數據。")
    return np.concatenate(all_s), np.concatenate(all_y), all_stages

# --- 2. Sigmoid 擬合函數 ---
def sigmoid(s, a, b, c, d):
    return a / (1.0 + np.exp(-b * (s - c))) + d

def fit_sigmoids(s_data, y_data):
    params = [curve_fit(sigmoid, s_data, y_data[:, k], maxfev=10000)[0] for k in range(4)]
    return np.array(params)

def get_sigmoid_derivatives(s_grid, params):
    y = np.zeros((len(s_grid), 4))
    dyds = np.zeros((len(s_grid), 4))
    for k in range(4):
        a, b, c, d = params[k]
        exp_term = np.exp(-b * (s_grid - c))
        y[:, k] = a / (1.0 + exp_term) + d
        dyds[:, k] = (a * b * exp_term) / ((1.0 + exp_term)**2)
    return y, dyds

# --- 3. BNN 模型定義與預訓練 ---
class BNN(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=128, output_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            bnn.BayesLinear(prior_mu=0, prior_sigma=0.01, in_features=input_dim, out_features=hidden_dim),
            nn.ReLU(),
            bnn.BayesLinear(prior_mu=0, prior_sigma=0.01, in_features=hidden_dim, out_features=hidden_dim),
            nn.ReLU(),
            bnn.BayesLinear(prior_mu=0, prior_sigma=0.01, in_features=hidden_dim, out_features=hidden_dim),
            nn.ReLU(),
            bnn.BayesLinear(prior_mu=0, prior_sigma=0.01, in_features=hidden_dim, out_features=output_dim),
            nn.Tanh(),
        )
        self.output_scaler = nn.Parameter(torch.tensor([0.1]), requires_grad=True)

    def forward(self, y):
        return self.net(y) * self.output_scaler

def pretrain_bnn_on_sigmoid(y_target, dyds_target, n_epochs=500, lr=1e-3):
    print("\n--- 階段一：在 Sigmoid 導數上預訓練 BNN (使用 Adam) ---")
    model = BNN()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    mse_loss = nn.MSELoss()
    kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)
    kl_weight = 0.01

    y_tensor = torch.tensor(y_target, dtype=torch.float32)
    dyds_tensor = torch.tensor(dyds_target, dtype=torch.float32)
    
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        dyds_pred = model(y_tensor)
        mse = mse_loss(dyds_pred, dyds_tensor)
        kl = kl_loss(model)
        loss = mse + kl_weight * kl
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 50 == 0:
            print(f"  Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.8f}, MSE: {mse.item():.8f}, KL: {kl.item():.8f}")
            
    model.eval()
    return model

# --- 4. 基於真實數據的 L-BFGS 微調 ---
class ODEModel(nn.Module):
    def __init__(self, bnn_model):
        super().__init__()
        self.bnn = bnn_model
    def forward(self, t, y):
        return self.bnn(y)

def calculate_loss(ode_model, patient_data, ab, pids, y0, kl_loss_func, kl_weight):
    all_s, all_y = [], []
    for pid in pids:
        dat = patient_data[pid]
        s_values = ab[pid]['a'] * dat['t'] + ab[pid]['b']
        all_s.append(s_values)
        all_y.append(dat['y'])
    
    s_global, y_global = torch.cat(all_s), torch.cat(all_y)
    s_sorted, sort_indices = torch.sort(s_global)
    y_sorted = y_global[sort_indices]
    
    try:
        y_pred = torch_odeint(ode_model, y0, s_sorted, method='dopri5', rtol=1e-4, atol=1e-5)
        mse = ((y_pred - y_sorted) ** 2).mean()
        kl = kl_loss_func(ode_model.bnn)
        loss = mse + kl_weight * kl
        return loss if torch.isfinite(loss) else torch.tensor(float('inf'))
    except Exception:
        return torch.tensor(float('inf'))

def train_bnn_on_data(initial_bnn, patient_data, y0, dps_path='dps.pth', n_epochs=10, lr=1e-4, max_iter_lbfgs=20):
    print("\n--- 在真實數據上使用 L-BFGS 訓練 BNN 和 DPS 參數 ---")
    ode_model = ODEModel(initial_bnn).train()
    kl_loss_func = bnn.BKLLoss(reduction='mean', last_layer_only=False)
    kl_weight = 0.01
    
    dps_params_loaded = torch.load(dps_path, weights_only=False)
    ab = {
        pid: {
            'a': nn.Parameter(torch.tensor(dps_params_loaded[pid]['a'], dtype=torch.float32)),
            'b': nn.Parameter(torch.tensor(dps_params_loaded[pid]['b'], dtype=torch.float32))
        }
        for pid in patient_data if pid in dps_params_loaded
    }
    patient_pids = list(ab.keys())

    dps_params = [p for pid in patient_pids for p in ab[pid].values()]
    optimizer = optim.LBFGS(
        list(ode_model.parameters()) + dps_params,
        lr=lr, max_iter=max_iter_lbfgs, line_search_fn="strong_wolfe"
    )

    for epoch in range(n_epochs):
        def closure():
            optimizer.zero_grad()
            loss = calculate_loss(ode_model, patient_data, ab, patient_pids, y0, kl_loss_func, kl_weight)
            if torch.isfinite(loss):
                loss.backward()
            return loss
        
        loss = optimizer.step(closure)
        print(f"  Epoch [{epoch+1}/{n_epochs}], L-BFGS Loss: {loss.item():.8f}")

    ode_model.eval()
    return ode_model, ab

# --- 5. 繪圖與保存 ---
def plot_pretrain_results(s_pop, y_pop_orig, stages_pop, s_grid, sigmoid_params, pretrained_bnn, y0):
    print("\n正在生成預訓練結果圖...")
    ode_model = ODEModel(pretrained_bnn).eval()
    
    with torch.no_grad():
        y_fnn_traj_norm = torch_odeint(ode_model, y0, torch.from_numpy(s_grid).float()).numpy()
    
    y_sigmoid_grid_norm, _ = get_sigmoid_derivatives(s_grid, sigmoid_params)
    y_sigmoid_grid_orig = pc.inv_nor(y_sigmoid_grid_norm)
    y_fnn_traj_orig = pc.inv_nor(y_fnn_traj_norm)

    TITLES = ['Aβ (A)', 'p-Tau (T)', 'N', 'Cognition (C)']
    colors = {'CN': 'orange', 'LMCI': 'green', 'AD': 'blue', 'Other': 'grey'}
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    axes = axes.flat
    
    for k in range(4):
        ax = axes[k]
        for stage in np.unique(stages_pop):
            mask = np.array(stages_pop) == stage
            ax.scatter(s_pop[mask], y_pop_orig[mask, k], s=15, alpha=0.4, c=colors[stage], label=f'{stage}')
        
        ax.plot(s_grid, y_sigmoid_grid_orig[:, k], 'r--', lw=2.5, label='Sigmoid Fit')
        ax.plot(s_grid, y_fnn_traj_orig[:, k], 'k-', lw=2.5, label='Pre-trained BNN Trajectory')
        ax.set(xlabel='Disease Progression Score (s)', ylabel=TITLES[k], title=TITLES[k])
        ax.legend()
        ax.grid(True, alpha=0.4)
    
    fig.suptitle('BNN Pre-training Result vs. Sigmoid Fit', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_and_save(s_pop, y_pop_orig, stages_pop, s_grid, final_model, y0, sigmoid_params, model_save_path='bnn.pth'):
    print("\n正在生成最終對比圖並保存模型...")
    torch.save(final_model.state_dict(), model_save_path)
    print(f"模型已成功保存到 {model_save_path}")

    n_samples = 100
    s_grid_tensor = torch.from_numpy(s_grid).float()

    print(f"從BNN後驗分佈中採樣 {n_samples} 次以進行不確定性量化...")
    pred_trajectories = [
        torch_odeint(final_model, y0, s_grid_tensor, method='dopri5').detach().numpy()
        for _ in range(n_samples)
    ]

    pred_trajectories = np.array(pred_trajectories)
    mean_pred_norm = np.mean(pred_trajectories, axis=0)
    ci_norm = np.percentile(pred_trajectories, [5, 95], axis=0)

    mean_pred_orig = pc.inv_nor(mean_pred_norm)
    ci_orig = pc.inv_nor(ci_norm)
    
    y_sigmoid_grid_norm, _ = get_sigmoid_derivatives(s_grid, sigmoid_params)
    y_sigmoid_grid_orig = pc.inv_nor(y_sigmoid_grid_norm)

    print("正在生成最终图表...")
    TITLES = ['Aβ (A)', 'p-Tau (T)', 'N', 'Cognition (C)']
    colors = {'CN': 'orange', 'LMCI': 'green', 'AD': 'blue', 'Other': 'grey'}
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    axes = axes.flat
    
    for k in range(4):
        ax = axes[k]
        for stage in np.unique(stages_pop):
            mask = np.array(stages_pop) == stage
            ax.scatter(s_pop[mask], y_pop_orig[mask, k], s=15, alpha=0.4, c=colors[stage], label=f'{stage}')
        
        ax.plot(s_grid, mean_pred_orig[:, k], 'k-', lw=2.5, label='Mean Trajectory')
        ax.fill_between(s_grid, ci_orig[0, :, k], ci_orig[1, :, k], color='lightgrey', alpha=0.8, label='90% CI (BNN Uncertainty)')
        ax.plot(s_grid, y_sigmoid_grid_orig[:, k], 'r--', lw=2.5, label='Sigmoid Fit')
        ax.set(xlabel='Disease Progression Score (s)', ylabel=TITLES[k], title=TITLES[k])
        ax.legend()
        ax.grid(True, alpha=0.4)
    
    fig.suptitle('BNN Model Trajectory with Uncertainty Quantification', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('bnn.png')
    plt.show()

if __name__ == '__main__':
    # 1. 準備數據
    s_pop_np, y_pop_norm_np, stages_pop_np = load_dps_and_create_scatter(csf_dict)
    y_pop_orig = pc.inv_nor(y_pop_norm_np)
    cn_y0s = [dat['y0'] for dat in patient_data.values() if dat['stage'] == 'CN']
    y0_cn_avg = torch.stack(cn_y0s).mean(dim=0) if cn_y0s else torch.tensor([0.1, 0, 0, 0])
    print(f"使用CN群體的平均初始值: {y0_cn_avg.numpy()}")

    # 2. 階段一：預訓練 BNN
    s_grid_np_pretrain = np.linspace(-10, 20, 500)
    sigmoid_p = fit_sigmoids(s_pop_np, y_pop_norm_np)
    y_sigmoid_norm, dyds_sigmoid_norm = get_sigmoid_derivatives(s_grid_np_pretrain, sigmoid_p)
    bnn_pretrained = pretrain_bnn_on_sigmoid(y_sigmoid_norm, dyds_sigmoid_norm)
    
    # 3. 繪製預訓練結果
    plot_pretrain_results(s_pop_np, y_pop_orig, stages_pop_np, s_grid_np_pretrain, sigmoid_p, bnn_pretrained, y0_cn_avg)
    
    # 4. 階段二：微調 BNN
    final_ode_model, final_ab = train_bnn_on_data(bnn_pretrained, patient_data, y0_cn_avg, dps_path='dps.pth')

    # 5. 繪圖與保存最終結果
    s_grid_np_final = np.linspace(-10, 20, 300)
    plot_and_save(s_pop_np, y_pop_orig, stages_pop_np, s_grid_np_final, final_ode_model, y0_cn_avg, sigmoid_p)
    
    print("\n完整流程執行完畢。")
