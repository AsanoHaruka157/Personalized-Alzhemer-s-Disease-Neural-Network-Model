import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
try:
    from torchdiffeq import odeint as torch_odeint
except ImportError:
    raise ImportError("請安裝 torchdiffeq 以運行此腳本: pip install torchdiffeq")
import pccmnn as pc # 確保 pccmnn.py 在您的工作目錄中

# --- 0. 數據加載和準備 ---
try:
    csf_dict = pc.load_data()
    stage_dict = pc.load_stage_dict()
    print(f"成功加載 {len(csf_dict)} 位患者的數據。")
except Exception as e:
    print(f"錯誤：無法加載數據，請確保 `pccmnn.py` 和相關數據文件存在。錯誤訊息: {e}")
    exit()

# 轉換數據格式以適應 PyTorch
patient_data = {}
for pid, sample in csf_dict.items():
    t = torch.from_numpy(sample[:, 0]).float()
    y = torch.from_numpy(sample[:, 1:5]).float()
    patient_data[pid] = {"t": t, "y": y, "y0": y[0].clone(), "stage": stage_dict.get(pid, 'Other')}

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
            
            a = dps_params[pid]['a']
            b = dps_params[pid]['b']
            
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


# --- 3. FNN 模型定義與預訓練 ---
class FNN(nn.Module):
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
            nn.Tanh(),
        )
        self.output_scaler = nn.Parameter(torch.tensor([0.1]), requires_grad=True)

    def forward(self, y):
        return self.net(y) * self.output_scaler


def pretrain_fnn_on_sigmoid(y_target, dyds_target, n_epochs=500, lr=1e-2):
    print("\n--- 階段一：在 Sigmoid 導數上預訓練 FNN (使用 L-BFGS) ---")
    model = FNN()
    optimizer = optim.LBFGS(model.parameters(), lr=lr, max_iter=20)
    criterion = nn.MSELoss()
    y_tensor = torch.tensor(y_target, dtype=torch.float32)
    dyds_tensor = torch.tensor(dyds_target, dtype=torch.float32)
    
    for epoch in range(n_epochs):
        def closure():
            optimizer.zero_grad()
            dyds_pred = model(y_tensor)
            loss = criterion(dyds_pred, dyds_tensor)
            loss.backward()
            return loss
        
        loss = optimizer.step(closure)
        
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.8f}")
            
    model.eval()
    return model


# --- 4. 基於真實數據的 L-BFGS 微調 ---
class ODEModel(nn.Module):
    def __init__(self, fnn_model):
        super().__init__()
        self.fnn = fnn_model
    def forward(self, t, y):
        return self.fnn(y)

def calculate_loss(ode_model, patient_data, ab, pids, y0):
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
        loss = ((y_pred - y_sorted) ** 2).mean()
        return loss if torch.isfinite(loss) else torch.tensor(float('inf'))
    except Exception:
        return torch.tensor(float('inf'))

def train_fnn_on_data(initial_fnn, patient_data, y0, dps_path='dps.pth', n_epochs=10, lr=1e-6, max_iter_lbfgs=20):
    print("\n--- 在真實數據上使用 L-BFGS 訓練 FNN 和 DPS 參數 ---")
    ode_model = ODEModel(initial_fnn).train()
    
    dps_params_loaded = torch.load(dps_path, weights_only=False)
    ab = {}
    for pid, data in patient_data.items():
        if pid in dps_params_loaded:
            ab[pid] = {
                'a': nn.Parameter(torch.tensor(dps_params_loaded[pid]['a'], dtype=torch.float32)),
                'b': nn.Parameter(torch.tensor(dps_params_loaded[pid]['b'], dtype=torch.float32))
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
            loss = calculate_loss(ode_model, patient_data, ab, patient_pids, y0)
            if torch.isfinite(loss):
                loss.backward()
            return loss
        
        loss = optimizer.step(closure)
        print(f"  Epoch [{epoch+1}/{n_epochs}], L-BFGS Loss: {loss.item():.8f}")

    ode_model.eval()
    return ode_model, ab

# --- 5. 繪圖與保存 ---

def plot_pretrain_results(s_pop, y_pop_orig, stages_pop, s_grid, sigmoid_params, pretrained_fnn, y0):
    print("\n正在生成預訓練結果圖...")

    # --- 準備繪圖數據 ---
    y_sigmoid_grid_norm, _ = get_sigmoid_derivatives(s_grid, sigmoid_params)
    y_sigmoid_grid_orig = pc.inv_nor(y_sigmoid_grid_norm)
    
    ode_model = ODEModel(pretrained_fnn).eval()
    with torch.no_grad():
        y_fnn_traj_norm = torch_odeint(ode_model, y0, torch.from_numpy(s_grid).float()).numpy()
    y_fnn_traj_orig = pc.inv_nor(y_fnn_traj_norm)

    # --- 繪圖 ---
    TITLES = ['Aβ (A)', 'p-Tau (T)', 'N', 'Cognition (C)']
    colors = {'CN': 'orange', 'LMCI': 'green', 'AD': 'blue', 'Other': 'grey'}
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    axes = axes.flat
    
    for k in range(4):
        ax = axes[k]
        for stage in np.unique(stages_pop):
            mask = np.array(stages_pop) == stage
            ax.scatter(s_pop[mask], y_pop_orig[mask, k], s=15, alpha=0.4, c=colors[stage], label=f'{stage}')
        
        ax.plot(s_grid, y_sigmoid_grid_orig[:, k], 'r--', lw=2.5, label='Sigmoid Fit', zorder=3)
        ax.plot(s_grid, y_fnn_traj_orig[:, k], 'k-', lw=2.5, label='Pre-trained FNN Trajectory', zorder=4)
        
        ax.set_xlabel('Disease Progression Score (s)')
        ax.set_ylabel(TITLES[k])
        ax.legend()
        ax.grid(True, alpha=0.4)
        ax.set_title(TITLES[k])
    
    fig.suptitle('FNN Pre-training Result vs. Sigmoid Fit', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def plot_and_save(s_pop, y_pop_orig, stages_pop, s_grid, final_model, y0, sigmoid_params, model_save_path='fnn.pth'):
    print("\n正在生成最終對比圖並保存模型...")
    
    torch.save(final_model.state_dict(), model_save_path)
    print(f"模型已成功保存到 {model_save_path}")

    # --- 准备不确定性量化 ---
    n_samples = 100
    nn_sigma = 1e-2
    pred_trajectories = []
    s_grid_tensor = torch.from_numpy(s_grid).float()

    print(f"从神经网络参数的 N(μ, {nn_sigma**2}) 分布中采样 {n_samples} 次以进行不确定性量化...")
    for i in range(n_samples):
        temp_fnn = FNN()
        temp_fnn.load_state_dict(final_model.fnn.state_dict())
        temp_ode_model = ODEModel(temp_fnn).eval()

        with torch.no_grad():
            for param in temp_ode_model.fnn.net.parameters():
                noise = torch.randn_like(param) * nn_sigma
                param.add_(noise)
            
            pred = torch_odeint(temp_ode_model, y0, s_grid_tensor, method='dopri5', rtol=1e-4, atol=1e-5)
            pred_trajectories.append(pred.numpy())

    # --- 数据后处理 ---
    pred_trajectories = np.array(pred_trajectories)
    mean_pred_norm = np.mean(pred_trajectories, axis=0)
    ci_norm = np.percentile(pred_trajectories, [5, 95], axis=0)

    mean_pred_orig = pc.inv_nor(mean_pred_norm)
    ci_orig = pc.inv_nor(ci_norm)
    
    # Sigmoid curve
    y_sigmoid_grid_norm, _ = get_sigmoid_derivatives(s_grid, sigmoid_params)
    y_sigmoid_grid_orig = pc.inv_nor(y_sigmoid_grid_norm)

    # --- 繪圖 ---
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
        
        ax.plot(s_grid, mean_pred_orig[:, k], 'k-', lw=2.5, label='Mean Trajectory', zorder=4)
        ax.fill_between(s_grid, ci_orig[0, :, k], ci_orig[1, :, k], color='lightgrey', alpha=0.8, label='90% CI (NN Uncertainty)', zorder=1)
        ax.plot(s_grid, y_sigmoid_grid_orig[:, k], 'r--', lw=2.5, label='Sigmoid Fit', zorder=3)

        ax.set_xlabel('Disease Progression Score (s)')
        ax.set_ylabel(TITLES[k])
        ax.legend()
        ax.grid(True, alpha=0.4)
        ax.set_title(TITLES[k])
    
    fig.suptitle('FNN Model Trajectory', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('fnn.png')
    plt.show()


if __name__ == '__main__':
    # --- 執行 Pipeline ---
    # 1. 準備數據
    s_pop_np, y_pop_norm_np, stages_pop_np = load_dps_and_create_scatter(csf_dict, dps_path='dps.pth')
    
    y_pop_orig = pc.inv_nor(y_pop_norm_np)

    cn_y0s = [dat['y0'] for pid, dat in patient_data.items() if dat['stage'] == 'CN']
    y0_cn_avg = torch.stack(cn_y0s).mean(dim=0) if cn_y0s else torch.tensor([0.1, 0, 0, 0])
    print(f"使用CN群體的平均初始值: {y0_cn_avg.numpy()}")

    # 2. 階段一：預訓練 FNN
    s_grid_np_pretrain = np.linspace(-10, 20, 500)
    sigmoid_p = fit_sigmoids(s_pop_np, y_pop_norm_np)
    y_sigmoid_norm, dyds_sigmoid_norm = get_sigmoid_derivatives(s_grid_np_pretrain, sigmoid_p)
    fnn_pretrained = pretrain_fnn_on_sigmoid(y_sigmoid_norm, dyds_sigmoid_norm)
    
    # 3. 繪製預訓練結果
    plot_pretrain_results(s_pop_np, y_pop_orig, stages_pop_np, s_grid_np_pretrain, sigmoid_p, fnn_pretrained, y0_cn_avg)
    
    # 4. 階段二：在真實數據上微調模型
    print("\n--- 階段二：在真實數據上使用 L-BFGS 微調 FNN 和 DPS 參數 ---")
    final_ode_model, final_ab = train_fnn_on_data(fnn_pretrained, patient_data, y0_cn_avg, dps_path='dps.pth')

    # 5. 繪圖與保存最終結果
    s_grid_np_final = np.linspace(-10, 20, 300)
    plot_and_save(s_pop_np, y_pop_orig, stages_pop_np, s_grid_np_final, final_ode_model, y0_cn_avg, sigmoid_p)
    
    print("\n完整流程執行完畢。")