import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import pccmnn as pc
try:
    from torchdiffeq import odeint as torch_odeint
except ImportError:
    raise ImportError("Please install torchdiffeq to run this script: pip install torchdiffeq")

# --- 0. 資料載入和準備 ---
csf_dict = pc.load_data()
stage_dict = pc.load_stage_dict()
print(f"成功載入 {len(csf_dict)} 位患者的資料。")

patient_data = {}
for pid, sample in csf_dict.items():
    t = torch.from_numpy(sample[:, 0]).float()
    y = torch.from_numpy(sample[:, 1:5]).float()
    patient_data[pid] = {"t": t, "y": y, "y0": y[0].clone()}

def get_cn_average_y0(patient_data, stage_dict):
    cn_y0s = []
    for pid, data in patient_data.items():
        if stage_dict.get(pid) == 'CN':
            cn_y0s.append(data['y0'])
    if not cn_y0s:
        print("警告: 未找到CN患者, 使用預設y0。")
        return torch.tensor([0.1, 0, 0, 0])
    avg_y0 = torch.stack(cn_y0s).mean(dim=0)
    print(f"使用CN群體的平均初始值: {avg_y0.numpy()}")
    return avg_y0

y0_cn_avg = get_cn_average_y0(patient_data, stage_dict)
name = 'fpp'

# --- 1. 定義混合ODE模型 ---
class ODEModel(nn.Module):
    def __init__(self, hidden_dim=1024):
        super().__init__()
        # 1a. 神經網路部分 f(y)
        self.net = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 4),
            nn.Tanh()  # **策略2: 使用Tanh限制輸出範圍**
        )
        self.output_scaler = nn.Parameter(torch.tensor([0.1]), requires_grad=True) # 縮放Tanh輸出

        # 1b. 多項式部分 p(y)
        self.wA = nn.Parameter(torch.zeros(3))
        self.wT = nn.Parameter(torch.zeros(6))
        self.wN = nn.Parameter(torch.zeros(6))
        self.wC = nn.Parameter(torch.zeros(6))

        self._initialize_weights()

    def _initialize_weights(self):
        """ **策略4: 初始化權重為非常小的值** """
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=1e-3) # 使用非常小的標準差
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def load_poly_params(self, path='poly.pth'):
        try:
            poly_coeffs = torch.load(path)
            self.wA.data = poly_coeffs['wA']
            self.wT.data = poly_coeffs['wT']
            self.wN.data = poly_coeffs['wN']
            self.wC.data = poly_coeffs['wC']
            print(f"成功從 {path} 載入預訓練的多項式模型係數。")
        except FileNotFoundError:
            print(f"警告: 未找到 {path}。")

    def poly(self, y: torch.Tensor) -> torch.Tensor:
        A, T, N, C = y[..., 0], y[..., 1], y[..., 2], y[..., 3]
        phi_A = torch.stack([torch.ones_like(A), A, A**2], dim=-1)
        phi_T = torch.stack([torch.ones_like(T), T, T**2, A, A**2, A*T], dim=-1)
        phi_N = torch.stack([torch.ones_like(N), N, N**2, T, T**2, T*N], dim=-1)
        phi_C = torch.stack([torch.ones_like(C), C, C**2, N, N**2, N*C], dim=-1)
        dAds = (phi_A @ self.wA)
        dTds = (phi_T @ self.wT)
        dNds = (phi_N @ self.wN)
        dCds = (phi_C @ self.wC)
        return torch.stack([dAds, dTds, dNds, dCds], dim=-1)

    def f(self, y: torch.Tensor) -> torch.Tensor:
        return self.net(y) * self.output_scaler

    def combined_dynamics(self, s: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # torchdiffeq的函數簽名是 func(t, y)
        return self.f(y) + self.poly(y)
    
    def net_dynamics(self, s: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.f(y)

    def poly_dynamics(self, s: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.poly(y)

    def forward(self, s_grid: torch.Tensor, y0: torch.Tensor, dynamics='combined') -> torch.Tensor:
        dynamics_map = {
            'combined': self.combined_dynamics,
            'net_only': self.net_dynamics,
            'poly_only': self.poly_dynamics
        }
        if dynamics not in dynamics_map:
            raise ValueError("dynamics must be 'combined', 'net_only', or 'poly_only'")
        
        # **策略3: 調整容忍度**
        return torch_odeint(dynamics_map[dynamics], y0, s_grid, method='dopri5', rtol=1e-4, atol=1e-5)

# --- 2. 定義訓練流程 ---
def calculate_loss(model, patient_data, ab, pids, y0, sigma=None, s_penalty_weight=0.0):
    all_s_values, all_y_values = [], []
    for pid in pids:
        dat = patient_data[pid]
        a = ab[pid]['a']
        b = ab[pid]['b']
        s_values = a * dat['t'] + b
        all_s_values.append(s_values)
        all_y_values.append(dat['y'])
    
    s_global = torch.cat(all_s_values)
    y_global = torch.cat(all_y_values)
    
    s_sorted, sort_indices = torch.sort(s_global)
    y_sorted = y_global[sort_indices]

    if s_sorted.numel() < 2: return torch.tensor(0.0)

    # --- **核心修改：S值範圍懲罰** ---
    penalty = torch.tensor(0.0)
    if s_penalty_weight > 0:
        safe_min, safe_max = -15.0, 25.0
        out_of_bounds_lower = torch.clamp(safe_min - s_sorted, min=0.0)
        out_of_bounds_upper = torch.clamp(s_sorted - safe_max, min=0.0)
        # 使用平方懲罰
        penalty = s_penalty_weight * (out_of_bounds_lower.pow(2).mean() + out_of_bounds_upper.pow(2).mean())

    try:
        y_pred = model(s_sorted, y0)
        mse_loss = ((y_pred - y_sorted) ** 2)
        if torch.isnan(mse_loss).any():
            return torch.tensor(float('inf')) # 如果求解失敗，返回無窮大loss
        
        if sigma is not None:
            mse_loss = mse_loss * sigma.clamp(min=1e-4)
        
        total_loss = mse_loss.mean() + penalty
        return total_loss
        
    except Exception as e:
        # 捕獲求解器錯誤
        return torch.tensor(float('inf'))


def fit_population(
    patient_data,
    y0,
    n_epochs=50,
    lr_model=1e-3, # **建议: 降低学习率**
    lr_lbfgs=1e-4,
    clip_value=1.0,
    s_penalty=1.0,
    max_iter_lbfgs=10
):
    sigma = torch.ones(4)
    model = ODEModel()
    model.load_poly_params()

    # --- 策略: 冻结多项式模型的参数 ---
    model.wA.requires_grad = False
    model.wT.requires_grad = False
    model.wN.requires_grad = False
    model.wC.requires_grad = False

    try:
        dps_params_loaded = torch.load('dps.pth', weights_only=False)
        ab = {}
        for pid, params in dps_params_loaded.items():
            if pid not in patient_data: continue
            ab[pid] = {
                'a': nn.Parameter(torch.tensor(params['a'], dtype=torch.float32)),
                'b': nn.Parameter(torch.tensor(params['b'], dtype=torch.float32))
            }
        print("成功從 dps.pth 載入並創建可訓練的 a, b 參數。")
    except FileNotFoundError:
        print("錯誤: 未找到 dps.pth。")
        return None, None
        
    patient_pids = list(ab.keys())

    # --- **策略2: 分離優化器** ---
    dps_params = [p for pid in patient_pids for p in ab[pid].values()]
    opt_model = optim.Adam(model.parameters(), lr=lr_model)

    # --- 策略: 为多项式和DPS参数创建一个LBFGS微调优化器 ---
    finetune_params = [model.wA, model.wT, model.wN, model.wC] + dps_params # **建议: 代码清理**
    opt_finetune = optim.LBFGS(finetune_params, lr=lr_lbfgs, max_iter=max_iter_lbfgs, line_search_fn="strong_wolfe") # LBFGS for fine-tuning


    for epoch in range(n_epochs):
        # --- 步驟 3a: 優化模型參數 (NN) ---
        opt_model.zero_grad()
        loss_model = calculate_loss(model, patient_data, ab, patient_pids, y0, sigma, s_penalty_weight=0.0)
        if torch.isfinite(loss_model):
            loss_model.backward()
            nn.utils.clip_grad_norm_(model.net.parameters(), clip_value) # 只裁剪被优化的参数
            opt_model.step()

        # --- 步驟 3b: 優化多项式和DPS参数 (LBFGS微调) ---
        def closure_finetune():
            opt_finetune.zero_grad()
            loss = calculate_loss(model, patient_data, ab, patient_pids, y0, sigma, s_penalty_weight=s_penalty)
            if torch.isfinite(loss):
                loss.backward()
            return loss
        
        loss_finetune = opt_finetune.step(closure_finetune)

        if epoch % 10 == 0:
            print(f"Epoch {epoch+1:02d}/{n_epochs} | Loss Model={loss_model.item():.6f} | Loss Finetune={loss_finetune.item():.6f}")

        with torch.no_grad():
            loss_val = calculate_loss(model, patient_data, ab, patient_pids, y0)
            if torch.isfinite(loss_val):
                sigma = torch.full((4,), loss_val.item())

    model.eval()
    return model, ab


# --- 4. 訓練和繪圖 ---
model, trained_ab = fit_population(patient_data, y0_cn_avg)
if model is None:
    exit()

torch.save(model.state_dict(), f'{name}.pth')
torch.save(trained_ab, f'dps_{name}.pth')

# --- 繪圖 ---
with torch.no_grad():
    s_grid = torch.linspace(-10, 20, 200)
    
    y_poly, y_net, y_combined = None, None, None
    try:
        y_poly = model(s_grid, y0_cn_avg, dynamics='poly_only').numpy()
        y_net = model(s_grid, y0_cn_avg, dynamics='net_only').numpy()
        y_combined = model(s_grid, y0_cn_avg, dynamics='combined').numpy()
    except Exception as e:
        print(f"繪圖時ODE求解失敗: {e}")

    if y_combined is None:
        print("無法生成繪圖，因為最終模型求解失敗。")
        exit()
        
    y_poly_orig = pc.inv_nor(y_poly)
    y_net_orig = pc.inv_nor(y_net)
    y_combined_orig = pc.inv_nor(y_combined)
    
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
                ax.scatter(s, y_orig, s=10, alpha=0.4, c=colors[stage])
        
        ax.plot(s_grid, y_poly_orig[:, k], 'g-.', lw=2, label='Polynomial Only', zorder=3)
        ax.plot(s_grid, y_net_orig[:, k], 'b:', lw=2, label='NN Correction Only (f(y))', zorder=3)
        ax.plot(s_grid, y_combined_orig[:, k], 'r-', lw=2.5, label='Combined (f(y) + p(y))', zorder=4)
        ax.set_xlabel('Disease Progression Score (s)')
        ax.set_ylabel(TITLES[k])
        ax.legend()
        ax.grid(True, alpha=0.4)
        ax.set_title(TITLES[k])
        
    fig.suptitle('Hybrid Model Trajectories (Stable Version)', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'{name}.png')
    plt.show()