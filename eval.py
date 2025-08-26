import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pccmnn as pc
from datetime import datetime

# ---------- Load model definition (same as in main.py) and weights ----------
name = 'fnn'
Message = 'Evaluation with loaded fnn.pth - Hybrid Model with Polynomial and Neural Network'

class ODEModel(nn.Module):
    def __init__(self, hidden_dim=32, num_layers=2):
        super().__init__()

        # 1. 输入层：5个输入（4个生物标记物 + s值）-> hidden_dim
        self.net = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),
            nn.Tanh()
        )
        
        # 2. 多项式系数：15个可学习参数
        # g(y0) = a0*y0^2 + a1*y0 + a2
        # g(y1) = b0*y1^2 + b1*y0*y1 + b2*y1 + b3
        # g(y2) = c0*y2^2 + c1*y1*y2 + c2*y2 + c3
        # g(y3) = d0*y3^2 + d1*y2*y3 + d2*y3 + d3
        self.poly_coeffs = nn.Parameter(torch.zeros(15))  # 初始化15个系数

    def f(self, y: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        y = y.float()
        s = s.float()
        # 确保s是标量，将其扩展为与y兼容的形状
        if s.dim() == 0:  # s是标量
            s_expanded = s.unsqueeze(0)  # 变成[1]
        else:
            s_expanded = s
        
        # 将y和s连接起来作为网络输入
        z = torch.cat([y, s_expanded], dim=0)  # 连接为[5]的向量
        
        # 神经网络部分
        net_output = self.net(z)
        
        # 多项式部分 g(y)
        y0, y1, y2, y3 = y[0], y[1], y[2], y[3]
        
        # 提取多项式系数
        a0, a1, a2 = self.poly_coeffs[0], self.poly_coeffs[1], self.poly_coeffs[2]
        b0, b1, b2, b3 = self.poly_coeffs[3], self.poly_coeffs[4], self.poly_coeffs[5], self.poly_coeffs[6]
        c0, c1, c2, c3 = self.poly_coeffs[7], self.poly_coeffs[8], self.poly_coeffs[9], self.poly_coeffs[10]
        d0, d1, d2, d3 = self.poly_coeffs[11], self.poly_coeffs[12], self.poly_coeffs[13], self.poly_coeffs[14]
        
        # 计算多项式项
        g0 = a0 * y0**2 + a1 * y0 + a2
        g1 = b0 * y1**2 + b1 * y0 * y1 + b2 * y1 + b3
        g2 = c0 * y2**2 + c1 * y1 * y2 + c2 * y2 + c3
        g3 = d0 * y3**2 + d1 * y2 * y3 + d2 * y3 + d3
        
        g_y = torch.stack([g0, g1, g2, g3])
        
        # 最终输出：f(y,s) = g(y) * net(y,s)
        return g_y * net_output

    def forward(self, s_grid: torch.Tensor, y0: torch.Tensor) -> torch.Tensor:
        # 4th-order Runge-Kutta integration to solve the ODE
        ys = [y0]
        for i in range(1, len(s_grid)):
            h = s_grid[i] - s_grid[i-1]
            y_i = ys[-1]
            
            k1 = self.f(y_i, s_grid[i-1])
            k2 = self.f(y_i + 0.5 * h * k1, s_grid[i-1])
            k3 = self.f(y_i + 0.5 * h * k2, s_grid[i-1])
            k4 = self.f(y_i + h * k3, s_grid[i-1])
            
            y_next = y_i + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            ys.append(y_next)
            
        return torch.stack(ys)          # (len_s, 4)

    def f_polynomial_only(self, y: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """Only polynomial component: g(y)"""
        y0, y1, y2, y3 = y[0], y[1], y[2], y[3]
        
        # 提取多项式系数
        a0, a1, a2 = self.poly_coeffs[0], self.poly_coeffs[1], self.poly_coeffs[2]
        b0, b1, b2, b3 = self.poly_coeffs[3], self.poly_coeffs[4], self.poly_coeffs[5], self.poly_coeffs[6]
        c0, c1, c2, c3 = self.poly_coeffs[7], self.poly_coeffs[8], self.poly_coeffs[9], self.poly_coeffs[10]
        d0, d1, d2, d3 = self.poly_coeffs[11], self.poly_coeffs[12], self.poly_coeffs[13], self.poly_coeffs[14]
        
        # 计算多项式项
        g0 = a0 * y0**2 + a1 * y0 + a2
        g1 = b0 * y1**2 + b1 * y0 * y1 + b2 * y1 + b3
        g2 = c0 * y2**2 + c1 * y1 * y2 + c2 * y2 + c3
        g3 = d0 * y3**2 + d1 * y2 * y3 + d2 * y3 + d3
        
        return torch.stack([g0, g1, g2, g3])

    def f_neural_only(self, y: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """Only neural network component: net(y,s)"""
        y = y.float()
        s = s.float()
        if s.dim() == 0:
            s_expanded = s.unsqueeze(0)
        else:
            s_expanded = s
        
        z = torch.cat([y, s_expanded], dim=0)
        return self.net(z)

    def forward_polynomial_only(self, s_grid: torch.Tensor, y0: torch.Tensor) -> torch.Tensor:
        """Forward pass using only polynomial component"""
        ys = [y0]
        for i in range(1, len(s_grid)):
            h = s_grid[i] - s_grid[i-1]
            y_i = ys[-1]
            
            k1 = self.f_polynomial_only(y_i, s_grid[i-1])
            k2 = self.f_polynomial_only(y_i + 0.5 * h * k1, s_grid[i-1])
            k3 = self.f_polynomial_only(y_i + 0.5 * h * k2, s_grid[i-1])
            k4 = self.f_polynomial_only(y_i + h * k3, s_grid[i-1])
            
            y_next = y_i + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            ys.append(y_next)
            
        return torch.stack(ys)

    def forward_neural_only(self, s_grid: torch.Tensor, y0: torch.Tensor) -> torch.Tensor:
        """Forward pass using only neural network component"""
        ys = [y0]
        for i in range(1, len(s_grid)):
            h = s_grid[i] - s_grid[i-1]
            y_i = ys[-1]
            
            k1 = self.f_neural_only(y_i, s_grid[i-1])
            k2 = self.f_neural_only(y_i + 0.5 * h * k1, s_grid[i-1])
            k3 = self.f_neural_only(y_i + 0.5 * h * k2, s_grid[i-1])
            k4 = self.f_neural_only(y_i + h * k3, s_grid[i-1])
            
            y_next = y_i + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            ys.append(y_next)
            
        return torch.stack(ys)

# Instantiate and load weights
model = ODEModel(hidden_dim=32, num_layers=2)
state = torch.load('fnn.pth', map_location=torch.device('cpu'))
model.load_state_dict(state)
model.eval()

# ---------- Minimal data loading to support plotting ----------
csf_dict = pc.load_csf_data()
patient_data = {}
for pid, sample in csf_dict.items():
    t = torch.from_numpy(sample[:, 0]).float()
    y = torch.from_numpy(sample[:, 1:5]).float()
    patient_data[pid] = {"t": t, "y": y, "y0": y[0].clone()}

# Create a simple ab_dict estimate for s mapping (deterministic, stage-based start)
stage_dict = pc.load_stage_dict()
ab_dict = {}
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
    alpha = torch.tensor(1.0)
    beta = s_range[0] - alpha * torch.min(t)
    ab_dict[pid] = (alpha, beta)

# Simple MSE for evaluation
def calculate_global_loss(y_pred, y_true):
    y_pred_t = torch.as_tensor(y_pred)
    return torch.mean((y_pred_t - y_true) ** 2)

# ---------- 2. 绘制人群四联图 (根据s的10%和90%分位数) -----------------
# 收集所有患者的s值
all_s_values = []
for p in patient_data:
    s_values = ab_dict[p][0] * patient_data[p]['t'] + ab_dict[p][1]
    all_s_values.append(s_values)

# 计算所有s值的10分位数和90分位数
all_s_flat = torch.cat(all_s_values)
s_10_percentile = torch.quantile(all_s_flat, 0.10)
s_90_percentile = torch.quantile(all_s_flat, 0.90)

print(f"S value range: 10th percentile = {s_10_percentile:.2f}, 90th percentile = {s_90_percentile:.2f}")

# 使用10分位数到90分位数的范围
s_min = s_10_percentile
s_max = s_90_percentile
s_curve = torch.linspace(s_min, s_max, 100)

# 过滤在10-90分位数范围内的数据点
keep = []
for p in patient_data:
    s_values = ab_dict[p][0] * patient_data[p]['t'] + ab_dict[p][1]
    # 检查是否有任何s值在范围内
    if torch.any((s_values >= s_min) & (s_values <= s_max)):
        keep.append(p)

stage_dict = pc.load_stage_dict()

# 准备绘图数据 - 使用10分位数对应的y值作为起始点
y0_pop = torch.tensor([0.1,0,0,0])

# ---------- 分别计算三种轨迹 ----------
with torch.no_grad():
    # 1. 完整混合模型轨迹
    y_curve_full = model(s_curve, y0_pop)
    y_curve_full = y_curve_full.detach().numpy()
    y_curve_full = pc.inv_nor(y_curve_full)
    
    # 2. 仅多项式模型轨迹
    y_curve_poly = model.forward_polynomial_only(s_curve, y0_pop)
    y_curve_poly = y_curve_poly.detach().numpy()
    y_curve_poly = pc.inv_nor(y_curve_poly)
    
    # 3. 仅神经网络轨迹
    y_curve_neural = model.forward_neural_only(s_curve, y0_pop)
    y_curve_neural = y_curve_neural.detach().numpy()
    y_curve_neural = pc.inv_nor(y_curve_neural)

TITLES = ['Aβ (A)', 'p-Tau (T)', 'N', 'Cognition (C)']

fig2, axes = plt.subplots(2, 2, figsize=(12, 8))
for k, ax in enumerate(axes.flat):
    # --- 分阶段准备散点数据 ---
    s_by_stage = {'CN': [], 'LMCI': [], 'AD': [], 'Other': []}
    y_by_stage = {'CN': [], 'LMCI': [], 'AD': [], 'Other': []}

    for p in keep:
        a, b = ab_dict[p]
        stage = stage_dict.get(p, 'Other')
        if stage not in s_by_stage:
            stage = 'Other'
        
        s_values = a * patient_data[p]['t'] + b
        y_values = patient_data[p]['y'][:, k]
        
        # 只保留在10-90分位数范围内的数据点
        mask = (s_values >= s_min) & (s_values <= s_max)
        if torch.any(mask):
            s_by_stage[stage].append(s_values[mask])
            y_by_stage[stage].append(y_values[mask])

    # --- 绘制散点 (分期颜色) ---
    colors = {'CN': 'orange', 'LMCI': 'green', 'AD': 'blue', 'Other': 'grey'}
    for stage, s_points_list in s_by_stage.items():
        if s_points_list:
            s_all = torch.cat(s_points_list).numpy()
            y_all = torch.cat(y_by_stage[stage]).numpy()
            y_all = pc.inv_nor(y_all, k)
            ax.scatter(s_all, y_all, s=15, alpha=0.6, c=colors[stage], label=stage)
    
    # --- 绘制三种轨迹 ---
    s_curve_np = s_curve.numpy()
    
    # 完整混合模型轨迹
    y_curve = y_curve_full[:,k]
    ax.plot(s_curve_np, y_curve, lw=2, c='red', linestyle='-', label='Hybrid Model (Full)')
    
    # 仅多项式模型轨迹
    y_curve_poly_k = y_curve_poly[:,k]
    ax.plot(s_curve_np, y_curve_poly_k, lw=1.5, c='blue', linestyle='--', label='Polynomial Only')
    
    # 仅神经网络轨迹
    y_curve_neural_k = y_curve_neural[:,k]
    ax.plot(s_curve_np, y_curve_neural_k, lw=1.5, c='green', linestyle=':', label='Neural Network Only')
    
    ax.set_xlabel('Disease progression score  s')
    ax.set_ylabel(TITLES[k])
    ax.legend(fontsize=8)

fig2.suptitle(f'Hybrid Model Comparison (s in 10-90 percentile: [{float(s_min):.2f}, {float(s_max):.2f}])')
plt.tight_layout(rect=[0,0,1,0.96])
plt.savefig(f'{name}_comparison.png', dpi=300, bbox_inches='tight')

plt.show()

# 计算损失
loss = 0
with torch.no_grad():
    for pid in patient_data:
        a, b = ab_dict[pid]
        s = a * patient_data[pid]['t'] + b
        y_pred = model(s, patient_data[pid]['y0'])
        y_pred = y_pred.numpy()
        loss += calculate_global_loss(y_pred, patient_data[pid]['y'])/len(y_pred)
    loss /= len(patient_data)

current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# 追加到输出文件
output_filename = f'experiments.out'
with open(output_filename, 'a') as f:  # 使用追加模式 'a'
    f.write(f"\n{name}_comparison\n")
    f.write(f"Time: {current_time}\n")
    f.write(Message)
    f.write("\nModel structure:\n")
    f.write(str(model))
    f.write(f"\nMSE: {loss:.4f}")
    f.write(f"\nPolynomial coefficients: {model.poly_coeffs.detach().numpy()}\n")
