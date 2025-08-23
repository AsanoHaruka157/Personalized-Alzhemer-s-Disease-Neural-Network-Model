import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pccmnn as pc
from datetime import datetime

# ---------- Load model definition (same as in main.py) and weights ----------
name = 'fnn'
Message = 'Evaluation with loaded fnn.pth'

class ODEModel(nn.Module):
    def __init__(self, hidden_dim=32, num_layers=2):
        super().__init__()

        # 1. 输入层：5个输入（4个生物标记物 + s值）-> hidden_dim
        self.net = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 4),
        )

    def f(self, y: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        y = y.float()
        s = s.float()
        if s.dim() == 0:
            s_expanded = s.unsqueeze(0)
        else:
            s_expanded = s
        z = torch.cat([y, s_expanded], dim=0)
        z = self.net(z)
        return z

    def forward(self, s_grid: torch.Tensor, y0: torch.Tensor) -> torch.Tensor:
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

# ---------- 2. 绘制人群四联图 (根据s的5%和95%分位数) -----------------
s_min = -10
s_max = 20
s_curve = torch.linspace(s_min, s_max, 100)
# -------- 根据s的分位数范围过滤病例 ---------
keep = []
for p in patient_data:
    s_values = ab_dict[p][0] * patient_data[p]['t'] + ab_dict[p][1]
    if s_values.min() >= s_min and s_values.max() <= s_max:
        keep.append(p)

stage_dict = pc.load_stage_dict()

# 准备绘图数据
if not keep:
    print("Warning: No patients left after filtering by s-quantiles. Plot will only show the mean trajectory based on all patients.")
    # Fallback to avoid crashing: use all patients for y0_pop
    y0_pop = torch.tensor([0.1,0,0,0])
else:
    y0_pop = torch.tensor([0.1,0,0,0])

# ---------- 添加新轨迹：从平均初值开始的完整轨迹（不使用窗口） ----------
with torch.no_grad():
    y_curve_full = model(s_curve, y0_pop)
    y_curve_full = y_curve_full.detach().numpy()
y_curve_full = pc.inv_nor(y_curve_full)

TITLES = ['Aβ (A)', 'p-Tau (T)', 'N', 'Cognition (C)']

fig2, axes = plt.subplots(2, 2, figsize=(9, 6))
for k, ax in enumerate(axes.flat):
    # --- 分阶段准备散点数据 ---
    s_by_stage = {'CN': [], 'LMCI': [], 'AD': [], 'Other': []}
    y_by_stage = {'CN': [], 'LMCI': [], 'AD': [], 'Other': []}

    for p in keep:
        a, b = ab_dict[p]
        stage = stage_dict.get(p, 'Other')
        if stage not in s_by_stage:
            stage = 'Other'
        
        s_by_stage[stage].append(a * patient_data[p]['t'] + b)
        y_by_stage[stage].append(patient_data[p]['y'][:, k])

    # --- 绘制散点 (分期颜色) ---
    colors = {'CN': 'orange', 'LMCI': 'green', 'AD': 'blue', 'Other': 'grey'}
    for stage, s_points_list in s_by_stage.items():
        if s_points_list:
            s_all = torch.cat(s_points_list).numpy()
            y_all = torch.cat(y_by_stage[stage]).numpy()
            y_all = pc.inv_nor(y_all, k)
            ax.scatter(s_all, y_all, s=15, alpha=0.6, c=colors[stage], label=stage)
    
    # --- 从平均初值开始的完整轨迹 ---
    y_curve = y_curve_full[:,k]
    ax.plot(s_curve, y_curve, lw=1.5, c='red', linestyle='--', label='Trajectory from Mean Initial')
    
    ax.set_xlabel('Disease progression score  s')
    ax.set_ylabel(TITLES[k])
    ax.legend(fontsize=8)

fig2.suptitle(f'Population Model (s in [{float(s_min):.2f}, {float(s_max):.2f}])')
plt.tight_layout(rect=[0,0,1,0.96])
plt.savefig(f'{name}.png')

plt.show()

loss = 0
with torch.no_grad():
    for pid in patient_data:
        a, b = ab_dict[pid]
        s = a * patient_data[pid]['t'] + b
        y_pred = model(s, patient_data[pid]['y0'])
        y_pred = y_pred.numpy()
        loss += calculate_global_loss(y_pred, patient_data[pid]['y'])/len(y_pred)
    loss /= len(patient_data)

plt.show()
current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# 追加到输出文件
output_filename = f'experiments.out'
with open(output_filename, 'a') as f:  # 使用追加模式 'a'
    f.write(name)
    f.write(f"Time: {current_time}\n")
    f.write(Message)
    f.write("Model structure:\n")
    f.write(str(model))
    f.write(f"MSE: {loss:.4f}")
