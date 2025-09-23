import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import pccmnn as pc
import pymc as pm
import pytensor.tensor as pt
import arviz as az

try:
    from torchdiffeq import odeint as torch_odeint
except ImportError:
    raise ImportError("Please install torchdiffeq: pip install torchdiffeq")

# --- 设定 ---
# 确保这个文件存在，它是之前训练好的包含NN权重的模型
PRETRAINED_MODEL_PATH = 'fpp.pth' 
# MCMC 采样设置
N_DRAWS = 2000
N_TUNE = 1000
N_CHAINS = 2

# --- 沿用之前的模型定义，方便加载权重 ---
class ODEModel(nn.Module):
    def __init__(self, hidden_dim=1024, dropout_rate=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden_dim), nn.Dropout(dropout_rate), nn.Tanh(),
            nn.Linear(hidden_dim, 4), nn.Dropout(dropout_rate), nn.Tanh()
        )
        self.output_scaler = nn.Parameter(torch.tensor([0.1]))
        self.wA = nn.Parameter(torch.zeros(3))
        self.wT = nn.Parameter(torch.zeros(6))
        self.wN = nn.Parameter(torch.zeros(6))
        self.wC = nn.Parameter(torch.zeros(6))

    def f(self, y: torch.Tensor) -> torch.Tensor:
        return self.net(y) * self.output_scaler

    def poly(self, y: torch.Tensor, wA, wT, wN, wC) -> torch.Tensor:
        A, T, N, C = y[..., 0], y[..., 1], y[..., 2], y[..., 3]
        phi_A = torch.stack([torch.ones_like(A), A, A**2], dim=-1)
        phi_T = torch.stack([torch.ones_like(T), T, T**2, A, A**2, A*T], dim=-1)
        phi_N = torch.stack([torch.ones_like(N), N, N**2, T, T**2, T*N], dim=-1)
        phi_C = torch.stack([torch.ones_like(C), C, C**2, N, N**2, N*C], dim=-1)
        dAds = (phi_A @ wA)
        dTds = (phi_T @ wT)
        dNds = (phi_N @ wN)
        dCds = (phi_C @ wC)
        return torch.stack([dAds, dTds, dNds, dCds], dim=-1)

    def combined_dynamics(self, s: torch.Tensor, y: torch.Tensor, wA, wT, wN, wC) -> torch.Tensor:
        return self.f(y) + self.poly(y, wA, wT, wN, wC)

# --- 1. 资料载入和准备 ---
csf_dict = pc.load_data()
stage_dict = pc.load_stage_dict()
dps_params_loaded = torch.load('dps_fpp.pth')
print(f"成功载入 {len(csf_dict)} 位患者的资料。")

# 聚合所有数据点到一个Tensor中
all_s, all_y = [], []
for pid, sample in csf_dict.items():
    if pid in dps_params_loaded:
        t = torch.from_numpy(sample[:, 0]).float()
        y = torch.from_numpy(sample[:, 1:5]).float()
        a = dps_params_loaded[pid]['a']
        b = dps_params_loaded[pid]['b']
        s = a * t + b
        all_s.append(s)
        all_y.append(y)

s_global = torch.cat(all_s)
y_global = torch.cat(all_y)
s_sorted, sort_indices = torch.sort(s_global)
y_sorted = y_global[sort_indices]

# 使用CN组的平均初始值作为ODE的y0
cn_y0s = [torch.from_numpy(csf_dict[pid][0, 1:5]).float() for pid, stage in stage_dict.items() if stage == 'CN' and pid in csf_dict]
y0_cn_avg = torch.stack(cn_y0s).mean(dim=0)
print(f"使用CN群體的平均初始值: {y0_cn_avg.numpy()}")

# --- 2. 加载并冻结神经网络 ---
frozen_nn_model = ODEModel()
try:
    frozen_nn_model.load_state_dict(torch.load(PRETRAINED_MODEL_PATH))
    frozen_nn_model.eval()
    for param in frozen_nn_model.parameters():
        param.requires_grad = False
    print(f"成功从 {PRETRAINED_MODEL_PATH} 加载并冻结了神经网络。")
except FileNotFoundError:
    print(f"错误: 未找到预训练模型 {PRETRAINED_MODEL_PATH}。请先运行之前的脚本生成此文件。")
    exit()

# --- 3. 定义PyMC与ODE的接口 ---
# 这个函数是PyMC和PyTorch ODE求解器之间的桥梁
def ode_solver_for_pymc(s_values, y0, wA, wT, wN, wC):
    # PyMC的自定义分布函数内不支持PyTorch的梯度计算
    with torch.no_grad():
        # 将PyTensor/Numpy变量转换为Torch Tensor
        wA_t = torch.from_numpy(np.array(wA, dtype=np.float32))
        wT_t = torch.from_numpy(np.array(wT, dtype=np.float32))
        wN_t = torch.from_numpy(np.array(wN, dtype=np.float32))
        wC_t = torch.from_numpy(np.array(wC, dtype=np.float32))
        
        # 定义带特定参数的动力学函数
        def dynamics(s, y):
            return frozen_nn_model.combined_dynamics(s, y, wA_t, wT_t, wN_t, wC_t)
            
        # 求解ODE
        pred = torch_odeint(dynamics, y0, s_values, method='dopri5', rtol=1e-4, atol=1e-5)
        return pred.numpy()

# pytensor.Op可以将一个普通的Python函数包装成PyMC可以使用的形式
class OdeOp(pt.Op):
    def __init__(self, s_values, y0):
        self.s_values = s_values
        self.y0 = y0

    def perform(self, node, inputs, output_storage):
        wA, wT, wN, wC = inputs
        result = ode_solver_for_pymc(self.s_values, self.y0, wA, wT, wN, wC)
        output_storage[0][0] = result

    # 定义输入和输出的类型
    def infer_shape(self, *args):
        # inputs are wA, wT, wN, wC.
        # output shape is (len(s_values), 4)
        return [(self.s_values.shape[0], 4)]

# 实例化Op
ode_op = OdeOp(s_sorted, y0_cn_avg)

# --- 4. 构建并运行PyMC模型 ---
with pm.Model() as pymc_model:
    # --- 先验分布 ---
    # 为多项式参数设置正态先验
    wA = pm.Normal('wA', mu=0., sigma=1., shape=3)
    wT = pm.Normal('wT', mu=0., sigma=1., shape=6)
    wN = pm.Normal('wN', mu=0., sigma=1., shape=6)
    wC = pm.Normal('wC', mu=0., sigma=1., shape=6)
    
    # --- 噪声项的先验 ---
    # 为4个生物标记的观测噪声设置半正态分布先验
    sigma = pm.HalfNormal('sigma', sigma=1., shape=4)
    
    # --- 模型的确定性部分 ---
    # 调用我们包装好的ODE求解器
    mu = ode_op(wA, wT, wN, wC)
    
    # --- 似然函数 ---
    # 假设观测值服从以ODE解为均值，sigma为标准差的正态分布
    Y_obs = pm.Normal('Y_obs', mu=mu, sigma=sigma, observed=y_sorted.numpy())

    # --- 运行采样器 ---
    print("开始PyMC采样...")
    trace = pm.sample(N_DRAWS, tune=N_TUNE, chains=N_CHAINS, target_accept=0.9)
    print("采样完成。")


# --- 5. 结果分析与绘图 ---
s_grid_plot = torch.linspace(-10, 20, 200)

# 从后验分布中抽取样本并生成预测轨迹
posterior_samples = az.extract(trace, num_samples=200)
pred_trajectories = []
for i in range(len(posterior_samples.sample)):
    wA_s = posterior_samples.wA.values[i]
    wT_s = posterior_samples.wT.values[i]
    wN_s = posterior_samples.wN.values[i]
    wC_s = posterior_samples.wC.values[i]
    
    traj = ode_solver_for_pymc(s_grid_plot, y0_cn_avg, wA_s, wT_s, wN_s, wC_s)
    pred_trajectories.append(traj)

# 转换成Numpy Array
pred_trajectories = np.array(pred_trajectories)

# 计算均值和94% HDI (贝叶斯置信区间)
mean_pred = np.mean(pred_trajectories, axis=0)
hdi_pred = np.percentile(pred_trajectories, [3, 97], axis=0) # 94% HDI

# 反正規化
mean_pred_orig = pc.inv_nor(mean_pred)
hdi_pred_orig = pc.inv_nor(hdi_pred)

# --- 绘图 ---
TITLES = ['Aβ (A)', 'p-Tau (T)', 'N', 'Cognition (C)']
colors = {'CN': 'orange', 'LMCI': 'green', 'AD': 'blue', 'Other': 'grey'}
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
axes = axes.flat

for k in range(4):
    ax = axes[k]
    for pid, dat in csf_dict.items():
        if pid in dps_params_loaded:
            stage = stage_dict.get(pid, 'Other')
            t = torch.from_numpy(dat[:, 0]).float()
            y = torch.from_numpy(dat[:, 1:5]).float()
            s = dps_params_loaded[pid]['a'] * t + dps_params_loaded[pid]['b']
            y_orig = pc.inv_nor(y[:, k].numpy(), k)
            ax.scatter(s, y_orig, s=10, alpha=0.4, c=colors.get(stage, 'grey'))

    ax.plot(s_grid_plot.numpy(), mean_pred_orig[:, k], 'r-', lw=2.5, label='Posterior Mean', zorder=4)
    ax.fill_between(s_grid_plot.numpy(), hdi_pred_orig[0, :, k], hdi_pred_orig[1, :, k], color='red', alpha=0.3, label='94% HDI', zorder=1)
    
    ax.set_xlabel('Disease Progression Score (s)')
    ax.set_ylabel(TITLES[k])
    ax.legend()
    ax.grid(True, alpha=0.4)
    ax.set_title(TITLES[k])

fig.suptitle('Polynomial UQ with PyMC (Frozen NN)', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('pymc_uq_polynomial.png')
plt.show()