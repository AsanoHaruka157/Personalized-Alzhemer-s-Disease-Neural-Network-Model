import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import odeint
from tqdm import tqdm
import pccmnn as pc # 假设您有这个文件来加载和反归一化数据

# --- 0. 数据加载和准备 ---
# 加载数据和患者分期信息
csf_dict = pc.load_data()
stage_dict = pc.load_stage_dict()
print(f"成功加载 {len(csf_dict)} 位患者的数据。")

# --- 1. 为每位患者分配DPS变换参数 ---
def assign_dps_params(csf_dict, stage_dict):
    """
    根据患者分期（CN, LMCI, AD）为每位患者指定 a 和 b 参数。
    a: CN=1, LMCI=2, AD=4
    b: 随机选择，使得初始s值落在指定区间内
    """
    patient_data = {}
    # 按照您的要求更新 s_ranges
    s_ranges = {
        'CN': (-10, 0),
        'LMCI': (-2, 8),
        'AD': (5, 20),
        'Other': (-10, 20) # 为其他类型提供一个默认范围
    }
    a_values = {'CN': 1.0, 'LMCI': 2.0, 'AD': 4.0, 'Other': 1.0}

    # 收集所有 (s, y) 点
    all_s_points = []
    all_y_points = []
    all_stages = []

    for pid, sample in csf_dict.items():
        stage = stage_dict.get(pid, 'Other')
        t = sample[:, 0]
        y = sample[:, 1:5]

        a = a_values[stage]
        s_min, s_max = s_ranges[stage]
        
        # 计算b，使s_initial落在目标区间
        t_initial = t[0]
        s_initial_target = np.random.uniform(s_min, s_max)
        b = s_initial_target - a * t_initial

        s = a * t + b
        
        patient_data[pid] = {'t': t, 'y': y, 's': s, 'stage': stage, 'a': a, 'b': b}
        
        all_s_points.append(s)
        all_y_points.append(y)
        all_stages.extend([stage] * len(t))

    # 将列表转换为Numpy数组
    s_population = np.concatenate(all_s_points)
    y_population = np.concatenate(all_y_points)
    
    return patient_data, s_population, y_population, all_stages

# --- 新增功能：计算CN群体的平均初始值 ---
def get_cn_average_y0(patient_data):
    """
    计算CN（认知正常）群体在第一次访问时的平均生物标记物值（忽略NaN）。
    """
    cn_y0s = []
    for pid, data in patient_data.items():
        if data['stage'] == 'CN':
            cn_y0s.append(data['y'][0]) # 取第一次访问的数据
    
    if not cn_y0s:
        print("警告：未找到CN患者数据，将使用默认初始值 [0.1, 0, 0, 0]。")
        return np.array([0.1, 0.0, 0.0, 0.0])
    
    # 使用nanmean对每个生物标志物分别计算非NaN值的平均
    cn_y0s_array = np.array(cn_y0s)  # shape: (num_cn_patients, 4)
    avg_y0 = np.nanmean(cn_y0s_array, axis=0)
    
    # 如果某个标志物全是NaN，用0填充
    avg_y0 = np.nan_to_num(avg_y0, nan=0.0)
    
    print(f"计算出的CN群体平均初始值（非NaN，归一化后）: {avg_y0}")
    return avg_y0

# --- 2. 用Sigmoid函数拟合人群散点 ---
def sigmoid(s, a, b, c, d):
    """Sigmoid函数定义"""
    return a / (1.0 + np.exp(-b * (s - c))) + d

def fit_sigmoids(s_data, y_data):
    """为4个biomarker分别拟合sigmoid函数（自动处理NaN）"""
    sigmoid_params = []
    print("正在为4个生物标记物拟合Sigmoid曲线...")
    for k in range(4):
        y_k = y_data[:, k]
        
        # 去掉NaN值
        valid_mask = ~np.isnan(y_k)
        s_k_valid = s_data[valid_mask]
        y_k_valid = y_k[valid_mask]
        
        print(f"  - Biomarker {k+1}: {len(y_k_valid)}/{len(y_k)} 个有效数据点")
        
        # 为curve_fit提供一个较好的初始猜测值
        p0 = [
            np.max(y_k_valid) - np.min(y_k_valid),  # a: 幅度
            10,                                     # b: 斜率
            np.median(s_k_valid),                    # c: 中心点
            np.min(y_k_valid)                        # d: 垂直偏移
        ]
        try:
            params, _ = curve_fit(sigmoid, s_k_valid, y_k_valid, p0=p0, maxfev=10000)
            sigmoid_params.append(params)
            print(f"    拟合成功。")
        except RuntimeError:
            print(f"    拟合失败，将使用初始值。")
            sigmoid_params.append(p0)
            
    return np.array(sigmoid_params)

# --- 3. 定义神经网络模型 ---
class ODENet(nn.Module):
    """单隐藏层神经网络用于学习dy/ds"""
    def __init__(self, input_dim=4, hidden_dim=4, output_dim=4):
        super(ODENet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.Tanh()
        
    def forward(self, y):
        """
        输入: y (batch_size, 4) - 4个生物标记物的值
        输出: dy/ds (batch_size, 4) - 4个导数
        """
        h = self.activation(self.fc1(y))
        dyds = self.activation(self.fc3(h))
        return dyds


class ODEModel(nn.Module):
    """ODE模型包装器，用于torchdiffeq"""
    def __init__(self, fnn_model):
        super().__init__()
        self.fnn = fnn_model
        
    def forward(self, t, y):
        """torchdiffeq的函数签名是 func(t, y)"""
        return self.fnn(y)


def get_sigmoid_derivatives(s_grid, params):
    """计算sigmoid函数在网格点上的值和解析导数"""
    y_on_grid = np.zeros((len(s_grid), 4))
    dyds_on_grid = np.zeros((len(s_grid), 4))
    
    for k in range(4):
        a, b, c, d = params[k]
        exp_term = np.exp(-b * (s_grid - c))
        y_on_grid[:, k] = a / (1.0 + exp_term) + d
        dyds_on_grid[:, k] = (a * b * exp_term) / ((1.0 + exp_term)**2)
        
    return y_on_grid, dyds_on_grid

def train_neural_network(y_target, dyds_target, epochs=20000, lr=1e-4, l1_lambda=1e-5):
    """训练神经网络来拟合sigmoid导数，使用Lasso稀疏化（L1正则化）"""
    print("正在训练神经网络模型（带L1稀疏化）...")
    print(f"L1正则化系数: {l1_lambda}")
    
    # 转换为PyTorch张量
    y_tensor = torch.tensor(y_target, dtype=torch.float32)
    dyds_tensor = torch.tensor(dyds_target, dtype=torch.float32)
    
    # 创建模型
    model = ODENet()
    
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 训练循环
    model.train()
    progress_bar = tqdm(range(epochs), desc="预训练神经网络", ncols=None)
    
    for epoch in progress_bar:
        optimizer.zero_grad()
        
        # 前向传播
        dyds_pred = model(y_tensor)
        
        # 计算MSE损失
        mse_loss = criterion(dyds_pred, dyds_tensor)
        
        # 添加L1正则化（Lasso稀疏化）
        l1_penalty = 0.0
        for param in model.parameters():
            l1_penalty += torch.sum(torch.abs(param))
        
        # 总损失 = MSE损失 + L1惩罚
        total_loss = mse_loss + l1_lambda * l1_penalty
        
        # 反向传播和优化
        total_loss.backward()
        optimizer.step()
        
        # 更新进度条
        progress_bar.set_postfix({
            'MSE': f'{mse_loss.item():.6f}',
            'L1': f'{l1_penalty.item():.6f}',
            'Total': f'{total_loss.item():.6f}'
        })
    
    print("神经网络训练完成！")
    model.eval()
    return model

# --- 4. 绘图与ODE求解 ---
def ode_system(y, s, model):
    """定义神经网络ODE系统，供求解器使用"""
    # 将numpy数组转换为tensor
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(0)  # (1, 4)
    
    # 使用神经网络预测导数
    with torch.no_grad():
        dyds_tensor = model(y_tensor)
    
    # 转换回numpy数组
    dyds = dyds_tensor.squeeze(0).numpy()
    
    return dyds

def plot_results(s_pop, y_pop, stages_pop, s_grid, sigmoid_params, model, y0_norm):
    """绘制最终结果图"""
    print("正在生成最终结果图...")
    # 反归一化，准备绘图
    y_pop_orig = pc.inv_nor(y_pop)

    # 计算Sigmoid函数值（直接计算）
    y_sigmoid_grid_norm, _ = get_sigmoid_derivatives(s_grid, sigmoid_params)
    y_sigmoid_grid_orig = pc.inv_nor(y_sigmoid_grid_norm)
    
    # 神经网络轨迹：从sigmoid在起点的值开始积分（使用相同的初值）
    y0_sigmoid = y_sigmoid_grid_norm[0]  # sigmoid在s_grid起点的值
    print(f"使用相同的初始值 (归一化): {y0_sigmoid}")
    y_nn_traj_norm = odeint(ode_system, y0_sigmoid, s_grid, args=(model,))
    y_nn_traj_orig = pc.inv_nor(y_nn_traj_norm)

    TITLES = ['Aβ (A)', 'p-Tau (T)', 'N', 'Cognition (C)']
    colors = {'CN': 'orange', 'LMCI': 'green', 'AD': 'blue', 'Other': 'grey'}
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flat

    # 创建散点图所需的数据
    unique_stages = np.unique(stages_pop)
    scatter_data = {}
    for stage in unique_stages:
        mask = np.array(stages_pop) == stage
        scatter_data[stage] = (s_pop[mask], y_pop_orig[mask])

    for k in range(4):
        ax = axes[k]
        
        # 绘制各阶段散点
        for stage in unique_stages:
            s_vals, y_vals = scatter_data[stage]
            ax.scatter(s_vals, y_vals[:, k], s=15, alpha=0.5, c=colors[stage], label=stage)

        # 绘制Sigmoid拟合曲线
        ax.plot(s_grid, y_sigmoid_grid_orig[:, k], 'r-', lw=2.5, label='Sigmoid Fit', zorder=3)
        
        # 绘制神经网络ODE轨迹（从相同初始值开始）
        ax.plot(s_grid, y_nn_traj_orig[:, k], 'k--', lw=2.5, label='Neural ODE', zorder=3)

        ax.set_xlabel('Disease Progression Score (s)')
        ax.set_ylabel(TITLES[k])
        
        # 设置横轴范围与s_grid一致
        ax.set_xlim(s_grid.min(), s_grid.max())
        
        ax.legend()
        ax.grid(True, alpha=0.4)
        ax.set_title(f'Trajectory for {TITLES[k]}')

    fig.suptitle('AD Biomarker Trajectories: Sigmoid Fit vs. Neural ODE', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('pretrain.png')
    plt.show()


def plot_parameter_distribution(model):
    """绘制神经网络参数分布的直方图"""
    print("\n正在生成神经网络参数分布图...")
    
    # 收集所有参数值
    all_params = []
    param_counts = {}
    
    for name, param in model.named_parameters():
        param_values = param.data.cpu().numpy().flatten()
        all_params.extend(param_values)
        param_counts[name] = len(param_values)
    
    all_params = np.array(all_params)
    
    # 统计信息
    print(f"\n神经网络参数统计:")
    print(f"  总参数数量: {len(all_params)}")
    print(f"  均值: {all_params.mean():.6f}")
    print(f"  标准差: {all_params.std():.6f}")
    print(f"  最小值: {all_params.min():.6f}")
    print(f"  最大值: {all_params.max():.6f}")
    
    # 统计接近0的参数（稀疏性指标）
    threshold = 1e-3
    near_zero = np.abs(all_params) < threshold
    sparsity = near_zero.sum() / len(all_params) * 100
    print(f"  接近0的参数比例 (|w| < {threshold}): {sparsity:.2f}%")
    
    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 子图1: 所有参数的直方图
    ax1 = axes[0, 0]
    ax1.hist(all_params, bins=100, alpha=0.7, color='blue', edgecolor='black')
    ax1.set_xlabel('Parameter Value')
    ax1.set_ylabel('Frequency')
    ax1.set_title('All Parameters Distribution')
    ax1.grid(True, alpha=0.3)
    ax1.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero')
    ax1.legend()
    
    # 子图2: 参数绝对值的直方图（对数刻度）
    ax2 = axes[0, 1]
    abs_params = np.abs(all_params)
    ax2.hist(abs_params, bins=100, alpha=0.7, color='green', edgecolor='black')
    ax2.set_xlabel('|Parameter Value|')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Absolute Parameter Values (Log Scale)')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    
    # 子图3: 各层参数的箱线图
    ax3 = axes[1, 0]
    layer_params = []
    layer_names = []
    for name, param in model.named_parameters():
        layer_params.append(param.data.cpu().numpy().flatten())
        layer_names.append(name)
    
    bp = ax3.boxplot(layer_params, labels=layer_names, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    ax3.set_ylabel('Parameter Value')
    ax3.set_title('Parameter Distribution by Layer')
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='x', rotation=45)
    
    # 子图4: 稀疏性统计（各层）
    ax4 = axes[1, 1]
    layer_sparsity = []
    for param_vals in layer_params:
        layer_near_zero = np.abs(param_vals) < threshold
        layer_sparsity.append(layer_near_zero.sum() / len(param_vals) * 100)
    
    bars = ax4.bar(range(len(layer_names)), layer_sparsity, color='orange', alpha=0.7, edgecolor='black')
    ax4.set_xlabel('Layer')
    ax4.set_ylabel(f'Sparsity (% with |w| < {threshold})')
    ax4.set_title('Sparsity by Layer')
    ax4.set_xticks(range(len(layer_names)))
    ax4.set_xticklabels(layer_names, rotation=45, ha='right')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 在柱状图上显示数值
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('parameter_distribution.png')
    print("参数分布图已保存到 parameter_distribution.png")
    plt.show()


if __name__ == '__main__':
    # --- 执行Pipeline ---
    # 1. 分配DPS参数并获取人群数据点
    patient_data, s_pop, y_pop_norm, stages_pop = assign_dps_params(csf_dict, stage_dict)

    # 1.5. 从CN群体计算平均初始值
    y0_cn_avg_norm = get_cn_average_y0(patient_data)

    # 2. 拟合Sigmoid函数
    sigmoid_params = fit_sigmoids(s_pop, y_pop_norm)

    # 3. 训练神经网络模型
    # 根据实际数据范围动态设置s_grid，并稍微扩展范围
    s_min, s_max = s_pop.min(), s_pop.max()
    s_margin = (s_max - s_min) * 0.1  # 扩展10%的边距
    s_grid = np.linspace(s_min - s_margin, s_max + s_margin, 300)
    print(f"s_grid范围: [{s_grid.min():.2f}, {s_grid.max():.2f}]")
    
    y_sigmoid_grid_norm, dyds_sigmoid_grid_norm = get_sigmoid_derivatives(s_grid, sigmoid_params)
    nn_model = train_neural_network(y_sigmoid_grid_norm, dyds_sigmoid_grid_norm)
    
    # 4. 求解ODE并绘图
    plot_results(s_pop, y_pop_norm, stages_pop, s_grid, sigmoid_params, nn_model, y0_cn_avg_norm)
    
    # 5. 保存模型参数
    # 保存神经网络模型
    torch.save(nn_model.state_dict(), 'fnn_pretrain.pth')
    print("神经网络模型已保存到 fnn_pretrain.pth")

    # 保存DPS参数
    dps_params_dict = {}
    for pid, data in patient_data.items():
        dps_params_dict[pid] = {'a': data['a'], 'b': data['b']}
    torch.save(dps_params_dict, 'dps_pretrain.pth')
    print("DPS参数已保存到 dps_pretrain.pth")
    
    # 6. 绘制神经网络参数分布图
    plot_parameter_distribution(nn_model)
    
    print("\n流程执行完毕。")