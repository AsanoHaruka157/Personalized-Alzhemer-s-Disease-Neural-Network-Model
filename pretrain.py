import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import odeint
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
        'CN': (-5, 5),
        'LMCI': (0, 10),
        'AD': (5, 15),
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
    计算CN（认知正常）群体在第一次访问时的平均生物标记物值。
    """
    cn_y0s = []
    for pid, data in patient_data.items():
        if data['stage'] == 'CN':
            cn_y0s.append(data['y'][0]) # 取第一次访问的数据
    
    if not cn_y0s:
        print("警告：未找到CN患者数据，将使用默认初始值 [0.1, 0, 0, 0]。")
        return np.array([0.1, 0.0, 0.0, 0.0])
        
    avg_y0 = np.mean(cn_y0s, axis=0)
    print(f"计算出的CN群体平均初始值 (归一化后): {avg_y0}")
    return avg_y0

# --- 2. 用Sigmoid函数拟合人群散点 ---
def sigmoid(s, a, b, c, d):
    """Sigmoid函数定义"""
    return a / (1.0 + np.exp(-b * (s - c))) + d

def fit_sigmoids(s_data, y_data):
    """为4个biomarker分别拟合sigmoid函数"""
    sigmoid_params = []
    print("正在为4个生物标记物拟合Sigmoid曲线...")
    for k in range(4):
        y_k = y_data[:, k]
        # 为curve_fit提供一个较好的初始猜测值
        p0 = [
            np.max(y_k) - np.min(y_k),  # a: 幅度
            0.1,                        # b: 斜率
            np.median(s_data),          # c: 中心点
            np.min(y_k)                 # d: 垂直偏移
        ]
        try:
            params, _ = curve_fit(sigmoid, s_data, y_k, p0=p0, maxfev=10000)
            sigmoid_params.append(params)
            print(f"  - Biomarker {k+1} 拟合成功。")
        except RuntimeError:
            print(f"  - Biomarker {k+1} 拟合失败，将使用初始值。")
            sigmoid_params.append(p0)
            
    return np.array(sigmoid_params)

# --- 3. 定义神经网络模型以匹配Sigmoid导数 ---
class ODENet(nn.Module):
    """单隐藏层神经网络用于学习dy/ds"""
    def __init__(self, input_dim=4, hidden_dim=512, output_dim=4):
        super(ODENet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.Tanh()  # 使用Tanh激活函数
        
    def forward(self, y):
        """
        输入: y (batch_size, 4) - 4个生物标记物的值
        输出: dy/ds (batch_size, 4) - 4个导数
        """
        h = self.activation(self.fc1(y))
        dyds = self.fc2(h)
        return dyds

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

def train_neural_network(y_target, dyds_target, epochs=10000, lr=1e-4):
    """训练神经网络来拟合sigmoid导数"""
    print("正在训练神经网络模型...")
    
    # 转换为PyTorch张量
    y_tensor = torch.tensor(y_target, dtype=torch.float32)
    dyds_tensor = torch.tensor(dyds_target, dtype=torch.float32)
    
    # 创建模型
    model = ODENet(input_dim=4, hidden_dim=256, output_dim=4)
    
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 训练循环
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # 前向传播
        dyds_pred = model(y_tensor)
        
        # 计算损失
        loss = criterion(dyds_pred, dyds_tensor)
        
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        
        # 打印进度
        if (epoch + 1) % 500 == 0:
            print(f"  Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")
    
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
        ax.plot(s_grid, y_nn_traj_orig[:, k], 'k--', lw=2.5, label='Neural Network ODE', zorder=3)

        ax.set_xlabel('Disease Progression Score (s)')
        ax.set_ylabel(TITLES[k])
        
        # 设置横轴范围与s_grid一致
        ax.set_xlim(s_grid.min(), s_grid.max())
        
        ax.legend()
        ax.grid(True, alpha=0.4)
        ax.set_title(f'Trajectory for {TITLES[k]}')

    fig.suptitle('AD Biomarker Trajectories: Sigmoid Fit vs. Neural Network ODE', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('pretrain.png')
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
    torch.save(nn_model.state_dict(), 'fnn.pth')
    print("神经网络模型已保存到 fnn.pth")

    # 保存DPS参数
    dps_params_dict = {}
    for pid, data in patient_data.items():
        dps_params_dict[pid] = {'a': data['a'], 'b': data['b']}
    torch.save(dps_params_dict, 'dps.pth')
    print("DPS参数已保存到 dps.pth")
    
    print("\n流程执行完毕。")