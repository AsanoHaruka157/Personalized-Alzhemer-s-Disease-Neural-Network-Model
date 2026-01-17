import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from tqdm import tqdm
import pccmnn as pc # 假设您有这个文件来加载和反归一化数据

# --- 0. 数据加载和准备 ---
# 加载数据和患者分期信息
csf_dict = pc.load_data()
stage_dict = pc.load_stage_dict()
print(f"成功加载 {len(csf_dict)} 位患者的数据。")

# --- 1. 为每位患者分配DPS变换参数（可优化版本）---
def assign_dps_params(csf_dict, stage_dict):
    """
    为每位患者创建可优化的DPS变换参数 a 和 b。
    a: CN=1, LMCI=2, AD=4 的初始值，但可优化
    b: 随机初始化，但可优化
    """
    patient_data = {}
    # 按照您的要求更新 s_ranges（用于初始化b参数）
    s_ranges = {
        'CN': (-10, 0),
        'LMCI': (-2, 8),
        'AD': (5, 20),
        'Other': (-10, 20) # 为其他类型提供一个默认范围
    }
    a_init_values = {'CN': 1.0, 'LMCI': 2.0, 'AD': 4.0, 'Other': 1.0}

    # 为每个患者创建可优化的a和b参数
    dps_params = {}
    for pid, sample in csf_dict.items():
        stage = stage_dict.get(pid, 'Other')
        t = sample[:, 0]

        # 初始化a参数（可优化）
        a_init = a_init_values[stage]
        a_param = nn.Parameter(torch.tensor(a_init, dtype=torch.float32))

        # 初始化b参数（可优化），使其初始s值落在合理区间
        s_min, s_max = s_ranges[stage]
        t_initial = t[0]
        s_initial_target = np.random.uniform(s_min, s_max)
        b_init = s_initial_target - a_init * t_initial
        b_param = nn.Parameter(torch.tensor(b_init, dtype=torch.float32))

        dps_params[pid] = {'a': a_param, 'b': b_param, 'stage': stage}

    return dps_params

def compute_s_values(csf_dict, dps_params):
    """
    根据当前的DPS参数计算所有患者的s值
    """
    patient_data = {}
    all_s_points = []
    all_y_points = []
    all_stages = []

    for pid, sample in csf_dict.items():
        t = sample[:, 0]
        y = sample[:, 1:5]

        params = dps_params[pid]
        a = params['a']
        b = params['b']
        stage = params['stage']

        # 计算s = a*t + b
        s = a * torch.tensor(t, dtype=torch.float32) + b

        patient_data[pid] = {'t': t, 'y': y, 's': s.detach().numpy(), 'stage': stage, 'a': a.item(), 'b': b.item()}

        all_s_points.append(s.detach().numpy())
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

        params, _ = curve_fit(sigmoid, s_k_valid, y_k_valid, p0=p0, maxfev=10000)
        sigmoid_params.append(params)

    return np.array(sigmoid_params)

class DifferentiableSigmoidFit(nn.Module):
    """可微分的sigmoid拟合模型"""
    def __init__(self, num_biomarkers=4):
        super().__init__()
        # 为每个biomarker创建可训练的sigmoid参数 [a, b, c, d]
        # a: 幅度, b: 斜率, c: 中心点, d: 偏移
        # 根据AD疾病特征设置初始斜率：
        # Biomarker 0 (Abeta): 负斜率 (下降)
        # Biomarker 1 (p-Tau): 正斜率 (上升)
        # Biomarker 2 (N): 负斜率 (下降)
        # Biomarker 3 (C): 正斜率 (上升)
        initial_params = [
            [1.0, -1.0, 0.0, 0.0],  # Abeta: 负斜率 (下降)
            [1.0, 1.0, 0.0, 0.0],   # p-Tau: 正斜率 (上升)
            [1.0, -1.0, 0.0, 0.0],  # N: 负斜率 (下降)
            [1.0, 1.0, 0.0, 0.0]    # C: 正斜率 (上升)
        ]
        self.sigmoid_params = nn.ParameterList([
            nn.Parameter(torch.tensor(initial_params[i], dtype=torch.float32))
            for i in range(num_biomarkers)
        ])

    def forward(self, s):
        """
        计算sigmoid函数值
        s: (batch_size,) - s值数组
        返回: (batch_size, 4) - 每个biomarker在每个s值处的预测
        """
        s = s.unsqueeze(-1) if s.dim() == 1 else s  # (batch_size, 1)
        results = []
        for i in range(4):
            a, b, c, d = self.sigmoid_params[i]
            # 广播计算: s.shape = (batch_size, 1), 参数都是标量
            y = a / (1.0 + torch.exp(-b * (s - c))) + d  # (batch_size, 1)
            results.append(y.squeeze(-1))  # (batch_size,)
        return torch.stack(results, dim=-1)  # (batch_size, 4)

def fit_sigmoids_differentiable(s_data, y_data, epochs=1000, lr=1e-3):
    """可微分版本的sigmoid拟合，使用改进的损失函数"""

    # 转换为PyTorch张量
    s_tensor = torch.tensor(s_data, dtype=torch.float32)
    y_tensor = torch.tensor(y_data, dtype=torch.float32)

    # 计算数据范围，用于防止坍缩成直线的约束
    s_range = s_data.max() - s_data.min()

    # 创建模型
    model = DifferentiableSigmoidFit()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    mse_criterion = nn.MSELoss()

    best_loss = float('inf')
    best_params = None

    # 训练
    model.train()
    progress_bar = tqdm(range(epochs), desc="Sigmoid拟合", ncols=None, leave=False)

    for epoch in progress_bar:
        optimizer.zero_grad()

        # 前向传播
        y_pred = model(s_tensor)

        # 基础损失：数据拟合损失
        valid_mask = ~torch.isnan(y_tensor)
        fit_loss = mse_criterion(y_pred[valid_mask], y_tensor[valid_mask])

        # 改进的损失项
        total_loss = fit_loss.clone()

        # 1. 锁定核心参数：B值接近1的正则化
        b_regularization = 0.0
        slope_max_penalty = 0.0
        curvature_penalty = 0.0

        for i in range(4):
            a, b, c, d = model.sigmoid_params[i]

            # B值正则化：根据biomarker特征设置目标B值
            # Biomarker 0 (Abeta): 目标B = -1 (下降)
            # Biomarker 1 (p-Tau): 目标B = 1 (上升)
            # Biomarker 2 (N): 目标B = -1 (下降)
            # Biomarker 3 (C): 目标B = 1 (上升)
            target_b_values = [-1.0, 1.0, -1.0, 1.0]  # 对应4个biomarker的目标B值
            target_b = target_b_values[i]
            b_regularization += (b - target_b) ** 2

            # 2. 控制导数极值：防止太陡或太扁
            # Sigmoid在拐点处的最大斜率：Slope_max = (a*b)/4
            # 这是因为sigmoid函数 y = a/(1+exp(-b(x-c))) + d 的导数在x=c处最大值为 (a*b)/4
            slope_max = torch.abs(a * b) / 4.0
            k_min, k_max = 0.1, 2.0  # 斜率合理范围

            # 防止太扁（坍缩成直线）
            slope_penalty_min = torch.relu(k_min - slope_max) ** 2
            # 防止太陡（像墙一样陡）
            slope_penalty_max = torch.relu(slope_max - k_max) ** 2
            slope_max_penalty += slope_penalty_min + slope_penalty_max

            # 3. 防止坍缩成直线的特殊技巧：控制B与数据范围的关系
            # 理想的B值应该在 4/(x_range) 到 6/(x_range) 之间
            ideal_b_min = 4.0 / s_range
            ideal_b_max = 6.0 / s_range
            b_range_penalty = torch.relu(ideal_b_min - torch.abs(b)) ** 2 + torch.relu(torch.abs(b) - ideal_b_max) ** 2
            curvature_penalty += b_range_penalty

        # 组合损失
        lambda_b = 1.0          # B值正则化权重
        lambda_slope = 0.5      # 斜率约束权重
        lambda_curvature = 0.2  # 曲率约束权重

        regularization_loss = lambda_b * b_regularization + \
                             lambda_slope * slope_max_penalty + \
                             lambda_curvature * curvature_penalty

        total_loss += regularization_loss

        # 反向传播
        total_loss.backward()
        optimizer.step()

        # 保存最佳参数
        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            best_params = [param.clone() for param in model.sigmoid_params]

        # 更新进度条
        progress_bar.set_postfix({
            'Fit': f'{fit_loss.item():.6f}',
            'Reg': f'{regularization_loss.item():.6f}',
            'Total': f'{total_loss.item():.6f}'
        })

    # 使用最佳参数
    if best_params is not None:
        for i in range(4):
            model.sigmoid_params[i].data = best_params[i].data

    model.eval()
    # 返回参数
    sigmoid_params = []
    for i in range(4):
        params = model.sigmoid_params[i].detach().numpy()
        sigmoid_params.append(params)

    return np.array(sigmoid_params)

def train_sigmoid_curves(csf_dict, dps_params, num_iterations=20):
    """
    只训练sigmoid曲线，不再优化DPS参数
    使用改进的损失函数确保sigmoid曲线有良好的形态
    """
    print(f"\n开始训练sigmoid曲线，共 {num_iterations} 次迭代...")

    best_curvature = -float('inf')
    best_sigmoid_params = None
    first_iteration_done = False

    progress_bar = tqdm(range(num_iterations), desc="Sigmoid训练", ncols=None)

    for iteration in progress_bar:
        # 使用固定的DPS参数计算s值并拟合sigmoid
        patient_data, s_pop, y_pop_norm, stages_pop = compute_s_values(csf_dict, dps_params)
        sigmoid_params = fit_sigmoids_differentiable(s_pop, y_pop_norm)

        # 计算s_grid并评估曲率
        s_min, s_max = s_pop.min(), s_pop.max()
        s_margin = (s_max - s_min) * 0.1
        s_grid = np.linspace(s_min - s_margin, s_max + s_margin, 300)

        curvature_grid = compute_sigmoid_curvature(s_grid, sigmoid_params)
        total_curvature = get_total_curvature(curvature_grid)

        # 保存最佳结果
        # 第一次迭代总是保存，即使曲率是NaN
        if not first_iteration_done or (not np.isnan(total_curvature) and total_curvature > best_curvature):
            best_curvature = total_curvature if not np.isnan(total_curvature) else best_curvature
            best_sigmoid_params = sigmoid_params.copy()
            first_iteration_done = True

        # 更新进度条
        progress_bar.set_postfix({
            '曲率': f'{total_curvature:.2f}' if not np.isnan(total_curvature) else 'NaN'
        })

    print(f"\nSigmoid训练完成！最佳曲率: {best_curvature:.6f}")

    return best_sigmoid_params

# --- 3. 定义神经网络模型 ---
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

def compute_sigmoid_curvature(s_grid, params):
    """计算sigmoid函数在网格点上的曲率（使用二阶导数的绝对值作为曲率指标）"""
    curvature_on_grid = np.zeros((len(s_grid), 4))

    for k in range(4):
        a, b, c, d = params[k]

        # 数值稳定性：限制参数范围
        b = np.clip(b, 0.1, 10.0)  # 限制b在合理范围内
        a = np.clip(a, -10.0, 10.0)  # 限制a的范围

        # 数值稳定的指数计算
        exp_arg = -b * (s_grid - c)
        exp_arg = np.clip(exp_arg, -50.0, 50.0)  # 防止溢出
        exp_term = np.exp(exp_arg)
        denominator = (1.0 + exp_term) ** 3

        # 二阶导数：d²y/ds² = a*b²*exp(-b*(s-c))*(exp(-b*(s-c))-1) / (1 + exp(-b*(s-c)))^3
        d2yds2 = a * b**2 * exp_term * (exp_term - 1) / denominator

        # 使用二阶导数的绝对值作为曲率指标
        curvature_on_grid[:, k] = np.abs(d2yds2)

    return curvature_on_grid

def get_total_curvature(curvature_grid):
    """计算总曲率（所有biomarker在所有网格点的曲率之和）"""
    return np.sum(curvature_grid)

# --- 4. 绘图 ---
def plot_results(s_pop, y_pop, stages_pop, s_grid, sigmoid_params):
    """绘制最终结果图"""
    print("📊 正在生成最终结果图...")

    # 反归一化，准备绘图
    y_pop_orig = pc.inv_nor(y_pop)

    # 计算Sigmoid函数值（直接计算）
    y_sigmoid_grid_norm, _ = get_sigmoid_derivatives(s_grid, sigmoid_params)
    y_sigmoid_grid_orig = pc.inv_nor(y_sigmoid_grid_norm)

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

        ax.set_xlabel('Disease Progression Score (s)')
        ax.set_ylabel(TITLES[k])
        
        # 设置横轴范围与s_grid一致
        ax.set_xlim(s_grid.min(), s_grid.max())
        
        ax.legend()
        ax.grid(True, alpha=0.4)
        ax.set_title(f'Trajectory for {TITLES[k]}')

    fig.suptitle('AD Biomarker Trajectories: Sigmoid Fit', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('pretrain.png')
    plt.show()



if __name__ == '__main__':
    # --- 执行Pipeline ---
    print("开始预训练流程...")

    # 1. 初始化可优化的DPS参数
    print("初始化DPS参数...")
    dps_params = assign_dps_params(csf_dict, stage_dict)

    # 2. 训练sigmoid曲线（固定DPS参数）
    print("训练sigmoid曲线...")
    sigmoid_params = train_sigmoid_curves(csf_dict, dps_params, num_iterations=20)

    # 3. 计算最终的s值和患者数据
    print("计算最终s值...")
    patient_data, s_pop, y_pop_norm, stages_pop = compute_s_values(csf_dict, dps_params)

    # 3.5. 从CN群体计算平均初始值
    print("计算CN群体平均初始值...")
    y0_cn_avg_norm = get_cn_average_y0(patient_data)

    # 4. 生成结果图表
    print("生成结果图表...")
    # 根据实际数据范围动态设置s_grid，并稍微扩展范围
    s_min, s_max = s_pop.min(), s_pop.max()
    s_margin = (s_max - s_min) * 0.1  # 扩展10%的边距
    s_grid = np.linspace(s_min - s_margin, s_max + s_margin, 300)

    plot_results(s_pop, y_pop_norm, stages_pop, s_grid, sigmoid_params)

    # 5. 保存DPS参数
    print("保存模型参数...")
    dps_params_dict = {}
    for pid, params in dps_params.items():
        dps_params_dict[pid] = {'a': params['a'].item(), 'b': params['b'].item()}
    torch.save(dps_params_dict, 'dps_pretrain.pth')
    print("DPS参数已保存到 dps_pretrain.pth")