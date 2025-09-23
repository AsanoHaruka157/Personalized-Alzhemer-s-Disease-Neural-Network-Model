import torch
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
        'CN': (-10, 5),
        'LMCI': (0, 10),
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

# --- 3. 拟合多项式模型以匹配Sigmoid导数 ---
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

def build_feature_matrix(y):
    """根据y构建多项式模型的特征矩阵 (Phi)"""
    A, T, N, C = y[:, 0], y[:, 1], y[:, 2], y[:, 3]
    
    # 对应论文中的公式(6)
    # dA/ds = wA0 + wA1*A + wA2*A^2
    phi_A = np.stack([np.ones_like(A), A, A**2], axis=1)
    
    # dT/ds = wT0 + wT1*T + wT2*T^2 + wT3*A + wT4*A^2 + wT5*A*T
    phi_T = np.stack([np.ones_like(T), T, T**2, A, A**2, A*T], axis=1)

    # dN/ds = wN0 + wN1*N + wN2*N^2 + wN3*T + wN4*T^2 + wN5*T*N
    phi_N = np.stack([np.ones_like(N), N, N**2, T, T**2, T*N], axis=1)

    # dC/ds = wC0 + wC1*C + wC2*C^2 + wC3*N + wC4*N^2 + wC5*N*C
    phi_C = np.stack([np.ones_like(C), C, C**2, N, N**2, N*C], axis=1)
    
    return [phi_A, phi_T, phi_N, phi_C]


def fit_polynomial_model(y_target, dyds_target):
    """使用最小二乘法求解多项式系数"""
    phi_list = build_feature_matrix(y_target)
    poly_coeffs = []
    print("正在求解多项式模型的系数...")
    for k in range(4):
        phi = phi_list[k]
        dyds = dyds_target[:, k]
        # 使用最小二乘法求解: w = (phi^T * phi)^-1 * phi^T * dyds
        coeffs, _, _, _ = np.linalg.lstsq(phi, dyds, rcond=None)
        poly_coeffs.append(coeffs)
        print(f"  - 方程 {k+1} 系数求解完毕。")
    return poly_coeffs

# --- 4. 绘图与ODE求解 ---
def ode_system(y, s, coeffs):
    """定义多项式ODE系统，供求解器使用"""
    A, T, N, C = y
    
    # 提取系数
    wA, wT, wN, wC = coeffs
    
    # 构建特征向量
    phi_A_vec = np.array([1, A, A**2])
    phi_T_vec = np.array([1, T, T**2, A, A**2, A*T])
    phi_N_vec = np.array([1, N, N**2, T, T**2, T*N])
    phi_C_vec = np.array([1, C, C**2, N, N**2, N*C])
    
    # 计算导数
    dAds = np.dot(wA, phi_A_vec)
    dTds = np.dot(wT, phi_T_vec)
    dNds = np.dot(wN, phi_N_vec)
    dCds = np.dot(wC, phi_C_vec)
    
    return [dAds, dTds, dNds, dCds]

def plot_results(s_pop, y_pop, stages_pop, s_grid, sigmoid_params, poly_coeffs, y0_norm):
    """绘制最终结果图"""
    print("正在生成最终结果图...")
    # 反归一化，准备绘图
    y_pop_orig = pc.inv_nor(y_pop)

    # 计算Sigmoid和多项式轨迹
    y_sigmoid_grid_norm, _ = get_sigmoid_derivatives(s_grid, sigmoid_params)
    y_sigmoid_grid_orig = pc.inv_nor(y_sigmoid_grid_norm)
    
    # 使用传入的 y0_norm 作为初始值
    y_poly_traj_norm = odeint(ode_system, y0_norm, s_grid, args=(poly_coeffs,))
    y_poly_traj_orig = pc.inv_nor(y_poly_traj_norm)

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

        # 绘制Sigmoid轨迹
        ax.plot(s_grid, y_sigmoid_grid_orig[:, k], 'r-', lw=2.5, label='Sigmoid Fit', zorder=3)
        
        # 绘制多项式ODE轨迹
        ax.plot(s_grid, y_poly_traj_orig[:, k], 'k--', lw=2.5, label='Polynomial ODE', zorder=3)

        ax.set_xlabel('Disease Progression Score (s)')
        ax.set_ylabel(TITLES[k])
        ax.legend()
        ax.grid(True, alpha=0.4)
        ax.set_title(f'Trajectory for {TITLES[k]}')

    fig.suptitle('AD Biomarker Trajectories: Sigmoid vs. Polynomial ODE Model', fontsize=16)
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

    # 3. 拟合多项式模型
    s_grid = np.linspace(-10, 20, 300)
    y_sigmoid_grid_norm, dyds_sigmoid_grid_norm = get_sigmoid_derivatives(s_grid, sigmoid_params)
    poly_coeffs = fit_polynomial_model(y_sigmoid_grid_norm, dyds_sigmoid_grid_norm)
    
    # 4. 求解ODE并绘图
    plot_results(s_pop, y_pop_norm, stages_pop, s_grid, sigmoid_params, poly_coeffs, y0_cn_avg_norm)
    
    # 5. 保存模型参数
    # 保存多项式模型系数
    poly_coeffs_dict = {
        'wA': torch.tensor(poly_coeffs[0], dtype=torch.float32),
        'wT': torch.tensor(poly_coeffs[1], dtype=torch.float32),
        'wN': torch.tensor(poly_coeffs[2], dtype=torch.float32),
        'wC': torch.tensor(poly_coeffs[3], dtype=torch.float32),
    }
    torch.save(poly_coeffs_dict, 'poly.pth')
    print("多项式模型系数已保存到 poly.pth")

    # 保存DPS参数
    dps_params_dict = {}
    for pid, data in patient_data.items():
        dps_params_dict[pid] = {'a': data['a'], 'b': data['b']}
    torch.save(dps_params_dict, 'dps.pth')
    print("DPS参数已保存到 dps.pth")
    
    print("\n流程执行完毕。")