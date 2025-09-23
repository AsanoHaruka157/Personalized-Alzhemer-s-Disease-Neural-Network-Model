import torch
import torch.nn as nn
import numpy as np
import json
import pccmnn as pc

try:
    from torchdiffeq import odeint as torch_odeint
except ImportError:
    raise ImportError("Please install torchdiffeq to run this script: pip install torchdiffeq")

# --- 定义与 main.py 相同的模型结构 ---
# 确保这里的模型定义与您训练和保存'fpp.pth'时使用的完全一致
class ODEModel(nn.Module):
    def __init__(self, hidden_dim=1024):
        super().__init__()
        # 神经网络 f(y)
        self.net = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 4),
            nn.Tanh()
        )
        self.output_scaler = nn.Parameter(torch.tensor([0.1]))
        # 多项式 p(y)
        self.wA = nn.Parameter(torch.zeros(3))
        self.wT = nn.Parameter(torch.zeros(6))
        self.wN = nn.Parameter(torch.zeros(6))
        self.wC = nn.Parameter(torch.zeros(6))

    def combined_dynamics(self, s, y):
        poly_A = torch.stack([torch.ones_like(y[..., 0]), y[..., 0], y[..., 0]**2], dim=-1) @ self.wA
        poly_T = torch.stack([torch.ones_like(y[..., 1]), y[..., 1], y[..., 1]**2, y[..., 0], y[..., 0]**2, y[..., 0]*y[..., 1]], dim=-1) @ self.wT
        poly_N = torch.stack([torch.ones_like(y[..., 2]), y[..., 2], y[..., 2]**2, y[..., 1], y[..., 1]**2, y[..., 1]*y[..., 2]], dim=-1) @ self.wN
        poly_C = torch.stack([torch.ones_like(y[..., 3]), y[..., 3], y[..., 3]**2, y[..., 2], y[..., 2]**2, y[..., 2]*y[..., 3]], dim=-1) @ self.wC
        poly_dyds = torch.stack([poly_A, poly_T, poly_N, poly_C], dim=-1)
        
        net_dyds = self.net(y) * self.output_scaler
        return net_dyds + poly_dyds

    def forward(self, s_grid, y0):
        # 确保所有参数都追踪梯度
        for param in self.parameters():
            param.requires_grad = True
        return torch_odeint(self.combined_dynamics, y0, s_grid, method='dopri5', rtol=1e-4, atol=1e-5)

# --- 主程序 ---
if __name__ == '__main__':
    # --- 1. 载入训练好的模型和数据 ---
    print("正在载入已训练的模型和数据...")
    model = ODEModel()
    try:
        model.load_state_dict(torch.load('fpp.pth'))
        print("成功载入模型 'fpp.pth'。")
    except FileNotFoundError:
        print("错误: 未找到模型文件 'fpp.pth'。请先运行 main.py 进行训练。")
        exit()

    # 将模型设置为评估模式，但梯度计算仍会进行
    model.eval() 
    
    # 获取初始条件 y0
    stage_dict = pc.load_stage_dict()
    csf_dict = pc.load_data()
    patient_data = {pid: {"y0": torch.from_numpy(sample[:1, 1:5]).float().squeeze(0)} 
                    for pid, sample in csf_dict.items()}
    y0_cn_avg = torch.stack([data['y0'] for pid, data in patient_data.items() 
                           if stage_dict.get(pid) == 'CN']).mean(dim=0)
    print(f"使用CN群体的平均初始值: {y0_cn_avg.numpy()}")

    # --- 2. 执行前向和后向传播以计算梯度 ---
    print("正在执行前向和后向传播以计算梯度...")
    
    # 定义我们关心的输出点
    s_of_interest = torch.tensor(15.0)
    s_grid = torch.linspace(-10, s_of_interest.item(), 100)
    
    # 清除旧的梯度
    model.zero_grad()
    
    # 前向传播
    trajectory = model(s_grid, y0_cn_avg)
    final_output = trajectory[-1, :] # 获取在s_of_interest时刻的 [A, T, N, C] 输出
    
    # 定义一个标量目标函数，用于反向传播。
    # 这里我们取四个输出绝对值的和，代表模型的“总体效应”
    scalar_target = torch.sum(torch.abs(final_output))
    
    # 反向传播，计算梯度
    scalar_target.backward()
    print("梯度计算完成。")

    # --- 3. 收集并分析梯度 ---
    print("\n--- 基于梯度的敏感性分析结果 ---")
    results = []
    param_index_counter = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            # 将多维梯度展平
            grads = param.grad.abs().detach().numpy().flatten()
            for i, grad_val in enumerate(grads):
                results.append({
                    'name': f"{name}_{i}",
                    'index': param_index_counter + i,
                    'sensitivity': grad_val
                })
            param_index_counter += len(grads)

    # 按敏感性（梯度绝对值）降序排序
    sorted_results = sorted(results, key=lambda x: x['sensitivity'], reverse=True)

    # --- 4. 解读、筛选并保存结果 ---
    # 打印最有影响的 TOP_N 个参数
    TOP_N = 20
    print(f"\n最重要的 {TOP_N} 个参数 (按梯度绝对值排序):")
    print("-" * 60)
    print(f"{'排名':<5}{'参数名称':<35}{'索引':<10}{'敏感度':<15}")
    print("-" * 60)
    for i, res in enumerate(sorted_results[:TOP_N]):
        print(f"{i+1:<5}{res['name']:<35}{res['index']:<10}{res['sensitivity']:.4e}")
    
    # 筛选出重要的参数用于保存
    # 使用动态阈值：例如，保存所有敏感度在前2%的参数
    num_to_save = int(len(sorted_results) * 0.02)
    sensitive_params_to_save = [
        {'name': res['name'], 'index': res['index']} 
        for res in sorted_results[:num_to_save]
    ]

    # 保存为JSON文件
    output_filename = 'sensitive_params.json'
    with open(output_filename, 'w') as f:
        json.dump(sensitive_params_to_save, f, indent=4)

    print("-" * 60)
    print(f"\n分析完成。已筛选出最重要的 {len(sensitive_params_to_save)} 个参数 (排名前2%)。")
    print(f"结果已保存到 '{output_filename}' 文件中。")
