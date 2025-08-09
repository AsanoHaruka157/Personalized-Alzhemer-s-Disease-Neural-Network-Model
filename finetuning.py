import torch
import torch.nn as nn
import torch.optim as optim
import copy
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import pccmnn as pc
import os
import pickle
from torch.utils.data import Dataset, DataLoader

# 复制模型定义
class ResidualBlock(nn.Module):
    """残差块"""
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(dim, dim),
        )
        
    def forward(self, x):
        return x + self.block(x)  # 残差连接

class PopulationODE(nn.Module):
    def __init__(self, hidden_dim=32, num_residual_blocks=2):
        super().__init__()
        
        # 1. 输入层：5个输入（4个生物标记物 + s值）-> hidden_dim
        self.input_layer = nn.Linear(5, hidden_dim)
        self.input_activation = nn.Tanh()
        self.input_dropout = nn.Dropout(0.1)
        
        # 2. 残差块序列
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(num_residual_blocks)
        ])
        
        # 3. 输出层：hidden_dim -> 4个生物标记物
        self.output_layer = nn.Linear(hidden_dim, 4)
        self.output_activation = nn.Sigmoid()
        self.output_dropout = nn.Dropout(0.1)
        
        # 4. A为可学习参数
        self.A = nn.Parameter(torch.zeros(4, dtype=torch.float32))

    def f(self, y, s):
        y = y.float()
        s = s.float()
        # 确保s是标量，将其扩展为与y兼容的形状
        if s.dim() == 0:  # s是标量
            s_expanded = s.unsqueeze(0)  # 变成[1]
        else:
            s_expanded = s
        
        # 将y和s连接起来作为网络输入
        z = torch.cat([y, s_expanded], dim=0)  # 连接为[5]的向量
        
        # 前向传播通过残差网络
        x = z.unsqueeze(0)  # 添加batch维度 -> [1, 5]
        
        # 输入层
        x = self.input_layer(x)  # [1, hidden_dim]
        x = self.input_activation(x)
        x = self.input_dropout(x)
        
        # 残差块序列
        for residual_block in self.residual_blocks:
            x = residual_block(x)
        
        # 输出层
        x = self.output_layer(x)  # [1, 4]
        x = self.output_activation(x)
        x = self.output_dropout(x)
        
        net_output = x.squeeze(0)  # 移除batch维度 -> [4]
        
        # 应用ODE动力学
        result = 1e-2*torch.sigmoid(y)*(self.A-torch.sigmoid(y))*net_output
        return result

    def forward(self, s_grid, y0):
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

# 损失函数定义
def calculate_combined_loss(model, dat, ab_pid_theta, sigma=None):
    """计算组合损失"""
    num_steps = dat['t'].shape[0]
    if num_steps < 2:
        device = next(model.parameters()).device
        return torch.tensor(0.0, device=device)

    alpha = torch.exp(ab_pid_theta[0]) + 1e-4
    beta  = ab_pid_theta[1]
    s = alpha * dat['t'] + beta
    
    y0 = dat['y'][0]
    y_pred_trajectory = model(s, y0)
    y_actual_trajectory = dat['y']
    
    loss = (y_pred_trajectory - y_actual_trajectory)**2
    if sigma is not None:
        loss = loss / sigma
        
    return loss.mean()

def load_data_and_model():
    """加载数据和模型"""
    print("=== 加载数据和模型 ===")
    
    # 加载数据
    csf_dict = pc.load_csf_data()
    
    # 数据清理
    keys_to_delete = []
    for key in csf_dict:
        sample = csf_dict[key]
        sample = sample[~np.isnan(sample).any(axis=1)]
        csf_dict[key] = sample
        if sample.shape[0] < 2:
            keys_to_delete.append(key)
    for key in keys_to_delete:
        del csf_dict[key]
    
    # 转换为patient_data格式
    patient_data = {}
    for pid, sample in csf_dict.items():
        t = torch.from_numpy(sample[:, 0]).float()
        y = torch.from_numpy(sample[:, 1:5]).float()
        patient_data[pid] = {"t": t, "y": y, "y0": y[0].clone()}
    
    # 加载阶段字典
    stage_dict = pc.load_stage_dict()
    
    print(f"加载了 {len(patient_data)} 个患者的数据")
    
    # 加载模型
    model_path = "model_9652.pt"
    if not os.path.exists(model_path):
        print(f"错误: 模型文件 {model_path} 不存在")
        return None, None, None, None
    
    try:
        model = torch.jit.load(model_path)
        model.eval()
        print(f"成功加载模型: {model_path}")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return None, None, None, None
    
    # 加载alpha和beta参数
    ab_path = "ab_9652.pkl"
    ab_dict = None
    if os.path.exists(ab_path):
        try:
            with open(ab_path, 'rb') as f:
                ab_dict = pickle.load(f)
            print(f"成功加载参数文件: {ab_path} (包含{len(ab_dict)}个患者的参数)")
        except Exception as e:
            print(f"参数文件加载失败: {e}")
    else:
        print(f"警告: 参数文件 {ab_path} 不存在")
    
    return model, patient_data, stage_dict, ab_dict

def finetune_model(model, patient_data, ab_dict, stage_dict, num_iterations=100):
    """微调模型"""
    print(f"\n=== 开始微调训练 ({num_iterations} 轮) ===")
    print(f"使用完整数据集进行微调，包含 {len(patient_data)} 个患者")
    
    # 统计各阶段患者数量
    stage_counts = {}
    for pid in patient_data.keys():
        stage = stage_dict.get(pid, 'Other')
        stage_counts[stage] = stage_counts.get(stage, 0) + 1
    
    print("各阶段患者分布:")
    for stage, count in sorted(stage_counts.items()):
        print(f"  {stage}: {count} 个患者")
    
    # 计算全局方差
    all_biomarkers = torch.cat([dat['y'] for dat in patient_data.values()], dim=0)
    sigma = all_biomarkers.var(dim=0).clamp_min(1e-8)
    
    # 获取所有模型参数
    all_params = list(model.parameters())
    param_names = []
    for name, param in model.named_parameters():
        param_names.append(name)
    
    print(f"模型共有 {len(all_params)} 个参数组")
    for i, name in enumerate(param_names):
        print(f"  {i}: {name} - shape: {all_params[i].shape}")
    
    # 记录损失历史
    loss_history = []
    accepted_updates = 0
    
    # 计算初始损失
    with torch.no_grad():
        total_loss = 0.0
        num_patients = 0
        for pid, dat in patient_data.items():
            if pid in ab_dict and dat['t'].shape[0] >= 2:
                alpha, beta = ab_dict[pid]
                theta = torch.tensor([torch.log(torch.tensor(alpha) - 1e-4), beta])
                loss = calculate_combined_loss(model, dat, theta, sigma)
                total_loss += loss.item()
                num_patients += 1
        
        initial_loss = total_loss / num_patients if num_patients > 0 else float('inf')
        loss_history.append(initial_loss)
        print(f"初始损失: {initial_loss:.6f}")
    
    # 微调循环
    for iteration in range(num_iterations):
        # 随机选择5个参数
        selected_indices = torch.randperm(len(all_params))[:5]
        selected_params = [all_params[i] for i in selected_indices]
        selected_names = [param_names[i] for i in selected_indices]
        
        # 保存原始参数值
        original_params = [param.clone() for param in selected_params]
        
        # 计算当前损失
        model.eval()
        with torch.no_grad():
            current_total_loss = 0.0
            current_num_patients = 0
            for pid, dat in patient_data.items():
                if pid in ab_dict and dat['t'].shape[0] >= 2:
                    alpha, beta = ab_dict[pid]
                    theta = torch.tensor([torch.log(torch.tensor(alpha) - 1e-4), beta])
                    loss = calculate_combined_loss(model, dat, theta, sigma)
                    current_total_loss += loss.item()
                    current_num_patients += 1
            
            current_loss = current_total_loss / current_num_patients if current_num_patients > 0 else float('inf')
        
        # 计算梯度并更新选中的参数
        model.train()
        
        # 清零梯度
        for param in selected_params:
            if param.grad is not None:
                param.grad.zero_()
        
        # 计算损失和梯度
        total_loss = 0.0
        num_patients = 0
        for pid, dat in patient_data.items():
            if pid in ab_dict and dat['t'].shape[0] >= 2:
                alpha, beta = ab_dict[pid]
                theta = torch.tensor([torch.log(torch.tensor(alpha) - 1e-4), beta])
                loss = calculate_combined_loss(model, dat, theta, sigma)
                total_loss += loss
                num_patients += 1
        
        if num_patients > 0:
            avg_loss = total_loss / num_patients
            avg_loss.backward()
            
            # 使用较大学习率更新选中的参数
            learning_rate = 1  # 1e-1
            with torch.no_grad():
                for param in selected_params:
                    if param.grad is not None:
                        param.data -= learning_rate * param.grad
        
        # 计算更新后的损失
        model.eval()
        with torch.no_grad():
            new_total_loss = 0.0
            new_num_patients = 0
            for pid, dat in patient_data.items():
                if pid in ab_dict and dat['t'].shape[0] >= 2:
                    alpha, beta = ab_dict[pid]
                    theta = torch.tensor([torch.log(torch.tensor(alpha) - 1e-4), beta])
                    loss = calculate_combined_loss(model, dat, theta, sigma)
                    new_total_loss += loss.item()
                    new_num_patients += 1
            
            new_loss = new_total_loss / new_num_patients if new_num_patients > 0 else float('inf')
        
        # 判断是否接受更新
        if new_loss < current_loss:
            # 接受更新
            accepted_updates += 1
            loss_history.append(new_loss)
            status = "✓ 接受"
        else:
            # 拒绝更新，恢复原始参数
            with torch.no_grad():
                for param, original in zip(selected_params, original_params):
                    param.data.copy_(original)
            loss_history.append(current_loss)
            status = "✗ 拒绝"
        
        # 打印进度
        if (iteration + 1) % 10 == 0 or iteration < 10:
            print(f"轮次 {iteration+1:3d}: 当前损失={current_loss:.6f}, 新损失={new_loss:.6f}, {status}")
            print(f"         选中参数: {', '.join(selected_names)}")
    
    print(f"\n微调完成! 接受的更新: {accepted_updates}/{num_iterations}")
    print(f"最终损失: {loss_history[-1]:.6f}")
    print(f"损失变化: {loss_history[-1] - loss_history[0]:.6f}")
    
    return model, loss_history

def visualize_results(model, patient_data, ab_dict, stage_dict, loss_history):
    """可视化结果"""
    print("\n=== 生成可视化结果 ===")
    print(f"使用完整数据集进行可视化，包含 {len(patient_data)} 个患者")
    
    # 统计各阶段患者数量
    stage_counts = {}
    for pid in patient_data.keys():
        stage = stage_dict.get(pid, 'Other')
        stage_counts[stage] = stage_counts.get(stage, 0) + 1
    
    print("散点图将包含以下阶段的数据:")
    for stage, count in sorted(stage_counts.items()):
        print(f"  {stage}: {count} 个患者")
    
    # 1. 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, 'b-', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Fine-tuning Loss History (Complete Dataset)')
    plt.grid(True, alpha=0.3)
    plt.savefig('finetuning_loss.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. 计算s值范围
    all_s_values = []
    for pid, dat in patient_data.items():
        if pid in ab_dict:
            alpha, beta = ab_dict[pid]
            s_values = alpha * dat['t'] + beta
            all_s_values.append(s_values)
    
    if not all_s_values:
        print("没有可用的数据进行可视化")
        return
    
    all_s_values = torch.cat(all_s_values)
    s_min = torch.quantile(all_s_values, 0.05)
    s_max = torch.quantile(all_s_values, 0.95)
    s_curve = torch.linspace(s_min.item(), s_max.item(), 100)
    
    print(f"s值范围: [{s_min.item():.3f}, {s_max.item():.3f}]")
    
    # 3. 绘制散点图和轨迹
    TITLES = ['Aβ (A)', 'p-Tau (T)', 'Neurodeg. (N)', 'Cognition (C)']
    # 更新颜色方案，使用更鲜明的颜色区分不同阶段
    stage_colors = {
        'CN': '#FF8C00',      # 深橙色 - 认知正常
        'LMCI': '#32CD32',    # 酸橙绿 - 轻度认知障碍
        'AD': '#1E90FF',      # 道奇蓝 - 阿尔茨海默病
        'Other': '#808080'    # 灰色 - 其他
    }
    
    # 更新标签名称，更加清晰
    stage_labels = {
        'CN': 'CN (Cognitively Normal)',
        'LMCI': 'LMCI (Mild Cognitive Impairment)', 
        'AD': 'AD (Alzheimer\'s Disease)',
        'Other': 'Other Stages'
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 收集所有患者的初值用于生成轨迹
    all_y0 = torch.stack([dat['y0'] for dat in patient_data.values()])
    y0_mean = all_y0.mean(dim=0)
    y0_std = all_y0.std(dim=0)
    
    # 生成5条轨迹的初值
    torch.manual_seed(42)
    trajectory_y0s = []
    for i in range(5):
        if i == 0:
            # 第一条使用均值
            trajectory_y0s.append(y0_mean)
        else:
            # 其他4条从分布中采样
            y0_sample = torch.normal(y0_mean, y0_std * 0.3)
            y0_sample = torch.clamp(y0_sample, 0.0, 3.0)
            trajectory_y0s.append(y0_sample)
    
    # 使用模型预测轨迹
    trajectories = []
    with torch.no_grad():
        for y0 in trajectory_y0s:
            traj = model(s_curve, y0)
            trajectories.append(traj.numpy())
    
    # 按阶段组织数据，确保颜色一致性
    stage_data = {}
    for pid, dat in patient_data.items():
        if pid in ab_dict:
            stage = stage_dict.get(pid, 'Other')
            if stage not in stage_data:
                stage_data[stage] = {'s_values': [], 'y_values': [[] for _ in range(4)]}
            
            alpha, beta = ab_dict[pid]
            s_patient = alpha * dat['t'] + beta
            y_patient = dat['y']
            
            stage_data[stage]['s_values'].extend(s_patient.numpy())
            for k in range(4):
                stage_data[stage]['y_values'][k].extend(y_patient[:, k].numpy())
    
    # 绘制每个生物标记物
    for k, ax in enumerate(axes.flat):
        # 按阶段绘制散点数据
        for stage in ['CN', 'LMCI', 'AD', 'Other']:  # 固定顺序确保一致性
            if stage in stage_data:
                s_vals = stage_data[stage]['s_values']
                y_vals = stage_data[stage]['y_values'][k]
                
                if s_vals and y_vals:  # 确保有数据
                    ax.scatter(s_vals, y_vals, 
                             s=20, alpha=0.7, 
                             c=stage_colors[stage], 
                             label=f"{stage_labels[stage]} (n={len(s_vals)})",
                             edgecolors='white', linewidth=0.5)
        
        # 绘制5条轨迹，使用不同的颜色和样式
        traj_colors = ['#FF0000', '#0000FF', '#00AA00', '#800080', '#FF8000']  # 红、蓝、绿、紫、橙
        traj_styles = ['-', '--', '-.', ':', '-']
        for i, (traj, color, style) in enumerate(zip(trajectories, traj_colors, traj_styles)):
            label = f'Predicted Trajectory {i+1}' if i < 3 else ""
            ax.plot(s_curve.numpy(), traj[:, k], 
                   color=color, linewidth=2.5, alpha=0.9, 
                   linestyle=style, label=label)
        
        ax.set_xlabel('Disease Progression Score (s)', fontsize=11)
        ax.set_ylabel(f'{TITLES[k]} Level', fontsize=11)
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'{TITLES[k]} - Fine-tuned Model (Complete Dataset)', fontsize=12, fontweight='bold')
    
    fig.suptitle('Fine-tuned Model Results: Complete Dataset Scatter Plot + 5 Trajectories', fontsize=14)
    plt.tight_layout()
    
    # 保存图片
    plt.savefig('finetuning_results.png', dpi=300, bbox_inches='tight')
    print("结果图已保存: finetuning_results.png")
    plt.show()
    
    # 打印轨迹初值信息
    print("\n生成的5条轨迹初值:")
    for i, y0 in enumerate(trajectory_y0s):
        print(f"轨迹 {i+1}: {y0.numpy()}")

def main():
    """主函数"""
    print("=== 模型微调程序 ===")
    
    # 加载数据和模型
    model, patient_data, stage_dict, ab_dict = load_data_and_model()
    if model is None:
        return
    
    if ab_dict is None:
        print("错误: 无法加载alpha和beta参数，无法进行微调")
        return
    
    # 微调模型
    finetuned_model, loss_history = finetune_model(model, patient_data, ab_dict, stage_dict, num_iterations=100)
    
    # 保存微调后的模型
    finetuned_model.save('model_9652_finetuned.pt')
    print("微调后的模型已保存: model_9652_finetuned.pt")
    
    # 可视化结果
    visualize_results(finetuned_model, patient_data, ab_dict, stage_dict, loss_history)
    
    print("\n=== 微调程序完成 ===")

if __name__ == "__main__":
    main()
