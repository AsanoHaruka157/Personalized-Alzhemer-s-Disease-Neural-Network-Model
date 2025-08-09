import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import pccmnn as pc
import os
import glob
from pathlib import Path

# 复制必要的类定义
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

def load_and_prepare_data():
    """加载并准备数据"""
    print("加载数据...")
    
    # 加载CSF数据
    csf_dict = pc.load_csf_data()
    
    # 数据清理
    keys_to_delete = []
    for key in csf_dict:
        sample = csf_dict[key]
        sample = sample[~np.isnan(sample).any(axis=1)]   # 删缺失
        csf_dict[key] = sample
        if sample.shape[0] < 2:
            keys_to_delete.append(key)
    for key in keys_to_delete:
        del csf_dict[key]
    
    # 转换为patient_data格式
    patient_data = {}
    for pid, sample in csf_dict.items():
        t = torch.from_numpy(sample[:, 0]).float()            # 年龄
        y = torch.from_numpy(sample[:, 1:5]).float()          # biomarker A/T/N/C
        patient_data[pid] = {"t": t, "y": y, "y0": y[0].clone()}
    
    # 加载阶段字典
    stage_dict = pc.load_stage_dict()
    
    print(f"加载了 {len(patient_data)} 个患者的数据")
    return patient_data, stage_dict

def select_model():
    """选择要评估的模型"""
    print("\n=== 选择模型 ===")
    
    # 查找所有模型文件
    model_files = glob.glob("model_*.pt")
    
    if not model_files:
        print("未找到任何模型文件 (model_*.pt)")
        return None, None
    
    print("可用的模型文件:")
    for i, model_file in enumerate(model_files):
        # 检查是否有对应的参数文件
        pid = model_file.replace("model_", "").replace(".pt", "")
        param_file = f"ab_{pid}.pkl"
        param_exists = os.path.exists(param_file)
        status = " (有参数文件)" if param_exists else " (无参数文件)"
        print(f"{i+1}. {model_file}{status}")
    
    while True:
        try:
            choice = input(f"请选择模型 (1-{len(model_files)}) 或输入文件名: ").strip()
            
            # 尝试作为数字解析
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(model_files):
                    selected_file = model_files[idx]
                    break
            except ValueError:
                pass
            
            # 尝试作为文件名
            if choice in model_files:
                selected_file = choice
                break
            elif os.path.exists(choice):
                selected_file = choice
                break
            else:
                print("无效选择，请重试")
                continue
                
        except KeyboardInterrupt:
            return None, None
    
    print(f"选择的模型: {selected_file}")
    
    # 加载模型
    try:
        model = torch.jit.load(selected_file)
        model.eval()
        print("模型加载成功")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return None, None
    
    # 尝试加载对应的参数文件
    pid = selected_file.replace("model_", "").replace(".pt", "")
    param_file = f"ab_{pid}.pkl"
    
    saved_params = None
    if os.path.exists(param_file):
        try:
            import pickle
            with open(param_file, 'rb') as f:
                saved_params = pickle.load(f)
            print(f"成功加载参数文件: {param_file} (包含{len(saved_params)}个患者的参数)")
        except Exception as e:
            print(f"参数文件加载失败: {e}")
            print("将使用估计的参数")
    else:
        print("未找到对应的参数文件，将使用估计的参数")
    
    return model, saved_params

def select_dataset(patient_data, stage_dict):
    """选择数据集"""
    print("\n=== 选择数据集 ===")
    print("1. 完整数据集")
    print("2. CN阶段")
    print("3. LMCI阶段")
    print("4. AD阶段")
    
    while True:
        try:
            choice = input("请选择数据集 (1-4): ").strip()
            
            if choice == "1":
                selected_data = patient_data
                dataset_name = "完整数据集"
                break
            elif choice == "2":
                selected_data = {pid: dat for pid, dat in patient_data.items() 
                               if stage_dict.get(pid) == 'CN'}
                dataset_name = "CN阶段"
                break
            elif choice == "3":
                selected_data = {pid: dat for pid, dat in patient_data.items() 
                               if stage_dict.get(pid) == 'LMCI'}
                dataset_name = "LMCI阶段"
                break
            elif choice == "4":
                selected_data = {pid: dat for pid, dat in patient_data.items() 
                               if stage_dict.get(pid) == 'AD'}
                dataset_name = "AD阶段"
                break
            else:
                print("无效选择，请重试")
                continue
                
        except KeyboardInterrupt:
            return None, None
    
    print(f"选择的数据集: {dataset_name} ({len(selected_data)} 个患者)")
    return selected_data, dataset_name

def select_evaluation_mode():
    """选择评估模式"""
    print("\n=== 选择评估模式 ===")
    print("1. 从初值出发画轨迹 (Global Trajectory)")
    print("2. 在不同窗口画轨迹 (Local Mean Stepwise)")
    
    while True:
        try:
            choice = input("请选择评估模式 (1-2): ").strip()
            
            if choice == "1":
                return "global", "从初值出发画轨迹"
            elif choice == "2":
                return "stepwise", "在不同窗口画轨迹"
            else:
                print("无效选择，请重试")
                continue
                
        except KeyboardInterrupt:
            return None, None

def estimate_alpha_beta(selected_data, stage_dict, saved_params=None):
    """估计alpha和beta参数"""
    ab_dict = {}
    
    if saved_params is not None:
        print("使用保存的alpha和beta参数...")
        # 使用保存的参数，但只包含选中数据集中的患者
        for pid in selected_data.keys():
            if pid in saved_params:
                ab_dict[pid] = saved_params[pid]
            else:
                print(f"警告: 患者 {pid} 的参数未找到，将使用估计值")
                # 对缺失的患者使用估计值
                stage = stage_dict.get(pid, 'Other')
                t_mean = selected_data[pid]['t'].mean()
                if stage == 'CN':
                    s_range = (-10.0, 0.0)
                elif stage == 'LMCI':
                    s_range = (0.0, 10.0)
                elif stage == 'AD':
                    s_range = (10.0, 20.0)
                else:
                    s_range = (-5.0, 5.0)
                alpha = 1.0
                s_target = (s_range[0] + s_range[1]) / 2
                beta = s_target - alpha * t_mean
                ab_dict[pid] = (alpha, beta)
        
        print(f"使用了 {len([pid for pid in selected_data.keys() if pid in saved_params])} 个保存的参数")
        print(f"估计了 {len([pid for pid in selected_data.keys() if pid not in saved_params])} 个缺失的参数")
    else:
        print("估计alpha和beta参数...")
        for pid, dat in selected_data.items():
            stage = stage_dict.get(pid, 'Other')
            t_mean = dat['t'].mean()

            # 根据阶段设置s范围
            if stage == 'CN':
                s_range = (-10.0, 0.0)
            elif stage == 'LMCI':
                s_range = (0.0, 10.0)
            elif stage == 'AD':
                s_range = (10.0, 20.0)
            else:  # Default for 'Other' or missing stages
                s_range = (-5.0, 5.0)
            
            # 简单的初始化策略
            alpha = 1.0  # 固定alpha为1
            s_target = (s_range[0] + s_range[1]) / 2  # s范围中点
            beta = s_target - alpha * t_mean
            
            ab_dict[pid] = (alpha, beta)
    
    return ab_dict

def evaluate_global_trajectory(model, selected_data, ab_dict, stage_dict, dataset_name):
    """从初值出发画轨迹的评估模式"""
    print("\n执行全局轨迹评估...")
    
    # 计算所有s值的范围
    all_s_values = []
    for pid, dat in selected_data.items():
        alpha, beta = ab_dict[pid]
        s_values = alpha * dat['t'] + beta
        all_s_values.append(s_values)
    
    if not all_s_values:
        print("没有可用的数据进行评估")
        return
    
    all_s_values = torch.cat(all_s_values)
    s_min = torch.quantile(all_s_values, 0.05)
    s_max = torch.quantile(all_s_values, 0.95)
    
    print(f"s值范围: [{s_min.item():.3f}, {s_max.item():.3f}]")
    
    # 生成s网格
    s_curve = torch.linspace(s_min.item(), s_max.item(), 50)
    
    # 计算平均初值
    y0_mean = torch.stack([dat['y0'] for dat in selected_data.values()]).mean(0)
    
    # 使用模型预测轨迹
    with torch.no_grad():
        y_curve = model(s_curve, y0_mean).numpy()
    
    # 绘图
    TITLES = ['Aβ (A)', 'p-Tau (T)', 'Neurodeg. (N)', 'Cognition (C)']
    colors = {'CN': 'orange', 'LMCI': 'green', 'AD': 'blue', 'Other': 'grey'}
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    for k, ax in enumerate(axes.flat):
        # 绘制数据点
        for pid, dat in selected_data.items():
            alpha, beta = ab_dict[pid]
            s_patient = alpha * dat['t'] + beta
            y_patient = dat['y'][:, k]
            stage = stage_dict.get(pid, 'Other')
            color = colors.get(stage, 'grey')
            
            ax.scatter(s_patient.numpy(), y_patient.numpy(), 
                      s=15, alpha=0.6, c=color, label=stage if pid == list(selected_data.keys())[0] else "")
        
        # 绘制预测轨迹
        ax.plot(s_curve.numpy(), y_curve[:, k], 'r-', linewidth=2, label='Predicted Trajectory')
        
        ax.set_xlabel('Disease progression score s')
        ax.set_ylabel(TITLES[k])
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(f'Global Trajectory Evaluation - {dataset_name}')
    plt.tight_layout()
    
    # 保存图片
    output_file = f'eval_global_{dataset_name.replace(" ", "_")}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"结果保存到: {output_file}")
    plt.show()

def evaluate_stepwise_trajectory(model, selected_data, ab_dict, stage_dict, dataset_name):
    """在不同窗口画轨迹的评估模式"""
    print("\n执行逐步轨迹评估...")
    
    # 计算所有s值的范围
    all_s_values = []
    all_s_points = []
    all_y_points = []
    
    for pid, dat in selected_data.items():
        alpha, beta = ab_dict[pid]
        s_values = alpha * dat['t'] + beta
        all_s_values.append(s_values)
        all_s_points.append(s_values)
        all_y_points.append(dat['y'])
    
    if not all_s_values:
        print("没有可用的数据进行评估")
        return
    
    all_s_values = torch.cat(all_s_values)
    all_s_points = torch.cat(all_s_points)
    all_y_points = torch.cat(all_y_points, dim=0)
    
    s_min = torch.quantile(all_s_values, 0.05)
    s_max = torch.quantile(all_s_values, 0.95)
    
    print(f"s值范围: [{s_min.item():.3f}, {s_max.item():.3f}]")
    
    # 生成s网格
    s_curve = torch.linspace(s_min.item(), s_max.item(), 20)
    window_size = 0.1 * (s_max - s_min)  # 窗口大小为s范围的10%
    
    print(f"窗口大小: {window_size:.3f}")
    
    # 逐步预测轨迹
    y_curve_stepwise = []
    
    # 初始点：使用s_min附近的数据点均值
    mask_init = torch.abs(all_s_points - s_min) <= window_size
    if mask_init.sum() > 0:
        y_current = all_y_points[mask_init].mean(dim=0)
    else:
        y_current = torch.stack([dat['y0'] for dat in selected_data.values()]).mean(0)
    
    y_curve_stepwise.append(y_current.clone())
    
    # 逐步预测
    with torch.no_grad():
        for i in range(1, len(s_curve)):
            s_current = s_curve[i-1]
            s_next = s_curve[i]
            
            # 在当前s点附近寻找数据点并计算均值
            mask = torch.abs(all_s_points - s_current) <= window_size
            if mask.sum() > 0:
                y_local_mean = all_y_points[mask].mean(dim=0)
            else:
                y_local_mean = y_current  # 如果没有附近的数据点，使用当前值
            
            # 使用模型从当前点预测到下一点
            s_step = torch.tensor([s_current.item(), s_next.item()])
            y_pred_step = model(s_step, y_local_mean)
            y_current = y_pred_step[1]  # 取预测的下一步
            
            y_curve_stepwise.append(y_current.clone())
    
    y_curve_stepwise = torch.stack(y_curve_stepwise).numpy()
    
    # 绘图
    TITLES = ['Aβ (A)', 'p-Tau (T)', 'Neurodeg. (N)', 'Cognition (C)']
    colors = {'CN': 'orange', 'LMCI': 'green', 'AD': 'blue', 'Other': 'grey'}
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    for k, ax in enumerate(axes.flat):
        # 绘制数据点
        for pid, dat in selected_data.items():
            alpha, beta = ab_dict[pid]
            s_patient = alpha * dat['t'] + beta
            y_patient = dat['y'][:, k]
            stage = stage_dict.get(pid, 'Other')
            color = colors.get(stage, 'grey')
            
            ax.scatter(s_patient.numpy(), y_patient.numpy(), 
                      s=15, alpha=0.6, c=color, label=stage if pid == list(selected_data.keys())[0] else "")
        
        # 绘制逐步预测轨迹
        ax.plot(s_curve.numpy(), y_curve_stepwise[:, k], 'r-', linewidth=2, label='Stepwise Trajectory')
        
        ax.set_xlabel('Disease progression score s')
        ax.set_ylabel(TITLES[k])
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(f'Stepwise Trajectory Evaluation - {dataset_name}')
    plt.tight_layout()
    
    # 保存图片
    output_file = f'eval_stepwise_{dataset_name.replace(" ", "_")}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"结果保存到: {output_file}")
    plt.show()

def main():
    """主函数"""
    print("=== 模型评估程序 ===")
    
    try:
        # 1. 加载数据
        patient_data, stage_dict = load_and_prepare_data()
        
        # 2. 选择模型
        model, saved_params = select_model()
        if model is None:
            print("未选择模型，退出程序")
            return
        
        # 3. 选择数据集
        selected_data, dataset_name = select_dataset(patient_data, stage_dict)
        if selected_data is None:
            print("未选择数据集，退出程序")
            return
        
        if len(selected_data) == 0:
            print("选择的数据集为空，退出程序")
            return
        
        # 4. 选择评估模式
        eval_mode, mode_name = select_evaluation_mode()
        if eval_mode is None:
            print("未选择评估模式，退出程序")
            return
        
        print(f"\n开始评估: {mode_name}")
        
        # 5. 估计alpha和beta参数
        ab_dict = estimate_alpha_beta(selected_data, stage_dict, saved_params)
        
        # 6. 执行评估
        if eval_mode == "global":
            evaluate_global_trajectory(model, selected_data, ab_dict, stage_dict, dataset_name)
        elif eval_mode == "stepwise":
            evaluate_stepwise_trajectory(model, selected_data, ab_dict, stage_dict, dataset_name)
        
        print("\n评估完成！")
        
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
    except Exception as e:
        print(f"\n程序执行出错: {e}")
        import traceback
        traceback.print_exc()

# ============ 可被外部调用的接口函数 ============

def evaluate_model_with_data(model, patient_data, ab_dict, stage_dict, 
                           dataset_filter=None, mode="global", 
                           output_prefix="eval", show_plot=True):
    """
    评估模型的接口函数，可被外部调用
    
    参数:
    - model: 训练好的模型
    - patient_data: 患者数据字典
    - ab_dict: alpha和beta参数字典
    - stage_dict: 阶段字典
    - dataset_filter: 数据集过滤器 ('CN', 'LMCI', 'AD', None表示全部)
    - mode: 评估模式 ('global' 或 'stepwise')
    - output_prefix: 输出文件前缀
    - show_plot: 是否显示图片
    
    返回:
    - 保存的图片文件名
    """
    
    # 根据过滤器选择数据
    if dataset_filter is None:
        selected_data = patient_data
        dataset_name = "完整数据集"
    else:
        selected_data = {pid: dat for pid, dat in patient_data.items() 
                        if stage_dict.get(pid) == dataset_filter}
        dataset_name = f"{dataset_filter}阶段"
    
    if len(selected_data) == 0:
        print(f"警告: {dataset_name} 没有数据")
        return None
    
    print(f"评估数据集: {dataset_name} ({len(selected_data)} 个患者)")
    
    # 过滤ab_dict，只保留选中数据集中的患者
    filtered_ab_dict = {pid: ab_dict[pid] for pid in selected_data.keys() if pid in ab_dict}
    
    if len(filtered_ab_dict) == 0:
        print("警告: 没有可用的alpha/beta参数")
        return None
    
    # 执行评估
    if mode == "global":
        return _evaluate_global_trajectory_internal(model, selected_data, filtered_ab_dict, 
                                                  stage_dict, dataset_name, output_prefix, show_plot)
    elif mode == "stepwise":
        return _evaluate_stepwise_trajectory_internal(model, selected_data, filtered_ab_dict, 
                                                    stage_dict, dataset_name, output_prefix, show_plot)
    else:
        raise ValueError(f"未知的评估模式: {mode}")

def _evaluate_global_trajectory_internal(model, selected_data, ab_dict, stage_dict, 
                                       dataset_name, output_prefix, show_plot):
    """内部全局轨迹评估函数"""
    print(f"执行全局轨迹评估...")
    
    # 计算所有s值的范围
    all_s_values = []
    for pid, dat in selected_data.items():
        if pid in ab_dict:
            alpha, beta = ab_dict[pid]
            s_values = alpha * dat['t'] + beta
            all_s_values.append(s_values)
    
    if not all_s_values:
        print("没有可用的数据进行评估")
        return None
    
    all_s_values = torch.cat(all_s_values)
    s_min = torch.quantile(all_s_values, 0.05)
    s_max = torch.quantile(all_s_values, 0.95)
    
    print(f"s值范围: [{s_min.item():.3f}, {s_max.item():.3f}]")
    
    # 生成s网格
    s_curve = torch.linspace(s_min.item(), s_max.item(), 50)
    
    # 计算平均初值
    y0_mean = torch.stack([dat['y0'] for dat in selected_data.values()]).mean(0)
    
    # 使用模型预测轨迹
    with torch.no_grad():
        y_curve = model(s_curve, y0_mean).numpy()
    
    # 绘图
    TITLES = ['Aβ (A)', 'p-Tau (T)', 'Neurodeg. (N)', 'Cognition (C)']
    colors = {'CN': 'orange', 'LMCI': 'green', 'AD': 'blue', 'Other': 'grey'}
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    for k, ax in enumerate(axes.flat):
        # 绘制数据点
        plotted_stages = set()
        for pid, dat in selected_data.items():
            if pid in ab_dict:
                alpha, beta = ab_dict[pid]
                s_patient = alpha * dat['t'] + beta
                y_patient = dat['y'][:, k]
                stage = stage_dict.get(pid, 'Other')
                color = colors.get(stage, 'grey')
                
                # 只为每个阶段添加一次标签
                label = stage if stage not in plotted_stages else ""
                if label:
                    plotted_stages.add(stage)
                
                ax.scatter(s_patient.numpy(), y_patient.numpy(), 
                          s=15, alpha=0.6, c=color, label=label)
        
        # 绘制预测轨迹
        ax.plot(s_curve.numpy(), y_curve[:, k], 'r-', linewidth=2, label='Predicted Trajectory')
        
        ax.set_xlabel('Disease progression score s')
        ax.set_ylabel(TITLES[k])
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(f'Global Trajectory Evaluation - {dataset_name}')
    plt.tight_layout()
    
    # 保存图片
    output_file = f'{output_prefix}_global_{dataset_name.replace(" ", "_").replace("阶段", "")}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"结果保存到: {output_file}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return output_file

def _evaluate_stepwise_trajectory_internal(model, selected_data, ab_dict, stage_dict, 
                                         dataset_name, output_prefix, show_plot):
    """内部逐步轨迹评估函数"""
    print(f"执行逐步轨迹评估...")
    
    # 计算所有s值的范围
    all_s_values = []
    all_s_points = []
    all_y_points = []
    
    for pid, dat in selected_data.items():
        if pid in ab_dict:
            alpha, beta = ab_dict[pid]
            s_values = alpha * dat['t'] + beta
            all_s_values.append(s_values)
            all_s_points.append(s_values)
            all_y_points.append(dat['y'])
    
    if not all_s_values:
        print("没有可用的数据进行评估")
        return None
    
    all_s_values = torch.cat(all_s_values)
    all_s_points = torch.cat(all_s_points)
    all_y_points = torch.cat(all_y_points, dim=0)
    
    s_min = torch.quantile(all_s_values, 0.05)
    s_max = torch.quantile(all_s_values, 0.95)
    
    print(f"s值范围: [{s_min.item():.3f}, {s_max.item():.3f}]")
    
    # 生成s网格
    s_curve = torch.linspace(s_min.item(), s_max.item(), 20)
    window_size = 0.1 * (s_max - s_min)  # 窗口大小为s范围的10%
    
    print(f"窗口大小: {window_size:.3f}")
    
    # 逐步预测轨迹
    y_curve_stepwise = []
    
    # 初始点：使用s_min附近的数据点均值
    mask_init = torch.abs(all_s_points - s_min) <= window_size
    if mask_init.sum() > 0:
        y_current = all_y_points[mask_init].mean(dim=0)
    else:
        y_current = torch.stack([dat['y0'] for dat in selected_data.values()]).mean(0)
    
    y_curve_stepwise.append(y_current.clone())
    
    # 逐步预测
    with torch.no_grad():
        for i in range(1, len(s_curve)):
            s_current = s_curve[i-1]
            s_next = s_curve[i]
            
            # 在当前s点附近寻找数据点并计算均值
            mask = torch.abs(all_s_points - s_current) <= window_size
            if mask.sum() > 0:
                y_local_mean = all_y_points[mask].mean(dim=0)
            else:
                y_local_mean = y_current  # 如果没有附近的数据点，使用当前值
            
            # 使用模型从当前点预测到下一点
            s_step = torch.tensor([s_current.item(), s_next.item()])
            y_pred_step = model(s_step, y_local_mean)
            y_current = y_pred_step[1]  # 取预测的下一步
            
            y_curve_stepwise.append(y_current.clone())
    
    y_curve_stepwise = torch.stack(y_curve_stepwise).numpy()
    
    # 绘图
    TITLES = ['Aβ (A)', 'p-Tau (T)', 'Neurodeg. (N)', 'Cognition (C)']
    colors = {'CN': 'orange', 'LMCI': 'green', 'AD': 'blue', 'Other': 'grey'}
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    for k, ax in enumerate(axes.flat):
        # 绘制数据点
        plotted_stages = set()
        for pid, dat in selected_data.items():
            if pid in ab_dict:
                alpha, beta = ab_dict[pid]
                s_patient = alpha * dat['t'] + beta
                y_patient = dat['y'][:, k]
                stage = stage_dict.get(pid, 'Other')
                color = colors.get(stage, 'grey')
                
                # 只为每个阶段添加一次标签
                label = stage if stage not in plotted_stages else ""
                if label:
                    plotted_stages.add(stage)
                
                ax.scatter(s_patient.numpy(), y_patient.numpy(), 
                          s=15, alpha=0.6, c=color, label=label)
        
        # 绘制逐步预测轨迹
        ax.plot(s_curve.numpy(), y_curve_stepwise[:, k], 'r-', linewidth=2, label='Stepwise Trajectory')
        
        ax.set_xlabel('Disease progression score s')
        ax.set_ylabel(TITLES[k])
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(f'Stepwise Trajectory Evaluation - {dataset_name}')
    plt.tight_layout()
    
    # 保存图片
    output_file = f'{output_prefix}_stepwise_{dataset_name.replace(" ", "_").replace("阶段", "")}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"结果保存到: {output_file}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return output_file

if __name__ == "__main__":
    main()
