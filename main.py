import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pccmnn as pc
from tqdm import tqdm
from pretrain import ODENet, ODEModel  # 从pretrain.py导入模型定义
from torchdiffeq import odeint as torch_odeint

# --- 0. 資料載入和準備 ---
csf_dict = pc.load_data()
stage_dict = pc.load_stage_dict()
print(f"成功載入 {len(csf_dict)} 位患者的資料。")

# 转换数据格式以适应 PyTorch
patient_data = {}
for pid, sample in csf_dict.items():
    t = torch.from_numpy(sample[:, 0]).float()
    y = torch.from_numpy(sample[:, 1:5]).float()
    patient_data[pid] = {"t": t, "y": y, "y0": y[0].clone(), "stage": stage_dict.get(pid, 'Other')}

def get_cn_average_y0(patient_data, stage_dict):
    cn_y0s = []
    for pid, data in patient_data.items():
        if stage_dict.get(pid) == 'CN':
            cn_y0s.append(data['y0'])
    if not cn_y0s:
        print("警告: 未找到CN患者, 使用預設y0。")
        return torch.tensor([0.1, 0, 0, 0])
    
    # 将所有CN患者的y0堆叠成矩阵
    cn_y0s_tensor = torch.stack(cn_y0s)  # shape: (num_cn_patients, 4)
    
    # 对每个生物标志物分别计算非NaN值的平均
    avg_y0 = torch.zeros(4)
    for k in range(4):
        y0_k = cn_y0s_tensor[:, k]
        valid_mask = ~torch.isnan(y0_k)
        if valid_mask.sum() > 0:
            avg_y0[k] = y0_k[valid_mask].mean()
        else:
            # 如果所有CN患者在该标志物上都是NaN，使用0
            avg_y0[k] = 0.0
            print(f"警告: 所有CN患者在生物标志物{k}上的初始值都是NaN，使用0。")
    
    print(f"使用CN群體的平均初始值（非NaN）: {avg_y0.numpy()}")
    return avg_y0

y0_cn_avg = get_cn_average_y0(patient_data, stage_dict)
name = 'fnn'

# --- 1. 定義訓練流程 ---
# ===== 优化后的 FNN 损失 =====
def calculate_loss_fnn(ode_model, patient_data, ab, pids, y0):
    """
    高效版 FNN loss：
    - 所有病人、所有 biomarker 的 s 值一次合并；
    - 去重、排序，确保 torchdiffeq 时间轴严格递增；
    - 在去重时间轴上解一次 ODE，再映射回原索引。
    - 添加正则化项：在[-5:20:1]的向量上约束ODE右侧函数
    """
    try:
        s_all_list, y_true_list, k_list = [], [], []
        for k in range(4):
            for pid in pids:
                dat = patient_data[pid]
                s_values = ab[pid]['a'] * dat['t'] + ab[pid]['b']
                y_values = dat['y'][:, k]
                valid_mask = ~torch.isnan(y_values)
                if valid_mask.any():
                    s_all_list.append(s_values[valid_mask])
                    y_true_list.append(y_values[valid_mask])
                    k_list.extend([k] * valid_mask.sum().item())

        if not s_all_list:
            return torch.tensor(0.0, requires_grad=True)

        s_all = torch.cat(s_all_list)
        y_true_all = torch.cat(y_true_list)
        k_all = torch.tensor(k_list, device=y_true_all.device, dtype=torch.long)

        # 排序和去重操作不进计算图
        with torch.no_grad():
            s_sorted, sort_idx = torch.sort(s_all)
            y_sorted = y_true_all[sort_idx]
            k_sorted = k_all[sort_idx]
            s_unique, inv = torch.unique_consecutive(s_sorted, return_inverse=True)

        # 一次 ODE 求解
        y_unique = torch_odeint(
            ode_model, y0, s_unique, method='dopri5', rtol=1e-4, atol=1e-5
        )  # (Nu, 4)

        # 映射回原索引
        y_all = y_unique[inv]  # (N, 4)
        y_pred_selected = y_all[torch.arange(y_all.size(0)), k_sorted]

        # 数据拟合损失 - 使用SmoothL1Loss
        smooth_l1_loss = nn.SmoothL1Loss(reduction='sum')
        data_loss = smooth_l1_loss(y_pred_selected, y_sorted)
        
        # === 正则化项 ===
        # FNN模型参数的L1正则化
        l1_reg = 0.0
        for param in ode_model.parameters():
            l1_reg += torch.sum(torch.abs(param))
        
        # 总损失：数据损失 + L1正则化
        lambda_l1 = 0.0001  # L1正则化权重
        
        total_loss = data_loss + lambda_l1 * l1_reg
        
        # 返回总损失和各分量
        loss_dict = {
            'total': total_loss.item() if torch.isfinite(total_loss) else float('inf'),
            'data': data_loss.item(),
            'l1': l1_reg.item(),
            'lambda': lambda_l1
        }
        
        if torch.isfinite(total_loss):
            return total_loss, loss_dict
        else:
            return torch.tensor(float('inf'), requires_grad=True), loss_dict

    except Exception as e:
        print(f"FNN loss 计算出错: {e}")
        import traceback
        traceback.print_exc()
        loss_dict = {'total': float('inf'), 'data': 0, 'l1': 0, 'lambda': 0.0001}
        return torch.tensor(float('inf'), requires_grad=True), loss_dict


# ===== 优化后的 DPS 损失 =====
def calculate_loss_dps(ode_model, patient_data, ab, pids, y0):
    """
    高效版 DPS loss：
    - 所有病人、所有时间点与 biomarker 合并；
    - 去重、排序，确保 torchdiffeq 时间轴严格递增；
    - 在去重时间轴上解一次 ODE，再映射回原索引。
    - 添加a,b参数的L2正则化
    """
    try:
        s_all_list, y_true_list, k_list = [], [], []
        for pid in pids:
            dat = patient_data[pid]
            s_values = ab[pid]['a'] * dat['t'] + ab[pid]['b']
            y_true = dat['y']
            for i in range(y_true.shape[0]):
                for k in range(4):
                    if not torch.isnan(y_true[i, k]):
                        s_all_list.append(s_values[i])
                        y_true_list.append(y_true[i, k])
                        k_list.append(k)

        if not s_all_list:
            return torch.tensor(0.0, requires_grad=True), {'total': 0.0, 'data': 0.0, 'l2': 0.0}

        s_all = torch.stack(s_all_list)
        y_true_all = torch.stack(y_true_list)
        k_all = torch.tensor(k_list, device=y_true_all.device, dtype=torch.long)

        with torch.no_grad():
            s_sorted, sort_idx = torch.sort(s_all)
            y_sorted = y_true_all[sort_idx]
            k_sorted = k_all[sort_idx]
            s_unique, inv = torch.unique_consecutive(s_sorted, return_inverse=True)

        y_unique = torch_odeint(
            ode_model, y0, s_unique, method='dopri5', rtol=1e-4, atol=1e-5
        )

        y_all = y_unique[inv]
        y_pred_selected = y_all[torch.arange(y_all.size(0)), k_sorted]

        # 数据拟合损失
        data_loss = ((y_pred_selected - y_sorted) ** 2).sum()
        
        # a,b参数的L2正则化
        l2_reg = 0.0
        for pid in pids:
            l2_reg += ab[pid]['a'] ** 2 + ab[pid]['b'] ** 2
        
        # 总损失
        lambda_l2 = 0.01  # L2正则化权重
        total_loss = data_loss + lambda_l2 * l2_reg
        
        # 返回损失和分量
        loss_dict = {
            'total': total_loss.item() if torch.isfinite(total_loss) else float('inf'),
            'data': data_loss.item(),
            'l2': l2_reg.item()
        }
        
        if torch.isfinite(total_loss):
            return total_loss, loss_dict
        else:
            return torch.tensor(float('inf'), requires_grad=True), loss_dict

    except Exception as e:
        print(f"DPS loss 计算出错: {e}")
        loss_dict = {'total': float('inf'), 'data': 0.0, 'l2': 0.0}
        return torch.tensor(float('inf'), requires_grad=True), loss_dict

def train_alternating(
    fnn_pretrained,
    patient_data,
    y0,
    dps_path='dps_pretrain.pth',
    n_outer=10,          # 外循环次数
    n_fnn=10,            # 每次外循环中FNN训练次数
    n_dps=5,             # 每次外循环中DPS训练次数
    lr_fnn=1e-3,         # FNN的Adam学习率
    lr_dps=1e-3,         # DPS的Adam学习率
):
    """
    交替优化FNN和DPS参数（两重循环）：
    外循环n_outer次，每次：
    1. 先用Adam优化FNN n_fnn次
    2. 再用Adam优化a,b参数 n_dps次
    """
    print(f"\n--- 开始交替优化训练 (Adam + Adam) ---")
    print(f"外循环: {n_outer}次, 每次训练FNN {n_fnn}次, 训练DPS {n_dps}次")
    
    # 加载预训练的FNN模型
    ode_model = ODEModel(fnn_pretrained).train()
    
    # 加载DPS参数
    try:
        dps_params_loaded = torch.load(dps_path, weights_only=False)
        ab = {}
        for pid, data in patient_data.items():
            if pid in dps_params_loaded:
                ab[pid] = {
                    'a': nn.Parameter(torch.tensor(dps_params_loaded[pid]['a'], dtype=torch.float32)),
                    'b': nn.Parameter(torch.tensor(dps_params_loaded[pid]['b'], dtype=torch.float32))
                }
        print(f"成功从 {dps_path} 加载DPS参数。")
    except FileNotFoundError:
        print(f"错误: 未找到 {dps_path}。")
        return None, None, None
        
    patient_pids = list(ab.keys())
    dps_params = [p for pid in patient_pids for p in ab[pid].values()]
    
    # 创建两个优化器：Adam用于FNN，Adam用于DPS
    opt_fnn = optim.Adam(ode_model.parameters(), lr=lr_fnn)
    opt_dps = optim.Adam(dps_params, lr=lr_dps)
    
    # 记录loss历史
    loss_history = {
        'outer_epoch': [],
        'inner_step': [],
        'step_type': [],  # 'FNN' or 'DPS'
        'fnn_loss': [],
        'dps_loss': []
    }
    
    # 计算总步数
    total_steps = n_outer * (n_fnn + n_dps)
    progress_bar = tqdm(total=total_steps, desc="训练进度", ncols=None)
    
    # 两重循环训练
    for outer_epoch in range(n_outer):
        # --- 阶段 1: 训练 FNN ---
        for fnn_step in range(n_fnn):
            opt_fnn.zero_grad()
            loss_fnn, loss_fnn_dict = calculate_loss_fnn(ode_model, patient_data, ab, patient_pids, y0)
            if torch.isfinite(loss_fnn):
                loss_fnn.backward()
                opt_fnn.step()
            
            # 计算当前DPS loss用于显示
            with torch.no_grad():
                loss_dps, loss_dps_dict = calculate_loss_dps(ode_model, patient_data, ab, patient_pids, y0)
            
            # 记录loss
            loss_history['outer_epoch'].append(outer_epoch + 1)
            loss_history['inner_step'].append(fnn_step + 1)
            loss_history['step_type'].append('FNN')
            loss_history['fnn_loss'].append(loss_fnn.item())
            loss_history['dps_loss'].append(loss_dps.item())
            
            # 更新进度条
            progress_bar.set_postfix({
                'Outer': f'{outer_epoch+1}/{n_outer}',
                'Phase': 'FNN',
                'FNN': f'{loss_fnn.item():.1f}',
                'D': f'{loss_fnn_dict["data"]:.1f}',
                'L1': f'{loss_fnn_dict["l1"]:.2f}',
                'DPS': f'{loss_dps.item():.1f}'
            })
            progress_bar.update(1)
        
        # --- 阶段 2: 训练 DPS (a,b) ---
        for dps_step in range(n_dps):
            opt_dps.zero_grad()
            loss_dps, loss_dps_dict = calculate_loss_dps(ode_model, patient_data, ab, patient_pids, y0)
            if torch.isfinite(loss_dps):
                loss_dps.backward()
                opt_dps.step()
            
            # 计算当前FNN loss用于显示
            with torch.no_grad():
                loss_fnn, loss_fnn_dict = calculate_loss_fnn(ode_model, patient_data, ab, patient_pids, y0)
            
            # 记录loss
            loss_history['outer_epoch'].append(outer_epoch + 1)
            loss_history['inner_step'].append(dps_step + 1)
            loss_history['step_type'].append('DPS')
            loss_history['fnn_loss'].append(loss_fnn.item())
            loss_history['dps_loss'].append(loss_dps.item())
            
            # 更新进度条
            progress_bar.set_postfix({
                'Outer': f'{outer_epoch+1}/{n_outer}',
                'Phase': 'DPS',
                'FNN': f'{loss_fnn.item():.1f}',
                'DPS': f'{loss_dps.item():.1f}',
                'D_dps': f'{loss_dps_dict["data"]:.1f}',
                'L2': f'{loss_dps_dict["l2"]:.2f}'
            })
            progress_bar.update(1)
    
    progress_bar.close()
    ode_model.eval()
    print("\n交替优化训练完成！")
    return ode_model, ab, loss_history


# --- 3. 辅助函数：自动加载预训练模型 ---
def load_pretrained_model(model_path='fnn_pretrain.pth'):
    """
    自动加载预训练模型，从state_dict推断模型结构
    """
    try:
        state_dict = torch.load(model_path, weights_only=True)
        
        # 处理键名不匹配的情况（移除"fnn."前缀）
        if any(k.startswith('fnn.') for k in state_dict.keys()):
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('fnn.'):
                    new_state_dict[k[4:]] = v  # 移除"fnn."前缀
                else:
                    new_state_dict[k] = v
            state_dict = new_state_dict
        
        # 从state_dict自动推断hidden_dim
        # fc1.weight的形状是 (hidden_dim, input_dim)
        fc1_weight = state_dict['fc1.weight']
        hidden_dim = fc1_weight.shape[0]
        input_dim = fc1_weight.shape[1]
        
        # fc3.weight的形状是 (output_dim, hidden_dim)
        fc3_weight = state_dict['fc3.weight']
        output_dim = fc3_weight.shape[0]
        
        print(f"从 {model_path} 检测到模型结构: input_dim={input_dim}, hidden_dim={hidden_dim}, output_dim={output_dim}")
        
        # 创建模型并加载参数
        model = ODENet(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
        model.load_state_dict(state_dict)
        print(f"成功加载预训练模型！")
        
        return model
        
    except FileNotFoundError:
        print(f"错误: 未找到 {model_path}。请先运行 pretrain.py 生成此文件。")
        exit()
    except Exception as e:
        print(f"加载模型时出错: {e}")
        exit()


# --- 4. 主程序 ---
if __name__ == '__main__':
    # 加载预训练的FNN模型
    print("\n--- 加载预训练的FNN模型 ---")
    fnn_pretrained = load_pretrained_model('fnn_pretrain.pth')
    
    # 交替优化训练
    final_model, trained_ab, loss_history = train_alternating(
        fnn_pretrained,
        patient_data,
        y0_cn_avg
    )
    
    if final_model is None:
        exit()
    
    # 保存训练后的模型和DPS参数
    torch.save(final_model.state_dict(), f'{name}.pth')
    torch.save(trained_ab, f'dps_{name}.pth')
    print(f"\n模型已保存到 {name}.pth")
    print(f"DPS参数已保存到 dps_{name}.pth")
    
    # --- 绘制损失曲线 ---
    print("\n--- 绘制损失曲线 ---")
    fig_loss, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 创建全局步数索引
    steps = list(range(len(loss_history['fnn_loss'])))
    
    # FNN Loss
    ax1.plot(steps, loss_history['fnn_loss'], 'b-', linewidth=2, alpha=0.7)
    ax1.set_xlabel('Training Step', fontsize=12)
    ax1.set_ylabel('FNN Loss', fontsize=12)
    ax1.set_title('FNN Loss Curve', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')  # 使用对数刻度更清晰
    
    # 标记FNN训练阶段
    fnn_steps = [i for i, t in enumerate(loss_history['step_type']) if t == 'FNN']
    if fnn_steps:
        ax1.scatter([fnn_steps[0]], [loss_history['fnn_loss'][fnn_steps[0]]], 
                   c='green', s=50, marker='o', label='FNN Phase', zorder=5)
    
    # DPS Loss
    ax2.plot(steps, loss_history['dps_loss'], 'r-', linewidth=2, alpha=0.7)
    ax2.set_xlabel('Training Step', fontsize=12)
    ax2.set_ylabel('DPS Loss', fontsize=12)
    ax2.set_title('DPS Loss Curve', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')  # 使用对数刻度更清晰
    
    # 标记DPS训练阶段
    dps_steps = [i for i, t in enumerate(loss_history['step_type']) if t == 'DPS']
    if dps_steps:
        ax2.scatter([dps_steps[0]], [loss_history['dps_loss'][dps_steps[0]]], 
                   c='orange', s=50, marker='s', label='DPS Phase', zorder=5)
    
    ax1.legend()
    ax2.legend()
    
    plt.tight_layout()
    loss_curve_filename = f'{name}_loss_curve.png'
    plt.savefig(loss_curve_filename)
    print(f"损失曲线已保存到 {loss_curve_filename}")
    plt.show()
    
    # --- 绘图 ---
    print("\n--- 生成可视化结果 ---")
    
    # 计算实际数据的s范围
    all_s_values = []
    for pid, dat in patient_data.items():
        if pid in trained_ab:
            a = trained_ab[pid]['a'].item()
            b = trained_ab[pid]['b'].item()
            s_values = a * dat['t'].numpy() + b
            all_s_values.extend(s_values)
    
    s_min, s_max = np.min(all_s_values), np.max(all_s_values)
    s_margin = (s_max - s_min) * 0.1  # 扩展10%的边距
    s_grid = torch.linspace(s_min - s_margin, s_max + s_margin, 300)
    print(f"s_grid范围: [{s_grid.min():.2f}, {s_grid.max():.2f}]")
    
    with torch.no_grad():
        try:
            y_pred = torch_odeint(final_model, y0_cn_avg, s_grid, method='dopri5', rtol=1e-4, atol=1e-5)
            y_pred_orig = pc.inv_nor(y_pred.numpy())
        except Exception as e:
            print(f"绘图时ODE求解失败: {e}")
            exit()
        
        TITLES = ['Aβ (A)', 'p-Tau (T)', 'N', 'Cognition (C)']
        colors = {'CN': 'orange', 'LMCI': 'green', 'AD': 'blue', 'Other': 'grey'}
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 11))
        axes = axes.flat
        
        for k in range(4):
            ax = axes[k]
            
            # 绘制数据点
            for pid, dat in patient_data.items():
                if pid in trained_ab:
                    stage = dat['stage']
                    a = trained_ab[pid]['a'].item()
                    b = trained_ab[pid]['b'].item()
                    s = a * dat['t'].numpy() + b
                    y_orig = pc.inv_nor(dat['y'][:, k].numpy(), k)
                    ax.scatter(s, y_orig, s=15, alpha=0.5, c=colors[stage])
            
            # 绘制FNN轨迹
            ax.plot(s_grid.numpy(), y_pred_orig[:, k], 'r-', lw=2.5, label='FNN Trajectory', zorder=3)
            
            ax.set_xlabel('Disease Progression Score (s)')
            ax.set_ylabel(TITLES[k])
            
            # 设置横轴范围与s_grid一致
            ax.set_xlim(s_grid.min().item(), s_grid.max().item())
            
            ax.legend()
            ax.grid(True, alpha=0.4)
            ax.set_title(TITLES[k])
        
        fig.suptitle('FNN Model with Alternating Optimization (Adam)', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f'{name}.png')
        print(f"结果图已保存到 {name}.png")
        plt.show()
    
    # --- 保存病人参数到Excel ---
    print("\n--- 保存病人参数 ---")
    patient_params_list = []
    for pid in sorted(trained_ab.keys()):
        a_value = trained_ab[pid]['a'].item()
        b_value = trained_ab[pid]['b'].item()
        stage = patient_data[pid]['stage'] if pid in patient_data else 'Unknown'
        patient_params_list.append({
            'PID': pid,
            'Stage': stage,
            'a': a_value,
            'b': b_value
        })
    
    # 创建DataFrame并保存为Excel
    params_df = pd.DataFrame(patient_params_list)
    excel_filename = f'dps_params.xlsx'
    params_df.to_excel(excel_filename, index=False)
    print(f"病人参数已保存到 {excel_filename}")
    print(f"共 {len(params_df)} 位病人的参数已记录")
    print(f"\n参数统计:")
    print(f"  a参数: 均值={params_df['a'].mean():.4f}, 标准差={params_df['a'].std():.4f}, 范围=[{params_df['a'].min():.4f}, {params_df['a'].max():.4f}]")
    print(f"  b参数: 均值={params_df['b'].mean():.4f}, 标准差={params_df['b'].std():.4f}, 范围=[{params_df['b'].min():.4f}, {params_df['b'].max():.4f}]")
    
    print("\n完整流程执行完毕。")