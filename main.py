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
        # === 函数形状正则化（已注释） ===
        # # 1. 在[-5:20:1]的s值上求解ODE和右侧函数
        # s_reg = torch.arange(-5, 21, 1, device=y0.device, dtype=y0.dtype)  # [-5, -4, ..., 19, 20]
        # 
        # # 确保从0开始积分到每个s值（需要包含0点）
        # s_reg_with_zero = torch.cat([torch.tensor([0.0], device=y0.device), s_reg[s_reg != 0]])
        # s_reg_with_zero, _ = torch.sort(s_reg_with_zero)
        # 
        # # 求解ODE得到各点的y值
        # y_reg_all = torch_odeint(ode_model, y0, s_reg_with_zero, method='dopri5', rtol=1e-4, atol=1e-5)
        # 
        # # 提取s_reg对应的y值
        # reg_indices = []
        # for s_val in s_reg:
        #     idx = (s_reg_with_zero == s_val).nonzero(as_tuple=True)[0]
        #     if len(idx) > 0:
        #         reg_indices.append(idx[0].item())
        # y_reg = y_reg_all[reg_indices]  # (26, 4)
        # 
        # # 计算每个点的dy/ds（ODE右侧函数）
        # dyds_list = []
        # for y_val in y_reg:
        #     dyds = ode_model(0, y_val)  # (4,)
        #     dyds_list.append(dyds)
        # dyds_all = torch.stack(dyds_list)  # (26, 4)
        # 
        # # 2. 正则化损失：要求[-5,0]和[15,20]区间平（接近0），[0,10]区间斜（绝对值大）
        # # [-5, 0]: indices 0-5
        # flat_region_1 = dyds_all[0:6]  # s=-5 to s=0
        # # [15, 20]: indices 20-25
        # flat_region_2 = dyds_all[20:26]  # s=15 to s=20
        # # [0, 10]: indices 5-15
        # steep_region = dyds_all[5:16]  # s=0 to s=10
        # 
        # # 平坦区域：惩罚绝对值大的导数
        # flat_loss = (flat_region_1 ** 2).sum() + (flat_region_2 ** 2).sum()
        # 
        # # 陡峭区域：鼓励绝对值大的导数（负向惩罚）
        # steep_loss = 1.0 / (steep_region.abs().sum() + 1e-6)
        
        # 参数正则化：惩罚a, b参数过大
        param_reg_loss = 0.0
        for pid in pids:
            param_reg_loss += ab[pid]['a'] ** 2 + ab[pid]['b'] ** 2
        
        # 总损失：加法形式的参数正则化
        lambda_param = 0.01  # 参数正则化权重
        
        total_loss = data_loss + lambda_param * param_reg_loss
        
        # 返回总损失和各分量
        loss_dict = {
            'total': total_loss.item() if torch.isfinite(total_loss) else float('inf'),
            'data': data_loss.item(),
            'param': param_reg_loss.item(),
            'lambda': lambda_param
        }
        
        if torch.isfinite(total_loss):
            return total_loss, loss_dict
        else:
            return torch.tensor(float('inf'), requires_grad=True), loss_dict

    except Exception as e:
        print(f"FNN loss 计算出错: {e}")
        import traceback
        traceback.print_exc()
        loss_dict = {'total': float('inf'), 'data': 0, 'param': 0, 'lambda': 0.01}
        return torch.tensor(float('inf'), requires_grad=True), loss_dict


# ===== 优化后的 DPS 损失 =====
def calculate_loss_dps(ode_model, patient_data, ab, pids, y0):
    """
    高效版 DPS loss：
    - 所有病人、所有时间点与 biomarker 合并；
    - 去重、排序，确保 torchdiffeq 时间轴严格递增；
    - 在去重时间轴上解一次 ODE，再映射回原索引。
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
            return torch.tensor(0.0, requires_grad=True)

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

        loss = ((y_pred_selected - y_sorted) ** 2).sum()
        return loss if torch.isfinite(loss) else torch.tensor(float('inf'), requires_grad=True)

    except Exception as e:
        print(f"DPS loss 计算出错: {e}")
        return torch.tensor(float('inf'), requires_grad=True)

def train_alternating(
    fnn_pretrained,
    patient_data,
    y0,
    dps_path='dps_pretrain.pth',
    n_epochs=80,
    lr_fnn=1e-3,         # FNN的Adam学习率
    lr_dps=1e-3,         # DPS的Adam学习率
):
    """
    交替优化FNN和DPS参数：
    1. 用Adam优化FNN
    2. 用Adam优化a,b参数
    """
    print("\n--- 开始交替优化训练 (Adam + Adam) ---")
    
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
        return None, None
        
    patient_pids = list(ab.keys())
    dps_params = [p for pid in patient_pids for p in ab[pid].values()]
    
    # 创建两个优化器：Adam用于FNN，Adam用于DPS
    opt_fnn = optim.Adam(ode_model.parameters(), lr=lr_fnn)
    opt_dps = optim.Adam(dps_params, lr=lr_dps)
    
    # 记录loss历史
    loss_history = {
        'epoch': [],
        'fnn_loss': [],
        'dps_loss': []
    }
    
    # 使用tqdm显示进度条
    progress_bar = tqdm(range(n_epochs), desc="训练进度", ncols=None)
    
    for epoch in progress_bar:
        # --- 步骤 1: 用Adam优化FNN（按算法1步骤3-4，对每个biomarker分别算loss）---
        opt_fnn.zero_grad()
        loss_fnn, loss_fnn_dict = calculate_loss_fnn(ode_model, patient_data, ab, patient_pids, y0)
        if torch.isfinite(loss_fnn):
            loss_fnn.backward()
            opt_fnn.step()
        
        # --- 步骤 2: 用Adam优化a,b参数（按算法1步骤8，对每个patient所有时间点算loss）---
        opt_dps.zero_grad()
        loss_dps = calculate_loss_dps(ode_model, patient_data, ab, patient_pids, y0)
        if torch.isfinite(loss_dps):
            loss_dps.backward()
            opt_dps.step()
        
        # 记录loss
        loss_history['epoch'].append(epoch + 1)
        loss_history['fnn_loss'].append(loss_fnn.item())
        loss_history['dps_loss'].append(loss_dps.item())
        
        # 更新进度条显示的信息
        progress_bar.set_postfix({
            'FNN': f'{loss_fnn.item():.1f}',
            'D': f'{loss_fnn_dict["data"]:.1f}',
            'Prm': f'{loss_fnn_dict["param"]:.1f}',
            'DPS': f'{loss_dps.item():.1f}'
        })
    
    ode_model.eval()
    print("交替优化训练完成！")
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
    
    # FNN Loss
    ax1.plot(loss_history['epoch'], loss_history['fnn_loss'], 'b-', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('FNN Loss', fontsize=12)
    ax1.set_title('FNN Loss Curve', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')  # 使用对数刻度更清晰
    
    # DPS Loss
    ax2.plot(loss_history['epoch'], loss_history['dps_loss'], 'r-', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('DPS Loss', fontsize=12)
    ax2.set_title('DPS Loss Curve', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')  # 使用对数刻度更清晰
    
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