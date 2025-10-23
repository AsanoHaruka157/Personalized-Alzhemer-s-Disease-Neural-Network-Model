import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import pccmnn as pc
try:
    from torchdiffeq import odeint as torch_odeint
except ImportError:
    raise ImportError("Please install torchdiffeq to run this script: pip install torchdiffeq")

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
    avg_y0 = torch.stack(cn_y0s).mean(dim=0)
    print(f"使用CN群體的平均初始值: {avg_y0.numpy()}")
    return avg_y0

y0_cn_avg = get_cn_average_y0(patient_data, stage_dict)
name = 'fnn'

# --- 1. 定義FNN模型（從pretrain.py復制ODENet結構）---
class ODENet(nn.Module):
    """单隐藏层神经网络用于学习dy/ds"""
    def __init__(self, input_dim=4, hidden_dim=256, output_dim=4):
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

class ODEModel(nn.Module):
    """ODE模型包装器"""
    def __init__(self, fnn_model):
        super().__init__()
        self.fnn = fnn_model
        
    def forward(self, t, y):
        """torchdiffeq的函数签名是 func(t, y)"""
        return self.fnn(y)

# --- 2. 定義訓練流程 ---
def calculate_loss(ode_model, patient_data, ab, pids, y0):
    """计算ODE模型在数据点上的损失"""
    all_s, all_y = [], []
    for pid in pids:
        dat = patient_data[pid]
        s_values = ab[pid]['a'] * dat['t'] + ab[pid]['b']
        all_s.append(s_values)
        all_y.append(dat['y'])
    
    s_global, y_global = torch.cat(all_s), torch.cat(all_y)
    s_sorted, sort_indices = torch.sort(s_global)
    y_sorted = y_global[sort_indices]
    
    try:
        y_pred = torch_odeint(ode_model, y0, s_sorted, method='dopri5', rtol=1e-4, atol=1e-5)
        loss = ((y_pred - y_sorted) ** 2).mean()
        return loss if torch.isfinite(loss) else torch.tensor(float('inf'))
    except Exception:
        return torch.tensor(float('inf'))

def train_alternating(
    fnn_pretrained,
    patient_data,
    y0,
    dps_path='dps.pth',
    n_epochs=50,
    lr_fnn=5e-4,         # FNN的LBFGS学习率
    lr_dps=1e-3,         # DPS的LBFGS学习率
    max_iter_fnn=20,     # FNN的LBFGS最大迭代次数
    max_iter_dps=10      # DPS的LBFGS最大迭代次数
):
    """
    交替优化FNN和DPS参数：
    1. 用LBFGS优化FNN
    2. 用LBFGS优化a,b参数
    """
    print("\n--- 开始交替优化训练 (LBFGS) ---")
    
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
    
    # 创建两个LBFGS优化器
    opt_fnn = optim.LBFGS(
        ode_model.parameters(),
        lr=lr_fnn,
        max_iter=max_iter_fnn,
        line_search_fn="strong_wolfe"
    )
    
    opt_dps = optim.LBFGS(
        dps_params,
        lr=lr_dps,
        max_iter=max_iter_dps,
        line_search_fn="strong_wolfe"
    )
    
    for epoch in range(n_epochs):
        # --- 步骤 1: 用LBFGS优化FNN ---
        def closure_fnn():
            opt_fnn.zero_grad()
            loss = calculate_loss(ode_model, patient_data, ab, patient_pids, y0)
            if torch.isfinite(loss):
                loss.backward()
            return loss
        
        loss_fnn = opt_fnn.step(closure_fnn)
        
        # --- 步骤 2: 用LBFGS优化a,b参数 ---
        def closure_dps():
            opt_dps.zero_grad()
            loss = calculate_loss(ode_model, patient_data, ab, patient_pids, y0)
            if torch.isfinite(loss):
                loss.backward()
            return loss
        
        loss_dps = opt_dps.step(closure_dps)
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{n_epochs}] | FNN Loss: {loss_fnn.item():.6f} | DPS Loss: {loss_dps.item():.6f}")
    
    ode_model.eval()
    print("交替优化训练完成！")
    return ode_model, ab


# --- 3. 主程序 ---
if __name__ == '__main__':
    # 加载预训练的FNN模型
    print("\n--- 加载预训练的FNN模型 ---")
    fnn_pretrained = ODENet(input_dim=4, hidden_dim=256, output_dim=4)
    try:
        state_dict = torch.load('fnn.pth', weights_only=True)
        
        # 处理键名不匹配的情况（移除"fnn."前缀）
        if any(k.startswith('fnn.') for k in state_dict.keys()):
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('fnn.'):
                    new_state_dict[k[4:]] = v  # 移除"fnn."前缀
                else:
                    new_state_dict[k] = v
            state_dict = new_state_dict
        
        fnn_pretrained.load_state_dict(state_dict)
        print("成功从 fnn.pth 加载预训练模型。")
    except FileNotFoundError:
        print("错误: 未找到 fnn.pth。请先运行 pretrain.py 生成此文件。")
        exit()
    
    # 交替优化训练
    final_model, trained_ab = train_alternating(
        fnn_pretrained,
        patient_data,
        y0_cn_avg,
        dps_path='dps.pth',
        n_epochs=50,
        lr_fnn=1e-3,
        lr_dps=1e-3,
        max_iter_fnn=20,
        max_iter_dps=10
    )
    
    if final_model is None:
        exit()
    
    # 保存训练后的模型和DPS参数
    torch.save(final_model.state_dict(), f'{name}.pth')
    torch.save(trained_ab, f'dps_{name}.pth')
    print(f"\n模型已保存到 {name}.pth")
    print(f"DPS参数已保存到 dps_{name}.pth")
    
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
        
        fig.suptitle('FNN Model with Alternating Optimization (LBFGS)', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f'{name}.png')
        print(f"结果图已保存到 {name}.png")
        plt.show()
    
    print("\n完整流程执行完毕。")