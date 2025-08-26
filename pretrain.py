import torch
import matplotlib.pyplot as plt
import numpy as np
import pccmnn as pc

# 加载数据：与 main.py 一致
csf_dict = pc.load_csf_data()
print("Number of valid patients:", len(csf_dict))

# 阶段信息
stage_dict = pc.load_stage_dict()
colors = {'CN': 'orange', 'LMCI': 'green', 'AD': 'blue', 'Other': 'grey'}
TITLES = ['Aβ (A)', 'p-Tau (T)', 'N', 'Cognition (C)']

# 绘图
fig, axes = plt.subplots(2, 2, figsize=(9, 6))
axes = axes.flat

for k in range(4):
    ax = axes[k]
    # 按阶段收集散点
    t_by_stage = {'CN': [], 'LMCI': [], 'AD': [], 'Other': []}
    y_by_stage = {'CN': [], 'LMCI': [], 'AD': [], 'Other': []}

    for pid, sample in csf_dict.items():
        t = torch.from_numpy(sample[:, 0]).float()
        yk = torch.from_numpy(sample[:, 1 + k]).float()
        stage = stage_dict.get(pid, 'Other')
        if stage not in t_by_stage:
            stage = 'Other'
        t_by_stage[stage].append(t)
        y_by_stage[stage].append(yk)

    # 绘制各阶段散点
    for stage, t_list in t_by_stage.items():
        if t_list:
            t_all = torch.cat(t_list).numpy()
            y_all = torch.cat(y_by_stage[stage]).numpy()
            ax.scatter(t_all, y_all, s=15, alpha=0.6, c=colors[stage], label=stage)

    ax.set_xlabel('Age (years)')
    ax.set_ylabel(TITLES[k])
    ax.legend(fontsize=8)

fig.suptitle('CSF biomarkers vs Age (no DPS transform)')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('raw_data.png')
plt.show()

############################################
# DPS 预训练与散点绘制（添加）
############################################

import torch.nn.functional as F

def build_patient_data(csf_dict):
    patient_data = {}
    for pid, sample in csf_dict.items():
        t = torch.from_numpy(sample[:, 0]).float()
        y = torch.from_numpy(sample[:, 1:5]).float()
        patient_data[pid] = {"t": t, "y": y}
    return patient_data

def compute_patient_score(y_tensor: torch.Tensor) -> float:
    # (A + N) - (T + C)
    mean_vals = y_tensor.mean(dim=0)
    score = float(mean_vals[0] + mean_vals[2] - mean_vals[1] - mean_vals[3])
    return score

def interval_from_score(score: float, lo_healthy=-10.0, hi_healthy=0.0, lo_diseased=10.0, hi_diseased=20.0,
                        threshold: float = 0.0) -> tuple:
    if score >= threshold:
        return (lo_healthy, hi_healthy)
    return (lo_diseased, hi_diseased)

def interval_penalty(s_values: torch.Tensor, interval: tuple) -> torch.Tensor:
    lo, hi = interval
    below = torch.clamp(lo - s_values, min=0.0)
    above = torch.clamp(s_values - hi, min=0.0)
    penalty = below.pow(2) + above.pow(2)
    return penalty.mean()

def pretrain_dps(patient_data: dict, stage_dict: dict, epochs: int = 500, lr: float = 1e-2,
                 weight_reg: float = 1e-3, verbose: bool = True):
    """
    最简单的连续诱导：不设 target。
    - 每位患者计算一个分数 z_pid = mean((-Aβ) + (+Tau) + (-N) + (+C))。
    - 优化目标：让 (a_pid + b_pid) 与 z_pid 同号并尽可能大：
        loss_pid = -(z_pid) * (a + b) + λ*(a^2 + b^2) + 边界惩罚
      其中 a=4*sigmoid(θ0)+eps ∈ (0,4]，b=θ1，自然满足 a≤4。
    - s = a*t + b，并对 s 超出 [-10,20] 的部分加二次惩罚。
    初始化：a=1, b=0（通过选择 θ0=logit(1/4)，θ1=0）。
    返回: ab（包含 θ）
    """
    # 预计算每位患者的分数 z_pid
    pid_to_score = {}
    for pid, dat in patient_data.items():
        y = dat['y']
        z = (-y[:, 0]) + (y[:, 1]) + (-y[:, 2]) + (y[:, 3])
        pid_to_score[pid] = float(z.mean())

    # 2) 初始化 α,β 
    ab = {}
    for pid, dat in patient_data.items():
        # 令 a 初始为 1，对应 raw θ0 = logit(a/4) = log(0.25/0.75)
        theta0_init = float(np.log(0.25 / 0.75))  # ≈ -1.098612288
        theta = torch.tensor([theta0_init, 0.0], dtype=torch.float32).requires_grad_(True)
        ab[pid] = {"theta": theta}

    optimizer = torch.optim.Adam([ab[pid]['theta'] for pid in ab], lr=lr)

    for ep in range(epochs):
        optimizer.zero_grad()
        total_loss = 0.0
        for pid, dat in patient_data.items():
            t = dat['t']
            theta = ab[pid]['theta']
            # α ∈ (0,4]
            alpha = 4.0 * torch.sigmoid(theta[0]) + 1e-4
            beta = theta[1]
            s_vals = alpha * t + beta

            # 诱导项：-(z_pid)*(a+b)
            z_pid = pid_to_score[pid]
            induce = - z_pid * (alpha + beta)

            # s 边界惩罚（超出 [-10,20] 的部分二次惩罚）
            below = torch.clamp(-10.0 - s_vals, min=0.0)
            above = torch.clamp(s_vals - 20.0, min=0.0)
            bound_penalty = (below.pow(2) + above.pow(2)).mean()

            # 轻微 L2 正则
            reg_loss = weight_reg * (alpha.pow(2) + beta.pow(2))

            total_loss = total_loss + induce + bound_penalty + reg_loss

        total_loss.backward()
        optimizer.step()
        if verbose and (ep % 50 == 0 or ep == epochs - 1):
            print(f"[DPS Pretrain] epoch {ep:04d} | loss={float(total_loss):.6f}")

    return ab, {"score": pid_to_score}

def plot_dps_scatter(patient_data: dict, stage_dict: dict, ab: dict, save_path: str = 'dps_pretrain.png'):
    s_points = []
    y_points = []
    color_points = []
    stage_colors = {'CN': 'orange', 'LMCI': 'green', 'AD': 'blue', 'Other': 'grey'}

    for pid, dat in patient_data.items():
        stage = stage_dict.get(pid, 'Other')
        color = stage_colors.get(stage, 'grey')
        theta = ab[pid]['theta'].detach()
        alpha = F.softplus(theta[0]).item() + 1e-4
        beta = theta[1].item()
        s_vals = alpha * dat['t'] + beta
        s_points.append(s_vals)
        y_points.append(dat['y'])
        color_points.extend([color] * s_vals.shape[0])

    s_all = torch.cat(s_points).numpy()
    y_all = torch.cat(y_points).numpy()
    y_all_orig = pc.inv_nor(y_all)

    # 仅对 s 按 10% 与 90% 分位做过滤
    s10 = np.quantile(s_all, 0.10)
    s90 = np.quantile(s_all, 0.90)
    mask_s = (s_all >= s10) & (s_all <= s90)
    s_all = s_all[mask_s]
    y_all_orig = y_all_orig[mask_s]
    color_points = np.array(color_points)[mask_s]

    titles = ['Aβ (A)', 'p-Tau (T)', 'N', 'Cognition (C)']
    fig, axes = plt.subplots(2, 2, figsize=(9, 6))
    axes = axes.flat
    for k in range(4):
        ax = axes[k]
        ax.scatter(s_all, y_all_orig[:, k], c=color_points, s=15, alpha=0.6)
        ax.set_xlabel('Disease Progression Score (s)')
        ax.set_ylabel(titles[k])
        ax.grid(True, alpha=0.3)
    fig.suptitle('DPS pretraining: s vs biomarkers (colored by stage)')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path)
    plt.show()



if __name__ == '__main__':
    # 构建 patient_data 结构并进行 DPS 预训练与绘制
    patient_data = build_patient_data(csf_dict)
    ab, _ = pretrain_dps(patient_data, stage_dict, epochs=400, lr=5e-2, verbose=True)
    model_path = 'dps.pth'
    torch.save(ab, model_path)
    print(f"Saved DPS parameters to {model_path}")
    plot_dps_scatter(patient_data, stage_dict, ab, save_path='dps_pretrain.png')
