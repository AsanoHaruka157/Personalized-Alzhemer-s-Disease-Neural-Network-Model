
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pccmnn as pc

# ------------------ 模型 ------------------
class PopSigmoidFree(nn.Module):
    """ 每个指标一条四参数sigmoid：g_k(s)= a_k*(1+exp(-b_k*(s-c_k)))^{-1} + d_k
        不做正负约束，允许上升/下降 """
    def __init__(self, num_markers=4):
        super().__init__()
        self.a = nn.Parameter(torch.empty(num_markers).uniform_(-1.0, 1.0))
        self.b = nn.Parameter(torch.empty(num_markers).uniform_(0.2, 1.5))  # 斜率>0 更稳定，方向交给 a 的正负
        self.c = nn.Parameter(torch.zeros(num_markers))                      # 中心
        self.d = nn.Parameter(torch.zeros(num_markers))                      # 基线

    def forward(self, s):
        s = torch.as_tensor(s, dtype=torch.float32).unsqueeze(-1)  # (...,1)
        return self.d + self.a * (1.0 / (1.0 + torch.exp(-self.b * (s - self.c))))


# ------------------ 初始化 (a,b) 确保 s ∈ [-10,20] ------------------
def init_ab_in_range(t, a_low=0.0, a_high=4.0, s_min=-10.0, s_max=20.0):
    a = torch.empty(1).uniform_(a_low, a_high)  # U(0,4)
    tmin, tmax = t.min(), t.max()
    # 若跨度过大，收缩 a 以保证 at+b 覆盖区间 ≤ 30
    span_t = (tmax - tmin).clamp(min=1e-6)
    a = torch.minimum(a, torch.tensor((s_max - s_min) / span_t))
    # 让均值点落在范围中部
    t_mean = t.mean()
    b = torch.empty(1).uniform_(s_min + 5.0, s_max - 5.0) - a * t_mean
    return torch.log(a + 1e-4), b.squeeze(0)


# ------------------ 主流程 ------------------
def run(
    outer_loops=6,         # 交替优化轮数（群体/个体）
    epochs_pop=200,        # 每轮群体参数步数
    epochs_ab=200,         # 每轮个体 a,b 步数
    lr_pop=5e-2,
    lr_ab=1e-2,
    seed=0
):
    torch.manual_seed(seed); np.random.seed(seed)

    # 读取原始数据（时间+原单位 A/T/N/C）
    csf_dict = pc.load_csf_data()
    print(f"原始数据加载完成，共 {len(csf_dict)} 个患者")

    # 构造 patient_data（t:Tensor, y:Tensor 标准化后的 4 维）
    patient = {}
    
    # 检查 patient 字典是否为空
    if len(patient) == 0:
        print("警告：patient 字典为空！请检查数据加载和预处理步骤。")
        return

    # 分期（用于配色和 xlsx）
    stage = pc.load_stage_dict()
    color_map = {'CN': '#f39c12', 'LMCI': '#27ae60', 'AD': '#2980b9'}

    # 群体模型
    model = PopSigmoidFree(4)

    # 初始化每人的 (a,b)；可学习参数为 theta0, theta1 其中 a=exp(theta0)+1e-4, b=theta1

    ab_params = nn.ParameterDict()
    for pid, dat in patient.items():
        th0, th1 = init_ab_in_range(dat['t'])
        # 统一用字符串 key；存一条 2 维向量参数 [theta0, theta1]
        ab_params[str(pid)] = nn.Parameter(torch.stack([th0, th1]))
    # 优化器
    opt_pop = optim.Adam(model.parameters(), lr=lr_pop, weight_decay=1e-5)
    opt_ab  = optim.Adam(ab_params.parameters(), lr=lr_ab)  # 不会再是空列表

    # 方差权重（在标准化空间下可取 1；也可按残差自适应，这里用 1 更稳）
    sigma = torch.ones(4)

    # 交替优化
    for L in range(outer_loops):
        # A) 固定 (a,b) 学群体
        for _ in range(epochs_pop):
            opt_pop.zero_grad()
            loss = 0.0
            for pid, dat in patient.items():
                t, y = dat['t'], dat['y']
                th = ab_params[str(pid)]
                a = torch.exp(th[0]) + 1e-4
                b = th[1]
                s = a * t + b
                # 越界惩罚，拉回 [-10,20]
                pen = torch.relu(s - 20).mean() + torch.relu(-10 - s).mean()
                yhat = model(s)
                mse = ((yhat - y)**2 / sigma).mean()
                loss = loss + mse + 1e-3 * pen
            loss.backward(); opt_pop.step()

        # B) 固定群体，更新每个体 (a,b)
        for _ in range(epochs_ab):
            opt_ab.zero_grad()
            loss = 0.0
            for pid, dat in patient.items():
                t, y = dat['t'], dat['y']
                th = ab_params[str(pid)]
                a = torch.exp(th[0]) + 1e-4
                b = th[1]
                s = a * t + b
                pen = torch.relu(s - 20).mean() + torch.relu(-10 - s).mean()
                yhat = model(s)
                mse = ((yhat - y)**2 / sigma).mean()
                loss = loss + mse + 1e-3 * pen + 1e-4 * (th[0]**2 + 0.1*th[1]**2)
            loss.backward(); opt_ab.step()

    # —— 导出 a,b 字典 ——（float）
    with torch.no_grad():
        ab_dict = {
            int(pid_str): (float(torch.exp(th[0]) + 1e-4), float(th[1]))
            for pid_str, th in ab_params.items()
        }



    # —— 画群体图（逆变换回原单位）——
    s_grid = torch.linspace(-10, 20, 300)
    with torch.no_grad():
        yhat_norm = model(s_grid).numpy()           # (300,4)  标准化空间
        yhat_orig = pc.inv_normalize(yhat_norm, stats)  # 原单位

    titles = ['Amyloid-beta (pg/ml)', 'Tau (pg/ml)', 'Hippocampus (ml)', 'ADAS13']
    fig, axes = plt.subplots(2, 2, figsize=(10, 6)); axes = axes.ravel()

    # 散点（逆变换）
    for k in range(4):
        for pid, dat in patient.items():
            th = ab_params[str(pid)]
            a = torch.exp(th[0]) + 1e-4; b = th[1]
            s_i = (a * dat['t'] + b).numpy()
            y_i = dat['y'][:, k].numpy()
            # 逆变换各自点
            y_i_orig = pc.inv_normalize(y_i.reshape(-1, 4) if k==0 else
                                        np.stack([np.zeros_like(y_i),
                                                  y_i, np.zeros_like(y_i), np.zeros_like(y_i)], axis=1)
                                        , stats)
            # 上面写法有点笨重，这里简化：把标准化数组还原时，仅替换第 k 列
            y_tmp = np.zeros((len(y_i), 4))
            y_tmp[:, k] = y_i
            y_i_orig = pc.inv_normalize(y_tmp, stats)[:, k]

            st = stage.get(pid, 'CN')
            col = color_map.get(st, '#7f8c8d')
            axes[k].scatter(s_i, y_i_orig, s=8, alpha=0.35, c=col)

        axes[k].plot(s_grid.numpy(), yhat_orig[:, k], lw=2, c='k')
        axes[k].set_title(titles[k]); axes[k].set_xlabel('DPS  s'); axes[k].set_ylabel('Value')

    plt.tight_layout(); fig.savefig('sigmoid.png', dpi=160); plt.close(fig)

    return ab_dict


if __name__ == "__main__":
    run()
