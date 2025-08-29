import torch
import matplotlib.pyplot as plt
import numpy as np
import pccmnn as pc
import torch.nn.functional as F

def load_patient_data():
    csf_dict = pc.load_csf_data()
    patient_data = {}
    for pid, sample in csf_dict.items():
        t = torch.from_numpy(sample[:, 0]).float()
        y = torch.from_numpy(sample[:, 1:5]).float()
        patient_data[pid] = {"t": t, "y": y}
    return patient_data


def load_ab_mapping(path: str):
    ab_raw = torch.load(path)
    ab = {}
    for pid, val in ab_raw.items():
        if isinstance(val, dict) and 'theta' in val:
            ab[pid] = {'theta': val['theta']}
        else:
            # assume tuple/list (a, b)
            try:
                a, b = val
                ab[pid] = {'theta': torch.tensor([float(a), float(b)])}
            except Exception:
                raise ValueError(f"Unsupported DPS format for pid {pid}: {type(val)}")
    return ab


def plot_scatter(patient_data, ab, stage_dict):
    titles = ['Aβ (A)', 'p-Tau (T)', 'N', 'Cognition (C)']
    colors = {'CN': 'orange', 'LMCI': 'green', 'AD': 'blue', 'Other': 'grey'}

    # compute global s range to display
    all_s = []
    for p, dat in patient_data.items():
        if p not in ab:
            continue
        a, b = F.softplus(ab[p]['theta'][0]).item() + 1e-4, ab[p]['theta'][1]
        s_values = a * dat['t'] + b
        all_s.append(s_values)
    if not all_s:
        raise RuntimeError('No patients with DPS parameters found.')
    all_s_flat = torch.cat(all_s)
    s_min = torch.quantile(all_s_flat, 0.10)
    s_max = torch.quantile(all_s_flat, 0.90)

    fig, axes = plt.subplots(2, 2, figsize=(9, 6))
    axes = axes.flat
    for k in range(4):
        ax = axes[k]
        grouped_s = {'CN': [], 'LMCI': [], 'AD': [], 'Other': []}
        grouped_y = {'CN': [], 'LMCI': [], 'AD': [], 'Other': []}

        for p, dat in patient_data.items():
            if p not in ab:
                continue
            a, b = F.softplus(ab[p]['theta'][0]).item() + 1e-4, ab[p]['theta'][1]
            s_values = a * dat['t'] + b
            y_values = dat['y'][:, k]

            mask = (s_values >= s_min) & (s_values <= s_max)
            if not torch.any(mask):
                continue

            stage = stage_dict.get(p, 'Other')
            if stage not in grouped_s:
                stage = 'Other'
            grouped_s[stage].append(s_values[mask])
            grouped_y[stage].append(y_values[mask])

        for stage, s_list in grouped_s.items():
            if not s_list:
                continue
            s_all = torch.cat(s_list).detach().numpy()
            y_all = torch.cat(grouped_y[stage]).detach().numpy()
            # de-normalize if needed
            y_all = pc.inv_nor(y_all, k)
            ax.scatter(s_all, y_all, s=15, alpha=0.6, c=colors[stage], label=stage)

        ax.set_xlabel('Disease progression score  s')
        ax.set_ylabel(titles[k])
        ax.legend(fontsize=8)

    fig.suptitle('CSF scatter vs DPS (10-90 percentile s-range)')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def main():
    patient_data = load_patient_data()
    stage_dict = pc.load_stage_dict()
    ab = load_ab_mapping('dps.pth')
    plot_scatter(patient_data, ab, stage_dict)


if __name__ == '__main__':
    main()


