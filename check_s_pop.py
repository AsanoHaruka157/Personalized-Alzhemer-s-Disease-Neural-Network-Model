import numpy as np
import torch
import pccmnn as pc


def compute_s_values(csf_dict, dps_params):
    all_s_points = []
    for pid, sample in csf_dict.items():
        t = sample[:, 0]
        params = dps_params[pid]
        a = params['a']
        b = params['b']
        s = a * torch.tensor(t, dtype=torch.float32) + b
        all_s_points.append(s.detach().numpy())
    return np.concatenate(all_s_points)


if __name__ == "__main__":
    csf_dict = pc.load_data()

    dps_best_dict = torch.load('dps.pth', weights_only=False)
    dps_params = {
        pid: {
            'a': torch.tensor(params['a'], dtype=torch.float32),
            'b': torch.tensor(params['b'], dtype=torch.float32)
        }
        for pid, params in dps_best_dict.items()
    }

    s_pop = compute_s_values(csf_dict, dps_params)
    nan_count = int(np.isnan(s_pop).sum())
    print(f"NaN timepoints: {nan_count} / {s_pop.size}")
    print("s_pop:")
    print(s_pop)
