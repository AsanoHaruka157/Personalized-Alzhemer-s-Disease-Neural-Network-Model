import numpy as np
import pandas as pd
import torch
import pccmnn as pc


def assign_dps_params(csf_dict, stage_dict):
    s_ranges = {
        'CN': (-10, 0),
        'LMCI': (-2, 8),
        'AD': (5, 20),
        'Other': (-10, 20)
    }
    a_init_values = {'CN': 1.0, 'LMCI': 2.0, 'AD': 4.0, 'Other': 1.0}

    dps_params = {}
    for pid, sample in csf_dict.items():
        stage = stage_dict.get(pid, 'Other')
        t = sample[:, 0]

        a_init = a_init_values[stage]
        a_param = torch.tensor(a_init, dtype=torch.float32)

        s_min, s_max = s_ranges[stage]
        t_initial = t[0]
        s_initial_target = np.random.uniform(s_min, s_max)
        b_init = s_initial_target - a_init * t_initial
        b_param = torch.tensor(b_init, dtype=torch.float32)

        dps_params[pid] = {'a': a_param, 'b': b_param, 'stage': stage}

    return dps_params


def compute_s_series(csf_dict, dps_params):
    rows = []
    for pid, sample in csf_dict.items():
        t = sample[:, 0]
        params = dps_params[pid]
        a = params['a'].item()
        b = params['b'].item()
        s = a * t + b
        for idx, (ti, si) in enumerate(zip(t, s)):
            rows.append({
                'pid': pid,
                'visit_index': idx,
                't': float(ti),
                's': float(si)
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    csf_dict = pc.load_data()
    stage_dict = pc.load_stage_dict()

    dps_params = assign_dps_params(csf_dict, stage_dict)

    dps_rows = []
    for pid, params in dps_params.items():
        dps_rows.append({
            'pid': pid,
            'stage': params['stage'],
            'a_init': params['a'].item(),
            'b_init': params['b'].item()
        })
    dps_df = pd.DataFrame(dps_rows)

    s_df = compute_s_series(csf_dict, dps_params)

    output_path = 'init_dps_and_s.xlsx'
    with pd.ExcelWriter(output_path) as writer:
        dps_df.to_excel(writer, sheet_name='dps_init', index=False)
        s_df.to_excel(writer, sheet_name='s_series', index=False)

    print(f"已保存到 {output_path}")
