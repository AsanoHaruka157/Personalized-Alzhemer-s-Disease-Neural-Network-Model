"""
Shared utility module for the Personalized Alzheimer's Disease Neural Network Model.

This module provides common data loading and inverse normalization utilities
used across all pipeline scripts (sigmoid fitting, FNN training, main training,
and personalization).

Note: This project currently contains two independent experimental pipelines
(main.py and fnn.py) with different model architectures and training strategies.
These will be unified in a future refactoring.
"""

import numpy as np


def load_data():
    """Load normalized patient biomarker data from data.xlsx (Sheet1).

    Returns:
        dict: Mapping from patient ID (int) to numpy array of shape (T, 5),
              where columns are [AGE, ABETA, TAU, N, C].
    """
    import pandas as pd

    data = pd.read_excel("../data/data.xlsx", sheet_name='Sheet1')
    data = data.to_numpy()

    csf_dict = {}
    for row in data:
        rid = int(row[0])
        values = row[1:]
        if rid not in csf_dict:
            csf_dict[rid] = []
        csf_dict[rid].append(values)

    for rid in csf_dict:
        csf_dict[rid] = np.array(csf_dict[rid])

    return csf_dict


def load_rawdata():
    """Load raw (unnormalized) patient data from data.xlsx (rawdata sheet).

    Returns:
        dict: Mapping from patient ID (int) to numpy array of shape (T, 5),
              where columns are [AGE, ABETA, TAU, N, C].
    """
    import pandas as pd

    data = pd.read_excel("../data/data.xlsx", sheet_name='rawdata')
    data = data.to_numpy()

    csf_dict = {}
    for row in data:
        rid = int(row[0])
        values = row[1:]
        if rid not in csf_dict:
            csf_dict[rid] = []
        csf_dict[rid].append(values)

    for rid in csf_dict:
        csf_dict[rid] = np.array(csf_dict[rid])

    return csf_dict


def load_stage_dict():
    """Load patient diagnostic stage labels from rawdata.xlsx.

    Returns:
        dict: Mapping from patient ID (int) to stage label ('CN', 'LMCI', 'AD', or 'Other').
    """
    import pandas as pd
    df = pd.read_excel('../data/rawdata.xlsx', sheet_name='ADNI Org.')
    stage_dict = {}
    for index, row in df.iterrows():
        rid = row['RID']
        stage = row['DX_bl']
        if rid not in stage_dict:
            stage_dict[rid] = stage
    return stage_dict


def inv_nor(data, k=None, is_std=False):
    """Inverse normalization: map z-scored values back to original physical scale.

    Uses the precomputed mean and standard deviation stored in mean_std.npy.

    Args:
        data (np.ndarray): Normalized data array of shape (N, 4) or (N,).
        k (int, optional): If provided, apply inverse normalization only to
            biomarker index k. Defaults to None (all 4 biomarkers).
        is_std (bool): If True, only scale by std without adding mean
            (used for standard deviation values). Defaults to False.

    Returns:
        np.ndarray: Data in original physical units.
    """
    mean_std = np.load('../data/mean_std.npy')
    means = mean_std[0]
    stds = mean_std[1]

    data = np.asarray(data)

    if is_std:
        if k is not None:
            return data.T * stds[k]
        else:
            return data * stds

    if k is not None:
        return (data.T * stds[k]) + means[k]
    else:
        return (data * stds) + means
