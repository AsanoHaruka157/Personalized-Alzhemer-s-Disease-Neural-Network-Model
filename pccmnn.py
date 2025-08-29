import numpy as np
import random
import torch

def load_data():
    import pandas as pd

    data = pd.read_excel("data.xlsx", sheet_name='Sheet1')

    data = data.to_numpy()

    csf_dict = {}

    # Group data by the first column (RID)
    for row in data:
        rid = int(row[0])
        values = row[1:]

        # Append the values to the corresponding key
        if rid not in csf_dict:
            csf_dict[rid] = []  # Initialize a new list if the key is not yet in the dictionary
        csf_dict[rid].append(values)

    for rid in csf_dict:
        csf_dict[rid] = np.array(csf_dict[rid])

    keys_to_delete = [key for key in csf_dict if csf_dict[key].shape[0] == 1]

    for key in keys_to_delete:
        del csf_dict[key]

    return csf_dict

def load_rawdata():
    import pandas as pd

    data = pd.read_excel("data.xlsx", sheet_name='rawdata')

    data = data.to_numpy()

    csf_dict = {}

    # Group data by the first column (RID)
    for row in data:
        rid = int(row[0])
        values = row[1:]

        # Append the values to the corresponding key
        if rid not in csf_dict:
            csf_dict[rid] = []  # Initialize a new list if the key is not yet in the dictionary
        csf_dict[rid].append(values)

    for rid in csf_dict:
        csf_dict[rid] = np.array(csf_dict[rid])

    keys_to_delete = [key for key in csf_dict if csf_dict[key].shape[0] == 1]

    for key in keys_to_delete:
        del csf_dict[key]

    return csf_dict

def load_stage_dict():
    import pandas as pd
    df = pd.read_excel('rawdata.xlsx', sheet_name='ADNI Org.')
    stage_dict = {}
    for index, row in df.iterrows():
        rid = row['RID']
        stage = row['DX_bl']
        if rid not in stage_dict:
            stage_dict[rid] = stage
    return stage_dict

<<<<<<< HEAD
def inv_nor(data, k=None):
    mean_std = np.load('mean_std.npy')
    
    means = mean_std[0]
    stds = mean_std[1]
    
    data = np.asarray(data)
    if k is not None:
        original_data = (data.T * stds[k]) + means[k]
    else:
        original_data = (data * stds) + means
    
    return original_data
=======
def sampling(csf_dict, lenth=2, num=1):
    sh_list = []
    for key in csf_dict:
        sh = csf_dict[key].shape[0]
        if sh >= lenth:
            sh_list.append(key)

    return random.sample(sh_list, num)
>>>>>>> 47820850003ccd5bb3d8b9a53a794d0819d12900
