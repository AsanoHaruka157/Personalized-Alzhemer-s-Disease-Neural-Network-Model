import numpy as np
import random
import torch

def load_csf_data():
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

def sampling(csf_dict, lenth=2, num=1):
    sh_list = []
    for key in csf_dict:
        sh = csf_dict[key].shape[0]
        if sh >= lenth:
            sh_list.append(key)

    return random.sample(sh_list, num)

def re_nor(x, i):
    percentile = np.load('quantile.npy')
    x95, x5 = percentile[0 , i], percentile[1 , i]
    x_inv = x5 + (x95 - x5) * x
    return x_inv

def predict(model, s, y0, x=None):
    # Get the device of the model
    device = next(model.parameters()).device

    # Ensure `s` is a torch.Tensor and on the correct device
    if isinstance(s, torch.Tensor):
        s = s.to(device)
    else:
        s = torch.tensor(s, dtype=torch.float64, device=device)

    # Ensure `y0` is on the correct device
    if isinstance(y0, torch.Tensor):
        y0 = y0.to(device)
    else:
        y0 = torch.tensor(y0, dtype=torch.float64, device=device)

    if x is None:
        # Initialize `y_pred` on the correct device
        y_pred = torch.zeros(s.shape, dtype=torch.float64, device=device)
        y_pred[0] = y0
        for i in range(len(y_pred) - 1):
            # Prepare input tensor and ensure it's on the correct device
            input = torch.clone(y_pred[i]).unsqueeze(0).to(device)
            dy = model(input)
            ds = s[i + 1] - s[i]
            y_pred[i + 1] = y_pred[i] + dy * ds
    else:
        # Ensure `x` is a torch.Tensor and on the correct device
        if isinstance(x, torch.Tensor):
            x = x.to(device)
        else:
            x = torch.tensor(x, dtype=torch.float64, device=device)

        # Initialize `y_pred` on the correct device
        y_pred = torch.zeros(s.shape, dtype=torch.float64, device=device)
        y_pred[0] = y0
        for i in range(len(y_pred) - 1):
            # Prepare input tensor and ensure it's on the correct device
            input = torch.tensor([x[i], y_pred[i]], dtype=torch.float64, device=device)
            dy = model(input)
            ds = s[i + 1] - s[i]
            y_pred[i + 1] = y_pred[i] + dy * ds

    return y_pred

def initialization(model):
    import torch.nn as nn
    import torch.nn.init as init
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
            init.zeros_(layer.bias)

    model = model.to(torch.float64)
    return model

import pandas as pd
import os

def save_ab_dict_to_xlsx(ab_dict, filename, sheetname):
    # 确保 sheetname 合法
    valid_sheets = {"ABETA", "TAU", "N", "C"}
    if sheetname not in valid_sheets:
        raise ValueError(f"无效的 sheetname: {sheetname}，必须是 {valid_sheets} 之一")

    # 转换字典为 DataFrame
    df = pd.DataFrame.from_dict(ab_dict, orient='index', columns=['a', 'b'])
    df.reset_index(inplace=True)  # 将 key 列从索引变为普通列
    df.rename(columns={'index': 'key'}, inplace=True)  # 重命名列名

    # 检查 Excel 文件是否存在
    if os.path.exists(filename):
        # 读取已有的 Excel 文件，保留其他 sheet
        with pd.ExcelWriter(filename, mode='a', engine='openpyxl') as writer:
            # 删除当前 sheet
            writer.book.remove(writer.book[sheetname]) if sheetname in writer.book.sheetnames else None
            # 重新写入该 sheet
            df.to_excel(writer, sheet_name=sheetname, index=False)
    else:
        # 文件不存在时，直接写入
        with pd.ExcelWriter(filename, mode='w', engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name=sheetname, index=False)

    print(f"ab_dict 已保存到 {filename} 的 {sheetname} sheet 中（已清空旧数据）")

def load_ab_dict(filename, sheetname):
    df = pd.read_excel(filename, sheet_name=sheetname)
    ab_dict = {}
    for _, row in df.iterrows():  # iterrows() 确保 row 是一行数据
        key = row['key']
        a = row['a']
        b = row['b']
        ab_dict[key] = [a, b]

    return ab_dict