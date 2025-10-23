import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

# 1. load data
# read rawdata.xlsx
file_path = 'rawdata.xlsx'
adni_org_df = pd.read_excel(file_path, sheet_name='ADNI Org.')
csf_biomarker_df = pd.read_excel(file_path, sheet_name='CSF Biomarker')

# initialize an empty dataframe
df = pd.DataFrame(columns=['RID', 'EXAMDATE', 'AGE', 'ABETA', 'TAU', 'N', 'C'])

# processing 'ADNI Org.' sheet to get RID, EXAMDATE, AGE, C, N (N is Hippocampus)
for index, row in adni_org_df.iterrows():
    rid = row['RID']
    
    # check if there is this RID in 'CSF Biomarker'
    if rid in csf_biomarker_df['RID'].values:
        # load 'EXAMDATE', 'AGE', 'C', 'Hippocampus' in 'ADNI Org.'
        examdate = row['EXAMDATE']
        age = row['AGE']
        c = row['C']
        n = row['Hippocampus']  # 读取Hippocampus列，但在程序中记为N
        
        # combining into result_df
        new_row = pd.DataFrame({
            'RID': [rid],
            'EXAMDATE': [examdate],
            'AGE': [age],
            'C': [c],
            'N': [n],
            'ABETA': [None],
            'TAU': [None],
        })
        df = pd.concat([df, new_row], ignore_index=True)

# processing 'CSF Biomarker'
for index, row in csf_biomarker_df.iterrows():
    rid = row['RID']
    drwdte = row['DRWDTE']
    
    # check if there is this RID in 'ADNI Org.'
    if rid in adni_org_df['RID'].values:
        # check if there  is same RID and DRWDTE in result_df
        match = df[(df['RID'] == rid) & (df['EXAMDATE'] == drwdte)]
        
        if not match.empty:
            # if true，upload 'ABETA' and 'TAU'
            df.loc[match.index, 'ABETA'] = row['ABETA']
            df.loc[match.index, 'TAU'] = row['TAU']
        else:
            # if false, create a new row
            new_row = pd.DataFrame({
                'RID': [rid],
                'EXAMDATE': [drwdte],
                'AGE': [None],
                'C': [None],
                'N': [None],
                'ABETA': [row['ABETA']],
                'TAU': [row['TAU']]
            })
            df = pd.concat([df, new_row], ignore_index=True)

# Sorting
df = df[['RID', 'EXAMDATE', 'AGE', 'ABETA', 'TAU', 'N', 'C']]

# delete rows whose ABETA, TAU, C, N are all empty or 0
condition = (df[['ABETA', 'TAU', 'C', 'N']].isna() | (df[['ABETA', 'TAU', 'C', 'N']] == 0)).all(axis=1)
df = df[~condition]

# 2.  recalculate the age for each RID, as it's the age at baseline in the document
grouped = df.groupby('RID')
updated_rows = []

for rid, group in grouped:
    group = group.sort_values(by='EXAMDATE')
    
    # get the first EXAMDATE and AGE for the group
    first_age = group['AGE'].iloc[0]
    
    # calculating the AGE for the rest of the rows
    for i, row in group.iterrows():
        if i == group.index[0]:
            # The first line remains the original AGE
            updated_rows.append(row)
        else:
            # calculate the new AGE based on the first EXAMDATE and AGE, keeping one decimal place
            date_diff = (row['EXAMDATE'] - group['EXAMDATE'].iloc[0]).days / 365
            new_age = round(first_age + date_diff, 1)
            row['AGE'] = new_age
            updated_rows.append(row)

# reassembly
df = pd.DataFrame(updated_rows)

# delete the rows whose EXAMDATE is empty
df = df.replace(0, np.nan)
df = df.dropna(subset=['EXAMDATE'])

# 3. combining the rows with same RID and AGE
grouped = df.groupby(['RID', 'AGE'])
# initialize an empty list to store the processed rows
merged_rows = []

for (rid, age), group in grouped:
    group = group.sort_values(by='AGE')
    merged_row = {
        'RID': rid,
        'AGE': group['AGE'].iloc[0],
        'ABETA': np.nan,
        'TAU': np.nan,
        'C': np.nan,
        'N': np.nan,
    }
    
    for _, row in group.iterrows():
        if not pd.isna(row['ABETA']) or not pd.isna(row['TAU']):
            # if there is ABETA or TAU at this time point, fill them in
            merged_row['ABETA'] = row['ABETA']
            merged_row['TAU'] = row['TAU']
        if not pd.isna(row['C']) or not pd.isna(row['N']):
            # if there is C or N at this time point, fill them in
            merged_row['C'] = row['C']
            merged_row['N'] = row['N']

    merged_rows.append(merged_row)

df = pd.DataFrame(merged_rows)
df = df[['RID', 'AGE', 'ABETA', 'TAU', 'N', 'C']]
df.rename(columns={'RID': 'PID'}, inplace=True)

#4. 使用基于AGE的Sigmoid函数拟合填补缺失值
# 定义sigmoid函数
def sigmoid(x, a, b, c, d):
    """
    Sigmoid函数: y = a / (1 + exp(-b*(x-c))) + d
    a: 幅度
    b: 斜率
    c: 中心点
    d: 垂直偏移
    """
    return a / (1.0 + np.exp(-b * (x - c))) + d

# 对每个PID分组，根据AGE进行sigmoid拟合和预测
interpolated_rows = []

for pid, group in df.groupby('PID'):
    # 按年龄排序
    group = group.sort_values(by='AGE').reset_index(drop=True)
    
    # 对每个生物标记物列进行基于AGE的sigmoid拟合
    for col in ['ABETA', 'TAU', 'N', 'C']:
        # 找到该列的有效数据点
        valid_mask = group[col].notna()
        valid_ages = group.loc[valid_mask, 'AGE'].values
        valid_values = group.loc[valid_mask, col].values
        
        # 如果至少有4个有效数据点，进行sigmoid拟合
        if len(valid_ages) >= 4:
            try:
                # 设置初始猜测值
                p0 = [
                    np.max(valid_values) - np.min(valid_values),  # a: 幅度
                    0.1,                                            # b: 斜率
                    np.median(valid_ages),                          # c: 中心点
                    np.min(valid_values)                            # d: 垂直偏移
                ]
                
                # 使用scipy进行sigmoid拟合
                params, _ = curve_fit(sigmoid, valid_ages, valid_values, p0=p0, maxfev=5000)
                
                # 对所有AGE（包括缺失值）进行预测
                all_ages = group['AGE'].values
                predicted_values = sigmoid(all_ages, *params)
                
                # 用预测值填充NaN
                group.loc[group[col].isna(), col] = predicted_values[group[col].isna()]
                
            except (RuntimeError, ValueError, TypeError):
                # 如果sigmoid拟合失败，退回到线性拟合
                coeffs = np.polyfit(valid_ages, valid_values, deg=1)
                k, b = coeffs[0], coeffs[1]
                all_ages = group['AGE'].values
                predicted_values = k * all_ages + b
                group.loc[group[col].isna(), col] = predicted_values[group[col].isna()]
                
        elif len(valid_ages) == 3:
            # 如果有3个数据点，使用二次多项式拟合
            coeffs = np.polyfit(valid_ages, valid_values, deg=2)
            all_ages = group['AGE'].values
            predicted_values = np.polyval(coeffs, all_ages)
            group.loc[group[col].isna(), col] = predicted_values[group[col].isna()]
            
        elif len(valid_ages) == 2:
            # 如果只有2个数据点，使用线性拟合
            coeffs = np.polyfit(valid_ages, valid_values, deg=1)
            k, b = coeffs[0], coeffs[1]
            all_ages = group['AGE'].values
            predicted_values = k * all_ages + b
            group.loc[group[col].isna(), col] = predicted_values[group[col].isna()]
            
        elif len(valid_ages) == 1:
            # 如果只有1个数据点，用该值填充所有缺失值（常数外推）
            group[col].fillna(valid_values[0], inplace=True)
        # 如果该列完全没有数据，保持NaN
    
    interpolated_rows.append(group)

# 合并所有插值后的数据
df = pd.concat(interpolated_rows, ignore_index=True)

# 删除仍然有NaN的行（某些列完全没有数据的情况）
df = df.dropna()

# 可选：删除数据点过少的PID（如果需要的话）
rid_counts = df['PID'].value_counts()
# to_remove = rid_counts[rid_counts < 2].index  # 可以设置最小数据点数
# df = df[~df['PID'].isin(to_remove)]

#4.5 删除离群值（保留2.5%到97.5%分位数之间的数据）
print(f"删除离群值前的数据行数: {len(df)}")

# 对每个生物标记物列计算分位数
for col in ['ABETA', 'TAU', 'N', 'C']:
    q_low = df[col].quantile(0.05)   # 2.5%分位数
    q_high = df[col].quantile(0.95)  # 97.5%分位数
    
    # 删除该列超出范围的行
    df = df[(df[col] >= q_low) & (df[col] <= q_high)]
    print(f"  {col}: [{q_low:.4f}, {q_high:.4f}]")

print(f"删除离群值后的数据行数: {len(df)}")

name = df[['PID','AGE']]
df = df[['ABETA', 'TAU', 'N', 'C']]

#5. normalizing and saving
mean = df.mean()
std = df.std()
df = (df - mean)/std
df = pd.concat([name, df], axis=1)
print(df.head())   # 打印前几行数据
with pd.ExcelWriter('data.xlsx', mode='a', engine='openpyxl', if_sheet_exists='replace') as writer: # delete the old sheet to guarantee no residual old data 
    df.to_excel(writer, sheet_name='Sheet1', index=False)

# save quantile to quantile.npy for convenience
mean_df = np.array(mean.values)
std_df = np.array(std.values)
mean_std = np.vstack((mean_df.T, std_df.T))
np.save('mean_std.npy', mean_std)
print(mean_std)