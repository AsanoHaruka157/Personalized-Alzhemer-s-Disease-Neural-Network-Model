import pandas as pd
import numpy as np

# 1. load data
# read rawdata.xlsx
file_path = 'rawdata.xlsx'
adni_org_df = pd.read_excel(file_path, sheet_name='ADNI Org.')
csf_biomarker_df = pd.read_excel(file_path, sheet_name='CSF Biomarker')

# initialize an empty dataframe
df = pd.DataFrame(columns=['RID', 'EXAMDATE', 'AGE', 'ABETA', 'TAU', 'N', 'C'])

# processing 'ADNI Org.' sheet to get RID, EXAMDATE, AGE, C, N
for index, row in adni_org_df.iterrows():
    rid = row['RID']
    
    # check if there is this RID in 'CSF Biomarker'
    if rid in csf_biomarker_df['RID'].values:
        # load 'EXAMDATE', 'AGE', 'C', 'N' in 'ADNI Org.'
        examdate = row['EXAMDATE']
        age = row['AGE']
        c = row['C']
        n = row['N']
        
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

#4. delete nan and low-quality data
df = df.dropna()
rid_counts = df['PID'].value_counts()
to_remove = rid_counts[rid_counts < 2].index
df = df[~df['PID'].isin(to_remove)]

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