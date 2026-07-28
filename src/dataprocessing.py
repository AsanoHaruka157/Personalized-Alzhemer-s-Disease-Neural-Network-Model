"""
Data preprocessing pipeline for the Personalized Alzheimer's Disease
Neural Network Model.

This script processes raw ADNI clinical data (rawdata.xlsx) to produce:
  1. data.xlsx — z-score normalized biomarker values per patient visit
  2. mean_std.npy — normalization parameters for inverse mapping

Workflow:
  1. Merge 'ADNI Org.' (demographics, cognition, hippocampus) with
     'CSF Biomarker' (ABETA, TAU) sheets by patient ID and exam date.
  2. Recalculate age for follow-up visits relative to baseline.
  3. Merge visits on the same date (same age) into single rows.
  4. Filter out subjects with fewer than 2 valid measurements for any biomarker.
  5. Z-score normalize and save.
"""

import pandas as pd
import numpy as np

# ==============================================================================
# 1. Load and merge raw data sheets
# ==============================================================================
file_path = '../data/rawdata.xlsx'
adni_org_df = pd.read_excel(file_path, sheet_name='ADNI Org.')
csf_biomarker_df = pd.read_excel(file_path, sheet_name='CSF Biomarker')

df = pd.DataFrame(columns=['RID', 'EXAMDATE', 'AGE', 'ABETA', 'TAU', 'N', 'C'])

# -- Step 1a: Process 'ADNI Org.' sheet (RID, EXAMDATE, AGE, C, Hippocampus as N) --
for index, row in adni_org_df.iterrows():
    rid = row['RID']

    # Only keep patients who also have CSF biomarker data
    if rid not in csf_biomarker_df['RID'].values:
        continue

    new_row = pd.DataFrame({
        'RID': [rid],
        'EXAMDATE': [row['EXAMDATE']],
        'AGE': [row['AGE']],
        'C': [row['C']],
        'N': [row['Hippocampus']],  # Hippocampus volume labeled as 'N'
        'ABETA': [None],
        'TAU': [None],
    })
    df = pd.concat([df, new_row], ignore_index=True)

# -- Step 1b: Process 'CSF Biomarker' sheet (RID, DRWDTE, ABETA, TAU) --
for index, row in csf_biomarker_df.iterrows():
    rid = row['RID']
    drwdte = row['DRWDTE']

    if rid not in adni_org_df['RID'].values:
        continue

    # Try to match with existing row by RID and exam date
    match = df[(df['RID'] == rid) & (df['EXAMDATE'] == drwdte)]

    if not match.empty:
        # Fill in ABETA and TAU on the matching row
        df.loc[match.index, 'ABETA'] = row['ABETA']
        df.loc[match.index, 'TAU'] = row['TAU']
    else:
        # Create a new row with only CSF data
        new_row = pd.DataFrame({
            'RID': [rid],
            'EXAMDATE': [drwdte],
            'AGE': [None],
            'C': [None],
            'N': [None],
            'ABETA': [row['ABETA']],
            'TAU': [row['TAU']],
        })
        df = pd.concat([df, new_row], ignore_index=True)

# Reorder columns
df = df[['RID', 'EXAMDATE', 'AGE', 'ABETA', 'TAU', 'N', 'C']]

# Drop rows where all four biomarkers are NaN or zero
condition = (df[['ABETA', 'TAU', 'C', 'N']].isna()
             | (df[['ABETA', 'TAU', 'C', 'N']] == 0)).all(axis=1)
df = df[~condition]

# ==============================================================================
# 2. Recalculate age for follow-up visits (baseline age is in the raw data)
# ==============================================================================
grouped = df.groupby('RID')
updated_rows = []

for rid, group in grouped:
    group = group.sort_values(by='EXAMDATE')
    first_age = group['AGE'].iloc[0]

    for i, row in group.iterrows():
        if i == group.index[0]:
            # Baseline visit: keep original age
            updated_rows.append(row)
        else:
            # Follow-up visit: compute age from date difference
            date_diff = (row['EXAMDATE'] - group['EXAMDATE'].iloc[0]).days / 365
            new_age = round(first_age + date_diff, 1)
            row['AGE'] = new_age
            updated_rows.append(row)

df = pd.DataFrame(updated_rows)

# Drop rows with missing exam dates
df = df.replace(0, np.nan)
df = df.dropna(subset=['EXAMDATE'])

# ==============================================================================
# 3. Merge rows sharing the same RID and AGE into a single visit
# ==============================================================================
grouped = df.groupby(['RID', 'AGE'])
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
            merged_row['ABETA'] = row['ABETA']
            merged_row['TAU'] = row['TAU']
        if not pd.isna(row['C']) or not pd.isna(row['N']):
            merged_row['C'] = row['C']
            merged_row['N'] = row['N']

    merged_rows.append(merged_row)

df = pd.DataFrame(merged_rows)
df = df[['RID', 'AGE', 'ABETA', 'TAU', 'N', 'C']]
df.rename(columns={'RID': 'PID'}, inplace=True)

# ==============================================================================
# 4. Filter subjects with insufficient measurements
# ==============================================================================
print(f"Subjects before filtering: {df['PID'].nunique()}")
print(f"Rows before filtering: {len(df)}")

pids_to_keep = []
for pid, group in df.groupby('PID'):
    keep = True
    for col in ['ABETA', 'TAU', 'N', 'C']:
        if group[col].notna().sum() < 2:
            keep = False
            break
    if keep:
        pids_to_keep.append(pid)

df = df[df['PID'].isin(pids_to_keep)]

print(f"Subjects after filtering: {df['PID'].nunique()}")
print(f"Rows after filtering: {len(df)}")
print(f"Remaining NaN count: {df[['ABETA', 'TAU', 'N', 'C']].isna().sum().sum()}")

# ==============================================================================
# 5. Z-score normalize and save
# ==============================================================================
name = df[['PID', 'AGE']]
df_biomarkers = df[['ABETA', 'TAU', 'N', 'C']]

mean = df_biomarkers.mean()
std = df_biomarkers.std()
df_normalized = (df_biomarkers - mean) / std
df_normalized = pd.concat([name, df_normalized], axis=1)

print(df_normalized.head())

# Save normalized data to data.xlsx (overwrite Sheet1)
with pd.ExcelWriter('../data/data.xlsx', mode='a', engine='openpyxl',
                    if_sheet_exists='replace') as writer:
    df_normalized.to_excel(writer, sheet_name='Sheet1', index=False)

# Save normalization parameters for inverse mapping
mean_arr = np.array(mean.values)
std_arr = np.array(std.values)
mean_std = np.vstack((mean_arr.T, std_arr.T))
np.save('../data/mean_std.npy', mean_std)
print("Normalization parameters (mean, std):")
print(mean_std)
