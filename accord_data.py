'''
ACCORD Trial Data Preprocessing
- Load ACCORD data
- Keep only the first (earliest) acquisition per subject
- Normalize H_MUSE volumes using pre-computed stats from training data
- Compute BAG = SPARE_BA - Age
- Normalize Age and BAG using pre-computed stats from normalization_stats.pkl
- Save processed data to data directory
'''

import os
import sys
import numpy as np
import pandas as pd
import pickle

data_dir = './data/'

# ---------------------------------------------------------------------------
# 1. Load ACCORD data
# ---------------------------------------------------------------------------
data = pd.read_csv('ACCORD_MARCH.csv')


print('Subjects', data['PTID.x'].nunique())
print(f'Loaded: {data.shape}')

print(data['X4'].describe())

rename_map = {f'X{i}': f'MUSE_Volume_{i}' for i in range(4, 208) if f'X{i}' in data.columns}
data = data.rename(columns=rename_map)
print(f'Renamed {len(rename_map)} columns (X4–X207 → MUSE_Volume_4–MUSE_Volume_207)')

# ---------------------------------------------------------------------------
# 2. Basic cleanup
# ---------------------------------------------------------------------------
data = data.drop_duplicates(subset=['PTID.x', 'Date.x'], keep='first')

data['Date.x'] = pd.to_datetime(data['Date.x'])

# Drop rows missing all H_MUSE ROIs
hmuse_cols = list(data.filter(regex='H_MUSE*').columns)
data = data.dropna(axis=0, subset=hmuse_cols)
print(f'After H_MUSE NaN removal: {data["PTID.x"].nunique()} subjects')

# ---------------------------------------------------------------------------
# 3. Sort and compute Time (months from first acquisition per subject)
# ---------------------------------------------------------------------------
data = data.sort_values(by=['PTID.x', 'Date.x'])

# Delta_Baseline: months since each subject's first scan
data['Delta_Baseline'] = data.groupby('PTID.x')['Date.x'].transform(lambda x: (x - x.iloc[0]).dt.days / 30)

# Time in months (ceiling, matching longitudinal_data.py)
data['Time'] = np.ceil(data['Delta_Baseline']).astype(int)

# Remove duplicate Time entries per subject (keep first occurrence)
data = data.groupby(['PTID.x', 'Time']).agg(lambda x: x.iloc[0]).reset_index()
print(f'Subjects after time deduplication: {data["PTID.x"].nunique()}')
print(f'Total acquisitions: {data.shape[0]}')

# ---------------------------------------------------------------------------
# 4. Compute BAG = SPARE_BA - Age  (before normalization)
# ---------------------------------------------------------------------------
data['BAG'] = data['SPARE_BA'] - data['Age.x']
print(f'BAG — mean: {data["BAG"].mean():.2f}, std: {data["BAG"].std():.2f}')

# ---------------------------------------------------------------------------
# 5. Normalize MUSE volumes using pre-computed training stats
# ---------------------------------------------------------------------------
hmuse_stats_path = os.path.join(data_dir, '145_MUSE_allstudies_mean_std_hmuse.pkl')
print(f'Loading H_MUSE normalization stats from: {hmuse_stats_path}')
with open(hmuse_stats_path, 'rb') as f:
    hmuse_stats = pickle.load(f)

mean_hmuse = hmuse_stats['mean']
std_hmuse = hmuse_stats['std']

# The stats were saved in the column order of training data; use the same cols
hmuse_df = data.filter(regex=r'^MUSE_')
for i, col in enumerate(hmuse_df.columns):
    data[col] = (data[col] - mean_hmuse[i]) / std_hmuse[i]

print(f'MUSE volumes normalized ({len(hmuse_df.columns)} ROIs)')

# ---------------------------------------------------------------------------
# 6. Normalize Age and BAG using pre-computed training stats
# ---------------------------------------------------------------------------
norm_stats_path = os.path.join(data_dir, 'normalization_stats.pkl')
print(f'Loading Age/BAG normalization stats from: {norm_stats_path}')
with open(norm_stats_path, 'rb') as f:
    norm_stats = pickle.load(f)

mean_age = norm_stats['Age']['mean']
std_age  = norm_stats['Age']['std']
mean_bag = norm_stats['BAG']['mean']
std_bag  = norm_stats['BAG']['std']

data['Age.x'] = (data['Age.x'] - mean_age) / std_age
data['BAG'] = (data['BAG'] - mean_bag) / std_bag

print(f'Age normalized  — training mean={mean_age:.2f}, std={std_age:.2f}')
print(f'BAG normalized  — training mean={mean_bag:.2f}, std={std_bag:.2f}')

# ---------------------------------------------------------------------------
# 7. Encode categorical variables (matching training pipeline)
# ---------------------------------------------------------------------------
if data['Sex'].dtype == object:
    data['Sex'].replace(['M', 'F'], [0, 1], inplace=True)

if data['Education_Years'].dtype != int:
    data['Education_Years'] = (data['Education_Years'] > 16).astype(int)

# ---------------------------------------------------------------------------
# 8. Save processed ACCORD data
# ---------------------------------------------------------------------------
os.makedirs(data_dir, exist_ok=True)


muse_cols = [c for c in data.columns if c.startswith('MUSE_Volume_')]
keep_cols = muse_cols + ['Sex.x', 'Age.x', 'BAG', 'PTID.x', 'Delta_Baseline', 'Time']
keep_cols = [c for c in keep_cols if c in data.columns]
data = data[keep_cols]
print(f'Kept {len(keep_cols)} columns: {len(muse_cols)} MUSE_Volume_* + meta columns')



data['PTID.x'] = data['PTID.x'].astype(str)
output_path = os.path.join(data_dir, 'accord_data_bag_processed.csv')
data.to_csv(output_path, index=False)
print(f'Saved processed ACCORD data: {output_path}')
print(f'Final shape: {data.shape}  ({data["PTID.x"].nunique()} subjects)')
