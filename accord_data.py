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
data = pd.read_csv('ACCORD_MARCH_enriched.csv')


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

# Date.x is stored as integer YYYYMMDD (e.g. 20040121) — parse explicitly
data['Date.x'] = pd.to_datetime(data['Date.x'].astype(str), format='%Y%m%d')

print(f'SPARE_BA available: {data["SPARE_BA"].notna().sum()}/{len(data)} rows')
print(f'SPARE_AD available: {data["SPARE_AD"].notna().sum()}/{len(data)} rows')

# Drop rows missing all MUSE volume ROIs
muse_volume_cols = [c for c in data.columns if c.startswith('MUSE_Volume_')]
if muse_volume_cols:
    data = data.dropna(axis=0, subset=muse_volume_cols)
print(f'After MUSE NaN removal: {data["PTID.x"].nunique()} subjects')

# ---------------------------------------------------------------------------
# 3. Sort and compute Time (months from first acquisition per subject)
# ---------------------------------------------------------------------------
data = data.sort_values(by=['PTID.x', 'Date.x'])

# Delta_Baseline: days since each subject's first scan
data['Delta_Baseline'] = data.groupby('PTID.x')['Date.x'].transform(lambda x: (x - x.iloc[0]).dt.days)

# Time in months (ceiling division, matching longitudinal_data.py)
data['Time'] = np.ceil(data['Delta_Baseline'] / 30).astype(int)

# Remove duplicate Time entries per subject (keep first occurrence)
data = data.groupby(['PTID.x', 'Time'], as_index=False).agg(lambda x: x.iloc[0])
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
hmuse_stats_path = os.path.join(data_dir, '145_MUSE_allstudies_mean_std.pkl')
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
# Rename .x-suffixed columns that originated from the ACCORD merge
rename_map_cols = {'PTID.x': 'PTID', 'Sex.x': 'Sex', 'Age.x': 'Age'}
data = data.rename(columns={k: v for k, v in rename_map_cols.items() if k in data.columns})

if data['Sex'].dtype == object:
    data['Sex'].replace(['M', 'F'], [0, 1], inplace=True)

if 'Education_Years' in data.columns and data['Education_Years'].dtype != int:
    data['Education_Years'] = (data['Education_Years'] > 16).astype(int)

# ---------------------------------------------------------------------------
# 8. Select exactly the features used during training
#    Load features_bag.pkl (saved by longitudinal_data.py) so the column
#    set and order are guaranteed to match the trained model.
# ---------------------------------------------------------------------------
features_pkl_path = os.path.join(data_dir, 'features_bag.pkl')
if not os.path.exists(features_pkl_path):
    raise FileNotFoundError(
        f'{features_pkl_path} not found. Run longitudinal_data.py first.')

with open(features_pkl_path, 'rb') as f:
    train_features = pickle.load(f)

# train_features = [MUSE_Volume_* <300, Sex, BAG, PTID, Delta_Baseline, Time]
# Model input = train_features minus PTID and Delta_Baseline (Time kept as
# last feature; the inference script overwrites it per future timepoint).
model_features = [c for c in train_features if c not in ('PTID', 'Delta_Baseline')]
# model_features = [MUSE_Volume_* <300, Sex, BAG, Time]  (148 - 2 = 146... + Time = 148 total with PTID excluded)

missing = [c for c in model_features if c not in data.columns]
if missing:
    raise ValueError(f'ACCORD data is missing columns required by the model: {missing}')

os.makedirs(data_dir, exist_ok=True)
data['PTID'] = data['PTID'].astype(str)
output_cols = ['PTID'] + model_features
data = data[output_cols]

# Drop any rows with NaN in model features (e.g. missing SPARE_BA → NaN BAG)
# A single NaN input causes the GP kernel matrix to produce NaN for all predictions
before = data['PTID'].nunique()
data = data.dropna(subset=model_features)
after = data['PTID'].nunique()
if before != after:
    print(f'Dropped {before - after} subjects with NaN in model features '
          f'(e.g. missing SPARE_BA/Sex)')

output_path = os.path.join(data_dir, 'accord_data_bag_processed.csv')
data.to_csv(output_path, index=False)
print(f'Saved processed ACCORD data: {output_path}')
print(f'Final shape: {data.shape}  ({data["PTID"].nunique()} subjects)')
print(f'Feature columns (excl. PTID): {len(model_features)}  '
      f'— matches training feature count')
