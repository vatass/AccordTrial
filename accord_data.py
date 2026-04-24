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
data = pd.read_csv('/cbica/home/harmang/harmonization_evaluation/istaging_3_0.csv')

data = data[data['Study'] == 'ACCORD']

print('Subjects', data['PTID'].nunique())
print(f'Loaded: {data.shape}')

# ---------------------------------------------------------------------------
# 2. Merge SPARE_BA by MRID
# ---------------------------------------------------------------------------
spare_ba = pd.read_csv('SPARE_BA_istaging_3_0_all.csv')
if 'SPARE_BA' in data.columns:
    data = data.drop(columns=['SPARE_BA'])
data = data.merge(spare_ba[['MRID', 'SPARE_BA']], on='MRID', how='left')
n_matched = data['SPARE_BA'].notna().sum()
print(f'After SPARE_BA merge: {n_matched}/{len(data)} rows have SPARE_BA ({100*n_matched/len(data):.1f}%)')

# ---------------------------------------------------------------------------
# 3. Extract date from MRID  (format: <ID>-YYYYMMDD, e.g. 611H90402-20080612)
# ---------------------------------------------------------------------------
data['Date'] = pd.to_datetime(
    data['MRID'].str.split('-').str[-1], format='%Y%m%d', errors='coerce'
)
n_parsed = data['Date'].notna().sum()
print(f'Date parsed from MRID: {n_parsed}/{len(data)} rows')

# ---------------------------------------------------------------------------
# 4. Basic cleanup
# ---------------------------------------------------------------------------
data = data.drop_duplicates(subset=['PTID', 'Date'], keep='first')

# Drop rows missing all MUSE volume ROIs
muse_volume_cols = [c for c in data.columns if c.startswith('DLMUSE_')]
if muse_volume_cols:
    data = data.dropna(axis=0, subset=muse_volume_cols)
print(f'After MUSE NaN removal: {data["PTID"].nunique()} subjects')

# ---------------------------------------------------------------------------
# 5. Sort and compute Time (months from first acquisition per subject)
# ---------------------------------------------------------------------------
data = data.sort_values(by=['PTID', 'Date'])

# Delta_Baseline: days since each subject's first scan
data['Delta_Baseline'] = data.groupby('PTID')['Date'].transform(
    lambda x: (x - x.iloc[0]).dt.days
)

# Time in months (ceiling division, matching longitudinal_data.py)
data['Time'] = np.ceil(data['Delta_Baseline'] / 30).astype(int)

# Remove duplicate Time entries per subject (keep first occurrence)
data = data.groupby(['PTID', 'Time'], as_index=False).agg(lambda x: x.iloc[0])
print(f'Subjects after time deduplication: {data["PTID"].nunique()}')
print(f'Total acquisitions: {data.shape[0]}')

# ---------------------------------------------------------------------------
# 6. Compute BAG = SPARE_BA - Age  (before normalization)
# ---------------------------------------------------------------------------
data['BAG'] = data['SPARE_BA'] - data['Age']
print(f'BAG — mean: {data["BAG"].mean():.2f}, std: {data["BAG"].std():.2f}')

# ---------------------------------------------------------------------------
# 7. Normalize MUSE volumes using pre-computed training stats
# ---------------------------------------------------------------------------
hmuse_stats_path = os.path.join(data_dir, '145_MUSE_allstudies_mean_std.pkl')
print(f'Loading H_MUSE normalization stats from: {hmuse_stats_path}')
with open(hmuse_stats_path, 'rb') as f:
    hmuse_stats = pickle.load(f)

mean_hmuse = hmuse_stats['mean']
std_hmuse  = hmuse_stats['std']

hmuse_df = data.filter(regex=r'^DLMUSE_')
for i, col in enumerate(hmuse_df.columns):
    data[col] = (data[col] - mean_hmuse[i]) / std_hmuse[i]

print(f'MUSE volumes normalized ({len(hmuse_df.columns)} ROIs)')

# ---------------------------------------------------------------------------
# 8. Normalize Age and BAG using pre-computed training stats
# ---------------------------------------------------------------------------
norm_stats_path = os.path.join(data_dir, 'normalization_stats.pkl')
print(f'Loading Age/BAG normalization stats from: {norm_stats_path}')
with open(norm_stats_path, 'rb') as f:
    norm_stats = pickle.load(f)

mean_age = norm_stats['Age']['mean']
std_age  = norm_stats['Age']['std']
mean_bag = norm_stats['BAG']['mean']
std_bag  = norm_stats['BAG']['std']

data['Age'] = (data['Age'] - mean_age) / std_age
data['BAG'] = (data['BAG'] - mean_bag) / std_bag

print(f'Age normalized  — training mean={mean_age:.2f}, std={std_age:.2f}')
print(f'BAG normalized  — training mean={mean_bag:.2f}, std={std_bag:.2f}')

# ---------------------------------------------------------------------------
# 9. Encode categorical variables (matching training pipeline)
# ---------------------------------------------------------------------------
if data['Sex'].dtype == object:
    data['Sex'].replace(['M', 'F'], [0, 1], inplace=True)

if 'Education_Years' in data.columns and data['Education_Years'].dtype != int:
    data['Education_Years'] = (data['Education_Years'] > 16).astype(int)

# ---------------------------------------------------------------------------
# 10. Select exactly the features used during training
# ---------------------------------------------------------------------------
features_pkl_path = os.path.join(data_dir, 'features_bag.pkl')
if not os.path.exists(features_pkl_path):
    raise FileNotFoundError(
        f'{features_pkl_path} not found. Run longitudinal_data.py first.')

with open(features_pkl_path, 'rb') as f:
    train_features = pickle.load(f)

model_features = [c for c in train_features if c not in ('PTID', 'Delta_Baseline')]

missing = [c for c in model_features if c not in data.columns]
if missing:
    raise ValueError(f'ACCORD data is missing columns required by the model: {missing}')

os.makedirs(data_dir, exist_ok=True)
data['PTID'] = data['PTID'].astype(str)
output_cols = ['PTID'] + model_features
data = data[output_cols]

# Drop any rows with NaN in model features (e.g. missing SPARE_BA → NaN BAG)
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

