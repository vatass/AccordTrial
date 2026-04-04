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
data = pd.read_pickle('/cbica/projects/ISTAGING/Pipelines/ISTAGING_Data_Consolidation_2020/v2.0/istaging.pkl.gz')

data = data[data['Study']=='ACCORD']

print('Subjects', data['PTID'].nunique())
print(f'Loaded: {data.shape}')

print(data['H_MUSE_Volume_4'].describe())

sys.exit(0)
# ---------------------------------------------------------------------------
# 2. Basic cleanup
# ---------------------------------------------------------------------------
data = data.drop_duplicates(subset=['PTID', 'Date'], keep='first')

data['Date'] = pd.to_datetime(data['Date'])

# Drop rows missing all H_MUSE ROIs
hmuse_cols = list(data.filter(regex='H_MUSE*').columns)
data = data.dropna(axis=0, subset=hmuse_cols)
print(f'After H_MUSE NaN removal: {data["PTID"].nunique()} subjects')

# ---------------------------------------------------------------------------
# 3. Keep only the first (earliest) acquisition per subject
# ---------------------------------------------------------------------------
data = data.sort_values(by=['PTID', 'Date'])
data = data.groupby('PTID', as_index=False).first()
print(f'After keeping first acquisition only: {data["PTID"].nunique()} subjects')

# ---------------------------------------------------------------------------
# 4. Compute BAG = SPARE_BA - Age  (before normalization)
# ---------------------------------------------------------------------------
data['BAG'] = data['SPARE_BA'] - data['Age']
print(f'BAG — mean: {data["BAG"].mean():.2f}, std: {data["BAG"].std():.2f}')

# ---------------------------------------------------------------------------
# 5. Normalize H_MUSE volumes using pre-computed training stats
# ---------------------------------------------------------------------------
hmuse_stats_path = os.path.join(data_dir, '145_harmonized_allstudies_mean_std_hmuse.pkl')
print(f'Loading H_MUSE normalization stats from: {hmuse_stats_path}')
with open(hmuse_stats_path, 'rb') as f:
    hmuse_stats = pickle.load(f)

mean_hmuse = hmuse_stats['mean']
std_hmuse = hmuse_stats['std']

# The stats were saved in the column order of training data; use the same cols
hmuse_df = data.filter(regex='H_MUSE*')
for i, col in enumerate(hmuse_df.columns):
    data[col] = (data[col] - mean_hmuse[i]) / std_hmuse[i]

print(f'H_MUSE volumes normalized ({len(hmuse_df.columns)} ROIs)')

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

data['Age'] = (data['Age'] - mean_age) / std_age
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

data['PTID'] = data['PTID'].astype(str)
output_path = os.path.join(data_dir, 'accord_data_processed.csv')
data.to_csv(output_path, index=False)
print(f'Saved processed ACCORD data: {output_path}')
print(f'Final shape: {data.shape}  ({data["PTID"].nunique()} subjects)')
