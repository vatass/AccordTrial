'''
ACCORD - CN Digital Twin - DKGP
'''

import os
import sys
import numpy as np
import pandas as pd
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

# Output directory
os.makedirs('data', exist_ok=True)
data_dir = 'data/'

# ---------------------------------------------------------------------------
# 1. Load Data
# ---------------------------------------------------------------------------
data = pd.read_pickle('/cbica/projects/ISTAGING/Pipelines/ISTAGING_Data_Consolidation_2020/v2.0/istaging.pkl.gz')

print(f'Loaded: {data.shape}')

# ---------------------------------------------------------------------------
# 2. Basic Filters
# ---------------------------------------------------------------------------
print('Removing BLSA 1.5T data...')
data = data[data['SITE'] != 'BLSA-1.5T']

data = data.drop_duplicates(subset=['PTID', 'Visit_Code'], keep='first')
data = data[data['Visit_Code'] != 'ADNI Screening']
data = data[data['Visit_Code'] != 'ADNIGO Screening MRI']

# Forward-fill missing diagnosis
data['Diagnosis_nearest_2.0'] = data['Diagnosis_nearest_2.0'].fillna(method='ffill')

# Prefix PTID for AIBL and PENN to avoid collisions
data.loc[data['Study'] == 'AIBL', 'PTID'] = 'aibl' + data.loc[data['Study'] == 'AIBL', 'PTID'].astype(str)
data.loc[data['Study'] == 'PENN', 'PTID'] = 'penn' + data.loc[data['Study'] == 'PENN', 'PTID'].astype(str)

data['Date'] = data['Date'].astype('datetime64[ns]')

# Drop rows missing all H_MUSE ROIs
hmuse = list(data.filter(regex='H_MUSE*'))
data = data.dropna(axis=0, subset=hmuse)
print(f'After H_MUSE NaN removal: {data["PTID"].nunique()} subjects')

# ---------------------------------------------------------------------------
# 3. Map Diagnosis
# ---------------------------------------------------------------------------
unique_diagnosis = list(data['Diagnosis_nearest_2.0'].unique())
dx_mapping = pd.read_csv('../LongGPClustering/DX_Mapping.csv')

old_diagnosis, new_diagnosis = [], []
for u in unique_diagnosis:
    old_diagnosis.append(u)
    indx = dx_mapping[dx_mapping['Diagnosis'] == u].index.values
    new_diagnosis.append(dx_mapping['Class'].iloc[indx[0]] if len(indx) > 0 else u)

data['Diagnosis_nearest_2.0'].replace(old_diagnosis, new_diagnosis, inplace=True)

# Remove non-AD-spectrum diagnoses
data = data[~data['Diagnosis_nearest_2.0'].isin(
    ['Vascular Dementia', 'other', 'FTD', '', 'PD', 'Lewy Body Dementia', 'Hydrocephalus', 'PCA', 'TBI']
)]

if data['Diagnosis_nearest_2.0'].isna().sum():
    data.loc[data['Diagnosis_nearest_2.0'].isna(), 'Diagnosis_nearest_2.0'] = 'unk'

data['Diagnosis_nearest_2.0'].replace(
    ['CN', 'MCI', 'AD', 'unk', 'other', 'early MCI', 'dementia'],
    [0,    1,     2,    -1,    -1,      1,            2], inplace=True
)

# ---------------------------------------------------------------------------
# 4. Keep only subjects that are CN (0) at all timepoints
# ---------------------------------------------------------------------------
cn_mask = data.groupby('PTID')['Diagnosis_nearest_2.0'].apply(lambda x: (x == 0).all())
data = data[data['PTID'].isin(cn_mask[cn_mask].index)]
print(f'CN-only subjects: {data["PTID"].nunique()}')

# Encode comorbidities
data['Hypertension'].replace(['Hypertension negative/absent', 'Hypertension positive/present'], [0, 1], inplace=True)
data['Hyperlipidemia'].replace(['Hyperlipidemia absent', 'Hyperlipidemia recent/active'], [0, 1], inplace=True)
data['Diabetes'].replace(['Diabetes negative/absent', 'Diabetes positive/present'], [0, 1], inplace=True)

# ---------------------------------------------------------------------------
# 5. Keep subjects with >1 acquisition and report per-study counts
# ---------------------------------------------------------------------------
data = data.groupby('PTID').filter(lambda x: x.shape[0] > 1)

print('\n=== Studies with Multiple Acquisitions ===')
studies_with_multiple = []
for study in sorted(data['Study'].unique()):
    study_data = data[data['Study'] == study]
    total = study_data['PTID'].nunique()
    n_multi = study_data.groupby('PTID').filter(lambda x: x.shape[0] > 1)['PTID'].nunique()
    if n_multi > 0:
        studies_with_multiple.append(study)
    print(f'  {study}: {n_multi}/{total} subjects with multiple acquisitions ({100 * n_multi / total:.1f}%)')
print(f'Total studies with multiple acquisitions: {len(studies_with_multiple)}')

data = data[data['Study'].isin(studies_with_multiple)]
print(f'Total subjects after study filter: {data["PTID"].nunique()}')

# ---------------------------------------------------------------------------
# 6. Fix Delta Baseline (first acquisition = 0)
# ---------------------------------------------------------------------------
def delta_baseline_fix(data):
    for pt in data['PTID'].unique():
        pt_indices = data[data['PTID'] == pt].index
        base = data.loc[pt_indices[0], 'Delta_Baseline']
        if base != 0:
            data.loc[pt_indices, 'Delta_Baseline'] -= base
    return data

data = delta_baseline_fix(data)

for pt in data['PTID'].unique():
    if data[data['PTID'] == pt].iloc[0]['Delta_Baseline'] != 0.0:
        print(f'Warning: {pt} has non-zero Delta_Baseline at baseline')

# Time in months (ceiling division)
data['Time'] = np.ceil(data['Delta_Baseline'] / 30).astype(int)

# Remove duplicate Time entries per subject
data = data.groupby(['PTID', 'Time']).agg(lambda x: x.iloc[0]).reset_index()
print(f'Subjects after time deduplication: {data["PTID"].nunique()}')

data_unnorm = data.copy()

# ---------------------------------------------------------------------------
# 7. Z-score MUSE ROIs
# ---------------------------------------------------------------------------
subjects_df_hmuse = data.filter(regex='H_MUSE*')
mean_hmuse = subjects_df_hmuse.mean(axis=0).tolist()
std_hmuse = subjects_df_hmuse.std(axis=0).tolist()

with open(data_dir + '145_harmonized_allstudies_mean_std_hmuse.pkl', 'wb') as f:
    pickle.dump({'mean': mean_hmuse, 'std': std_hmuse}, f)

for i, c in enumerate(subjects_df_hmuse.columns):
    data[c] = (subjects_df_hmuse[c] - mean_hmuse[i]) / std_hmuse[i]

# ---------------------------------------------------------------------------
# 8. Baseline columns
# ---------------------------------------------------------------------------
data['Baseline_Age'] = data.groupby('PTID')['Age'].transform('min')

for col in [c for c in data.columns if c.startswith('H_MUSE_')]:
    data['Baseline_' + col] = data.groupby('PTID')[col].transform('first')

for col in ['SPARE_AD', 'SPARE_BA', 'Diagnosis_nearest_2.0']:
    data['Baseline_' + col] = data.groupby('PTID')[col].transform('first')

# Keep only non-negative timepoints
data = data[data['Time'] >= 0]

# ---------------------------------------------------------------------------
# 9. BAG = SPARE_BA - Age  (computed before normalization)
# ---------------------------------------------------------------------------
data['BAG'] = data['SPARE_BA'] - data['Age']
print(f'BAG — mean: {data["BAG"].mean():.2f}, std: {data["BAG"].std():.2f}')

# ---------------------------------------------------------------------------
# 10. Normalize / encode clinical variables
# ---------------------------------------------------------------------------
mean_age, std_age = data['Age'].mean(), data['Age'].std()
data['Age'] = (data['Age'] - mean_age) / std_age

data['Education_Years'] = (data['Education_Years'] > 16).astype(int)
data['Sex'].replace(['M', 'F'], [0, 1], inplace=True)

mean_spareba, std_spareba = data['SPARE_BA'].mean(), data['SPARE_BA'].std()
data['SPARE_BA'] = (data['SPARE_BA'] - mean_spareba) / std_spareba

clinical_features = ['Sex', 'PTID', 'Delta_Baseline', 'Time']
for cf in clinical_features:
    data[cf] = data[cf].fillna(-1)

# ---------------------------------------------------------------------------
# 11. Save CSV (BAG biomarker)
# ---------------------------------------------------------------------------
all_subjects = list(data['PTID'].unique())
print(f'Total subjects: {len(all_subjects)}')

data['PTID'] = data['PTID'].astype(str)
data.to_csv(data_dir + 'longitudinal_covariates_bag_allstudies.csv', index=False)
print(f'Saved: {data_dir}longitudinal_covariates_bag_allstudies.csv')

# ---------------------------------------------------------------------------
# 12. Save features pickle
# ---------------------------------------------------------------------------
features = [name for name in data.columns if name.startswith('H_MUSE_Volume') and int(name[14:]) < 300]
features.extend(clinical_features)

with open(data_dir + 'features_bag.pkl', 'wb') as f:
    pickle.dump(features, f)

target = [name for name in data.columns if name.startswith('H_MUSE_Volume') and int(name[14:]) < 300]

# ---------------------------------------------------------------------------
# 13. 5-Fold Cross Validation
# ---------------------------------------------------------------------------
print(f'\nCreating 5-fold splits for {len(all_subjects)} subjects...')
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for i, (train_index, test_index) in enumerate(kf.split(all_subjects)):
    train_ids = [all_subjects[t] for t in train_index]
    test_ids  = [all_subjects[t] for t in test_index]
    assert not (set(train_ids) & set(test_ids)), f'Data leak in fold {i}!'
    print(f'  Fold {i}: {len(train_ids)} train / {len(test_ids)} test')
    with open(data_dir + f'train_subject_bag_allstudies_ids_hmuse{i}.pkl', 'wb') as f:
        pickle.dump(train_ids, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(data_dir + f'test_subject_bag_allstudies_ids_hmuse{i}.pkl', 'wb') as f:
        pickle.dump(test_ids, f, protocol=pickle.HIGHEST_PROTOCOL)

print('Done.')
