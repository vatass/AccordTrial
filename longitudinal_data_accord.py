'''
ACCORD - CN Digital Twin - DKGP
'''

import os
import sys
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection import StratifiedKFold, KFold


def create_baseline_temporal_dataset(subjects, dataframe, dataframeunnorm, target, features,hmuse):
    '''
    subjects: list of the subject ids
    dataframe: dataframe with all the data
    target: H_MUSE ROI features
    '''
    print('Target', target)
    cnt = 0
    num_samples = 0
    list_of_subjects, list_of_subject_ids = [], []
    data_x, data_y, data_xbase = [], [], []

    samples = {'PTID': [], 'X': [], 'Y': []}

    # remove the PTID from the features!
    features.remove('PTID.x')
    features.remove('Delta_Baseline')
    features.remove('Time')
    # hmuse = [i for i in features if i.startswith('H_MUSE')]

    # print('Features', features)
    clinical_features = [f for f in features if not f.startswith('MUSE')]
    # print('Clinical Features', clinical_features)

    # target = [t for t in target if t.startswith('H_')]
    print('Target', len(target))
    print('Input Features', features)

    for i, subject_id in enumerate(subjects):

        subject = dataframe[dataframe['PTID.x']==subject_id]
        subject_unnorm = dataframeunnorm[dataframeunnorm['PTID.x']==subject_id]

        # print(subject)
        for k in range(0, subject.shape[0]):
            samples['PTID'].append(subject_id)

            # print('Baseline Features',  features)

            x = subject[features].iloc[0].to_list()

            # print(x)

            delta = subject['Time'].iloc[k]
            # man_device = subject['MRI_Scanner_Model'].iloc[k]

            # print('Delta', delta)
            x.extend([delta])

            # print('Input', x)
            # print('Target', target)
            t = subject[target].iloc[k] #.to_list()

            samples['X'].append(x)
            samples['Y'].append(t.tolist())

            data_x.append(x)
            data_y.append(t)

        subject_data = list(zip(data_x, data_y))
        num_samples +=len(subject_data)
        list_of_subjects.append(subject_data)
        list_of_subject_ids.append(subject_id)

    assert len(samples['PTID']) == len(samples['X'])
    assert len(samples['X']) == len(samples['Y'])

    return samples, subject_data, num_samples, list_of_subjects, list_of_subject_ids, cnt

"""**Data Selection**
1. Read Data and remove all ADNI Screening and BLSA 1.5 T
2. Drop all NaN MUSE
3. Map the Diagnosis Column

"""

data_dir = './data/'

data = pd.read_csv('ACCORD_MARCH_enriched.csv')

rename_map = {f'X{i}': f'MUSE_Volume_{i}' for i in range(4, 208) if f'X{i}' in data.columns}
data = data.rename(columns=rename_map)
print(f'Renamed {len(rename_map)} columns (X4–X207 → MUSE_Volume_4–MUSE_Volume_207)')


data['Date.x'] = data['Date.x'].astype('datetime64[ns]')

# Sort by subject and visit date to ensure correct temporal ordering
data = data.sort_values(['PTID.x', 'Date.x']).reset_index(drop=True)

# Drop rows missing all H_MUSE ROIs
hmuse = list(data.filter(regex=r'^MUSE_'))
data = data.dropna(axis=0, subset=hmuse)
print(f'After MUSE NaN removal: {data["PTID.x"].nunique()} subjects')

# ---------------------------------------------------------------------------
# 6. Compute Delta_Baseline: days from each subject's first acquisition date
# ---------------------------------------------------------------------------

data['Delta_Baseline'] = data.groupby('PTID.x')['Date.x'].transform(
    lambda x: (x - x.min()).dt.days
)

def delta_baseline_fix(data):
    for pt in data['PTID.x'].unique():
        pt_indices = data[data['PTID.x'] == pt].index
        base = data.loc[pt_indices[0], 'Delta_Baseline']
        if base != 0:
            data.loc[pt_indices, 'Delta_Baseline'] -= base
    return data

data = delta_baseline_fix(data)

for pt in data['PTID.x'].unique():
    if data[data['PTID.x'] == pt].iloc[0]['Delta_Baseline'] != 0.0:
        print(f'Warning: {pt} has non-zero Delta_Baseline at baseline')

# Time in months (ceiling division)
data['Time'] = np.ceil(data['Delta_Baseline'] / 30).astype(int)

# Remove duplicate Time entries per subject
data = data.groupby(['PTID.x', 'Time']).agg(lambda x: x.iloc[0]).reset_index()
print(f'Subjects after time deduplication: {data["PTID.x"].nunique()}')

data_unnorm = data.copy()

# ---------------------------------------------------------------------------
# 7. Z-score MUSE ROIs using pre-computed combined normalization stats
# ---------------------------------------------------------------------------
subjects_df_hmuse = data.filter(regex=r'^MUSE_')

muse_pkl = data_dir + '145_MUSE_allstudies_mean_std.pkl'
if not os.path.exists(muse_pkl):
    raise FileNotFoundError(
        f'{muse_pkl} not found. Run compute_combined_normalization_stats.py first.')
print(f'Loading MUSE stats from: {muse_pkl}')
with open(muse_pkl, 'rb') as f:
    muse_stats = pickle.load(f)
mean_hmuse = muse_stats['mean']
std_hmuse  = muse_stats['std']

for i, c in enumerate(subjects_df_hmuse.columns):
    data[c] = (subjects_df_hmuse[c] - mean_hmuse[i]) / std_hmuse[i]


# Keep only non-negative timepoints
data = data[data['Time'] >= 0]

# ---------------------------------------------------------------------------
# 9. BAG = SPARE_BA - Age  (computed before normalization)
# ---------------------------------------------------------------------------
data['BAG'] = data['SPARE_BA'] - data['Age.x']
print(f'BAG — mean: {data["BAG"].mean():.2f}, std: {data["BAG"].std():.2f}')

# ---------------------------------------------------------------------------
# 10. Normalize clinical variables using pre-computed combined stats
# ---------------------------------------------------------------------------
norm_pkl = data_dir + 'normalization_stats.pkl'
if not os.path.exists(norm_pkl):
    raise FileNotFoundError(
        f'{norm_pkl} not found. Run compute_combined_normalization_stats.py first.')
print(f'Loading normalization stats from: {norm_pkl}')
with open(norm_pkl, 'rb') as f:
    normalization_stats = pickle.load(f)

mean_age     = normalization_stats['Age']['mean']
std_age      = normalization_stats['Age']['std']
mean_spareba = normalization_stats['SPARE_BA']['mean']
std_spareba  = normalization_stats['SPARE_BA']['std']
mean_bag     = normalization_stats['BAG']['mean']
std_bag      = normalization_stats['BAG']['std']

data['Age.x']      = (data['Age.x']      - mean_age)     / std_age
data['SPARE_BA'] = (data['SPARE_BA'] - mean_spareba) / std_spareba
data['BAG']      = (data['BAG']      - mean_bag)     / std_bag

data['Education_Years'] = (data['Education_Years'] > 16).astype(int)
data['Sex.x'].replace(['M', 'F'], [0, 1], inplace=True)

print(f'  Age:      mean={mean_age:.2f}, std={std_age:.2f}')
print(f'  SPARE_BA: mean={mean_spareba:.2f}, std={std_spareba:.2f}')
print(f'  BAG:      mean={mean_bag:.2f}, std={std_bag:.2f}')

clinical_features = ['Sex.x', 'Age.x', 'BAG', 'PTID.x', 'Delta_Baseline', 'Time']
for cf in clinical_features:
    data[cf] = data[cf].fillna(-1)

# ---------------------------------------------------------------------------
# 12. Save features pickle
# ---------------------------------------------------------------------------
features = [name for name in data.columns if name.startswith('MUSE_Volume') and int(name[12:]) < 300]
print(features)
features.extend(clinical_features)

print(len(features)) 

target = ['BAG']
all_subjects = list(data['PTID.x'].unique())

samples, subject_data, num_samples, list_of_subjects, list_of_subject_ids, cnt  = create_baseline_temporal_dataset(subjects=all_subjects, dataframe=data, dataframeunnorm=data_unnorm,  target=target, features=features, hmuse=hmuse)

samples_df = pd.DataFrame(data=samples)
samples_df.to_csv(data_dir + 'subjectsamples_bag_'+'accord'+'.csv')
