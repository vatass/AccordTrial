'''
ACCORD - CN Digital Twin - DKGP
'''

import os
import sys
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import StratifiedKFold, KFold


def create_baseline_temporal_dataset(subjects, dataframe, dataframeunnorm, target, features,hmuse, genomic, followup, derivedroi,  visualize=False):
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
    covariates = {'PTID': [], 'Age': [], 'BaselineDiagnosis': [], 'BaselineAge': [], 'Sex': [] , 'APOE4_Alleles': [], 'Education_Years': [], 'Status': []}

    longitudinal_covariates = {'PTID': [], 'Time': [], 'Age': [],  'Diagnosis': [], 'Hypertension': [],
                               'Diabetes': [], 'DLICV': [], 'Study': [], 'Education_Years': [], 'Race': [], 'Sex': [], 'APOE4_Alleles': [], 'SPARE_BA': [], 'SPARE_AD': [], 'MRI_Scanner_Model': [], 
                               'CDR_Global': [], 'Tau_CSF': [], 'Abeta_CSF': [], 'PTau_CSF': [], 'MMSE_nearest_2.0': [] }


    if visualize:
        vdata = {'target': [], 'class': [], 'time': [], 'id': []}
        cnt = 0

    # remove the PTID from the features!
    features.remove('PTID')
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

        subject = dataframe[dataframe['PTID']==subject_id]
        subject_unnorm = dataframeunnorm[dataframeunnorm['PTID']==subject_id]

        first_diagnosis = subject['Diagnosis_nearest_2.0'].iloc[0]
        last_diagnosis = subject['Diagnosis_nearest_2.0'].iloc[-1]

        if first_diagnosis == 0 and last_diagnosis == 0:
            status = 'Non-Progressor'
        elif first_diagnosis == 0 and last_diagnosis != 0: 
            status = 'Progressor'
        else: 
            status = 'MCI/Demented Stable'

        # print(subject)
        for k in range(0, subject.shape[0]):
            samples['PTID'].append(subject_id)
            covariates['PTID'].append(subject_id)

            print('Baseline Features',  features)

            x = subject[features].iloc[0].to_list()

            # print(x)

            delta = subject['Time'].iloc[k]
            # man_device = subject['MRI_Scanner_Model'].iloc[k]
            diagnosis = subject['Diagnosis_nearest_2.0'].iloc[k]
            baseline_diagnosis = subject['Diagnosis_nearest_2.0'].iloc[0]
            baseline_age = subject_unnorm['Age'].iloc[0]
            age = subject_unnorm['Age'].iloc[k]
            dlicv = subject_unnorm['DLICV'].iloc[k]
            study = subject_unnorm['Study'].iloc[k]
            edu_years = subject['Education_Years'].iloc[k]
            race = subject_unnorm['Race'].iloc[k]
            sex = subject['Sex'].iloc[k]
            apoe4 = subject['APOE4_Alleles'].iloc[k]
            hypertension = subject_unnorm['Hypertension'].iloc[k]
            diabetes = subject_unnorm['Diabetes'].iloc[k]
            spba = subject_unnorm['SPARE_BA'].iloc[k]
            spad = subject_unnorm['SPARE_AD'].iloc[k]
            scanner = subject_unnorm['MRI_Scanner_Model'].iloc[k]
            cdr_global = subject_unnorm['CDR_Global'].iloc[k]
            tau_csf = subject_unnorm['Tau_CSF'].iloc[k]
            abeta_csf = subject_unnorm['Abeta_CSF'].iloc[k]
            ptau_csf = subject_unnorm['PTau_CSF'].iloc[k]
            mmse = subject_unnorm['MMSE_nearest_2.0'].iloc[k]

            # print('Delta', delta)
            x.extend([delta])

            # print('Input', x)
            # print('Target', target)
            t = subject[target].iloc[k] #.to_list()

            print('Target', t)
            # covariates['MRI_Scanner_Model'].append(man_device)
            covariates['Age'].append(age)
            covariates['BaselineDiagnosis'].append(baseline_diagnosis)
            covariates['BaselineAge'].append(baseline_age)
            covariates['Sex'].append(sex) 
            covariates['APOE4_Alleles'].append(apoe4)
            covariates['Education_Years'].append(edu_years) 
            covariates['Status'].append(status)
                                                 
            longitudinal_covariates['PTID'].append(subject_id)
            longitudinal_covariates['Time'].append(delta)
            longitudinal_covariates['Age'].append(age)
            longitudinal_covariates['Diagnosis'].append(diagnosis)
            longitudinal_covariates['DLICV'].append(dlicv)
            longitudinal_covariates['Study'].append(study)
            longitudinal_covariates['Education_Years'].append(edu_years)
            longitudinal_covariates['Race'].append(race)
            longitudinal_covariates['Sex'].append(sex)
            longitudinal_covariates['APOE4_Alleles'].append(apoe4)
            longitudinal_covariates['Hypertension'].append(hypertension)
            longitudinal_covariates['Diabetes'].append(diabetes)
            longitudinal_covariates['SPARE_BA'].append(spba)
            longitudinal_covariates['SPARE_AD'].append(spba)
            longitudinal_covariates['MRI_Scanner_Model'].append(scanner)
            longitudinal_covariates['CDR_Global'].append(cdr_global)
            longitudinal_covariates['Tau_CSF'].append(tau_csf)
            longitudinal_covariates['PTau_CSF'].append(ptau_csf)
            longitudinal_covariates['Abeta_CSF'].append(abeta_csf)
            longitudinal_covariates['MMSE_nearest_2.0'].append(mmse)

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

    return samples, subject_data, num_samples, list_of_subjects, list_of_subject_ids, cnt, covariates, longitudinal_covariates

"""**Data Selection**
1. Read Data and remove all ADNI Screening and BLSA 1.5 T
2. Drop all NaN MUSE
3. Map the Diagnosis Column

"""

data_dir = './data/'

data = pd.read_pickle('/cbica/projects/ISTAGING/Pipelines/ISTAGING_Data_Consolidation_2020/v2.0/istaging.pkl.gz')

print(f'Loaded: {data.shape}')

# ---------------------------------------------------------------------------
# 2. Basic Filters
# ---------------------------------------------------------------------------
print('Removing BLSA 1.5T data and BIOCARD...')
data = data[data['SITE'] != 'BLSA-1.5T']
data = data[data['Study'] != 'BIOCARD']

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
hmuse = list(data.filter(regex=r'^MUSE_'))
data = data.dropna(axis=0, subset=hmuse)
print(f'After MUSE NaN removal: {data["PTID"].nunique()} subjects')

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
subjects_df_hmuse = data.filter(regex=r'^MUSE_')
mean_hmuse = subjects_df_hmuse.mean(axis=0).tolist()
std_hmuse = subjects_df_hmuse.std(axis=0).tolist()

with open(data_dir + '145_MUSE_allstudies_mean_std_hmuse.pkl', 'wb') as f:
    pickle.dump({'mean': mean_hmuse, 'std': std_hmuse}, f)

for i, c in enumerate(subjects_df_hmuse.columns):
    data[c] = (subjects_df_hmuse[c] - mean_hmuse[i]) / std_hmuse[i]

# ---------------------------------------------------------------------------
# 8. Baseline columns
# ---------------------------------------------------------------------------
data['Baseline_Age'] = data.groupby('PTID')['Age'].transform('min')

for col in [c for c in data.columns if c.startswith('MUSE_')]:
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
# 10. Normalize / encode clinical variables  (save all stats for later use)
# ---------------------------------------------------------------------------
mean_age,     std_age     = data['Age'].mean(),     data['Age'].std()
mean_spareba, std_spareba = data['SPARE_BA'].mean(), data['SPARE_BA'].std()
mean_bag,     std_bag     = data['BAG'].mean(),      data['BAG'].std()

data['Age']     = (data['Age']     - mean_age)     / std_age
data['SPARE_BA'] = (data['SPARE_BA'] - mean_spareba) / std_spareba
data['BAG']     = (data['BAG']     - mean_bag)     / std_bag

data['Education_Years'] = (data['Education_Years'] > 16).astype(int)
data['Sex'].replace(['M', 'F'], [0, 1], inplace=True)

# Persist normalization statistics for downstream scripts
normalization_stats = {
    'Age':      {'mean': mean_age,     'std': std_age},
    'SPARE_BA': {'mean': mean_spareba, 'std': std_spareba},
    'BAG':      {'mean': mean_bag,     'std': std_bag},
}
with open(data_dir + 'normalization_stats.pkl', 'wb') as f:
    pickle.dump(normalization_stats, f)
print('Normalization stats saved.')
print(f'  Age:      mean={mean_age:.2f}, std={std_age:.2f}')
print(f'  SPARE_BA: mean={mean_spareba:.2f}, std={std_spareba:.2f}')
print(f'  BAG:      mean={mean_bag:.2f}, std={std_bag:.2f}')

clinical_features = ['Sex', 'BAG', 'PTID', 'Delta_Baseline', 'Time']
for cf in clinical_features:
    data[cf] = data[cf].fillna(-1)

# ---------------------------------------------------------------------------
# 11. Save CSV (BAG biomarker)
# ---------------------------------------------------------------------------
all_subjects = list(data['PTID'].unique())
print(f'Total subjects: {len(all_subjects)}')

data['PTID'] = data['PTID'].astype(str)
data.to_csv(data_dir + 'data_bag_allstudies.csv', index=False)
print(f'Saved: {data_dir}data_bag_allstudies.csv')

# ---------------------------------------------------------------------------
# 12. Save features pickle
# ---------------------------------------------------------------------------
features = [name for name in data.columns if name.startswith('MUSE_Volume') and int(name[12:]) < 300]
features.extend(clinical_features)

with open(data_dir + 'features_bag.pkl', 'wb') as f:
    pickle.dump(features, f)

target = ['BAG']

samples, subject_data, num_samples, list_of_subjects, list_of_subject_ids, cnt, covs, longitudinal_covariates = create_baseline_temporal_dataset(subjects=all_subjects, dataframe=data, dataframeunnorm=data_unnorm,  target=target, features=features, hmuse=hmuse,  genomic=0, followup=0, derivedroi='all', visualize=False)

# samples, subject_data, num_samples, list_of_subjects, list_of_subject_ids, cnt = create_n_acquisition_temporal_dataset(n=3, subjects=all_subjects, dataframe=data, dataframeunnorm=data_unnorm,  target=target, features=features, hmuse=hmuse,  genomic=0, followup=0, derivedroi='all', visualize=False)

samples_df = pd.DataFrame(data=samples)
longitudinal_covariates_df = pd.DataFrame(data=longitudinal_covariates)
longitudinal_covariates_df.to_csv(data_dir + 'longitudinal_covariates_bag_allstudies.csv', index=False)
samples_df.to_csv(data_dir + 'subjectsamples_bag_'+'allstudies'+'.csv')
sys.exit(0)

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
