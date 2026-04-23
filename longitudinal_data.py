'''
ACCORD - CN Digital Twin - DKGP
'''

import os
import re
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

data = pd.read_csv('/cbica/home/harmang/harmonization_evaluation/istaging_3_0.csv')


print(f'Loaded: {data.shape}')
print(data['MRID'].head(10))
print(data['MRID'].tail(10))

print(data['MRID'].nunique())


# ---------------------------------------------------------------------------
# 2. Basic Filters
# ---------------------------------------------------------------------------
print('Removing BLSA 1.5T data and BIOCARD...')
data = data[data['SITE'] != 'BLSA-1.5T']
data = data[data['Study'] != 'BIOCARD']

# data = data.drop_duplicates(subset=['PTID', 'Visit_Code'], keep='first')
# data = data[data['Visit_Code'] != 'ADNI Screening']
# data = data[data['Visit_Code'] != 'ADNIGO Screening MRI']

# Forward-fill missing diagnosis
data['DX_AD'] = data['DX_AD'].fillna(method='ffill')

# Prefix PTID for AIBL and PENN to avoid collisions
data.loc[data['Study'] == 'AIBL', 'PTID'] = 'aibl' + data.loc[data['Study'] == 'AIBL', 'PTID'].astype(str)
data.loc[data['Study'] == 'PENN', 'PTID'] = 'penn' + data.loc[data['Study'] == 'PENN', 'PTID'].astype(str)



# Drop rows missing all H_MUSE ROIs
hmuse = list(data.filter(regex=r'^DMUSE_'))
data = data.dropna(axis=0, subset=hmuse)
print(f'After MUSE NaN removal: {data["PTID"].nunique()} subjects')

# ---------------------------------------------------------------------------
# 3. Map Diagnosis
# ---------------------------------------------------------------------------
unique_diagnosis = list(data['DX_AD'].unique())
dx_mapping = pd.read_csv('../LongGPClustering/DX_Mapping.csv')

old_diagnosis, new_diagnosis = [], []
for u in unique_diagnosis:
    old_diagnosis.append(u)
    indx = dx_mapping[dx_mapping['Diagnosis'] == u].index.values
    new_diagnosis.append(dx_mapping['Class'].iloc[indx[0]] if len(indx) > 0 else u)

data['DX_AD'].replace(old_diagnosis, new_diagnosis, inplace=True)

# Remove non-AD-spectrum diagnoses
data = data[~data['DX_AD'].isin(
    ['Vascular Dementia', 'other', 'FTD', '', 'PD', 'Lewy Body Dementia', 'Hydrocephalus', 'PCA', 'TBI']
)]

if data['DX_AD'].isna().sum():
    data.loc[data['DX_AD'].isna(), 'DX_AD'] = 'unk'

data['DX_AD'].replace(
    ['CN', 'MCI', 'AD', 'unk', 'other', 'early MCI', 'dementia'],
    [0,    1,     2,    -1,    -1,      1,            2], inplace=True
)

# ---------------------------------------------------------------------------
# 4. Keep only subjects that are CN (0) at all timepoints
# ---------------------------------------------------------------------------
cn_mask = data.groupby('PTID')['DX_AD'].apply(lambda x: (x == 0).all())
data = data[data['PTID'].isin(cn_mask[cn_mask].index)]
print(f'CN-only subjects: {data["PTID"].nunique()}')

# Encode comorbidities
# data['Hypertension'].replace(['Hypertension negative/absent', 'Hypertension positive/present'], [0, 1], inplace=True)
# data['Hyperlipidemia'].replace(['Hyperlipidemia absent', 'Hyperlipidemia recent/active'], [0, 1], inplace=True)
# data['Diabetes'].replace(['Diabetes negative/absent', 'Diabetes positive/present'], [0, 1], inplace=True)

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


# MRID formats differ across the 25 iSTAGING studies.  The naive
# split('-')[-1] only yields a valid YYYYMMDD for a handful of cohorts
# (ACCORD, ADNI_DOD, CARDIA, SPRINT, WHIMS, lookAHEAD).  All others
# need study-specific logic; some have no calendar date in the MRID at all.
print('\n=== MRID format samples per study ===')
for _study in sorted(data['Study'].unique()):
    _samples = data[data['Study'] == _study]['MRID'].dropna().head(3).tolist()
    print(f'  {_study}: {_samples}')


def extract_date_from_mrid(mrid: str, study: str) -> 'pd.Timestamp':
    """Return a Timestamp parsed from the MRID, or NaT when not possible."""
    try:
        if study in ('ACCORD', 'ADNI_DOD', 'CARDIA', 'SPRINT', 'WHIMS', 'lookAHEAD'):
            # Format: <ID>-YYYYMMDD
            return pd.to_datetime(mrid.split('-')[-1], format='%Y%m%d')
        elif study == 'ADNI':
            # Format: 002_S_0295_YYYY-MM-DD
            return pd.to_datetime(mrid.split('_')[-1], format='%Y-%m-%d')
        elif study in ('AIBL', 'BIOCARD', 'FITBIR', 'HANDLS', 'MESA', 'PENN-PMC'):
            # Format: <ID>_YYYYMMDD  or  PREFIX_<ID>_YYYYMMDD
            return pd.to_datetime(mrid.split('_')[-1], format='%Y%m%d')
        elif study == 'PENN-ADC':
            # Format: YYYYMMDD_<ID>
            return pd.to_datetime(mrid.split('_')[0], format='%Y%m%d')
        elif study == 'HABS':
            # Format: P_<ID>_YYYY-MM-DD_<SITE>_<N>
            m = re.search(r'(\d{4}-\d{2}-\d{2})', mrid)
            if m:
                return pd.to_datetime(m.group(1), format='%Y-%m-%d')
        # BLSA, GSP, HCP-Aging, HCP-YA, OASIS3, OASIS4,
        # PreventAD, SHIP, UKBIOBANK, WRAP — no calendar date in MRID
        return pd.NaT
    except (ValueError, IndexError):
        return pd.NaT


data['Date'] = [
    extract_date_from_mrid(mrid, study)
    for mrid, study in zip(data['MRID'], data['Study'])
]

print('\n=== Date extraction coverage per study ===')
for _study in sorted(data['Study'].unique()):
    _mask = data['Study'] == _study
    _total = _mask.sum()
    _parsed = data.loc[_mask, 'Date'].notna().sum()
    print(f'  {_study}: {_parsed}/{_total} dates parsed')
print()

print('\n=== MRID samples for studies with unparsed dates ===')
for _study in sorted(data['Study'].unique()):
    _mask = data['Study'] == _study
    _parsed = data.loc[_mask, 'Date'].notna().sum()
    _total = _mask.sum()
    if _parsed < _total:
        _unparsed_mrids = data.loc[_mask & data['Date'].isna(), 'MRID'].dropna().head(5).tolist()
        print(f'  {_study} ({_total - _parsed} unparsed): {_unparsed_mrids}')

data['Delta_Baseline'] = data.groupby('PTID')['Date'].transform(lambda x: (x - x.iloc[0]).dt.days)

# Keep only specific columns
# 1. Identify all columns that start with 'DLMUSE'
dlmuse_cols = [col for col in data.columns if col.startswith('DLMUSE')]

# 2. Define your fixed columns
fixed_cols = ['Age', 'Sex', 'MRID', 'PTID', 'DX_AD', 'Delta_Baseline', 'Study']

# 3. Combine them and filter the DataFrame
data = data[fixed_cols + dlmuse_cols]

# Optional: Verify the new shape
print(data.head())

# load the istaging 2.0 studies with the harmonized DLMUSE and SPARE_BA
additional_data = pd.read_csv('additional_studies.csv')

print('Studies in data', data['Study'].unique())
print('Studies in additional data', additional_data['Study'].unique())
sys.exit(0)

# stach the additional data to the data. 


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

# Convert Delta_Baseline from days to months
data['Delta_Baseline'] = data['Delta_Baseline'] / 30

# Remove duplicate Time entries per subject
data = data.groupby(['PTID', 'Time']).agg(lambda x: x.iloc[0]).reset_index()
print(f'Subjects after time deduplication: {data["PTID"].nunique()}')

data_unnorm = data.copy()

# ---------------------------------------------------------------------------
# 7. Z-score MUSE ROIs using pre-computed combined normalization stats
# ---------------------------------------------------------------------------
subjects_df_hmuse = data.filter(regex=r'^DLMUSE_')

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

# ---------------------------------------------------------------------------
# 8. Baseline columns
# ---------------------------------------------------------------------------
data['Baseline_Age'] = data.groupby('PTID')['Age'].transform('min')

# for col in [c for c in data.columns if c.startswith('DLMUSE_')]:
#     data['Baseline_' + col] = data.groupby('PTID')[col].transform('first')

# for col in ['SPARE_AD', 'SPARE_BA', 'Diagnosis_nearest_2.0']:
#     data['Baseline_' + col] = data.groupby('PTID')[col].transform('first')

# Keep only non-negative timepoints
data = data[data['Time'] >= 0]

# ---------------------------------------------------------------------------
# 9. BAG = SPARE_BA - Age  (computed before normalization)
# ---------------------------------------------------------------------------
data['BAG'] = data['SPARE_BA'] - data['Age']
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

data['Age']      = (data['Age']      - mean_age)     / std_age
data['SPARE_BA'] = (data['SPARE_BA'] - mean_spareba) / std_spareba
data['BAG']      = (data['BAG']      - mean_bag)     / std_bag

data['Education_Years'] = (data['Education_Years'] > 16).astype(int)
data['Sex'].replace(['M', 'F'], [0, 1], inplace=True)

print(f'  Age:      mean={mean_age:.2f}, std={std_age:.2f}')
print(f'  SPARE_BA: mean={mean_spareba:.2f}, std={std_spareba:.2f}')
print(f'  BAG:      mean={mean_bag:.2f}, std={std_bag:.2f}')

clinical_features = ['Sex', 'Age', 'BAG', 'PTID', 'Delta_Baseline', 'Time']
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
features = [name for name in data.columns if name.startswith('MUSE_Volume') and int(name[12:]) < 300]
features.extend(clinical_features)

with open(data_dir + 'features_bag.pkl', 'wb') as f:
    pickle.dump(features, f)


target = ['BAG']

samples, subject_data, num_samples, list_of_subjects, list_of_subject_ids, cnt, covs, longitudinal_covariates = create_baseline_temporal_dataset(subjects=all_subjects, dataframe=data, dataframeunnorm=data_unnorm,  target=target, features=features, hmuse=hmuse,  genomic=0, followup=0, derivedroi='all', visualize=False)

# samples, subject_data, num_samples, list_of_subjects, list_of_subject_ids, cnt = create_n_acquisition_temporal_dataset(n=3, subjects=all_subjects, dataframe=data, dataframeunnorm=data_unnorm,  target=target, features=features, hmuse=hmuse,  genomic=0, followup=0, derivedroi='all', visualize=False)

samples_df = pd.DataFrame(data=samples)
longitudinal_covariates_df = pd.DataFrame(data=longitudinal_covariates)
# longitudinal_covariates_df.to_csv(data_dir + 'longitudinal_covariates_bag_allstudies.csv', index=False)
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
