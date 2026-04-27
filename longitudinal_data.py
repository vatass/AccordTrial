'''
ACCORD Trial  - CN Digital Twin - DKGP
iSTAGING 3.0 
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

    longitudinal_covariates = {'PTID': [], 'Time': [], 'Age': [],  'Diagnosis': [], 'Study': [], 'SPARE_BA': [], 'DLICV': [], 'Race': [], 'Sex':[] }


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
    # print('Target', len(target))
    # print('Input Features', features)

    for i, subject_id in enumerate(subjects):

        subject = dataframe[dataframe['PTID']==subject_id]
        subject_unnorm = dataframeunnorm[dataframeunnorm['PTID']==subject_id]


        # print(subject)
        for k in range(0, subject.shape[0]):
            samples['PTID'].append(subject_id)

            # print('Baseline Features',  features)

            x = subject[features].iloc[0].to_list()

            # print(x)

            delta = subject['Time'].iloc[k]
            # man_device = subject['MRI_Scanner_Model'].iloc[k]
            diagnosis = subject['DX_AD'].iloc[k]
            baseline_diagnosis = subject['DX_AD'].iloc[0]
            baseline_age = subject_unnorm['Age'].iloc[0]
            age = subject_unnorm['Age'].iloc[k]
            dlicv = subject_unnorm['DLICV'].iloc[k]
            study = subject_unnorm['Study'].iloc[k]
            race = subject_unnorm['Race'].iloc[k]
            sex = subject['Sex'].iloc[k]
      
            spba = subject_unnorm['SPARE_BA'].iloc[k]

            # print('Delta', delta)
            x.extend([delta])

            # print('Input', x)
            # print('Target', target)
            t = subject[target].iloc[k] #.to_list()

            # print('Target', t)
      
            longitudinal_covariates['PTID'].append(subject_id)
            longitudinal_covariates['Time'].append(delta)
            longitudinal_covariates['Age'].append(age)
            longitudinal_covariates['Diagnosis'].append(diagnosis)
            longitudinal_covariates['DLICV'].append(dlicv)
            longitudinal_covariates['Study'].append(study)
            longitudinal_covariates['Race'].append(race)
            longitudinal_covariates['Sex'].append(sex)
            longitudinal_covariates['SPARE_BA'].append(spba)

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

    return samples, subject_data, num_samples, list_of_subjects, list_of_subject_ids, cnt, longitudinal_covariates

"""**Data Selection**
1. Read Data and remove all ADNI Screening and BLSA 1.5 T
2. Drop all NaN MUSE
3. Map the Diagnosis Column

"""

data_dir = './data/'

data = pd.read_csv('/cbica/home/harmang/harmonization_evaluation/istaging_3_0.csv')

print(f'Loaded: {data.shape}')
print(f'Unique MRIDs in data: {data["MRID"].nunique()}')

spare_ba = pd.read_csv('SPARE_BA_istaging_3_0_all.csv')
print(f'Unique MRIDs in SPARE_BA file: {spare_ba["MRID"].nunique()}')

# Drop existing SPARE_BA if present to avoid duplicates after merge
if 'SPARE_BA' in data.columns:
    data = data.drop(columns=['SPARE_BA'])

data = data.merge(spare_ba[['MRID', 'SPARE_BA']], on='MRID', how='left')
n_matched = data['SPARE_BA'].notna().sum()
print(f'After SPARE_BA merge: {n_matched}/{len(data)} rows have SPARE_BA ({100*n_matched/len(data):.1f}%)')

accord_data = data[data['Study']=='ACCORD']

print('ACCORD Study:', accord_data['PTID'].nunique())

# Spot-check: pick a random MRID that exists in spare_ba and verify the value survived the merge
_sample = spare_ba[spare_ba['MRID'].isin(data['MRID'])].sample(1, random_state=42).iloc[0]
_mrid, _expected = _sample['MRID'], _sample['SPARE_BA']
_actual = data.loc[data['MRID'] == _mrid, 'SPARE_BA'].iloc[0]
assert _actual == _expected, f'SPARE_BA mismatch for MRID {_mrid}: got {_actual}, expected {_expected}'
print(f'Spot-check passed — MRID {_mrid}: SPARE_BA={_actual:.4f} (matches source)')

# ---------------------------------------------------------------------------
# 2. Basic Filters
# ---------------------------------------------------------------------------
print('Removing BLSA 1.5T data and BIOCARD...')
data = data[data['SITE'] != 'BLSA-1.5T']
data = data[data['Study'] != 'BIOCARD']


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
# 4. Keep only subjects that are CN (0 or -1 ) at all timepoints
# ---------------------------------------------------------------------------
cn_mask = data.groupby('PTID')['DX_AD'].apply(lambda x: x.isin([0, -1]).all())
data = data[data['PTID'].isin(cn_mask[cn_mask].index)]
print(f'CN-only subjects: {data["PTID"].nunique()}')

print(list(data['Study'].unique()))

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

# Step 1: Remove studies where dates are ENTIRELY unavailable (no MRID contains a date).
# These are handled separately via additional_data.csv.
study_any_date = data.groupby('Study')['Date'].apply(lambda x: x.notna().any())
undatable_studies = study_any_date[~study_any_date].index.tolist()
if undatable_studies:
    print(f'Removing {len(undatable_studies)} entirely undatable studies: {undatable_studies}')
    data = data[~data['Study'].isin(undatable_studies)]
else:
    print('All studies in data have at least one parseable date.')
print(f'Subjects after undatable-study removal: {data["PTID"].nunique()}')

# Step 2: Within datable studies, drop individual rows whose MRID date could not be parsed.
before_rows = data.shape[0]
before_subj = data['PTID'].nunique()
data = data[data['Date'].notna()].copy()
print(f'Dropped {before_rows - data.shape[0]} rows with unparseable dates within datable studies '
      f'({before_subj - data["PTID"].nunique()} subjects lost entirely)')
print(f'Subjects after per-row date filter: {data["PTID"].nunique()}')
print('Remaining studies:', sorted(data['Study'].unique()))


additional_data = pd.read_csv('additional_data.csv')

additional_data = additional_data.rename(columns={
    col: col.replace("H_DL_MUSE_Volume", "DLMUSE")
    for col in additional_data.columns
    if col.startswith("H_DL_MUSE_Volume")
})


dlmuse_cols = [c for c in additional_data.columns if c.startswith('DLMUSE_') and int(c[7:]) < 300]

print('Studies in additional data:', additional_data['Study'].unique())
print('Columns in additional data:', additional_data.columns.tolist())
print('Columns in data', data.columns.tolist())

# ---------------------------------------------------------------------------
# NaN analysis of additional_data before concatenation
# ---------------------------------------------------------------------------
print(f'\n=== additional_data NaN analysis ({additional_data.shape[0]} rows, '
      f'{additional_data["PTID"].nunique()} subjects) ===')

nan_counts = additional_data.isna().sum()
nan_cols = nan_counts[nan_counts > 0].sort_values(ascending=False)

if nan_cols.empty:
    print('  No NaN values found in additional_data — clean.')
else:
    total_cells = additional_data.shape[0] * additional_data.shape[1]
    total_nan   = nan_cols.sum()
    print(f'  Total NaN cells : {total_nan} / {total_cells} '
          f'({100 * total_nan / total_cells:.2f}%)')
    print(f'  Columns with NaN: {len(nan_cols)} / {additional_data.shape[1]}')

    # Separate DLMUSE ROI columns from clinical/metadata columns
    dlmuse_nan = nan_cols[nan_cols.index.str.startswith('DLMUSE_')]
    other_nan  = nan_cols[~nan_cols.index.str.startswith('DLMUSE_')]

    if not other_nan.empty:
        print('\n  Non-DLMUSE columns with NaN:')
        for col, cnt in other_nan.items():
            pct = 100 * cnt / additional_data.shape[0]
            print(f'    {col}: {cnt} ({pct:.1f}%)')

    if not dlmuse_nan.empty:
        print(f'\n  DLMUSE columns with NaN: {len(dlmuse_nan)} columns')
        print('  Top-10 worst DLMUSE NaN columns:')
        for col, cnt in dlmuse_nan.head(10).items():
            pct = 100 * cnt / additional_data.shape[0]
            print(f'    {col}: {cnt} ({pct:.1f}%)')
    else:
        print('  No NaN values in DLMUSE ROI columns — good.')

# Zero analysis for DLMUSE ROIs in additional_data
add_dlmuse_cols = [c for c in additional_data.columns
                   if c.startswith('DLMUSE_') and int(c[7:]) < 300]
if add_dlmuse_cols:
    zero_counts = (additional_data[add_dlmuse_cols] == 0).sum()
    zero_cols   = zero_counts[zero_counts > 0].sort_values(ascending=False)
    print(f'\n  DLMUSE zero analysis ({len(add_dlmuse_cols)} ROIs):')
    if zero_cols.empty:
        print('  No zero values in DLMUSE ROI columns.')
    else:
        print(f'  Rows with any zero DLMUSE value: '
              f'{(additional_data[add_dlmuse_cols] == 0).any(axis=1).sum()} '
              f'/ {additional_data.shape[0]}')
        print('  Top-10 worst DLMUSE zero columns:')
        for col, cnt in zero_cols.head(10).items():
            pct = 100 * cnt / additional_data.shape[0]
            print(f'    {col}: {cnt} ({pct:.1f}%)')

print('=== End additional_data analysis ===\n')

# ---------------------------------------------------------------------------
# DLMUSE quality filter on additional_data (before concat)
# Each dataframe is filtered using only its OWN columns so that column
# differences between istaging_3_0.csv and additional_data.csv do not
# cause cross-contamination NaN after concat.
# ---------------------------------------------------------------------------
add_dlmuse_filter_cols = [c for c in additional_data.columns
                          if c.startswith('DLMUSE_') and int(c[7:]) < 300]
if add_dlmuse_filter_cols:
    add_nan_mask  = additional_data[add_dlmuse_filter_cols].isna().any(axis=1)
    add_zero_mask = (additional_data[add_dlmuse_filter_cols] == 0).any(axis=1)
    add_bad_mask  = add_nan_mask | add_zero_mask
    n_before = additional_data.shape[0]
    additional_data = additional_data[~add_bad_mask].reset_index(drop=True)
    print(f'additional_data DLMUSE filter: dropped {n_before - additional_data.shape[0]} rows '
          f'({additional_data["PTID"].nunique()} subjects remain)')

print(data['delta_days_imaging_baseline'].describe())
data = data.rename(columns={'delta_days_imaging_baseline': 'Delta_Baseline'})

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
before_dedup = data['PTID'].nunique()
data = data.groupby(['PTID', 'Time']).agg(lambda x: x.iloc[0]).reset_index()
print(f'Subjects after time deduplication: {data["PTID"].nunique()} '
      f'(was {before_dedup}, lost {before_dedup - data["PTID"].nunique()})')

# ---------------------------------------------------------------------------
# DLMUSE quality filter on main data (before concat)
# ---------------------------------------------------------------------------
data_dlmuse_cols = [c for c in data.columns if c.startswith('DLMUSE_') and int(c[7:]) < 300]
before_rows = data.shape[0]
before_subj = data['PTID'].nunique()

nan_muse_mask  = data[data_dlmuse_cols].isna().any(axis=1)
zero_muse_mask = (data[data_dlmuse_cols] == 0).any(axis=1)
bad_muse_mask  = nan_muse_mask | zero_muse_mask

print(f'\n=== DLMUSE quality filter — main data ({len(data_dlmuse_cols)} ROIs) ===')
print(f'  Rows with NaN  in any ROI: {nan_muse_mask.sum()}')
print(f'  Rows with zero in any ROI: {zero_muse_mask.sum()}')
for _study in sorted(data['Study'].unique()):
    _smask = data['Study'] == _study
    _dropped = (bad_muse_mask & _smask).sum()
    if _dropped > 0:
        print(f'    {_study}: {_dropped}/{_smask.sum()} rows dropped')
data = data[~bad_muse_mask].reset_index(drop=True)
print(f'  Dropped {before_rows - data.shape[0]} rows total '
      f'({before_subj - data["PTID"].nunique()} subjects lost entirely)')
print(f'  Remaining: {data.shape[0]} rows, {data["PTID"].nunique()} subjects')

# Concatenate additional_data (undatable studies, already filtered above)
print(f'\nadditional_data: {additional_data.shape[0]} rows, {additional_data["PTID"].nunique()} subjects')
data = pd.concat([data, additional_data], ignore_index=True)
print(f'After concat: {data.shape[0]} rows, {data["PTID"].nunique()} subjects')

# Validation
n_dup = data.duplicated(subset=['PTID', 'MRID']).sum()
assert n_dup == 0, f'{n_dup} duplicate (PTID, MRID) pairs found after concat!'
print('Validation passed: no duplicate (PTID, MRID) pairs.')

print('\n=== Post-concatenation summary ===')
print(f'Total rows    : {data.shape[0]}')
print(f'Total subjects: {data["PTID"].nunique()}')
print(f'Total studies : {data["Study"].nunique()}')
print('\nPer-study breakdown:')
for _study in sorted(data['Study'].unique()):
    _n_subj = data[data['Study'] == _study]['PTID'].nunique()
    _n_rows = (data['Study'] == _study).sum()
    print(f'  {_study}: {_n_subj} subjects, {_n_rows} rows')

# ---------------------------------------------------------------------------
# Drop rows with zero or NaN in any of the 145 DLMUSE ROI features
# Zero volume = segmentation failure; NaN = missing segmentation entirely
# ---------------------------------------------------------------------------
dlmuse_cols = [c for c in data.columns if c.startswith('DLMUSE_') and int(c[7:]) < 300]
before_rows = data.shape[0]
before_subj = data['PTID'].nunique()

nan_muse_mask  = data[dlmuse_cols].isna().any(axis=1)
bad_muse_mask  = nan_muse_mask 

print(f'\n=== DLMUSE quality filter ({len(dlmuse_cols)} ROIs) ===')
print(f'  Rows with NaN  in any ROI: {nan_muse_mask.sum()}')
# Per-study breakdown of dropped rows
print('  Dropped rows per study:')
for _study in sorted(data['Study'].unique()):
    _smask = data['Study'] == _study
    _dropped = (bad_muse_mask & _smask).sum()
    _total   = _smask.sum()
    if _dropped > 0:
        print(f'    {_study}: {_dropped}/{_total} rows dropped')
data = data[~bad_muse_mask].reset_index(drop=True)

print(f'  Dropped {before_rows - data.shape[0]} rows total '
      f'({before_subj - data["PTID"].nunique()} subjects lost entirely)')
print(f'  Remaining: {data.shape[0]} rows, {data["PTID"].nunique()} subjects')

data_unnorm = data.copy()

print('Studies', data['Study'].unique())
print('Subjects', data['PTID'].nunique())

# ---------------------------------------------------------------------------
# 7. Z-score MUSE ROIs — compute stats from current data and save
# ---------------------------------------------------------------------------
subjects_df_hmuse = data.filter(regex=r'^DLMUSE_')

mean_hmuse = subjects_df_hmuse.mean(axis=0).tolist()
std_hmuse  = subjects_df_hmuse.std(axis=0).tolist()

muse_pkl = data_dir + '145_MUSE_allstudies_mean_std.pkl'
with open(muse_pkl, 'wb') as f:
    pickle.dump({'mean': mean_hmuse, 'std': std_hmuse}, f)
print(f'MUSE stats computed and saved to: {muse_pkl}')

for i, c in enumerate(subjects_df_hmuse.columns):
    data[c] = (subjects_df_hmuse[c] - mean_hmuse[i]) / std_hmuse[i]

# ---------------------------------------------------------------------------
# 8. Baseline columns
# ---------------------------------------------------------------------------
data['Baseline_Age'] = data.groupby('PTID')['Age'].transform('min')

# Keep only non-negative timepoints
data = data[data['Time'] >= 0]

# ---------------------------------------------------------------------------
# 9. BAG = SPARE_BA - Age  (computed before normalization)
# ---------------------------------------------------------------------------
data['BAG'] = data['SPARE_BA'] - data['Age']
print(f'BAG — mean: {data["BAG"].mean():.2f}, std: {data["BAG"].std():.2f}')

# ---------------------------------------------------------------------------
# 10. Normalize clinical variables — compute stats from current data and save
# ---------------------------------------------------------------------------
mean_age,     std_age     = data['Age'].mean(),      data['Age'].std()
mean_spareba, std_spareba = data['SPARE_BA'].mean(),  data['SPARE_BA'].std()
mean_bag,     std_bag     = data['BAG'].mean(),       data['BAG'].std()

normalization_stats = {
    'Age':      {'mean': mean_age,     'std': std_age},
    'SPARE_BA': {'mean': mean_spareba, 'std': std_spareba},
    'BAG':      {'mean': mean_bag,     'std': std_bag},
}
norm_pkl = data_dir + 'normalization_stats.pkl'
with open(norm_pkl, 'wb') as f:
    pickle.dump(normalization_stats, f)
print(f'Normalization stats computed and saved to: {norm_pkl}')

data['Age']      = (data['Age']      - mean_age)     / std_age
data['SPARE_BA'] = (data['SPARE_BA'] - mean_spareba) / std_spareba
data['BAG']      = (data['BAG']      - mean_bag)     / std_bag
data['Sex'].replace(['M', 'F'], [0, 1], inplace=True)

print(f'  Age:      mean={mean_age:.2f}, std={std_age:.2f}')
print(f'  SPARE_BA: mean={mean_spareba:.2f}, std={std_spareba:.2f}')
print(f'  BAG:      mean={mean_bag:.2f}, std={std_bag:.2f}')

# Capture ACCORD slice NOW — after BAG is computed and normalized
accord_data        = data[data['Study'] == 'ACCORD']
accord_data_unnorm = data_unnorm[data_unnorm['Study'] == 'ACCORD']

print(f'ACCORD subjects: {accord_data["PTID"].nunique()}')
print('ACCORD BAG (normalized):')
print(accord_data['BAG'].describe())

clinical_features = ['Sex', 'Age', 'BAG', 'PTID', 'Delta_Baseline', 'Time']
for cf in clinical_features:
    data[cf] = data[cf].fillna(-1)

# ---------------------------------------------------------------------------
# 11. Save CSV (BAG biomarker)
# ---------------------------------------------------------------------------
# Exclude ACCORD from the training set (held out as prospective test cohort)
data = data[data['Study'] != 'ACCORD']
all_subjects = list(data['PTID'].unique())
print(f'Total subjects (training, ACCORD excluded): {len(all_subjects)}')
print('Studies apart from ACCORD:', data['Study'].unique())

# ---------------------------------------------------------------------------
# 12. Save features pickle
# ---------------------------------------------------------------------------
features = [name for name in data.columns if name.startswith('DLMUSE_') and int(name[7:]) < 300]
features.extend(clinical_features)

with open(data_dir + 'features_bag.pkl', 'wb') as f:
    pickle.dump(features, f)

target = ['BAG']

samples, subject_data, num_samples, list_of_subjects, list_of_subject_ids, cnt, longitudinal_covariates = create_baseline_temporal_dataset(subjects=all_subjects, dataframe=data, dataframeunnorm=data_unnorm,  target=target, features=features, hmuse=hmuse,  genomic=0, followup=0, derivedroi='all', visualize=False)

samples_df = pd.DataFrame(data=samples)
longitudinal_covariates_df = pd.DataFrame(data=longitudinal_covariates)
longitudinal_covariates_df.to_csv(data_dir + 'longitudinal_covariates_bag_allstudies.csv', index=False)
samples_df.to_csv(data_dir + 'subjectsamples_bag_'+'allstudies'+'.csv')

features = [name for name in data.columns if name.startswith('DLMUSE_') and int(name[7:]) < 300]
features.extend(clinical_features)


accord_subjects = list(accord_data['PTID'].unique())
print('ACCORD Subjects', len(accord_subjects))
accord_samples, accord_subject_data, accord_num_samples, accord_list_of_subjects, accord_list_of_subject_ids, accord_cnt, accord_longitudinal_covariates = create_baseline_temporal_dataset(subjects=accord_subjects, dataframe=accord_data, dataframeunnorm=accord_data_unnorm,  target=target, features=features, hmuse=hmuse,  genomic=0, followup=0, derivedroi='all', visualize=False)
accord_samples_df = pd.DataFrame(data=samples)
accord_samples_df.to_csv(data_dir + 'subjectsamples_bag_'+'accord'+'.csv')

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
