'''
ACCORD Trial - Data Processing for DKGP Model Input

Produces a SubjectSamples CSV in the same format as
longitudinal_data.py (columns: PTID, X, Y) by applying the
population normalization statistics that were saved during iSTAGING
training.

Expected ACCORD data columns (configurable via COLUMN_MAP below):
  - PTID / MaskID  : subject identifier
  - Visit_Code     : visit label
  - Delta_Baseline : days from baseline (0 at baseline visit)
  - Age            : age at scan (years)
  - Sex            : 'M' / 'F'
  - Education_Years: years of education
  - SPARE_BA       : brain age score
  - SPARE_AD       : AD-likelihood score
  - H_MUSE_Volume_*: MUSE ROI volumes (same set as iSTAGING)

Usage
-----
python accord_data_processing.py \
    --accord_data   /path/to/accord_mri.csv \
    --stats_dir     data/ \
    --output_dir    data/ \
    [--ptid_col     MaskID] \
    [--visit_col    Visit_Code]
'''

import os
import sys
import argparse
import pickle
import math

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Prepare ACCORD data for DKGP model input')
parser.add_argument('--accord_data',  required=True,
                    help='Path to ACCORD MRI data file (.csv or .pkl / .pkl.gz)')
parser.add_argument('--stats_dir',    default='data/',
                    help='Directory containing population normalization stats '
                         '(normalization_stats.pkl, '
                         '145_harmonized_allstudies_mean_std_hmuse.pkl, '
                         'features_bag.pkl)')
parser.add_argument('--output_dir',   default='data/',
                    help='Directory for output files')
parser.add_argument('--ptid_col',     default=None,
                    help='Column name for subject ID (auto-detected if omitted)')
parser.add_argument('--visit_col',    default=None,
                    help='Column name for visit code (auto-detected if omitted)')
parser.add_argument('--min_visits',   type=int, default=1,
                    help='Minimum number of visits per subject to include (default: 1)')
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load population normalization statistics
# ---------------------------------------------------------------------------
print('Loading population normalization statistics...')

norm_stats_path = os.path.join(args.stats_dir, 'normalization_stats.pkl')
hmuse_stats_path = os.path.join(args.stats_dir, '145_harmonized_allstudies_mean_std_hmuse.pkl')
features_path = os.path.join(args.stats_dir, 'features_bag.pkl')

if not os.path.isfile(norm_stats_path):
    sys.exit(f'ERROR: {norm_stats_path} not found. Run longitudinal_data.py first.')
if not os.path.isfile(hmuse_stats_path):
    sys.exit(f'ERROR: {hmuse_stats_path} not found. Run longitudinal_data.py first.')
if not os.path.isfile(features_path):
    sys.exit(f'ERROR: {features_path} not found. Run longitudinal_data.py first.')

with open(norm_stats_path, 'rb') as f:
    norm_stats = pickle.load(f)

with open(hmuse_stats_path, 'rb') as f:
    hmuse_stats = pickle.load(f)

with open(features_path, 'rb') as f:
    population_features = pickle.load(f)

mean_age,     std_age     = norm_stats['Age']['mean'],     norm_stats['Age']['std']
mean_spareba, std_spareba = norm_stats['SPARE_BA']['mean'], norm_stats['SPARE_BA']['std']
mean_bag,     std_bag     = norm_stats['BAG']['mean'],     norm_stats['BAG']['std']

# H_MUSE mean/std lists are ordered the same as subjects_df_hmuse.columns
hmuse_means = hmuse_stats['mean']
hmuse_stds  = hmuse_stats['std']

# H_MUSE columns used as features (same selection as longitudinal_data.py)
hmuse_feature_cols = [f for f in population_features if f.startswith('H_MUSE_Volume')]

print(f'  Age      : mean={mean_age:.2f}, std={std_age:.2f}')
print(f'  SPARE_BA : mean={mean_spareba:.2f}, std={std_spareba:.2f}')
print(f'  BAG      : mean={mean_bag:.2f}, std={std_bag:.2f}')
print(f'  H_MUSE features : {len(hmuse_feature_cols)}')

# Build a name->index lookup for H_MUSE normalization
# hmuse_stats was built from subjects_df_hmuse which used data.filter(regex='H_MUSE*')
# The order must match; we derive it from the feature list saved during training.
hmuse_all_cols = [f for f in population_features if f.startswith('H_MUSE_Volume')]
hmuse_idx_map  = {col: i for i, col in enumerate(hmuse_all_cols)}

# ---------------------------------------------------------------------------
# 2. Load ACCORD data
# ---------------------------------------------------------------------------
print(f'\nLoading ACCORD data from: {args.accord_data}')

if args.accord_data.endswith('.csv'):
    data = pd.read_csv(args.accord_data)
elif args.accord_data.endswith('.pkl') or args.accord_data.endswith('.pkl.gz'):
    data = pd.read_pickle(args.accord_data)
else:
    sys.exit('ERROR: --accord_data must be a .csv, .pkl, or .pkl.gz file.')

print(f'Loaded {data.shape[0]} rows, {data.shape[1]} columns')

# ---------------------------------------------------------------------------
# 3. Identify / rename key columns
# ---------------------------------------------------------------------------
# Auto-detect PTID column
PTID_CANDIDATES = ['PTID', 'MaskID', 'SubjectID', 'ID', 'subject_id', 'maskid']
VISIT_CANDIDATES = ['Visit_Code', 'Visit', 'visit', 'VISIT', 'VisitCode']

def _find_col(df, candidates, override=None, label=''):
    if override and override in df.columns:
        return override
    for c in candidates:
        if c in df.columns:
            return c
    print(f'WARNING: Could not auto-detect {label} column. '
          f'Available columns: {list(df.columns)}')
    return None

ptid_col  = _find_col(data, PTID_CANDIDATES,  args.ptid_col,  'PTID')
visit_col = _find_col(data, VISIT_CANDIDATES,  args.visit_col, 'Visit_Code')

if ptid_col is None:
    sys.exit('ERROR: Could not identify PTID column. Use --ptid_col.')

# Standardize to 'PTID'
if ptid_col != 'PTID':
    data = data.rename(columns={ptid_col: 'PTID'})
    print(f'  Renamed {ptid_col} -> PTID')

data['PTID'] = data['PTID'].astype(str)

# Standardize to 'Visit_Code' if present
if visit_col and visit_col != 'Visit_Code' and visit_col in data.columns:
    data = data.rename(columns={visit_col: 'Visit_Code'})
    print(f'  Renamed {visit_col} -> Visit_Code')

print(f'  Subjects: {data["PTID"].nunique()}')

# ---------------------------------------------------------------------------
# 4. Compute / validate Delta_Baseline (days from baseline)
# ---------------------------------------------------------------------------
if 'Delta_Baseline' not in data.columns:
    # Try to derive from a date column
    date_candidates = ['Date', 'ScanDate', 'scan_date', 'ExamDate']
    date_col = _find_col(data, date_candidates, label='Date')
    if date_col:
        data[date_col] = pd.to_datetime(data[date_col])
        data = data.sort_values(['PTID', date_col])
        data['Delta_Baseline'] = (
            data.groupby('PTID')[date_col]
            .transform(lambda x: (x - x.iloc[0]).dt.days)
        )
        print(f'  Delta_Baseline derived from {date_col}')
    else:
        # Assume a single-visit dataset; all timepoints = 0
        print('  WARNING: No Delta_Baseline or date column found. Assuming single-visit (time=0).')
        data['Delta_Baseline'] = 0.0
else:
    # Fix: ensure first visit per subject is 0
    data = data.sort_values(['PTID', 'Delta_Baseline'])
    data['Delta_Baseline'] = data.groupby('PTID')['Delta_Baseline'].transform(
        lambda x: x - x.iloc[0]
    )

# Convert days -> months (ceiling, consistent with longitudinal_data.py)
data['Time'] = np.ceil(data['Delta_Baseline'] / 30).astype(int)

# Remove duplicate (PTID, Time) rows
data = data.groupby(['PTID', 'Time']).agg(lambda x: x.iloc[0]).reset_index()
data = data[data['Time'] >= 0]
print(f'  Subjects after time deduplication: {data["PTID"].nunique()}')

# ---------------------------------------------------------------------------
# 5. Filter by minimum visits
# ---------------------------------------------------------------------------
if args.min_visits > 1:
    data = data.groupby('PTID').filter(lambda x: x.shape[0] >= args.min_visits)
    print(f'  Subjects with >= {args.min_visits} visits: {data["PTID"].nunique()}')

# ---------------------------------------------------------------------------
# 6. Validate and prepare H_MUSE columns
# ---------------------------------------------------------------------------
available_hmuse = [c for c in hmuse_feature_cols if c in data.columns]
missing_hmuse   = [c for c in hmuse_feature_cols if c not in data.columns]

if missing_hmuse:
    print(f'  WARNING: {len(missing_hmuse)} H_MUSE columns missing from ACCORD data '
          f'(will be filled with column mean from population normalization):')
    for c in missing_hmuse[:10]:
        print(f'    {c}')
    if len(missing_hmuse) > 10:
        print(f'    ... and {len(missing_hmuse) - 10} more')
    # Fill missing columns with 0 (= population mean in z-score space)
    for c in missing_hmuse:
        data[c] = np.nan

if not available_hmuse:
    sys.exit('ERROR: No H_MUSE columns found in ACCORD data. '
             'Ensure MUSE processing has been run on ACCORD MRI data.')

# Drop rows missing ALL H_MUSE ROIs
data = data.dropna(axis=0, subset=available_hmuse)
print(f'  Subjects after H_MUSE NaN removal: {data["PTID"].nunique()}')

# ---------------------------------------------------------------------------
# 7. Normalize H_MUSE with population statistics
# ---------------------------------------------------------------------------
print('\nApplying population H_MUSE normalization...')

data_unnorm = data.copy()  # keep original values for longitudinal covariates

for col in hmuse_feature_cols:
    idx = hmuse_idx_map.get(col)
    if idx is None:
        continue
    mu  = hmuse_means[idx]
    sig = hmuse_stds[idx]
    if col in data.columns:
        data[col] = (data[col] - mu) / sig
    else:
        data[col] = 0.0  # fill missing with population mean (z=0)

# ---------------------------------------------------------------------------
# 8. Normalize / encode clinical variables with population statistics
# ---------------------------------------------------------------------------
print('Applying population clinical normalization...')

# Sex: M=0, F=1
if 'Sex' in data.columns:
    data['Sex'] = data['Sex'].replace({'M': 0, 'F': 1, 'MALE': 0, 'FEMALE': 1,
                                       'male': 0, 'female': 1})
else:
    print('  WARNING: Sex column not found; defaulting to 0 (Male).')
    data['Sex'] = 0

# Education_Years: binarized (>16 = 1)
if 'Education_Years' in data.columns:
    data['Education_Years'] = (data['Education_Years'] > 16).astype(int)
else:
    data['Education_Years'] = 0

# SPARE_BA normalization (population z-score)
if 'SPARE_BA' in data.columns:
    data['SPARE_BA'] = (data['SPARE_BA'] - mean_spareba) / std_spareba
else:
    print('  WARNING: SPARE_BA not found; BAG target will not be available.')

# Age normalization (needed only for BAG; keep unnorm version separately)
if 'Age' not in data.columns:
    print('  WARNING: Age column not found; BAG target will not be available.')

# ---------------------------------------------------------------------------
# 9. Compute BAG target (normalized)
#    BAG = SPARE_BA_original - Age_original, then z-norm with population stats
# ---------------------------------------------------------------------------
if 'SPARE_BA' in data_unnorm.columns and 'Age' in data_unnorm.columns:
    data['BAG'] = (data_unnorm['SPARE_BA'] - data_unnorm['Age'] - mean_bag) / std_bag
    has_bag = True
    print(f'  BAG computed — mean (normalized): {data["BAG"].mean():.3f}')
else:
    data['BAG'] = np.nan
    has_bag = False
    print('  WARNING: BAG could not be computed (missing SPARE_BA or Age).')

# ---------------------------------------------------------------------------
# 10. Create baseline columns (first observation per subject for each H_MUSE)
# ---------------------------------------------------------------------------
print('\nCreating baseline feature columns...')

data = data.sort_values(['PTID', 'Time'])

for col in hmuse_feature_cols:
    data['Baseline_' + col] = data.groupby('PTID')[col].transform('first')

if 'Age' in data.columns:
    data['Baseline_Age'] = data.groupby('PTID')['Age'].transform('first')

# ---------------------------------------------------------------------------
# 11. Build SubjectSamples (PTID, X, Y) matching longitudinal_data.py format
#
# X = [Baseline_H_MUSE_Volume_0, ..., Baseline_H_MUSE_Volume_N, Sex, Time]
#     identical ordering to population_features list + Time appended last
# Y = [BAG_normalized]
# ---------------------------------------------------------------------------
print('\nBuilding SubjectSamples...')

# Determine which baseline features are present
baseline_hmuse_cols = ['Baseline_' + c for c in hmuse_feature_cols]

# Clinical features from population_features (exclude PTID, Delta_Baseline, Time)
clinical_cols = [f for f in population_features
                 if not f.startswith('H_MUSE') and f not in ('PTID', 'Delta_Baseline', 'Time')]

samples = {'PTID': [], 'X': [], 'Y': []}

longitudinal_covariates = {
    'PTID': [], 'Time': [], 'Age': [], 'Sex': [],
    'Education_Years': [], 'SPARE_BA': [], 'SPARE_AD': [],
    'Hypertension': [], 'Diabetes': [], 'BAG': [],
}

all_subjects = list(data['PTID'].unique())
print(f'  Total subjects: {len(all_subjects)}')

skipped = 0
for subject_id in all_subjects:
    subject     = data[data['PTID'] == subject_id]
    subject_raw = data_unnorm[data_unnorm['PTID'] == subject_id]

    if subject.empty:
        skipped += 1
        continue

    for k in range(subject.shape[0]):
        row     = subject.iloc[k]
        row_raw = subject_raw.iloc[k] if k < subject_raw.shape[0] else subject_raw.iloc[-1]

        # --- Build X ---
        # Baseline H_MUSE features (normalized)
        x = []
        for col in hmuse_feature_cols:
            bcol = 'Baseline_' + col
            x.append(float(subject[bcol].iloc[0]) if bcol in subject.columns else 0.0)

        # Clinical features (Sex only in population_features beyond H_MUSE)
        for cf in clinical_cols:
            if cf in subject.columns:
                x.append(float(row[cf]) if not pd.isna(row[cf]) else 0.0)
            else:
                x.append(0.0)

        # Time (months) — always last, matching population feature order
        x.append(int(row['Time']))

        # --- Build Y ---
        bag_val = float(row['BAG']) if has_bag and not pd.isna(row['BAG']) else float('nan')
        y = [bag_val]

        samples['PTID'].append(subject_id)
        samples['X'].append(x)
        samples['Y'].append(y)

        # Longitudinal covariates
        longitudinal_covariates['PTID'].append(subject_id)
        longitudinal_covariates['Time'].append(int(row['Time']))
        longitudinal_covariates['Age'].append(
            float(row_raw['Age']) if 'Age' in row_raw.index and not pd.isna(row_raw['Age']) else float('nan'))
        longitudinal_covariates['Sex'].append(float(row['Sex']))
        longitudinal_covariates['Education_Years'].append(float(row['Education_Years']))
        longitudinal_covariates['SPARE_BA'].append(
            float(row_raw['SPARE_BA']) if 'SPARE_BA' in row_raw.index and not pd.isna(row_raw['SPARE_BA']) else float('nan'))
        longitudinal_covariates['SPARE_AD'].append(
            float(row_raw['SPARE_AD']) if 'SPARE_AD' in row_raw.index and not pd.isna(row_raw['SPARE_AD']) else float('nan'))
        longitudinal_covariates['Hypertension'].append(
            float(row_raw['Hypertension']) if 'Hypertension' in row_raw.index and not pd.isna(row_raw['Hypertension']) else float('nan'))
        longitudinal_covariates['Diabetes'].append(
            float(row_raw['Diabetes']) if 'Diabetes' in row_raw.index and not pd.isna(row_raw['Diabetes']) else float('nan'))
        longitudinal_covariates['BAG'].append(bag_val)

assert len(samples['PTID']) == len(samples['X']) == len(samples['Y'])

if skipped:
    print(f'  Skipped {skipped} subjects (empty after filtering)')

print(f'  Total samples (rows): {len(samples["PTID"])}')
print(f'  Feature vector length: {len(samples["X"][0]) if samples["X"] else "N/A"}')

# ---------------------------------------------------------------------------
# 12. Save outputs
# ---------------------------------------------------------------------------
print('\nSaving outputs...')

samples_df = pd.DataFrame(samples)
out_samples = os.path.join(args.output_dir, 'subjectsamples_bag_accord.csv')
samples_df.to_csv(out_samples, index=False)
print(f'  Saved SubjectSamples: {out_samples}')

long_cov_df = pd.DataFrame(longitudinal_covariates)
out_cov = os.path.join(args.output_dir, 'longitudinal_covariates_bag_accord.csv')
long_cov_df.to_csv(out_cov, index=False)
print(f'  Saved longitudinal covariates: {out_cov}')

# Also save the processed (normalized) ACCORD dataframe for reference
out_data = os.path.join(args.output_dir, 'data_bag_accord.csv')
data.to_csv(out_data, index=False)
print(f'  Saved normalized ACCORD data: {out_data}')

print('\nDone.')
print(f'  Output directory : {args.output_dir}')
print(f'  SubjectSamples   : {out_samples}')
print(f'  Use this file as --data_file argument to dkgp_inference.py')
