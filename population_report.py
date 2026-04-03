'''
ACCORD - Population Statistics Report
Generates a manuscript-ready summary table from the processed BAG dataset.
All continuous variables are reported in their original (denormalized) units.
'''

import os
import pickle
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
data_dir = 'data/'
os.makedirs(data_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# Load data & denormalize
# ---------------------------------------------------------------------------
df = pd.read_csv(data_dir + 'data_bag_allstudies.csv', low_memory=False)
df['PTID'] = df['PTID'].astype(str)

# Denormalize if normalization stats exist, otherwise treat columns as already raw
try:
    with open(data_dir + 'normalization_stats.pkl', 'rb') as f:
        norm_stats = pickle.load(f)
    df['Age_raw']      = df['Age']      * norm_stats['Age']['std']      + norm_stats['Age']['mean']
    df['SPARE_BA_raw'] = df['SPARE_BA'] * norm_stats['SPARE_BA']['std'] + norm_stats['SPARE_BA']['mean']
    df['BAG_raw']      = df['BAG']      * norm_stats['BAG']['std']      + norm_stats['BAG']['mean']
    print('Normalization stats loaded — values denormalized.')
except FileNotFoundError:
    print('normalization_stats.pkl not found — assuming columns are already in raw units.')
    df['Age_raw']      = df['Age']
    df['SPARE_BA_raw'] = df['SPARE_BA']
    df['BAG_raw']      = df['BAG']

# Baseline age: first observed Age per subject (sorted by Time)
df = df.sort_values(['PTID', 'Time'])
df['Baseline_Age'] = df.groupby('PTID')['Age_raw'].transform('first')

# Baseline rows only (one row per subject)
baseline = df[df['Time'] == 0].copy()
baseline = baseline.drop_duplicates(subset='PTID')

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def mean_std(series):
    return f'{series.mean():.2f} ± {series.std():.2f}'

def median_iqr(series):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    return f'{series.median():.2f} [{q1:.2f}–{q3:.2f}]'

def pct(count, total):
    return f'{count} ({100 * count / total:.1f}%)'

def visits_per_subject(data):
    return data.groupby('PTID')['Time'].count()

def followup_months(data):
    return data.groupby('PTID')['Time'].max()

# ---------------------------------------------------------------------------
# Build summary — overall + per study
# ---------------------------------------------------------------------------
def summarize(data, label='Overall'):
    base = data[data['Time'] == 0].drop_duplicates(subset='PTID')
    n_subj  = base['PTID'].nunique()
    n_scans = len(data)

    visits = visits_per_subject(data)
    fup    = followup_months(data)

    # Sex (0=M, 1=F after encoding)
    n_female = (base['Sex'] == 1).sum()
    n_male   = (base['Sex'] == 0).sum()

    # Education (binarized: 1 = >16 yrs)
    n_highedu = (base['Education_Years'] == 1).sum()

    # APOE4
    n_noncarrier  = (base['APOE4_Alleles'] == 0).sum()
    n_hetero      = (base['APOE4_Alleles'] == 1).sum()
    n_homo        = (base['APOE4_Alleles'] == 2).sum()

    row = {
        'Study':                        label,
        'N subjects':                   n_subj,
        'N scans':                      n_scans,
        'Visits/subject (mean±std)':    mean_std(visits),
        'Follow-up months (mean±std)':  mean_std(fup),
        'Baseline Age (mean±std)':      mean_std(base['Age_raw']),
        'Age range':                    f'{base["Age_raw"].min():.1f}–{base["Age_raw"].max():.1f}',
        'Female':                       pct(n_female, n_subj),
        'Male':                         pct(n_male,   n_subj),
        'Education > 16 yrs':           pct(n_highedu, n_subj),
        'APOE4 non-carrier':            pct(n_noncarrier, n_subj),
        'APOE4 heterozygous':           pct(n_hetero,     n_subj),
        'APOE4 homozygous':             pct(n_homo,       n_subj),
        'Baseline BAG (mean±std)':      mean_std(base['BAG_raw']),
        'BAG range':                    f'{base["BAG_raw"].min():.1f}–{base["BAG_raw"].max():.1f}',
        'BAG median [IQR]':             median_iqr(base['BAG_raw']),
    }
    return row

rows = [summarize(df, label='Overall')]
for study in sorted(df['Study'].unique()):
    rows.append(summarize(df[df['Study'] == study], label=study))

report = pd.DataFrame(rows).set_index('Study')

# ---------------------------------------------------------------------------
# Print to console
# ---------------------------------------------------------------------------
print('\n' + '=' * 80)
print('POPULATION STATISTICS REPORT  —  CN subjects, longitudinal, all studies')
print('=' * 80)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.max_colwidth', 30)
print(report.T.to_string())
print('=' * 80)

# ---------------------------------------------------------------------------
# Save CSV report
# ---------------------------------------------------------------------------
report_path = data_dir + 'population_report_bag.csv'
report.to_csv(report_path)
print(f'\nReport saved: {report_path}')

# ---------------------------------------------------------------------------
# Per-study BAG trajectory summary (mean slope via linear fit)
# ---------------------------------------------------------------------------
print('\n--- BAG Trajectory Summary (linear slope per study) ---')
slope_rows = []
for study in sorted(df['Study'].unique()):
    sdf = df[df['Study'] == study].copy()
    slopes = []
    for ptid, grp in sdf.groupby('PTID'):
        grp = grp.sort_values('Time')
        if len(grp) < 2:
            continue
        # simple OLS slope: ΔBAG per month
        x = grp['Time'].values.astype(float)
        y = grp['BAG_raw'].values
        if x.std() == 0:
            continue
        slope = np.polyfit(x, y, 1)[0]
        slopes.append(slope)
    if slopes:
        slope_rows.append({
            'Study':                     study,
            'N subjects (slope)':        len(slopes),
            'BAG slope yr/month (mean)': f'{np.mean(slopes):.4f}',
            'BAG slope yr/month (std)':  f'{np.std(slopes):.4f}',
            'BAG slope yr/year (mean)':  f'{np.mean(slopes)*12:.4f}',
        })
        print(f'  {study}: {np.mean(slopes)*12:.4f} ± {np.std(slopes)*12:.4f} yr/year  '
              f'(n={len(slopes)})')

slope_df = pd.DataFrame(slope_rows).set_index('Study')
slope_path = data_dir + 'bag_slope_report.csv'
slope_df.to_csv(slope_path)
print(f'\nSlope report saved: {slope_path}')

# ---------------------------------------------------------------------------
# BAG slope by baseline age group
# ---------------------------------------------------------------------------
age_bins   = [0,  60,  70,  80,  999]
age_labels = ['<60', '60–70', '70–80', '≥80']
df['Age_group'] = pd.cut(df['Baseline_Age'], bins=age_bins, labels=age_labels, right=False)

# Compute per-subject linear BAG slope
subject_slopes = []
for ptid, grp in df.groupby('PTID'):
    grp = grp.sort_values('Time')
    if len(grp) < 2 or grp['Time'].std() == 0:
        continue
    slope = np.polyfit(grp['Time'].values.astype(float), grp['BAG_raw'].values, 1)[0]
    subject_slopes.append({
        'PTID':         ptid,
        'slope_yr_yr':  slope * 12,
        'Age_group':    grp['Age_group'].iloc[0],
        'Baseline_Age': grp['Baseline_Age'].iloc[0],
    })
slopes_df = pd.DataFrame(subject_slopes)

print('\n--- BAG Slope by Baseline Age Group ---')
age_group_rows = []
for ag in age_labels:
    grp = slopes_df[slopes_df['Age_group'] == ag]['slope_yr_yr'].dropna()
    baseline_grp = df[df['Age_group'] == ag].drop_duplicates('PTID')
    row = {
        'Age_group':                   ag,
        'N subjects':                  len(grp),
        'Baseline Age (mean±std)':     mean_std(baseline_grp['Age_raw']),
        'Baseline BAG (mean±std)':     mean_std(baseline_grp['BAG_raw']),
        'BAG slope yr/yr (mean±std)':  f'{grp.mean():.4f} ± {grp.std():.4f}',
        'BAG slope yr/yr (median)':    f'{grp.median():.4f}',
        'BAG slope yr/yr (min–max)':   f'{grp.min():.4f}–{grp.max():.4f}',
    }
    age_group_rows.append(row)
    print(f'  {ag}: slope={grp.mean():.4f} ± {grp.std():.4f} yr/yr  (n={len(grp)}, '
          f'baseline age={baseline_grp["Age_raw"].mean():.1f}±{baseline_grp["Age_raw"].std():.1f})')

age_group_df = pd.DataFrame(age_group_rows).set_index('Age_group')
age_group_path = data_dir + 'bag_slope_by_age_group.csv'
age_group_df.to_csv(age_group_path)
print(f'\nAge-group slope report saved: {age_group_path}')
