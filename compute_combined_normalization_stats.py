'''
Compute Combined Normalization Statistics
=========================================
Pools the 11 iSTAGING longitudinal studies + ACCORD baseline visits to
derive normalization statistics for MUSE volumes, Age, BAG, and SPARE_BA.

Run this BEFORE longitudinal_data.py so that the training data is
normalized with stats that represent both populations.

Prerequisites:
  - data/data_bag_allstudies.csv         (iSTAGING data, already processed)
  - data/145_MUSE_allstudies_mean_std_hmuse.pkl  (existing iSTAGING-only stats)
  - data/normalization_stats.pkl          (existing iSTAGING-only stats)
  - ACCORD_MARCH_enriched.csv             (run enrich_accord_data.py first)

Overwrites:
  - data/145_MUSE_allstudies_mean_std_hmuse.pkl
  - data/normalization_stats.pkl

Workflow:
  python compute_combined_normalization_stats.py
  python longitudinal_data.py     ← normalizes training data with combined stats
  python accord_data.py           ← normalizes ACCORD with the same combined stats
'''

import os
import pickle
import numpy as np
import pandas as pd

DATA_DIR      = './data/'
ACCORD_CSV    = 'ACCORD_MARCH_enriched.csv'
ISTAGING_CSV  = os.path.join(DATA_DIR, 'data_bag_allstudies.csv')
MUSE_PKL      = os.path.join(DATA_DIR, '145_MUSE_allstudies_mean_std_hmuse.pkl')
NORM_PKL      = os.path.join(DATA_DIR, 'normalization_stats.pkl')

# ---------------------------------------------------------------------------
# 1. Load existing (iSTAGING-only) stats — needed to denormalize the CSV
# ---------------------------------------------------------------------------
print('Loading existing iSTAGING-only stats for denormalization...')
with open(MUSE_PKL, 'rb') as f:
    old_muse = pickle.load(f)
with open(NORM_PKL, 'rb') as f:
    old_norm = pickle.load(f)

old_mean_muse = np.array(old_muse['mean'])
old_std_muse  = np.array(old_muse['std'])

# ---------------------------------------------------------------------------
# 2. Load iSTAGING normalized CSV and denormalize to raw values
# ---------------------------------------------------------------------------
print(f'Loading {ISTAGING_CSV} ...')
istaging = pd.read_csv(ISTAGING_CSV)
print(f'  {istaging["PTID"].nunique()} subjects, {len(istaging)} rows')

muse_cols = [c for c in istaging.columns if c.startswith('MUSE_')]
print(f'  {len(muse_cols)} MUSE columns')

# Denormalize MUSE
istaging_muse_raw = pd.DataFrame(index=istaging.index, columns=muse_cols, dtype=float)
for i, col in enumerate(muse_cols):
    istaging_muse_raw[col] = istaging[col] * old_std_muse[i] + old_mean_muse[i]

# Denormalize Age, BAG, SPARE_BA
istaging_age_raw    = istaging['Age']     * old_norm['Age']['std']     + old_norm['Age']['mean']
istaging_bag_raw    = istaging['BAG']     * old_norm['BAG']['std']     + old_norm['BAG']['mean']
istaging_spareba_raw = istaging['SPARE_BA'] * old_norm['SPARE_BA']['std'] + old_norm['SPARE_BA']['mean']

# ---------------------------------------------------------------------------
# 3. Load ACCORD enriched CSV — keep baseline only (first visit per subject)
# ---------------------------------------------------------------------------
print(f'\nLoading {ACCORD_CSV} ...')
accord = pd.read_csv(ACCORD_CSV)

# Rename X4–X207 → MUSE_Volume_*
rename_map = {f'X{i}': f'MUSE_Volume_{i}' for i in range(4, 208) if f'X{i}' in accord.columns}
accord = accord.rename(columns=rename_map)

# Parse dates (stored as integer YYYYMMDD)
accord['Date.x'] = pd.to_datetime(accord['Date.x'].astype(str), format='%Y%m%d')
accord = accord.sort_values(['PTID.x', 'Date.x'])
accord_bl = accord.groupby('PTID.x', as_index=False).first()

# Compute raw BAG
accord_bl['BAG'] = accord_bl['SPARE_BA'] - accord_bl['Age.x']

# Drop subjects missing SPARE_BA (can't compute BAG)
n_before = len(accord_bl)
accord_bl = accord_bl.dropna(subset=['SPARE_BA', 'Age.x', 'BAG'])
print(f'  {len(accord_bl)} baseline subjects '
      f'({n_before - len(accord_bl)} dropped — missing SPARE_BA)')

# ---------------------------------------------------------------------------
# 4. Compute combined statistics
# ---------------------------------------------------------------------------
print('\nComputing combined statistics (iSTAGING + ACCORD baseline)...')

# MUSE volumes
new_mean_muse, new_std_muse = [], []
n_pooled = 0
for i, col in enumerate(muse_cols):
    ist_vals = istaging_muse_raw[col].dropna().values
    if col in accord_bl.columns:
        acc_vals = accord_bl[col].dropna().values
        combined = np.concatenate([ist_vals, acc_vals])
        n_pooled += 1
    else:
        combined = ist_vals
    new_mean_muse.append(float(combined.mean()))
    new_std_muse.append(float(combined.std()))
print(f'  MUSE columns pooled with ACCORD data: {n_pooled}/{len(muse_cols)}')

# Age
age_comb = np.concatenate([istaging_age_raw.dropna().values,
                            accord_bl['Age.x'].dropna().values])
new_mean_age, new_std_age = float(age_comb.mean()), float(age_comb.std())

# BAG
bag_comb = np.concatenate([istaging_bag_raw.dropna().values,
                            accord_bl['BAG'].dropna().values])
new_mean_bag, new_std_bag = float(bag_comb.mean()), float(bag_comb.std())

# SPARE_BA
sba_comb = np.concatenate([istaging_spareba_raw.dropna().values,
                            accord_bl['SPARE_BA'].dropna().values])
new_mean_sba, new_std_sba = float(sba_comb.mean()), float(sba_comb.std())

# ---------------------------------------------------------------------------
# 5. Print comparison
# ---------------------------------------------------------------------------
print(f'\n{"Statistic":<22} {"iSTAGING-only":>20}   {"Combined":>20}')
print('-' * 65)
def _fmt(m, s): return f'{m:.3f} ± {s:.3f}'
print(f'{"Age  (mean ± std)":<22} {_fmt(old_norm["Age"]["mean"], old_norm["Age"]["std"]):>20}   {_fmt(new_mean_age, new_std_age):>20}')
print(f'{"BAG  (mean ± std)":<22} {_fmt(old_norm["BAG"]["mean"], old_norm["BAG"]["std"]):>20}   {_fmt(new_mean_bag, new_std_bag):>20}')
print(f'{"SPARE_BA (mean ± std)":<22} {_fmt(old_norm["SPARE_BA"]["mean"], old_norm["SPARE_BA"]["std"]):>20}   {_fmt(new_mean_sba, new_std_sba):>20}')
print(f'{"iSTAGING subjects":<22} {istaging["PTID"].nunique():>20}')
print(f'{"ACCORD baseline subj.":<22} {len(accord_bl):>20}')

# ---------------------------------------------------------------------------
# 6. Save updated stats
# ---------------------------------------------------------------------------
new_muse_stats = {'mean': new_mean_muse, 'std': new_std_muse}
with open(MUSE_PKL, 'wb') as f:
    pickle.dump(new_muse_stats, f)
print(f'\nSaved: {MUSE_PKL}')

new_norm_stats = {
    'Age':      {'mean': new_mean_age, 'std': new_std_age},
    'SPARE_BA': {'mean': new_mean_sba, 'std': new_std_sba},
    'BAG':      {'mean': new_mean_bag, 'std': new_std_bag},
}
with open(NORM_PKL, 'wb') as f:
    pickle.dump(new_norm_stats, f)
print(f'Saved: {NORM_PKL}')

print('\nDone. Run longitudinal_data.py next to retrain with combined normalization.')
