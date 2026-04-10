'''
Enrich ACCORD_MARCH.csv with SPARE_BA and SPARE_AD scores.

Merges on MRID (format: PTID-YYYYMMDD), which is already present in all
three files. Saves the result as ACCORD_MARCH_enriched.csv so downstream
scripts (accord_data.py, analyses) can load a single file.

Columns added / overwritten:
  SPARE_BA        — from SPARE_BA_out_20260319.csv
  Age_actual      — chronological age at scan (from SPARE_BA file)
  SPARE_AD        — continuous SPARE-AD score
  SPARE_AD_binary — binary AD classification (0/1)
'''

import os
import pandas as pd

ACCORD_CSV   = 'ACCORD_MARCH.csv'
SPARE_BA_CSV = 'SPARE_BA_out_20260319.csv'
SPARE_AD_CSV = 'SPARE_AD_out_20260319.csv'
OUTPUT_CSV   = 'ACCORD_MARCH_enriched.csv'

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
print(f'Loading {ACCORD_CSV} ...')
accord = pd.read_csv(ACCORD_CSV)
print(f'  {accord.shape[0]} rows, {accord.shape[1]} columns')

print(f'Loading {SPARE_BA_CSV} ...')
spare_ba = pd.read_csv(SPARE_BA_CSV)[['MRID', 'SPARE_BA', 'Age_actual']]
print(f'  {len(spare_ba)} rows')

print(f'Loading {SPARE_AD_CSV} ...')
spare_ad = pd.read_csv(SPARE_AD_CSV)[['MRID', 'SPARE_AD', 'SPARE_AD_binary']]
print(f'  {len(spare_ad)} rows')

# ---------------------------------------------------------------------------
# Drop any existing stale SPARE columns from ACCORD before merging
# ---------------------------------------------------------------------------
drop_cols = [c for c in ['SPARE_BA', 'Age_actual', 'SPARE_AD', 'SPARE_AD_binary']
             if c in accord.columns]
if drop_cols:
    accord = accord.drop(columns=drop_cols)
    print(f'Dropped existing columns to overwrite: {drop_cols}')

# ---------------------------------------------------------------------------
# Merge on MRID (left join — keep all ACCORD rows)
# ---------------------------------------------------------------------------
accord = accord.merge(spare_ba, on='MRID', how='left')
n_ba = accord['SPARE_BA'].notna().sum()
print(f'SPARE_BA matched: {n_ba}/{len(accord)} rows '
      f'({accord["PTID.x"].nunique()} subjects)')

accord = accord.merge(spare_ad, on='MRID', how='left')
n_ad = accord['SPARE_AD'].notna().sum()
print(f'SPARE_AD matched: {n_ad}/{len(accord)} rows')

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
accord.to_csv(OUTPUT_CSV, index=False)
print(f'\nSaved: {OUTPUT_CSV}')
print(f'Final shape: {accord.shape}')
print(f'New columns: SPARE_BA, Age_actual, SPARE_AD, SPARE_AD_binary')
