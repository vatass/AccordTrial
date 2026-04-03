'''
ACCORD - BAG Trajectory Visualization
Plots Brain Age Gap (BAG) trajectories across time for CN subjects.
Loads the processed CSV and denormalizes BAG using saved normalization stats.
'''

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
data_dir  = 'data/'
out_dir   = 'data/plots/'
os.makedirs(out_dir, exist_ok=True)

RANDOM_SEED     = 42
N_SAMPLE_TRAJS  = 80   # subjects shown in spaghetti plot
MIN_TIMEPOINTS  = 3    # minimum visits for a subject to appear in trajectories

# ---------------------------------------------------------------------------
# Load data & denormalize BAG
# ---------------------------------------------------------------------------
df = pd.read_csv(data_dir + 'data_bag_allstudies.csv', low_memory=False)
df['PTID'] = df['PTID'].astype(str)
df = df.sort_values(['PTID', 'Time'])

# Denormalize if normalization stats exist, otherwise treat columns as already raw
try:
    with open(data_dir + 'normalization_stats.pkl', 'rb') as f:
        norm_stats = pickle.load(f)
    df['BAG_raw'] = df['BAG'] * norm_stats['BAG']['std'] + norm_stats['BAG']['mean']
    df['Age_raw'] = df['Age'] * norm_stats['Age']['std'] + norm_stats['Age']['mean']
    print('Normalization stats loaded — values denormalized.')
except FileNotFoundError:
    print('normalization_stats.pkl not found — assuming columns are already in raw units.')
    df['BAG_raw'] = df['BAG']
    df['Age_raw'] = df['Age']

# Baseline age: first observed Age per subject
df['Baseline_Age'] = df.groupby('PTID')['Age_raw'].transform('first')

print(f'Loaded {df["PTID"].nunique()} subjects, {len(df)} observations')
print(f'BAG (raw) — mean: {df["BAG_raw"].mean():.2f}, std: {df["BAG_raw"].std():.2f}')

studies  = sorted(df['Study'].unique())
n_studies = len(studies)
study_colors = {s: cm.tab10(i / max(n_studies - 1, 1)) for i, s in enumerate(studies)}

# ---------------------------------------------------------------------------
# Helper: filter subjects with enough timepoints
# ---------------------------------------------------------------------------
def subjects_with_min_visits(data, n=MIN_TIMEPOINTS):
    counts = data.groupby('PTID')['Time'].count()
    return counts[counts >= n].index.tolist()

# ---------------------------------------------------------------------------
# Figure 1 — Spaghetti plot: individual BAG trajectories (random sample)
# ---------------------------------------------------------------------------
eligible = subjects_with_min_visits(df)
rng = np.random.default_rng(RANDOM_SEED)
sample_ids = rng.choice(eligible, size=min(N_SAMPLE_TRAJS, len(eligible)), replace=False)

fig, ax = plt.subplots(figsize=(12, 6))
for ptid in sample_ids:
    sub = df[df['PTID'] == ptid].sort_values('Time')
    study = sub['Study'].iloc[0]
    ax.plot(sub['Time'], sub['BAG_raw'], color=study_colors[study],
            alpha=0.35, linewidth=0.9)

# Legend proxies
for s in studies:
    ax.plot([], [], color=study_colors[s], label=s, linewidth=2)

ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
ax.set_xlabel('Time from Baseline (months)', fontsize=13)
ax.set_ylabel('BAG (years)', fontsize=13)
ax.set_title(f'Individual BAG Trajectories — CN subjects (n={len(sample_ids)} sample)', fontsize=13)
ax.legend(title='Study', fontsize=9, title_fontsize=9, loc='upper left',
          bbox_to_anchor=(1.01, 1), borderaxespad=0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(out_dir + 'bag_spaghetti.svg', bbox_inches='tight')
plt.savefig(out_dir + 'bag_spaghetti.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: bag_spaghetti')

# ---------------------------------------------------------------------------
# Figure 2 — Mean ± std BAG trajectory per study
# ---------------------------------------------------------------------------
# Bin Time into yearly windows for smoother averaging
df['Time_year'] = (df['Time'] / 12).round().astype(int) * 12  # round to nearest year in months

fig, ax = plt.subplots(figsize=(11, 6))
for study in studies:
    sdf = df[df['Study'] == study]
    grouped = sdf.groupby('Time_year')['BAG_raw'].agg(['mean', 'std', 'count']).reset_index()
    grouped = grouped[grouped['count'] >= 5]   # only bins with enough subjects
    if grouped.empty:
        continue
    color = study_colors[study]
    ax.plot(grouped['Time_year'], grouped['mean'], color=color, label=study, linewidth=2)
    ax.fill_between(grouped['Time_year'],
                    grouped['mean'] - grouped['std'],
                    grouped['mean'] + grouped['std'],
                    color=color, alpha=0.15)

ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
ax.set_xlabel('Time from Baseline (months)', fontsize=13)
ax.set_ylabel('BAG (years)', fontsize=13)
ax.set_title('Mean ± SD BAG Trajectory by Study — CN subjects', fontsize=13)
ax.legend(title='Study', fontsize=9, title_fontsize=9, loc='upper left',
          bbox_to_anchor=(1.01, 1), borderaxespad=0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(out_dir + 'bag_mean_std_by_study.svg', bbox_inches='tight')
plt.savefig(out_dir + 'bag_mean_std_by_study.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: bag_mean_std_by_study')

# ---------------------------------------------------------------------------
# Figure 3 — Baseline BAG distribution (histogram per study)
# ---------------------------------------------------------------------------
baseline = df[df['Time'] == 0].copy()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Overall
axes[0].hist(baseline['BAG_raw'], bins=40, color='steelblue', edgecolor='white', alpha=0.85)
axes[0].axvline(baseline['BAG_raw'].mean(), color='red', linestyle='--', linewidth=1.5,
                label=f'Mean = {baseline["BAG_raw"].mean():.1f} yr')
axes[0].set_xlabel('BAG at Baseline (years)', fontsize=12)
axes[0].set_ylabel('Count', fontsize=12)
axes[0].set_title('Baseline BAG Distribution — All Studies', fontsize=12)
axes[0].legend(fontsize=10)
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)

# Per study (overlaid)
for study in studies:
    sub = baseline[baseline['Study'] == study]['BAG_raw']
    if len(sub) < 5:
        continue
    axes[1].hist(sub, bins=30, alpha=0.45, label=study, color=study_colors[study], edgecolor='none')

axes[1].set_xlabel('BAG at Baseline (years)', fontsize=12)
axes[1].set_ylabel('Count', fontsize=12)
axes[1].set_title('Baseline BAG Distribution by Study', fontsize=12)
axes[1].legend(title='Study', fontsize=8, title_fontsize=9)
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(out_dir + 'bag_baseline_distribution.svg', bbox_inches='tight')
plt.savefig(out_dir + 'bag_baseline_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: bag_baseline_distribution')

# ---------------------------------------------------------------------------
# Figure 4 — BAG change from baseline per subject (longitudinal delta)
# ---------------------------------------------------------------------------
df_sorted = df.sort_values(['PTID', 'Time'])
baseline_bag = df[df['Time'] == 0][['PTID', 'BAG_raw']].rename(columns={'BAG_raw': 'BAG_baseline'})
df_delta = df_sorted.merge(baseline_bag, on='PTID', how='left')
df_delta['ΔBAG'] = df_delta['BAG_raw'] - df_delta['BAG_baseline']

fig, ax = plt.subplots(figsize=(11, 5))
for study in studies:
    sdf = df_delta[df_delta['Study'] == study]
    grouped = sdf.groupby('Time_year')['ΔBAG'].agg(['mean', 'std', 'count']).reset_index()
    grouped = grouped[grouped['count'] >= 5]
    if grouped.empty:
        continue
    color = study_colors[study]
    ax.plot(grouped['Time_year'], grouped['mean'], color=color, label=study, linewidth=2)
    ax.fill_between(grouped['Time_year'],
                    grouped['mean'] - grouped['std'],
                    grouped['mean'] + grouped['std'],
                    color=color, alpha=0.15)

ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.6)
ax.set_xlabel('Time from Baseline (months)', fontsize=13)
ax.set_ylabel('ΔBAG from Baseline (years)', fontsize=13)
ax.set_title('Mean ± SD Change in BAG from Baseline — CN subjects', fontsize=13)
ax.legend(title='Study', fontsize=9, title_fontsize=9, loc='upper left',
          bbox_to_anchor=(1.01, 1), borderaxespad=0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(out_dir + 'bag_delta_from_baseline.svg', bbox_inches='tight')
plt.savefig(out_dir + 'bag_delta_from_baseline.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: bag_delta_from_baseline')

# ---------------------------------------------------------------------------
# Age-group setup
# ---------------------------------------------------------------------------
age_bins   = [0,  60,  70,  80,  999]
age_labels = ['<60', '60–70', '70–80', '≥80']
age_colors = {'<60': '#2166ac', '60–70': '#4dac26', '70–80': '#d6604d', '≥80': '#762a83'}

df['Age_group'] = pd.cut(df['Baseline_Age'], bins=age_bins, labels=age_labels, right=False)

# Per-subject linear BAG slope (yr / month)
def compute_slopes(data):
    rows = []
    for ptid, grp in data.groupby('PTID'):
        grp = grp.sort_values('Time')
        if len(grp) < 2 or grp['Time'].std() == 0:
            continue
        slope = np.polyfit(grp['Time'].values.astype(float), grp['BAG_raw'].values, 1)[0]
        rows.append({
            'PTID':         ptid,
            'slope_yr_mo':  slope,
            'slope_yr_yr':  slope * 12,
            'Age_group':    grp['Age_group'].iloc[0],
            'Baseline_Age': grp['Baseline_Age'].iloc[0],
            'Study':        grp['Study'].iloc[0],
        })
    return pd.DataFrame(rows)

slopes_df = compute_slopes(df)
print(f'\nComputed BAG slopes for {len(slopes_df)} subjects')

# ---------------------------------------------------------------------------
# Figure 5 — Mean ± SD BAG trajectory by age group
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 6))
for ag in age_labels:
    sdf = df[df['Age_group'] == ag]
    grouped = sdf.groupby('Time_year')['BAG_raw'].agg(['mean', 'std', 'count']).reset_index()
    grouped = grouped[grouped['count'] >= 5]
    if grouped.empty:
        continue
    color = age_colors[ag]
    n = sdf['PTID'].nunique()
    ax.plot(grouped['Time_year'], grouped['mean'], color=color, label=f'{ag} (n={n})', linewidth=2)
    ax.fill_between(grouped['Time_year'],
                    grouped['mean'] - grouped['std'],
                    grouped['mean'] + grouped['std'],
                    color=color, alpha=0.15)

ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
ax.set_xlabel('Time from Baseline (months)', fontsize=13)
ax.set_ylabel('BAG (years)', fontsize=13)
ax.set_title('Mean ± SD BAG Trajectory by Baseline Age Group — CN subjects', fontsize=13)
ax.legend(title='Age group', fontsize=10, title_fontsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(out_dir + 'bag_trajectory_by_age_group.svg', bbox_inches='tight')
plt.savefig(out_dir + 'bag_trajectory_by_age_group.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: bag_trajectory_by_age_group')

# ---------------------------------------------------------------------------
# Figure 6 — Spaghetti by age group (random sample per group)
# ---------------------------------------------------------------------------
N_PER_GROUP = 30
fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
axes = axes.flatten()

for idx, ag in enumerate(age_labels):
    ax = axes[idx]
    group_ids = df[df['Age_group'] == ag]['PTID'].unique()
    eligible_ids = [p for p in group_ids if df[df['PTID'] == p]['Time'].count() >= MIN_TIMEPOINTS]
    sample = rng.choice(eligible_ids, size=min(N_PER_GROUP, len(eligible_ids)), replace=False)
    color = age_colors[ag]
    for ptid in sample:
        sub = df[df['PTID'] == ptid].sort_values('Time')
        ax.plot(sub['Time'], sub['BAG_raw'], color=color, alpha=0.4, linewidth=0.9)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.set_title(f'Baseline Age {ag} (n={len(group_ids)})', fontsize=12)
    ax.set_xlabel('Time from Baseline (months)', fontsize=10)
    ax.set_ylabel('BAG (years)', fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.suptitle('Individual BAG Trajectories by Baseline Age Group — CN subjects', fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(out_dir + 'bag_spaghetti_by_age_group.svg', bbox_inches='tight')
plt.savefig(out_dir + 'bag_spaghetti_by_age_group.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: bag_spaghetti_by_age_group')

# ---------------------------------------------------------------------------
# Figure 7 — Boxplot of individual BAG slopes by age group
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 6))
data_to_plot = [slopes_df[slopes_df['Age_group'] == ag]['slope_yr_yr'].dropna().values
                for ag in age_labels]
bp = ax.boxplot(data_to_plot, labels=age_labels, patch_artist=True,
                medianprops=dict(color='black', linewidth=2),
                flierprops=dict(marker='o', markersize=3, alpha=0.4))

for patch, ag in zip(bp['boxes'], age_labels):
    patch.set_facecolor(age_colors[ag])
    patch.set_alpha(0.7)

ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.6)
ax.set_xlabel('Baseline Age Group', fontsize=13)
ax.set_ylabel('BAG Slope (years / year)', fontsize=13)
ax.set_title('Individual BAG Slopes by Baseline Age Group — CN subjects', fontsize=13)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Annotate with n and mean
for i, ag in enumerate(age_labels):
    grp = slopes_df[slopes_df['Age_group'] == ag]['slope_yr_yr'].dropna()
    ax.text(i + 1, ax.get_ylim()[0], f'n={len(grp)}\n{grp.mean():.3f}±{grp.std():.3f}',
            ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(out_dir + 'bag_slopes_boxplot_by_age_group.svg', bbox_inches='tight')
plt.savefig(out_dir + 'bag_slopes_boxplot_by_age_group.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: bag_slopes_boxplot_by_age_group')

# Save slopes table
slopes_df.to_csv(out_dir + 'individual_bag_slopes_by_age_group.csv', index=False)
print('Saved: individual_bag_slopes_by_age_group.csv')

print(f'\nAll plots saved to {out_dir}')
