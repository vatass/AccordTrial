'''
ACCORD BAG Trajectory Analysis
- Ensemble predictions across 5 folds
- Plot population-level and individual BAG trajectories 8 years ahead
- Overlay real observed BAG at measured timepoints (~0, 36, 48 months)
'''

import os
import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

parser = argparse.ArgumentParser(description='Visualize ACCORD BAG trajectories')
parser.add_argument('--inference_dir', default='inference',
                    help='Root directory containing accord_bag_fold{i}/ subdirs')
parser.add_argument('--accord_data', default='data/accord_data_bag_processed.csv',
                    help='Processed ACCORD CSV (for Sex/Age demographics)')
parser.add_argument('--norm_stats', default='data/normalization_stats.pkl',
                    help='Normalization stats pickle (to denormalize Age)')
parser.add_argument('--output_dir', default='analysis/accord_bag',
                    help='Directory to save figures and CSVs')
parser.add_argument('--n_folds', type=int, default=5)
parser.add_argument('--n_traj', type=int, default=20,
                    help='Number of individual trajectories to plot in the grid')
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load all fold predictions
# ---------------------------------------------------------------------------
fold_dfs = []
for fold in range(args.n_folds):
    path = os.path.join(args.inference_dir,
                        f'accord_bag_fold{fold}',
                        f'predictions_accord_BAG_{fold}.csv')
    if os.path.exists(path):
        df = pd.read_csv(path)
        df['fold'] = fold
        fold_dfs.append(df)
        print(f'Fold {fold}: {len(df)} rows ({df["PTID"].nunique()} subjects)')
    else:
        print(f'WARNING: {path} not found — skipping fold {fold}')

if not fold_dfs:
    raise FileNotFoundError('No fold inference results found. Run run_accord_inference.sh first.')

all_preds = pd.concat(fold_dfs, ignore_index=True)
all_preds['PTID'] = all_preds['PTID'].astype(str)

# ---------------------------------------------------------------------------
# 2. Ensemble across folds: average predictions, keep real_BAG (same per fold)
# ---------------------------------------------------------------------------
ensemble = (
    all_preds.groupby(['PTID', 'time_months'])
    .agg(
        predicted_value  = ('predicted_value', 'mean'),
        lower_bound      = ('lower_bound',     'mean'),
        upper_bound      = ('upper_bound',     'mean'),
        variance         = ('variance',        'mean'),
        interval_width   = ('interval_width',  'mean'),
        real_BAG         = ('real_BAG',        'first'),  # identical across folds
    )
    .reset_index()
)
print(f'\nEnsemble: {ensemble["PTID"].nunique()} subjects, '
      f'{ensemble["time_months"].nunique()} timepoints')

# ---------------------------------------------------------------------------
# 3. Load demographics and merge
# ---------------------------------------------------------------------------
accord_data = pd.read_csv(args.accord_data)
accord_data['PTID'] = accord_data['PTID'].astype(str)

with open(args.norm_stats, 'rb') as f:
    norm_stats = pickle.load(f)


print(norm_stats)
mean_age = norm_stats['Age']['mean']
std_age  = norm_stats['Age']['std']



# Per-subject Sex from processed CSV (Time == 0 row)
demo = (accord_data[accord_data['Time'] == 0][['PTID', 'Sex']]
        .drop_duplicates('PTID')
        .copy())

# Age from SPARE_BA file (has Age_actual per MRID = PTID-Date)
# Extract baseline age (first visit per subject)
spare_ba_path = 'SPARE_BA_out_20260319.csv'
if os.path.exists(spare_ba_path):
    sba = pd.read_csv(spare_ba_path)[['MRID', 'Age_actual']]
    sba['PTID'] = sba['MRID'].str.rsplit('-', n=1).str[0]
    age_baseline = (sba.groupby('PTID')['Age_actual']
                    .first()
                    .reset_index()
                    .rename(columns={'Age_actual': 'Age_years'}))
    demo = demo.merge(age_baseline, on='PTID', how='left')
    print(f'Age loaded from {spare_ba_path}: '
          f'{demo["Age_years"].notna().sum()}/{len(demo)} subjects matched')
else:
    demo['Age_years'] = np.nan
    print(f'WARNING: {spare_ba_path} not found — Age will be unavailable')

ensemble = ensemble.merge(demo[['PTID', 'Sex', 'Age_years']], on='PTID', how='left')
ensemble['Sex_label'] = ensemble['Sex'].map({0: 'Male', 1: 'Female'})

timepoints = sorted(ensemble['time_months'].unique())
sex_colors = {'Male': '#2196F3', 'Female': '#E91E63'}

print(f'Sex breakdown: '
      f'{(demo["Sex"] == 0).sum()} Male, '
      f'{(demo["Sex"] == 1).sum()} Female')

# Identify timepoints where real observations exist
obs_timepoints = sorted(ensemble.loc[ensemble['real_BAG'].notna(), 'time_months'].unique())
print(f'Timepoints with real BAG observations: {obs_timepoints}')

# ---------------------------------------------------------------------------
# Figure 1: Population-level trajectory + observed BAG
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 6))

# Population mean prediction ± 95% CI of the mean
pop = (ensemble.groupby('time_months')['predicted_value']
       .agg(['mean', 'std', 'count'])
       .reset_index())
pop['se']    = pop['std'] / np.sqrt(pop['count'])
pop['ci_lo'] = pop['mean'] - 1.96 * pop['se']
pop['ci_hi'] = pop['mean'] + 1.96 * pop['se']

ax.plot(pop['time_months'], pop['mean'], 'k-', lw=2.5,
        label='Population mean prediction', zorder=3)
ax.fill_between(pop['time_months'], pop['ci_lo'], pop['ci_hi'],
                alpha=0.20, color='gray', label='95% CI (population mean)')

# Mean model uncertainty band (average GP CI across subjects)
unc = ensemble.groupby('time_months')[['lower_bound', 'upper_bound']].mean().reset_index()
ax.fill_between(unc['time_months'], unc['lower_bound'], unc['upper_bound'],
                alpha=0.10, color='steelblue', label='Mean prediction interval')

# Observed BAG: mean ± 95% CI across subjects at each measured timepoint
real_obs = ensemble[ensemble['real_BAG'].notna()].copy()
if len(real_obs) > 0:
    obs_agg = (real_obs.groupby('time_months')['real_BAG']
               .agg(['mean', 'std', 'count'])
               .reset_index())
    obs_agg['se'] = obs_agg['std'] / np.sqrt(obs_agg['count'])
    ax.errorbar(obs_agg['time_months'], obs_agg['mean'],
                yerr=1.96 * obs_agg['se'],
                fmt='D', color='crimson', ms=9, lw=2, capsize=6,
                label='Observed BAG (mean ± 95% CI)', zorder=5)

ax.set_xlabel('Time (months)', fontsize=13)
ax.set_ylabel('BAG (years)', fontsize=13)
ax.set_title('ACCORD: Population BAG Trajectory — 8-Year Prediction', fontsize=14)
ax.set_xticks(timepoints)
ax.set_xticklabels([f'{t}m\n({t//12}yr)' if t > 0 else '0m\n(baseline)'
                    for t in timepoints], fontsize=9)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(args.output_dir, 'fig1_population_trajectory.png'), dpi=150)
plt.close(fig)
print('Saved fig1_population_trajectory.png')

# ---------------------------------------------------------------------------
# Figure 2: Sex-stratified population trajectories
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
for ax, sex_label in zip(axes, ['Male', 'Female']):
    sub = ensemble[ensemble['Sex_label'] == sex_label]
    if len(sub) == 0:
        ax.set_visible(False)
        continue

    pop_s = (sub.groupby('time_months')['predicted_value']
             .agg(['mean', 'std', 'count'])
             .reset_index())
    pop_s['se']    = pop_s['std'] / np.sqrt(pop_s['count'])
    pop_s['ci_lo'] = pop_s['mean'] - 1.96 * pop_s['se']
    pop_s['ci_hi'] = pop_s['mean'] + 1.96 * pop_s['se']

    c = sex_colors[sex_label]
    ax.plot(pop_s['time_months'], pop_s['mean'], '-', color=c, lw=2.5,
            label='Mean prediction')
    ax.fill_between(pop_s['time_months'], pop_s['ci_lo'], pop_s['ci_hi'],
                    alpha=0.25, color=c, label='95% CI (mean)')

    unc_s = sub.groupby('time_months')[['lower_bound', 'upper_bound']].mean().reset_index()
    ax.fill_between(unc_s['time_months'], unc_s['lower_bound'], unc_s['upper_bound'],
                    alpha=0.10, color=c, label='Mean prediction interval')

    real_s = sub[sub['real_BAG'].notna()]
    if len(real_s) > 0:
        obs_s = (real_s.groupby('time_months')['real_BAG']
                 .agg(['mean', 'std', 'count'])
                 .reset_index())
        obs_s['se'] = obs_s['std'] / np.sqrt(obs_s['count'])
        ax.errorbar(obs_s['time_months'], obs_s['mean'],
                    yerr=1.96 * obs_s['se'],
                    fmt='D', color='crimson', ms=8, lw=2, capsize=5,
                    label='Observed BAG')

    n_subj = sub['PTID'].nunique()
    ax.set_title(f'{sex_label}  (n={n_subj})', fontsize=13)
    ax.set_xlabel('Time (months)', fontsize=12)
    ax.set_xticks(timepoints)
    ax.set_xticklabels([f'{t}m' for t in timepoints], rotation=45, fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

axes[0].set_ylabel('BAG (years)', fontsize=12)
fig.suptitle('ACCORD BAG Trajectories by Sex — 8-Year Prediction', fontsize=14)
plt.tight_layout()
fig.savefig(os.path.join(args.output_dir, 'fig2_sex_stratified_trajectories.png'), dpi=150)
plt.close(fig)
print('Saved fig2_sex_stratified_trajectories.png')

# ---------------------------------------------------------------------------
# Figure 3: Individual subject trajectories (random sample)
# ---------------------------------------------------------------------------
# Prefer subjects that have at least one real observation
ptids_with_obs = ensemble.loc[ensemble['real_BAG'].notna(), 'PTID'].unique()
ptids_all      = ensemble['PTID'].unique()

rng = np.random.default_rng(42)
n_want = min(args.n_traj, len(ptids_all))
# Fill from subjects with obs first, then pad with random others
n_with = min(n_want, len(ptids_with_obs))
chosen_obs  = rng.choice(ptids_with_obs, size=n_with, replace=False)
remaining   = [p for p in ptids_all if p not in set(chosen_obs)]
n_pad       = n_want - n_with
chosen_pad  = rng.choice(remaining, size=n_pad, replace=False) if n_pad > 0 else []
sample_ptids = list(chosen_obs) + list(chosen_pad)
rng.shuffle(sample_ptids)

ncols = 4
nrows = int(np.ceil(len(sample_ptids) / ncols))
fig, axes = plt.subplots(nrows, ncols,
                         figsize=(4.5 * ncols, 3.5 * nrows),
                         sharey=False)
axes_flat = np.array(axes).flatten()

for ax, ptid in zip(axes_flat, sample_ptids):
    subj = ensemble[ensemble['PTID'] == ptid].sort_values('time_months')
    sex_label = subj['Sex_label'].iloc[0] if not subj['Sex_label'].isna().all() else 'Unknown'
    age_val   = subj['Age_years'].iloc[0]
    c = sex_colors.get(sex_label, 'gray')

    ax.plot(subj['time_months'], subj['predicted_value'], '-', color=c, lw=2,
            label='Predicted')
    ax.fill_between(subj['time_months'], subj['lower_bound'], subj['upper_bound'],
                    alpha=0.25, color=c, label='95% CI')

    real = subj[subj['real_BAG'].notna()]
    if len(real) > 0:
        ax.scatter(real['time_months'], real['real_BAG'],
                   color='crimson', s=70, zorder=5, marker='D', label='Observed')

    age_str = f'{age_val:.0f}yr' if not np.isnan(age_val) else ''
    ax.set_title(f'{sex_label}, {age_str}', fontsize=8.5)
    ax.set_xlabel('months', fontsize=8)
    ax.set_ylabel('BAG (yr)', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_xticks(timepoints[::2])
    ax.grid(True, alpha=0.3)

# Show legend once in last visible axis
handles, labels = axes_flat[0].get_legend_handles_labels()
axes_flat[min(len(sample_ptids) - 1, len(axes_flat) - 1)].legend(
    handles, labels, fontsize=7, loc='upper left')

for ax in axes_flat[len(sample_ptids):]:
    ax.set_visible(False)

fig.suptitle('ACCORD Individual BAG Trajectories (sample)', fontsize=13)
plt.tight_layout()
fig.savefig(os.path.join(args.output_dir, 'fig3_individual_trajectories.png'), dpi=150)
plt.close(fig)
print('Saved fig3_individual_trajectories.png')

# ---------------------------------------------------------------------------
# Figure 4: Predicted vs Observed at matched timepoints (scatter)
# ---------------------------------------------------------------------------
matched = ensemble[ensemble['real_BAG'].notna()].copy()
obs_tps = sorted(matched['time_months'].unique())
if len(matched) >= 10 and len(obs_tps) > 0:
    ncols_s = len(obs_tps)
    fig, axes = plt.subplots(1, ncols_s, figsize=(5 * ncols_s, 5), squeeze=False)
    for ax, tp in zip(axes[0], obs_tps):
        tp_data = matched[matched['time_months'] == tp].dropna(subset=['real_BAG', 'predicted_value'])
        mae = np.mean(np.abs(tp_data['real_BAG'] - tp_data['predicted_value']))
        if len(tp_data) >= 2:
            r, pval = stats.pearsonr(tp_data['real_BAG'], tp_data['predicted_value'])
            corr_str = f'r={r:.3f}'
        else:
            corr_str = 'n<2'

        ax.scatter(tp_data['real_BAG'], tp_data['predicted_value'],
                   alpha=0.5, s=30, color='steelblue', edgecolors='none')
        lo = min(tp_data['real_BAG'].min(), tp_data['predicted_value'].min()) - 2
        hi = max(tp_data['real_BAG'].max(), tp_data['predicted_value'].max()) + 2
        ax.plot([lo, hi], [lo, hi], 'r--', lw=1.5, label='Identity')
        ax.set_xlabel('Observed BAG (years)', fontsize=11)
        ax.set_ylabel('Predicted BAG (years)', fontsize=11)
        ax.set_title(f't={tp}m\n{corr_str}, MAE={mae:.2f}yr\n(n={len(tp_data)})',
                     fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    fig.suptitle('Predicted vs Observed BAG at Measured Timepoints', fontsize=13)
    plt.tight_layout()
    fig.savefig(os.path.join(args.output_dir, 'fig4_predicted_vs_observed.png'), dpi=150)
    plt.close(fig)
    print('Saved fig4_predicted_vs_observed.png')
else:
    print(f'Fewer than 10 matched observations ({len(matched)}) — skipping fig4')

# ---------------------------------------------------------------------------
# Figure 5: Violin of predicted BAG distribution at each timepoint
#           with real observations overlaid as red dots
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(13, 6))
viol_data = [ensemble[ensemble['time_months'] == tp]['predicted_value'].values
             for tp in timepoints]
positions = list(range(len(timepoints)))
vp = ax.violinplot(viol_data, positions=positions, showmedians=True, showextrema=False)
for body in vp['bodies']:
    body.set_facecolor('steelblue')
    body.set_alpha(0.55)
vp['cmedians'].set_color('navy')
vp['cmedians'].set_linewidth(2)

# Real observations
for j, tp in enumerate(timepoints):
    real = ensemble.loc[(ensemble['time_months'] == tp) & ensemble['real_BAG'].notna(), 'real_BAG']
    if len(real) > 0:
        jitter = rng.uniform(-0.15, 0.15, size=len(real))
        ax.scatter(j + jitter, real, color='crimson', s=18, alpha=0.65, zorder=5,
                   label='Observed' if j == 0 else '')

ax.set_xticks(positions)
ax.set_xticklabels([f'{tp}m\n({tp//12}yr)' if tp > 0 else '0m\n(baseline)'
                    for tp in timepoints], fontsize=9)
ax.set_xlabel('Time (months)', fontsize=12)
ax.set_ylabel('BAG (years)', fontsize=12)
ax.set_title('ACCORD Predicted BAG Distribution Over Time', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
fig.savefig(os.path.join(args.output_dir, 'fig5_bag_distribution_over_time.png'), dpi=150)
plt.close(fig)
print('Saved fig5_bag_distribution_over_time.png')

# ---------------------------------------------------------------------------
# Save ensemble CSV and print summary
# ---------------------------------------------------------------------------
out_csv = os.path.join(args.output_dir, 'accord_bag_ensemble_predictions.csv')
ensemble.to_csv(out_csv, index=False)
print(f'\nSaved ensemble predictions: {out_csv}')

print('\n=== Summary ===')
print(f'Subjects: {ensemble["PTID"].nunique()}')
print(f'Folds ensembled: {len(fold_dfs)}')
print(f'Timepoints with real BAG: {obs_timepoints}')

baseline = ensemble[ensemble['time_months'] == 0]
final    = ensemble[ensemble['time_months'] == max(timepoints)]
print(f'\nBaseline (t=0): mean BAG = {baseline["predicted_value"].mean():.2f} '
      f'± {baseline["predicted_value"].std():.2f} yr')
print(f'Year 8  (t=96): mean BAG = {final["predicted_value"].mean():.2f} '
      f'± {final["predicted_value"].std():.2f} yr')
change = final['predicted_value'].mean() - baseline['predicted_value'].mean()
print(f'Mean predicted change over 8 years: {change:+.2f} yr')

if len(matched) > 0:
    mae_all = np.mean(np.abs(matched['real_BAG'] - matched['predicted_value']))
    print(f'\nOverall MAE at observed timepoints: {mae_all:.2f} yr '
          f'({len(matched)} matched observations)')
