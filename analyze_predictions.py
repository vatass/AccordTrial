'''
Prediction Trajectory Analysis and Demographic Error Assessment

Loads per-sample predictions produced by dkgp_training.py and:
  1. Plots predicted trajectories with 95% confidence bounds vs ground truth
  2. Scatter plot: predicted vs ground truth
  3. Absolute-error histograms (per observation and per subject)
  4. Prediction error stratified by Sex (violin, bar, Mann-Whitney U test)
  5. Prediction error stratified by Age group (violin, bar, Kruskal-Wallis test)
  6. Sex × Age group MAE / RMSE heatmaps
  7. Saves CSV summary tables for each demographic breakdown

Usage:
    python analyze_predictions.py \
        --predictions_file models/predictions_BAG_0_0.csv \
        --covariates_file  data/longitudinal_covariates_bag_allstudies.csv \
        --output_dir       analysis/fold0 \
        --biomarker_name   BAG \
        [--normalization_stats_file data/normalization_stats.pkl] \
        [--n_traj_subjects 20] \
        [--min_timepoints  2]
'''

import os
import argparse
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import r2_score

# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description='Analyze DKGP predictions: trajectories and demographic error assessment'
)
parser.add_argument('--predictions_file', required=True,
                    help='CSV produced by dkgp_training.py with per-sample predictions')
parser.add_argument('--covariates_file',
                    default='./data/longitudinal_covariates_bag_allstudies.csv',
                    help='Longitudinal covariates CSV (default: ./data/longitudinal_covariates_bag_allstudies.csv)')
parser.add_argument('--output_dir', default='./prediction_analysis',
                    help='Directory to save plots and tables')
parser.add_argument('--biomarker_name', default='Biomarker',
                    help='Biomarker name used in plot titles and file names')
parser.add_argument('--normalization_stats_file', default=None,
                    help='Pickle file with normalization stats for optional denormalization')
parser.add_argument('--n_traj_subjects', type=int, default=20,
                    help='Number of subjects to display in the trajectory grid (default 20)')
parser.add_argument('--min_timepoints', type=int, default=2,
                    help='Minimum number of observations for a subject to appear in trajectory plots')
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

BIOMARKER  = args.biomarker_name
OUT_DIR    = args.output_dir
N_TRAJ     = args.n_traj_subjects
MIN_TP     = args.min_timepoints
AGE_BINS   = [0, 60, 70, 80, 200]
AGE_LABELS = ['<60', '60-70', '70-80', '>=80']
AGE_COLORS = {'<60': '#2166ac', '60-70': '#4dac26', '70-80': '#d6604d', '>=80': '#762a83'}
SEX_COLORS = {'Male': '#4C72B0', 'Female': '#DD8452'}

plt.rcParams.update({
    'font.size': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

rng = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# 1. Load predictions
# ---------------------------------------------------------------------------
print(f'Loading predictions from: {args.predictions_file}')
pred_df = pd.read_csv(args.predictions_file)
pred_df['PTID'] = pred_df['PTID'].astype(str)
print(f'  {len(pred_df):,} observations, {pred_df["PTID"].nunique()} subjects')

# ---------------------------------------------------------------------------
# 2. Optional denormalization
# ---------------------------------------------------------------------------
target_mean, target_std = 0.0, 1.0
if args.normalization_stats_file and os.path.exists(args.normalization_stats_file):
    with open(args.normalization_stats_file, 'rb') as fh:
        norm_stats = pickle.load(fh)
    # Normalize biomarker name variants to match stats keys (e.g. SPARE-BA -> SPARE_BA)
    _norm_key_map = {'SPARE-BA': 'SPARE_BA', 'SPARE_BA': 'SPARE_BA', 'BAG': 'BAG'}
    key = _norm_key_map.get(BIOMARKER.upper(), BIOMARKER.upper())
    if key in norm_stats:
        target_mean = float(norm_stats[key]['mean'])
        target_std  = float(norm_stats[key]['std'])
        print(f'  Denormalizing {key}: mean={target_mean:.3f}, std={target_std:.3f}')
        for col in ['ground_truth', 'predicted', 'lower_bound', 'upper_bound']:
            pred_df[col] = pred_df[col] * target_std + target_mean
        pred_df['interval_width'] = pred_df['upper_bound'] - pred_df['lower_bound']
        pred_df['abs_error']      = np.abs(pred_df['ground_truth'] - pred_df['predicted'])
        pred_df['squared_error']  = (pred_df['ground_truth'] - pred_df['predicted']) ** 2
    else:
        print(f'  Key "{key}" not found in normalization stats. Plotting in normalized scale.')
else:
    print('  No normalization stats provided — plotting in normalized scale.')

y_label = BIOMARKER

# ---------------------------------------------------------------------------
# 3. Load and merge covariates (Sex, Age)
# ---------------------------------------------------------------------------
has_covariates = False

# Check if Sex / BaselineAge were already embedded in the predictions file
if {'Sex', 'BaselineAge'}.issubset(pred_df.columns) and pred_df['Sex'].notna().any():
    has_covariates = True
    print('  Sex and BaselineAge columns found directly in predictions file.')
elif args.covariates_file and os.path.exists(args.covariates_file):
    print(f'  Loading covariates from: {args.covariates_file}')
    cov_df = pd.read_csv(args.covariates_file)
    cov_df['PTID'] = cov_df['PTID'].astype(str)
    baseline_cov = (cov_df.sort_values('Time')
                          .groupby('PTID')
                          .first()
                          .reset_index()[['PTID', 'Sex', 'Age']]
                          .rename(columns={'Age': 'BaselineAge'}))
    pred_df = pred_df.merge(baseline_cov, on='PTID', how='left')
    has_covariates = True
    print(f'  Covariates merged for {pred_df["Sex"].notna().sum():,} observations')

if has_covariates:
    pred_df['Sex_label'] = pred_df['Sex'].map({0: 'Male', 1: 'Female'}).fillna('Unknown')
    pred_df['Age_group'] = pd.cut(
        pred_df['BaselineAge'], bins=AGE_BINS, labels=AGE_LABELS, right=False
    )
    n_sex = pred_df.drop_duplicates('PTID')['Sex_label'].value_counts().to_dict()
    print(f'  Sex distribution (subjects): {n_sex}')

# ---------------------------------------------------------------------------
# 4. Per-subject aggregated metrics
# ---------------------------------------------------------------------------
def _compute_subject_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ptid, grp in df.groupby('PTID'):
        n   = len(grp)
        mae = grp['abs_error'].mean()
        mse = grp['squared_error'].mean()
        rmse = float(np.sqrt(mse))
        cov  = grp['covered'].mean() if 'covered' in grp.columns else np.nan
        miw  = grp['interval_width'].mean()
        r2   = np.nan
        if n >= 2:
            ss_res = grp['squared_error'].sum()
            ss_tot = ((grp['ground_truth'] - grp['ground_truth'].mean()) ** 2).sum()
            r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan
        row = {
            'PTID': ptid,
            'n_timepoints': n,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'coverage': cov,
            'mean_interval_width': miw,
        }
        for col in ['Sex_label', 'Age_group', 'BaselineAge', 'Sex']:
            if col in grp.columns:
                row[col] = grp[col].iloc[0]
        rows.append(row)
    return pd.DataFrame(rows)


subj_df = _compute_subject_metrics(pred_df)
subj_df.to_csv(os.path.join(OUT_DIR, f'subject_summary_{BIOMARKER}.csv'), index=False)
print(f'\nPer-subject summary saved ({len(subj_df)} subjects)')

# ---------------------------------------------------------------------------
# Helper: print overall metrics
# ---------------------------------------------------------------------------
def _print_metrics(df_obs: pd.DataFrame, label: str = 'Overall') -> None:
    mae  = df_obs['abs_error'].mean()
    rmse = float(np.sqrt(df_obs['squared_error'].mean()))
    r2   = r2_score(df_obs['ground_truth'], df_obs['predicted'])
    cov  = df_obs['covered'].mean() if 'covered' in df_obs.columns else float('nan')
    miw  = df_obs['interval_width'].mean()
    print(f'  [{label}] MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}  '
          f'Coverage={cov:.4f}  IntWidth={miw:.4f}')


print('\n=== Overall Metrics ===')
_print_metrics(pred_df)

# ---------------------------------------------------------------------------
# Figure 1: Trajectory grid — predicted + CI bounds + ground truth
# ---------------------------------------------------------------------------
print('\nGenerating trajectory plots...')

eligible_ids = list(
    pred_df.groupby('PTID').filter(lambda x: len(x) >= MIN_TP)['PTID'].unique()
)

# Balance by sex when covariates are available
if has_covariates and 'Sex_label' in pred_df.columns:
    elig_df    = pred_df[pred_df['PTID'].isin(eligible_ids)].drop_duplicates('PTID')
    male_ids   = elig_df[elig_df['Sex_label'] == 'Male']['PTID'].values
    female_ids = elig_df[elig_df['Sex_label'] == 'Female']['PTID'].values
    n_each     = min(N_TRAJ // 2, len(male_ids), len(female_ids))
    sel_males  = rng.choice(male_ids,   size=n_each, replace=False) if n_each > 0 else male_ids[:0]
    sel_females= rng.choice(female_ids, size=n_each, replace=False) if n_each > 0 else female_ids[:0]
    sample_ids = list(sel_males) + list(sel_females)
    if len(sample_ids) == 0:
        n_s = min(N_TRAJ, len(eligible_ids))
        sample_ids = list(rng.choice(eligible_ids, size=n_s, replace=False))
else:
    n_s = min(N_TRAJ, len(eligible_ids))
    sample_ids = list(rng.choice(eligible_ids, size=n_s, replace=False))

n_plots = len(sample_ids)
n_cols  = 4
n_rows  = max(1, int(np.ceil(n_plots / n_cols)))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4.2, n_rows * 3.5))
axes_flat  = np.array(axes).flatten()

for i, ptid in enumerate(sample_ids):
    ax   = axes_flat[i]
    subj = pred_df[pred_df['PTID'] == ptid].sort_values('time_months')

    times = subj['time_months'].values
    gt    = subj['ground_truth'].values
    pred  = subj['predicted'].values
    lb    = subj['lower_bound'].values
    ub    = subj['upper_bound'].values

    ax.fill_between(times, lb, ub, alpha=0.25, color='steelblue', label='95% CI')
    ax.plot(times, pred, color='steelblue', linewidth=1.8, zorder=3, label='Predicted')
    ax.scatter(times, gt, color='crimson', s=28, zorder=4, label='Ground Truth')

    title = str(ptid)[-8:]
    if has_covariates and 'Sex_label' in subj.columns:
        sex_lbl = subj['Sex_label'].iloc[0]
        age_val = subj['BaselineAge'].iloc[0] if 'BaselineAge' in subj.columns else '?'
        age_str = f'{age_val:.0f}' if not pd.isna(age_val) else '?'
        title  += f'\n{sex_lbl}, Age {age_str}'
    ax.set_title(title, fontsize=8.5)

    if i % n_cols == 0:
        ax.set_ylabel(y_label, fontsize=9)
    if i >= (n_rows - 1) * n_cols:
        ax.set_xlabel('Time (months)', fontsize=9)
    ax.tick_params(labelsize=8)

# Legend on first axis only
axes_flat[0].legend(fontsize=7, loc='best', framealpha=0.7)

# Hide unused axes
for j in range(n_plots, len(axes_flat)):
    axes_flat[j].set_visible(False)

fig.suptitle(f'Predicted Trajectories with 95% CI vs Ground Truth — {BIOMARKER}',
             fontsize=12, y=1.01)
plt.tight_layout()
for ext in ('png', 'svg'):
    plt.savefig(os.path.join(OUT_DIR, f'trajectories_{BIOMARKER}.{ext}'),
                dpi=150, bbox_inches='tight')
plt.close()
print('  Saved: trajectory grid')

# ---------------------------------------------------------------------------
# Figure 2: Scatter — predicted vs ground truth
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 6))
ax.scatter(pred_df['ground_truth'], pred_df['predicted'],
           alpha=0.25, s=8, color='steelblue', rasterized=True)

vmin = min(pred_df['ground_truth'].min(), pred_df['predicted'].min())
vmax = max(pred_df['ground_truth'].max(), pred_df['predicted'].max())
ax.plot([vmin, vmax], [vmin, vmax], 'k--', linewidth=1, alpha=0.6, label='Identity')

r2_all  = r2_score(pred_df['ground_truth'], pred_df['predicted'])
mae_all = pred_df['abs_error'].mean()
ax.text(0.05, 0.95,
        f'R² = {r2_all:.3f}\nMAE = {mae_all:.3f}',
        transform=ax.transAxes, fontsize=11, va='top',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='lightgray', alpha=0.9))

ax.set_xlabel(f'Ground Truth ({y_label})', fontsize=12)
ax.set_ylabel(f'Predicted ({y_label})', fontsize=12)
ax.set_title(f'Predicted vs Ground Truth — {BIOMARKER}', fontsize=13)
plt.tight_layout()
for ext in ('png', 'svg'):
    plt.savefig(os.path.join(OUT_DIR, f'scatter_{BIOMARKER}.{ext}'),
                dpi=150, bbox_inches='tight')
plt.close()
print('  Saved: scatter plot')

# ---------------------------------------------------------------------------
# Figure 3: Error histograms
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
ax.hist(pred_df['abs_error'], bins=60, color='steelblue', edgecolor='white', alpha=0.85)
ax.axvline(pred_df['abs_error'].mean(), color='crimson', linestyle='--', linewidth=1.5,
           label=f'Mean = {pred_df["abs_error"].mean():.3f}')
ax.set_xlabel(f'Absolute Error ({y_label})', fontsize=12)
ax.set_ylabel('Observations', fontsize=12)
ax.set_title(f'Abs. Error Distribution — All Observations', fontsize=12)
ax.legend(fontsize=10)

ax = axes[1]
ax.hist(subj_df['mae'], bins=40, color='coral', edgecolor='white', alpha=0.85)
ax.axvline(subj_df['mae'].mean(), color='navy', linestyle='--', linewidth=1.5,
           label=f'Mean = {subj_df["mae"].mean():.3f}')
ax.set_xlabel(f'Per-Subject MAE ({y_label})', fontsize=12)
ax.set_ylabel('Subjects', fontsize=12)
ax.set_title(f'Per-Subject MAE Distribution', fontsize=12)
ax.legend(fontsize=10)

plt.tight_layout()
for ext in ('png', 'svg'):
    plt.savefig(os.path.join(OUT_DIR, f'error_histograms_{BIOMARKER}.{ext}'),
                dpi=150, bbox_inches='tight')
plt.close()
print('  Saved: error histograms')

# ===========================================================================
# Demographic analyses (Sex and Age) — only when covariates are available
# ===========================================================================
if has_covariates:

    # -----------------------------------------------------------------------
    # Figure 4: Error by Sex — violin + bar chart
    # -----------------------------------------------------------------------
    sex_groups  = [s for s in ['Male', 'Female'] if s in subj_df.get('Sex_label', pd.Series()).values]
    sex_mae     = {s: subj_df[subj_df['Sex_label'] == s]['mae'].dropna().values for s in sex_groups}
    sex_mae     = {k: v for k, v in sex_mae.items() if len(v) > 0}

    if len(sex_mae) >= 1:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

        # — Violin
        ax = axes[0]
        keys   = list(sex_mae.keys())
        colors = [SEX_COLORS.get(s, '#888888') for s in keys]
        parts  = ax.violinplot([sex_mae[s] for s in keys],
                               positions=range(len(keys)),
                               showmedians=True, showextrema=True)
        for pc, col in zip(parts['bodies'], colors):
            pc.set_facecolor(col)
            pc.set_alpha(0.72)
        parts['cmedians'].set_color('black')
        parts['cmedians'].set_linewidth(2)

        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels([f'{s}\n(n={len(sex_mae[s])})' for s in keys], fontsize=12)
        ax.set_ylabel('MAE (per subject)', fontsize=12)
        ax.set_title(f'{BIOMARKER} — Prediction Error by Sex', fontsize=12)

        # Mann-Whitney U
        if len(keys) == 2:
            stat_mw, p_mw = stats.mannwhitneyu(sex_mae[keys[0]], sex_mae[keys[1]],
                                                alternative='two-sided')
            sig = '***' if p_mw < 0.001 else '**' if p_mw < 0.01 else '*' if p_mw < 0.05 else 'n.s.'
            ax.text(0.5, 0.97,
                    f'Mann-Whitney U  p = {p_mw:.4f}  {sig}',
                    transform=ax.transAxes, fontsize=9,
                    ha='center', va='top', style='italic',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow', alpha=0.8))

        # — Bar chart  (MAE mean ± std per sex)
        ax = axes[1]
        metric_rows = []
        for s in keys:
            group = subj_df[subj_df['Sex_label'] == s]
            metric_rows.append({
                'Sex': s,
                'MAE_mean':  group['mae'].mean(),
                'MAE_std':   group['mae'].std(),
                'RMSE_mean': group['rmse'].mean(),
                'RMSE_std':  group['rmse'].std(),
                'R2_mean':   group['r2'].mean(),
                'Coverage_mean': group['coverage'].mean() * 100,
                'n':         len(group),
            })
        sex_metrics_df = pd.DataFrame(metric_rows)

        x     = np.arange(len(sex_metrics_df))
        width = 0.35
        bar_colors = [SEX_COLORS.get(s, '#888') for s in sex_metrics_df['Sex']]
        ax.bar(x - width / 2, sex_metrics_df['MAE_mean'],  width,
               yerr=sex_metrics_df['MAE_std'],  capsize=5,
               color=bar_colors, alpha=0.80, label='MAE')
        ax.bar(x + width / 2, sex_metrics_df['RMSE_mean'], width,
               yerr=sex_metrics_df['RMSE_std'], capsize=5,
               color=bar_colors, alpha=0.45, label='RMSE', hatch='//')

        ax.set_xticks(x)
        ax.set_xticklabels([f"{r['Sex']}\n(n={r['n']})" for _, r in sex_metrics_df.iterrows()],
                           fontsize=11)
        ax.set_ylabel('Error', fontsize=12)
        ax.set_title(f'{BIOMARKER} — MAE & RMSE by Sex', fontsize=12)
        ax.legend(fontsize=10)

        plt.tight_layout()
        for ext in ('png', 'svg'):
            plt.savefig(os.path.join(OUT_DIR, f'error_by_sex_{BIOMARKER}.{ext}'),
                        dpi=150, bbox_inches='tight')
        plt.close()
        print('  Saved: error by sex')

        sex_metrics_df.to_csv(
            os.path.join(OUT_DIR, f'metrics_by_sex_{BIOMARKER}.csv'), index=False
        )

        print('\n=== Metrics by Sex ===')
        for _, row in sex_metrics_df.iterrows():
            print(f"  {row['Sex']} (n={row['n']}): "
                  f"MAE={row['MAE_mean']:.4f}±{row['MAE_std']:.4f}  "
                  f"RMSE={row['RMSE_mean']:.4f}  "
                  f"R²={row['R2_mean']:.4f}  "
                  f"Coverage={row['Coverage_mean']:.1f}%")

    # -----------------------------------------------------------------------
    # Figure 5: Error by Age group — violin + bar chart
    # -----------------------------------------------------------------------
    valid_age_groups = [ag for ag in AGE_LABELS
                        if ag in subj_df.get('Age_group', pd.Series()).astype(str).values]
    age_mae = {ag: subj_df[subj_df['Age_group'].astype(str) == ag]['mae'].dropna().values
               for ag in valid_age_groups}
    age_mae = {k: v for k, v in age_mae.items() if len(v) > 0}

    if len(age_mae) >= 1:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

        # — Violin
        ax = axes[0]
        vkeys  = list(age_mae.keys())
        vcolors= [AGE_COLORS.get(ag, '#888888') for ag in vkeys]
        parts  = ax.violinplot([age_mae[ag] for ag in vkeys],
                               positions=range(len(vkeys)),
                               showmedians=True, showextrema=True)
        for pc, col in zip(parts['bodies'], vcolors):
            pc.set_facecolor(col)
            pc.set_alpha(0.72)
        parts['cmedians'].set_color('black')
        parts['cmedians'].set_linewidth(2)

        ax.set_xticks(range(len(vkeys)))
        ax.set_xticklabels([f'{ag}\n(n={len(age_mae[ag])})' for ag in vkeys], fontsize=10)
        ax.set_ylabel('MAE (per subject)', fontsize=12)
        ax.set_title(f'{BIOMARKER} — Prediction Error by Age Group', fontsize=12)

        # Kruskal-Wallis
        if len(vkeys) >= 2:
            kw_stat, kw_p = stats.kruskal(*[age_mae[ag] for ag in vkeys])
            sig_kw = '***' if kw_p < 0.001 else '**' if kw_p < 0.01 else '*' if kw_p < 0.05 else 'n.s.'
            ax.text(0.5, 0.97,
                    f'Kruskal-Wallis  p = {kw_p:.4f}  {sig_kw}',
                    transform=ax.transAxes, fontsize=9,
                    ha='center', va='top', style='italic',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow', alpha=0.8))

        # — Bar chart
        ax = axes[1]
        age_metric_rows = []
        for ag in vkeys:
            group = subj_df[subj_df['Age_group'].astype(str) == ag]
            age_metric_rows.append({
                'Age_group':   ag,
                'MAE_mean':    group['mae'].mean(),
                'MAE_std':     group['mae'].std(),
                'RMSE_mean':   group['rmse'].mean(),
                'RMSE_std':    group['rmse'].std(),
                'R2_mean':     group['r2'].mean(),
                'Coverage_mean': group['coverage'].mean() * 100,
                'n':           len(group),
            })
        age_metrics_df = pd.DataFrame(age_metric_rows)

        x      = np.arange(len(age_metrics_df))
        width  = 0.35
        acolors= [AGE_COLORS.get(ag, '#888') for ag in age_metrics_df['Age_group']]
        ax.bar(x - width / 2, age_metrics_df['MAE_mean'],  width,
               yerr=age_metrics_df['MAE_std'],  capsize=5,
               color=acolors, alpha=0.80, label='MAE')
        ax.bar(x + width / 2, age_metrics_df['RMSE_mean'], width,
               yerr=age_metrics_df['RMSE_std'], capsize=5,
               color=acolors, alpha=0.45, label='RMSE', hatch='//')

        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"{r['Age_group']}\n(n={r['n']})" for _, r in age_metrics_df.iterrows()],
            fontsize=10
        )
        ax.set_ylabel('Error', fontsize=12)
        ax.set_title(f'{BIOMARKER} — MAE & RMSE by Age Group', fontsize=12)
        ax.legend(fontsize=10)

        plt.tight_layout()
        for ext in ('png', 'svg'):
            plt.savefig(os.path.join(OUT_DIR, f'error_by_age_{BIOMARKER}.{ext}'),
                        dpi=150, bbox_inches='tight')
        plt.close()
        print('  Saved: error by age group')

        age_metrics_df.to_csv(
            os.path.join(OUT_DIR, f'metrics_by_age_{BIOMARKER}.csv'), index=False
        )

        print('\n=== Metrics by Age Group ===')
        for _, row in age_metrics_df.iterrows():
            print(f"  {row['Age_group']} (n={row['n']}): "
                  f"MAE={row['MAE_mean']:.4f}±{row['MAE_std']:.4f}  "
                  f"RMSE={row['RMSE_mean']:.4f}  "
                  f"R²={row['R2_mean']:.4f}  "
                  f"Coverage={row['Coverage_mean']:.1f}%")

    # -----------------------------------------------------------------------
    # Figure 6: Sex-stratified trajectory grids
    # -----------------------------------------------------------------------
    for sex_label in ['Male', 'Female']:
        if 'Sex_label' not in subj_df.columns:
            break
        sex_eligible = [p for p in subj_df[subj_df['Sex_label'] == sex_label]['PTID'].values
                        if p in eligible_ids]
        if not sex_eligible:
            continue

        n_s  = min(12, len(sex_eligible))
        samp = list(rng.choice(sex_eligible, size=n_s, replace=False))
        nc, nr = 4, max(1, int(np.ceil(n_s / 4)))
        color  = SEX_COLORS[sex_label]

        fig, axes = plt.subplots(nr, nc, figsize=(nc * 4.2, nr * 3.5))
        axes_flat = np.array(axes).flatten()

        for i, ptid in enumerate(samp):
            ax   = axes_flat[i]
            subj = pred_df[pred_df['PTID'] == ptid].sort_values('time_months')
            times, gt, pred_v = (subj['time_months'].values,
                                 subj['ground_truth'].values,
                                 subj['predicted'].values)
            lb, ub = subj['lower_bound'].values, subj['upper_bound'].values

            ax.fill_between(times, lb, ub, alpha=0.25, color=color)
            ax.plot(times, pred_v, color=color, linewidth=1.8, zorder=3)
            ax.scatter(times, gt, color='crimson', s=28, zorder=4)

            age_str = ''
            if 'BaselineAge' in subj.columns and not pd.isna(subj['BaselineAge'].iloc[0]):
                age_str = f', Age {subj["BaselineAge"].iloc[0]:.0f}'
            ax.set_title(f'{str(ptid)[-8:]}{age_str}', fontsize=8.5)
            ax.tick_params(labelsize=8)
            if i % nc == 0:
                ax.set_ylabel(y_label, fontsize=9)

        for j in range(n_s, len(axes_flat)):
            axes_flat[j].set_visible(False)

        fig.suptitle(f'{BIOMARKER} Trajectories — {sex_label}', fontsize=12, y=1.01)
        plt.tight_layout()
        for ext in ('png', 'svg'):
            plt.savefig(
                os.path.join(OUT_DIR, f'trajectories_{sex_label.lower()}_{BIOMARKER}.{ext}'),
                dpi=150, bbox_inches='tight'
            )
        plt.close()
    print('  Saved: sex-stratified trajectory grids')

    # -----------------------------------------------------------------------
    # Figure 7: Sex × Age group heatmaps (MAE and RMSE)
    # -----------------------------------------------------------------------
    if 'Sex_label' in subj_df.columns and 'Age_group' in subj_df.columns:
        cross_rows = []
        for (sex, age), grp in subj_df.groupby(['Sex_label', 'Age_group'], observed=True):
            if sex not in ('Male', 'Female') or str(age) in ('nan', 'None', ''):
                continue
            cross_rows.append({
                'Sex': sex,
                'Age_group': str(age),
                'n': len(grp),
                'MAE':  grp['mae'].mean(),
                'RMSE': grp['rmse'].mean(),
                'R2':   grp['r2'].mean(),
                'Coverage': grp['coverage'].mean() * 100,
            })

        if cross_rows:
            cross_df = pd.DataFrame(cross_rows)
            cross_df.to_csv(
                os.path.join(OUT_DIR, f'metrics_sex_x_age_{BIOMARKER}.csv'), index=False
            )

            for metric in ['MAE', 'RMSE']:
                pivot = cross_df.pivot(index='Sex', columns='Age_group', values=metric)
                ordered_cols = [c for c in AGE_LABELS if c in pivot.columns]
                pivot = pivot[ordered_cols]

                fig, ax = plt.subplots(figsize=(max(6, len(ordered_cols) * 2), 3.0))
                vals = pivot.values.astype(float)
                vmin_h = np.nanmin(vals)
                vmax_h = np.nanmax(vals)
                im = ax.imshow(vals, cmap='YlOrRd', aspect='auto',
                               vmin=vmin_h, vmax=vmax_h)
                plt.colorbar(im, ax=ax, label=metric, shrink=0.85)

                ax.set_xticks(range(len(pivot.columns)))
                ax.set_xticklabels(pivot.columns, fontsize=11)
                ax.set_yticks(range(len(pivot.index)))
                ax.set_yticklabels(pivot.index, fontsize=11)
                ax.set_xlabel('Age Group', fontsize=11)

                mid = (vmin_h + vmax_h) / 2
                for (ir, ic), val in np.ndenumerate(vals):
                    if not np.isnan(val):
                        txt_col = 'white' if val > mid else 'black'
                        ax.text(ic, ir, f'{val:.3f}',
                                ha='center', va='center', fontsize=10, color=txt_col)

                # Annotate with n
                n_pivot = cross_df.pivot(index='Sex', columns='Age_group', values='n')
                n_pivot = n_pivot[[c for c in ordered_cols if c in n_pivot.columns]]
                for (ir, ic), n_val in np.ndenumerate(n_pivot.values.astype(float)):
                    if not np.isnan(n_val):
                        ax.text(ic, ir + 0.32, f'n={int(n_val)}',
                                ha='center', va='center', fontsize=7.5, color='dimgray')

                ax.set_title(f'{BIOMARKER} {metric} — Sex × Age Group', fontsize=12)
                plt.tight_layout()
                for ext in ('png', 'svg'):
                    plt.savefig(
                        os.path.join(OUT_DIR, f'heatmap_{metric.lower()}_{BIOMARKER}.{ext}'),
                        dpi=150, bbox_inches='tight'
                    )
                plt.close()
            print('  Saved: Sex × Age heatmaps')

    # -----------------------------------------------------------------------
    # Figure 8: Coverage analysis by Sex and Age group
    # -----------------------------------------------------------------------
    if 'covered' in pred_df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        # Coverage by Sex
        ax = axes[0]
        if 'Sex_label' in pred_df.columns:
            cov_sex = (pred_df.groupby('Sex_label')['covered']
                              .agg(['mean', 'count'])
                              .reset_index()
                              .rename(columns={'mean': 'Coverage', 'count': 'n'}))
            cov_sex = cov_sex[cov_sex['Sex_label'].isin(['Male', 'Female'])]
            bar_c = [SEX_COLORS.get(s, '#888') for s in cov_sex['Sex_label']]
            ax.bar(range(len(cov_sex)), cov_sex['Coverage'] * 100,
                   color=bar_c, alpha=0.8, edgecolor='white')
            ax.axhline(95, color='black', linestyle='--', linewidth=1, alpha=0.6,
                       label='Nominal 95%')
            ax.set_xticks(range(len(cov_sex)))
            ax.set_xticklabels(
                [f"{r['Sex_label']}\n(n={r['n']})" for _, r in cov_sex.iterrows()],
                fontsize=11
            )
            ax.set_ylabel('Coverage (%)', fontsize=12)
            ax.set_ylim(0, 105)
            ax.set_title(f'{BIOMARKER} — 95% CI Coverage by Sex', fontsize=12)
            ax.legend(fontsize=10)

        # Coverage by Age group
        ax = axes[1]
        if 'Age_group' in pred_df.columns:
            cov_age = (pred_df.groupby('Age_group', observed=True)['covered']
                              .agg(['mean', 'count'])
                              .reset_index()
                              .rename(columns={'mean': 'Coverage', 'count': 'n'}))
            cov_age['Age_group'] = cov_age['Age_group'].astype(str)
            cov_age = cov_age[cov_age['Age_group'].isin(AGE_LABELS)]
            cov_age = cov_age.set_index('Age_group').reindex(
                [a for a in AGE_LABELS if a in cov_age['Age_group'].values]
            ).reset_index()
            bar_c = [AGE_COLORS.get(ag, '#888') for ag in cov_age['Age_group']]
            ax.bar(range(len(cov_age)), cov_age['Coverage'] * 100,
                   color=bar_c, alpha=0.8, edgecolor='white')
            ax.axhline(95, color='black', linestyle='--', linewidth=1, alpha=0.6,
                       label='Nominal 95%')
            ax.set_xticks(range(len(cov_age)))
            ax.set_xticklabels(
                [f"{r['Age_group']}\n(n={r['n']})" for _, r in cov_age.iterrows()],
                fontsize=10
            )
            ax.set_ylabel('Coverage (%)', fontsize=12)
            ax.set_ylim(0, 105)
            ax.set_title(f'{BIOMARKER} — 95% CI Coverage by Age Group', fontsize=12)
            ax.legend(fontsize=10)

        plt.tight_layout()
        for ext in ('png', 'svg'):
            plt.savefig(os.path.join(OUT_DIR, f'coverage_by_demographics_{BIOMARKER}.{ext}'),
                        dpi=150, bbox_inches='tight')
        plt.close()
        print('  Saved: coverage by demographics')

else:
    print('\nNo covariates available — skipping demographic analyses.')
    print('  Re-run with --covariates_file to enable Sex/Age stratification.')

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------
print(f'\n=== Final Summary — {BIOMARKER} ===')
print(f'  Observations : {len(pred_df):,}')
print(f'  Subjects     : {len(subj_df)}')
_print_metrics(pred_df)

print(f'\nAll outputs saved to: {OUT_DIR}')
