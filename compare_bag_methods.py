'''
BAG Comparison: Direct BAG Model  vs  SPARE-BA-Derived BAG
===========================================================

Two strategies for predicting Brain Age Gap (BAG):

  Method A — Direct   : BAG predicted by the DKGP BAG model
  Method B — Derived  : BAG = predicted_SPARE-BA  −  Age_at_timepoint
                        where  Age_at_tp = BaselineAge + time_months / 12

This script merges both prediction sets per fold, computes comprehensive
comparative metrics, and generates publication-ready plots so you can
decide which approach to use.

Key outputs
-----------
  compare_fold_metrics.csv          — per-fold metric table (both methods)
  compare_subject_mae.csv           — per-subject MAE for both methods
  fig1_fold_metrics.png/svg         — cross-fold bar chart (MAE & RMSE)
  fig2_scatter_both_methods.png/svg — scatter: each method vs ground truth
  fig3_bland_altman.png/svg         — agreement between the two predictions
  fig4_error_distributions.png/svg  — absolute-error histograms side-by-side
  fig5_subject_mae_comparison.png/svg — subject-level MAE: direct vs derived
  fig6_error_by_sex.png/svg         — MAE by Sex for both methods
  fig7_error_by_age.png/svg         — MAE by Age group for both methods
  fig8_coverage.png/svg             — 95% CI coverage comparison
  interpretation.txt                — plain-English summary of findings

Usage
-----
  python compare_bag_methods.py                          # all defaults
  python compare_bag_methods.py \\
      --bag_models_dir    models \\
      --spare_ba_models_dir models \\
      --normalization_stats data/normalization_stats.pkl \\
      --output_dir        analysis/bag_comparison \\
      --n_folds           5 \\
      --biomarker_index   0
'''

import os, argparse, pickle, textwrap
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
parser = argparse.ArgumentParser(description='Compare direct BAG vs SPARE-BA-derived BAG')
parser.add_argument('--bag_models_dir',     default='models',
                    help='Parent dir containing bag_fold{i}/ sub-dirs')
parser.add_argument('--spare_ba_models_dir', default='models',
                    help='Parent dir containing spare_ba_fold{i}/ sub-dirs')
parser.add_argument('--normalization_stats', default='data/normalization_stats.pkl',
                    help='Pickle with BAG and SPARE_BA normalization stats')
parser.add_argument('--output_dir',         default='analysis/bag_comparison')
parser.add_argument('--n_folds',  type=int, default=5)
parser.add_argument('--biomarker_index', type=int, default=0)
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)
OUT = args.output_dir
IDX = args.biomarker_index

plt.rcParams.update({'font.size': 11,
                     'axes.spines.top': False,
                     'axes.spines.right': False})

AGE_BINS   = [0, 60, 70, 80, 200]
AGE_LABELS = ['<60', '60-70', '70-80', '>=80']
AGE_COLORS = {'<60': '#2166ac', '60-70': '#4dac26',
              '70-80': '#d6604d', '>=80': '#762a83'}
SEX_COLORS = {'Male': '#4C72B0', 'Female': '#DD8452'}
METHOD_COLORS = {'Direct': '#1f77b4', 'Derived': '#d62728'}

# ---------------------------------------------------------------------------
# 1.  Load normalization stats
# ---------------------------------------------------------------------------
if not os.path.exists(args.normalization_stats):
    raise FileNotFoundError(f'Normalization stats not found: {args.normalization_stats}')

with open(args.normalization_stats, 'rb') as fh:
    norm = pickle.load(fh)

mean_bag,  std_bag  = float(norm['BAG']['mean']),      float(norm['BAG']['std'])
mean_sba,  std_sba  = float(norm['SPARE_BA']['mean']), float(norm['SPARE_BA']['std'])
print(f'Norm stats  BAG    : mean={mean_bag:.3f}  std={std_bag:.3f}')
print(f'Norm stats  SPARE-BA: mean={mean_sba:.3f}  std={std_sba:.3f}')

# ---------------------------------------------------------------------------
# 2.  Load & merge predictions for every fold
# ---------------------------------------------------------------------------
def _denorm(series, mean, std):
    return series * std + mean

fold_dfs   = []
fold_stats = []

for fold in range(args.n_folds):
    bag_csv  = os.path.join(args.bag_models_dir,
                            f'bag_fold{fold}',
                            f'predictions_BAG_{IDX}_{fold}.csv')
    sba_csv  = os.path.join(args.spare_ba_models_dir,
                            f'spare_ba_fold{fold}',
                            f'predictions_SPARE-BA_{IDX}_{fold}.csv')

    if not os.path.exists(bag_csv):
        print(f'  Fold {fold}: BAG predictions not found — skipping')
        continue
    if not os.path.exists(sba_csv):
        print(f'  Fold {fold}: SPARE-BA predictions not found — skipping')
        continue

    bag = pd.read_csv(bag_csv)
    sba = pd.read_csv(sba_csv)
    bag['PTID'] = bag['PTID'].astype(str)
    sba['PTID'] = sba['PTID'].astype(str)

    # --- denormalise BAG predictions ---
    for col in ['ground_truth', 'predicted', 'lower_bound', 'upper_bound']:
        bag[col] = _denorm(bag[col], mean_bag, std_bag)
    bag.rename(columns={
        'ground_truth': 'gt_BAG',
        'predicted':    'pred_direct',
        'lower_bound':  'lower_direct',
        'upper_bound':  'upper_direct',
        'covered':      'covered_direct',
    }, inplace=True)

    # --- denormalise SPARE-BA predictions ---
    for col in ['ground_truth', 'predicted', 'lower_bound', 'upper_bound']:
        sba[col] = _denorm(sba[col], mean_sba, std_sba)

    # Age at each timepoint (years) — BaselineAge is raw (from unnorm covariates)
    sba['Age_at_tp']    = sba['BaselineAge'] + sba['time_months'] / 12.0
    sba['pred_derived'] = sba['predicted']    - sba['Age_at_tp']
    sba['lower_derived']= sba['lower_bound']  - sba['Age_at_tp']
    sba['upper_derived']= sba['upper_bound']  - sba['Age_at_tp']

    sba_cols = ['PTID', 'time_months',
                'pred_derived', 'lower_derived', 'upper_derived']
    merged = bag.merge(sba[sba_cols], on=['PTID', 'time_months'], how='inner')

    # Coverage for derived method
    merged['covered_derived'] = (
        (merged['gt_BAG'] >= merged['lower_derived']) &
        (merged['gt_BAG'] <= merged['upper_derived'])
    ).astype(int)

    # Absolute errors
    merged['ae_direct']  = np.abs(merged['gt_BAG'] - merged['pred_direct'])
    merged['ae_derived'] = np.abs(merged['gt_BAG'] - merged['pred_derived'])
    merged['sq_direct']  = (merged['gt_BAG'] - merged['pred_direct'])  ** 2
    merged['sq_derived'] = (merged['gt_BAG'] - merged['pred_derived']) ** 2

    # Sex label & Age group (from BAG file which already has covariates)
    if 'Sex' in merged.columns:
        merged['Sex_label'] = merged['Sex'].map({0: 'Male', 1: 'Female'}).fillna('Unknown')
    if 'BaselineAge' in merged.columns:
        merged['Age_group'] = pd.cut(merged['BaselineAge'],
                                     bins=AGE_BINS, labels=AGE_LABELS, right=False)
    merged['fold'] = fold
    fold_dfs.append(merged)

    # ---- per-fold summary stats ----
    n = len(merged)
    row = dict(
        fold        = fold,
        n_obs       = n,
        n_subjects  = merged['PTID'].nunique(),
        MAE_direct  = merged['ae_direct'].mean(),
        MAE_derived = merged['ae_derived'].mean(),
        RMSE_direct = float(np.sqrt(merged['sq_direct'].mean())),
        RMSE_derived= float(np.sqrt(merged['sq_derived'].mean())),
        R2_direct   = r2_score(merged['gt_BAG'], merged['pred_direct']),
        R2_derived  = r2_score(merged['gt_BAG'], merged['pred_derived']),
        Cov_direct  = merged['covered_direct'].mean()  * 100,
        Cov_derived = merged['covered_derived'].mean() * 100,
        IW_direct   = (merged['upper_direct']  - merged['lower_direct']).mean(),
        IW_derived  = (merged['upper_derived'] - merged['lower_derived']).mean(),
    )
    fold_stats.append(row)
    print(f'  Fold {fold}: n={n}  '
          f'MAE direct={row["MAE_direct"]:.3f}  derived={row["MAE_derived"]:.3f}  |  '
          f'RMSE direct={row["RMSE_direct"]:.3f}  derived={row["RMSE_derived"]:.3f}')

if not fold_dfs:
    raise RuntimeError('No fold data loaded — check paths.')

all_df   = pd.concat(fold_dfs, ignore_index=True)
stats_df = pd.DataFrame(fold_stats)
stats_df.to_csv(os.path.join(OUT, 'compare_fold_metrics.csv'), index=False)
print(f'\nFold-level metrics saved.')

# ---------------------------------------------------------------------------
# 3.  Cross-fold aggregate metrics (reported in interpretation)
# ---------------------------------------------------------------------------
def _agg(col):
    return all_df[col].mean(), all_df[col].std()

agg = {
    'MAE_direct' : _agg('ae_direct'),
    'MAE_derived': _agg('ae_derived'),
    'RMSE_direct' : (float(np.sqrt(all_df['sq_direct'].mean())), None),
    'RMSE_derived': (float(np.sqrt(all_df['sq_derived'].mean())), None),
    'R2_direct'   : (r2_score(all_df['gt_BAG'], all_df['pred_direct']),  None),
    'R2_derived'  : (r2_score(all_df['gt_BAG'], all_df['pred_derived']), None),
    'Cov_direct'  : (all_df['covered_direct'].mean()  * 100, None),
    'Cov_derived' : (all_df['covered_derived'].mean() * 100, None),
}

# Wilcoxon signed-rank on per-observation absolute errors (paired)
wstat, wpval = stats.wilcoxon(all_df['ae_direct'], all_df['ae_derived'])

# ---------------------------------------------------------------------------
# 4.  Per-subject MAE
# ---------------------------------------------------------------------------
subj = (all_df.groupby('PTID')
        .agg(mae_direct =('ae_direct',  'mean'),
             mae_derived=('ae_derived', 'mean'),
             n_obs      =('gt_BAG',     'count'))
        .reset_index())
if 'Sex_label' in all_df.columns:
    subj = subj.merge(all_df.drop_duplicates('PTID')[['PTID', 'Sex_label', 'Age_group']],
                      on='PTID', how='left')
subj.to_csv(os.path.join(OUT, 'compare_subject_mae.csv'), index=False)

# Who benefits from which method?
subj['winner'] = np.where(subj['mae_direct'] < subj['mae_derived'],
                           'Direct', 'Derived')
winner_counts = subj['winner'].value_counts()

# ---------------------------------------------------------------------------
# Fig 1 — Cross-fold bar chart: MAE and RMSE
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
x = np.arange(len(stats_df))
w = 0.35

for ax, metric, ylabel in zip(axes,
                               [('MAE_direct', 'MAE_derived'),
                                ('RMSE_direct', 'RMSE_derived')],
                               ['MAE (years)', 'RMSE (years)']):
    m_d, m_r = metric
    ax.bar(x - w/2, stats_df[m_d], w, label='Direct BAG',
           color=METHOD_COLORS['Direct'],  alpha=0.82)
    ax.bar(x + w/2, stats_df[m_r], w, label='Derived BAG',
           color=METHOD_COLORS['Derived'], alpha=0.82)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Fold {i}' for i in stats_df['fold']], fontsize=11)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f'{ylabel.split()[0]} per Fold — Direct vs Derived BAG', fontsize=12)
    ax.legend(fontsize=10)

plt.suptitle('Direct vs Derived BAG — Per-Fold Error Metrics', fontsize=13)
plt.tight_layout()
for ext in ('png', 'svg'):
    plt.savefig(os.path.join(OUT, f'fig1_fold_metrics.{ext}'), dpi=150, bbox_inches='tight')
plt.close()
print('Saved: fig1_fold_metrics')

# ---------------------------------------------------------------------------
# Fig 2 — Scatter: each method vs ground truth (side-by-side)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 6))
lim_min = all_df['gt_BAG'].quantile(0.01)
lim_max = all_df['gt_BAG'].quantile(0.99)

for ax, pred_col, method, color in [
        (axes[0], 'pred_direct',  'Direct',  METHOD_COLORS['Direct']),
        (axes[1], 'pred_derived', 'Derived', METHOD_COLORS['Derived']),
]:
    ax.scatter(all_df['gt_BAG'], all_df[pred_col],
               alpha=0.15, s=5, color=color, rasterized=True)
    ax.plot([lim_min, lim_max], [lim_min, lim_max],
            'k--', linewidth=1, alpha=0.6)
    r2  = r2_score(all_df['gt_BAG'], all_df[pred_col])
    mae = np.abs(all_df['gt_BAG'] - all_df[pred_col]).mean()
    ax.text(0.05, 0.95,
            f'R² = {r2:.3f}\nMAE = {mae:.3f} yr',
            transform=ax.transAxes, fontsize=11, va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='lightgray', alpha=0.9))
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.set_xlabel('Ground Truth BAG (years)', fontsize=12)
    ax.set_ylabel(f'Predicted BAG — {method} (years)', fontsize=12)
    ax.set_title(f'{method} BAG  vs  Ground Truth', fontsize=12)
    ax.set_aspect('equal', adjustable='box')

plt.tight_layout()
for ext in ('png', 'svg'):
    plt.savefig(os.path.join(OUT, f'fig2_scatter_both_methods.{ext}'),
                dpi=150, bbox_inches='tight')
plt.close()
print('Saved: fig2_scatter_both_methods')

# ---------------------------------------------------------------------------
# Fig 3 — Bland-Altman: agreement between the two methods
# ---------------------------------------------------------------------------
mean_pred  = (all_df['pred_direct'] + all_df['pred_derived']) / 2
diff_pred  = all_df['pred_direct'] - all_df['pred_derived']   # Direct − Derived
mean_diff  = diff_pred.mean()
std_diff   = diff_pred.std()
loa_upper  = mean_diff + 1.96 * std_diff
loa_lower  = mean_diff - 1.96 * std_diff

fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(mean_pred, diff_pred, alpha=0.15, s=5, color='steelblue', rasterized=True)
ax.axhline(mean_diff,  color='black',  linewidth=1.5, label=f'Bias = {mean_diff:.3f} yr')
ax.axhline(loa_upper,  color='crimson', linewidth=1.2, linestyle='--',
           label=f'+1.96 SD = {loa_upper:.3f} yr')
ax.axhline(loa_lower,  color='crimson', linewidth=1.2, linestyle='--',
           label=f'−1.96 SD = {loa_lower:.3f} yr')
ax.axhline(0, color='gray', linewidth=0.8, linestyle=':', alpha=0.6)
ax.fill_between(ax.get_xlim(), loa_lower, loa_upper, alpha=0.06, color='crimson')
ax.set_xlabel('Mean of Direct & Derived BAG (years)', fontsize=12)
ax.set_ylabel('Direct − Derived BAG (years)', fontsize=12)
ax.set_title('Bland-Altman: Direct BAG  vs  Derived BAG', fontsize=13)
ax.legend(fontsize=10, loc='upper right')
plt.tight_layout()
for ext in ('png', 'svg'):
    plt.savefig(os.path.join(OUT, f'fig3_bland_altman.{ext}'), dpi=150, bbox_inches='tight')
plt.close()
print('Saved: fig3_bland_altman')

# ---------------------------------------------------------------------------
# Fig 4 — Error distributions (absolute error histograms, side-by-side)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, col, method, color in [
        (axes[0], 'ae_direct',  'Direct',  METHOD_COLORS['Direct']),
        (axes[1], 'ae_derived', 'Derived', METHOD_COLORS['Derived']),
]:
    ax.hist(all_df[col], bins=60, color=color, edgecolor='white', alpha=0.82)
    m = all_df[col].mean()
    ax.axvline(m, color='black', linestyle='--', linewidth=1.5,
               label=f'Mean = {m:.3f} yr')
    ax.set_xlabel('Absolute Error (years)', fontsize=12)
    ax.set_ylabel('Observations', fontsize=12)
    ax.set_title(f'Abs Error — {method} BAG', fontsize=12)
    ax.legend(fontsize=10)

plt.suptitle('Absolute Error Distribution: Direct vs Derived BAG', fontsize=13)
plt.tight_layout()
for ext in ('png', 'svg'):
    plt.savefig(os.path.join(OUT, f'fig4_error_distributions.{ext}'),
                dpi=150, bbox_inches='tight')
plt.close()
print('Saved: fig4_error_distributions')

# ---------------------------------------------------------------------------
# Fig 5 — Subject-level MAE: Direct vs Derived (scatter + diagonal)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 7))
lim = max(subj[['mae_direct', 'mae_derived']].max()) * 1.05
ax.scatter(subj['mae_direct'], subj['mae_derived'],
           alpha=0.4, s=18, color='steelblue', edgecolors='none')
ax.plot([0, lim], [0, lim], 'k--', linewidth=1, alpha=0.6, label='Equal error')
ax.axhline(subj['mae_derived'].mean(), color=METHOD_COLORS['Derived'],
           linewidth=1, linestyle=':', alpha=0.7,
           label=f'Derived mean = {subj["mae_derived"].mean():.3f}')
ax.axvline(subj['mae_direct'].mean(),  color=METHOD_COLORS['Direct'],
           linewidth=1, linestyle=':', alpha=0.7,
           label=f'Direct mean  = {subj["mae_direct"].mean():.3f}')

n_dir = (winner_counts.get('Direct',  0))
n_der = (winner_counts.get('Derived', 0))
ax.text(0.04, 0.96,
        f'Direct better:  {n_dir} subjects ({100*n_dir/len(subj):.0f}%)\n'
        f'Derived better: {n_der} subjects ({100*n_der/len(subj):.0f}%)',
        transform=ax.transAxes, fontsize=10, va='top',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.9))
ax.set_xlabel('Per-Subject MAE — Direct BAG (years)', fontsize=12)
ax.set_ylabel('Per-Subject MAE — Derived BAG (years)', fontsize=12)
ax.set_title('Subject-Level MAE: Direct vs Derived BAG', fontsize=13)
ax.set_xlim(0, lim); ax.set_ylim(0, lim)
ax.legend(fontsize=9, loc='lower right')
plt.tight_layout()
for ext in ('png', 'svg'):
    plt.savefig(os.path.join(OUT, f'fig5_subject_mae_comparison.{ext}'),
                dpi=150, bbox_inches='tight')
plt.close()
print('Saved: fig5_subject_mae_comparison')

# ---------------------------------------------------------------------------
# Fig 6 — MAE by Sex for both methods
# ---------------------------------------------------------------------------
if 'Sex_label' in all_df.columns:
    sex_rows = []
    for sex in ['Male', 'Female']:
        g = all_df[all_df['Sex_label'] == sex]
        if len(g) == 0:
            continue
        sex_rows.append(dict(
            Sex=sex, n=g['PTID'].nunique(),
            MAE_direct =g['ae_direct'].mean(),  MAE_d_std=g['ae_direct'].std(),
            MAE_derived=g['ae_derived'].mean(), MAE_r_std=g['ae_derived'].std(),
        ))
    sex_cmp = pd.DataFrame(sex_rows)
    sex_cmp.to_csv(os.path.join(OUT, 'compare_by_sex.csv'), index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(sex_cmp))
    w = 0.35
    ax.bar(x - w/2, sex_cmp['MAE_direct'],  w, label='Direct BAG',
           color=METHOD_COLORS['Direct'],  alpha=0.82,
           yerr=sex_cmp['MAE_d_std'], capsize=5)
    ax.bar(x + w/2, sex_cmp['MAE_derived'], w, label='Derived BAG',
           color=METHOD_COLORS['Derived'], alpha=0.82,
           yerr=sex_cmp['MAE_r_std'], capsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['Sex']}\n(n={r['n']})" for _, r in sex_cmp.iterrows()],
                       fontsize=12)
    ax.set_ylabel('MAE (years)', fontsize=12)
    ax.set_title('MAE by Sex — Direct vs Derived BAG', fontsize=13)
    ax.legend(fontsize=11)
    plt.tight_layout()
    for ext in ('png', 'svg'):
        plt.savefig(os.path.join(OUT, f'fig6_error_by_sex.{ext}'),
                    dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved: fig6_error_by_sex')

# ---------------------------------------------------------------------------
# Fig 7 — MAE by Age group for both methods
# ---------------------------------------------------------------------------
if 'Age_group' in all_df.columns:
    age_rows = []
    for ag in AGE_LABELS:
        g = all_df[all_df['Age_group'].astype(str) == ag]
        if len(g) == 0:
            continue
        age_rows.append(dict(
            Age_group=ag, n=g['PTID'].nunique(),
            MAE_direct =g['ae_direct'].mean(),  MAE_d_std=g['ae_direct'].std(),
            MAE_derived=g['ae_derived'].mean(), MAE_r_std=g['ae_derived'].std(),
        ))
    age_cmp = pd.DataFrame(age_rows)
    age_cmp.to_csv(os.path.join(OUT, 'compare_by_age.csv'), index=False)

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(age_cmp))
    w = 0.35
    bar_colors = [AGE_COLORS.get(ag, '#888') for ag in age_cmp['Age_group']]
    ax.bar(x - w/2, age_cmp['MAE_direct'],  w, label='Direct BAG',
           color=bar_colors, alpha=0.82, yerr=age_cmp['MAE_d_std'], capsize=5)
    ax.bar(x + w/2, age_cmp['MAE_derived'], w, label='Derived BAG',
           color=bar_colors, alpha=0.45, yerr=age_cmp['MAE_r_std'], capsize=5,
           hatch='//')
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['Age_group']}\n(n={r['n']})"
                        for _, r in age_cmp.iterrows()], fontsize=11)
    ax.set_ylabel('MAE (years)', fontsize=12)
    ax.set_title('MAE by Age Group — Direct vs Derived BAG', fontsize=13)
    ax.legend(fontsize=11)
    plt.tight_layout()
    for ext in ('png', 'svg'):
        plt.savefig(os.path.join(OUT, f'fig7_error_by_age.{ext}'),
                    dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved: fig7_error_by_age')

# ---------------------------------------------------------------------------
# Fig 8 — 95% CI Coverage comparison
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Overall coverage per fold
ax = axes[0]
x = np.arange(len(stats_df))
ax.plot(x, stats_df['Cov_direct'],  'o-', color=METHOD_COLORS['Direct'],
        linewidth=2, markersize=8, label='Direct BAG')
ax.plot(x, stats_df['Cov_derived'], 's-', color=METHOD_COLORS['Derived'],
        linewidth=2, markersize=8, label='Derived BAG')
ax.axhline(95, color='black', linestyle='--', linewidth=1, alpha=0.5,
           label='Nominal 95%')
ax.set_xticks(x)
ax.set_xticklabels([f'Fold {i}' for i in stats_df['fold']])
ax.set_ylabel('Coverage (%)', fontsize=12)
ax.set_ylim(50, 105)
ax.set_title('95% CI Coverage per Fold', fontsize=12)
ax.legend(fontsize=10)

# Interval width per fold
ax = axes[1]
ax.plot(x, stats_df['IW_direct'],  'o-', color=METHOD_COLORS['Direct'],
        linewidth=2, markersize=8, label='Direct BAG')
ax.plot(x, stats_df['IW_derived'], 's-', color=METHOD_COLORS['Derived'],
        linewidth=2, markersize=8, label='Derived BAG')
ax.set_xticks(x)
ax.set_xticklabels([f'Fold {i}' for i in stats_df['fold']])
ax.set_ylabel('Mean Interval Width (years)', fontsize=12)
ax.set_title('Mean 95% CI Width per Fold', fontsize=12)
ax.legend(fontsize=10)

plt.suptitle('Uncertainty Calibration: Direct vs Derived BAG', fontsize=13)
plt.tight_layout()
for ext in ('png', 'svg'):
    plt.savefig(os.path.join(OUT, f'fig8_coverage.{ext}'), dpi=150, bbox_inches='tight')
plt.close()
print('Saved: fig8_coverage')

# ---------------------------------------------------------------------------
# 5.  Print & save interpretation
# ---------------------------------------------------------------------------
direct_wins_mae  = agg['MAE_direct'][0]  < agg['MAE_derived'][0]
direct_wins_rmse = agg['RMSE_direct'][0] < agg['RMSE_derived'][0]
direct_wins_r2   = agg['R2_direct'][0]   > agg['R2_derived'][0]
overall_winner   = 'Direct BAG model' if direct_wins_mae else 'Derived BAG (SPARE-BA − Age)'
sig_str = ('significant' if wpval < 0.05 else 'not significant')
sig_sym = '***' if wpval < 0.001 else '**' if wpval < 0.01 else '*' if wpval < 0.05 else 'n.s.'

lines = [
    '=' * 72,
    'BAG PREDICTION: DIRECT MODEL vs SPARE-BA-DERIVED  —  INTERPRETATION',
    '=' * 72,
    '',
    'METRIC DEFINITIONS',
    '  Direct BAG   : predicted directly by the DKGP BAG model',
    '  Derived BAG  : predicted_SPARE-BA  −  (BaselineAge + time_months/12)',
    '',
    'OVERALL PERFORMANCE  (all folds combined)',
    f'  Method         MAE (yr)   RMSE (yr)   R²      Coverage   CI Width',
    f'  Direct BAG     {agg["MAE_direct"][0]:.3f}      {agg["RMSE_direct"][0]:.3f}       '
    f'{agg["R2_direct"][0]:.3f}   {agg["Cov_direct"][0]:.1f}%      {stats_df["IW_direct"].mean():.3f} yr',
    f'  Derived BAG    {agg["MAE_derived"][0]:.3f}      {agg["RMSE_derived"][0]:.3f}       '
    f'{agg["R2_derived"][0]:.3f}   {agg["Cov_derived"][0]:.1f}%      {stats_df["IW_derived"].mean():.3f} yr',
    '',
    f'  Wilcoxon signed-rank test on per-observation |error|:',
    f'    p = {wpval:.4f}  ({sig_str})  {sig_sym}',
    '',
    f'WINNER: {overall_winner}',
    f'  (lower MAE {"✓" if direct_wins_mae else "✗"}  '
    f'lower RMSE {"✓" if direct_wins_rmse else "✗"}  '
    f'higher R² {"✓" if direct_wins_r2 else "✗"})',
    '',
    'SUBJECT-LEVEL BREAKDOWN',
    f'  Direct BAG is better for  {winner_counts.get("Direct",0)} / {len(subj)} subjects '
    f'({100*winner_counts.get("Direct",0)/len(subj):.0f}%)',
    f'  Derived BAG is better for {winner_counts.get("Derived",0)} / {len(subj)} subjects '
    f'({100*winner_counts.get("Derived",0)/len(subj):.0f}%)',
    '',
]

if 'Sex_label' in all_df.columns and len(sex_rows) >= 2:
    lines += ['BY SEX']
    for _, r in sex_cmp.iterrows():
        w_sex = 'Direct' if r['MAE_direct'] < r['MAE_derived'] else 'Derived'
        delta = abs(r['MAE_direct'] - r['MAE_derived'])
        lines.append(f"  {r['Sex']:7s} (n={r['n']:4d})  "
                     f"Direct={r['MAE_direct']:.3f}  Derived={r['MAE_derived']:.3f}  "
                     f"→ {w_sex} wins by {delta:.3f} yr")
    lines.append('')

if 'Age_group' in all_df.columns and len(age_rows) >= 2:
    lines += ['BY AGE GROUP']
    for _, r in age_cmp.iterrows():
        w_age = 'Direct' if r['MAE_direct'] < r['MAE_derived'] else 'Derived'
        delta = abs(r['MAE_direct'] - r['MAE_derived'])
        lines.append(f"  {r['Age_group']:7s} (n={r['n']:4d})  "
                     f"Direct={r['MAE_direct']:.3f}  Derived={r['MAE_derived']:.3f}  "
                     f"→ {w_age} wins by {delta:.3f} yr")
    lines.append('')

lines += [
    'UNCERTAINTY CALIBRATION',
    f'  Nominal 95% CI coverage — Direct: {agg["Cov_direct"][0]:.1f}%   '
    f'Derived: {agg["Cov_derived"][0]:.1f}%',
    f'  Mean CI width          — Direct: {stats_df["IW_direct"].mean():.3f} yr   '
    f'Derived: {stats_df["IW_derived"].mean():.3f} yr',
    '  (well-calibrated models should achieve ~95% coverage)',
    '',
    'BLAND-ALTMAN (agreement between the two methods)',
    f'  Bias (Direct − Derived) = {mean_diff:.3f} yr',
    f'  95% limits of agreement = [{loa_lower:.3f},  {loa_upper:.3f}] yr',
    '  A large LoA range means the two methods disagree substantially',
    '  on individual subjects even if population-level MAE is similar.',
    '',
    'RECOMMENDATION',
]

if abs(agg['MAE_direct'][0] - agg['MAE_derived'][0]) < 0.05:
    rec = ('Both methods perform comparably (MAE difference < 0.05 yr). '
           'Prefer the Direct BAG model when interpretability of the BAG '
           'trajectory itself matters. Use the Derived approach when a single '
           'SPARE-BA model is sufficient and you want to avoid training a '
           'separate BAG model.')
elif direct_wins_mae:
    delta = agg['MAE_derived'][0] - agg['MAE_direct'][0]
    rec = (f'The Direct BAG model is {delta:.3f} yr more accurate on average. '
           'Training a dedicated BAG model is worthwhile. The derived approach '
           'accumulates error from both SPARE-BA prediction and the age-subtraction '
           'step, particularly for older subjects where age variance is higher.')
else:
    delta = agg['MAE_direct'][0] - agg['MAE_derived'][0]
    rec = (f'The Derived BAG (SPARE-BA − Age) is {delta:.3f} yr more accurate on average. '
           'This suggests that the SPARE-BA model captures brain aging dynamics '
           'more precisely than the direct BAG model, and the age-subtraction '
           'step does not add significant noise.')

for line in textwrap.wrap(rec, width=70):
    lines.append('  ' + line)
lines += ['', '=' * 72]

interp_text = '\n'.join(lines)
print('\n' + interp_text)

with open(os.path.join(OUT, 'interpretation.txt'), 'w') as fh:
    fh.write(interp_text + '\n')
print(f'\nInterpretation saved to {os.path.join(OUT, "interpretation.txt")}')
print(f'All outputs in: {OUT}')
