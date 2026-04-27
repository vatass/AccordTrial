"""
evaluate_bag.py — Comprehensive evaluation of DKGP BAG models

Covers:
  Part 1  5-fold cross-validation evaluation (held-out test sets)
  Part 2  ACCORD observed-timepoint predictions (ensemble across folds)
  Part 3  ACCORD 8-year prospective forecast (ensemble across folds)

Usage:
  python evaluate_bag.py [options]

Reads from:
  models/bag_fold{i}/predictions_BAG_0_{i}.csv
  models/bag_fold{i}/accord_predictions_BAG_0_{i}.csv
  models/bag_fold{i}/accord_eight_year_forecast_BAG_0_{i}.csv
  data/accord_data_bag_processed.csv
  data/normalization_stats.pkl
"""

import argparse
import os
import pickle

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Evaluate DKGP BAG models')
parser.add_argument('--models_dir',      default='models')
parser.add_argument('--data_dir',        default='data')
parser.add_argument('--output_dir',      default='analysis/bag_evaluation')
parser.add_argument('--n_folds',         type=int, default=5)
parser.add_argument('--biomarker',       default='BAG')
parser.add_argument('--biomarker_index', type=int, default=0)
parser.add_argument('--n_traj',          type=int, default=20,
                    help='Number of individual trajectories to show in grid plots')
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)
B  = args.biomarker
BI = args.biomarker_index

# ---------------------------------------------------------------------------
# Publication-quality style (Okabe-Ito colorblind palette)
# ---------------------------------------------------------------------------
plt.rcParams.update({
    'font.family':        'sans-serif',
    'font.size':          9,
    'axes.titlesize':     10,
    'axes.labelsize':     9,
    'xtick.labelsize':    8,
    'ytick.labelsize':    8,
    'legend.fontsize':    8,
    'lines.linewidth':    1.2,
    'axes.linewidth':     0.8,
    'xtick.major.width':  0.8,
    'ytick.major.width':  0.8,
    'figure.dpi':         150,
    'savefig.dpi':        300,
    'savefig.bbox':       'tight',
    'savefig.pad_inches': 0.1,
})

C_BLUE   = '#0072B2'
C_ORANGE = '#E69F00'
C_GREEN  = '#009E73'
C_RED    = '#D55E00'
C_PURPLE = '#CC79A7'
C_BLACK  = '#333333'


def despine(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def savefig(fig, name):
    path = os.path.join(args.output_dir, name)
    fig.savefig(path)
    plt.close(fig)
    print(f'  Saved {name}')


# ---------------------------------------------------------------------------
# Load normalization stats (BAG mean/std from training set)
# ---------------------------------------------------------------------------
bag_mean, bag_std = 0.0, 1.0
age_mean, age_std = 65.0, 10.0  # sensible fallback if pkl missing
norm_path = os.path.join(args.data_dir, 'normalization_stats.pkl')
if os.path.exists(norm_path):
    with open(norm_path, 'rb') as f:
        ns = pickle.load(f)
    if 'BAG' in ns:
        bag_mean = float(ns['BAG']['mean'])
        bag_std  = float(ns['BAG']['std'])
    if 'Age' in ns:
        age_mean = float(ns['Age']['mean'])
        age_std  = float(ns['Age']['std'])
    print(f'Normalization stats: BAG mean={bag_mean:.3f} yr, std={bag_std:.3f} yr')
    print(f'                     Age mean={age_mean:.3f} yr, std={age_std:.3f} yr')
else:
    print(f'WARNING: {norm_path} not found — predictions assumed already in years')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_fold_files(prefix):
    """Concatenate models/bag_fold{i}/{prefix}_{B}_{BI}_{i}.csv for all folds."""
    frames = []
    for fold in range(args.n_folds):
        path = os.path.join(args.models_dir, f'bag_fold{fold}',
                            f'{prefix}_{B}_{BI}_{fold}.csv')
        if os.path.exists(path):
            df = pd.read_csv(path)
            df['fold'] = fold
            frames.append(df)
        else:
            print(f'  [missing] {path}')
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def print_metrics(df, label):
    mae  = df['abs_error'].mean()
    rmse = np.sqrt(df['squared_error'].mean())
    r2   = r2_score(df['ground_truth'], df['predicted'])
    cov  = df['covered'].mean()
    ciw  = df['interval_width'].mean()
    print(f'\n{label}')
    print(f'  Subjects     : {df["PTID"].nunique()}')
    print(f'  Observations : {len(df)}')
    print(f'  MAE          : {mae:.3f} yr')
    print(f'  RMSE         : {rmse:.3f} yr')
    print(f'  R²           : {r2:.4f}')
    print(f'  Coverage 90% : {cov:.3f}')
    print(f'  Mean CI width: {ciw:.3f} yr')
    return dict(mae=mae, rmse=rmse, r2=r2, coverage=cov, ci_width=ciw,
                n_subj=df['PTID'].nunique(), n_obs=len(df))


def ensemble_predictions(df):
    """Average per (PTID, time_months) across folds."""
    agg = {
        'ground_truth':   ('ground_truth',   'mean'),
        'predicted':      ('predicted',       'mean'),
        'lower_bound':    ('lower_bound',     'mean'),
        'upper_bound':    ('upper_bound',     'mean'),
        'interval_width': ('interval_width',  'mean'),
        'abs_error':      ('abs_error',       'mean'),
        'squared_error':  ('squared_error',   'mean'),
        'covered':        ('covered',         'mean'),
    }
    return df.groupby(['PTID', 'time_months'], as_index=False).agg(**agg)


def pred_vs_obs_fig(df, color, title, fname):
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.scatter(df['ground_truth'], df['predicted'],
               s=5, alpha=0.25, color=color, linewidths=0, rasterized=True)
    lo = min(df['ground_truth'].min(), df['predicted'].min()) - 1
    hi = max(df['ground_truth'].max(), df['predicted'].max()) + 1
    ax.plot([lo, hi], [lo, hi], color=C_BLACK, lw=0.8, ls='--')
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel('Observed BAG (years)')
    ax.set_ylabel('Predicted BAG (years)')
    ax.set_title(title)
    m = print_metrics.__wrapped__(df) if hasattr(print_metrics, '__wrapped__') else None
    mae  = df['abs_error'].mean()
    rmse = np.sqrt(df['squared_error'].mean())
    r2   = r2_score(df['ground_truth'], df['predicted'])
    ax.text(0.05, 0.95,
            f'MAE = {mae:.2f} yr\nRMSE = {rmse:.2f} yr\nR² = {r2:.3f}',
            transform=ax.transAxes, va='top', fontsize=7.5,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='none', alpha=0.85))
    despine(ax)
    savefig(fig, fname)


def trajectory_grid(df, n, color, suptitle, fname):
    """Grid of individual longitudinal trajectories (subjects with >1 timepoint)."""
    multi = df.groupby('PTID').filter(lambda g: len(g) > 1)['PTID'].unique()
    if len(multi) == 0:
        print(f'  (no multi-timepoint subjects for {fname})')
        return
    rng = np.random.RandomState(42)
    ptids = rng.choice(multi, size=min(n, len(multi)), replace=False)
    n_cols = 5
    n_rows = int(np.ceil(len(ptids) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 2.6, n_rows * 2.2), squeeze=False)
    for idx, ptid in enumerate(ptids):
        ax = axes[idx // n_cols][idx % n_cols]
        s  = df[df['PTID'] == ptid].sort_values('time_months')
        t  = s['time_months'] / 12
        ax.fill_between(t, s['lower_bound'], s['upper_bound'], color=color, alpha=0.2)
        ax.plot(t, s['predicted'],    color=color, lw=1.4, label='Predicted')
        ax.scatter(t, s['ground_truth'], s=16, color=C_BLACK, zorder=3, label='Observed')
        ax.set_title(str(ptid)[:14], fontsize=6.5)
        ax.set_xlabel('Time (yr)', fontsize=7)
        ax.set_ylabel('BAG (yr)',  fontsize=7)
        ax.tick_params(labelsize=6)
        despine(ax)
    for idx in range(len(ptids), n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=2, frameon=False,
               fontsize=8, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(suptitle, fontsize=10)
    fig.tight_layout()
    savefig(fig, fname)


def error_vs_time_fig(df, color, title, fname):
    """Mean ± SD absolute error binned by year, with jittered scatter.

    time_months can be irregular (e.g. 33, 34, 35 …); we round to the
    nearest integer year so the x-axis stays clean.
    """
    df = df.copy()
    df['year_bin'] = (df['time_months'] / 12).round().astype(int)

    bins   = sorted(df['year_bin'].unique())
    means  = [df[df['year_bin'] == b]['abs_error'].mean()  for b in bins]
    sds    = [df[df['year_bin'] == b]['abs_error'].std()   for b in bins]
    counts = [df[df['year_bin'] == b]['abs_error'].count() for b in bins]

    fig, ax = plt.subplots(figsize=(6, 3.8))
    rng = np.random.RandomState(1)
    for b in bins:
        y      = df[df['year_bin'] == b]['abs_error'].values
        jitter = rng.uniform(-0.25, 0.25, size=len(y))
        ax.scatter(b + jitter, y,
                   s=4, alpha=0.18, color=color, linewidths=0, rasterized=True)

    ax.errorbar(bins, means, yerr=sds,
                fmt='o', color=C_BLACK, ms=5, lw=1.2, capsize=3,
                label='Mean ± SD')
    ax.plot(bins, means, color=C_BLACK, lw=1, alpha=0.6)

    ax.set_xlabel('Time from baseline (years)')
    ax.set_ylabel('Absolute Error (years)')
    ax.set_title(title)
    ax.set_xticks(bins)
    ax.set_xticklabels(
        [f'{"BL" if b == 0 else str(b)}\n(n={n})' for b, n in zip(bins, counts)],
        fontsize=7,
    )
    ax.legend(frameon=False)
    despine(ax)
    fig.tight_layout()
    savefig(fig, fname)


def error_by_age_fig(df, color, title, fname, age_col='BaselineAge'):
    """Box-plot of absolute error stratified by age group (5 bins)."""
    if age_col not in df.columns or df[age_col].isna().all():
        print(f'  (skipping {fname} — {age_col} not available)')
        return

    df = df.copy()
    # Age is stored as a z-score; denormalize to years
    df['age_yr'] = df[age_col] * age_std + age_mean

    bins   = [0,  55, 65, 75, 85, 200]
    labels = ['<55', '55–65', '65–75', '75–85', '≥85']
    df['age_group'] = pd.cut(df['age_yr'], bins=bins, labels=labels, right=False)

    groups = [df[df['age_group'] == lbl]['abs_error'].dropna().values
              for lbl in labels]
    counts = [len(g) for g in groups]

    # Build all box/scatter data in one pass to guarantee consistent lengths
    nonempty = [(i, g) for i, g in enumerate(groups) if len(g) > 0]
    if not nonempty:
        print(f'  (skipping {fname} — no data after age binning)')
        return
    bp_positions = [i for i, _ in nonempty]
    bp_data      = [g for _, g in nonempty]
    bp_means     = [g.mean() for _, g in nonempty]

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    rng = np.random.RandomState(2)
    for i, g in nonempty:
        jitter = rng.uniform(-0.18, 0.18, size=len(g))
        ax.scatter(i + jitter, g, s=4, alpha=0.18,
                   color=color, linewidths=0, rasterized=True)

    ax.boxplot(
        bp_data,
        positions=bp_positions,
        widths=0.45, patch_artist=True,
        medianprops=dict(color=C_RED, lw=1.8),
        boxprops=dict(facecolor='none', edgecolor=C_BLACK, lw=0.9),
        whiskerprops=dict(color=C_BLACK, lw=0.8),
        capprops=dict(color=C_BLACK, lw=0.8),
        flierprops=dict(marker='', markersize=0),
    )

    # Mean markers
    ax.scatter(bp_positions, bp_means,
               marker='D', s=22, color=C_BLACK, zorder=5, label='Mean')

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([f'{lbl}\n(n={n})' for lbl, n in zip(labels, counts)],
                       fontsize=8)
    ax.set_xlabel('Baseline Age Group (years)')
    ax.set_ylabel('Absolute Error (years)')
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    despine(ax)
    fig.tight_layout()
    savefig(fig, fname)


def tp_xtick_labels(tps_yr):
    return ['BL' if t == 0 else str(int(t)) for t in tps_yr]


# ===========================================================================
# PART 1 — 5-Fold Cross-Validation
# ===========================================================================
print('\n' + '=' * 62)
print('  PART 1 — 5-Fold Cross-Validation')
print('=' * 62)

cv_raw = load_fold_files('predictions')
if cv_raw is None:
    print('No CV prediction files found — skipping Part 1.')
else:
    cv_raw['PTID'] = cv_raw['PTID'].astype(str)

    # Pool across folds (each subject appears in exactly one test fold)
    cv = cv_raw.copy()
    m_cv = print_metrics(cv, 'Pooled 5-fold CV (all test observations)')

    # Per-fold breakdown
    print('\nPer-fold breakdown:')
    fold_rows = []
    for fold, g in cv.groupby('fold'):
        row = dict(
            fold   = int(fold),
            n_subj = g['PTID'].nunique(),
            n_obs  = len(g),
            mae    = g['abs_error'].mean(),
            rmse   = np.sqrt(g['squared_error'].mean()),
            r2     = r2_score(g['ground_truth'], g['predicted']),
            cov    = g['covered'].mean(),
            ciw    = g['interval_width'].mean(),
        )
        fold_rows.append(row)
        print(f"  fold {fold}: n={row['n_subj']:3d}  "
              f"MAE={row['mae']:.3f}  RMSE={row['rmse']:.3f}  "
              f"R²={row['r2']:.4f}  Cov={row['cov']:.3f}")
    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(os.path.join(args.output_dir, 'cv_fold_metrics.csv'), index=False)

    # Fig 1 — Predicted vs. Observed
    pred_vs_obs_fig(cv, C_BLUE, '5-Fold CV: Predicted vs. Observed',
                    'cv_fig1_pred_vs_obs.png')

    # Fig 2 — Absolute error distribution
    fig, ax = plt.subplots(figsize=(4.5, 3))
    ax.hist(cv['abs_error'], bins=50, color=C_BLUE, edgecolor='none', alpha=0.8)
    med = cv['abs_error'].median()
    ax.axvline(med, color=C_RED, lw=1.3, label=f'Median = {med:.2f} yr')
    ax.set_xlabel('Absolute Error (years)')
    ax.set_ylabel('Observations')
    ax.set_title('5-Fold CV: Absolute Error Distribution')
    ax.legend(frameon=False)
    despine(ax)
    savefig(fig, 'cv_fig2_error_dist.png')

    # Fig 3 — Per-fold metric bars (MAE / RMSE / R²)
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    for ax, (col, ylabel) in zip(axes, [('mae', 'MAE (yr)'), ('rmse', 'RMSE (yr)'), ('r2', 'R²')]):
        vals = fold_df[col].values
        ax.bar(fold_df['fold'], vals, color=C_BLUE, alpha=0.8, width=0.55)
        ax.axhline(vals.mean(), color=C_RED, lw=1.1, ls='--',
                   label=f'mean = {vals.mean():.3f}')
        ax.set_xlabel('Fold'); ax.set_ylabel(ylabel); ax.set_title(ylabel)
        ax.set_xticks(fold_df['fold'])
        ax.legend(frameon=False, fontsize=7)
        despine(ax)
    fig.suptitle('Per-Fold Metrics — 5-Fold CV', fontsize=10, y=1.02)
    fig.tight_layout()
    savefig(fig, 'cv_fig3_fold_metrics.png')

    # Fig 4 — Sample longitudinal trajectories
    trajectory_grid(cv, args.n_traj, C_BLUE,
                    '5-Fold CV: Sample Longitudinal Trajectories',
                    'cv_fig4_trajectories.png')

    # Fig 5 — Absolute error vs. time
    error_vs_time_fig(cv, C_BLUE,
                      '5-Fold CV: Absolute Error vs. Time',
                      'cv_fig5_error_vs_time.png')

    # Fig 6 — Absolute error by age group
    error_by_age_fig(cv, C_BLUE,
                     '5-Fold CV: Absolute Error by Age Group',
                     'cv_fig6_error_by_age.png',
                     age_col='BaselineAge')

# ===========================================================================
# PART 2 — ACCORD Observed-Timepoint Predictions
# ===========================================================================
print('\n' + '=' * 62)
print('  PART 2 — ACCORD Observed-Timepoint Predictions')
print('=' * 62)

accord_obs_raw = load_fold_files('accord_predictions')
if accord_obs_raw is None:
    print('No ACCORD observed prediction files found — skipping Part 2.')
else:
    accord_obs_raw['PTID'] = accord_obs_raw['PTID'].astype(str)
    accord_obs = ensemble_predictions(accord_obs_raw)

    accord_obs_path = os.path.join(args.output_dir, 'accord_observed_ensemble.csv')
    accord_obs.to_csv(accord_obs_path, index=False)
    print(f'  Saved accord_observed_ensemble.csv  '
          f'({accord_obs["PTID"].nunique()} subjects, {len(accord_obs)} observations)')

    m_acc_obs = print_metrics(accord_obs, 'ACCORD Observed (5-fold ensemble)')

    # Merge baseline age from demographics (stored as z-score; helper denormalizes)
    demo_path = os.path.join(args.data_dir, 'accord_data_bag_processed.csv')
    if os.path.exists(demo_path):
        demo = pd.read_csv(demo_path)
        demo['PTID'] = demo['PTID'].astype(str)
        if 'Age' in demo.columns:
            baseline_age = (demo.sort_values('Time')
                                .groupby('PTID', as_index=False)
                                .first()[['PTID', 'Age']]
                                .rename(columns={'Age': 'BaselineAge'}))
            accord_obs = accord_obs.merge(baseline_age, on='PTID', how='left')

    # Per-fold breakdown
    print('\nPer-fold breakdown:')
    for fold, g in accord_obs_raw.groupby('fold'):
        mae  = g['abs_error'].mean()
        rmse = np.sqrt(g['squared_error'].mean())
        r2   = r2_score(g['ground_truth'], g['predicted'])
        cov  = g['covered'].mean()
        print(f"  fold {fold}: n={g['PTID'].nunique():3d}  "
              f"MAE={mae:.3f}  RMSE={rmse:.3f}  R²={r2:.4f}  Cov={cov:.3f}")

    # Fig 1 — Predicted vs. Observed
    pred_vs_obs_fig(accord_obs, C_GREEN,
                    'ACCORD: Predicted vs. Observed (ensemble)',
                    'accord_obs_fig1_pred_vs_obs.png')

    # Fig 2 — Sample ACCORD trajectories
    trajectory_grid(accord_obs, args.n_traj, C_GREEN,
                    'ACCORD: Sample Longitudinal Trajectories',
                    'accord_obs_fig2_trajectories.png')

    # Fig 3 — Absolute error vs. time
    error_vs_time_fig(accord_obs, C_GREEN,
                      'ACCORD Observed: Absolute Error vs. Time',
                      'accord_obs_fig3_error_vs_time.png')

    # Fig 4 — Absolute error by age group
    error_by_age_fig(accord_obs, C_GREEN,
                     'ACCORD Observed: Absolute Error by Age Group',
                     'accord_obs_fig4_error_by_age.png',
                     age_col='BaselineAge')

# ===========================================================================
# PART 3 — ACCORD 8-Year Prospective Forecast
# ===========================================================================
print('\n' + '=' * 62)
print('  PART 3 — ACCORD 8-Year Prospective Forecast')
print('=' * 62)

forecast_raw = load_fold_files('accord_eight_year_forecast')
if forecast_raw is None:
    print('No 8-year forecast files found — skipping Part 3.')
else:
    forecast_raw['PTID'] = forecast_raw['PTID'].astype(str)

    # Ensemble across folds
    forecast = (forecast_raw
                .groupby(['PTID', 'time_months'], as_index=False)
                .agg(predicted      = ('predicted',      'mean'),
                     lower_bound    = ('lower_bound',    'mean'),
                     upper_bound    = ('upper_bound',    'mean'),
                     variance       = ('variance',       'mean'),
                     interval_width = ('interval_width', 'mean')))

    n_subj = forecast['PTID'].nunique()
    n_folds_found = forecast_raw['fold'].nunique()
    print(f'\n8-year forecast ensemble: {n_subj} subjects, '
          f'{len(forecast)} rows ({n_folds_found} folds)')

    forecast.to_csv(os.path.join(args.output_dir, 'accord_forecast_ensemble.csv'), index=False)
    print('  Saved accord_forecast_ensemble.csv')

    # Load sex from demographics for stratification
    sex_map = None
    demo_path = os.path.join(args.data_dir, 'accord_data_bag_processed.csv')
    if os.path.exists(demo_path):
        demo = pd.read_csv(demo_path)
        demo['PTID'] = demo['PTID'].astype(str)
        sex_map = (demo.sort_values('Time')
                       .groupby('PTID', as_index=False)
                       .first()[['PTID', 'Sex']])
        forecast = forecast.merge(sex_map, on='PTID', how='left')
        n_sex = forecast['Sex'].notna().sum()
        print(f'  Sex mapped for {n_sex}/{len(forecast)} rows')

    tps     = sorted(forecast['time_months'].unique())
    tps_yr  = [t / 12 for t in tps]
    xlabels = tp_xtick_labels(tps_yr)

    # Population summary per timepoint
    pop = (forecast
           .groupby('time_months', as_index=False)
           .agg(mean_pred = ('predicted',   'mean'),
                q10       = ('predicted',   lambda x: np.percentile(x, 10)),
                q90       = ('predicted',   lambda x: np.percentile(x, 90)),
                mean_lo   = ('lower_bound', 'mean'),
                mean_hi   = ('upper_bound', 'mean')))
    t_yr = pop['time_months'] / 12

    # Fig 1 — Population mean trajectory
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.fill_between(t_yr, pop['q10'], pop['q90'],
                    color=C_BLUE, alpha=0.12, label='10th–90th pctile')
    ax.fill_between(t_yr, pop['mean_lo'], pop['mean_hi'],
                    color=C_BLUE, alpha=0.28, label='Mean 90% CI')
    ax.plot(t_yr, pop['mean_pred'], color=C_BLUE, lw=2, label='Population mean')
    ax.set_xlabel('Time from baseline (years)')
    ax.set_ylabel('Predicted BAG (years)')
    ax.set_title(f'ACCORD 8-Year BAG Forecast — Population Trajectory\n'
                 f'(n = {n_subj} subjects, {n_folds_found}-fold ensemble)')
    ax.set_xticks(tps_yr); ax.set_xticklabels(xlabels)
    ax.legend(frameon=False, loc='upper left', fontsize=8)
    despine(ax)
    fig.tight_layout()
    savefig(fig, 'accord_forecast_fig1_population.png')

    # Fig 2 — Sex-stratified trajectories
    if sex_map is not None and forecast['Sex'].notna().any():
        fig, ax = plt.subplots(figsize=(5.5, 3.8))
        for sex_val, label, color in [(0, 'Male', C_BLUE), (1, 'Female', C_RED)]:
            sub = forecast[forecast['Sex'] == sex_val]
            if sub.empty:
                continue
            g = (sub.groupby('time_months', as_index=False)
                    .agg(mean_pred = ('predicted',   'mean'),
                         mean_lo   = ('lower_bound', 'mean'),
                         mean_hi   = ('upper_bound', 'mean')))
            t = g['time_months'] / 12
            ax.fill_between(t, g['mean_lo'], g['mean_hi'], color=color, alpha=0.2)
            ax.plot(t, g['mean_pred'], color=color, lw=2,
                    label=f'{label} (n={sub["PTID"].nunique()})')
        ax.set_xlabel('Time from baseline (years)')
        ax.set_ylabel('Predicted BAG (years)')
        ax.set_title('ACCORD 8-Year Forecast: Sex-Stratified')
        ax.set_xticks(tps_yr); ax.set_xticklabels(xlabels)
        ax.legend(frameon=False)
        despine(ax)
        fig.tight_layout()
        savefig(fig, 'accord_forecast_fig2_sex_stratified.png')

    # Fig 3 — Individual spaghetti (random sample) + population mean
    rng = np.random.RandomState(0)
    sample_ptids = rng.choice(forecast['PTID'].unique(),
                              size=min(args.n_traj * 4, n_subj), replace=False)
    fig, ax = plt.subplots(figsize=(6, 4))
    for ptid in sample_ptids:
        s = forecast[forecast['PTID'] == ptid].sort_values('time_months')
        ax.plot(s['time_months'] / 12, s['predicted'],
                color=C_BLUE, lw=0.4, alpha=0.25)
    ax.plot(t_yr, pop['mean_pred'], color=C_BLACK, lw=2,
            label=f'Population mean (n={n_subj})', zorder=5)
    ax.set_xlabel('Time from baseline (years)')
    ax.set_ylabel('Predicted BAG (years)')
    ax.set_title(f'ACCORD 8-Year Forecast: Individual Trajectories\n'
                 f'({len(sample_ptids)} of {n_subj} subjects shown)')
    ax.set_xticks(tps_yr); ax.set_xticklabels(xlabels)
    ax.legend(frameon=False)
    despine(ax)
    fig.tight_layout()
    savefig(fig, 'accord_forecast_fig3_individual.png')

    # Fig 4 — Distribution over time (violin)
    data_by_tp = [forecast[forecast['time_months'] == tp]['predicted'].values
                  for tp in tps]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    parts = ax.violinplot(data_by_tp, positions=tps_yr, widths=0.5,
                          showmedians=True, showextrema=False)
    for pc in parts['bodies']:
        pc.set_facecolor(C_BLUE)
        pc.set_alpha(0.55)
    parts['cmedians'].set_color(C_RED)
    parts['cmedians'].set_linewidth(1.5)
    ax.set_xlabel('Time from baseline (years)')
    ax.set_ylabel('Predicted BAG (years)')
    ax.set_title('ACCORD Forecast: Predicted BAG Distribution Over Time')
    ax.set_xticks(tps_yr); ax.set_xticklabels(xlabels)
    despine(ax)
    fig.tight_layout()
    savefig(fig, 'accord_forecast_fig4_distribution.png')

    # Fig 5 — Prediction uncertainty (CI width) over time
    # No ground truth available for the forecast, so CI width shows how
    # model uncertainty grows as predictions extend further from baseline.
    ciw_by_tp = (forecast
                 .groupby('time_months', as_index=False)
                 .agg(mean_ciw = ('interval_width', 'mean'),
                      sd_ciw   = ('interval_width', 'std')))
    t_ciw = ciw_by_tp['time_months'] / 12

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.fill_between(t_ciw,
                    ciw_by_tp['mean_ciw'] - ciw_by_tp['sd_ciw'],
                    ciw_by_tp['mean_ciw'] + ciw_by_tp['sd_ciw'],
                    color=C_ORANGE, alpha=0.25)
    ax.plot(t_ciw, ciw_by_tp['mean_ciw'], color=C_ORANGE, lw=2,
            marker='o', ms=4, label='Mean CI width')
    ax.set_xlabel('Time from baseline (years)')
    ax.set_ylabel('90% CI Width (years)')
    ax.set_title('ACCORD Forecast: Prediction Uncertainty Over Time')
    ax.set_xticks(tps_yr); ax.set_xticklabels(xlabels)
    ax.legend(frameon=False)
    despine(ax)
    fig.tight_layout()
    savefig(fig, 'accord_forecast_fig5_uncertainty_vs_time.png')

    # ------------------------------------------------------------------
    # Fig 6 — Individual subject trajectories (Nature-quality, 1 per file)
    # 10 random subjects; special markers at 48 m (4 yr) and 96 m (8 yr).
    # ------------------------------------------------------------------
    NATURE_W   = 3.504   # 89 mm — single Nature column, inches
    NATURE_H   = 3.0
    NATURE_DPI = 600

    # Nature-specific rcParams override for this block only
    nature_rc = {
        'font.family':        'sans-serif',
        'font.sans-serif':    ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size':          7,
        'axes.titlesize':     8,
        'axes.labelsize':     7,
        'xtick.labelsize':    6,
        'ytick.labelsize':    6,
        'legend.fontsize':    6,
        'lines.linewidth':    1.2,
        'axes.linewidth':     0.6,
        'xtick.major.width':  0.6,
        'ytick.major.width':  0.6,
        'xtick.major.size':   2.5,
        'ytick.major.size':   2.5,
        'savefig.dpi':        NATURE_DPI,
        'savefig.bbox':       'tight',
        'savefig.pad_inches': 0.04,
    }

    # Key timepoints of interest (months → years)
    KEY_MONTHS = {48: '4 yr', 96: '8 yr'}
    KEY_COLORS = {48: C_RED, 96: C_ORANGE}

    subj_dir = os.path.join(args.output_dir, 'accord_subject_trajectories')
    os.makedirs(subj_dir, exist_ok=True)

    rng_s = np.random.RandomState(7)
    subject_ptids = rng_s.choice(forecast['PTID'].unique(), size=10, replace=False)

    with plt.rc_context(nature_rc):
        for ptid in subject_ptids:
            s = forecast[forecast['PTID'] == ptid].sort_values('time_months')
            t = s['time_months'] / 12

            fig, ax = plt.subplots(figsize=(NATURE_W, NATURE_H))

            # 90 % CI band
            ax.fill_between(t, s['lower_bound'], s['upper_bound'],
                            color=C_BLUE, alpha=0.18, linewidth=0)

            # Trajectory line
            ax.plot(t, s['predicted'], color=C_BLUE, lw=1.4, zorder=3)

            # Regular timepoint markers (small, open)
            regular_mask = ~s['time_months'].isin(KEY_MONTHS)
            ax.scatter(t[regular_mask],
                       s.loc[regular_mask, 'predicted'],
                       s=12, color=C_BLUE, zorder=4,
                       edgecolors='white', linewidths=0.4)

            # Special markers and annotations at 48 m and 96 m
            for km, klabel in KEY_MONTHS.items():
                kc = KEY_COLORS[km]
                row = s[s['time_months'] == km]
                if row.empty:
                    continue
                t_k   = float(row['time_months'].iloc[0]) / 12
                y_k   = float(row['predicted'].iloc[0])
                y_lo  = float(row['lower_bound'].iloc[0])
                y_hi  = float(row['upper_bound'].iloc[0])

                # Vertical reference line
                ax.axvline(t_k, color=kc, lw=0.7, ls='--', alpha=0.55, zorder=2)

                # Star marker
                ax.scatter(t_k, y_k, marker='*', s=80, color=kc,
                           edgecolors='white', linewidths=0.4, zorder=6)

                # Value annotation (above or below depending on space)
                y_range = s['upper_bound'].max() - s['lower_bound'].min()
                offset  = y_range * 0.10
                va      = 'bottom'
                y_ann   = y_hi + offset * 0.4
                ax.annotate(
                    f'{klabel}\n{y_k:+.1f} yr',
                    xy=(t_k, y_k), xytext=(t_k, y_ann),
                    ha='center', va=va, fontsize=5.5, color=kc,
                    arrowprops=dict(arrowstyle='-', color=kc,
                                   lw=0.5, alpha=0.7),
                )

            # Axes
            ax.set_xlabel('Time from baseline (years)')
            ax.set_ylabel('Brain Age Gap (years)')
            ax.set_xticks(tps_yr)
            ax.set_xticklabels(xlabels)
            ax.set_title(f'Subject {ptid}', pad=4)

            # Subtle legend for key timepoints only
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color=C_BLUE, lw=1.4, label='Predicted BAG'),
                Line2D([0], [0], color=C_BLUE, lw=0, marker='*',
                       markersize=6, markerfacecolor=C_RED,
                       label='4-yr & 8-yr marks'),
            ]
            ax.legend(handles=legend_elements, frameon=False,
                      loc='upper left', handlelength=1.2)

            despine(ax)
            fig.tight_layout()

            fname_subj = os.path.join(subj_dir, f'subject_{ptid}_forecast.png')
            fig.savefig(fname_subj)
            plt.close(fig)

    print(f'  Saved 10 individual subject trajectories → {subj_dir}/')

    # Companion panel: 2 × 5 grid of the same 10 subjects (one figure for
    # convenient manuscript review)
    with plt.rc_context(nature_rc):
        fig, axes = plt.subplots(2, 5,
                                 figsize=(NATURE_W * 5 + 0.3, NATURE_H * 2 + 0.3),
                                 sharey=False)
        for ax, ptid in zip(axes.flat, subject_ptids):
            s = forecast[forecast['PTID'] == ptid].sort_values('time_months')
            t = s['time_months'] / 12

            ax.fill_between(t, s['lower_bound'], s['upper_bound'],
                            color=C_BLUE, alpha=0.18, linewidth=0)
            ax.plot(t, s['predicted'], color=C_BLUE, lw=1.2, zorder=3)

            regular_mask = ~s['time_months'].isin(KEY_MONTHS)
            ax.scatter(t[regular_mask],
                       s.loc[regular_mask, 'predicted'],
                       s=9, color=C_BLUE, zorder=4,
                       edgecolors='white', linewidths=0.3)

            for km, klabel in KEY_MONTHS.items():
                kc  = KEY_COLORS[km]
                row = s[s['time_months'] == km]
                if row.empty:
                    continue
                t_k = float(row['time_months'].iloc[0]) / 12
                y_k = float(row['predicted'].iloc[0])
                ax.axvline(t_k, color=kc, lw=0.6, ls='--', alpha=0.5, zorder=2)
                ax.scatter(t_k, y_k, marker='*', s=55, color=kc,
                           edgecolors='white', linewidths=0.3, zorder=6)

            ax.set_xticks(tps_yr)
            ax.set_xticklabels(xlabels, fontsize=5)
            ax.set_title(f'{ptid}', fontsize=6.5, pad=2)
            ax.set_xlabel('Time (yr)', fontsize=6)
            ax.set_ylabel('BAG (yr)',  fontsize=6)
            despine(ax)

        fig.suptitle('ACCORD 8-Year BAG Forecast — Individual Trajectories\n'
                     '★ = 4-yr and 8-yr key timepoints',
                     fontsize=8, y=1.01)
        fig.tight_layout(h_pad=1.2, w_pad=0.8)
        savefig(fig, 'accord_forecast_fig6_subject_panel.png')

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print('\n' + '=' * 62)
print(f'  Outputs written to: {args.output_dir}/')
print('=' * 62)
