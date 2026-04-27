'''
ACCORD BAG Trajectory Analysis
Reads the per-fold ACCORD prediction CSVs written by dkgp_training.py,
ensembles across folds, and produces:
  Fig 1 — population spaghetti + mean trajectory + observed BAG
  Fig 2 — sex-stratified trajectories
  Fig 3 — individual subject trajectory grid (random sample)
  Fig 4 — predicted vs observed scatter at measured timepoints
  Fig 5 — predicted BAG distribution (violin) over time
  ensemble CSV saved alongside the figures

Prediction CSV columns (denormalized, in years):
  PTID, time_months, ground_truth, predicted, lower_bound, upper_bound,
  variance, interval_width, abs_error, squared_error, covered
'''

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Analyse ACCORD BAG predictions from dkgp_training.py')
parser.add_argument('--models_dir',   default='models',
                    help='Root dir containing bag_fold{i}/ subdirectories')
parser.add_argument('--accord_data',  default='data/accord_data_bag_processed.csv',
                    help='Flat ACCORD CSV with PTID, Time, Sex, Age (written by accord_data.py)')
parser.add_argument('--covariates_file', default='data/longitudinal_covariates_bag_allstudies.csv',
                    help='Longitudinal covariates CSV (fallback source of Sex/Age)')
parser.add_argument('--output_dir',   default='analysis/accord_bag')
parser.add_argument('--n_folds',      type=int, default=5)
parser.add_argument('--biomarker',    default='BAG')
parser.add_argument('--biomarker_index', type=int, default=0)
parser.add_argument('--n_traj',       type=int, default=20,
                    help='Number of individual trajectories to show in Fig 3')
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

bm   = args.biomarker
bm_i = args.biomarker_index

# ---------------------------------------------------------------------------
# 1. Load per-fold predictions
# ---------------------------------------------------------------------------
fold_dfs = []
for fold in range(args.n_folds):
    path = os.path.join(args.models_dir,
                        f'bag_fold{fold}',
                        f'accord_predictions_{bm}_{bm_i}_{fold}.csv')
    if os.path.exists(path):
        df = pd.read_csv(path)
        df['fold'] = fold
        fold_dfs.append(df)
        print(f'Fold {fold}: {len(df)} rows, {df["PTID"].nunique()} subjects  ({path})')
    else:
        print(f'WARNING: {path} not found — skipping fold {fold}')

if not fold_dfs:
    raise FileNotFoundError(
        f'No ACCORD prediction files found under {args.models_dir}/bag_fold*/. '
        f'Run train_bag_5fold.sh first.'
    )

all_preds = pd.concat(fold_dfs, ignore_index=True)
all_preds['PTID'] = all_preds['PTID'].astype(str)

# ---------------------------------------------------------------------------
# 2. Ensemble across folds (mean of predictions; ground_truth is identical)
# ---------------------------------------------------------------------------
ensemble = (
    all_preds.groupby(['PTID', 'time_months'])
    .agg(
        predicted      = ('predicted',      'mean'),
        lower_bound    = ('lower_bound',    'mean'),
        upper_bound    = ('upper_bound',    'mean'),
        variance       = ('variance',       'mean'),
        interval_width = ('interval_width', 'mean'),
        ground_truth   = ('ground_truth',   'first'),  # same across folds
        covered        = ('covered',        'mean'),
    )
    .reset_index()
)
print(f'\nEnsemble: {ensemble["PTID"].nunique()} subjects, '
      f'{ensemble["time_months"].nunique()} timepoints')

# ---------------------------------------------------------------------------
# 3. Demographics (Sex, Age)
#    Priority: accord_data_bag_processed.csv → longitudinal_covariates_bag_allstudies.csv
# ---------------------------------------------------------------------------
demo = pd.DataFrame({'PTID': ensemble['PTID'].unique().astype(str)})

def _load_demo(path, ptid_col='PTID', time_col='Time'):
    """Return a per-subject baseline DataFrame with whatever columns exist."""
    df = pd.read_csv(path)
    df[ptid_col] = df[ptid_col].astype(str)
    if time_col in df.columns:
        df = df[df[time_col] == 0]
    keep = [ptid_col] + [c for c in ['Sex', 'Age'] if c in df.columns]
    return df[keep].drop_duplicates(ptid_col).rename(columns={ptid_col: 'PTID'})

demo_loaded = False
for demo_path in [args.accord_data, args.covariates_file]:
    if os.path.exists(demo_path):
        try:
            baseline = _load_demo(demo_path)
            demo = demo.merge(baseline, on='PTID', how='left')
            matched = demo['Sex'].notna().sum() if 'Sex' in demo.columns else 0
            print(f'Demographics from {demo_path}: '
                  f'{matched}/{len(demo)} subjects matched '
                  f'(cols: {[c for c in ["Sex","Age"] if c in demo.columns]})')
            demo_loaded = True
            break
        except Exception as e:
            print(f'WARNING: could not load demographics from {demo_path}: {e}')

if not demo_loaded:
    print('WARNING: no demographics file found — Sex/Age will be unavailable')

if 'Sex' not in demo.columns:
    demo['Sex'] = np.nan
if 'Age' not in demo.columns:
    demo['Age'] = np.nan

ensemble = ensemble.merge(demo, on='PTID', how='left')
ensemble['Sex_label'] = ensemble['Sex'].map({0: 'Male', 1: 'Female'})

# ---------------------------------------------------------------------------
# 4. Summary statistics
# ---------------------------------------------------------------------------
timepoints = sorted(ensemble['time_months'].unique())
obs_mask   = ensemble['ground_truth'].notna()
obs_tps    = sorted(ensemble.loc[obs_mask, 'time_months'].unique())

print(f'\nTimepoints in data        : {timepoints}')
print(f'Timepoints with real BAG  : {obs_tps}')

baseline_pred = ensemble[ensemble['time_months'] == 0]['predicted']
final_pred    = ensemble[ensemble['time_months'] == max(timepoints)]['predicted']
print(f'\nBaseline mean predicted BAG : {baseline_pred.mean():.2f} ± {baseline_pred.std():.2f} yr')
print(f'Year-8 mean predicted BAG   : {final_pred.mean():.2f} ± {final_pred.std():.2f} yr')
print(f'Mean change over 8 years    : {final_pred.mean() - baseline_pred.mean():+.2f} yr')

if obs_mask.any():
    matched = ensemble[obs_mask]
    mae_obs = np.mean(np.abs(matched['ground_truth'] - matched['predicted']))
    print(f'\nMAE at observed timepoints  : {mae_obs:.2f} yr  (n={len(matched)} observations)')
    coverage_pct = matched['covered'].mean() * 100
    print(f'Coverage at observed tps    : {coverage_pct:.1f}%')

print(f'\nSex breakdown (n subjects): '
      f'{(demo["Sex"] == 0).sum()} Male, {(demo["Sex"] == 1).sum()} Female')

# ---------------------------------------------------------------------------
# Publication-quality global style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    'font.family':        'DejaVu Sans',
    'font.size':          11,
    'axes.titlesize':     12,
    'axes.labelsize':     11,
    'xtick.labelsize':    9,
    'ytick.labelsize':    9,
    'legend.fontsize':    9,
    'legend.framealpha':  0.92,
    'legend.edgecolor':   '0.75',
    'legend.borderpad':   0.5,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'axes.linewidth':     0.8,
    'grid.color':         '0.88',
    'grid.linewidth':     0.5,
    'lines.linewidth':    1.5,
    'savefig.dpi':        300,
    'savefig.bbox':       'tight',
    'savefig.pad_inches': 0.05,
})

# Colorblind-friendly palette
C_PRED = '#2166AC'   # dark blue   — predicted trajectories
C_OBS  = '#D73027'   # crimson     — observed BAG
C_CI   = '#AEC6E8'   # light blue  — GP prediction interval
C_POP  = '#BABABA'   # light gray  — population mean SEM band
C_MALE = '#4393C3'   # mid blue    — male
C_FEM  = '#D6604D'   # salmon      — female

def _despine(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def _xtick_labels(tps):
    """Year labels with months in parentheses."""
    labels = []
    for t in tps:
        if t == 0:
            labels.append('Baseline\n(0 m)')
        elif t % 12 == 0:
            labels.append(f'Year {t//12}\n({t} m)')
        else:
            labels.append(f'{t} m')
    return labels

rng = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# Fig 1: Population spaghetti + mean trajectory + observed BAG
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5.5))

# Ultra-transparent spaghetti — shows density without visual clutter
for ptid, subj in ensemble.groupby('PTID'):
    subj = subj.sort_values('time_months')
    ax.plot(subj['time_months'], subj['predicted'],
            color=C_PRED, lw=0.3, alpha=0.06, zorder=1)

# Population mean ± SEM band
pop = (ensemble.groupby('time_months')['predicted']
       .agg(['mean', 'std', 'count']).reset_index())
pop['se']    = pop['std'] / np.sqrt(pop['count'])
pop['ci_lo'] = pop['mean'] - 1.96 * pop['se']
pop['ci_hi'] = pop['mean'] + 1.96 * pop['se']

ax.fill_between(pop['time_months'], pop['ci_lo'], pop['ci_hi'],
                alpha=0.35, color=C_POP, zorder=2)
ax.plot(pop['time_months'], pop['mean'], '-', color=C_PRED, lw=2.5,
        label='Population mean (± 95% SEM)', zorder=3)

# Observed BAG at measured timepoints
if obs_mask.any():
    obs_agg = (ensemble[obs_mask].groupby('time_months')['ground_truth']
               .agg(['mean', 'std', 'count']).reset_index())
    obs_agg['se'] = obs_agg['std'] / np.sqrt(obs_agg['count'])
    ax.errorbar(obs_agg['time_months'], obs_agg['mean'],
                yerr=1.96 * obs_agg['se'], fmt='D', color=C_OBS,
                ms=7, lw=1.5, capsize=4, capthick=1.5,
                label='Observed BAG (mean ± 95% SEM)', zorder=5)

ax.set_xlabel('Time')
ax.set_ylabel('BAG (years)')
ax.set_title('ACCORD: Individual and Population BAG Trajectories')
ax.set_xticks(timepoints)
ax.set_xticklabels(_xtick_labels(timepoints))
ax.yaxis.grid(True, zorder=0)
ax.set_axisbelow(True)
_despine(ax)
ax.legend(loc='best')
fig.tight_layout()
fig.savefig(os.path.join(args.output_dir, 'fig1_population_trajectory.png'))
plt.close(fig)
print('Saved fig1_population_trajectory.png')

# ---------------------------------------------------------------------------
# Fig 2: Sex-stratified mean trajectories (no spaghetti)
# ---------------------------------------------------------------------------
sex_colors = {'Male': C_MALE, 'Female': C_FEM}
fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)

for ax, sex_label in zip(axes, ['Male', 'Female']):
    col = sex_colors[sex_label]
    sub = ensemble[ensemble['Sex_label'] == sex_label]
    if len(sub) == 0:
        ax.set_visible(False)
        continue

    pop_s = (sub.groupby('time_months')['predicted']
             .agg(['mean', 'std', 'count']).reset_index())
    pop_s['se']    = pop_s['std'] / np.sqrt(pop_s['count'])
    pop_s['ci_lo'] = pop_s['mean'] - 1.96 * pop_s['se']
    pop_s['ci_hi'] = pop_s['mean'] + 1.96 * pop_s['se']

    # Mean GP prediction interval band (averaged over subjects)
    mean_bounds = sub.groupby('time_months')[['lower_bound', 'upper_bound']].mean().reset_index()
    ax.fill_between(mean_bounds['time_months'],
                    mean_bounds['lower_bound'], mean_bounds['upper_bound'],
                    alpha=0.18, color=col, label='Mean GP interval', zorder=1)
    # Population SEM band
    ax.fill_between(pop_s['time_months'], pop_s['ci_lo'], pop_s['ci_hi'],
                    alpha=0.35, color=col, label='95% SEM band', zorder=2)
    ax.plot(pop_s['time_months'], pop_s['mean'], '-', color=col, lw=2.5,
            label='Mean prediction', zorder=3)

    # Observed
    real_s = sub[sub['ground_truth'].notna()]
    if len(real_s) > 0:
        obs_s = (real_s.groupby('time_months')['ground_truth']
                 .agg(['mean', 'std', 'count']).reset_index())
        obs_s['se'] = obs_s['std'] / np.sqrt(obs_s['count'])
        ax.errorbar(obs_s['time_months'], obs_s['mean'],
                    yerr=1.96 * obs_s['se'], fmt='D', color=C_OBS,
                    ms=7, lw=1.5, capsize=4, capthick=1.5, label='Observed BAG', zorder=5)

    ax.set_title(f'{sex_label}  (n = {sub["PTID"].nunique()})')
    ax.set_xlabel('Time')
    ax.set_xticks(timepoints)
    ax.set_xticklabels(_xtick_labels(timepoints))
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    _despine(ax)
    ax.legend(loc='best', fontsize=8)

axes[0].set_ylabel('BAG (years)')
fig.suptitle('ACCORD BAG Trajectories by Sex', y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(args.output_dir, 'fig2_sex_stratified.png'))
plt.close(fig)
print('Saved fig2_sex_stratified.png')

# ---------------------------------------------------------------------------
# Fig 3: Individual subject trajectory grid
# ---------------------------------------------------------------------------
ptids_with_obs = ensemble.loc[obs_mask, 'PTID'].unique()
ptids_all      = ensemble['PTID'].unique()
n_want    = min(args.n_traj, len(ptids_all))
n_with    = min(n_want, len(ptids_with_obs))
chosen    = list(rng.choice(ptids_with_obs, size=n_with, replace=False))
remaining = [p for p in ptids_all if p not in set(chosen)]
if n_want > n_with:
    chosen += list(rng.choice(remaining, size=n_want - n_with, replace=False))
rng.shuffle(chosen)

ncols = 4
nrows = int(np.ceil(len(chosen) / ncols))
fig, axes = plt.subplots(nrows, ncols,
                         figsize=(4.2 * ncols, 3.2 * nrows),
                         sharey=True, sharex=False)
axes_flat = np.array(axes).flatten()

legend_drawn = False
for ax, ptid in zip(axes_flat, chosen):
    subj = ensemble[ensemble['PTID'] == ptid].sort_values('time_months')
    sex_lbl = subj['Sex_label'].iloc[0] if not subj['Sex_label'].isna().all() else '?'
    age_val = subj['Age'].iloc[0]

    ax.fill_between(subj['time_months'], subj['lower_bound'], subj['upper_bound'],
                    alpha=0.28, color=C_CI, zorder=1, label='90% PI')
    ax.plot(subj['time_months'], subj['predicted'], '-', color=C_PRED, lw=1.8,
            zorder=2, label='Predicted')

    real = subj[subj['ground_truth'].notna()]
    if len(real) > 0:
        ax.scatter(real['time_months'], real['ground_truth'],
                   color=C_OBS, s=50, zorder=5, marker='D', label='Observed')

    age_str = f'{age_val:.0f} yr' if pd.notna(age_val) else ''
    ax.set_title(f'{sex_lbl}  {age_str}', fontsize=8)
    ax.set_xticks(timepoints[::2])
    ax.set_xticklabels([_xtick_labels(timepoints[::2])[i]
                        for i in range(len(timepoints[::2]))], fontsize=6.5)
    ax.tick_params(labelsize=7)
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    _despine(ax)

    if not legend_drawn:
        handles, labels = ax.get_legend_handles_labels()
        legend_drawn = True

for ax in axes_flat[len(chosen):]:
    ax.set_visible(False)

# Single shared legend below the grid
fig.legend(handles, labels, loc='lower center', ncol=3,
           bbox_to_anchor=(0.5, -0.03), fontsize=9, framealpha=0.9)
# Shared axis labels
fig.supxlabel('Time', y=-0.04)
fig.supylabel('BAG (years)', x=-0.01)
fig.suptitle('ACCORD Individual BAG Trajectories (sample)', y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(args.output_dir, 'fig3_individual_trajectories.png'))
plt.close(fig)
print('Saved fig3_individual_trajectories.png')

# ---------------------------------------------------------------------------
# Fig 4: Predicted vs Observed at measured timepoints
# ---------------------------------------------------------------------------
if obs_mask.any() and len(obs_tps) > 0:
    matched  = ensemble[obs_mask].copy()
    ncols_s  = len(obs_tps)
    panel_w  = 3.8
    fig, axes = plt.subplots(1, ncols_s,
                             figsize=(panel_w * ncols_s, panel_w + 0.5),
                             squeeze=False)
    for ax, tp in zip(axes[0], obs_tps):
        tp_data = matched[matched['time_months'] == tp].dropna(
            subset=['ground_truth', 'predicted'])
        if len(tp_data) < 2:
            ax.set_visible(False)
            continue

        mae_tp = np.mean(np.abs(tp_data['ground_truth'] - tp_data['predicted']))
        r, _   = stats.pearsonr(tp_data['ground_truth'], tp_data['predicted'])

        lo = min(tp_data['ground_truth'].min(), tp_data['predicted'].min()) - 1.5
        hi = max(tp_data['ground_truth'].max(), tp_data['predicted'].max()) + 1.5

        ax.scatter(tp_data['ground_truth'], tp_data['predicted'],
                   alpha=0.55, s=22, color=C_PRED, edgecolors='none', zorder=3)
        ax.plot([lo, hi], [lo, hi], '--', color='0.55', lw=1.2, zorder=2)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect('equal', adjustable='box')

        label = f'$r$ = {r:.3f}\nMAE = {mae_tp:.2f} yr\n$n$ = {len(tp_data)}'
        ax.text(0.05, 0.95, label, transform=ax.transAxes,
                va='top', ha='left', fontsize=8.5,
                bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                          edgecolor='0.8', linewidth=0.7))

        tp_yr = tp / 12
        ax.set_title(f'{"Baseline" if tp == 0 else f"Year {tp_yr:.1g}"}  (t = {tp} m)')
        ax.set_xlabel('Observed BAG (yr)')
        if tp == obs_tps[0]:
            ax.set_ylabel('Predicted BAG (yr)')
        ax.yaxis.grid(True, zorder=0)
        ax.xaxis.grid(True, zorder=0)
        ax.set_axisbelow(True)
        _despine(ax)

    fig.suptitle('Predicted vs Observed BAG at Measured Timepoints')
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, 'fig4_predicted_vs_observed.png'))
    plt.close(fig)
    print('Saved fig4_predicted_vs_observed.png')
else:
    print('No matched observations — skipping fig4')

# ---------------------------------------------------------------------------
# Fig 5: Violin of predicted BAG distribution at each timepoint
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 5.5))

viol_data = [ensemble[ensemble['time_months'] == tp]['predicted'].values
             for tp in timepoints]
positions = list(range(len(timepoints)))

vp = ax.violinplot(viol_data, positions=positions,
                   showmedians=True, showextrema=False, widths=0.65)
for body in vp['bodies']:
    body.set_facecolor(C_PRED)
    body.set_edgecolor(C_PRED)
    body.set_alpha(0.50)
vp['cmedians'].set_color(C_PRED)
vp['cmedians'].set_linewidth(2.2)

# Overlay jittered observed points
obs_handle = None
for j, tp in enumerate(timepoints):
    real = ensemble.loc[(ensemble['time_months'] == tp) & obs_mask, 'ground_truth']
    if len(real) > 0:
        jitter = rng.uniform(-0.18, 0.18, size=len(real))
        sc = ax.scatter(j + jitter, real, color=C_OBS, s=14,
                        alpha=0.70, zorder=5, linewidths=0)
        if obs_handle is None:
            obs_handle = sc

from matplotlib.patches import Patch
legend_handles = [
    Patch(facecolor=C_PRED, alpha=0.65, label='Predicted BAG'),
]
if obs_handle is not None:
    legend_handles.append(
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=C_OBS,
                   markersize=6, label='Observed BAG'))
ax.legend(handles=legend_handles, loc='upper left')

ax.set_xticks(positions)
ax.set_xticklabels(_xtick_labels(timepoints))
ax.set_xlabel('Time')
ax.set_ylabel('BAG (years)')
ax.set_title('ACCORD Predicted BAG Distribution Over Time')
ax.yaxis.grid(True, zorder=0)
ax.set_axisbelow(True)
_despine(ax)
fig.tight_layout()
fig.savefig(os.path.join(args.output_dir, 'fig5_distribution_over_time.png'))
plt.close(fig)
print('Saved fig5_distribution_over_time.png')

# ---------------------------------------------------------------------------
# Save ensemble CSV
# ---------------------------------------------------------------------------
out_csv = os.path.join(args.output_dir, 'accord_bag_ensemble.csv')
ensemble.to_csv(out_csv, index=False)
print(f'\nSaved ensemble predictions: {out_csv}')
print(f'All figures in: {args.output_dir}/')
