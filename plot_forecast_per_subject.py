"""
Per-subject evaluation plot: 8-year BAG forecast vs real BAG observations.

For each subject one subplot is drawn showing:
  - Predicted BAG trajectory (line) with 90 % CI shading over 0–96 months
  - Real BAG observations plotted at their *actual* timepoints in months from
    baseline (these will generally NOT fall on the 12-month forecast grid)

Two data-source modes are supported:

  Mode A – inference CSVs (real_BAG already embedded):
      python plot_forecast_per_subject.py \\
          --inference_dir inference \\
          --output_dir plots/per_subject

  Mode B – training forecast CSVs + SPARE_BA file:
      python plot_forecast_per_subject.py \\
          --forecast_dir models/bag_fold0 \\
          --biomarker_name BAG --biomarker_index 0 \\
          --spare_ba SPARE_BA_out_20260319.csv \\
          --output_dir plots/per_subject

In both modes multiple folds are automatically ensembled (averaged).

Output
------
  <output_dir>/per_subject_forecasts.pdf   – all subjects in one PDF
  <output_dir>/per_subject_forecasts.png   – first page preview PNG
  <output_dir>/subject_<PTID>.png          – individual PNG per subject
                                             (only when --save_individual is set)
"""

import argparse
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pdf_backend
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Per-subject 8-year BAG forecast vs real BAG plot"
)

# --- forecast source (Mode A) ---
parser.add_argument(
    "--inference_dir",
    default=None,
    help="Directory containing accord_bag_fold{i}/ sub-dirs with "
         "predictions_accord_BAG_{i}.csv (Mode A)",
)

# --- forecast source (Mode B) ---
parser.add_argument(
    "--forecast_dir",
    default=None,
    help="Directory (or glob pattern) containing "
         "accord_eight_year_forecast_<name>_<idx>_<fold>.csv files (Mode B)",
)
parser.add_argument("--biomarker_name",  default="BAG",
                    help="Biomarker name used in training forecast filenames (Mode B)")
parser.add_argument("--biomarker_index", type=int, default=0,
                    help="Biomarker index used in training forecast filenames (Mode B)")

# --- real BAG source ---
parser.add_argument(
    "--spare_ba",
    default="SPARE_BA_out_20260319.csv",
    help="SPARE_BA CSV with columns MRID, SPARE_BA, Age_actual "
         "(used to derive real BAG; Mode B always needs this; "
         "Mode A uses it to enrich real_BAG when present)",
)

# --- general ---
parser.add_argument("--n_folds",    type=int,   default=5,
                    help="Number of folds to ensemble")
parser.add_argument("--output_dir", default="plots/per_subject",
                    help="Where to save the output files")
parser.add_argument("--max_subjects", type=int, default=None,
                    help="Limit to first N subjects (useful for quick checks)")
parser.add_argument("--subjects",   default=None,
                    help="Comma-separated list of PTIDs to plot (default: all)")
parser.add_argument("--ncols",      type=int,   default=4,
                    help="Subplot columns per page (multi-subject layout)")
parser.add_argument("--page_size",  type=int,   default=20,
                    help="Subjects per PDF page")
parser.add_argument("--save_individual", action="store_true",
                    help="Also save one PNG per subject")
parser.add_argument("--dpi",        type=int,   default=150)

args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)


# ---------------------------------------------------------------------------
# Helper: extract PTID from MRID (format PTID-Date, e.g. 301D00252-20040121)
# ---------------------------------------------------------------------------
def mrid_to_ptid(mrid: str) -> str:
    """Return the PTID portion of an MRID string."""
    # ACCORD MRIDs are <PTID>-<8-digit-date>
    m = re.match(r"^(.+)-\d{8}$", str(mrid).strip())
    return m.group(1) if m else str(mrid).strip()


def mrid_to_date(mrid: str):
    """Return a pandas Timestamp from the date embedded in an MRID."""
    m = re.match(r"^.+-(\d{8})$", str(mrid).strip())
    if m:
        return pd.to_datetime(m.group(1), format="%Y%m%d")
    return pd.NaT


# ---------------------------------------------------------------------------
# 1.  Load / build real BAG from SPARE_BA file
# ---------------------------------------------------------------------------
def load_real_bag(spare_ba_path: str) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
        PTID, visit_date (Timestamp), time_months (float), real_BAG (float)
    sorted by PTID, visit_date.
    """
    if not os.path.exists(spare_ba_path):
        print(f"WARNING: SPARE_BA file not found: {spare_ba_path}  — real BAG will be NaN")
        return pd.DataFrame(columns=["PTID", "visit_date", "time_months", "real_BAG"])

    sba = pd.read_csv(spare_ba_path)
    required = {"MRID", "SPARE_BA", "Age_actual"}
    missing  = required - set(sba.columns)
    if missing:
        print(f"WARNING: SPARE_BA file missing columns {missing}  — real BAG will be NaN")
        return pd.DataFrame(columns=["PTID", "visit_date", "time_months", "real_BAG"])

    sba["PTID"]       = sba["MRID"].apply(mrid_to_ptid)
    sba["visit_date"] = sba["MRID"].apply(mrid_to_date)
    sba["real_BAG"]   = sba["SPARE_BA"] - sba["Age_actual"]

    sba = sba.dropna(subset=["visit_date"]).sort_values(["PTID", "visit_date"])

    # Time in months from each subject's first (baseline) scan
    sba["baseline_date"] = sba.groupby("PTID")["visit_date"].transform("first")
    sba["time_months"]   = (sba["visit_date"] - sba["baseline_date"]).dt.days / 30.4375

    return sba[["PTID", "visit_date", "time_months", "real_BAG"]].reset_index(drop=True)


real_bag_df = load_real_bag(args.spare_ba)
print(f"Loaded real BAG: {len(real_bag_df)} observations, "
      f"{real_bag_df['PTID'].nunique()} subjects")


# ---------------------------------------------------------------------------
# 2.  Load forecast predictions and ensemble across folds
# ---------------------------------------------------------------------------

def _norm_forecast_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise column names so we always have 'predicted', 'lower_bound',
    'upper_bound' regardless of whether the file came from training or inference."""
    rename = {}
    if "predicted_value" in df.columns and "predicted" not in df.columns:
        rename["predicted_value"] = "predicted"
    df = df.rename(columns=rename)
    return df


fold_dfs = []

# --- Mode A: inference directory ---
if args.inference_dir is not None:
    for fold in range(args.n_folds):
        path = os.path.join(
            args.inference_dir,
            f"accord_bag_fold{fold}",
            f"predictions_accord_BAG_{fold}.csv",
        )
        if os.path.exists(path):
            df = pd.read_csv(path)
            df = _norm_forecast_cols(df)
            df["fold"] = fold
            fold_dfs.append(df)
            print(f"  Fold {fold} (inference): {len(df)} rows, "
                  f"{df['PTID'].nunique()} subjects  [{path}]")
        else:
            print(f"  WARNING: {path} not found — skipping fold {fold}")

# --- Mode B: training forecast directory ---
if args.forecast_dir is not None:
    for fold in range(args.n_folds):
        fname = (f"accord_eight_year_forecast_"
                 f"{args.biomarker_name}_{args.biomarker_index}_{fold}.csv")
        path  = os.path.join(args.forecast_dir, fname)
        if os.path.exists(path):
            df = pd.read_csv(path)
            df = _norm_forecast_cols(df)
            df["fold"] = fold
            fold_dfs.append(df)
            print(f"  Fold {fold} (training forecast): {len(df)} rows, "
                  f"{df['PTID'].nunique()} subjects  [{path}]")
        else:
            print(f"  WARNING: {path} not found — skipping fold {fold}")

if not fold_dfs:
    sys.exit(
        "ERROR: No forecast files found.\n"
        "  Run with --inference_dir <dir>   (Mode A)\n"
        "  or  --forecast_dir <dir>         (Mode B)"
    )

all_preds = pd.concat(fold_dfs, ignore_index=True)
all_preds["PTID"] = all_preds["PTID"].astype(str)

# Ensemble: average predicted / bounds across folds per (PTID, time_months)
agg_dict = dict(
    predicted   =("predicted",    "mean"),
    lower_bound =("lower_bound",  "mean"),
    upper_bound =("upper_bound",  "mean"),
    variance    =("variance",     "mean"),
)
if "real_BAG" in all_preds.columns:
    # Keep the first non-NaN real_BAG value across folds
    agg_dict["real_BAG_forecast"] = ("real_BAG", "first")

ensemble = (
    all_preds.groupby(["PTID", "time_months"])
    .agg(**agg_dict)
    .reset_index()
)
ensemble["interval_width"] = ensemble["upper_bound"] - ensemble["lower_bound"]

print(f"\nEnsemble: {ensemble['PTID'].nunique()} subjects, "
      f"{ensemble['time_months'].nunique()} forecast timepoints")
print(f"Forecast timepoints: {sorted(ensemble['time_months'].unique())}")


# ---------------------------------------------------------------------------
# 3.  Merge real BAG observations into ensemble
#     (use embedded real_BAG when available, otherwise use SPARE_BA-derived)
# ---------------------------------------------------------------------------

# Build per-subject real-BAG table from SPARE_BA (authoritative source)
real_bag_per_subject = {}
for ptid, grp in real_bag_df.groupby("PTID"):
    real_bag_per_subject[ptid] = grp[["time_months", "real_BAG"]].copy()

# Fallback: if SPARE_BA file was absent but inference CSVs had real_BAG, use those
if not real_bag_df.empty:
    source_label = "SPARE_BA file"
else:
    # Build from embedded real_BAG_forecast column if it exists
    source_label = "embedded real_BAG (from inference CSV)"
    if "real_BAG_forecast" in ensemble.columns:
        grp_obs = ensemble[ensemble["real_BAG_forecast"].notna()].copy()
        for ptid, grp in grp_obs.groupby("PTID"):
            real_bag_per_subject[ptid] = (
                grp[["time_months", "real_BAG_forecast"]]
                .rename(columns={"real_BAG_forecast": "real_BAG"})
                .copy()
            )

print(f"Real BAG source: {source_label}")
print(f"Subjects with real BAG observations: {len(real_bag_per_subject)}")


# ---------------------------------------------------------------------------
# 4.  Decide which subjects to plot
# ---------------------------------------------------------------------------
all_subject_ids = sorted(ensemble["PTID"].unique())

if args.subjects:
    wanted = [s.strip() for s in args.subjects.split(",")]
    subject_ids = [s for s in wanted if s in set(all_subject_ids)]
    missing = set(wanted) - set(all_subject_ids)
    if missing:
        print(f"WARNING: subjects not found in forecast data: {missing}")
else:
    subject_ids = all_subject_ids

if args.max_subjects:
    subject_ids = subject_ids[: args.max_subjects]

print(f"\nPlotting {len(subject_ids)} subjects …")

forecast_timepoints = sorted(ensemble["time_months"].unique())
PRED_COLOR  = "steelblue"
OBS_COLOR   = "crimson"
SHADE_ALPHA = 0.25


# ---------------------------------------------------------------------------
# 5.  Per-subject plot helper
# ---------------------------------------------------------------------------

def plot_subject(ax, ptid: str):
    """Draw forecast + real BAG for one subject onto *ax*."""
    subj = ensemble[ensemble["PTID"] == ptid].sort_values("time_months")

    # --- Forecast trajectory ---
    ax.plot(
        subj["time_months"], subj["predicted"],
        color=PRED_COLOR, lw=2, label="Forecast",
    )
    ax.fill_between(
        subj["time_months"], subj["lower_bound"], subj["upper_bound"],
        color=PRED_COLOR, alpha=SHADE_ALPHA, label="90 % CI",
    )

    # --- Real BAG observations (at their actual timepoints) ---
    real = real_bag_per_subject.get(ptid)
    if real is not None and not real.empty:
        # Clip to the 8-year window for clarity
        real_plot = real[real["time_months"] <= 100].copy()
        if not real_plot.empty:
            ax.scatter(
                real_plot["time_months"], real_plot["real_BAG"],
                color=OBS_COLOR, s=60, zorder=5, marker="D",
                label="Real BAG",
            )

    # Axis decoration
    ax.set_title(str(ptid), fontsize=8, pad=3)
    ax.set_xlabel("months", fontsize=7)
    ax.set_ylabel("BAG (yr)", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.set_xticks([0, 24, 48, 72, 96])
    ax.set_xticklabels(["0\n(BL)", "24m", "48m", "72m", "96m\n(8yr)"], fontsize=6)
    ax.set_xlim(-3, 100)
    ax.grid(True, alpha=0.3, lw=0.5)


# ---------------------------------------------------------------------------
# 6.  Multi-page PDF + individual PNGs
# ---------------------------------------------------------------------------

ncols      = args.ncols
page_size  = args.page_size
nrows_page = int(np.ceil(page_size / ncols))

pdf_path = os.path.join(args.output_dir, "per_subject_forecasts.pdf")
png_preview = os.path.join(args.output_dir, "per_subject_forecasts_preview.png")

pages  = [subject_ids[i : i + page_size] for i in range(0, len(subject_ids), page_size)]
n_pages = len(pages)

print(f"Writing {n_pages} page(s) → {pdf_path}")

with pdf_backend.PdfPages(pdf_path) as pdf:
    for page_idx, page_ptids in enumerate(pages):
        nrows = int(np.ceil(len(page_ptids) / ncols))
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(4.5 * ncols, 3.8 * nrows),
            squeeze=False,
        )
        axes_flat = axes.flatten()

        for ax, ptid in zip(axes_flat, page_ptids):
            plot_subject(ax, ptid)

        # Shared legend on the first panel
        handles, labels = axes_flat[0].get_legend_handles_labels()
        if handles:
            axes_flat[0].legend(handles, labels, fontsize=6, loc="upper left")

        # Hide unused panels
        for ax in axes_flat[len(page_ptids) :]:
            ax.set_visible(False)

        fig.suptitle(
            f"Per-Subject 8-Year BAG Forecast vs Real BAG"
            f"  (page {page_idx + 1}/{n_pages})",
            fontsize=12,
        )
        plt.tight_layout(rect=[0, 0, 1, 0.97])

        pdf.savefig(fig, dpi=args.dpi)

        # Save the first page as PNG preview
        if page_idx == 0:
            fig.savefig(png_preview, dpi=args.dpi)
            print(f"Preview PNG saved → {png_preview}")

        plt.close(fig)
        print(f"  Page {page_idx + 1}/{n_pages}: {len(page_ptids)} subjects")

print(f"PDF saved → {pdf_path}")


# ---------------------------------------------------------------------------
# 7.  Optional: individual PNG per subject
# ---------------------------------------------------------------------------
if args.save_individual:
    ind_dir = os.path.join(args.output_dir, "individual")
    os.makedirs(ind_dir, exist_ok=True)
    for ptid in subject_ids:
        fig, ax = plt.subplots(figsize=(6, 4))
        plot_subject(ax, ptid)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, fontsize=8)
        fig.suptitle(f"Subject {ptid} — 8-Year BAG Forecast", fontsize=10)
        plt.tight_layout()
        safe_ptid = re.sub(r"[^\w\-]", "_", str(ptid))
        fig.savefig(os.path.join(ind_dir, f"subject_{safe_ptid}.png"), dpi=args.dpi)
        plt.close(fig)
    print(f"Individual PNGs saved → {ind_dir}/")


# ---------------------------------------------------------------------------
# 8.  Brief summary stats
# ---------------------------------------------------------------------------
matched_count = sum(
    1
    for p in subject_ids
    if p in real_bag_per_subject and not real_bag_per_subject[p].empty
)
print(f"\nSummary")
print(f"  Subjects plotted            : {len(subject_ids)}")
print(f"  With real BAG observations  : {matched_count}")
print(f"  Without real BAG            : {len(subject_ids) - matched_count}")

pop_base = ensemble[ensemble["time_months"] == 0]["predicted"]
pop_end  = ensemble[ensemble["time_months"] == 96]["predicted"]
if not pop_base.empty and not pop_end.empty:
    print(f"  Forecast BAG at baseline    : {pop_base.mean():.2f} ± {pop_base.std():.2f} yr")
    print(f"  Forecast BAG at 8 years     : {pop_end.mean():.2f} ± {pop_end.std():.2f} yr")
    print(f"  Mean predicted change       : {pop_end.mean() - pop_base.mean():+.2f} yr")

print("\nDone.")
