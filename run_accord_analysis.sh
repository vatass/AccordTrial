#!/bin/bash
# Analyse ACCORD BAG predictions produced by dkgp_training.py / train_bag_5fold.sh
#
# Pipeline order:
#   1. python longitudinal_data.py          → data/ splits + normalization stats
#   2. python accord_data.py               → data/subjectsamples_bag_accord.csv
#                                             data/accord_data_bag_processed.csv
#   3. ./train_bag_5fold.sh                → models/bag_fold{0-4}/ checkpoints
#                                             models/bag_fold{i}/accord_predictions_BAG_0_{i}.csv
#   4. ./run_accord_analysis.sh  (THIS)    → analysis/accord_bag/ figures + CSV

set -e

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODELS_DIR="models"
ACCORD_DATA="data/accord_data_bag_processed.csv"
COVARIATES="data/longitudinal_covariates_bag_allstudies.csv"
NORM_STATS="data/normalization_stats.pkl"
OUTPUT_DIR="analysis/accord_bag"
N_FOLDS=5
BIOMARKER="BAG"
BIOMARKER_INDEX=0
N_TRAJ=20

echo "============================================="
echo "  ACCORD BAG Trajectory Analysis"
echo "  Biomarker  : ${BIOMARKER} (index ${BIOMARKER_INDEX})"
echo "  Folds      : ${N_FOLDS}"
echo "  Output dir : ${OUTPUT_DIR}"
echo "============================================="
echo ""

# ---------------------------------------------------------------------------
# Prerequisite checks
# ---------------------------------------------------------------------------
MISSING=0

if [ ! -f "${ACCORD_DATA}" ]; then
    echo "ERROR: ${ACCORD_DATA} not found."
    echo "  → Run: python accord_data.py"
    MISSING=1
fi

if [ ! -f "${COVARIATES}" ]; then
    echo "ERROR: ${COVARIATES} not found."
    echo "  → Run: python longitudinal_data.py"
    MISSING=1
fi

if [ ! -f "${NORM_STATS}" ]; then
    echo "ERROR: ${NORM_STATS} not found."
    echo "  → Run: python longitudinal_data.py"
    MISSING=1
fi

# Count available prediction files
FOUND=0
for FOLD in $(seq 0 $((N_FOLDS - 1))); do
    PRED="${MODELS_DIR}/bag_fold${FOLD}/accord_predictions_${BIOMARKER}_${BIOMARKER_INDEX}_${FOLD}.csv"
    if [ -f "${PRED}" ]; then
        FOUND=$((FOUND + 1))
        echo "  [ok] ${PRED}"
    else
        echo "  [missing] ${PRED}"
    fi
done

if [ "${FOUND}" -eq 0 ]; then
    echo ""
    echo "ERROR: No ACCORD prediction files found in ${MODELS_DIR}/bag_fold*/."
    echo "  → Run: ./train_bag_5fold.sh"
    MISSING=1
fi

if [ "${MISSING}" -ne 0 ]; then
    echo ""
    echo "Aborting — fix the missing prerequisites above."
    exit 1
fi

echo ""
echo "Using ${FOUND}/${N_FOLDS} fold prediction files."
echo ""

# ---------------------------------------------------------------------------
# Run analysis
# ---------------------------------------------------------------------------
mkdir -p "${OUTPUT_DIR}"

python analyze_accord_predictions.py \
    --models_dir        "${MODELS_DIR}" \
    --accord_data       "${ACCORD_DATA}" \
    --covariates_file   "${COVARIATES}" \
    --output_dir        "${OUTPUT_DIR}" \
    --n_folds           ${N_FOLDS} \
    --biomarker         "${BIOMARKER}" \
    --biomarker_index   ${BIOMARKER_INDEX} \
    --n_traj            ${N_TRAJ}

echo ""
echo "============================================="
echo "  Figures saved to: ${OUTPUT_DIR}/"
echo "    fig1_population_trajectory.png"
echo "    fig2_sex_stratified.png"
echo "    fig3_individual_trajectories.png"
echo "    fig4_predicted_vs_observed.png"
echo "    fig5_distribution_over_time.png"
echo "    accord_bag_ensemble.csv"
echo "============================================="
