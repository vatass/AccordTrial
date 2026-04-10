#!/bin/bash
# Run analyze_predictions.py for BAG and SPARE-BA across all 5 folds
#
# Prerequisites: both training scripts must have completed so that
#   models/bag_fold{0-4}/predictions_BAG_0_{fold}.csv
#   models/spare_ba_fold{0-4}/predictions_SPARE-BA_0_{fold}.csv
# are present.

set -e

COVARIATES_FILE="data/longitudinal_covariates_bag_allstudies.csv"
NORM_STATS="data/normalization_stats.pkl"
N_TRAJ=20
MIN_TP=2

# ---------------------------------------------------------------------------
# BAG
# ---------------------------------------------------------------------------
BIOMARKER_NAME="BAG"
BIOMARKER_INDEX=0

echo "============================================="
echo "  Analyzing predictions for: ${BIOMARKER_NAME}"
echo "============================================="

for FOLD in 0 1 2 3 4; do
    PREDICTIONS_FILE="models/bag_fold${FOLD}/predictions_${BIOMARKER_NAME}_${BIOMARKER_INDEX}_${FOLD}.csv"
    OUTPUT_DIR="analysis/bag_fold${FOLD}"

    echo ""
    echo "--- Fold ${FOLD} ---"
    echo "  Predictions : ${PREDICTIONS_FILE}"
    echo "  Output      : ${OUTPUT_DIR}"

    if [ ! -f "${PREDICTIONS_FILE}" ]; then
        echo "  WARNING: ${PREDICTIONS_FILE} not found — skipping fold ${FOLD}"
        continue
    fi

    python analyze_predictions.py \
        --predictions_file      "${PREDICTIONS_FILE}" \
        --covariates_file       "${COVARIATES_FILE}" \
        --normalization_stats_file "${NORM_STATS}" \
        --output_dir            "${OUTPUT_DIR}" \
        --biomarker_name        "${BIOMARKER_NAME}" \
        --n_traj_subjects       ${N_TRAJ} \
        --min_timepoints        ${MIN_TP}

    echo "  Fold ${FOLD} done."
done

# echo ""
# echo "============================================="
# echo "  Analyzing predictions for: SPARE-BA"
# echo "============================================="

# BIOMARKER_NAME="SPARE-BA"
# BIOMARKER_INDEX=0
# COVARIATES_FILE_SPAREBA="data/longitudinal_covariates_spare_ba_allstudies.csv"

# # Fall back to BAG covariates file if SPARE-BA specific one is absent
# if [ ! -f "${COVARIATES_FILE_SPAREBA}" ]; then
#     echo "  NOTE: ${COVARIATES_FILE_SPAREBA} not found — using BAG covariates file"
#     COVARIATES_FILE_SPAREBA="${COVARIATES_FILE}"
# fi

# for FOLD in 0 1 2 3 4; do
#     PREDICTIONS_FILE="models/spare_ba_fold${FOLD}/predictions_${BIOMARKER_NAME}_${BIOMARKER_INDEX}_${FOLD}.csv"
#     OUTPUT_DIR="analysis/spare_ba_fold${FOLD}"

#     echo ""
#     echo "--- Fold ${FOLD} ---"
#     echo "  Predictions : ${PREDICTIONS_FILE}"
#     echo "  Output      : ${OUTPUT_DIR}"

#     if [ ! -f "${PREDICTIONS_FILE}" ]; then
#         echo "  WARNING: ${PREDICTIONS_FILE} not found — skipping fold ${FOLD}"
#         continue
#     fi

#     python analyze_predictions.py \
#         --predictions_file      "${PREDICTIONS_FILE}" \
#         --covariates_file       "${COVARIATES_FILE_SPAREBA}" \
#         --normalization_stats_file "${NORM_STATS}" \
#         --output_dir            "${OUTPUT_DIR}" \
#         --biomarker_name        "${BIOMARKER_NAME}" \
#         --n_traj_subjects       ${N_TRAJ} \
#         --min_timepoints        ${MIN_TP}

#     echo "  Fold ${FOLD} done."
# done

# echo ""
# echo "============================================="
# echo "  All analysis complete."
# echo "  BAG outputs    : analysis/bag_fold{0-4}/"
# echo "  SPARE-BA outputs: analysis/spare_ba_fold{0-4}/"
# echo "============================================="
