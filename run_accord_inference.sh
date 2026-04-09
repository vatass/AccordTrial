#!/bin/bash
# Run DKGP inference on ACCORD data for BAG across all 5 folds.
#
# Prerequisites:
#   1. accord_data.py has been run  →  data/accord_data_bag_processed.csv exists
#   2. train_bag_5fold.sh has been run  →  models/bag_fold{0-4}/ checkpoints exist

set -e

DATA_FILE="data/accord_data_bag_processed.csv"
BIOMARKER_NAME="BAG"
BIOMARKER_TYPE="BAG"
BIOMARKER_INDEX=0
STATS_DIR="data"
GPU_ID=0

echo "============================================="
echo "  ACCORD Inference — ${BIOMARKER_NAME}"
echo "  Data file : ${DATA_FILE}"
echo "============================================="

if [ ! -f "${DATA_FILE}" ]; then
    echo "ERROR: ${DATA_FILE} not found."
    echo "Run accord_data.py first to generate the preprocessed ACCORD CSV."
    exit 1
fi

for FOLD in 0 1 2 3 4; do
    MODEL_FILE="models/bag_fold${FOLD}/deep_kernel_gp_${BIOMARKER_NAME}_${BIOMARKER_INDEX}_${FOLD}.pth"
    OUTPUT_DIR="inference/accord_bag_fold${FOLD}"
    OUTPUT_FILE="${OUTPUT_DIR}/predictions_accord_${BIOMARKER_NAME}_${FOLD}.csv"

    echo ""
    echo "--- Fold ${FOLD} ---"
    echo "  Model  : ${MODEL_FILE}"
    echo "  Output : ${OUTPUT_FILE}"

    if [ ! -f "${MODEL_FILE}" ]; then
        echo "  WARNING: model not found — skipping fold ${FOLD}"
        continue
    fi

    mkdir -p "${OUTPUT_DIR}"

    python dkgp_inference.py \
        --data_file         "${DATA_FILE}" \
        --model_file        "${MODEL_FILE}" \
        --biomarker_index   ${BIOMARKER_INDEX} \
        --biomarker_name    "${BIOMARKER_NAME}" \
        --biomarker         "${BIOMARKER_TYPE}" \
        --output_file       "${OUTPUT_FILE}" \
        --stats_dir         "${STATS_DIR}" \
        --gpu_id            ${GPU_ID}

    echo "  Fold ${FOLD} done → ${OUTPUT_FILE}"
done

echo ""
echo "============================================="
echo "  All folds complete."
echo "  Results in: inference/accord_bag_fold{0-4}/"
echo "============================================="
