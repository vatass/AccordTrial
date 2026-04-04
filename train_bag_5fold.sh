#!/bin/bash
# Train DKGP models for BAG prediction across all 5 folds

set -e

DATA_FILE="data/subjectsamples_bag_allstudies.csv"
BIOMARKER_INDEX=0
BIOMARKER_NAME="BAG"
GPU_ID=0

echo "=== 5-Fold BAG Training ==="
echo "Data file: ${DATA_FILE}"
echo "Biomarker: ${BIOMARKER_NAME} (index ${BIOMARKER_INDEX})"
echo ""

for FOLD in 0 1 2 3 4; do
    TRAIN_IDS="data/train_subject_bag_allstudies_ids_hmuse${FOLD}.pkl"
    TEST_IDS="data/test_subject_bag_allstudies_ids_hmuse${FOLD}.pkl"
    OUTPUT_DIR="models/bag_fold${FOLD}"

    echo "--- Fold ${FOLD} ---"
    echo "Train IDs: ${TRAIN_IDS}"
    echo "Test IDs:  ${TEST_IDS}"
    echo "Output:    ${OUTPUT_DIR}"
    echo ""

    python dkgp_training.py \
        --data_file "${DATA_FILE}" \
        --train_ids_file "${TRAIN_IDS}" \
        --test_ids_file "${TEST_IDS}" \
        --biomarker_index ${BIOMARKER_INDEX} \
        --biomarker_name "${BIOMARKER_NAME}" \
        --output_dir "${OUTPUT_DIR}" \
        --gpu_id ${GPU_ID} \
        --fold ${FOLD}

    echo ""
    echo "Fold ${FOLD} complete."
    echo ""
done

echo "=== All 5 folds complete ==="
