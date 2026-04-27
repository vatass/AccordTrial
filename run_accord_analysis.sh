#!/bin/bash
# Analyze ACCORD BAG inference results — ensemble 5 folds and plot trajectories

set -e

python analyze_accord_predictions.py \
    --inference_dir   models \
    --accord_data     data/accord_data_bag_processed.csv \
    --norm_stats      data/normalization_stats.pkl \
    --output_dir      analysis/accord_bag \
    --n_folds         5 \
    --n_traj          20

echo ""
echo "Figures saved to: analysis/accord_bag/"
