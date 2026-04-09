'''
Population DKGP Model Inference 
'''

import pandas as pd
import numpy as np
import torch
import gpytorch
from utils import *
from models import SingleTaskDeepKernel
import argparse
import json
import time
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

parser = argparse.ArgumentParser(description='Future time point inference with trained DKGP model')
parser.add_argument("--data_file", help="Path to the data CSV file", required=True)
parser.add_argument("--model_file", help="Path to the trained model file", required=True)
parser.add_argument("--biomarker_index", help="biomarker index for inference", type=int, required=True)
parser.add_argument("--biomarker_name", help="biomarker name for inference", type=str, required=True)
parser.add_argument("--output_file", help="Path to save inference results CSV", required=True)
parser.add_argument("--biomarker", help="Biomarker type (MUSE, SPARE_AD, BAG, SPARE_BA)", required=True)
parser.add_argument("--gpu_id", help="GPU ID to use", type=int, default=0)
parser.add_argument("--stats_dir", help="Directory containing normalization stats files", default="./data")

args = parser.parse_args()

# Parse arguments
gpu_id = args.gpu_id
roi_idx = args.biomarker_index
data_file = args.data_file
model_file = args.model_file
output_file = args.output_file
biomarker = args.biomarker.upper()
stats_dir = args.stats_dir

# Define future time points (8 years = 96 months, every 12 months)
future_timepoints = [0, 12, 24, 36, 48, 60, 72, 84, 96]

print(f"Starting future time point inference for ROI {roi_idx}")
print(f"Future time points: {future_timepoints}")
print(f"Loading data from: {data_file}")
print(f"Loading model from: {model_file}")

# Load data
datasamples = pd.read_csv(data_file)
print(f"Loaded data with {len(datasamples)} samples")


# Load model
print("Loading trained model...")
checkpoint = torch.load(model_file, map_location=f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')

# Extract model components
model_state_dict = checkpoint['model_state_dict']
optimizer_state_dict = checkpoint['optimizer_state_dict']
likelihood_state_dict = checkpoint['likelihood_state_dict']

# Check if training data is available in checkpoint
if 'train_x' in checkpoint and 'train_y' in checkpoint:
    train_x = checkpoint['train_x']
    train_y = checkpoint['train_y']
    print(f"Model loaded successfully with training data")
    print(f"Training data shape: {train_x.shape}")
    print(f"Training targets shape: {train_y.shape}")
else:
    print("Error: Training data not found in checkpoint. This model was not saved with training data.")
    print("Please retrain the model using the updated training script.")
    exit(1)

# Initialize model components
likelihood = gpytorch.likelihoods.GaussianLikelihood()
deepkernelmodel = SingleTaskDeepKernel(
    input_dim=train_x.shape[1], 
    train_x=train_x, 
    train_y=train_y, 
    likelihood=likelihood, 
    depth=[(train_x.shape[1], int(train_x.shape[1]/2))], 
    dropout=0.2, 
    activation='relu', 
    kernel_choice='RBF', 
    mean='Constant',
    pretrained=False, 
    feature_extractor=None, 
    latent_dim=int(train_x.shape[1]/2), 
    gphyper=None
)

# Load state dicts
deepkernelmodel.load_state_dict(model_state_dict)
likelihood.load_state_dict(likelihood_state_dict)

# Move to GPU if available
if torch.cuda.is_available():
    deepkernelmodel = deepkernelmodel.cuda(gpu_id)
    likelihood = likelihood.cuda(gpu_id)
    train_x = train_x.cuda(gpu_id)
    train_y = train_y.cuda(gpu_id)

# Set to evaluation mode
deepkernelmodel.eval()
likelihood.eval()

# Get baseline data (time = 0) for test subjects
print("Preparing baseline test data...")

test_data = datasamples

# remove any Unnamed columns
test_data = test_data.loc[:, ~test_data.columns.str.contains('^Unnamed')]

print('Columns: ', test_data.columns)

# For each subject, extract only the first (baseline) row so that
# baseline_ptids[i] and baseline_data[i] are guaranteed to correspond.
# Taking all rows and indexing by enumerate(unique_ptids) breaks the
# alignment whenever a subject has more than one row in the file.
baseline_data = []
baseline_ptids = []

for ptid in test_data['PTID.x'].unique():
    subject_data = test_data[test_data['PTID.x'] == ptid]
    if len(subject_data) > 0:
        # Get the first record (baseline, time = 0) for this subject
        baseline_record = subject_data.iloc[0]
        features = baseline_record.drop(labels=['PTID.x']).to_numpy(dtype=np.float32)
        baseline_data.append(features)
        baseline_ptids.append(ptid)

baseline_data = np.array(baseline_data)

print(f"Extracted baseline data for {len(baseline_ptids)} subjects")
print(f"Baseline feature shape: {baseline_data.shape}")

# ------------------- Denormalization Setup -------------------
def load_target_stats(biomarker, roi_idx, stats_dir):
    """Return (mean, std) used to normalize the target variable during training,
    so predictions can be converted back to the original scale."""
    import pickle

    # read the 
    muse_cols = np.load(os.path.join(stats_dir, 'muse_list.npy'), allow_pickle=True)

    def _load(path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    def _unpack(obj):
        if isinstance(obj, dict):
            return float(obj['mean']), float(obj['std'])
        elif isinstance(obj, list) and len(obj) == 2:
            return float(obj[0]), float(obj[1])
        raise ValueError(f"Unexpected stats format: {type(obj)}")

    if biomarker == 'MUSE':
        stats = _load(os.path.join(stats_dir, '145_MUSE_allstrudies_mean_std.pkl'))
        roi_col = muse_cols[roi_idx]
        return float(stats['mean']['DL_MUSE_Volume_' + str(roi_col)]), \
               float(stats['std']['DL_MUSE_Volume_'  + str(roi_col)])
    elif biomarker in ('BAG', 'SPARE_BA', 'SPARE-BA'):
        # Both BAG and SPARE_BA stats live in normalization_stats.pkl
        norm_stats = _load(os.path.join(stats_dir, 'normalization_stats.pkl'))
        key = 'SPARE_BA' if biomarker in ('SPARE_BA', 'SPARE-BA') else 'BAG'
        return float(norm_stats[key]['mean']), float(norm_stats[key]['std'])
    elif biomarker == 'SPARE_AD':
        return _unpack(_load(os.path.join(stats_dir, 'spare_ad_mean_std.pkl')))
    elif biomarker == 'MMSE':
        return _unpack(_load(os.path.join(stats_dir, 'mmse_mean_std.pkl')))
    elif biomarker == 'ADAS':
        return _unpack(_load(os.path.join(stats_dir, 'adas_mean_std.pkl')))
    else:
        raise ValueError(f"Unknown biomarker: {biomarker}")

print(f"\nLoading denormalization stats for {biomarker} (roi_idx={roi_idx})...")
try:
    target_mean, target_std = load_target_stats(biomarker, roi_idx, stats_dir)
    print(f"Target stats: mean={target_mean:.4f}, std={target_std:.4f}")
    denormalize = True
except Exception as e:
    print(f"⚠️  Could not load denormalization stats: {e}")
    print("Predictions will remain in normalized scale.")
    denormalize = False

# Create future time point data
all_results = []

for time_point in future_timepoints:
    print(f"\n=== Processing time point: {time_point} months ===")
    
    # Create future data by modifying the time component (last feature)
    future_data = baseline_data.copy()
    
    print(f"Time point: {time_point}")

    future_data[:, -1] = time_point  # Set time to future time point
    print(f"Future data: {future_data.shape}")
    # Convert to tensor
    future_tensor = torch.Tensor(future_data)
    if torch.cuda.is_available():
        future_tensor = future_tensor.cuda(gpu_id)
    
    print(f"Future data shape: {future_tensor.shape}")
    # Make predictions
    print(f"Making predictions for time point {time_point}...")
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        f_preds = deepkernelmodel(future_tensor)
        y_preds = likelihood(f_preds)
        
        mean = y_preds.mean
        variance = y_preds.variance
        lower, upper = y_preds.confidence_region()
    
    # Convert to numpy for processing
    mean_np = mean.cpu().detach().numpy()
    variance_np = variance.cpu().detach().numpy()
    lower_np = lower.cpu().detach().numpy()
    upper_np = upper.cpu().detach().numpy()

    # Inverse-normalize predictions back to original scale:
    #   y_original = y_normalized * std + mean
    #   Var(y_original) = Var(y_normalized) * std^2
    if denormalize:
        mean_np    = mean_np    * target_std + target_mean
        lower_np   = lower_np   * target_std + target_mean
        upper_np   = upper_np   * target_std + target_mean
        variance_np = variance_np * (target_std ** 2)

    # Create results for this time point
    for i, ptid in enumerate(baseline_ptids):
        result = {
            'PTID': ptid,
            'time_months': time_point,
            'predicted_value': mean_np[i],
            'variance': variance_np[i],
            'lower_bound': lower_np[i],
            'upper_bound': upper_np[i],
            'interval_width': upper_np[i] - lower_np[i],
            'biomarker': biomarker
        }
        all_results.append(result)
    
    print(f"Completed predictions for {len(baseline_ptids)} subjects at {time_point} months")

# Create results DataFrame
results_df = pd.DataFrame(all_results)

# Add summary statistics
print(f"\n=== Summary Statistics ===")
print(f"Total predictions: {len(results_df)}")
print(f"Subjects: {len(baseline_ptids)}")
print(f"Time points: {len(future_timepoints)}")
print(f"ROI: {roi_idx}")

for time_point in future_timepoints:
    tp_data = results_df[results_df['time_months'] == time_point]
    mean_pred = tp_data['predicted_value'].mean()
    std_pred = tp_data['predicted_value'].std()
    mean_uncertainty = tp_data['interval_width'].mean()
    print(f"Time {time_point}m: Mean={mean_pred:.4f} ± {std_pred:.4f}, Uncertainty={mean_uncertainty:.4f}")

# Save results
results_df.to_csv(output_file, index=False)
print(f"\nResults saved to: {output_file}")

# Save summary metrics
summary_metrics = {
    'roi_idx': roi_idx,
    'n_subjects': len(baseline_ptids),
    'n_timepoints': len(future_timepoints),
    'future_timepoints': future_timepoints,
    'n_total_predictions': len(results_df),
    'mean_prediction_by_timepoint': {
        str(tp): float(results_df[results_df['time_months'] == tp]['predicted_value'].mean())
        for tp in future_timepoints
    },
    'mean_uncertainty_by_timepoint': {
        str(tp): float(results_df[results_df['time_months'] == tp]['interval_width'].mean())
        for tp in future_timepoints
    }
}

summary_file = output_file.replace('.csv', '_summary.json')
with open(summary_file, 'w') as f:
    json.dump(summary_metrics, f, indent=2)

print(f"Summary metrics saved to: {summary_file}")
print("Future time point inference completed successfully!")