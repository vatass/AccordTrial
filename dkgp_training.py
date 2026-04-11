'''
Population DKGP Model Training
'''

import pandas as pd
import numpy as np
import sys
import torch
import gpytorch
from utils import *
import pickle
from models import SingleTaskDeepKernel
import argparse
import json
import time
import logging
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

parser = argparse.ArgumentParser(description='Temporal Deep Kernel Single Task GP model for a Biomarker')
## Production Parameters
parser.add_argument("--data_file", help="Path to the data CSV file", required=True)
parser.add_argument("--train_ids_file", help="Path to the train IDs pickle file", required=True)
parser.add_argument("--test_ids_file", help="Path to the test IDs pickle file", required=True)
parser.add_argument("--biomarker_index", help="biomarker index to train on", type=int, required=True)
parser.add_argument("--biomarker_name", help="Biomarker name", type=str, required=True)
parser.add_argument("--output_dir", help="Directory to save model outputs", default="./models")
parser.add_argument("--gpu_id", help="GPU ID to use", type=int, default=0)
parser.add_argument("--fold", help="fold", type=int, default=0)
parser.add_argument("--covariates_file", help="Path to longitudinal covariates CSV (for per-subject Sex/Age metadata)",
                    default="./data/longitudinal_covariates_bag_allstudies.csv")



t0 = time.time()
args = parser.parse_args()

# Parse arguments
gpu_id = args.gpu_id
biomarker_index = args.biomarker_index
biomarker_name = args.biomarker_name
data_file = args.data_file
fold = args.fold
train_ids_file = args.train_ids_file
test_ids_file = args.test_ids_file
output_dir = args.output_dir

# Create output directory if it doesn't exist
import os
os.makedirs(output_dir, exist_ok=True)

# Set up logging to both console and file
log_filename = os.path.join(output_dir, f'training_{biomarker_name}_{biomarker_index}_{fold}.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)
logger.info(f"Log file: {log_filename}")

# Load data
logger.info(f"Loading data from {data_file}")
datasamples = pd.read_csv(data_file)
subject_ids = list(datasamples['PTID'].unique())
logger.info(f"Loaded {len(subject_ids)} subjects")

accord_test_data = pd.read_csv('./data/subjectsamples_bag_accord.csv')

# Load normalization stats once — used for both per-sample predictions and 8-year forecast
target_mean, target_std = 0.0, 1.0
denormalize = False
_norm_path = './data/normalization_stats.pkl'
if os.path.exists(_norm_path):
    with open(_norm_path, 'rb') as _f:
        _norm = pickle.load(_f)
    _key = (biomarker_name if biomarker_name in _norm
            else biomarker_name.upper() if biomarker_name.upper() in _norm
            else None)
    if _key:
        target_mean = float(_norm[_key]['mean'])
        target_std  = float(_norm[_key]['std'])
        denormalize = True
        logger.info(f"Denormalization stats for '{_key}': mean={target_mean:.4f}, std={target_std:.4f}")
    else:
        logger.info(f"Biomarker '{biomarker_name}' not found in normalization_stats.pkl; predictions stay normalized")
else:
    logger.info(f"normalization_stats.pkl not found; predictions will stay in normalized scale")

# Load train/test split
logger.info(f"Loading train IDs from {train_ids_file}")
with open(train_ids_file, "rb") as openfile:
    train_ids = []
    while True:
        try:
            train_ids.append(pickle.load(openfile))
        except EOFError:
            break
train_ids = train_ids[0]

logger.info(f"Loading test IDs from {test_ids_file}")
with open(test_ids_file, "rb") as openfile:
    test_ids = []
    while True:
        try:
            test_ids.append(pickle.load(openfile))
        except EOFError:
            break
test_ids = test_ids[0]

logger.info(f'Train IDs: {len(train_ids)}')
logger.info(f'Test IDs: {len(test_ids)}')

# Verify no overlap
for t in test_ids:
    if t in train_ids:
        raise ValueError('Test Samples belong to the train!')

# Prepare data
train_x = datasamples[datasamples['PTID'].isin(train_ids)]['X']
train_y = datasamples[datasamples['PTID'].isin(train_ids)]['Y']
test_x = datasamples[datasamples['PTID'].isin(test_ids)]['X']
test_y = datasamples[datasamples['PTID'].isin(test_ids)]['Y']

accord_test_x = accord_test_data['X']
accord_test_y = accord_test_data['Y']

# Extract per-sample metadata for ACCORD predictions (before tensor conversion)
accord_ptids_list = accord_test_data['PTID'].tolist()
accord_time_list = [float(x_str.strip('][').split(', ')[-1]) for x_str in accord_test_data['X']]

# Process ACCORD data from string representation to tensors
accord_x_data = []
for x_str in accord_test_data['X']:
    a = x_str.strip('][').split(', ')
    accord_x_data.append(np.expand_dims(np.array([float(v) for v in a]), 0))

accord_y_data = []
for y_str in accord_test_data['Y']:
    a = y_str.strip('][').split(', ')
    accord_y_data.append(np.expand_dims(np.array([float(v) for v in a]), 0))

accord_test_x = torch.Tensor(np.concatenate(accord_x_data, axis=0))
accord_test_y = torch.Tensor(np.concatenate(accord_y_data, axis=0))

# Extract per-sample metadata for prediction tracking (before tensor conversion)
test_data_raw = datasamples[datasamples['PTID'].isin(test_ids)]
test_ptids_list = test_data_raw['PTID'].tolist()
# Time is the last element of the feature vector in X
test_time_list = [float(x_str.strip('][').split(', ')[-1]) for x_str in test_data_raw['X']]

logger.info(f'Train data shape: {train_x.shape}')
logger.info(f'Test data shape: {test_x.shape}')
# Process data
train_x, train_y, test_x, test_y = process_temporal_singletask_data(train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y, test_ids=test_ids)

# Move to GPU if available
if torch.cuda.is_available():
    train_x = train_x.cuda(gpu_id)
    train_y = train_y.cuda(gpu_id)
    test_x = test_x.cuda(gpu_id)
    test_y = test_y.cuda(gpu_id)

    accord_test_x = accord_test_x.cuda(gpu_id)
    accord_test_y = accord_test_y.cuda(gpu_id)

logger.info(f'Processed Train Data: {train_x.shape}')
logger.info(f'Processed Test Data: {test_x.shape}')

logger.info(f'Processed ACCORD Test Data: {accord_test_x.shape}')
logger.info(f'Processed ACCORD Test Data: {accord_test_y.shape}')


logger.info("=== FEATURE VERIFICATION ===")
logger.info(f"Number of features in training data: {train_x.shape[1]}")
logger.info(f"Number of features in accord test data: {accord_test_x.shape[1]}")
logger.info("=== END VERIFICATION ===")

# Select ROI
test_y = test_y[:, biomarker_index]
train_y = train_y[:, biomarker_index]
accord_test_y = accord_test_y[:, biomarker_index]
train_y = train_y.squeeze()
test_y = test_y.squeeze()
accord_test_y = accord_test_y.squeeze()


# Define model with fixed architecture
depth = [(train_x.shape[1], int(train_x.shape[1]/2))]
likelihood = gpytorch.likelihoods.GaussianLikelihood()
deepkernelmodel = SingleTaskDeepKernel(
    input_dim=train_x.shape[1],
    train_x=train_x,
    train_y=train_y,
    likelihood=likelihood,
    depth=depth,
    dropout=0.2,
    activation='relu',
    kernel_choice='RBF',
    mean='Constant',
    pretrained=False,
    feature_extractor=None,
    latent_dim=int(train_x.shape[1]/2),
    gphyper=None
)

if torch.cuda.is_available():
    likelihood = likelihood.cuda(gpu_id)
    deepkernelmodel = deepkernelmodel.cuda(gpu_id)

# Training setup
deepkernelmodel.feature_extractor.train()
deepkernelmodel.train()
deepkernelmodel.likelihood.train()

optimizer = torch.optim.Adam([
    {'params': deepkernelmodel.feature_extractor.parameters(), 'lr': 0.02, 'weight_decay': 0.01},
    {'params': deepkernelmodel.covar_module.parameters(), 'lr': 0.02},
    {'params': deepkernelmodel.mean_module.parameters(), 'lr': 0.02},
    {'params': deepkernelmodel.likelihood.parameters(), 'lr': 0.02}
])

mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, deepkernelmodel)

# Training loop
iterations = 300
logger.info(f"Training for {iterations} iterations...")
for i in range(iterations):
    deepkernelmodel.train()
    likelihood.train()
    optimizer.zero_grad()
    output = deepkernelmodel(train_x)
    loss = -mll(output, train_y)
    loss.backward()
    optimizer.step()

    if (i+1) % 50 == 0:
        logger.info(f'Iteration {i+1}/{iterations} - Loss: {loss.item():.3f}')

# Evaluation
deepkernelmodel.eval()
likelihood.eval()

with torch.no_grad(), gpytorch.settings.fast_pred_var():
    f_preds = deepkernelmodel(test_x)
    mean = f_preds.mean
    variance = f_preds.variance
    lower = mean - 1.645 * f_preds.stddev
    upper = mean + 1.645 * f_preds.stddev

# Calculate metrics
mae_pop = mean_absolute_error(test_y.cpu().detach().numpy(), mean.cpu().detach().numpy())
mse_pop = mean_squared_error(test_y.cpu().detach().numpy(), mean.cpu().detach().numpy())
rmse_pop = np.sqrt(mse_pop)
rsq = r2_score(test_y.cpu().detach().numpy(), mean.cpu().detach().numpy())

coverage, interval_width, mean_coverage, mean_interval_width = calc_coverage(
    predictions=mean.cpu().detach().numpy(),
    groundtruth=test_y.cpu().detach().numpy(),
    intervals=[lower.cpu().detach().numpy(), upper.cpu().detach().numpy()]
)

coverage, interval_width, mean_coverage, mean_interval_width = coverage.numpy().astype(int), interval_width.numpy(), mean_coverage.numpy(), mean_interval_width.numpy()

logger.info(f"Results for Biomarker {biomarker_name}:")
logger.info(f"MAE: {mae_pop:.4f}")
logger.info(f"MSE: {mse_pop:.4f}")
logger.info(f"RMSE: {rmse_pop:.4f}")
logger.info(f"R²: {rsq:.4f}")
logger.info(f"Coverage: {np.mean(coverage):.4f}")
logger.info(f"Interval Width: {mean_interval_width:.4f}")

# ------------------------------------------------------------------
# ACCORD Inference
# ------------------------------------------------------------------
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    accord_f_preds = deepkernelmodel(accord_test_x)
    accord_mean = accord_f_preds.mean
    accord_variance = accord_f_preds.variance
    accord_lower = accord_mean - 1.645 * accord_f_preds.stddev
    accord_upper = accord_mean + 1.645 * accord_f_preds.stddev

# Calculate ACCORD metrics
accord_mae = mean_absolute_error(accord_test_y.cpu().detach().numpy(), accord_mean.cpu().detach().numpy())
accord_mse = mean_squared_error(accord_test_y.cpu().detach().numpy(), accord_mean.cpu().detach().numpy())
accord_rmse = np.sqrt(accord_mse)
accord_rsq = r2_score(accord_test_y.cpu().detach().numpy(), accord_mean.cpu().detach().numpy())

accord_coverage, accord_interval_width, accord_mean_coverage, accord_mean_interval_width = calc_coverage(
    predictions=accord_mean.cpu().detach().numpy(),
    groundtruth=accord_test_y.cpu().detach().numpy(),
    intervals=[accord_lower.cpu().detach().numpy(), accord_upper.cpu().detach().numpy()]
)
accord_coverage, accord_interval_width, accord_mean_coverage, accord_mean_interval_width = (
    accord_coverage.numpy().astype(int),
    accord_interval_width.numpy(),
    accord_mean_coverage.numpy(),
    accord_mean_interval_width.numpy()
)

logger.info(f"ACCORD Results for Biomarker {biomarker_name}:")
logger.info(f"MAE: {accord_mae:.4f}")
logger.info(f"MSE: {accord_mse:.4f}")
logger.info(f"RMSE: {accord_rmse:.4f}")
logger.info(f"R²: {accord_rsq:.4f}")
logger.info(f"Coverage: {np.mean(accord_coverage):.4f}")
logger.info(f"Interval Width: {accord_mean_interval_width:.4f}")

# ACCORD per-sample predictions DataFrame
accord_test_y_np   = accord_test_y.cpu().detach().numpy()
accord_mean_np     = accord_mean.cpu().detach().numpy()
accord_lower_np    = accord_lower.cpu().detach().numpy()
accord_upper_np    = accord_upper.cpu().detach().numpy()
accord_variance_np = accord_variance.cpu().detach().numpy()

# Denormalize to original scale (same stats used by the 8-year forecast)
if denormalize:
    accord_test_y_np   = accord_test_y_np   * target_std + target_mean
    accord_mean_np     = accord_mean_np     * target_std + target_mean
    accord_lower_np    = accord_lower_np    * target_std + target_mean
    accord_upper_np    = accord_upper_np    * target_std + target_mean
    accord_variance_np = accord_variance_np * (target_std ** 2)

accord_interval_np = accord_upper_np - accord_lower_np
accord_abs_error   = np.abs(accord_test_y_np - accord_mean_np)
accord_sq_error    = (accord_test_y_np - accord_mean_np) ** 2

accord_predictions_df = pd.DataFrame({
    'PTID':           accord_ptids_list,
    'time_months':    accord_time_list,
    'ground_truth':   accord_test_y_np,
    'predicted':      accord_mean_np,
    'lower_bound':    accord_lower_np,
    'upper_bound':    accord_upper_np,
    'variance':       accord_variance_np,
    'interval_width': accord_interval_np,
    'abs_error':      accord_abs_error,
    'squared_error':  accord_sq_error,
    'covered':        accord_coverage.astype(int),
})

accord_predictions_filename = os.path.join(output_dir, f'accord_predictions_{biomarker_name}_{biomarker_index}_{fold}.csv')
accord_predictions_df.to_csv(accord_predictions_filename, index=False)
logger.info(f"ACCORD per-sample predictions saved to {accord_predictions_filename}")

# ACCORD per-subject aggregated metrics
accord_subject_metrics = (
    accord_predictions_df.groupby('PTID')
    .agg(
        n_timepoints        = ('ground_truth', 'count'),
        mae                 = ('abs_error', 'mean'),
        mse                 = ('squared_error', 'mean'),
        coverage_rate       = ('covered', 'mean'),
        mean_interval_width = ('interval_width', 'mean'),
        mean_predicted      = ('predicted', 'mean'),
        mean_ground_truth   = ('ground_truth', 'mean'),
    )
    .reset_index()
)
accord_subject_metrics['rmse'] = np.sqrt(accord_subject_metrics['mse'])

accord_subject_metrics_filename = os.path.join(output_dir, f'accord_subject_metrics_{biomarker_name}_{biomarker_index}_{fold}.csv')
accord_subject_metrics.to_csv(accord_subject_metrics_filename, index=False)
logger.info(f"ACCORD per-subject metrics saved to {accord_subject_metrics_filename}")

logger.info(f"ACCORD per-subject metric summary (n={len(accord_subject_metrics)} subjects):")
logger.info(f"  MAE  — mean: {accord_subject_metrics['mae'].mean():.4f}, median: {accord_subject_metrics['mae'].median():.4f}")
logger.info(f"  RMSE — mean: {accord_subject_metrics['rmse'].mean():.4f}")
logger.info(f"  Coverage — mean: {accord_subject_metrics['coverage_rate'].mean():.4f}")

# ------------------------------------------------------------------
# Per-sample predictions DataFrame
# ------------------------------------------------------------------
test_y_np      = test_y.cpu().detach().numpy()
mean_np        = mean.cpu().detach().numpy()
lower_np       = lower.cpu().detach().numpy()
upper_np       = upper.cpu().detach().numpy()
variance_np    = variance.cpu().detach().numpy()
interval_np    = upper_np - lower_np
abs_error_np   = np.abs(test_y_np - mean_np)
sq_error_np    = (test_y_np - mean_np) ** 2

predictions_df = pd.DataFrame({
    'PTID':           test_ptids_list,
    'time_months':    test_time_list,
    'ground_truth':   test_y_np,
    'predicted':      mean_np,
    'lower_bound':    lower_np,
    'upper_bound':    upper_np,
    'variance':       variance_np,
    'interval_width': interval_np,
    'abs_error':      abs_error_np,
    'squared_error':  sq_error_np,
    'covered':        coverage.astype(int),
})

# Optionally enrich with baseline Sex and Age from covariates file
if args.covariates_file and os.path.exists(args.covariates_file):
    cov_df = pd.read_csv(args.covariates_file)
    cov_df['PTID'] = cov_df['PTID'].astype(str)
    predictions_df['PTID'] = predictions_df['PTID'].astype(str)
    # Use age at first (baseline) timepoint per subject
    baseline_cov = (cov_df.sort_values('Time')
                          .groupby('PTID')
                          .first()
                          .reset_index()[['PTID', 'Sex', 'Age']]
                          .rename(columns={'Age': 'BaselineAge'}))
    predictions_df = predictions_df.merge(baseline_cov, on='PTID', how='left')
    logger.info(f"Covariates merged: Sex/BaselineAge added for {predictions_df['Sex'].notna().sum()} observations")

# Save per-sample predictions
predictions_filename = os.path.join(output_dir, f'predictions_{biomarker_name}_{biomarker_index}_{fold}.csv')
predictions_df.to_csv(predictions_filename, index=False)
logger.info(f"Per-sample predictions saved to {predictions_filename}")

# ------------------------------------------------------------------
# Per-subject aggregated metrics
# ------------------------------------------------------------------
subject_metrics = (
    predictions_df.groupby('PTID')
    .agg(
        n_timepoints      = ('ground_truth', 'count'),
        mae               = ('abs_error', 'mean'),
        mse               = ('squared_error', 'mean'),
        coverage_rate     = ('covered', 'mean'),
        mean_interval_width = ('interval_width', 'mean'),
        mean_predicted    = ('predicted', 'mean'),
        mean_ground_truth = ('ground_truth', 'mean'),
    )
    .reset_index()
)
subject_metrics['rmse'] = np.sqrt(subject_metrics['mse'])

# Carry over baseline covariates if present
for col in ['Sex', 'BaselineAge']:
    if col in predictions_df.columns:
        first_vals = predictions_df.groupby('PTID')[col].first().reset_index()
        subject_metrics = subject_metrics.merge(first_vals, on='PTID', how='left')

subject_metrics_filename = os.path.join(output_dir, f'subject_metrics_{biomarker_name}_{biomarker_index}_{fold}.csv')
subject_metrics.to_csv(subject_metrics_filename, index=False)
logger.info(f"Per-subject metrics saved to {subject_metrics_filename}")

logger.info(f"Per-subject metric summary (n={len(subject_metrics)} subjects):")
logger.info(f"  MAE  — mean: {subject_metrics['mae'].mean():.4f}, median: {subject_metrics['mae'].median():.4f}")
logger.info(f"  RMSE — mean: {subject_metrics['rmse'].mean():.4f}")
logger.info(f"  Coverage — mean: {subject_metrics['coverage_rate'].mean():.4f}")

# Save model
model_filename = os.path.join(output_dir, f'deep_kernel_gp_{biomarker_name}_{biomarker_index}_{fold}.pth')
save_model(deepkernelmodel, optimizer, likelihood, filename=model_filename, train_x=train_x, train_y=train_y)

# Save results
results = {
    'mae': float(mae_pop),
    'mse': float(mse_pop),
    'rmse': float(rmse_pop),
    'r2': float(rsq),
    'coverage': float(np.mean(coverage)),
    'interval_width': float(mean_interval_width),
    'training_time': float(time.time() - t0),
    'predictions_file': predictions_filename,
    'subject_metrics_file': subject_metrics_filename,
    'n_test_subjects': int(len(test_ids)),
    'n_test_observations': int(len(predictions_df)),
    'accord_mae': float(accord_mae),
    'accord_mse': float(accord_mse),
    'accord_rmse': float(accord_rmse),
    'accord_r2': float(accord_rsq),
    'accord_coverage': float(np.mean(accord_coverage)),
    'accord_interval_width': float(accord_mean_interval_width),
    'accord_predictions_file': accord_predictions_filename,
    'accord_subject_metrics_file': accord_subject_metrics_filename,
    'n_accord_subjects': int(accord_test_data['PTID'].nunique()),
    'n_accord_observations': int(len(accord_predictions_df)),
}


# ------------------------------------------------------------------
# ACCORD Inference for 8 years ahead
# ------------------------------------------------------------------
logger.info("=== ACCORD 8-Year Forecast ===")
future_timepoints = [0, 12, 24, 36, 48, 60, 72, 84, 96]

# Extract per-subject baseline features (first visit per PTID).
# Feature vector layout: 147 base features + 1 time value (last position).
accord_forecast_ptids = []
accord_baseline_features = []
for ptid in accord_test_data['PTID'].unique():
    first_row = accord_test_data[accord_test_data['PTID'] == ptid].iloc[0]
    a = first_row['X'].strip('][').split(', ')
    features = np.array([float(v) for v in a])
    accord_forecast_ptids.append(ptid)
    accord_baseline_features.append(features)

accord_baseline_np = np.array(accord_baseline_features)  # shape: [n_subjects, 148]
logger.info(f"Extracted baseline features for {len(accord_forecast_ptids)} ACCORD subjects "
            f"(shape: {accord_baseline_np.shape})")

# Normalization stats already loaded at startup — reuse for forecast
forecast_denormalize = denormalize
forecast_target_mean = target_mean
forecast_target_std  = target_std

# Run model inference for each future timepoint
all_forecast_rows = []
deepkernelmodel.eval()
likelihood.eval()

for tp in future_timepoints:
    # Copy baseline features and replace time (last feature) with the future timepoint
    forecast_data = accord_baseline_np.copy()
    forecast_data[:, -1] = float(tp)

    forecast_tensor = torch.Tensor(forecast_data)
    if torch.cuda.is_available():
        forecast_tensor = forecast_tensor.cuda(gpu_id)

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        fp_preds = deepkernelmodel(forecast_tensor)
        fc_mean     = fp_preds.mean
        fc_variance = fp_preds.variance
        fc_lower    = fc_mean - 1.645 * fp_preds.stddev
        fc_upper    = fc_mean + 1.645 * fp_preds.stddev

    fc_mean_np     = fc_mean.cpu().detach().numpy()
    fc_variance_np = fc_variance.cpu().detach().numpy()
    fc_lower_np    = fc_lower.cpu().detach().numpy()
    fc_upper_np    = fc_upper.cpu().detach().numpy()

    # Inverse-normalize to original scale: y_orig = y_norm * std + mean
    if forecast_denormalize:
        fc_mean_np     = fc_mean_np     * forecast_target_std + forecast_target_mean
        fc_lower_np    = fc_lower_np    * forecast_target_std + forecast_target_mean
        fc_upper_np    = fc_upper_np    * forecast_target_std + forecast_target_mean
        fc_variance_np = fc_variance_np * (forecast_target_std ** 2)

    for i, ptid in enumerate(accord_forecast_ptids):
        all_forecast_rows.append({
            'PTID':           ptid,
            'time_months':    tp,
            'predicted':      fc_mean_np[i],
            'lower_bound':    fc_lower_np[i],
            'upper_bound':    fc_upper_np[i],
            'variance':       fc_variance_np[i],
            'interval_width': fc_upper_np[i] - fc_lower_np[i],
        })

    logger.info(f"  t={tp:>2}m: mean={fc_mean_np.mean():.4f}, "
                f"interval_width={(fc_upper_np - fc_lower_np).mean():.4f}")

forecast_df = pd.DataFrame(all_forecast_rows)
forecast_filename = os.path.join(
    output_dir,
    f'accord_eight_year_forecast_{biomarker_name}_{biomarker_index}_{fold}.csv'
)
forecast_df.to_csv(forecast_filename, index=False)
logger.info(f"ACCORD 8-year forecast saved to {forecast_filename} ({len(forecast_df)} rows, "
            f"{len(accord_forecast_ptids)} subjects × {len(future_timepoints)} timepoints)")

results['accord_eight_year_forecast_file'] = forecast_filename
results['accord_eight_year_forecast_n_subjects'] = int(len(accord_forecast_ptids))
results['accord_eight_year_forecast_timepoints'] = future_timepoints
results['accord_eight_year_forecast_denormalized'] = bool(forecast_denormalize)

results_filename = os.path.join(output_dir, f'results_biomarker_{biomarker_name}_{biomarker_index}_{fold}.json')
with open(results_filename, 'w') as f:
    json.dump(results, f, indent=2)

logger.info(f"Model and results saved to {output_dir}")
logger.info(f"Training completed in {time.time() - t0:.2f} seconds")
