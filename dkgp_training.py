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



t0 = time.time()
args = parser.parse_args()

# Parse arguments
gpu_id = args.gpu_id
biomarker_index = args.biomarker_index
biomarker_name = args.biomarker_index
data_file = args.data_file
fold = args.fold
train_ids_file = args.train_ids_file
test_ids_file = args.test_ids_file
output_dir = args.output_dir

# Create output directory if it doesn't exist
import os
os.makedirs(output_dir, exist_ok=True)

# Load data
print(f"Loading data from {data_file}")
datasamples = pd.read_csv(data_file)
subject_ids = list(datasamples['PTID'].unique()) 
print(f"Loaded {len(subject_ids)} subjects")

# Load train/test split
print(f"Loading train IDs from {train_ids_file}")
with open(train_ids_file, "rb") as openfile:
    train_ids = []
    while True:
        try:
            train_ids.append(pickle.load(openfile))
        except EOFError:
            break 
train_ids = train_ids[0]

print(f"Loading test IDs from {test_ids_file}")
with open(test_ids_file, "rb") as openfile:
    test_ids = []
    while True:
        try:
            test_ids.append(pickle.load(openfile))
        except EOFError:
            break
test_ids = test_ids[0]

print(f'Train IDs: {len(train_ids)}')
print(f'Test IDs: {len(test_ids)}')

# Verify no overlap
for t in test_ids: 
    if t in train_ids: 
        raise ValueError('Test Samples belong to the train!')

# Prepare data
train_x = datasamples[datasamples['PTID'].isin(train_ids)]['X']
train_y = datasamples[datasamples['PTID'].isin(train_ids)]['Y']    
test_x = datasamples[datasamples['PTID'].isin(test_ids)]['X']
test_y = datasamples[datasamples['PTID'].isin(test_ids)]['Y']

print('Train data shape:', train_x.shape)
print('Test data shape:', test_x.shape)

# Process data
train_x, train_y, test_x, test_y = process_temporal_singletask_data(train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y, test_ids=test_ids)

# Move to GPU if available
if torch.cuda.is_available():
    train_x = train_x.cuda(gpu_id) 
    train_y = train_y.cuda(gpu_id)
    test_x = test_x.cuda(gpu_id) 
    test_y = test_y.cuda(gpu_id)

print('Processed Train Data:', train_x.shape)
print('Processed Test Data:', test_x.shape)

print("\n=== FEATURE VERIFICATION ===")
print(f"Number of features in training data: {train_x.shape[1]}")
print("=== END VERIFICATION ===\n")

# Select ROI
test_y = test_y[:, biomarker_index]
train_y = train_y[:, biomarker_index]
train_y = train_y.squeeze() 
test_y = test_y.squeeze()

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
    {'params': deepkernelmodel.feature_extractor.parameters(), 'lr': 0.01, 'weight_decay': 1e-4},
    {'params': deepkernelmodel.covar_module.parameters(), 'lr': 0.01},
    {'params': deepkernelmodel.mean_module.parameters(), 'lr': 0.01},
    {'params': deepkernelmodel.likelihood.parameters(), 'lr': 0.01}
])

mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, deepkernelmodel)

# Training loop
iterations = 1500
print(f"Training for {iterations} iterations...")
for i in range(iterations):
    deepkernelmodel.train()
    likelihood.train()
    optimizer.zero_grad()
    output = deepkernelmodel(train_x)
    loss = -mll(output, train_y)
    loss.backward()
    optimizer.step()
    
    if (i+1) % 50 == 0:
        print(f'Iteration {i+1}/{iterations} - Loss: {loss.item():.3f}')

# Evaluation
deepkernelmodel.eval()
likelihood.eval()

with torch.no_grad(), gpytorch.settings.fast_pred_var():
    f_preds = deepkernelmodel(test_x)
    y_preds = likelihood(f_preds)
    mean = y_preds.mean
    variance = y_preds.variance
    lower, upper = y_preds.confidence_region()

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

print(f"\nResults for Biomarker {biomarker_name}:")
print(f"MAE: {mae_pop:.4f}")
print(f"MSE: {mse_pop:.4f}")
print(f"RMSE: {rmse_pop:.4f}")
print(f"R²: {rsq:.4f}")
print(f"Coverage: {np.mean(coverage):.4f}")
print(f"Interval Width: {mean_interval_width:.4f}")

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
    'training_time': float(time.time() - t0)
}

results_filename = os.path.join(output_dir, f'results_biomarker_{biomarker_name}_{biomarker_index}_{fold}.json')
with open(results_filename, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nModel and results saved to {output_dir}")
print(f"Training completed in {time.time() - t0:.2f} seconds") 



