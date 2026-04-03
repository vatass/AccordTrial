'''
ACCORD - CN Digital Twin - DKGP 
'''


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KernelDensity
import sklearn
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, mean_absolute_error
from sklearn.model_selection import StratifiedKFold, KFold
import plotly.graph_objects as go
# from synthcity.metrics import Metrics
# from synthcity.plugins.core.dataloader import GenericDataLoader
import fnmatch
import seaborn as sns
import torch
import sys
import pickle



def create_baseline_temporal_dataset(subjects, dataframe, dataframeunnorm, target, features,hmuse, genomic, followup, derivedroi,  visualize=False):
    '''
    subjects: list of the subject ids
    dataframe: dataframe with all the data
    target: H_MUSE ROI features
    '''
    print('Target', target)
    cnt = 0
    num_samples = 0
    list_of_subjects, list_of_subject_ids = [], []
    data_x, data_y, data_xbase = [], [], []

    samples = {'PTID': [], 'X': [], 'Y': []}
    covariates = {'PTID': [], 'Age': [], 'BaselineDiagnosis': [], 'BaselineAge': [], 'Sex': [] , 'APOE4_Alleles': [], 'Education_Years': [], 'Status': []}

    longitudinal_covariates = {'PTID': [], 'Time': [], 'Age': [],  'Diagnosis': [], 'Hypertension': [],
                               'Diabetes': [], 'DLICV': [], 'Study': [], 'Education_Years': [], 'Race': [], 'Sex': [], 'APOE4_Alleles': [], 'SPARE_BA': [], 'SPARE_AD': [], 'MRI_Scanner_Model': [], 
                               'CDR_Global': [], 'Tau_CSF': [], 'Abeta_CSF': [], 'PTau_CSF': [], 'MMSE_nearest_2.0': [] }


    if visualize:
        vdata = {'target': [], 'class': [], 'time': [], 'id': []}
        cnt = 0

    # remove the PTID from the features!
    features.remove('PTID')
    features.remove('Delta_Baseline')
    features.remove('Time')
    # hmuse = [i for i in features if i.startswith('H_MUSE')]

    # print('Features', features)
    clinical_features = [f for f in features if not f.startswith('H_MUSE')]
    # print('Clinical Features', clinical_features)

    # target = [t for t in target if t.startswith('H_')]
    print('Target', len(target))
    print('Input Features', features)

    for i, subject_id in enumerate(subjects):

        subject = dataframe[dataframe['PTID']==subject_id]
        subject_unnorm = dataframeunnorm[dataframeunnorm['PTID']==subject_id]

        first_diagnosis = subject['Diagnosis_nearest_2.0'].iloc[0]
        last_diagnosis = subject['Diagnosis_nearest_2.0'].iloc[-1]

        if first_diagnosis == 0 and last_diagnosis == 0:
            status = 'Non-Progressor'
        elif first_diagnosis == 0 and last_diagnosis != 0: 
            status = 'Progressor'
        else: 
            status = 'MCI/Demented Stable'

        # print(subject)
        for k in range(0, subject.shape[0]):
            samples['PTID'].append(subject_id)
            covariates['PTID'].append(subject_id)

            print('Baseline Features',  features)

            x = subject[features].iloc[0].to_list()

            # print(x)

            delta = subject['Time'].iloc[k]
            # man_device = subject['MRI_Scanner_Model'].iloc[k]
            diagnosis = subject['Diagnosis_nearest_2.0'].iloc[k]
            baseline_diagnosis = subject['Diagnosis_nearest_2.0'].iloc[0]
            baseline_age = subject_unnorm['Age'].iloc[0]
            age = subject_unnorm['Age'].iloc[k]
            dlicv = subject_unnorm['DLICV'].iloc[k]
            study = subject_unnorm['Study'].iloc[k]
            edu_years = subject['Education_Years'].iloc[k]
            race = subject_unnorm['Race'].iloc[k]
            sex = subject['Sex'].iloc[k]
            apoe4 = subject['APOE4_Alleles'].iloc[k]
            hypertension = subject_unnorm['Hypertension'].iloc[k]
            diabetes = subject_unnorm['Diabetes'].iloc[k]
            spba = subject_unnorm['SPARE_BA'].iloc[k]
            spad = subject_unnorm['SPARE_AD'].iloc[k]
            scanner = subject_unnorm['MRI_Scanner_Model'].iloc[k]
            cdr_global = subject_unnorm['CDR_Global'].iloc[k]
            tau_csf = subject_unnorm['Tau_CSF'].iloc[k]
            abeta_csf = subject_unnorm['Abeta_CSF'].iloc[k]
            ptau_csf = subject_unnorm['PTau_CSF'].iloc[k]
            mmse = subject_unnorm['MMSE_nearest_2.0'].iloc[k]

            # print('Delta', delta)
            x.extend([delta])

            # print('Input', x)
            # print('Target', target)
            t = subject[target].iloc[k] #.to_list()

            print('Target', t)
            # covariates['MRI_Scanner_Model'].append(man_device)
            covariates['Age'].append(age)
            covariates['BaselineDiagnosis'].append(baseline_diagnosis)
            covariates['BaselineAge'].append(baseline_age)
            covariates['Sex'].append(sex) 
            covariates['APOE4_Alleles'].append(apoe4)
            covariates['Education_Years'].append(edu_years) 
            covariates['Status'].append(status)
                                                 
            longitudinal_covariates['PTID'].append(subject_id)
            longitudinal_covariates['Time'].append(delta)
            longitudinal_covariates['Age'].append(age)
            longitudinal_covariates['Diagnosis'].append(diagnosis)
            longitudinal_covariates['DLICV'].append(dlicv)
            longitudinal_covariates['Study'].append(study)
            longitudinal_covariates['Education_Years'].append(edu_years)
            longitudinal_covariates['Race'].append(race)
            longitudinal_covariates['Sex'].append(sex)
            longitudinal_covariates['APOE4_Alleles'].append(apoe4)
            longitudinal_covariates['Hypertension'].append(hypertension)
            longitudinal_covariates['Diabetes'].append(diabetes)
            longitudinal_covariates['SPARE_BA'].append(spba)
            longitudinal_covariates['SPARE_AD'].append(spba)
            longitudinal_covariates['MRI_Scanner_Model'].append(scanner)
            longitudinal_covariates['CDR_Global'].append(cdr_global)
            longitudinal_covariates['Tau_CSF'].append(tau_csf)
            longitudinal_covariates['PTau_CSF'].append(ptau_csf)
            longitudinal_covariates['Abeta_CSF'].append(abeta_csf)
            longitudinal_covariates['MMSE_nearest_2.0'].append(mmse)

            samples['X'].append(x)
            samples['Y'].append(t.tolist())

            data_x.append(x)
            data_y.append(t)

        subject_data = list(zip(data_x, data_y))
        num_samples +=len(subject_data)
        list_of_subjects.append(subject_data)
        list_of_subject_ids.append(subject_id)

    assert len(samples['PTID']) == len(samples['X'])
    assert len(samples['X']) == len(samples['Y'])

    return samples, subject_data, num_samples, list_of_subjects, list_of_subject_ids, cnt, covariates, longitudinal_covariates

"""**Data Selection**
1. Read Data and remove all ADNI Screening and BLSA 1.5 T
2. Drop all NaN MUSE
3. Map the Diagnosis Column

"""
data = pd.read_pickle('/cbica/projects/ISTAGING/Pipelines/ISTAGING_Data_Consolidation_2020/v2.0/istaging.pkl.gz')

for c in data.columns:
    print(c)


# studies = [ 'ADNI', 'BLSA', 'WRAP', 'HABS', 'CARDIA', 'AIBL', 'PENN', 'PreventAD', 'OASIS'] 

## WHIMS seems to have an issue
# data = data.loc[data.Study.isin(studies)] # was: ADNI, BLSA
print(data.shape)

print('Total BLSA data')
print(data[data['Study']=='BLSA'].shape)
print('BLSA Subjects', len(data[data['Study']=='BLSA']['PTID'].unique()))

print('Remove the 1.5T BLSA Data')
data = data[data['SITE']!='BLSA-1.5T']

print('BLSA Subjects', len(data[data['Study']=='BLSA']['PTID'].unique()))

# revome duplicate visit codes per subject
data = data.drop_duplicates(subset=['PTID', 'Visit_Code'], keep='first')

print('Revome all the rows that have Visit_Code == ADNI Screening')
data = data[data['Visit_Code']!='ADNI Screening']
data = data[data['Visit_Code']!='ADNIGO Screening MRI']
print('After', data.shape)

# replace all the NaN diagnosis with the closest diagnosis
data['Diagnosis_nearest_2.0'] = data['Diagnosis_nearest_2.0'].fillna(method='ffill')
print('Diagnosis Before')

# Only for the subjects at AIBL Study, replace the PTID with AIBL+PTID
data.loc[data['Study']=='AIBL', 'PTID'] = 'aibl' + data.loc[data['Study']=='AIBL', 'PTID'].astype(str)

data.loc[data['Study']=='PENN', 'PTID'] = 'penn' + data.loc[data['Study']=='PENN', 'PTID'].astype(str)

data['Date'] = data['Date'].astype('datetime64[ns]')
print('SUBJECTS::', len(list(data['PTID'].unique())))

studies = data['Study'].unique()
print(studies)

###### Filter out ROWS that have all H_MUSE NANs ######
print('1. Filter NAN MUSE...') # ok
hmuse = list(data.filter(regex='H_MUSE*'))
data = data.dropna(axis=0, subset=hmuse)

print('BLSA Subjects', len(data[data['Study']=='BLSA']['PTID'].unique()))

studies = data['Study'].unique()
print(studies)
print('SUBJECTS::', len(list(data['PTID'].unique())))

unique_diagnosis = list(data['Diagnosis_nearest_2.0'].unique())
subject_list = list(data['PTID'].unique())
dx_mapping = pd.read_csv('../LongGPClustering/DX_Mapping.csv')

print('Diagnosis Before')
print(data['Diagnosis_nearest_2.0'].unique())

# using the dx_mapping file, map the diagnosis to the new diagnosis
old_diagnosis, new_diagnosis = [], []

for i, u in enumerate(unique_diagnosis):
    old_diagnosis.append(u)
    indx = dx_mapping[dx_mapping['Diagnosis']==u].index.values
    if len(indx) == 0:
        new_diagnosis.append(u)
    else:
        new_diagnosis.append(dx_mapping['Class'].iloc[indx[0]])

print('Old Diagnosis', old_diagnosis)
print('New Diagnosis', new_diagnosis)

data['Diagnosis_nearest_2.0'].replace(old_diagnosis, new_diagnosis, inplace=True)

print('Diagnosis After')
print(data['Diagnosis_nearest_2.0'].unique())

## Remove the subjects that have Vascular Dementia, FTD, PD, Lewy Body Dementia, Hydrocephalus, PCA, TBI
data = data[~data['Diagnosis_nearest_2.0'].isin(['Vascular Dementia', 'other', 'FTD', '', 'PD', 'Lewy Body Dementia', 'Hydrocephalus', 'PCA', 'TBI'])]

# Missing Diagnosis just place it as -1 ##
nan_diagnosis_count = data['Diagnosis_nearest_2.0'].isna().sum()
print('NAN Diagnosis', nan_diagnosis_count)
if nan_diagnosis_count: 
    data.loc[data['Diagnosis_nearest_2.0'].isna(), 'Diagnosis_nearest_2.0'] = 'unk'

print('SUBJECTS::', len(list(data['PTID'].unique())))

data['Diagnosis_nearest_2.0'].replace(['CN', 'MCI', 'AD', 'unk','other', 'early MCI', 'dementia'] ,
[0, 1, 2, -1,  -1, 1, 2], inplace=True)
print(data['Diagnosis_nearest_2.0'].unique())
print('Initial SUBJECTS::', len(list(data['PTID'].unique())))

# Keep only subjects that are CN (0) at all timepoints
cn_subjects = data.groupby('PTID')['Diagnosis_nearest_2.0'].apply(lambda x: (x == 0).all())
cn_subjects = cn_subjects[cn_subjects].index.tolist()
data = data[data['PTID'].isin(cn_subjects)]
print('Subjects after keeping CN-only at all timepoints:', len(data['PTID'].unique()))

data['Hypertension'].replace(['Hypertension negative/absent', 'Hypertension positive/present'], [0,1], inplace=True)
data['Hyperlipidemia'].replace(['Hyperlipidemia absent', 'Hyperlipidemia recent/active'], [0,1], inplace=True)
data['Diabetes'].replace(['Diabetes negative/absent', 'Diabetes positive/present'], [0,1], inplace=True)

list(data['Study'].unique())

"""**Data Selection**
1. Keep the Subjects with more than one aquisition
"""
print('Before the Single-Visit Subject Removal')
print('BLSA Subjects', len(data[data['Study']=='BLSA']['PTID'].unique()))

# prompt: remove all the subjects with only one acquisition.
data = data.groupby('PTID').filter(lambda x: x.shape[0] > 1)
print('After the Single-Visit Subject Removal')
print('BLSA Subjects', len(data[data['Study']=='BLSA']['PTID'].unique()))
print('Total subjects after filtering:', len(data['PTID'].unique()))

print('Subjects after removing the single-sampled ones', len(list(data['PTID'].unique())))
studies_with_multiple_acquisitions = []
print("\n=== Studies with Multiple Acquisitions ===")
for study in data['Study'].unique():
    study_data = data[data['Study'] == study]
    total_subjects = study_data['PTID'].nunique()
    subjects_with_multiple = study_data.groupby('PTID').filter(lambda x: x.shape[0] > 1)
    n_multi = subjects_with_multiple['PTID'].nunique()
    if n_multi > 0:
        studies_with_multiple_acquisitions.append(study)
        pct = 100 * n_multi / total_subjects
        print(f"  {study}: {n_multi}/{total_subjects} subjects with multiple acquisitions ({pct:.1f}%)")
    else:
        print(f"  {study}: 0/{total_subjects} subjects with multiple acquisitions (0.0%)")
print(f"Total studies with multiple acquisitions: {len(studies_with_multiple_acquisitions)}")

# Filter data to keep only studies with multiple acquisitions
data = data[data['Study'].isin(studies_with_multiple_acquisitions)]
print("\nStudies kept after filtering:", studies_with_multiple_acquisitions)
print('Total subjects after filtering:', len(data['PTID'].unique()))
print('Total samples after filtering:', len(data))


'''
Calculate the Population Statistics for all available studies
'''
studies = data['Study'].unique()
print(studies)
for s in studies: 

    s_data = data[data['Study'] == s]

    # Calculate the number of subjects
    num_subjects = s_data['PTID'].nunique()

    education_years_median = s_data['Education_Years'].median()
    education_years_std = s_data['Education_Years'].std()
    print('Edu Years Stats', education_years_median, education_years_std)

    apoe_noncarriers = s_data[s_data['APOE4_Alleles'] == 0].shape[0]
    apoe_heterozygous = s_data[s_data['APOE4_Alleles'] == 1].shape[0]
    apoe_homozygous = s_data[s_data['APOE4_Alleles'] == 2].shape[0]

    # Calculate the number of samples per subject
    samples_per_subject = s_data.groupby('PTID').size()
    avg_samples_per_subject = samples_per_subject.mean()
    std_samples_per_subject = samples_per_subject.std()

    # Calculate the total number of samples
    total_samples = s_data.shape[0]

    # Calculate the race instances
    # Count the number of different races
    number_of_races = s_data['Race'].nunique()

    print("Number of different races:", number_of_races)

    # Calculate the mean age and standard deviation
    mean_age = s_data['Age'].mean()
    std_age = s_data['Age'].std()

    min_age = s_data['Age'].min()
    max_age = s_data['Age'].max()

    # Calculate the percentage of male subjects
    percentage_male = (s_data[s_data['Sex'] == 'M' ].shape[0] / total_samples) * 100

    # Assuming the diagnoses are given in a column named 'Diagnosis'
    # Calculate the percentage of each diagnosis category
    diagnosis_counts = s_data['Diagnosis_nearest_2.0'].value_counts(normalize=True) * 100

    # race_counts = s_data['Race'].value_counts(normalize=True) * 100

    education_years_mean = s_data['Education_Years'].mean()
    education_years_std = s_data['Education_Years'].std()

    # Results
    results = {
        'num_subjects': num_subjects,
        'avg_samples_per_subject': avg_samples_per_subject,
        'std_samples_per_subject': std_samples_per_subject,
        'total_samples': total_samples,
        'mean_age': mean_age,
        'std_age': std_age,
        'min_age': min_age, 
        'max_age': max_age, 
        'percentage_male': percentage_male,
        'diagnosis_counts': diagnosis_counts.to_dict(), 
        'apoe_counts': {'noncarriers': apoe_noncarriers, 'heterozygous': apoe_heterozygous, 'homoyzgous': apoe_homozygous},
        'education_years_mean': education_years_mean,
        'education_years_std': education_years_std
    }
    print('STUDY::', s)
    print(results) 


# What is the average number of visits per subject in the datasets ?
print("=== Average Number of Visits per Subject Analysis ===")

# Calculate visits per subject for all studies combined
visits_per_subject = data.groupby('PTID').size()
avg_visits_all = visits_per_subject.mean()
std_visits_all = visits_per_subject.std()
median_visits_all = visits_per_subject.median()
min_visits_all = visits_per_subject.min()
max_visits_all = visits_per_subject.max()

print(f"All Studies Combined:")
print(f"  Average visits per subject: {avg_visits_all:.2f} ± {std_visits_all:.2f}")
print(f"  Median visits per subject: {median_visits_all:.1f}")
print(f"  Range: {min_visits_all} - {max_visits_all} visits")
print(f"  Total subjects: {len(visits_per_subject)}")
print(f"  Total visits: {visits_per_subject.sum()}")

# Calculate visits per subject for each study
print(f"\nPer Study Breakdown:")
for study in data['Study'].unique():
    study_data = data[data['Study'] == study]
    study_visits = study_data.groupby('PTID').size()
    
    avg_visits = study_visits.mean()
    std_visits = study_visits.std()
    median_visits = study_visits.median()
    min_visits = study_visits.min()
    max_visits = study_visits.max()
    
    print(f"  {study}:")
    print(f"    Average visits per subject: {avg_visits:.2f} ± {std_visits:.2f}")
    print(f"    Median visits per subject: {median_visits:.1f}")
    print(f"    Range: {min_visits} - {max_visits} visits")
    print(f"    Total subjects: {len(study_visits)}")
    print(f"    Total visits: {study_visits.sum()}")


print(f"\nSummary Statistics:")
print(f"  Overall average visits per subject: {avg_visits_all:.2f} ± {std_visits_all:.2f}")
print(f"  Study with most visits per subject: {max([(study, data[data['Study']==study].groupby('PTID').size().mean()) for study in data['Study'].unique()], key=lambda x: x[1])[0]}")
print(f"  Study with least visits per subject: {min([(study, data[data['Study']==study].groupby('PTID').size().mean()) for study in data['Study'].unique()], key=lambda x: x[1])[0]}")



"""**Data Preprocessing**
1. Fix the Delta Baseline.The first aquisition should always have Delta Baseline 0 and then create the Time column which is the Time from Baseline expressed in months
"""

def delta_baseline_fix(data):
    for pt in list(data['PTID'].unique()):
        # print(pt)
        # Identifying indices where the current patient's data is located
        pt_indices = data[data['PTID'] == pt].index
        # Calculating the baseline to subtract
        base = data.loc[pt_indices[0], 'Delta_Baseline']

        # print('Delta Baseline Before', data.loc[pt_indices, 'Delta_Baseline'].tolist())

        if base != 0:  # Only adjust if the base is not already 0
           
            print(pt, data.loc[pt_indices[0], 'Study'])
            print('Delta Baseline Before', data.loc[pt_indices, 'Delta_Baseline'].tolist())

            # Subtracting the base from Delta_Baseline for all entries of the current patient
            data.loc[pt_indices, 'Delta_Baseline'] -= base

            print('After', data.loc[pt_indices, 'Delta_Baseline'].tolist())

    return data


data = delta_baseline_fix(data)

print('Subjects')
print(len(list(data['PTID'].unique())))

# prompt: verify that for every subject the Delta_Baseline on the first acquisition is zero
for pt in list(data['PTID'].unique()):
    # print('ID', pt)
    pt_data = data[data['PTID'] == pt]
    # print(pt_data.iloc[0]['Delta_Baseline'])
    
    # Remove any subject that has any value at Delta_Baseline to be negative 
    if pt_data.iloc[0]['Delta_Baseline'] != 0.0:
        print('Error')
    
# prompt: create a new column that is the Delta_Baseline divided by 30 and keep only the integer part and round to the greater integer
import numpy as np
data['Time'] = np.ceil((data['Delta_Baseline'] / 30)).astype(int)

# prompt: calculate the time intervals among consecutive aquisitions within a subject and then plot the distribution of time intervals. Also calculate the mean and the std

import matplotlib.pyplot as plt
import numpy as np
# Calculate the time intervals between consecutive acquisitions
time_intervals, acquisitions = [], [] 
for subject_id in data['PTID'].unique():
  subject_data = data[data['PTID'] == subject_id].sort_values(by='Time')
  acquisitions.append(subject_data.shape[0])
  for i in range(1, len(subject_data)):
    time_interval = subject_data['Time'].iloc[i] - subject_data['Time'].iloc[i-1]
    time_intervals.append(time_interval)

# Plot the distribution of time intervals
plt.hist(time_intervals, bins=20)
plt.xlabel('Time interval (months)')
plt.ylabel('Frequency')
plt.title('Distribution of time intervals between consecutive acquisitions')
plt.show()

# Calculate the mean and standard deviation of the time intervals
mean_interval = np.mean(time_intervals)
std_interval = np.std(time_intervals)

print(f"Mean time interval: {mean_interval}")
print(f"Standard deviation of time interval: {std_interval}")

# prompt: within a subject remove the duplicate entries in Time column
print(data.shape)
print(len(list(data['PTID'].unique())))
data = data.groupby(['PTID', 'Time']).agg(lambda x: x.iloc[0]).reset_index()
print(data.shape)
print(len(list(data['PTID'].unique())))

data_unnorm = data.copy()

"""**Z-scoring on MUSE**"""
def data_normalization_all(data):
    # print('Extract Statistics from', data.shape)
    mean_list, std_list = [],[]
    mean_list = data.mean(axis=0).tolist()
    std_list = data.std(axis=0).tolist()
    for m in mean_list:
        print(m)
    return mean_list, std_list

print('MUSE Data Normalization...')
subjects_df_hmuse = data.filter(regex='H_MUSE*')
mean, std = data_normalization_all(data=subjects_df_hmuse)
print('Unnormalized MUSE ROIS', subjects_df_hmuse.shape)

## Store the mean and std 
with open(data_dir + '145_harmonized_allstudies_mean_std_hmuse.pkl', 'wb') as file:
    pickle.dump({'mean': mean, 'std': std}, file)   


for i, c in enumerate(list(subjects_df_hmuse)):
    m,s= mean[i], std[i]
    subjects_df_hmuse[c] = (subjects_df_hmuse[c] - m)/s
print('Normalized MUSE ROIS', subjects_df_hmuse.shape)
for h in hmuse:
    # print(h)
    data[h] = subjects_df_hmuse[h]

"""**Verify the z-scoring**"""
print(data['H_MUSE_Volume_4'].head(10))

# prompt: create an additional column named Baseline_Age that is the Age of the first acquisition in every subject
data['Baseline_Age'] = data.groupby('PTID')['Age'].transform('min')

# # prompt: for all the features that start from H_MUSE_* create an additional column named Baseline_H_MUSE_*  that contains the initial H_MUSE_ value for every subjet
hmuse_cols = [col for col in data.columns if col.startswith('H_MUSE_')]

for col in hmuse_cols:
    data['Baseline_' + col] = data.groupby('PTID')[col].transform('first')

for col in ['SPARE_AD', 'SPARE_BA', 'Diagnosis_nearest_2.0']: 
    data['Baseline_' + col] = data.groupby('PTID')[col].transform('first')


'''
Calculate the Population Statistics for all available studies
'''
studies = data['Study'].unique()
print(studies)
for s in studies: 

    s_data = data[data['Study'] == s]
    # s_data = data 
    # Calculate the number of subjects
    num_subjects = s_data['PTID'].nunique()

    # Calculate the number of samples per subject
    samples_per_subject = s_data.groupby('PTID').size()
    avg_samples_per_subject = samples_per_subject.mean()
    std_samples_per_subject = samples_per_subject.std()

    # Calculate the total number of samples
    total_samples = s_data.shape[0]

    # Calculate the race instances
    # Count the number of different races
    number_of_races = s_data['Race'].nunique()

    print("Number of different races:", number_of_races)

    # Calculate the mean age and standard deviation
    mean_age = s_data['Age'].mean()
    std_age = s_data['Age'].std()

    min_age = s_data['Age'].min()
    max_age = s_data['Age'].max()

    # Calculate the percentage of male subjects
    percentage_male = (s_data[s_data['Sex'] == 'M' ].shape[0] / total_samples) * 100

    # Assuming the diagnoses are given in a column named 'Diagnosis'
    # Calculate the percentage of each diagnosis category
    diagnosis_counts = s_data['Diagnosis'].value_counts(normalize=True) * 100

    race_counts = s_data['Race'].value_counts(normalize=True) * 100

    education_years_mean = s_data['Education_Years'].mean()
    education_years_std = s_data['Education_Years'].std()

    # Results
    results = {
        'num_subjects': num_subjects,
        'avg_samples_per_subject': avg_samples_per_subject,
        'std_samples_per_subject': std_samples_per_subject,
        'total_samples': total_samples,
        'mean_age': mean_age,
        'std_age': std_age,
        'min_age': min_age, 
        'max_age': max_age, 
        'percentage_male': percentage_male,
        'diagnosis_counts': diagnosis_counts.to_dict(), 
        'race_counts': race_counts.to_dict(),
        'education_years_mean': education_years_mean,
        'education_years_std': education_years_std
    }
    print('STUDY::', s)
    print(results)

### Plot the historgram of Age range for the BLSA Presentation 
data= data[data['Time']>=0]
# Increasing font sizes for the presentation
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fontsize = 18
# Histogram for Age
axes[0].hist(data['Age'], bins=20, color='#377eb8', edgecolor='black', alpha=0.7)  # Color Universal Design blue
axes[0].set_xlabel('Age', fontsize=fontsize)
axes[0].set_ylabel('Frequency', fontsize=fontsize)
axes[0].set_title('Distribution of Age', fontsize=fontsize)
axes[0].tick_params(axis='both', which='major', labelsize=fontsize-2)  # Increasing tick size
axes[0].spines['right'].set_visible(False)
axes[0].spines['top'].set_visible(False)

# Histogram for Time from Baseline
axes[1].hist(data['Time'], bins=20, color='#e41a1c', edgecolor='black', alpha=0.7)  # Color Universal Design red
axes[1].set_xlabel('Time from Baseline', fontsize=fontsize)
axes[1].set_ylabel('Frequency', fontsize=fontsize)
axes[1].set_title('Distribution of Time from Baseline', fontsize=fontsize)
axes[1].tick_params(axis='both', which='major', labelsize=fontsize-2)  # Increasing tick size
axes[1].spines['right'].set_visible(False)
axes[1].spines['top'].set_visible(False)


# Applying a tight layout for better spacing
plt.tight_layout()

# Saving the plots with increased font sizes for presentation
plt.savefig('BLSA_Presentation_Age_Time.svg')

"""Processing of Clinical Covariates
1. Normalization of Baseline Age
2. Binarization of Education Years
3. Binarization of Sex
4. Missing Data Indication -1
"""
education_years_median = data['Education_Years'].median()
education_years_std = data['Education_Years'].std()
print('Edu Years Stats', education_years_median, education_years_std)

# BAG: Brain Age Gap = SPARE_BA - Age (computed on raw/unnormalized values)
data['BAG'] = data['SPARE_BA'] - data['Age']
print('BAG column created. Mean BAG:', data['BAG'].mean(), '  STD BAG:', data['BAG'].std())

print('Normalize Age...')
mean_age, std_age = data['Age'].mean() ,data['Age'].std()
data['Age'] = data['Age'].apply(lambda x: (x-mean_age)/std_age)
print('Mean Age', mean_age)
print('STD Age', std_age)

print('Binarize Education Years...')
# mean_ey, std_ey = subjects_df['Education_Years'].mean(),subjects_df['Education_Years'].std()
data['Education_Years'] = data['Education_Years'].apply(lambda x: 1 if x>16 else 0)
data['Education_Years'] = pd.to_numeric(data['Education_Years'])

print('Binarize Sex')
data['Sex'].replace(['M', 'F'], [0,1], inplace=True)

# Substitute Missing Clinical Features with -1
# remove SPARE_BA and SPARE_AD when we infer only the imaging rois
clinical_features = ['Sex', 'PTID', 'Delta_Baseline', 'Time']

mean_spareba, std_spareba = data['SPARE_BA'].mean() ,data['SPARE_BA'].std()
data['SPARE_BA'] = data['SPARE_BA'].apply(lambda x: (x-mean_spareba)/std_spareba)

for cf in clinical_features:
    print(cf)
    data[cf] = data[cf].fillna(-1)

all_subjects = list(data['PTID'].unique())


print('Evaluate Here the Scanner Changes')
all_subjects = ['002_S_5230', '002_S_4447', '011_S_0362'] # '002_S_0413', '003_S_0908','007_S_4272', '002_S_0729', '006_S_4485']  # Subjects in the manuscript.

for s in all_subjects: 
    subject_data = data[data['PTID'] == s]
    print('Subject', s)
    # print(subject_data[['Sex', 'Age', 'Diagnosis_nearest_2.0']])

    print('Scanner', subject_data['MRI_Scanner_Model'])
    # print('Protocol', subject_data['MRI_Protocol'].unique())
    # print('Manufacturer', subject_data['MRI_Manufacturer'].unique())
    # print('Site', subject_data['SITE'].unique())
    # print('Study', subject_data['Study'].unique())


"""**LMM Data**"""

print('Store the LMM data')
print(data.shape)
# keep in data all the columns that start from Baseline and Time and the H_MUSE columns
data = data.filter(regex='Baseline*|Time*|H_MUSE*|PTID|Diagnosis_nearest_2.0|Age|Sex|APOE4_Alleles|Education_Years|SPARE_BA|SPARE_AD|BAG|Delta_Baseline|Study|MRI_Scanner_Model')
print(data.shape)
# then check for the columns that have Nan values
for c in data.columns:
    if data[c].isnull().sum() > 0:
        print(c)
    
# cast all the PTID to string
data['PTID'] = data['PTID'].astype(str)

data.to_csv('LMM_data_allstudies.csv')

print('Total Number of Subjects::', len(list(data['PTID'].unique())))


"""**Save the pickle files**"""
import pickle
clinical_features = ['Sex', 'PTID', 'Delta_Baseline', 'Time']
features = [name for name in data.columns if (name.startswith('H_MUSE_Volume') and int(name[14:])<300)]
features.extend(clinical_features)

# Saving with pickle
with open(data_dir + "features.pkl", "wb") as file:
    pickle.dump(features, file)

target = [ name for name in data.columns if (name.startswith('H_MUSE_Volume') and  int(name[14:])<300)]
# target= ['SPARE_AD', 'SPARE_BA']
all_subjects = list(data['PTID'].unique())

print('Total Number of Subjects::', len(all_subjects)) 
sys.exit(0)
samples, subject_data, num_samples, list_of_subjects, list_of_subject_ids, cnt, covs, longitudinal_covariates = create_baseline_temporal_dataset(subjects=all_subjects, dataframe=data, dataframeunnorm=data_unnorm,  target=target, features=features, hmuse=hmuse,  genomic=0, followup=0, derivedroi='all', visualize=False)

# samples, subject_data, num_samples, list_of_subjects, list_of_subject_ids, cnt = create_n_acquisition_temporal_dataset(n=3, subjects=all_subjects, dataframe=data, dataframeunnorm=data_unnorm,  target=target, features=features, hmuse=hmuse,  genomic=0, followup=0, derivedroi='all', visualize=False)

samples_df = pd.DataFrame(data=samples)
covariates_df = pd.DataFrame(data=covs)
longitudinal_covariates_df = pd.DataFrame(data=longitudinal_covariates)
longitudinal_covariates_df.to_csv(data_dir + 'longitudinal_covariates_subjectsamples_longclean_hmuse_convs_'+'allstudies' +'.csv')
# samples_df.to_csv(data_dir + 'subjectsamples_longclean_hmuse_'+'allstudies'+'.csv')
# covariates_df.to_csv(data_dir + 'covariates_subjectsamples_longclean_'+'allstudies'+ '.csv')

sys.exit(0)
print('Subjects', len(longitudinal_covariates_df['PTID'].unique()))

"""**5 Fold Cross Validation**"""
from sklearn.model_selection import KFold

#### Check of duplicates in  list_of_subject_ids #####
print('Check for Duplicates...')
assert len(list(set(list_of_subject_ids))) == len(list_of_subject_ids)

print('Data for K-FOLD Splitting...', len(list_of_subject_ids))
## CREATE 5 FOLDS
kf = KFold(n_splits=5, random_state=None, shuffle=True)
for i, (train_index, test_index) in enumerate(kf.split(list_of_subject_ids)):
    print('Fold::', i)

    train_subject_ids = []
    test_subject_ids = []
    print("TRAIN:", len(train_index), "TEST:", len(test_index))

    for tr in train_index:
        train_subject_ids.append(list_of_subject_ids[tr])

    for te in test_index:
        test_subject_ids.append(list_of_subject_ids[te])

    for t in test_subject_ids:
        if t in train_subject_ids:
            print('There is a leak!!!!')
            sys.exit(0)

    print('Train IDs', len(train_subject_ids))
    print('Test IDs', len(test_subject_ids))

    with open( data_dir  + 'train_subject_allstudies_ids_hmuse'  + str(i) + '.pkl', 'wb') as handle:
        pickle.dump(train_subject_ids, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open( data_dir +  'test_subject_allstudies_ids_hmuse' +  str(i) + '.pkl', 'wb') as handle:
        pickle.dump(test_subject_ids, handle, protocol=pickle.HIGHEST_PROTOCOL)


sys.exit(0)


# Calculating median and standard deviation for Age and Education Years
age_median = data['Baseline_Age'].median()
age_std = data['Baseline_Age'].std()
education_years_median = data['Education_Years'].median()
education_years_std = data['Education_Years'].std()

print(f"Age Median: {age_median}, Age Standard Deviation: {age_std:.2f}")
print(f"Education Years Median: {education_years_median}, Education Years Standard Deviation: {education_years_std:.2f}")

# Calculating percentages for Sex
sex_counts = data['Sex'].value_counts(normalize=True) * 100

# Calculating percentages for Clinical Status at Baseline
clinical_status_counts = data['Diagnosis_nearest_2.0'].value_counts(normalize=True) * 100

# Calculating percentages for Race
race_counts = data['Race'].value_counts(normalize=True) * 100

# Calculating percentages for APOE4 Alleles
apoe4_counts = data['APOE4_Alleles'].value_counts(normalize=True) * 100

print(f"Sex Percentages:\n{sex_counts}")
print(f"Clinical Status at Baseline Percentages:\n{clinical_status_counts}")
print(f"Race Percentages:\n{race_counts}")
print(f"APOE4 Alleles Percentages:\n{apoe4_counts}")

print('Total Samples', data.shape)
print('Total Subjects', len(list(data['PTID'].unique())))
print('Mean/Std of Acquisitions', np.mean(acquisitions), np.std(acquisitions))

# prompt: Visualize the Baseline Age Distribution for every study  in a plotly boxplot
import plotly.express as px
# Creating the boxplot with specified customizations
fig = px.box(data, x="Study", y="Baseline_Age", title="OASIS Baseline Age Distribution",color_discrete_sequence=['green'])  # This sets all boxplots to red

# Centering the title and setting layout properties
fig.update_layout(
    title={
        'text': "Baseline Age Distribution by Study",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
     width=500, # Fixed width
    height=500, # Fixed height
    yaxis_title='Baseline Age',
    xaxis_title='Study',
    plot_bgcolor='white', # White background
    font=dict(size=15) # Setting font size for all text in the plot
)

# Highlighting a specific boxplot with a different color
# Assuming you want to change the color of the first boxplot
# fig.update_traces(selector=dict(type='box', x=8), marker_color='#FFA07A')  # Adjust 'x' based on the position
fig.write_image(data_dir + 'baseline_age_distribution_of_long_studies.svg')



# prompt: Visualize the Time distribution for all studies

# fig = px.histogram(data, x="Time", nbins=len(data['Time'].unique()), title="Time Distribution")
fig = go.Figure(data=[go.Histogram(x=data['Time'], marker_color='orange')])

# Centering the title and setting layout properties
fig.update_layout(
    title={
        'text': "Time Distribution",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
     width=800, # Fixed width
    height=600, # Fixed height

    plot_bgcolor='white', # White background
    font=dict(size=17) # Setting font size for all text in the plot
)

# prompt: plot the distribution of H_MUSE_Volume_4

fig = px.histogram(data, x="H_MUSE_Volume_4", nbins=100, title="Distribution of H_MUSE_Volume_4")

# Centering the title and setting layout properties
fig.update_layout(
    title={
        'text': "Distribution of H_MUSE_Volume_4",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
     width=800, # Fixed width
    height=600, # Fixed height

    plot_bgcolor='white', # White background
    font=dict(size=17) # Setting font size for all text in the plot
)

# prompt: normalize all the H_MUSE_Volume*

for col in hmuse_cols:
  data['Baseline_' + col] = (data['Baseline_' + col] - data['Baseline_' + col].min()) / (data['Baseline_' + col].max() - data['Baseline_' + col].min())

# prompt: normalize the Age

data['Baseline_Age'] = (data['Baseline_Age'] - data['Baseline_Age'].min()) / (data['Baseline_Age'].max() - data['Baseline_Age'].min())

# prompt: plot the distribution of H_MUSE_Volume_4

fig = px.histogram(data, x="H_MUSE_Volume_4", nbins=100, title="Distribution of Normalized H_MUSE_Volume_4")

# Centering the title and setting layout properties
fig.update_layout(
    title={
        'text': "Distribution of Normalized H_MUSE_Volume_4",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
     width=800, # Fixed width
    height=600, # Fixed height

    plot_bgcolor='white', # White background
    font=dict(size=17) # Setting font size for all text in the plot
)
fig.show()
