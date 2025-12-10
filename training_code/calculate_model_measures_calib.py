"""
-----------------------------------------------------------------------------------------
calculate_model_measures_calib.py
-----------------------------------------------------------------------------------------
Goal of the script:
Calculate and save euclidean error and pearson correlation for comparing 
DeepMReyeCalib models to eye tracking data. 
-----------------------------------------------------------------------------------------
Input(s):
sys.argv[1]: main project directory
sys.argv[2]: project name (correspond to directory)
sys.argv[3]: subject name
sys.argv[4]: task

sys.argv[6]: group of shared data (e.g. 327)
-----------------------------------------------------------------------------------------
Output(s):
tsv.gz file of ee 
-----------------------------------------------------------------------------------------
To run:
cd /Users/sinakling/disks/meso_H/projects/deepmreye/training_code
python calculate_model_measures_calib.py /Users/sinakling/disks/meso_shared deepmreye sub-01 DeepMReyeCalib no_interpol 327
------------------------------------------------------------------------------------------------------------
"""
import pandas as pd
import json
import numpy as np
import re
import matplotlib.pyplot as plt
import glob 
import os
from sklearn.preprocessing import MinMaxScaler
import sys
from statistics import median
from pathlib import Path



sys.path.append("{}/utils".format(os.getcwd()))
from training_utils import adapt_evaluation,euclidean_distance,chunk_and_median

# --------------------- Load settings and inputs -------------------------------------

def load_settings(settings_file):
    with open(settings_file) as f:
        settings = json.load(f)
    return settings

def load_inputs():
    return sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6]


#--------------------- MAIN ----------------------------------------------------------
# Load inputs and settings
main_dir, project_dir, subject, task, model, group = load_inputs()   
with open("settings.json") as f:
    settings = json.load(f)


nTR = settings["num_TR_calib"]
# Create masks for tasks
# Define task TR distributions
interim_TRs = 5  
task_TRs = [50, 54, 30]  
runs = settings["num_run"]
#subjects = settings["subjects"]

# Create the task_labels array
task_labels = np.full(nTR, "interim", dtype=object)  

# Assign tasks to TRs
start_idx = interim_TRs  # Start after initial interim period
for i, task_duration in enumerate(task_TRs, start=1):
    task_labels[start_idx:start_idx + task_duration] = f"task_{i}"
    start_idx += task_duration + interim_TRs  # Move to the next block

# Create a boolean mask for TRs where the desired task is active
tr_mask_task_fix = task_labels == "task_1"
tr_mask_task_pur = task_labels == "task_2"
tr_mask_task_fv = task_labels == "task_3"


# Calculate EE model vs Eyetracking and save
tasks = ["fixation", "pursuit", "freeview", "all"]
evaluation = np.load(f"{main_dir}/{project_dir}/derivatives/int_deepmreye/deepmreye_calib/pred/evaluation_dict_calib_{model}.npy", allow_pickle=True).item() # load correct model

for subject in subjects: 
    for run in range(runs): 
         # load eye tracking 
            eye_data = pd.read_csv(f"/Users/sinakling/disks/meso_shared/deepmreye/derivatives/pp_data/{subject}/eyetracking/{subject}_task-DeepMReyeCalib_run_0{run+1}_eyedata.tsv.gz", compression='gzip', delimiter='\t')
            eye_data = eye_data[['timestamps','x', 'y']].to_numpy()     
            # downsample eye data 
            eye_data_downsampled_x = chunk_and_median(eye_data[:,1])
            eye_data_downsampled_y = chunk_and_median(eye_data[:,2])
            eye_data_downsampled = np.stack((eye_data_downsampled_x, eye_data_downsampled_y), axis=1)

            for task in tasks: 
                subject_data = evaluation[f"/scratch/mszinte/data/deepmreye/derivatives/int_deepmreye/deepmreye_calib/pp_data_{model}/{subject}_DeepMReyeCalib_label_{model}.npz"]
                df_pred_median, df_pred_subtr = adapt_evaluation(subject_data)

                if run == 0: 
                    sub_run_X = np.array(df_pred_median['X'])[:nTR]
                    sub_run_Y = np.array(df_pred_median['Y'])[:nTR]
                    
                    
                    if task == 'fixation':
                        eye_data_downsampled_task_1 = eye_data_downsampled[tr_mask_task_fix] 
                        sub_run_X = sub_run_X[tr_mask_task_fix]   
                        sub_run_Y = sub_run_Y[tr_mask_task_fix] 

                        ee = euclidean_distance(eye_data_downsampled_task_1, sub_run_X, sub_run_Y)
                        eucl_dist_df_1 = pd.DataFrame(ee, columns=['ee'])
                        ee_file_path = f'/Users/sinakling/disks/meso_shared/deepmreye/derivatives/pp_data/{subject}/eyetracking/{subject}_task-DeepMReyeCalib_subtask-{task}_run_0{run+1}_ee_{model}.tsv.gz'
                        eucl_dist_df_1.to_csv(ee_file_path, sep='\t', index=False, compression='gzip')

                    elif task == "pursuit": 
                        eye_data_downsampled_task_2 = eye_data_downsampled[tr_mask_task_pur] 

                        sub_run_X = sub_run_X[tr_mask_task_pur]   
                        sub_run_Y = sub_run_Y[tr_mask_task_pur] 
                        
                        ee = euclidean_distance(eye_data_downsampled_task_2, sub_run_X, sub_run_Y)
                        eucl_dist_df_1 = pd.DataFrame(ee, columns=['ee'])
                        ee_file_path = f'/Users/sinakling/disks/meso_shared/deepmreye/derivatives/pp_data/{subject}/eyetracking/{subject}_task-DeepMReyeCalib_subtask-{task}_run_0{run+1}_ee_{model}.tsv.gz'
                        eucl_dist_df_1.to_csv(ee_file_path, sep='\t', index=False, compression='gzip')

                    elif task == "freeview":
                        eye_data_downsampled_task_3 = eye_data_downsampled[tr_mask_task_fv] 

                        sub_run_X = sub_run_X[tr_mask_task_fv]   
                        sub_run_Y = sub_run_Y[tr_mask_task_fv] 
                        
                        ee = euclidean_distance(eye_data_downsampled_task_3, sub_run_X, sub_run_Y)
                        eucl_dist_df_1 = pd.DataFrame(ee, columns=['ee'])
                        ee_file_path = f'/Users/sinakling/disks/meso_shared/deepmreye/derivatives/pp_data/{subject}/eyetracking/{subject}_task-DeepMReyeCalib_subtask-{task}_run_0{run+1}_ee_{model}.tsv.gz'
                        eucl_dist_df_1.to_csv(ee_file_path, sep='\t', index=False, compression='gzip')
                    
                    elif task == "all":
                        eye_data_downsampled_task_all = eye_data_downsampled 

                        ee_ft = euclidean_distance(eye_data_downsampled_task_all, sub_run_X, sub_run_Y)
                        eucl_dist_df_1 = pd.DataFrame(ee, columns=['ee'])
                        ee_file_path = f'/Users/sinakling/disks/meso_shared/deepmreye/derivatives/pp_data/{subject}/eyetracking/{subject}_task-DeepMReyeCalib_subtask-{task}_run_0{run+1}_ee_{model}.tsv.gz'
                        eucl_dist_df_1.to_csv(ee_file_path, sep='\t', index=False, compression='gzip')
                    

                
                elif run == 1: 
                    sub_run_X = np.array(df_pred_median['X'])[nTR:(nTR+nTR)]
                    sub_run_Y = np.array(df_pred_median['Y'])[nTR:(nTR+nTR)]
                    
                    if task == 'fixation':
                        eye_data_downsampled_task_1 = eye_data_downsampled[tr_mask_task_fix]
                        sub_run_X = sub_run_X[tr_mask_task_fix]   
                        sub_run_Y = sub_run_Y[tr_mask_task_fix] 
                        
                        ee = euclidean_distance(eye_data_downsampled_task_1, sub_run_X, sub_run_Y)
                        eucl_dist_df_2 = pd.DataFrame(ee, columns=['ee'])
                        ee_file_path = f'/Users/sinakling/disks/meso_shared/deepmreye/derivatives/pp_data/{subject}/eyetracking/{subject}_task-DeepMReyeCalib_subtask-{task}_run_0{run+1}_ee_{model}.tsv.gz'
                        eucl_dist_df_2.to_csv(ee_file_path, sep='\t', index=False, compression='gzip')

                    elif task == "pursuit": 
                        eye_data_downsampled_task_2 = eye_data_downsampled[tr_mask_task_pur] 
                        sub_run_X = sub_run_X[tr_mask_task_pur]   
                        sub_run_Y = sub_run_Y[tr_mask_task_pur] 

                        ee = euclidean_distance(eye_data_downsampled_task_2, sub_run_X, sub_run_Y)
                        eucl_dist_df_2 = pd.DataFrame(ee, columns=['ee'])
                        ee_file_path = f'/Users/sinakling/disks/meso_shared/deepmreye/derivatives/pp_data/{subject}/eyetracking/{subject}_task-DeepMReyeCalib_subtask-{task}_run_0{run+1}_ee_{model}.tsv.gz'
                        eucl_dist_df_2.to_csv(ee_file_path, sep='\t', index=False, compression='gzip')

                    elif task == "freeview":
                        eye_data_downsampled_task_3 = eye_data_downsampled[tr_mask_task_fv] 

                        sub_run_X = sub_run_X[tr_mask_task_fv]   
                        sub_run_Y = sub_run_Y[tr_mask_task_fv] 

                        ee = euclidean_distance(eye_data_downsampled_task_3, sub_run_X, sub_run_Y)
                        eucl_dist_df_2 = pd.DataFrame(ee, columns=['ee'])
                        ee_file_path = f'/Users/sinakling/disks/meso_shared/deepmreye/derivatives/pp_data/{subject}/eyetracking/{subject}_task-DeepMReyeCalib_subtask-{task}_run_0{run+1}_ee_{model}.tsv.gz'
                        eucl_dist_df_2.to_csv(ee_file_path, sep='\t', index=False, compression='gzip')
                    
                    elif task == "all":
                        eye_data_downsampled_task_all = eye_data_downsampled 

                        ee = euclidean_distance(eye_data_downsampled_task_all, sub_run_X, sub_run_Y)
                        eucl_dist_df_2 = pd.DataFrame(ee, columns=['ee'])
                        ee_file_path = f'/Users/sinakling/disks/meso_shared/deepmreye/derivatives/pp_data/{subject}/eyetracking/{subject}_task-DeepMReyeCalib_subtask-{task}_run_0{run+1}_ee_{model}.tsv.gz'
                        eucl_dist_df_2.to_csv(ee_file_path, sep='\t', index=False, compression='gzip')
                        
                        
                elif run == 2: 
                    sub_run_X = np.array(df_pred_median['X'])[(nTR+nTR):]
                    sub_run_Y = np.array(df_pred_median['Y'])[(nTR+nTR):]
                    

                    if task == 'fixation':
                        eye_data_downsampled_task_1 = eye_data_downsampled[tr_mask_task_fix] 
                        sub_run_X = sub_run_X[tr_mask_task_fix]   
                        sub_run_Y = sub_run_Y[tr_mask_task_fix] 

                        ee = euclidean_distance(eye_data_downsampled_task_1, sub_run_X, sub_run_Y)
                        eucl_dist_df_3 = pd.DataFrame(ee, columns=['ee'])
                        ee_file_path = f'/Users/sinakling/disks/meso_shared/deepmreye/derivatives/pp_data/{subject}/eyetracking/{subject}_task-DeepMReyeCalib_subtask-{task}_run_0{run+1}_ee_{model}.tsv.gz'
                        eucl_dist_df_3.to_csv(ee_file_path, sep='\t', index=False, compression='gzip')
                        
                    elif task == "pursuit": 
                        eye_data_downsampled_task_2 = eye_data_downsampled[tr_mask_task_pur] 
                        sub_run_X = sub_run_X[tr_mask_task_pur]   
                        sub_run_Y = sub_run_Y[tr_mask_task_pur] 

                        ee = euclidean_distance(eye_data_downsampled_task_2, sub_run_X, sub_run_Y)
                        eucl_dist_df_3 = pd.DataFrame(ee, columns=['ee'])
                        ee_file_path = f'/Users/sinakling/disks/meso_shared/deepmreye/derivatives/pp_data/{subject}/eyetracking/{subject}_task-DeepMReyeCalib_subtask-{task}_run_0{run+1}_ee_{model}.tsv.gz'
                        eucl_dist_df_3.to_csv(ee_file_path, sep='\t', index=False, compression='gzip')
                        

                    elif task == "freeview":
                        eye_data_downsampled_task_3 = eye_data_downsampled[tr_mask_task_fv] 
                        sub_run_X = sub_run_X[tr_mask_task_fv]   
                        sub_run_Y = sub_run_Y[tr_mask_task_fv] 
                        
                        ee = euclidean_distance(eye_data_downsampled_task_3, sub_run_X, sub_run_Y)
                        eucl_dist_df_3 = pd.DataFrame(ee, columns=['ee'])
                        ee_file_path = f'/Users/sinakling/disks/meso_shared/deepmreye/derivatives/pp_data/{subject}/eyetracking/{subject}_task-DeepMReyeCalib_subtask-{task}_run_0{run+1}_ee_{model}.tsv.gz'
                        eucl_dist_df_3.to_csv(ee_file_path, sep='\t', index=False, compression='gzip')
                    
                    elif task == "all":
                        eye_data_downsampled_task_all = eye_data_downsampled 

                        ee = euclidean_distance(eye_data_downsampled_task_all, sub_run_X, sub_run_Y)
                        eucl_dist_df_3 = pd.DataFrame(ee, columns=['ee'])
                        ee_file_path = f'/Users/sinakling/disks/meso_shared/deepmreye/derivatives/pp_data/{subject}/eyetracking/{subject}_task-DeepMReyeCalib_subtask-{task}_run_0{run+1}_ee_{model}.tsv.gz'
                        eucl_dist_df_3.to_csv(ee_file_path, sep='\t', index=False, compression='gzip')



