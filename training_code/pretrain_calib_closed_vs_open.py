"""
-----------------------------------------------------------------------------------------
pretrain_calib_closed_vs_open.py
-----------------------------------------------------------------------------------------
Goal of the script:
Run deepmreye on fmriprep output 
-----------------------------------------------------------------------------------------
Input(s):
-----------------------------------------------------------------------------------------
Output(s):
Dictionary with all predictions
TSV with gaze position
-----------------------------------------------------------------------------------------
To run:
1. cd to function
>> cd /home/mszinte/projects/gaze_prf/analysis_code/deepmreye
2. pretrain_calib_closed_vs_open.py [main directory] [project name] [group]
-----------------------------------------------------------------------------------------
Exemple:
cd ~/projects/deepmreye/training_code
python pretrain_calib_closed_vs_open.py /scratch/mszinte/data deepmreye 327 
-----------------------------------------------------------------------------------------
"""
# Import modules and add library to path
import sys
import json
import os
import pickle
import glob
import warnings
import numpy as np
import pandas as pd

# DeepMReye imports
from deepmreye import analyse, preprocess, train
from deepmreye.util import data_generator, model_opts 
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

sys.path.append("{}/utils".format(os.getcwd()))
from training_utils import detrending


# Define paths to functional data
main_dir = f"{sys.argv[1]}/{sys.argv[2]}/derivatives/int_deepmreye/deepmreye_calib" 
func_dir = f"{main_dir}/func"  
model_dir = f"{main_dir}/model/"
model_file = f"{model_dir}dataset6_openclosed.h5"
pp_dir = f"{main_dir}/pp_data_closed_vs_open_calibpretrained/"
mask_dir = f"{main_dir}/mask"
report_dir = f"{main_dir}/report"
pred_dir = f"{main_dir}/pred"

# Make directories
os.makedirs(pp_dir, exist_ok=True)
os.makedirs(mask_dir, exist_ok=True)
os.makedirs(report_dir, exist_ok=True)
os.makedirs(pred_dir, exist_ok=True)

# Define settings
with open('settings.json') as f:
    json_s = f.read()
    settings = json.loads(json_s)

subjects = settings['subjects']
ses = settings["session"]
num_run = settings["num_run"]
subTRs = settings['subTRs']
TR = settings['TR']

noise_std = settings["noise_std"]

opts = model_opts.get_opts()

# Define environment cuda
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # stop warning
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # use 4 gpu cards 

# Preload masks to save time within subject loop
(eyemask_small, eyemask_big, dme_template, mask, x_edges, y_edges, z_edges) = preprocess.get_masks()

for subject in subjects:
    print(f"Running {subject}")
    func_sub_dir = f"{func_dir}/{subject}"
    mask_sub_dir = f"{mask_dir}/{subject}"
    func_files = glob.glob(f"{func_sub_dir}/*.nii.gz")

    
    print(mask_sub_dir)

    for func_file in func_files:
        mask_sub_dir_check = os.listdir(mask_sub_dir) 
        print(mask_sub_dir_check)
        
        if len(mask_sub_dir_check) != 0: 
            print(f"Mask for {subject} exists. Continuing")
        else:
            preprocess.run_participant(fp_func=func_file, 
                                       dme_template=dme_template, 
                                       eyemask_big=eyemask_big, 
                                       eyemask_small=eyemask_small,
                                       x_edges=x_edges, y_edges=y_edges, z_edges=z_edges,
                                       transforms=['Affine', 'Affine', 'SyNAggro'])

# Pre-process data
for subject in subjects:    
    subject_data = []
    subject_labels = [] 
    subject_ids = []

    
    for run in range(num_run): 
         # Identify mask and label files
            mask_filename = f"mask_{subject}_ses-02_task-DeepMReyeCalib_run-0{run + 1}_space-T1w_desc-preproc_bold.p"
            

            mask_path = os.path.join(mask_dir, subject, mask_filename)
            

            if not os.path.exists(mask_path):
                print(f"WARNING --- Mask file {mask_filename} not found for Subject {subject} Run {run + 1}.")
                continue

            # Load mask and normalize it
            this_mask = pickle.load(open(mask_path, "rb"))
            this_mask = preprocess.normalize_img(this_mask)

            # No labels bc pretrained
            this_label = this_label = np.zeros(
                (this_mask.shape[3], 10, 2)
            )


            # Check if each functional image has a corresponding label
            if this_mask.shape[3] != this_label.shape[0]:
                print(
                    f"WARNING --- Skipping Subject {subject} Run {run + 1} "
                    f"--- Wrong alignment (Mask {this_mask.shape} - Label {this_label.shape})."
                )
                continue

            # Store across runs
            subject_data.append(this_mask)  # adds data per run to list
            subject_labels.append(this_label)
            subject_ids.append(([subject] * this_label.shape[0],
                                    [run + 1] * this_label.shape[0]))
            
    
    # Save participant file
    preprocess.save_data(participant=f"{subject}_DeepMReyeCalib_no_label",
                            participant_data=subject_data,
                            participant_labels=subject_labels,
                            participant_ids=subject_ids,
                            processed_data=pp_dir,
                            center_labels=False)

try:
    os.system(f'rm {pp_dir}/.DS_Store')
    print('.DS_Store file deleted successfully.')
except Exception as e:
    print(f'An error occurred: {e}')

# Define paths to dataset
datasets = [
    pp_dir + p for p in os.listdir(pp_dir) if "no_label" in p
]

# Load data from one participant to showcase input/output
X, y = data_generator.get_all_subject_data(datasets[0])
print(f"Input: {X.shape}, Output: {y.shape}")

test_participants = [
    pp_dir + p for p in os.listdir(pp_dir) if "no_label" in p
]
generators = data_generator.create_generators(test_participants,
                                              test_participants)
generators = (*generators, test_participants, test_participants
              )  # Add participant list

# Train and evaluate model
# Get untrained model and load with trained weights
(model, model_inference) = train.train_model(dataset="DeepMRyeCalib_pretrained",
                                             generators=generators,
                                             opts=opts,
                                             return_untrained=True)
model_inference.load_weights(model_file)

# === adapted evaluation for eyes open vs closed schema ===
(evaluation, scores) = dict(), dict()

for idx, subj in enumerate(test_participants):
    # Load all subject data
    X, real_y = data_generator.get_all_subject_data(subj)  # real_y may be zeros if no labels
    pred_y, confidence = model_inference.predict(X, verbose=0, batch_size=16)
    
    # Store predictions in dict
    evaluation[subj] = {
        "pred_y": pred_y,         # shape (n_TRs, 10, 2)
        "confidence": confidence  # optional, if model outputs it
    }
    
    # If real labels exist, compute metrics only on the eyes-closed channel (channel 0, ignore fake channel 1 (saccades))
    if np.any(real_y): 
        nan_indices = np.any(np.isnan(real_y), axis=(1,2))
        real_y_clean = real_y[~nan_indices, ...]
        pred_y_clean = pred_y[~nan_indices, ...]
        
        if real_y_clean.size > 0:
            # Flatten across subTRs for evaluation
            pred_flat = pred_y_clean[..., 0].reshape(-1)
            real_flat = real_y_clean[..., 0].reshape(-1)
            
            # Compute metrics
            r2 = r2_score(real_flat, pred_flat)
            pearson = np.corrcoef(real_flat, pred_flat)[0,1]
            mae = mean_absolute_error(real_flat, pred_flat)
            
            scores[subj] = {"R2": r2, "Pearson": pearson, "MAE": mae}
            
            print(f"{idx+1}/{len(test_participants)} --- {subj} --- Eyes-Closed Predictions")
            print(f"    R2: {r2:.5f} | Pearson: {pearson:.5f} | MAE: {mae:.5f}")
        else:
            print(f"{idx+1}/{len(test_participants)} --- {subj} --- No valid labels")
    else:
        print(f"{idx+1}/{len(test_participants)} --- {subj} --- No labels, saving predictions only")

# Save the evaluation and scores dicts
np.save(f"{pred_dir}/evaluation_dict_closed_vs_open_calibpretrained.npy", evaluation)
np.save(f"{pred_dir}/scores_dict_closed_vs_open_calibpretrained.npy", scores)

# Optional: extract only the eyes-closed channel for easier downstream analysis
eyes_closed_preds = {subj: data['pred_y'][..., 0] for subj, data in evaluation.items()}
np.save(f"{pred_dir}/calib_eyes_closed_predictions.npy", eyes_closed_preds)


   

        
      

