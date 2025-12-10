"""
-----------------------------------------------------------------------------------------
train_closed.py
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
2. python deepmreye_analysis.py [main directory] [project name] [group]
-----------------------------------------------------------------------------------------
Exemple:
cd ~/projects/deepmreye/training_code
python train_closed_vs_open.py /scratch/mszinte/data deepmreye 327 
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
main_dir = f"{sys.argv[1]}/{sys.argv[2]}/derivatives/int_deepmreye/deepmreye_closed" 
func_dir = f"{main_dir}/func"  
model_dir = f"{main_dir}/model/"
model_file = f"{model_dir}dataset6_openclosed.h5"
pp_dir = f"{main_dir}/pp_data_closed_vs_open_closed"
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
opts['epochs'] = settings['epochs']
opts['batch_size'] = settings['batch_size']
opts['steps_per_epoch'] = settings['steps_per_epoch']
opts['validation_steps'] = settings['validation_steps']
opts['lr'] = settings['lr']
opts['lr_decay'] = settings['lr_decay']
opts['rotation_y'] = settings['rotation_y']
opts['rotation_x'] = settings['rotation_x']
opts['rotation_z'] = settings['rotation_z']
opts['shift'] = settings['shift']
opts['zoom'] = settings['zoom']
opts['gaussian_noise'] = settings['gaussian_noise']
opts['mc_dropout'] = False
opts['dropout_rate'] = settings['dropout_rate']
opts['loss_euclidean'] = settings['loss_euclidean']
opts['error_weighting'] = settings['error_weighting']
opts['num_fc'] = settings['num_fc']
opts['load_pretrained'] = model_file
opts["train_test_split"] = settings["train_test_split"]  #80/20
#opts['samples_per_epoch'] = 1000
#opts['validation_steps'] = 1000
opts['loss_euclidean'] = 1
opts['loss_confidence'] = 0  #ft only with euclidean loss, ignore confidence 

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
            mask_filename = f"mask_{subject}_ses-02_task-DeepMReyeClosed_run-0{run + 1}_space-T1w_desc-preproc_bold.p"
            label_filename = f"{subject}_run_0{run + 1}_eyesclosed_labels.npy"

            mask_path = os.path.join(mask_dir, subject, mask_filename)
            label_path = os.path.join(model_dir, "eyesclosed_labels", label_filename)
            print(label_path)


            if not os.path.exists(mask_path):
                print(f"WARNING --- Mask file {mask_filename} not found for Subject {subject} Run {run + 1}.")
                continue

            if not os.path.exists(label_path):
                print(f"WARNING --- Label file {label_filename} not found for Subject {subject} Run {run + 1}.")
                continue

            # Load mask and normalize it
            this_mask = pickle.load(open(mask_path, "rb"))
            this_mask = preprocess.normalize_img(this_mask)

            # Load labels
            this_label = np.load(label_path)  # load here proportion labels
            

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
    preprocess.save_data(participant=f"{subject}_DeepMReyeClosed_label_closed",
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

# Train and evaluate model
evaluation, scores = dict(), dict()

# cross validation dataset creation
cv_generators = data_generator.create_cv_generators(pp_dir+'/',
                                                    num_cvs=len(subjects),
                                                    batch_size=opts['batch_size'], 
                                                    augment_list=((opts['rotation_x'], 
                                                                    opts['rotation_y'], 
                                                                    opts['rotation_z']), 
                                                                    opts['shift'], 
                                                                    opts['zoom']), 
                                                    mixed_batches=True)

# Loop across each cross-validation split
for cv_idx, generators in enumerate(cv_generators):    
    
    # pre-load the model
    (preload_model, preload_model_inference) = train.train_model(dataset="pt_gaze_DeepMReyeClosed", 
                                                                    generators=generators, 
                                                                    opts=opts, 
                                                                    return_untrained=True)
    preload_model.load_weights(opts['load_pretrained'])
    
    (_,_,_,_,_,_,full_testing_list,_) = generators
    print(f"CV fold {cv_idx + 1}: Testing subjects: {full_testing_list}")
    
    # train the model
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        (model, model_inference) = train.train_model(dataset="closed_DeepMReyeClosed", 
                                                        generators=generators, 
                                                        opts=opts, 
                                                        use_multiprocessing=False,
                                                        return_untrained=False, 
                                                        verbose=1, 
                                                        save=True,
                                                        model_path=model_dir, 
                                                        models=[preload_model, preload_model_inference])

    print(f"Training done for CV fold {cv_idx + 1}")
  
    (
        training_generator,
        testing_generator,
        single_testing_generators,
        single_testing_names,
        single_training_generators,
        single_training_names,
        full_testing_list,
        full_training_list,
    ) = generators
    
    # Save predictions for each subject in this CV fold
    for idx, subj in enumerate(full_testing_list):
        X, real_y = data_generator.get_all_subject_data(subj)
        
        # Keep original prediction format (2D pipeline)
        (pred_y, euc_pred) = model_inference.predict(X, verbose=0, batch_size=16)
        
        # Store results - each subject appears only once across all CV folds
        evaluation[subj] = {
            "real_y": real_y, 
            "pred_y": pred_y, 
            "euc_pred": euc_pred
        }
        
        # EVALUATE ONLY BLINKS (dimension 0) - ignore saccades (dimension 1)
        if real_y.size > 0:
            # Handle NaN values
            nan_indices = np.any(np.isnan(real_y), axis=(1,2))
            real_y_clean = real_y[~nan_indices, ...]
            pred_y_clean = pred_y[~nan_indices, ...]
            
            if real_y_clean.size > 0:
                # Reshape for evaluation - focus only on dimension 0 (blinks/eye closure)
                pred_y_flat = np.reshape(pred_y_clean, (pred_y_clean.shape[0] * pred_y_clean.shape[1], -1))
                real_y_flat = np.reshape(real_y_clean, (real_y_clean.shape[0] * real_y_clean.shape[1], -1))

                # Only evaluate blinks/eye closure (dimension 0)
                r2_mean_blinks = r2_score(np.mean(real_y_clean[..., 0], axis=1), np.mean(pred_y_clean[..., 0], axis=1))
                r2_blinks = r2_score(real_y_flat[:, 0], pred_y_flat[:, 0])
                pearson_blinks = np.corrcoef(real_y_flat[:, 0], pred_y_flat[:, 0])[0, 1]
                mse_blinks = mean_squared_error(real_y_flat[:, 0], pred_y_flat[:, 0])
                mae_blinks = mean_absolute_error(real_y_flat[:, 0], pred_y_flat[:, 0])

                print(f"CV {cv_idx + 1} - Subject {idx + 1}/{len(full_testing_list)}: {subj} --- EYE CLOSURE PREDICTIONS ---")
                print(f"    R2: {r2_blinks:.5f} | Mean R2: {r2_mean_blinks:.5f} | Pearson: {pearson_blinks:.5f}")
                print(f"    MSE: {mse_blinks:.5f} | MAE: {mae_blinks:.5f}")
            else:
                print(f"CV {cv_idx + 1} - Subject {idx + 1}/{len(full_testing_list)}: {subj} --- No valid labels")
        else:
            print(f"CV {cv_idx + 1} - Subject {idx + 1}/{len(full_testing_list)}: {subj} --- No labels - Pred shape: {pred_y.shape}")

# Save predictions ONCE after all CV folds are complete
np.save(f"{pred_dir}/evaluation_dict_gaze_closed_vs_open_closed.npy", evaluation)
print(f"All predictions saved to {pred_dir}/evaluation_dict_gaze_closed_vs_open_closed.npy")
print(f"Total subjects evaluated: {len(evaluation.keys())}")