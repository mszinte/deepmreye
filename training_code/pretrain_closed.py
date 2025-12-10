"""
-----------------------------------------------------------------------------------------
pretrain_closed.py
-----------------------------------------------------------------------------------------
Goal of the script:
Run deepmreye on fmriprep output 
-----------------------------------------------------------------------------------------
Input(s):
-----------------------------------------------------------------------------------------
Output(s):
TSV with gaze position
-----------------------------------------------------------------------------------------
To run:
1. cd to function
>> cd /home/mszinte/projects/gaze_prf/analysis_code/deepmreye
2. python deepmreye_analysis.py [main directory] [project name] [group]
-----------------------------------------------------------------------------------------
Exemple:
cd ~/projects/deepmreye/training_code
python pretrain_closed.py /scratch/mszinte/data deepmreye 327 
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

sys.path.append("{}/utils".format(os.getcwd()))
from training_utils import detrending


def adapt_evaluation(participant_evaluation):
    pred_y = participant_evaluation["pred_y"]
    pred_y_median = np.nanmedian(pred_y, axis=1)
    pred_uncertainty = abs(participant_evaluation["euc_pred"])
    pred_uncertainty_median = np.nanmedian(pred_uncertainty, axis=1)
    df_pred_median = pd.DataFrame(
        np.concatenate(
            (pred_y_median, pred_uncertainty_median[..., np.newaxis]), axis=1),
        columns=["X", "Y", "Uncertainty"],
    )
    # With subTR
    subtr_values = np.concatenate((pred_y, pred_uncertainty[..., np.newaxis]),
                                  axis=2)
    index = pd.MultiIndex.from_product(
        [range(subtr_values.shape[0]),
         range(subtr_values.shape[1])],
        names=["TR", "subTR"])
    df_pred_subtr = pd.DataFrame(subtr_values.reshape(-1,
                                                      subtr_values.shape[-1]),
                                 index=index,
                                 columns=["X", "Y", "pred_error"])

    return df_pred_median, df_pred_subtr

# Define paths to functional data
main_dir = f"{sys.argv[1]}/{sys.argv[2]}/derivatives/int_deepmreye/deepmreye_closed" 
func_dir = f"{main_dir}/func"  
model_dir = f"{main_dir}/model/"
model_file = f"{model_dir}datasets_1to5.h5"
#f"{model_dir}modelinference_DeepMReyeCalib_no_interpol.h5"
#f"{model_dir}datasets_1to5.h5"
#f"{model_dir}modelinference_gaze_DeepMReyeClosed.h5"
pp_dir = f"{main_dir}/pp_data_pretrained/"
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

opts = model_opts.get_opts()
opts["train_test_split"] = settings["train_test_split"]  #80/20

# Define environment cuda
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # stop warning
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"  # use 3 gpu cards 



# Preload masks to save time within subject loop
(eyemask_small, eyemask_big, dme_template, mask, x_edges, y_edges, z_edges) = preprocess.get_masks()

for subject in subjects:
    print(f"Running {subject}")
    func_sub_dir = f"{func_dir}/{subject}"
    mask_sub_dir = f"{mask_dir}/{subject}"
    func_files = glob.glob(f"{func_sub_dir}/*.nii.gz")


    for func_file in func_files:
        mask_sub_dir_check = os.listdir(mask_sub_dir) 
        
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
    preprocess.save_data(participant=f"{subject}_DeepMReyeClosed_no_label",
                            participant_data=subject_data,
                            participant_labels=subject_labels,
                            participant_ids=subject_ids,
                            processed_data=pp_dir,
                            center_labels=False)

try:
    os.system(f'rm {pp_dir}.DS_Store')
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
(model, model_inference) = train.train_model(dataset="DeepMRyeClosed_pretrained",
                                             generators=generators,
                                             opts=opts,
                                             return_untrained=True)
model_inference.load_weights(model_file)

(evaluation, scores) = train.evaluate_model(
    dataset="DeepMRyeClosed_PT",
    model=model_inference,
    generators=generators,
    save=False,
    model_path=model_dir,
    model_description="",
    verbose=2,
    percentile_cut=80,
)    


   
# Sava data      
np.save(f"{pred_dir}/evaluation_dict_closed_pretrained.npy",evaluation)
   
np.save(f"{pred_dir}/scores_dict_closed_pretrained.npy",scores)

# Save predictions as tsv
labels_list = os.listdir(pp_dir)

#for label in labels_list: 
#    df_pred_median, df_pred_subtr = adapt_evaluation(evaluation[f'{main_dir}/pp_data_pretrained/{label}'])
#    df_pred_median.to_csv(f'{model_dir}/{os.path.basename(label)[:6]}_pred_pt_median.tsv', sep='\t', index=False)
#    df_pred_subtr.to_csv(f'{model_dir}/{os.path.basename(label)[:6]}_pred_pt_subtr.tsv', sep='\t', index=False)




# Add chmod/chgrp
print(f"Changing files permissions in {sys.argv[1]}/{sys.argv[2]}")
os.system(f"chmod -Rf 771 {sys.argv[1]}/{sys.argv[2]}")
os.system(f"chgrp -Rf {sys.argv[3]} {sys.argv[1]}/{sys.argv[2]}")