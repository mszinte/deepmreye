"""
-----------------------------------------------------------------------------------------
calib_ee_sequments.py
-----------------------------------------------------------------------------------------
Goal of the script:
Run deepmreye on fmriprep output 
-----------------------------------------------------------------------------------------
Input(s):
sys.argv[1]: main project directory
sys.argv[2]: project name (correspond to directory)
sys.argv[3]: model name (e.g pretrained)
sys.argv[4]: group (e.g. 327)
-----------------------------------------------------------------------------------------
Output(s):
TSV with gaze position
-----------------------------------------------------------------------------------------
To run:
1. cd to function
   ~/projects/deepmreye/analysis_code/postproc/
2. python calib_ee_sequments.py [main directory] [project name] [model][group]
-----------------------------------------------------------------------------------------
Exemple:
cd ~/projects/deepmreye/analysis_code/postproc
python calib_ee_sequments.py /scratch/mszinte/data deepmreye pretrained 327 
-----------------------------------------------------------------------------------------
"""
# Debug
import ipdb
deb = ipdb.set_trace

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple
import os
import pickle
import sys 

# Personal imports
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, "..", "utils"))
from plot_utils import plot_scatter_ecc_sector_plotly 
from training_utils import chunk_and_median, adapt_evaluation, compute_ee_by_ecc_sector, euclidean_distance, create_task_labels

# Inputs
main_dir = sys.argv[1]
project_dir = sys.argv[2]
model = sys.argv[3]
group = sys.argv[4]


base_path = f"{main_dir}/{project_dir}/derivatives/pp_data"
subjects = ["sub-02","sub-03","sub-04", "sub-05", "sub-06", "sub-07","sub-08", "sub-09","sub-10", "sub-11", "sub-13", "sub-14", "sub-15", "sub-16", "sub-17"]
tasks = ["fixation", "pursuit", "freeview", "all"]

def adapt_scaled_evaluation(subject_data: dict) -> pd.DataFrame:
    """
    Adapts scaled evaluation data to match df_pred_median format.
    scaled_x/y are (462, 1) — 3 runs × 154 TRs, already TR-aligned.
    """
    scaled_x = np.array(subject_data['scaled_x']).squeeze()  # (462,)
    scaled_y = np.array(subject_data['scaled_y']).squeeze()  # (462,)

    return pd.DataFrame({"X": scaled_x, "Y": scaled_y})



# AVERAGED OVER ALL SUBJECTS
run = 0
tasks = ["fixation", "pursuit", "freeview", "all"]

dfs_by_task = {task: [] for task in tasks}

deepmreye_path = f"{main_dir}/{project_dir}/derivatives/int_deepmreye/deepmreye_calibration/pred"

if model == "scaled":
    scaled_path = os.path.join(deepmreye_path, "scaled_prediction_calib.pkl")
    with open(scaled_path, 'rb') as f:
        evaluation = pickle.load(f)
else:
    model_path = os.path.join(deepmreye_path, f"evaluation_dict_calib_{model}.npy")
    evaluation = np.load(model_path, allow_pickle=True).item()

for subject in subjects:
    run_slices = [(0, 154), (154, 308), (308, None)]

    _, tr_mask_task_1, tr_mask_task_2, tr_mask_task_3 = create_task_labels()
    task_bool_mapping = {
        "fixation" : tr_mask_task_1,
        "pursuit"  : tr_mask_task_2,
        "freeview" : tr_mask_task_3,
        "all"      : slice(None)
    }

    eye_data_path = f"{base_path}/{subject}/eyetracking/timeseries/{subject}_task-DeepMReyeCalib_run_0{run+1}_eyedata.tsv.gz"
    try:
        eye_data = pd.read_csv(eye_data_path, compression='gzip', delimiter='\t')
        eye_data = eye_data[['timestamps', 'x', 'y']].to_numpy()
    except FileNotFoundError:
        print(f"Warning: eye data not found for {subject}, skipping")
        continue

    eye_data_downsampled_x = chunk_and_median(eye_data[:, 1])
    eye_data_downsampled_y = chunk_and_median(eye_data[:, 2])
    eye_data_downsampled = np.stack((eye_data_downsampled_x, eye_data_downsampled_y), axis=1)

    # load predictions — branched cleanly by model type
    if model == "scaled":
        if subject not in evaluation:
            print(f"Warning: {subject} not found in scaled evaluation, skipping")
            continue
        df_pred_median = adapt_scaled_evaluation(evaluation[subject])

    else:
        if model == "pretrained":
            subject_key = f"/scratch/mszinte/data/deepmreye/derivatives/int_deepmreye/deepmreye_calib/pp_data_{model}/{subject}_DeepMReyeCalib_no_label.npz"
        else:
            subject_key = f"/scratch/mszinte/data/deepmreye/derivatives/int_deepmreye/deepmreye_calib/pp_data_{model}/{subject}_DeepMReyeCalib_label_{model}.npz"

        if subject_key not in evaluation:
            print(f"Warning: {subject_key} not found in evaluation, skipping")
            continue

        subject_data = evaluation[subject_key]
        df_pred_median, _ = adapt_evaluation(subject_data)

    sub_run_X = np.array(df_pred_median['X'])[run_slices[0][0]:run_slices[0][1]]
    sub_run_Y = np.array(df_pred_median['Y'])[run_slices[0][0]:run_slices[0][1]]

    for task in tasks:
        task_bool = task_bool_mapping[task]
        eye_data_downsampled_task = eye_data_downsampled[task_bool]
        sub_run_X_task = sub_run_X[task_bool]
        sub_run_Y_task = sub_run_Y[task_bool]

        if model == "pretrained":
            sub_run_Y_task = -1 * sub_run_Y_task

        df_both = compute_ee_by_ecc_sector(
            eye_data_downsampled_task, sub_run_X_task, sub_run_Y_task,
            ecc_bins   = [0, 3.375, 6.75, 10.125, 12.75],
            ecc_labels = ["foveal", "parafoveal", "peripheral", "far peripheral"],
        )
        df_both["subject"] = subject
        dfs_by_task[task].append(df_both)


def average_ee_across_subjects(dfs: list) -> pd.DataFrame:
    """
    Average mean_ee across subjects per ecc_bin × sector cell.
    Returns mean, std, subject count, and total sample count per cell.
    """
    combined = pd.concat(dfs, ignore_index=True)
    agg = (
        combined
        .groupby(["ecc_bin", "sector"], sort=False)
        .agg(
            mean_ee    = ("mean_ee",   "mean"),
            std_ee     = ("mean_ee",   "std"),
            n_subjects = ("mean_ee",   "count"),
            n_samples  = ("n_samples", "sum"),
        )
        .reset_index()
    )
    return agg


for task in tasks:
    if not dfs_by_task[task]:
        print(f"No data collected for task {task}, skipping")
        continue

    df_avg = average_ee_across_subjects(dfs_by_task[task])
    print(f"\n{task} — group average (n={len(dfs_by_task[task])} subjects):")
    print(df_avg)

    fig = plot_scatter_ecc_sector_plotly(
    df_avg,
    bins    = [0, 3.375, 6.75, 10.125, 12.75],
    subject = f"group (n={len(dfs_by_task[task])})",
    task    = task,
    model   = model,
    save_path = f"{main_dir}/{project_dir}/derivatives/int_deepmreye/deepmreye_calibration/figures/group/group_task-{task}_model-{model}_ecc_sector_ee.pdf"
    )
