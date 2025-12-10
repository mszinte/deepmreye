"""
-----------------------------------------------------------------------------------------
simulated_labels_calib.py
-----------------------------------------------------------------------------------------
Goal:
Extract expected position for the Guided Fixation and Pursuit Task in the calibration 
experiment, convert to TR format, and save as label. Free-viewing trials are marked as NaNs.
-----------------------------------------------------------------------------------------
Inputs:
sys.argv[1]: main project directory
sys.argv[2]: project name (corresponds to directory)
sys.argv[3]: subject name
sys.argv[4]: task
sys.argv[5]: group of shared data (e.g., 327)
-----------------------------------------------------------------------------------------
Output:
Cleaned time-series data per run
-----------------------------------------------------------------------------------------
To run:
cd /Users/sinakling/disks/meso_H/projects/deepmreye/training_code
python simulated_labels_calib.py /Users/sinakling/disks/meso_shared deepmreye sub-03 DeepMReyeCalib 327
------------------------------------------------------------------------------------------------------------
"""
import pandas as pd
import numpy as np
import os
import json
import sys
import matplotlib.pyplot as plt

# --------------------- Data saving  ------------------------------------------------
def save_preprocessed_data(data, file_path):
    """Save preprocessed data as a compressed TSV file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)  # Ensure directory exists
    data.to_csv(file_path, sep='\t', index=False, compression='gzip')

# --------------------- Load settings and inputs -------------------------------------

def load_settings(settings_file):
    with open(settings_file) as f:
        return json.load(f)

def load_inputs():
    """Load command-line arguments."""
    return sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]

# Load inputs and settings
main_dir, project_dir, subject, task, group = load_inputs()
settings = load_settings("settings.json")

ses = settings['session']
num_trials = settings["num_trials_calib"]

fixations_positions = [
    [-9.0, 9.0], [-4.5, 9.0], [0.0, 9.0], [4.5, 9.0], [9.0, 9.0],
    [-9.0, 4.5], [-4.5, 4.5], [0.0, 4.5], [4.5, 4.5], [9.0, 4.5],
    [-9.0, 0.0], [-4.5, 0.0], [0.0, 0.0], [4.5, 0.0], [9.0, 0.0],
    [-9.0, -4.5], [-4.5, -4.5], [0.0, -4.5], [4.5, -4.5], [9.0, -4.5],
    [-9.0, -9.0], [-4.5, -9.0], [0.0, -9.0], [4.5, -9.0], [9.0, -9.0]
]

# Load event files for each run
dfs_runs = [
    pd.read_csv(f"{main_dir}/{project_dir}/{subject}/ses-02/func/{subject}_ses-02_task-DeepMReyeCalib_run-0{i+1}_events.tsv", sep='\t')
    for i in range(3)
]

# Initialize lists
all_runs_expected_fix_x, all_runs_expected_fix_y = [], []
all_runs_expected_purs_x, all_runs_expected_purs_y = [], []

# --------------------- Extract expected fixation positions ---------------------

for df_run in dfs_runs:
    expected_position_fix = [
        fixations_positions[int(val) - 1] if not pd.isna(val) else np.nan
        for val in df_run['fixation_location']
    ]
    
    all_runs_expected_fix_x.append([pos[0] if isinstance(pos, list) else np.nan for pos in expected_position_fix])
    all_runs_expected_fix_y.append([pos[1] if isinstance(pos, list) else np.nan for pos in expected_position_fix])

# --------------------- Extract expected pursuit positions ---------------------

legend_ang = {i: (i - 1) * 20 for i in range(1, 19)}
legend_amp = {1: 3, 2: 5, 3: 7}

for df_run in dfs_runs:
    pursuit_coord_on_list, pursuit_coord_off_list = [], []
    pursuit_amp_values = df_run['pursuit_amplitude'].fillna(0).map(legend_amp)
    pursuit_angle_values = df_run['pursuit_angle'].fillna(0).map(legend_ang)

    for index, (amp, ang) in enumerate(zip(pursuit_amp_values, pursuit_angle_values)):
        if index < 52 or index > 105:
            pursuit_coord_on = np.array([0, 0])
            pursuit_coord_off = np.array([0, 0])
        elif index == 52:
            pursuit_coord_on = np.array([0, 0])
            pursuit_coord_off = pursuit_coord_on + np.array([amp * np.cos(np.radians(ang)), -amp * np.sin(np.radians(ang))])
        elif index == 105:
            pursuit_coord_on = pursuit_coord_off
            pursuit_coord_off = np.array([0, 0])
        else:
            pursuit_coord_on = pursuit_coord_off
            pursuit_coord_off = pursuit_coord_on + np.array([amp * np.cos(np.radians(ang)), -amp * np.sin(np.radians(ang))])

        pursuit_coord_on_list.append(pursuit_coord_on)
        pursuit_coord_off_list.append(pursuit_coord_off)

    pursuit_coord_on_arr = np.array(pursuit_coord_on_list)
    all_runs_expected_purs_x.append(pursuit_coord_on_arr[:, 0])
    all_runs_expected_purs_y.append(-1 * pursuit_coord_on_arr[:, 1])

# --------------------- Construct & Save Simulated Labels Per Run ---------------------

for run_idx in range(3):
    trials_fv = np.full(30, np.nan)  # Freeviewing as NaNs

    simulated_label_x = np.concatenate((
        np.zeros(4),  # 4 TRs of 0 at the start
        all_runs_expected_fix_x[run_idx][:51],  # 51 fixation trials
        np.zeros(4),  # 4 TRs of 0
        all_runs_expected_purs_x[run_idx][51:111],  # 54 pursuit trials
        trials_fv,  # 30 TRs of NaN
        np.zeros(5)  # 5 TRs of 0 at the end
    ))

    simulated_label_y = np.concatenate((
        np.zeros(4),  # 4 TRs of 0 at the start
        all_runs_expected_fix_y[run_idx][:51],  # 51 fixation trials
        np.zeros(4),  # 4 TRs of 0
        all_runs_expected_purs_y[run_idx][51:111],  # 54 pursuit trials
        trials_fv,  # 30 TRs of NaN
        np.zeros(5)  # 5 TRs of 0 at the end
    ))

    simulated_labels = np.stack((simulated_label_x, 
                                 simulated_label_y), axis=1)
    print(simulated_labels.shape)
    
    simulated_labels = np.tile(simulated_labels[:, np.newaxis, :], (1, 10, 1))  # Duplicate along axis 1

    print(simulated_labels.shape)
                
    plt.plot(simulated_labels[:,:,0])
    plt.show()
    # Save per run
    np.save(f"{main_dir}/{project_dir}/derivatives/int_deepmreye/deepmreye_calib/model/sim_labels/{subject}_run_0{run_idx + 1}_simulated_labels.npy", simulated_labels)


print("Simulated labels saved for all runs.")

