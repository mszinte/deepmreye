"""
-----------------------------------------------------------------------------------------
eyetracking_preproc.py
-----------------------------------------------------------------------------------------
Goal of the script:
Preprocess BIDS formatted eyetracking data for creating training labels for 
fine tuning DeepMReye
- blinks are removed by excluding samples at which the pupil was lost entirely, 
excluding data before and after each occurrence. 
- removing slow signal drift by linear detrending
- median-centering of the gaze position time series (X and Y) 
(assumes that the median gaze position corresponds to central fixation)
- smoothing using a 50-ms running average
- downsampling to TR for deepmreye labels
-----------------------------------------------------------------------------------------
Input(s):
sys.argv[1]: main project directory
sys.argv[2]: project name (correspond to directory)
sys.argv[3]: subject name
sys.argv[4]: task
sys.argv[5]: type
sys.argv[6]: group of shared data (e.g. 327)
-----------------------------------------------------------------------------------------
Output(s):
Cleaned timeseries data per run 
-----------------------------------------------------------------------------------------
To run:
cd /Users/sinakling/disks/meso_H/projects/deepmreye/training_code
python eyetracking_preproc.py /Users/sinakling/disks/meso_shared deepmreye sub-01 DeepMReyeCalib labels 327
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
from scipy.signal import detrend, resample


sys.path.append("{}/utils".format(os.getcwd()))
from eyetrack_utils import load_event_files, extract_data, blinkrm_pupil_off, \
    blinkrm_pupil_off_smooth, interpol_nans, detrending, downsample_to_targetrate, \
    moving_average_smoothing, gaussian_smoothing, extract_eye_data_and_triggers, convert_to_dva, \
    segment_and_filter_eyetracking, extract_eye_data_and_triggers_debugging

# --------------------- Data saving  ------------------------------------------------
def save_preprocessed_data(data, file_path):
    data.to_csv(file_path, sep='\t', index=False, compression='gzip')

# --------------------- Load settings and inputs -------------------------------------

def load_settings(settings_file):
    with open(settings_file) as f:
        settings = json.load(f)
    return settings

def load_inputs():
    return sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6]

def ensure_save_dir_labels(base_dir):
    save_dir = f"{base_dir}/model/labels"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return save_dir

def ensure_save_dir_et(base_dir, subject):
    save_dir = f"{base_dir}/pp_data/{subject}/eyetracking/timeseries"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return save_dir

def ensure_design_dir(base_dir, subject):
    save_dir = f"{base_dir}/exp_design"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return save_dir

def load_events(main_dir, subject, ses, task): 
    data_events = load_event_files(main_dir, subject, ses, task)
    return data_events 

# --------------------- Data Extraction ------------------------------------------------

def extract_event_and_physio_data(main_dir, project_dir, subject, task, ses, num_run, eye):
    df_event_runs = extract_data(main_dir, project_dir, subject, task, ses, num_run, eye, file_type="physioevents")
    df_data_runs = extract_data(main_dir, project_dir, subject, task, ses, num_run, eye, file_type="physio")
    return df_event_runs, df_data_runs

# --------------------- Preprocessing Methods -----------------------------------------
# Options in behaviour_settings
def remove_blinks(data, method, sampling_rate):
    if method == 'pupil_off':
        return blinkrm_pupil_off(data, sampling_rate)
    elif method == 'pupil_off_smooth':
        return blinkrm_pupil_off_smooth(data, sampling_rate)
    else:
        print("No blink removal method specified")
        return data

def drift_correction(data, method, fixation_periods):
    if method == "linear":
        return detrend(data, type='linear')
    elif method == 'median':
        fixation_median = median([item for row in fixation_periods for item in row])
        return np.array([elem - fixation_median for elem in data])
    else:
        print("No drift correction method specified")
        return data

def interpolate_nans(data):
    return interpol_nans(data)

def normalize_data(data):
    print('- normalizing pupil data')
    scaler = MinMaxScaler(feature_range=(-1, 1))
    return scaler.fit_transform(data.reshape(-1, 1)).flatten()


def apply_smoothing(data, method, settings):
    if method == 'moving_avg':
        sampling_rate = settings["eyetrack_sampling"]
        window = settings["window"]
        return moving_average_smoothing(data, sampling_rate, window)
    elif method == 'gaussian':
        sigma = settings.get("sigma")
        return gaussian_smoothing(data, 'x_coordinate', sigma)
    else:
        print(f"Unknown smoothing method: {method}")
        return data

def downsample_to_tr(original_data):
    """
    ----------------------------------------------------------------------
    downsample_to_tr(original_data)
    ----------------------------------------------------------------------
    Goal of the function :
    Resample eyetracking signal to TR resolution for deepmreye training 
    label of DeepMReyeCalib
    ----------------------------------------------------------------------
    Input(s) :
    original_data : 1D eyetracking data to be resampled (x or y coordinates)
    ----------------------------------------------------------------------
    Output(s) :
    reshaped_data : 1D resampled and reshaped data (x or y coordinates)
    ----------------------------------------------------------------------
    Function created by Sina Kling
    ----------------------------------------------------------------------
    """
    from scipy.signal import resample
    eyetracking_rate = 1000  # Original sampling rate in Hz
    target_points_per_tr = 10  # 10 data points per 1.2 seconds
    tr_duration = 1.2  # 1.2 sec
    target_rate = target_points_per_tr / tr_duration  # 8.33 Hz

    # Calculate total number of data points in target rate
    eyetracking_in_sec = len(original_data) / eyetracking_rate  # 185 sec
    total_target_points = int(eyetracking_in_sec * target_rate) # 1541

    # Resample the data
    downsampled_data = resample(original_data, total_target_points) # resample into amount of wanted data points

    # Reshape into TRs
    num_trs = int(eyetracking_in_sec / tr_duration) 

    reshaped_data = downsampled_data[:num_trs * target_points_per_tr].reshape(num_trs, target_points_per_tr)

    # Check new shape
    print(reshaped_data.shape)

    plt_2 = plt.figure(figsize=(15, 6))
    plt.title("Downsampled timeseries")
    plt.xlabel('x-coordinate', fontweight='bold')
    plt.plot(reshaped_data)
    plt.show()

    return reshaped_data


# Load inputs and settings
main_dir, project_dir, subject, task, data_type, group = load_inputs()   
with open("settings.json") as f:
    settings = json.load(f)
ses = settings['session']
eye = settings['eye']

# Prepare save directory
if data_type == "labels":
    file_dir_save = ensure_save_dir_labels(f'{main_dir}/{project_dir}/derivatives/int_deepmreye/deepmreye_calib')  
    if task == "DeepMReyeClosed": 
        file_dir_save = ensure_save_dir_labels(f'{main_dir}/{project_dir}/derivatives/int_deepmreye/deepmreye_closed')  
elif data_type == "eyetracking":
    file_dir_save = ensure_save_dir_et(f'{main_dir}/{project_dir}/derivatives', subject)  

# Prepare design matrix directory
design_dir_save = ensure_design_dir(f'{main_dir}/{project_dir}/derivatives/int_deepmreye/deepmreye_calib', subject)
if task == "DeepMReyeClosed": 
    design_dir_save = ensure_design_dir(f'{main_dir}/{project_dir}/derivatives/int_deepmreye/deepmreye_closed', subject)

# Load data
df_event_runs, df_data_runs = extract_event_and_physio_data(main_dir, project_dir, subject, task, ses, settings['num_run'], eye)

# Preprocessing for each run
for run_idx, (df_event, df_data) in enumerate(zip(df_event_runs, df_data_runs)):
    if task == "DeepMReyeCalib": 
        eye_data_run, time_start_eye, time_end_eye = extract_eye_data_and_triggers(df_event, df_data,settings['first_trial_pattern_calib'], settings['last_trial_pattern_calib'])

        if data_type == 'pupil_labels': 
            pupil = eye_data_run[:, 3]

            # Define blinks: pupil == 0
            eye_closed = (pupil == 0).astype(float)

            samples_per_tr = int(settings["TR"] * 1000)  # assuming ms precision
            total_trs = len(eye_closed) // samples_per_tr
            eye_closed = eye_closed[:total_trs * samples_per_tr]
            eye_closed = eye_closed.reshape((total_trs, samples_per_tr))

            # Compute % closed per TR, repeat across 10 bins
            num_bins = 10
            blink_matrix = []

            for tr_data in eye_closed:
                closed_pct = np.mean(tr_data) * 100
                tr_bins = [[closed_pct]] * num_bins  # repeat same value
                blink_matrix.append(tr_bins)

            blink_matrix = np.array(blink_matrix)  # shape: (TRs, 10, 1)
            print("Blink label shape:", blink_matrix.shape)

            save_path = f'{main_dir}/{project_dir}/derivatives/int_deepmreye/deepmreye_calib/model/eyesclosed_labels/{subject}_run_0{run_idx + 1}_eyesclosed_labels.npy'
            np.save(save_path, blink_matrix)

        # Apply preprocessing steps based on settings
        # --------- remove blinks ------------------
        eye_data_run = remove_blinks(eye_data_run, settings['blinks_remove'], settings['eyetrack_sampling'])
        # ------ convert to dva and center ---------
        eye_data_run = convert_to_dva(eye_data_run, settings['center'], settings['ppd'])
        # ------------ interpolate -----------------
        if data_type == 'eyetracking': 
            eye_data_run_x = interpolate_nans(eye_data_run[:,1])
            eye_data_run_y = interpolate_nans(eye_data_run[:,2])
        
            # ------------- detrending ----------------
            if settings.get('drift_corr'):
                eye_data_run_x = detrending(eye_data_run_x, subject, ses, run_idx, settings["fixation_column"], task, design_dir_save)
                eye_data_run_y = detrending(eye_data_run_y, subject, ses, run_idx, settings["fixation_column"], task, design_dir_save)

        
                
            eye_data_run = np.stack((eye_data_run[:,0],
                                            eye_data_run_x, 
                                            eye_data_run_y), axis=1)
                    
                    
            eye_data_run_df = pd.DataFrame(eye_data_run, columns=['timestamps','x', 'y'])
                   

            # Save the preprocessed data as tsv.gz
            tsv_file_path = f'{file_dir_save}/{subject}_task-{task}_run_0{run_idx+1}_eyedata.tsv.gz'
            print(f"saving to: {tsv_file_path}")
            save_preprocessed_data(eye_data_run_df, tsv_file_path)

        if data_type == 'labels': 
            print("processing labels")
            # dont interpolate
            eye_data_run_df = pd.DataFrame(eye_data_run, columns=['timestamps','x', 'y', 'pupil'])
            # ------------ downsampling --------------
            eye_data_run_x = segment_and_filter_eyetracking(eye_data_run_df['x'].to_numpy()) # downsample to 10 points per TR if TR is valid using mean pooling over 10 bins
            eye_data_run_y = segment_and_filter_eyetracking(eye_data_run_df['y'].to_numpy())


            eye_data_run = np.stack((eye_data_run_x, 
                                        eye_data_run_y), axis=-1)

            # ----------- save -----------------------

            print(eye_data_run.shape)
            np.save(f'{main_dir}/{project_dir}/derivatives/int_deepmreye/deepmreye_calib/model/gaze_labels/{subject}_run_0{run_idx + 1}_training_labels', eye_data_run)
     

    elif task == "DeepMReyeClosed": 
        eye_data_run, time_start_eye, time_end_eye = extract_eye_data_and_triggers(
            df_event, df_data,
            settings['first_trial_pattern_closed'],
            settings['last_trial_pattern_closed']
        )
        
        if data_type == 'pupil_labels': 
            pupil = eye_data_run[:, 3]

            # Define blinks: pupil == 0
            eye_closed = (pupil == 0).astype(float)

            samples_per_tr = int(settings["TR"] * 1000)  # assuming ms precision
            total_trs = len(eye_closed) // samples_per_tr
            eye_closed = eye_closed[:total_trs * samples_per_tr]
            eye_closed = eye_closed.reshape((total_trs, samples_per_tr))

            # Compute % closed per TR, repeat across 10 bins
            num_bins = 10
            blink_matrix = []

            for tr_data in eye_closed:
                closed_pct = np.mean(tr_data) * 100
                tr_bins = [[closed_pct]] * num_bins  # repeat same value
                blink_matrix.append(tr_bins)

            blink_matrix = np.array(blink_matrix)  # shape: (TRs, 10, 1)
            print("Blink label shape:", blink_matrix.shape)

            save_path = f'{main_dir}/{project_dir}/derivatives/int_deepmreye/deepmreye_closed/model/eyesclosed_labels/{subject}_run_0{run_idx + 1}_eyesclosed_labels.npy'
            np.save(save_path, blink_matrix)



        # Apply preprocessing steps based on settings
        # --------- remove blinks ------------------
        eye_data_run = remove_blinks(eye_data_run, settings['blinks_remove'], settings['eyetrack_sampling'])
        # ------ convert to dva and center ---------
        eye_data_run = convert_to_dva(eye_data_run, settings['center'], settings['ppd'])
        if data_type == 'eyetracking': 
            print('processing eyedata')
            design_matrix = pd.read_csv(f"{design_dir_save}/{subject}/{subject}_{ses}_task-{task}_run-0{run_idx+1}_design_matrix.tsv", sep ="\t")
            task_1_trials = np.array(design_matrix["trial_type_eyes_open_task"])
            task_3_trials = np.array(design_matrix["trial_type_no_stimulus_task"])
           
            resampled_task_1_trials = resample(task_1_trials, len(eye_data_run[:,1]))
            resampled_task_3_trials = resample(task_3_trials, len(eye_data_run[:,1]))

            task_1_trials_bool = resampled_task_1_trials > 0.5
            task_3_trials_bool = resampled_task_3_trials > 0.5

            task_1_trials_data_X = eye_data_run[:,1][task_1_trials_bool]
            task_1_trials_data_Y = eye_data_run[:,2][task_1_trials_bool]

            task_3_trials_data_X = eye_data_run[:,1][task_3_trials_bool]
            task_3_trials_data_Y = eye_data_run[:,2][task_3_trials_bool]

            # ---------------- interpolate -------------------------------

            task_1_trials_data_X = interpolate_nans(task_1_trials_data_X)
            task_1_trials_data_Y = interpolate_nans(task_1_trials_data_Y)

            task_3_trials_data_X = interpolate_nans(task_3_trials_data_X)
            task_3_trials_data_Y = interpolate_nans(task_3_trials_data_Y)


            # Replace the interpolated data for task 1
            eye_data_run[:, 1][task_1_trials_bool] = task_1_trials_data_X
            eye_data_run[:, 2][task_1_trials_bool] = task_1_trials_data_Y

            # Replace the interpolated data for task 3
            eye_data_run[:, 1][task_3_trials_bool] = task_3_trials_data_X
            eye_data_run[:, 2][task_3_trials_bool] = task_3_trials_data_Y

            eye_data_run = np.stack((eye_data_run[:,0],
                                        eye_data_run[:, 1], 
                                        eye_data_run[:, 2]), axis=1) 
                
                
            eye_data_run_df = pd.DataFrame(eye_data_run, columns=['timestamps','x', 'y']) 
            
            # Save the preprocessed data as tsv.gz
            tsv_file_path = f'{file_dir_save}/{subject}_task-{task}_run_0{run_idx+1}_eyedata.tsv.gz'
            save_preprocessed_data(eye_data_run_df, tsv_file_path)

        elif data_type == 'labels': 
            print('processing labels')
            # dont interpolate
            eye_data_run_df = pd.DataFrame(eye_data_run, columns=['timestamps','x', 'y', 'pupil'])
            # ------------ downsampling --------------
            eye_data_run_x = segment_and_filter_eyetracking(eye_data_run_df['x'].to_numpy()) # downsample to 10 points per TR if TR is valid using mean pooling over 10 bins
            eye_data_run_y = segment_and_filter_eyetracking(eye_data_run_df['y'].to_numpy())

            eye_data_run = np.stack((eye_data_run_x, 
                                        eye_data_run_y), axis=-1)

            design_matrix = pd.read_csv(f"{design_dir_save}/{subject}/{subject}_{ses}_task-{task}_run-0{run_idx+1}_design_matrix.tsv", sep ="\t")
            task_4_trials = np.array(design_matrix["trial_type_eyes_closed_task"])
            resampled_task_4_trials = resample(task_4_trials, len(eye_data_run[:,0]))
            task_4_trials_bool = resampled_task_4_trials > 0.5

            # Replace task 4 values with NaN
            eye_data_run[task_4_trials_bool, :] = np.nan

            # ----------- save -----------------------

            print(eye_data_run.shape)
            np.save(f'{main_dir}/{project_dir}/derivatives/int_deepmreye/deepmreye_closed/model/gaze_labels/{subject}_run_0{run_idx + 1}_training_labels', eye_data_run)


        
                
       
    

    
    