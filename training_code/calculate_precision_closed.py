"""
-----------------------------------------------------------------------------------------
calculate_precision_closed.py
-----------------------------------------------------------------------------------------
Goal of the script:
Generate prediction for eyemovements and calculate euclidean distance 
-----------------------------------------------------------------------------------------
Input(s):
sys.argv[1]: main directory 
sys.argv[2]: project directory 
sys.argv[3]: subject 
sys.argv[4]: task 
sys.argv[5]: subtask
sys.argv[6]: group 
-----------------------------------------------------------------------------------------
Output(s):
tsv of fraction under thresholds
tsv.gz timeseries of Euclidean distance 
tsv.gz timeseries of Prediction
-----------------------------------------------------------------------------------------
To run:
cd ~/projects/deepmreye/training_code
python calculate_precision.py /scratch/mszinte/data deepmreye sub-03 DeepMReyeClosed task_1 327
-----------------------------------------------------------------------------------------
"""


import pandas as pd
import json
import numpy as np
import re
import matplotlib.pyplot as plt
import glob 
import os
import sys
import math 
import h5py
import scipy.io 
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy.signal import detrend, resample

# path of utils folder  
sys.path.append("{}/utils".format(os.getcwd()))
from eyetrack_utils import load_event_files, euclidean_distance, fraction_under_threshold,adapt_evaluation, split_predictions, downsample_to_targetrate, load_data_events, extract_triggers
# --------------------- Load settings and inputs -------------------------------------

def load_settings(settings_file):
    with open(settings_file) as f:
        settings = json.load(f)
    return settings

def load_events(main_dir, subject, ses, task): 
    data_events = load_event_files(main_dir, subject, ses, task)
    return data_events 

def load_inputs():
    return sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6]

def ensure_save_dir(base_dir, subject):
    save_dir = f"{base_dir}/{subject}/eyetracking"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return save_dir

def ensure_design_dir(base_dir, subject):
    save_dir = f"{base_dir}/exp_design"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return save_dir

# Load inputs and setting
main_dir, project_dir, subject, task, subtask, group = load_inputs()
with open("settings.json") as f:
    settings = json.load(f)
# Load main experiment settings
ses = settings['session'] 
eye = settings['eye']
num_run = settings['num_run']
eyetracking_sampling = settings['eyetrack_sampling']
target_sampling = settings['target_sampling']
screen_size = settings['screen_size']
ppd = settings['ppd']

#pt = "pretrained"
pt = None

file_dir_save = ensure_save_dir(f'{main_dir}/{project_dir}/derivatives/pp_data', subject)
design_dir_save = ensure_design_dir(f'{main_dir}/{project_dir}/derivatives/deepmreye_closed', subject)
data_events = load_event_files(main_dir, project_dir, subject, ses, task)
dfs_runs = [pd.read_csv(run, sep="\t") for run in data_events]

precision_all_runs = []
precision_one_thrs_list = []

threshold = settings['threshold']


if pt == "pretrained":
    subject_file = f'{main_dir}/{project_dir}/derivatives/deepmreye_closed/pp_data_pretrained/{subject}_DeepMReyeClosed_no_label.npz'
else: 
    subject_file = f'{main_dir}/{project_dir}/derivatives/deepmreye_closed/pp_data/{subject}_DeepMReyeClosed_label.npz'


#Load the prediction
if pt == "pretrained": 
    prediction_dict = np.load(f"{main_dir}/{project_dir}/derivatives/deepmreye_closed/pred/evaluation_dict_closed_pretrained.npy", allow_pickle=True).item()
else: 
    prediction_dict = np.load(f"{main_dir}/{project_dir}/derivatives/deepmreye_closed/pred/evaluation_dict_closed.npy", allow_pickle=True).item()

subject_prediction = prediction_dict[subject_file]
df_pred_median, df_pred_subtr = adapt_evaluation(subject_prediction)
print(df_pred_subtr)

#Split prediction into runs
segment_lengths = [int((len(df_pred_subtr)/3)), int((len(df_pred_subtr)/3)),int((len(df_pred_subtr)/3))]

subject_prediction_X = split_predictions(df_pred_subtr, 'X', segment_lengths)
subject_prediction_Y = split_predictions(df_pred_subtr, 'Y', segment_lengths)
print("prediction loaded")

for run in range(num_run):
    # ----------------------- eye tracking processing -----------------------------------------
    #Load the eye data
    eye_data = pd.read_csv(f"{file_dir_save}/{subject}_task-{task}_run_0{run + 1}_eyedata.tsv.gz", compression='gzip', delimiter='\t')
    eye_data = eye_data[['timestamps','x', 'y']].to_numpy()


    # Downsample eye tracking data to match prediction rate 
    eye_data_downsampled = downsample_to_targetrate(eye_data, eyetracking_sampling, target_sampling)
    print("eye tracking downsampled")

    timestamps = eye_data_downsampled[:,0]
    # Extract task triggers eye tracking
    # apply interpolation and smoothing to 1 and 3 
    design_matrix = pd.read_csv(f"{design_dir_save}/{subject}/{subject}_{ses}_task-{task}_run-0{run+1}_design_matrix.tsv", sep ="\t")
    task_1_trials = np.array(design_matrix["task_eyes_open_task"])
    task_3_trials = np.array(design_matrix["task_no_stimulus_task"])
    task_4_trials = np.array(design_matrix["task_eyes_closed_task"])
    
    resampled_task_1_trials = resample(task_1_trials, len(eye_data_downsampled[:,1]))
    resampled_task_3_trials = resample(task_3_trials, len(eye_data_downsampled[:,1]))
    resampled_task_4_trials = resample(task_4_trials, len(eye_data_downsampled[:,1]))

    task_1_trials_bool = resampled_task_1_trials > 0.5
    task_3_trials_bool = resampled_task_3_trials > 0.5
    task_4_trials_bool = resampled_task_4_trials > 0.5

    eye_data_task_1 = eye_data_downsampled[task_1_trials_bool]
    eye_data_task_3 = eye_data_downsampled[task_3_trials_bool]
    eye_data_task_4 = eye_data_downsampled[task_4_trials_bool]
  

    # ------------------------ prediction processing ------------------------------------------

    # Add time dimension to prediction
    sampling_frequency = 10  # 10 data points per 1.2 seconds
    time_step = 1.2 / sampling_frequency  # 0.12 seconds per sample

    # ----------------------- cutting in tasks -----------------------------------------------
    # Slice Prediction on newly created time dimension 
    

    if subtask == "task_1": 
        
        subject_prediction_X_run = subject_prediction_X[run][:int(len(eye_data_downsampled))][task_1_trials_bool]
        subject_prediction_Y_run = subject_prediction_Y[run][:int(len(eye_data_downsampled))][task_1_trials_bool]

        eye_data = eye_data_task_1
        print(eye_data.shape)
        print(subject_prediction_X_run.shape)

    elif subtask == "task_3": 
        subject_prediction_X_run = subject_prediction_X[run][:int(len(eye_data_downsampled))][task_3_trials_bool]
        subject_prediction_Y_run = subject_prediction_Y[run][:int(len(eye_data_downsampled))][task_3_trials_bool]

        eye_data = eye_data_task_3
        print(eye_data.shape)
        print(subject_prediction_X_run.shape)

    elif subtask == "task_4": 
        subject_prediction_X_run = subject_prediction_X[run][:int(len(eye_data_downsampled))][task_4_trials_bool]
        subject_prediction_Y_run = subject_prediction_Y[run][:int(len(eye_data_downsampled))][task_4_trials_bool]

        eye_data = eye_data_task_4
        print(eye_data.shape)
        print(subject_prediction_X_run.shape)


    print("plotting....")

    plot_rows = 1
    plot_cols = 3

    fig = make_subplots(rows=plot_rows, cols=plot_cols,shared_xaxes=True,vertical_spacing=0.05, subplot_titles= ['Hor. Coord. Fixation', 'Ver. Coord. Fixation'])

    # Set a common y-axis range
    common_y_range = [-11, 11]  # Adjust the range as needed


    fig.add_trace(go.Scatter(y=eye_data[:,1] ,showlegend=True, name='Eyetracking',line=dict(color='#0E1C36', width=2)), row = 1, col = 1)
    fig.add_trace(go.Scatter(y=subject_prediction_X_run,showlegend=True, name='Fine Tuned', line=dict(color='#069D6B', width=2)), row = 1, col = 1)
    fig.add_trace(go.Scatter(y=eye_data[:,2] ,showlegend=True, name='Eyetracking',line=dict(color='#0E1C36', width=2)), row = 1, col = 2)
    fig.add_trace(go.Scatter(y=subject_prediction_Y_run,showlegend=True, name='Fine Tuned', line=dict(color='#069D6B', width=2)), row = 1, col = 2)
    fig.add_trace(go.Scatter(x=eye_data[:,1], y=eye_data[:,2],showlegend=True, name='Eyetracking', line=dict(color='#0E1C36', width=2)), row = 1, col = 3) # 2D plot
    fig.add_trace(go.Scatter(x= subject_prediction_X_run, y=subject_prediction_Y_run,showlegend=True, name='Pretrained', line=dict(color='#069D6B', width=2)), row = 1, col = 3) # 2D plot

    fig.update_traces(showlegend=False, row=1, col=2)
    fig.update_traces(showlegend=False, row=1, col=3)
    # Format and show fig
    fig.update_layout(height=600, width=1500, template="simple_white", title_text=f"Eyetracker Gaze Position (X,Y) vs. Fine Tuned Gaze Position", 
                yaxis1 = dict(title = "<b>Hor. coord. (dva)<b>", title_font=dict(size=12)),
                yaxis2 = dict(title = "<b>Ver. coord. (dva)<b>",title_font=dict(size=12)))
            


    # Update subplot titles font
    fig.update_annotations(font=dict(size=14))


    #fig.show()
    if pt == "pretrained": 
        fig.write_image(f'{main_dir}/{project_dir}/derivatives/deepmreye_closed/figures/{subject}/{subject}_{subtask}_test_pretrained.pdf')
    else: 
        fig.write_image(f'{main_dir}/{project_dir}/derivatives/deepmreye_closed/figures/{subject}/{subject}_{subtask}_test.pdf')


    

    eucl_dist = euclidean_distance(eye_data,subject_prediction_X_run, subject_prediction_Y_run)
    #fig = px.line( y=eucl_dist)
    #fig.write_image(f'{subject}_{subtask}__ee_test.pdf')
    print(np.mean(eucl_dist))

    eucl_dist_df = pd.DataFrame(eucl_dist, columns=['ee'])
    # Save eucl_dist as tsv.gz
    if pt == "pretrained": 
        ee_file_path = f'{file_dir_save}/{subject}_task-{task}_subtask-{subtask}_run_0{run+1}_ee_pretrained.tsv.gz'
    else: 
        ee_file_path = f'{file_dir_save}/{subject}_task-{task}_subtask-{subtask}_run_0{run+1}_ee.tsv.gz'
    eucl_dist_df.to_csv(ee_file_path, sep='\t', index=False, compression='gzip')

    precision_fraction = fraction_under_threshold(eucl_dist)
    
        
    
    # Store precision for this run
    precision_all_runs.append(precision_fraction)
   



# Combine all precision data into a single DataFrame
precision_df = pd.DataFrame(precision_all_runs).T  # Transpose so each column is a run

print(precision_df)
# Rename columns to match `run_01`, `run_02`, etc.
precision_df.columns = [f"run_{i+1:02d}" for i in range(num_run)]


#precision_df["threshold"] = np.linspace(0, 9.0, 100)
# Add a column for the mean across runs
precision_df["precision_mean"] = precision_df.mean(axis=1)



# Save the DataFrame to a TSV file

if pt == "pretrained": 
    output_tsv_file = f"{file_dir_save}/{subject}_task-{task}_subtask-{subtask}_precision_summary_pretrained.tsv"
else: 
    output_tsv_file = f"{file_dir_save}/{subject}_task-{task}_subtask-{subtask}_precision_summary.tsv"
precision_df.to_csv(output_tsv_file, sep="\t", index=False)


print(f"Saved precision summary to {output_tsv_file}")
    
# Define permission cmd
#print('Changing files permissions in {}/{}'.format(main_dir, project_dir))
#os.system("chmod -Rf 771 {}/{}".format(main_dir, project_dir))
#os.system("chgrp -Rf {} {}/{}".format(group, main_dir, project_dir))


