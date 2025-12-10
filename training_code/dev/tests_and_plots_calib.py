"""
-----------------------------------------------------------------------------------------
plots_and_tests_calib.py
-----------------------------------------------------------------------------------------
Goal of the script:
Calculate mean ee, apply permutation tests and plot model comparison 
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
python plots_and_tests_calib.py /Users/sinakling/disks/meso_shared deepmreye sub-01 DeepMReyeCalib 327
------------------------------------------------------------------------------------------------------------
"""
import pandas as pd
import json
import numpy as np
import re
import matplotlib.pyplot as plt
import os
import sys
from statistics import median
from scipy.stats import permutation_test
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots



sys.path.append("{}/utils".format(os.getcwd()))
from training_utils import flatten_dicts, statistic

# --------------------- Load settings and inputs -------------------------------------

def load_settings(settings_file):
    with open(settings_file) as f:
        settings = json.load(f)
    return settings

def load_inputs():
    return sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]


#--------------------- MAIN ----------------------------------------------------------
# Load inputs and settings
main_dir, project_dir, subject, task, model, group = load_inputs()   
with open("settings.json") as f:
    settings = json.load(f)

subjects = settings["subjects"]

# Calculate mean and 75th percentile of EE 
# Define tasks
tasks = ['fixation', 'pursuit', 'freeview', 'all']

def calc_mean_ee(subjects, tasks, model): 
    subject_ee_means = {}
    subject_ee_perc = {}

    for subject in subjects:
        subject_ee_means[subject] = {}
        subject_ee_perc[subject] = {}
        
        for task in tasks:
            # Load data for each run
            ee_run_01 = pd.read_csv(f"/Users/sinakling/disks/meso_shared/deepmreye/derivatives/pp_data/{subject}/eyetracking/{subject}_task-DeepMReyeCalib_subtask-{task}_run_01_ee_{model}.tsv.gz", compression='gzip', delimiter='\t')[['ee']].to_numpy()
            ee_run_02 = pd.read_csv(f"/Users/sinakling/disks/meso_shared/deepmreye/derivatives/pp_data/{subject}/eyetracking/{subject}_task-DeepMReyeCalib_subtask-{task}_run_02_ee_{model}.tsv.gz", compression='gzip', delimiter='\t')[['ee']].to_numpy()
            ee_run_03 = pd.read_csv(f"/Users/sinakling/disks/meso_shared/deepmreye/derivatives/pp_data/{subject}/eyetracking/{subject}_task-DeepMReyeCalib_subtask-{task}_run_03_ee_{model}.tsv.gz", compression='gzip', delimiter='\t')[['ee']].to_numpy()
            
            # Compute mean across all runs
            all_ee = np.concatenate([ee_run_01, ee_run_02, ee_run_03])
            mean_ee = np.mean(all_ee)
            perc_ee = np.percentile(all_ee, 75)
            
            # Save in dictionaries
            subject_ee_means[subject][task] = mean_ee
            subject_ee_perc[subject][task] = perc_ee
            print(f"model {model}:{task}, {subject}: {perc_ee}")


    return subject_ee_means, subject_ee_perc



# apply function for all wanted models 

# do permutation test 

# PERMUTATION TEST CORRELATION FT VS PT
from scipy.stats import permutation_test

for task in tasks:
    data1 = flatten_dicts(subject_ee_means_model_1, task)
    data2 = flatten_dicts(subject_ee_means_model_2, task)

    res = permutation_test((data1, data2), statistic, permutation_type = 'samples',vectorized = True, n_resamples=10000, alternative='two-sided')
    print(f"Task: {task}")
    print(f"statistic: {res.statistic}")
    print(f"P-value: {res.pvalue:.4f}")
    print("-" * 30)



# Prepare a list of dictionaries for the DataFrame
tasks = ['fix', 'pur', 'fv', 'all']

data_list = []
for subject in subjects:
    for task in tasks:
        data_list.append({
            'subject': subject,
            'task': task,
            'model_1': subject_ee_means_model_1, 
            "model_2": subject_ee_means_model_2
        })
        #

# Convert the list of dictionaries into a DataFrame
df = pd.DataFrame(data_list)

# Reshape data into long format for easier plotting
df_long = df.melt(id_vars=['subject', 'task'], value_vars=['model_1', 'model_2'], 
                   var_name='model', value_name='value')

# Define colors for each subject
colormap_subject_dict = {
    'sub-01': '#AA0DFE', 'sub-02': '#3283FE', 'sub-03': '#85660D', 'sub-04': '#782AB6',
    'sub-05': '#565656', 'sub-06': '#1C8356', 'sub-07': '#16FF32', 'sub-08': '#F7E1A0',
    'sub-09': '#E2E2E2', 'sub-11': '#1CBE4F', 'sub-13': '#DEA0FD', 'sub-14': '#FBE426', 'sub-15': '#325A9B'
}

# Create subplots with 2 rows and 2 columns
fig = make_subplots(
    rows=2, cols=2,  # Two rows, two columns
    subplot_titles=["<b>Guided Fixation task</b>", "<b>Pursuit task</b>", "<b>Freeview task</b>", "<b>All tasks</b>"],
    shared_yaxes=False, 
    shared_xaxes=False
)

# Add strip plots, subject connections, and median/confidence intervals for each task
for i, task in enumerate(tasks):
    row, col = divmod(i, 2)
    row += 1  # Adjusting for subplot indexing
    col += 1
    
    task_df = df_long[df_long['task'] == task]
    
    # Create a strip plot for the data distribution
    strip_fig = px.strip(task_df, x='model', y='value', color_discrete_sequence = ['gray'], stripmode='overlay')
    
    for trace in strip_fig.data:
        fig.add_trace(trace, row=row, col=col)
    
    for subject in task_df['subject'].unique():
        subject_data = task_df[task_df['subject'] == subject].set_index('model').loc[['model_1', 'model_2']].reset_index()
        color = colormap_subject_dict.get(subject, 'gray')
        
        # Add subject connecting lines
        fig.add_trace(go.Scatter(
            x=subject_data['model'],
            y=subject_data['value'],
            mode='lines',
            opacity=0.7,
            line=dict(color=color, width=0.8),  
            name=subject,
            legendgroup=subject,
            showlegend=False
        ), row=row, col=col)
    
    # Compute median and 95% confidence interval
    for model in ['model_1', 'model_2']:
        model_data = task_df[task_df['model'] == model]['value']
        median = np.median(model_data)
        ci_lower, ci_upper = np.percentile(model_data, [2.5, 97.5])  # 95% CI
        
        # Add median marker
        fig.add_trace(go.Scatter(
            x=[model],
            y=[median],
            mode='markers',
            marker=dict(color='black', size=15, symbol='circle'),
            showlegend=False
        ), row=row, col=col)
        
        # Add confidence interval
        fig.add_trace(go.Scatter(
            x=[model, model],
            y=[ci_lower, ci_upper],
            mode='lines',
            line=dict(color='black', width=2),
            showlegend=False
        ), row=row, col=col)

# Update layout
fig.update_layout(
    height=1000,  # Set the height of the plot
    width=1500,  # Set the width of the plot
    showlegend=True,
    template="simple_white",
    yaxis_title="Euclidean Distance (dva)"
)

# Ensure y-axis range is consistent for all subplots
fig.update_yaxes(range=[1, 7], row=1, col=1)
fig.update_yaxes(range=[1, 7], row=1, col=2)
fig.update_yaxes(range=[1, 7], row=2, col=1)
fig.update_yaxes(range=[1, 7], row=2, col=2)

fig.show()
#fig.write_image("/Users/sinakling/disks/meso_shared/deepmreye/derivatives/deepmreye_calib/figures/group/correlation_calib_plots.pdf")
