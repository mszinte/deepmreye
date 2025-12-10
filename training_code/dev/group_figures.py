"""
-----------------------------------------------------------------------------------------
group_figures.py
-----------------------------------------------------------------------------------------
Goal of the script:
Create figures for all subjects together showing the percentage of amount of data of the 
euclidean error under each threshold (precision) as well as under one specific 
threshold 
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
{subject}_{task}_threshold_precision.pdf
group_{task}_threshold_precision.pdf
{subject}_{task}_threshold_ranking.pdf
group_{task}_threshold_ranking.pdf
{subject}_{task}_stats_figure.pdf
group_{task}_stats_figure.pdf
-----------------------------------------------------------------------------------------
To run:
cd meso_H/projects/deepmreye/training_code
python group_figures.py /Users/sinakling/disks/meso_shared deepmreye sub-02 DeepMReyeCalib fixation 327
python group_figures.py /Users/sinakling/disks/meso_shared deepmreye group DeepMReyeCalib fixation 327
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
import plotly.express as px
from plotly.subplots import make_subplots

# path of utils folder  
sys.path.append("{}/utils".format(os.getcwd()))
from eyetrack_utils import load_event_files

# --------------------- Load settings and inputs -------------------------------------

def load_inputs():
    main_dir = sys.argv[1]
    project_dir = sys.argv[2]
    subject = sys.argv[3]
    #subjects = [sub.strip().strip('[]"\'') for sub in subjects_input.split(',')]
    task = sys.argv[4]
    subtask = sys.argv[5]
    group = sys.argv[6]

    return main_dir, project_dir, subject, task, subtask, group 

def load_events(main_dir, subject, ses, task): 
    data_events = load_event_files(main_dir, subject, ses, task)
    return data_events 


def ensure_save_dir(base_dir, subject, is_group=False):
    if is_group:
        save_dir = f"{base_dir}/group/eyetracking"
    else:
        save_dir = f"{base_dir}/pp_data/{subject}/eyetracking"
    #if not os.path.exists(save_dir):
        #os.makedirs(save_dir)
    return save_dir


# Load inputs and settings
main_dir, project_dir, subject_input, task, subtask, group = load_inputs()
print(subject_input)


with open("settings.json") as f:
    settings = json.load(f)


eye = settings['eye']
num_run = settings['num_run']
eyetracking_sampling = settings['eyetrack_sampling']
screen_size = settings['screen_size']
ppd = settings['ppd']

threshold = settings['threshold']


def process_subject(main_dir, project_dir, subject, task, subtask): 
    #pt = "pretrained"
    pt  = None
    file_dir_save = ensure_save_dir(f'{main_dir}/{project_dir}/derivatives', subject)
    if pt == "pretrained": 
        precision_summary = pd.read_csv(f"{file_dir_save}/{subject}_task-{task}_subtask-{subtask}_precision_summary_pretrained.tsv", delimiter='\t')
    else: 
        precision_summary = pd.read_csv(f"{file_dir_save}/{subject}_task-{task}_subtask-{subtask}_precision_summary_no_invers.tsv", delimiter='\t')
    return {
        "precision_mean": precision_summary["precision_mean"]
    }

    
def process_all_subjects(main_dir, project_dir, subjects, subject_input, task, subtask, threshold):

    precision_data = {
        subject: (
            result := process_subject(main_dir, project_dir, subject, task, subtask),
            {
                "precision_mean": result["precision_mean"]
            }
        )[1]
        for subject in subjects
    }
    colormap_subject_dict = {
    'sub-01': '#AA0DFE',
    'sub-02': '#3283FE',
    'sub-03': '#85660D',
    'sub-04': '#782AB6',
    'sub-05': '#565656',
    'sub-06': '#1C8356',
    'sub-07': '#16FF32',
    'sub-08': '#F7E1A0',
    'sub-09': '#E2E2E2',
    'sub-11': '#1CBE4F',
    'sub-13': '#DEA0FD',
    'sub-14': '#FBE426',
    'sub-15': '#325A9B'}

    generate_final_figure(precision_data, colormap_subject_dict, threshold, main_dir, project_dir, subject_input, task, subtask, thresholds=np.linspace(0, 9, 100))
    

def generate_final_figure(precision_data, colormap, threshold, main_dir, project_dir, subject_input, task, subtask, thresholds):
    #pt = "pretrained"
    pt = None
    traces = [
        go.Scatter(
            x=thresholds,
            y=data["precision_mean"],  
            mode='lines',
            name=f'Subject {subject}',
            line=dict(color=colormap.get(subject, '#000000'))  # Default to black if subject not in colormap
        )
        for subject, data in precision_data.items()  
    ]

    # Define layout
    layout = go.Layout(
        xaxis=dict(
            title='Euclidean distance error in dva', range=[0, 9], zeroline=True, linecolor='black', showgrid=False, tickmode='linear', dtick=2 
        ),
        yaxis=dict(
            title=r'% amount of data', range=[0, 1], zeroline=True, linecolor='black', showgrid=False
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Arial", size=12, color="black"),
        height=700,
        width=480,  
        shapes=[dict(
            type="line", x0=0, x1=9, y0=0.75, y1=0.75, line=dict(color="black", dash="dash")
        )]
    )

    # Create figure and show it
    fig = go.Figure(data=traces, layout=layout)
    fig.show()

    # Save figure dynamically
    is_group = subject_input == "group"
    save_dir = f"{main_dir}/{project_dir}/derivatives/deepmreye_calib/figures"
    if pt == "pretrained": 
        fig_fn = f"{save_dir}/{subject_input}/{subject_input}_{task}_{subtask}_threshold_precision_pretrained.pdf" if not is_group else f"{save_dir}/group/group_{task}_{subtask}_threshold_precision_pretrained.pdf"
    else: 
        fig_fn = f"{save_dir}/{subject_input}/{subject_input}_{task}_{subtask}_threshold_precision_no_invers.pdf" if not is_group else f"{save_dir}/group/group_{task}_{subtask}_threshold_precision.pdf"

    print(f"Saving {fig_fn}")
    fig.write_image(fig_fn)

    


# Check if the input is "group" or a specific subject
if subject_input == "group":
    subjects = settings["subjects"]
else:
    subjects = [subject_input]


process_all_subjects(main_dir, project_dir, subjects, subject_input, task, subtask, threshold)