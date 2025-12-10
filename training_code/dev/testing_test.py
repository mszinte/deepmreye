import pandas as pd
import json
import numpy as np
import sys
import os
import itertools
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import permutation_test

sys.path.append("{}/utils".format(os.getcwd()))
from training_utils import flatten_dicts, statistic

def load_inputs():
    main_dir, project_dir, task, models = sys.argv[1:5]
    model_list = models.split(",")  
    return main_dir, project_dir, task, model_list

main_dir, project_dir, task, models = load_inputs()
with open("settings.json") as f:
    settings = json.load(f)
subjects = settings["subjects"]

tasks = ['fixation', 'pursuit', 'freeview', 'all']

def calc_mean_ee(subjects, tasks, models):
    results = {model: {} for model in models}
    
    for model in models:
        for subject in subjects:
            results[model][subject] = {}
            for task in tasks:
                all_ee = []
                for run in range(1, 4):
                    file_path = f"{main_dir}/{project_dir}/derivatives/pp_data/{subject}/eyetracking/{subject}_task-DeepMReyeCalib_subtask-{task}_run_0{run}_ee_{model}.tsv.gz"
                    print(file_path)
                    if os.path.exists(file_path):
                        ee_data = pd.read_csv(file_path, compression='gzip', delimiter='\t')[['ee']].to_numpy()
                        all_ee.append(ee_data)
                
                if all_ee:
                    all_ee = np.concatenate(all_ee)
                    results[model][subject][task] = np.mean(all_ee)
                    print(np.mean(all_ee))
                else:
                    results[model][subject][task] = np.nan
    
    return results

# Compute mean EE for each model
subject_ee_means = calc_mean_ee(subjects, tasks, models)
print(subject_ee_means)

# Perform pairwise permutation tests
for model1, model2 in itertools.combinations(models, 2):
    print(f"Comparing {model1} vs {model2}")
    for task in tasks:
        data1 = flatten_dicts(subject_ee_means[model1], task)
        data2 = flatten_dicts(subject_ee_means[model2], task)
        
        if not (np.isnan(data1).all() or np.isnan(data2).all()):
            res = permutation_test((data1, data2), statistic, permutation_type='samples', vectorized=True, n_resamples=10000, alternative='two-sided')
            print(f"Task: {task}, Statistic: {res.statistic}, P-value: {res.pvalue:.4f}")

# Prepare Data for Plotting
data_list = []
for subject in subjects:
    for task in tasks:
        model_values = {model: subject_ee_means[model].get(subject, {}).get(task, np.nan) for model in models}
        data_entry = {"subject": subject, "task": task, **model_values}
        data_list.append(data_entry)
        
df = pd.DataFrame(data_list)
df_long = df.melt(id_vars=['subject', 'task'], value_vars=models, var_name='model', value_name='value')

# Define subject colors
colormap_subject_dict = {
    'sub-01': '#AA0DFE', 'sub-02': '#3283FE', 'sub-03': '#85660D', 'sub-04': '#782AB6',
    'sub-05': '#565656', 'sub-06': '#1C8356', 'sub-07': '#16FF32', 'sub-08': '#F7E1A0',
    'sub-09': '#E2E2E2', 'sub-11': '#1CBE4F', 'sub-13': '#DEA0FD', 'sub-14': '#FBE426', 'sub-15': '#325A9B'
}

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=["<b>Guided Fixation task</b>", "<b>Pursuit task</b>", "<b>Freeview task</b>", "<b>All tasks</b>"],
    shared_yaxes=False,
    shared_xaxes=False
)

for i, task in enumerate(tasks):
    row, col = divmod(i, 2)
    row += 1
    col += 1
    
    task_df = df_long[df_long['task'] == task]
    
    # Strip plot (individual data points)
    strip_fig = px.strip(task_df, x='model', y='value', color_discrete_sequence=['gray'], stripmode='overlay')
    
    for trace in strip_fig.data:
        fig.add_trace(trace, row=row, col=col)
    
    # Connecting subject lines
    for subject in task_df['subject'].unique():
        subject_data = task_df[task_df['subject'] == subject]
        subject_data = subject_data.pivot(index='model', columns='subject', values='value').reset_index()
        
        if len(subject_data) == len(models):  # Ensure valid data
            color = colormap_subject_dict.get(subject, 'gray')
            
            fig.add_trace(go.Scatter(
                x=subject_data['model'],
                y=subject_data.iloc[:, 1],  # First column after 'model' is the subject's values
                mode='lines',
                opacity=0.7,
                line=dict(color=color, width=0.8),
                name=subject,
                legendgroup=subject,
                showlegend=False
            ), row=row, col=col)
    
    # Median and confidence intervals
    for model in models:
        model_data = task_df[task_df['model'] == model]['value'].dropna()
        
        if not model_data.empty:
            median = np.median(model_data)
            ci_lower, ci_upper = np.percentile(model_data, [2.5, 97.5])
            
            fig.add_trace(go.Scatter(
                x=[model],
                y=[median],
                mode='markers',
                marker=dict(color='black', size=15, symbol='circle'),
                showlegend=False
            ), row=row, col=col)
            
            fig.add_trace(go.Scatter(
                x=[model, model],
                y=[ci_lower, ci_upper],
                mode='lines',
                line=dict(color='black', width=2),
                showlegend=False
            ), row=row, col=col)

fig.update_layout(
    height=1000,
    width=1500,
    showlegend=True,
    template="simple_white",
    font=dict(size=15),
    yaxis_title="Euclidean Distance (dva)"
)

for r in range(1, 3):
    for c in range(1, 3):
        fig.update_yaxes(range=[1, 7], row=r, col=c)

fig.show()
