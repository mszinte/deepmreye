"""
-----------------------------------------------------------------------------------------
classifier.py
-----------------------------------------------------------------------------------------
Goal: Load predictions per model and predict triangle rotation with logistic classifier.
-----------------------------------------------------------------------------------------
Input(s):
    sys.argv[1]: main project directory
    sys.argv[2]: project name (correspond to directory)
    sys.argv[3]: model name
    sys.argv[4]: condition (ordered or shuffled)
    sys.argv[5]: group of shared data (e.g. 327)
-----------------------------------------------------------------------------------------
Output(s):
    accuracies.csv per model
-----------------------------------------------------------------------------------------
Usage:
    cd ~/projects/deepmreye/analysis_code/postproc
    python classifier.py /scratch/mszinte/data deepmreye pretrained 327
-----------------------------------------------------------------------------------------
Written by Sina Kling (sina.kling@outlook.de)
Modified: Improved structure and removed code repetition
-----------------------------------------------------------------------------------------
"""

import numpy as np
import pandas as pd
import sys
import os
import random
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Personal imports
sys.path.append("{}/../analysis_code/utils".format(os.getcwd()))
from classifier_utils import load_subject_evaluations, create_labeled_dataframe, get_task_label, run_classfier_loo
from training_utils import adapt_evaluation

main_dir = sys.argv[1]
project_dir = sys.argv[2]
model = sys.argv[3]
condition = sys.argv[4]

# Set up paths 
task_dir = "derivatives/int_deepmreye/deepmreye_eyestate_tracking"
save_dir = f"{main_dir}/{project_dir}/{task_dir}/classifier/pred"
pred_dir = f"{main_dir}/{project_dir}/{task_dir}/pred"

# Small helper functions
def get_label_suffix(model):
    """Get label suffix based on model type."""
    return "label" if model == "gaze" else "no_label"

def invert_y_axis_if_needed(subject_dfs, model):
    """Invert Y axis for pretrained model."""
    if model == "pretrained":
        for df in subject_dfs.values():
            df["Y"] *= -1

def shuffle_coordinates(df_final, slice_length=9, seed=42):
    """
    Shuffle coordinate sequences while keeping labels intact.

    Parameters
    ----------
    df_final : pd.DataFrame
        DataFrame with coordinate columns
    slice_length : int
        Length of coordinate sequences
    seed : int
        Random seed for reproducibility

    Returns
    -------
    pd.DataFrame
        DataFrame with shuffled coordinates
    """
    np.random.seed(seed)
    random.seed(seed)
    print(f"Random seed set to {seed}")

    def shuffle_sequence(row):
        """Shuffle (x, y) coordinate pairs within a row."""
        coords = list(zip(
            [row[f'x{i}'] for i in range(1, slice_length + 1)],
            [row[f'y{i}'] for i in range(1, slice_length + 1)]
        ))
        random.shuffle(coords)
        xs, ys = zip(*coords)
        for i in range(slice_length):
            row[f'x{i+1}'] = xs[i]
            row[f'y{i+1}'] = ys[i]
        return row

    return df_final.copy().apply(shuffle_sequence, axis=1)


# Parse command-line arguments
if len(sys.argv) < 5:
    raise ValueError(
        "Usage: python classifier.py <main_dir> <project_dir> <model> <condition>"
    )

# MAIN -----------------------

label = get_label_suffix(model)

# Load evaluation data
print("Loading model predictions...")
evaluation = np.load(
    f"{pred_dir}/evaluation_dict_closed_{model}.npy",
    allow_pickle=True
).item()

# Load and adapt subject data
print("Preparing subject data...")
subject_dfs = load_subject_evaluations(
    evaluation, main_dir, project_dir, task_dir, model, label
)

# Invert Y axis if pretrained model
invert_y_axis_if_needed(subject_dfs, model)

# Create labeled dataframe
print("Creating labeled dataframe...")
df_final, tr_indices = create_labeled_dataframe(subject_dfs)

# Shuffle if condition is shuffled
if condition == "shuffled":
    print("Shuffling coordinate sequences...")
    df_final = shuffle_coordinates(df_final)

# Define feature columns
feature_cols = [f'x{i}' for i in range(1, 10)] + [f'y{i}' for i in range(1, 10)]

# Run classifier for each task
print("Running Leave-One-Group-Out cross-validation...")
results = {}
for task_num in range(1, 5):
    task_name = f"task_{task_num}"
    df_task = df_final[df_final['task'] == task_name].copy()

    if len(df_task) > 0:
        print(f"  Processing {task_name} ({len(df_task)} samples)...")
        df_acc, cm = run_classfier_loo(
            df_task, task_name, condition, model, save_dir
        )
        results[task_name] = {'accuracy': df_acc, 'confusion_matrix': cm}
    else:
        print(f"  Warning: No data for {task_name}")

print("Classification complete!")


