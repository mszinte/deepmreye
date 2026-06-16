#!/usr/bin/env python3
"""
DeepMReye Eye Tracking Error Calculator

This script calculates estimation error (EE) between eye tracking data and model predictions
for different tasks (fixation, pursuit, freeview) with optional 5-degree window filtering.

Usage:
    python deepmreye_ee_calculator.py --subjects sub-01 sub-02 --evaluation-path path/to/evaluation.npz --model model_name
    python deepmreye_ee_calculator.py --subjects sub-01 --evaluation-path path/to/evaluation.npz --model model_name --five-degree-window
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import json
from typing import List, Tuple, Optional
import os



def load_subjects_from_settings(settings_path: str) -> List[str]:
    """
    Load subjects list from a JSON settings file.
    
    Parameters:
    - settings_path: Path to the JSON settings file
    
    Returns:
    - subjects: List of subject IDs
    
    Expected JSON format:
    {
        "subjects": ["sub-01", "sub-02", "sub-03", ...]
    }
    """
    try:
        with open(settings_path, 'r') as f:
            settings = json.load(f)
        
        if 'subjects' not in settings:
            raise KeyError("'subjects' key not found in settings file")
        
        subjects = settings['subjects']
        
        if not isinstance(subjects, list):
            raise TypeError("'subjects' must be a list in the settings file")
        
        if not subjects:
            raise ValueError("'subjects' list is empty in the settings file")
        
        print(f"Loaded {len(subjects)} subjects from settings file: {subjects}")
        return subjects
        
    except FileNotFoundError:
        raise FileNotFoundError(f"Settings file not found: {settings_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in settings file: {e}")


def chunk_and_median(eyetracking_data: np.ndarray, sampling_rate: int = 1000, chunk_duration: float = 1.2) -> np.ndarray:
    """
    Splits continuous eyetracking data into chunks of specified duration
    and computes the median for each chunk, ensuring no NaNs are returned.

    Parameters:
    - eyetracking_data: 1D NumPy array (continuous signal)
    - sampling_rate: int, samples per second (default: 1000 Hz)
    - chunk_duration: float, duration of each chunk in seconds (default: 1.2 s)

    Returns:
    - medians: 1D NumPy array with median values per chunk
    """
    # Remove NaNs from input
    eyetracking_data = np.nan_to_num(eyetracking_data, nan=0.0)

    chunk_size = int(sampling_rate * chunk_duration)
    num_chunks = len(eyetracking_data) // chunk_size

    medians = np.array([
        np.nanmedian(eyetracking_data[i * chunk_size: (i + 1) * chunk_size])
        for i in range(num_chunks)
    ])

    return medians


def adapt_evaluation(participant_evaluation: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Adapts participant evaluation data for analysis.
    
    Parameters:
    - participant_evaluation: dict containing evaluation data
    
    Returns:
    - df_pred_median: DataFrame with median predictions
    - df_pred_subtr: DataFrame with sub-TR predictions
    """
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
    subtr_values = np.concatenate((pred_y, pred_uncertainty[..., np.newaxis]), axis=2)
    index = pd.MultiIndex.from_product(
        [range(subtr_values.shape[0]), range(subtr_values.shape[1])],
        names=["TR", "subTR"]
    )
    df_pred_subtr = pd.DataFrame(
        subtr_values.reshape(-1, subtr_values.shape[-1]),
        index=index,
        columns=["X", "Y", "pred_error"]
    )

    return df_pred_median, df_pred_subtr


def euclidean_distance(eye_data: np.ndarray, pred_x: np.ndarray, pred_y: np.ndarray) -> np.ndarray:
    """
    Calculate Euclidean distance between eye tracking data and predictions.
    
    Parameters:
    - eye_data: Eye tracking data array
    - pred_x: X predictions
    - pred_y: Y predictions
    
    Returns:
    - eucl_dist: Euclidean distances
    """
    min_len = min(len(eye_data), len(pred_x), len(pred_y))
    eucl_dist = np.sqrt(
        (eye_data[:min_len, 0] - pred_x[:min_len])**2 + 
        (eye_data[:min_len, 1] - pred_y[:min_len])**2
    )
    return eucl_dist


def filter_positions(target_positions: np.ndarray, limit: float = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filters datapoints where the target stays within a specified window (centered at 0,0).
    
    Parameters:
    - target_positions: Shape (num_datapoints, 2), containing (x, y) positions
    - limit: Half of the desired window size (default 5 for a 10x10 window)
    
    Returns:
    - filtered_positions: Subset of target_positions within the window
    - indices: Indices of selected datapoints
    """
    within_bounds = (np.abs(target_positions[:, 0]) <= limit) & (np.abs(target_positions[:, 1]) <= limit)
    filtered_positions = target_positions[within_bounds]
    return filtered_positions, np.where(within_bounds)[0]


def create_task_labels(nTR: int = 154) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create task labels for different TR periods.
    
    Parameters:
    - nTR: Total number of TRs
    
    Returns:
    - task_labels: Array of task labels
    - tr_mask_task_1: Boolean mask for task 1 (fixation)
    - tr_mask_task_2: Boolean mask for task 2 (pursuit)
    - tr_mask_task_3: Boolean mask for task 3 (freeview)
    """
    # Define task TR distributions
    interim_TRs = 5  # Before, between, and after tasks
    task_TRs = [50, 54, 30]  # Task 1, Task 2, Task 3

    # Create the task_labels array
    task_labels = np.full(nTR, "interim", dtype=object)  # Default to "interim"

    # Assign tasks to TRs
    start_idx = interim_TRs  # Start after initial interim period
    for i, task_duration in enumerate(task_TRs, start=1):
        task_labels[start_idx:start_idx + task_duration] = f"task_{i}"
        start_idx += task_duration + interim_TRs  # Move to the next block

    # Create boolean masks for TRs where the desired task is active
    tr_mask_task_1 = task_labels == "task_1"
    tr_mask_task_2 = task_labels == "task_2"
    tr_mask_task_3 = task_labels == "task_3"
    
    return task_labels, tr_mask_task_1, tr_mask_task_2, tr_mask_task_3


def calculate_ee(subjects: List[str], model: str, evaluation_path: str = "/Users/sinakling/disks/meso_shared/deepmreye/derivatives/deepmreye_calib/pred/",
                base_path: str = "/Users/sinakling/disks/meso_shared/deepmreye/derivatives/pp_data",
                runs: int = 3, show_plots: bool = False) -> None:
    """
    Calculate estimation error for eye tracking data.
    
    Parameters:
    - subjects: List of subject IDs
    - evaluation_path: Path to evaluation data
    - model: Model name
    - base_path: Base path for data files
    - runs: Number of runs to process
    - show_plots: Whether to show plots
    """
    tasks = ["fixation", "pursuit", "freeview", "all"]
    evaluation_path = os.path.join(evaluation_path, f"evaluation_dict_calib_{model}.npy")
    
    evaluation = np.load(evaluation_path, allow_pickle=True).item()
    run_slices = [(0, 154), (154, 308), (308, None)]
    
    # Create task labels
    _, tr_mask_task_1, tr_mask_task_2, tr_mask_task_3 = create_task_labels()
    
    task_bool_mapping = {
        "fixation": tr_mask_task_1,
        "pursuit": tr_mask_task_2,
        "freeview": tr_mask_task_3,
        "all": slice(None)  # Select all for "all"
    }
    
    for subject in subjects:
        print(f"Processing subject: {subject}")
        
        for run in range(runs):
            print(f"  Processing run: {run + 1}")
            
            # Load eye tracking data
            eye_data_path = f"{base_path}/{subject}/eyetracking/timeseries/{subject}_task-DeepMReyeCalib_run_0{run+1}_eyedata.tsv.gz"
            try:
                eye_data = pd.read_csv(eye_data_path, compression='gzip', delimiter='\t')
                eye_data = eye_data[['timestamps', 'x', 'y']].to_numpy()
            except FileNotFoundError:
                print(f"Warning: Eye data file not found: {eye_data_path}")
                continue
            
            # Downsample eye data
            eye_data_downsampled_x = chunk_and_median(eye_data[:, 1])
            eye_data_downsampled_y = chunk_and_median(eye_data[:, 2])
            eye_data_downsampled = np.stack((eye_data_downsampled_x, eye_data_downsampled_y), axis=1)
            
            # Load prediction data
            if model == "pretrained": 
                subject_key = f"/scratch/mszinte/data/deepmreye/derivatives/int_deepmreye/deepmreye_calib/pp_data_{model}/{subject}_DeepMReyeCalib_no_label.npz"
            else: 
                subject_key = f"/scratch/mszinte/data/deepmreye/derivatives/int_deepmreye/deepmreye_calib/pp_data_{model}/{subject}_DeepMReyeCalib_label_{model}.npz"
            
            if subject_key not in evaluation:
                print(f"    Warning: Subject key not found in evaluation: {subject_key}")
                continue
                
            subject_data = evaluation[subject_key]
            df_pred_median, _ = adapt_evaluation(subject_data)
            
            sub_run_X = np.array(df_pred_median['X'])[run_slices[run][0]:run_slices[run][1]]
            sub_run_Y = np.array(df_pred_median['Y'])[run_slices[run][0]:run_slices[run][1]]
            
            for task in tasks:
                print(f"Processing task: {task}")
                
                task_bool = task_bool_mapping[task]
                eye_data_downsampled_task = eye_data_downsampled[task_bool]
                sub_run_X_task = sub_run_X[task_bool]
                if model == "pretrained": 
                    sub_run_Y_task = sub_run_Y[task_bool]
                    sub_run_Y_task = -1 * sub_run_Y_task # Y axis inversion for pretrained model
                else: 
                    sub_run_Y_task = sub_run_Y[task_bool]

                # Compute Euclidean distance
                ee = euclidean_distance(eye_data_downsampled_task, sub_run_X_task, sub_run_Y_task)
                print(f"Mean EE: {np.mean(ee):.4f}")
                
                # Save results
                eucl_dist_df = pd.DataFrame(ee, columns=['ee'])
                ee_file_path = f"{base_path}/{subject}/eyetracking/timeseries/{subject}_task-DeepMReyeCalib_subtask-{task}_run_0{run+1}_ee_{model}.tsv.gz"
                
                # Create directory if it doesn't exist
                Path(ee_file_path).parent.mkdir(parents=True, exist_ok=True)
                
                eucl_dist_df.to_csv(ee_file_path, sep='\t', index=False, compression='gzip')
                print(f"Saved: {ee_file_path}")


def calculate_ee_5_deg(subjects: List[str], model: str, evaluation_path: str = "/Users/sinakling/disks/meso_shared/deepmreye/derivatives/deepmreye_calib/pred/",
                      base_path: str = "/Users/sinakling/disks/meso_shared/deepmreye/derivatives/pp_data",
                      runs: int = 3, show_plots: bool = False) -> None:
    """
    Calculate estimation error with 5-degree window filtering.
    
    Parameters:
    - subjects: List of subject IDs
    - evaluation_path: Path to evaluation data
    - model: Model name
    - base_path: Base path for data files
    - runs: Number of runs to process
    - show_plots: Whether to show plots
    """
    tasks = ["fixation", "pursuit", "freeview", "all"]
    
    evaluation_path = os.path.join(evaluation_path, f"evaluation_dict_calib_{model}.npy")

    evaluation = np.load(evaluation_path, allow_pickle=True).item()
    run_slices = [(0, 154), (154, 308), (308, None)]
    
    # Create task labels
    _, tr_mask_task_1, tr_mask_task_2, tr_mask_task_3 = create_task_labels()
    
    task_bool_mapping = {
        "fixation": tr_mask_task_1,
        "pursuit": tr_mask_task_2,
        "freeview": tr_mask_task_3,
        "all": slice(None)  # Select all for "all"
    }
    
    for subject in subjects:
        print(f"Processing subject (5-deg window): {subject}")
        
        for run in range(runs):
            print(f"  Processing run: {run + 1}")
            
            # Load eye tracking data
            eye_data_path = f"{base_path}/{subject}/eyetracking/timeseries/{subject}_task-DeepMReyeCalib_run_0{run+1}_eyedata.tsv.gz"
            try:
                eye_data = pd.read_csv(eye_data_path, compression='gzip', delimiter='\t')
                eye_data = eye_data[['timestamps', 'x', 'y']].to_numpy()
            except FileNotFoundError:
                print(f"    Warning: Eye data file not found: {eye_data_path}")
                continue
            
            # Downsample eye data
            eye_data_downsampled_x = chunk_and_median(eye_data[:, 1])
            eye_data_downsampled_y = chunk_and_median(eye_data[:, 2])
            eye_data_downsampled = np.stack((eye_data_downsampled_x, eye_data_downsampled_y), axis=1)
            
            # Load fine-tuned prediction data (note: different path for 5-deg version)
            if model == "pretrained": 
                subject_key = f"/scratch/mszinte/data/deepmreye/derivatives/int_deepmreye/deepmreye_calib/pp_data_{model}/{subject}_DeepMReyeCalib_no_label.npz"
            else: 
                subject_key = f"/scratch/mszinte/data/deepmreye/derivatives/int_deepmreye/deepmreye_calib/pp_data_{model}/{subject}_DeepMReyeCalib_label_{model}.npz"
            
            if subject_key not in evaluation:
                print(f"    Warning: Subject key not found in evaluation: {subject_key}")
                continue
                
            subject_data = evaluation[subject_key]
            df_pred_median, _ = adapt_evaluation(subject_data)
            
            sub_run_X = np.array(df_pred_median['X'])[run_slices[run][0]:run_slices[run][1]]
            sub_run_Y = np.array(df_pred_median['Y'])[run_slices[run][0]:run_slices[run][1]]
            
            for task in tasks:
                print(f"    Processing task: {task}")
                
                task_bool = task_bool_mapping[task]
                eye_data_downsampled_task = eye_data_downsampled[task_bool]
                
                # Apply 5-degree window filtering
                eye_data_downsampled_task, indices = filter_positions(eye_data_downsampled_task)
                
                sub_run_X_task = sub_run_X[task_bool][indices]
                sub_run_Y_task = sub_run_Y[task_bool][indices]
                
                # Compute Euclidean distance
                ee = euclidean_distance(eye_data_downsampled_task, sub_run_X_task, sub_run_Y_task)
                print(f"      Mean EE (5-deg): {np.mean(ee):.4f}")
                
                # Save results
                eucl_dist_df = pd.DataFrame(ee, columns=['ee'])
                if model == 'pretrained':
                    ee_file_path = f"{base_path}/{subject}/eyetracking/timeseries/{subject}_task-DeepMReyeCalib_subtask-{task}_run_0{run+1}_ee_pt_fivedegree.tsv.gz"
                else: 
                    ee_file_path = f"{base_path}/{subject}/eyetracking/timeseries/{subject}_task-DeepMReyeCalib_subtask-{task}_run_0{run+1}_ee_ft_fivedegree.tsv.gz"
                
                # Create directory if it doesn't exist
                Path(ee_file_path).parent.mkdir(parents=True, exist_ok=True)
                
                eucl_dist_df.to_csv(ee_file_path, sep='\t', index=False, compression='gzip')
                print(f"      Saved: {ee_file_path}")


def main():
    """Main function to run the script."""
   
   # Inputs
    main_dir = sys.argv[1]
    project_dir = sys.argv[2]
    base_path = os.path.join(main_dir, project_dir, "/derivatives/pp_data")
    subjects = sys.argv[3]
    model = sys.argv[4]
    evaluation_path = sys.argv[5]
    five_degree_window = sys.argv[6]

    with open('../settings.json') as f:
        settings = f.read()
    
    runs = settings["num_runs"]
    
    # Resolve subjects (handle 'all' case)
    try:
        subjects = resolve_subjects(subjects, settings)
    except Exception as e:
        print(f"Error resolving subjects: {e}")
        sys.exit(1)
    
    # Validate inputs
    if not Path(evaluation_path).exists():
        print(f"Error: Evaluation file not found: {evaluation_path}")
        sys.exit(1)
    
    if not Path(base_path).exists():
        print(f"Error: Base path not found: {base_path}")
        sys.exit(1)
    
    print(f"Starting processing with:")
    print(f"  Subjects: {subjects}")
    print(f"  Evaluation path: {evaluation_path}")
    print(f"  Model: {args.model}")
    print(f"  Base path: {base_path}")
    print(f"  Runs: {runs}")
    print(f"  Five-degree window: {five_degree_window}")
    print()
    
    try:
        if five_degree_window:
            calculate_ee_5_deg(
                subjects=subjects,
                evaluation_path=evaluation_path,
                model=args.model,
                base_path=base_path,
                runs=runs,
                show_plots=args.show_plots
            )
        else:
            calculate_ee(
                subjects=subjects,
                evaluation_path=evaluation_path,
                model=args.model,
                base_path=base_path,
                runs=runs,
                show_plots=args.show_plots
            )
        
        print("\nProcessing completed successfully!")
        
    except Exception as e:
        print(f"Error during processing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()