import pickle 
import numpy as np 
import pandas as pd
import sys
import os 
import json


# Inputs
main_dir = sys.argv[1]
project_dir = sys.argv[2]
subjects = sys.argv[3]

def chunk_and_median(eyetracking_data, sampling_rate=1000, chunk_duration=1.2):
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
        np.nanmedian(eyetracking_data[i * chunk_size: (i + 1) * chunk_size])  # Ensures NaN-safe median
        for i in range(num_chunks)
    ])

    return medians

def filter_positions(target_positions, limit=5):
    """
    Filters datapoints where the target stays within a 10x10 dva window (centered at 0,0).
    
    Parameters:
        target_positions (numpy array): Shape (num_datapoints, 2), containing (x, y) positions.
        limit (float): Half of the desired window size (default 5 for a 10x10 window).
    
    Returns:
        filtered_positions (numpy array): Subset of target_positions within the 10x10 dva window.
        indices (numpy array): Indices of selected datapoints.
    """
    # Check which points fall within the 5 dva window
    within_bounds = (np.abs(target_positions[:, 0]) <= limit) & (np.abs(target_positions[:, 1]) <= limit)
    
    # Extract only the valid positions
    filtered_positions = target_positions[within_bounds]
    
    return filtered_positions, np.where(within_bounds)[0]


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

nTR = 154  # Total TRs

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



# Create a boolean mask for TRs where the desired task is active
tr_mask_task_1 = task_labels == "task_1"
tr_mask_task_2 = task_labels == "task_2"
tr_mask_task_3 = task_labels == "task_3"




def calculate_correlation(subjects, model, evaluation_path=None, scaled_path=None, runs=3):
    tasks = ["fixation", "pursuit", "freeview", "all"]
    run_slices = [(0, 154), (154, 308), (308, None)]

    if model == 'scaled':
        with open(scaled_path, 'rb') as f:
            scaled_data_dict = pickle.load(f)
    else:
        evaluation = np.load(evaluation_path, allow_pickle=True).item()

    results = []

    for subject in subjects:
        for run in range(runs):
            eye_data = pd.read_csv(
                f"/Users/sinakling/disks/meso_shared/deepmreye/derivatives/pp_data/{subject}/eyetracking/timeseries/{subject}_task-DeepMReyeCalib_run_0{run+1}_eyedata.tsv.gz",
                compression='gzip', delimiter='\t')
            eye_data = eye_data[['timestamps', 'x', 'y']].to_numpy()
            eye_data_downsampled_x = chunk_and_median(eye_data[:, 1])
            eye_data_downsampled_y = chunk_and_median(eye_data[:, 2])
            eye_data_downsampled = np.stack((eye_data_downsampled_x, eye_data_downsampled_y), axis=1)

            start, end = run_slices[run]

            task_bool_mapping = {
                "fixation": tr_mask_task_1,
                "pursuit": tr_mask_task_2,
                "freeview": tr_mask_task_3,
                "all": slice(None)
            }

            if model == 'scaled':
                pred_x = np.array(scaled_data_dict[subject]['scaled_x'])[start:end]
                pred_y = np.array(scaled_data_dict[subject]['scaled_y'])[start:end]

                for task in tasks:
                    task_mask = task_bool_mapping[task]
                    eye_task = eye_data_downsampled[task_mask]
                    model_x = pred_x[task_mask]
                    model_y = pred_y[task_mask]

                    pearson_x = np.corrcoef(eye_task[:, 0].ravel(), model_x.ravel())[0, 1]
                    pearson_y = np.corrcoef(eye_task[:, 1].ravel(), model_y.ravel())[0, 1]

                    mean_corr = (pearson_x + pearson_y) / 2

                    results.append({
                        'subject': subject,
                        'run': run + 1,
                        'task': task,
                        'model': model,
                        'mean_pearson': mean_corr
                    })

            elif "ft_fivedegree" in model:
                subject_data = evaluation[
                    f"/scratch/mszinte/data/deepmreye/derivatives/int_deepmreye/deepmreye_calibration/pp_data_no_interpol/{subject}_DeepMReyeCalib_label_no_interpol.npz"]
                df_pred_median, _ = adapt_evaluation(subject_data)

                sub_run_X = np.array(df_pred_median['X'])[start:end]
                sub_run_Y = np.array(df_pred_median['Y'])[start:end]

                for task in tasks:
                    task_bool = task_bool_mapping[task]
                    eye_data_downsampled_task = eye_data_downsampled[task_bool]
                    eye_data_downsampled_task, indices = filter_positions(eye_data_downsampled_task)

                    sub_run_X_task = sub_run_X[task_bool][indices]
                    sub_run_Y_task = sub_run_Y[task_bool][indices] 
                    

                    pearson_x = np.corrcoef(eye_data_downsampled_task[:, 0].ravel(), sub_run_X_task.ravel())[0, 1]
                    pearson_y = np.corrcoef(eye_data_downsampled_task[:, 1].ravel(), sub_run_Y_task.ravel())[0, 1]
                    mean_corr = (pearson_x + pearson_y) / 2

                    results.append({
                        'subject': subject,
                        'run': run + 1,
                        'task': task,
                        'model': model,
                        'mean_pearson': mean_corr
                    })

            elif "pt_fivedegree" in model:
                subject_data = evaluation[
                    f"/scratch/mszinte/data/deepmreye/derivatives/int_deepmreye/deepmreye_calibration/pp_data_pretrained/{subject}_DeepMReyeCalib_no_label.npz"]
                df_pred_median, _ = adapt_evaluation(subject_data)

                sub_run_X = np.array(df_pred_median['X'])[start:end]
                sub_run_Y = np.array(df_pred_median['Y'])[start:end]

                for task in tasks:
                    task_bool = task_bool_mapping[task]
                    eye_data_downsampled_task = eye_data_downsampled[task_bool]
                    eye_data_downsampled_task, indices = filter_positions(eye_data_downsampled_task)

                    sub_run_X_task = sub_run_X[task_bool][indices]
                    sub_run_Y_task = -1 * sub_run_Y[task_bool][indices]  # Y inversion

                    pearson_x = np.corrcoef(eye_data_downsampled_task[:, 0].ravel(), sub_run_X_task.ravel())[0, 1]
                    pearson_y = np.corrcoef(eye_data_downsampled_task[:, 1].ravel(), sub_run_Y_task.ravel())[0, 1]
                    mean_corr = (pearson_x + pearson_y) / 2

                    results.append({
                        'subject': subject,
                        'run': run + 1,
                        'task': task,
                        'model': model,
                        'mean_pearson': mean_corr
                    })


            elif model == 'pretrained':
                subject_data = evaluation[
                    f"/scratch/mszinte/data/deepmreye/derivatives/int_deepmreye/deepmreye_calibration/pp_data_pretrained/{subject}_DeepMReyeCalib_no_label.npz"]
                df_pred_median, _ = adapt_evaluation(subject_data)
                pred_x = np.array(df_pred_median['X'][start:end])
                pred_y = -np.array(df_pred_median['Y'][start:end])  # invert Y-axis

                for task in tasks:
                    task_mask = task_bool_mapping[task]
                    eye_task = eye_data_downsampled[task_mask]
                    model_x = pred_x[task_mask]
                    model_y = pred_y[task_mask]

                    pearson_x = np.corrcoef(eye_task[:, 0].ravel(), model_x.ravel())[0, 1]
                    pearson_y = np.corrcoef(eye_task[:, 1].ravel(), model_y.ravel())[0, 1]

                    mean_corr = (pearson_x + pearson_y) / 2

                    results.append({
                        'subject': subject,
                        'run': run + 1,
                        'task': task,
                        'model': model,
                        'mean_pearson': mean_corr
                    })

            else:
                subject_data = evaluation[
                    f"/scratch/mszinte/data/deepmreye/derivatives/int_deepmreye/deepmreye_calibration/pp_data_{model}/{subject}_DeepMReyeCalib_label_{model}.npz"]
                df_pred_median, _ = adapt_evaluation(subject_data)
                pred_x = np.array(df_pred_median['X'][start:end])
                pred_y = np.array(df_pred_median['Y'][start:end])

                for task in tasks:
                    task_mask = task_bool_mapping[task]
                    eye_task = eye_data_downsampled[task_mask]
                    model_x = pred_x[task_mask]
                    model_y = pred_y[task_mask]

                    pearson_x = np.corrcoef(eye_task[:, 0].ravel(), model_x.ravel())[0, 1]
                    pearson_y = np.corrcoef(eye_task[:, 1].ravel(), model_y.ravel())[0, 1]

                    mean_corr = (pearson_x + pearson_y) / 2

                    results.append({
                        'subject': subject,
                        'run': run + 1,
                        'task': task,
                        'model': model,
                        'mean_pearson': mean_corr
                    })

    return pd.DataFrame(results)


# calculate correlation
df_corr_pt_calib = calculate_correlation(subjects, model='no_interpol', evaluation_path= f'{main_dir}/{project_dir}/derivatives/int_deepmreye/deepmreye_calibration/pred/evaluation_dict_calib_no_interpol.npy')
df_corr_scaled = calculate_correlation(subjects, model='scaled', scaled_path= f'{main_dir}/{project_dir}/derivatives/int_deepmreye/deepmreye_calibration/pred/scaled_prediction_calib.pkl')
df_corr_pt = calculate_correlation(subjects,  model='pretrained', evaluation_path= f'{main_dir}/{project_dir}/derivatives/int_deepmreye/deepmreye_calibration/pred/evaluation_dict_calib_pretrained.npy')
df_corr_pt_5deg = calculate_correlation(subjects,  model='pt_fivedegree', evaluation_path= f'{main_dir}/{project_dir}/derivatives/int_deepmreye/deepmreye_calibration/pred/evaluation_dict_calib_pretrained.npy')
df_corr_ft_5deg = calculate_correlation(subjects,  model='ft_fivedegree', evaluation_path= f'{main_dir}/{project_dir}/derivatives/int_deepmreye/deepmreye_calibration/pred/evaluation_dict_calib_no_interpol.npy')
df_corr_sim = calculate_correlation(subjects,  model='sim', evaluation_path= f'{main_dir}/{project_dir}/derivatives/int_deepmreye/deepmreye_calibration/pred/evaluation_dict_calib_sim.npy')


# concatonate finetuned vs scaled vs pretrained
df_corr = pd.concat([df_corr_pt_calib, df_corr_scaled, df_corr_pt], ignore_index=True)
# average over runs for each subject and task
df_mean_corr = (
    df_corr
    .groupby(['subject', 'task', 'model'], as_index=False)['mean_pearson']
    .mean()
)
df_mean_corr = df_mean_corr.sort_values(by=['subject', 'task'])


# concatonate fivedegree pretrained and fivedegree finetuned
df_corr_5deg = pd.concat([df_corr_ft_5deg, df_corr_pt_5deg,], ignore_index=True)
# average over runs for each subject and task
df_mean_corr_5deg = (
    df_corr_5deg
    .groupby(['subject', 'task', 'model'], as_index=False)['mean_pearson']
    .mean()
)
df_mean_corr_5deg = df_mean_corr_5deg.sort_values(by=['subject', 'task'])


# simulated labels model
# average over runs for each subject and task
df_mean_corr_sim = (
    df_corr_sim
    .groupby(['subject', 'task', 'model'], as_index=False)['mean_pearson']
    .mean()
)
df_mean_corr_sim = df_mean_corr_sim.sort_values(by=['subject', 'task'])


# save as csv
df_mean_corr.to_csv(f'{main_dir}/{project_dir}/derivatives/pp_data/group/correlations/correlation_calib.csv', index=False)
df_mean_corr_5deg.to_csv(f'{main_dir}/{project_dir}/derivatives/pp_data/group/correlations/correlation_5deg_calib.csv', index=False)
df_mean_corr_sim.to_csv(f'{main_dir}/{project_dir}/derivatives/pp_data/group/correlations/correlation_sim_calib.csv', index=False)

