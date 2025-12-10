def detrending(eyetracking_1D, subject, ses, run, fixation_column, task, design_dir_save): 
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import resample
    """
    Remove linear trends from eye-tracking data and median-center it during fixation periods for drift correction.

    Args:
        eyetracking_1D (np.array): 1D array of eye-tracking data to detrend.
        task (str): Task type, currently 'pRF' or other.

    Returns:
        np.array: Detrended eye-tracking data with trends removed and median-centered.
    """
    
    # Load and resample fixation data 
    fixation_trials = load_design_matrix_fixations(subject, ses, run, fixation_column, task, design_dir_save)  # Requires design matrix from task (see create_design_matrix.py)
    resampled_fixation_type = resample(fixation_trials, len(eyetracking_1D))
    fixation_bool = resampled_fixation_type > 0.5

    fixation_data = eyetracking_1D[fixation_bool]

    # Fit a linear model for the trend during fixation periods
    fixation_indices = np.where(fixation_bool)[0]
    trend_coefficients = np.polyfit(fixation_indices, fixation_data, deg=1)

    # Apply the linear trend to the entire dataset
    full_indices = np.arange(len(eyetracking_1D))
    linear_trend_full = np.polyval(trend_coefficients, full_indices)

    # Subtract the trend from the full dataset
    detrended_full_data = eyetracking_1D - linear_trend_full

    # Median centering using numpy's median function for consistency with numpy arrays
    fixation_median = np.median(detrended_full_data)
    detrended_full_data -= fixation_median

    # Plot the original and detrended data
    plt.plot(eyetracking_1D, label="Original Data")
    plt.plot(detrended_full_data, label="Detrended Data")
    plt.title("Detrended Full Eye Data")
    plt.xlabel("Time")
    plt.ylabel("Detrended Eye Position")
    plt.legend()
    plt.show()

    return detrended_full_data


def load_design_matrix_fixations(subject, ses, run, fixation_column, task, design_dir_save): 
    """
    Load the design matrix and extract fixation trial information.

    Args:
        fixation_column (str): Column name in the design matrix that contains fixation data.

    Returns:
        np.array: Array containing fixation trial information.
    """

    import pandas as pd
    import numpy as np
   
    design_matrix = pd.read_csv(f"{design_dir_save}/{subject}/{subject}_{ses}_task-{task}_run-0{run+1}_design_matrix.tsv", sep ="\t")
    fixation_trials = np.array(design_matrix[fixation_column])

    return fixation_trials

def load_event_files(main_dir, project_dir, subject, ses, task): 
    """
    Load event files from eye-tracking experiments.

    Args:
        main_dir (str): Main directory containing all experiment data.
        project_dir (str): Main project directory
        subject (str): Subject ID.
        ses (str): Session identifier.
        task (str): Task name.

    Returns:
        list: Sorted list of event file paths.
    """
    import glob
    
    data_events = sorted(glob.glob(r'{main_dir}/{project_dir}/{sub}/{ses}/func/{sub}_{ses}_task-{task}_*_events*.tsv'.format(
        main_dir=main_dir, project_dir=project_dir, sub=subject, ses = ses, task = task)))
    
    assert len(data_events) > 0, "No event files found"

    return data_events



def adapt_evaluation(participant_evaluation):
    import pandas as pd
    import numpy as np
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

def euclidean_distance(eye_data, pred_x, pred_y): 
     import numpy as np
     eucl_dist = np.sqrt(
        (eye_data[:int(len(pred_x)), 0] - pred_x[:int(len(eye_data))])**2 + (eye_data[:int(len(pred_y)), 1] - pred_y[:int(len(eye_data))])**2)

     return eucl_dist

import numpy as np

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
    Filters datapoints where the target stays within a 5 dva window (centered at 0,0).
    
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


def flatten_dicts(model_results, task):
    correlations = []
    for subject in model_results:
        if task in model_results[subject]:
            correlations.append(model_results[subject][task])
    return np.array(correlations)

def statistic(x, y, axis):
    return np.mean(x, axis=axis) - np.mean(y, axis=axis)