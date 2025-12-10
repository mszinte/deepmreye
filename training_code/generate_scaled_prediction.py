
"""
Scaled Prediction Script using Fully Cross-Validated 80/20 Train-Test Split and Linear Regression

This script processes eye-tracking and pretrained model data to create scaled predictions
using cross-validation across subjects.
"""

import argparse
import sys
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split



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


def load_eye_tracking_data(subject, run_idx, eye_data_dir):
    """
    Load and downsample eye-tracking data for a subject and run.
    
    Args:
        subject: Subject identifier
        run_idx: Run index (0-based)
        eye_data_dir: Directory containing eye-tracking data
    
    Returns:
        Downsampled eye-tracking data as numpy array
    """
    eye_data_file = os.path.join(
        eye_data_dir, subject, "eyetracking", "timeseries",
        f"{subject}_task-DeepMReyeCalib_run_0{run_idx+1}_eyedata.tsv.gz"
    )
    
    if not os.path.exists(eye_data_file):
        raise FileNotFoundError(f"Eye tracking data file not found: {eye_data_file}")
    
    eye_data = pd.read_csv(eye_data_file, compression='gzip', delimiter='\t')
    eye_data = eye_data[['timestamps', 'x', 'y']].to_numpy()
    
    eye_data_downsampled_x = chunk_and_median(eye_data[:, 1])
    eye_data_downsampled_y = chunk_and_median(eye_data[:, 2])
    
    return np.stack((eye_data_downsampled_x, eye_data_downsampled_y), axis=1)


def process_subject_data(subject, evaluation_data, eye_data_dir, slices):
    """
    Process data for a single subject across all runs.
    
    Args:
        subject: Subject identifier
        evaluation_data: Evaluation dictionary containing pretrained model data
        eye_data_dir: Directory containing eye-tracking data
        slices: List of slice ranges per run
    
    Returns:
        Tuple of (orig_x, orig_y, target_x, target_y) data
    """
    # Get subject data from evaluation dictionary
    subject_key = f"/scratch/mszinte/data/deepmreye/derivatives/int_deepmreye/deepmreye_calib/pp_data_pretrained/{subject}_DeepMReyeCalib_no_label.npz"
    
    if subject_key not in evaluation_data:
        raise KeyError(f"Subject data not found in evaluation data: {subject_key}")
    
    subject_data = evaluation_data[subject_key]
    df_pred_median, df_pred_subtr = adapt_evaluation(subject_data)
    
    subject_orig_x, subject_orig_y, subject_target_x, subject_target_y = [], [], [], []
    
    for run_idx, (start, end) in enumerate(slices):
        # Load and downsample eye-tracking data
        eye_data_downsampled = load_eye_tracking_data(subject, run_idx, eye_data_dir)
        
        # Extract pretrained model predictions
        orig_x = np.array(df_pred_median['X'][start:end]).reshape(-1, 1)
        orig_y = np.array(df_pred_median['Y'][start:end]).reshape(-1, 1)
        target_x = eye_data_downsampled[:, 0].reshape(-1, 1)
        target_y = eye_data_downsampled[:, 1].reshape(-1, 1)
        
        subject_orig_x.append(orig_x)
        subject_orig_y.append(orig_y)
        subject_target_x.append(target_x)
        subject_target_y.append(target_y)
    
    return (np.concatenate(subject_orig_x), np.concatenate(subject_orig_y),
            np.concatenate(subject_target_x), np.concatenate(subject_target_y))


def train_and_predict(all_orig_x, all_orig_y, all_target_x, all_target_y, subjects):
    """
    Perform fully cross-validated training and prediction.
    
    Args:
        all_orig_x, all_orig_y: Original prediction data for all subjects
        all_target_x, all_target_y: Target (ground truth) data for all subjects
        subjects: List of subject identifiers
    
    Returns:
        Dictionary containing scaled predictions for each subject
    """
    scaled_data_dict = {}
    num_subjects = len(subjects)
    
    for test_idx in range(num_subjects):
        test_subject = subjects[test_idx]
        print(f"Testing on subject: {test_subject}")
        
        # Get train-validation indices (exclude test subject)
        train_val_idx = [i for i in range(num_subjects) if i != test_idx]
        
        # Split remaining subjects into 80% train, 20% validation
        train_idx, val_idx = train_test_split(train_val_idx, test_size=0.2, random_state=42)
        
        # Prepare training data
        train_orig_x = np.concatenate([all_orig_x[i] for i in train_idx])
        train_orig_y = np.concatenate([all_orig_y[i] for i in train_idx])
        train_target_x = np.concatenate([all_target_x[i] for i in train_idx])
        train_target_y = np.concatenate([all_target_y[i] for i in train_idx])
        
        # Train regression models
        model_x = LinearRegression().fit(train_orig_x, train_target_x)
        model_y = LinearRegression().fit(train_orig_y, train_target_y)
        
        # Apply trained model to the test subject
        test_orig_x = all_orig_x[test_idx]
        test_orig_y = all_orig_y[test_idx]
        
        scaled_x = model_x.predict(test_orig_x)  # PT -> ET X
        scaled_y = model_y.predict(test_orig_y)  # PT -> ET Y
        
        # Store results
        scaled_data_dict[test_subject] = {
            'scaled_x': scaled_x.tolist(),
            'scaled_y': scaled_y.tolist()
        }
    
    return scaled_data_dict


def plot_results(scaled_data_dict, all_orig_x, all_target_x, subjects, output_dir):
    """
    Plot comparison of original, scaled, and target data.
    
    Args:
        scaled_data_dict: Dictionary containing scaled predictions
        all_orig_x: Original prediction data
        all_target_x: Target data
        subjects: List of subject identifiers
        output_dir: Directory to save plots
    """
    num_subjects = len(subjects)
    cols = 2
    rows = (num_subjects + 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(10, 5 * rows))
    axes = axes.flatten()
    
    for i, subject in enumerate(subjects):
        eye_data_downsampled_x = all_target_x[i].flatten()
        df_pred_x = all_orig_x[i].flatten()
        scaled_x_concat = np.array(scaled_data_dict[subject]['scaled_x'])
        
        axes[i].plot(eye_data_downsampled_x, label="ET X")
        axes[i].plot(scaled_x_concat, label="Scaled X (Regressed PT)")
        axes[i].plot(df_pred_x, label="Original PT X")
        
        axes[i].set_title(f"Subject: {subject}")
        axes[i].legend(loc="upper right")
        axes[i].set_xlabel("Time")
        axes[i].set_ylabel("Position")
    
    # Hide unused subplots
    for i in range(num_subjects, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, "scaled_predictions_comparison.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {plot_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Create scaled predictions using cross-validated linear regression"
    )
    
    parser.add_argument(
        "--evaluation_file",
        required=True,
        help="Path to evaluation dictionary file (.npy)"
    )
    
    parser.add_argument(
        "--eye_data_dir",
        required=True,
        help="Directory containing eye-tracking data"
    )
    
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to save output files"
    )
    
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=["sub-02", "sub-03", "sub-04", "sub-05", "sub-06", "sub-07",
                "sub-08", "sub-09", "sub-10", "sub-11", "sub-13", "sub-14",
                "sub-15", "sub-16", "sub-17"],
        help="List of subject identifiers"
    )
    
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of runs per subject"
    )
    
    parser.add_argument(
        "--slice_ranges",
        nargs="+",
        default=["None,154", "154,308", "308,None"],
        help="Slice ranges per run (format: 'start,end' where None means no limit)"
    )
    
    parser.add_argument(
        "--no_plot",
        action="store_true",
        help="Skip plotting results"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.evaluation_file):
        print(f"Error: Evaluation file not found: {args.evaluation_file}")
        sys.exit(1)
    
    if not os.path.exists(args.eye_data_dir):
        print(f"Error: Eye data directory not found: {args.eye_data_dir}")
        sys.exit(1)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Parse slice ranges
    slices = []
    for slice_str in args.slice_ranges:
        start, end = slice_str.split(',')
        start = None if start == 'None' else int(start)
        end = None if end == 'None' else int(end)
        slices.append((start, end))
    
    print(f"Processing {len(args.subjects)} subjects with {args.runs} runs each")
    print(f"Slice ranges: {slices}")
    
    # Load evaluation data
    print("Loading evaluation data...")
    try:
        evaluation_data = np.load(args.evaluation_file, allow_pickle=True).item()
    except Exception as e:
        print(f"Error loading evaluation file: {e}")
        sys.exit(1)
    
    # Process all subjects
    print("Processing subject data...")
    all_orig_x, all_orig_y, all_target_x, all_target_y = [], [], [], []
    
    for subject in args.subjects:
        try:
            orig_x, orig_y, target_x, target_y = process_subject_data(
                subject, evaluation_data, args.eye_data_dir, slices
            )
            all_orig_x.append(orig_x)
            all_orig_y.append(orig_y)
            all_target_x.append(target_x)
            all_target_y.append(target_y)
            print(f"Processed subject: {subject}")
        except Exception as e:
            print(f"Error processing subject {subject}: {e}")
            continue
    
    if not all_orig_x:
        print("Error: No subjects were successfully processed")
        sys.exit(1)
    
    # Perform cross-validated training and prediction
    print("Performing cross-validated training and prediction...")
    scaled_data_dict = train_and_predict(
        all_orig_x, all_orig_y, all_target_x, all_target_y, args.subjects
    )
    
    # Save results
    output_file = os.path.join(args.output_dir, "scaled_prediction_calib.pkl")
    with open(output_file, "wb") as f:
        pickle.dump(scaled_data_dict, f)
    
    print(f"Results saved to: {output_file}")
    print(f"Subjects in results: {list(scaled_data_dict.keys())}")
    
    # Plot results
    if not args.no_plot:
        print("Generating plots...")
        plot_results(scaled_data_dict, all_orig_x, all_target_x, args.subjects, args.output_dir)
    
    print("Script completed successfully!")


if __name__ == "__main__":
    main()