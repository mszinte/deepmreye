"""
=========================================================================================
train_eyelid_state.py
=========================================================================================
Train DeepMReye to predict eyelid state (open vs closed) with flexible configuration.

Predicts 1D continuous output (eye closure proportion) starting from different 
pre-trained weights to compare fine-tuning strategies.

Usage:
    python train_eyelid_state.py <main_dir> <project_name> <dataset_name> \
        <label_type> <pretrained_weights> <output_name>

Arguments (positional):
    1. main_dir: Root data directory
    2. project_name: Project name (used to build paths)
    3. dataset_name: Dataset identifier (calib, eyestate, base, etc.)
    4. label_type: Type of labels to load (eyesclosed, eyeopen)
    5. pretrained_weights: Path to pre-trained model weights (.h5 file)
    6. output_name: Name for output files (DeepMReye, DeepMReye_Calibration)

Example:
    python train_eyelid_state.py /scratch/mszinte/data deepmreye calib eyesclosed \
            weights.h5 DeepMReye_Calibration

=========================================================================================
"""

import sys
import json
import os
import pickle
import glob
import warnings
import numpy as np
import pandas as pd

from deepmreye import preprocess, train
from deepmreye.util import data_generator, model_opts
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def setup_directories(base_dir, dataset_name):
    """Create necessary directories."""
    directories = {
        "pp_dir": os.path.join(base_dir, f"pp_data_{dataset_name}/"),
        "mask_dir": os.path.join(base_dir, "mask"),
        "report_dir": os.path.join(base_dir, "report"),
        "pred_dir": os.path.join(base_dir, "pred"),
    }

    for dir_path in directories.values():
        os.makedirs(dir_path, exist_ok=True)

    return directories


def load_settings(settings_file="settings.json"):
    """Load settings from JSON file."""
    with open(settings_file) as f:
        return json.load(f)


def preprocess_data(
    subjects, func_dir, mask_dir, mask_filename_template, preload_masks
):
    """Extract masks from functional data for each subject."""
    eyemask_small, eyemask_big, dme_template, mask, x_edges, y_edges, z_edges = (
        preload_masks
    )

    for subject in subjects:
        print(f"\nPreprocessing {subject}...")
        func_sub_dir = os.path.join(func_dir, subject)
        mask_sub_dir = os.path.join(mask_dir, subject)
        func_files = glob.glob(os.path.join(func_sub_dir, "*.nii.gz"))

        if not func_files:
            print(f"  WARNING: No functional files found for {subject}")
            continue

        # Check if masks already exist
        if os.path.exists(mask_sub_dir) and len(os.listdir(mask_sub_dir)) > 0:
            print(f"  Masks already exist, skipping")
            continue

        # Create mask directory
        os.makedirs(mask_sub_dir, exist_ok=True)

        # Process each functional file
        for func_file in func_files:
            print(f"  Processing {os.path.basename(func_file)}...")
            preprocess.run_participant(
                fp_func=func_file,
                dme_template=dme_template,
                eyemask_big=eyemask_big,
                eyemask_small=eyemask_small,
                x_edges=x_edges,
                y_edges=y_edges,
                z_edges=z_edges,
                transforms=["Affine", "Affine", "SyNAggro"],
            )


def prepare_training_data(
    subjects, mask_dir, label_dir, pp_dir, num_run, label_type
):
    """Prepare and save training data for each subject."""
    print("\n" + "=" * 80)
    print("PREPARING TRAINING DATA")
    print("=" * 80)

    for subject in subjects:
        print(f"\nPreparing {subject}...")
        subject_data = []
        subject_labels = []
        subject_ids = []

        for run in range(num_run):
            run_num = run + 1

            # Define file paths
            mask_filename = (
                f"mask_{subject}_ses-02_task-DeepMReyeClosed_run-0{run_num}"
                f"_space-T1w_desc-preproc_bold.p"
            )
            label_filename = f"{subject}_run_0{run_num}_{label_type}_labels.npy"

            mask_path = os.path.join(mask_dir, subject, mask_filename)
            label_path = os.path.join(label_dir, label_filename)

            # Validate files exist
            if not os.path.exists(mask_path):
                print(f"  WARNING: Mask not found - {mask_filename}")
                continue

            if not os.path.exists(label_path):
                print(f"  WARNING: Labels not found - {label_filename}")
                continue

            # Load and normalize mask
            this_mask = pickle.load(open(mask_path, "rb"))
            this_mask = preprocess.normalize_img(this_mask)

            # Load labels
            this_label = np.load(label_path)

            # Validate alignment
            if this_mask.shape[3] != this_label.shape[0]:
                print(
                    f"  WARNING: Shape mismatch Run {run_num} "
                    f"(Mask {this_mask.shape[3]} TRs vs Label {this_label.shape[0]} TRs)"
                )
                continue

            print(f"  Run {run_num}: {this_mask.shape} with labels {this_label.shape}")

            # Store data
            subject_data.append(this_mask)
            subject_labels.append(this_label)
            subject_ids.append(
                ([subject] * this_label.shape[0], [run_num] * this_label.shape[0])
            )

        # Save participant data if runs were collected
        if subject_data:
            preprocess.save_data(
                participant=f"{subject}_eyelid_state_labels",
                participant_data=subject_data,
                participant_labels=subject_labels,
                participant_ids=subject_ids,
                processed_data=pp_dir,
                center_labels=False,
            )
            print(f"  Saved {subject} ({len(subject_data)} runs)")
        else:
            print(f"  WARNING: No valid runs for {subject}")

    # Clean up system files
    try:
        os.system(f"rm {pp_dir}/.DS_Store")
        print("\nCleaned up .DS_Store files")
    except Exception as e:
        print(f"Note: Could not remove .DS_Store - {e}")


def train_and_evaluate(
    pp_dir, model_dir, pretrained_weights, opts, output_name, pred_dir
):
    """Train model with cross-validation and evaluate performance."""
    print("\n" + "=" * 80)
    print("TRAINING AND EVALUATION")
    print("=" * 80)

    # Find all preprocessed datasets
    datasets = [
        os.path.join(pp_dir, p)
        for p in os.listdir(pp_dir)
        if "eyelid_state_labels" in p
    ]

    if not datasets:
        raise ValueError(f"No preprocessed datasets found in {pp_dir}")

    print(f"\nFound {len(datasets)} subjects for training")

    # Create cross-validation generators
    cv_generators = data_generator.create_cv_generators(
        pp_dir,
        num_cvs=len(datasets),
        batch_size=opts["batch_size"],
        augment_list=(
            (opts["rotation_x"], opts["rotation_y"], opts["rotation_z"]),
            opts["shift"],
            opts["zoom"],
        ),
        mixed_batches=True,
    )

    evaluation = {}
    all_scores = {}

    # Train and evaluate for each cross-validation fold
    for cv_idx, generators in enumerate(cv_generators):
        print(f"\n{'='*80}")
        print(f"CV FOLD {cv_idx + 1}/{len(cv_generators)}")
        print(f"{'='*80}")

        # Pre-load model architecture
        (preload_model, preload_model_inference) = train.train_model(
            dataset="eyelid_state_pretrain",
            generators=generators,
            opts=opts,
            return_untrained=True,
        )

        # Load pre-trained weights
        print(f"Loading pre-trained weights from: {pretrained_weights}")
        preload_model.load_weights(pretrained_weights)

        # Extract testing subjects
        (_, _, _, _, _, _, full_testing_list, full_training_list) = generators
        print(f"Training subjects: {len(full_training_list)}")
        print(f"Testing subject: {full_testing_list}")

        # Fine-tune model
        print("\nFine-tuning model...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            (model, model_inference) = train.train_model(
                dataset=f"eyelid_state_{output_name}_finetune",
                generators=generators,
                opts=opts,
                use_multiprocessing=False,
                return_untrained=False,
                verbose=1,
                save=True,
                model_path=model_dir,
                models=[preload_model, preload_model_inference],
            )

        print(f"Training complete for CV fold {cv_idx + 1}")

        # Make predictions for test subject
        for idx, subj in enumerate(full_testing_list):
            print(f"\nEvaluating: {subj}")
            X, real_y = data_generator.get_all_subject_data(subj)

            # Get predictions
            pred_y, euc_pred = model_inference.predict(X, verbose=0, batch_size=16)

            # Store raw results
            evaluation[subj] = {
                "real_y": real_y,
                "pred_y": pred_y,
                "euc_pred": euc_pred,
                "cv_fold": cv_idx + 1,
            }

            # Evaluate if labels exist
            if real_y.size > 0:
                # Remove NaN values
                nan_indices = np.any(np.isnan(real_y), axis=(1, 2))
                real_y_clean = real_y[~nan_indices, ...]
                pred_y_clean = pred_y[~nan_indices, ...]

                if real_y_clean.size > 0:
                    # Flatten for metrics (dimension 0 = eye closure)
                    real_flat = real_y_clean[..., 0].reshape(-1)
                    pred_flat = pred_y_clean[..., 0].reshape(-1)

                    # Compute metrics
                    r2 = r2_score(real_flat, pred_flat)
                    pearson = np.corrcoef(real_flat, pred_flat)[0, 1]
                    mse = mean_squared_error(real_flat, pred_flat)
                    mae = mean_absolute_error(real_flat, pred_flat)

                    all_scores[subj] = {
                        "R2": r2,
                        "Pearson": pearson,
                        "MSE": mse,
                        "MAE": mae,
                        "cv_fold": cv_idx + 1,
                    }

                    print(f"  Eye Closure Prediction Performance:")
                    print(f"    R2: {r2:.5f} | Pearson: {pearson:.5f}")
                    print(f"    MSE: {mse:.5f} | MAE: {mae:.5f}")
                else:
                    print(f"  No valid labels after cleaning")
            else:
                print(f"  No labels available")

    return evaluation, all_scores


# ========================================================================================
# SCRIPT EXECUTION
# ========================================================================================

# Validate command line arguments
if len(sys.argv) != 7:
    print(__doc__)
    print(f"ERROR: Expected 6 arguments, got {len(sys.argv) - 1}")
    print(f"Usage: python train_eyelid_state.py <main_dir> <project_name> <dataset_name> \\")
    print(f"           <label_type> <pretrained_weights> <output_name>")
    sys.exit(1)

main_dir = sys.argv[1]
project_name = sys.argv[2]
# Build directory structure
base_dir = (f"{main_dir}/{project_name}/derivatives/int_deepmreye/deepmreye_eyelid_state")
dataset_name = sys.argv[3]
label_type = sys.argv[4]
pretrained_weights = sys.argv[5]
pretrained_weights = os.path.join(base_dir, "model", pretrained_weights)
print(pretrained_weights)
output_name = sys.argv[6]

print("\n" + "=" * 80)
print("DEEPMREYE EYELID STATE TRAINING")
print("=" * 80)


func_dir = os.path.join(base_dir, "func")
model_dir = os.path.join(base_dir, "model/")
label_dir = os.path.join(model_dir, f"{label_type}_labels")

# Validate pretrained weights exist
if not os.path.exists(pretrained_weights):
    raise FileNotFoundError(f"Pre-trained weights not found: {pretrained_weights}")

print(f"\nConfiguration:")
print(f"  Main directory: {main_dir}")
print(f"  Project: {project_name}")
print(f"  Dataset: {dataset_name}")
print(f"  Labels: {label_type}")
print(f"  Output name: {output_name}")
print(f"  Pre-trained weights: {pretrained_weights}")
print(f"  Base directory: {base_dir}")

# Setup directories
directories = setup_directories(base_dir, dataset_name)

# Load settings
settings = load_settings()
subjects = settings["subjects"]
num_run = settings["num_run"]

print(f"\nSubjects ({len(subjects)}): {subjects}")
print(f"Runs per subject: {num_run}")

# Setup model options
opts = model_opts.get_opts()
opts.update(
    {
        "epochs": settings.get("epochs", 100),
        "batch_size": settings.get("batch_size", 16),
        "steps_per_epoch": settings.get("steps_per_epoch", None),
        "validation_steps": settings.get("validation_steps", None),
        "lr": settings.get("lr", 0.001),
        "lr_decay": settings.get("lr_decay", 0.0),
        "rotation_x": settings.get("rotation_x", 0),
        "rotation_y": settings.get("rotation_y", 0),
        "rotation_z": settings.get("rotation_z", 0),
        "shift": settings.get("shift", 0),
        "zoom": settings.get("zoom", 0),
        "gaussian_noise": settings.get("gaussian_noise", 0.0),
        "dropout_rate": settings.get("dropout_rate", 0.0),
        "mc_dropout": False,
        "loss_euclidean": 1,  # For eye closure (not spatial coordinates)
        "loss_confidence": 0,  # For eye closure (not 2D gaze)
        "load_pretrained": pretrained_weights,
        "train_test_split": settings.get("train_test_split", 0.8),
    }
)

# Setup CUDA
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

# Preload masks
print("\nPreloading masks...")
preload_masks = preprocess.get_masks()

# Preprocess functional data
preprocess_data(
    subjects,
    func_dir,
    directories["mask_dir"],
    "mask_{subject}_ses-02_task-DeepMReyeClosed_run-0{run}_space-T1w_desc-preproc_bold.p",
    preload_masks,
)

# Prepare training data
prepare_training_data(
    subjects,
    directories["mask_dir"],
    label_dir,
    directories["pp_dir"],
    num_run,
    label_type,
)

# Train and evaluate
evaluation, scores = train_and_evaluate(
    directories["pp_dir"],
    model_dir,
    pretrained_weights,
    opts,
    output_name,
    directories["pred_dir"],
)

# Save results
eval_filename = f"evaluation_dict_eyelid_state_{output_name}.npy"
scores_filename = f"scores_dict_eyelid_state_{output_name}.npy"

eval_path = os.path.join(directories["pred_dir"], eval_filename)
scores_path = os.path.join(directories["pred_dir"], scores_filename)

np.save(eval_path, evaluation)
np.save(scores_path, scores)

print("\n" + "=" * 80)
print("TRAINING COMPLETE")
print("=" * 80)
print(f"Evaluation saved: {eval_path}")
print(f"Scores saved: {scores_path}")
print(f"Total subjects evaluated: {len(evaluation)}")
print(f"Subjects with metrics: {len(scores)}")
