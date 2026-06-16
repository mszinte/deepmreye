import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from training_utils import adapt_evaluation


def load_subject_evaluations(evaluation_dict, main_dir, project_dir, task_dir, model, label):
    """Load and adapt evaluation data for all subjects."""
    subject_list = [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]
    subject_dfs = {}

    for sub_id in subject_list:
        key = f"/{main_dir}/{project_dir}/{task_dir}/pp_data_{model}/sub-{sub_id:02d}_DeepMReyeClosed_{label}.npz"
        print(key)
        if key in evaluation_dict:
            df_median, _ = adapt_evaluation(evaluation_dict[key])
            subject_dfs[sub_id] = df_median
        else:
            print(f"Warning: Subject {sub_id} data not found in evaluation dict")

    return subject_dfs

def create_labeled_dataframe(subject_dfs, slice_length=9, nTR_run=313):
    """
    Create labeled dataframe with sliced coordinate sequences for each task.

    Parameters
    ----------
    subject_dfs : dict
        Dictionary mapping subject IDs to their DataFrames
    slice_length : int
        Length of coordinate slices (default: 9)
    nTR_run : int
        Number of TRs per run (default: 313)

    Returns
    -------
    tuple
        (df_final, tr_indices) - DataFrame with all sliced data and TR indices
    """
    # Define labeling pattern: (TR index within task, label)
    pattern = [
        (5, 'up'), (41, 'up'), (82, 'up'), (118, 'up'),
        (159, 'up'), (195, 'up'), (236, 'up'), (272, 'up'),
        (14, 'right'), (50, 'right'), (91, 'right'), (127, 'right'),
        (168, 'right'), (204, 'right'), (245, 'right'), (281, 'right'),
        (23, 'down'), (59, 'down'), (100, 'down'), (136, 'down'),
        (177, 'down'), (213, 'down'), (254, 'down'), (290, 'down'),
        (32, 'left'), (68, 'left'), (109, 'left'), (145, 'left'),
        (186, 'left'), (222, 'left'), (263, 'left'), (299, 'left')
    ]

    # Shift TRs by run offset
    indices_and_labels = []
    for run_idx in range(3):  # 3 runs
        indices_and_labels.extend(
            [(tr + run_idx * nTR_run, label) for tr, label in pattern]
        )

    # Create slices for all subjects
    all_data = []
    tr_indices = []

    for subject_id, df in subject_dfs.items():
        df = df.copy()
        for start_index, label in indices_and_labels:
            end_index = start_index + slice_length
            if end_index <= len(df):
                x_slice = df['X'].iloc[start_index:end_index].values
                y_slice = df['Y'].iloc[start_index:end_index].values

                if len(x_slice) == slice_length and len(y_slice) == slice_length:
                    all_data.append([subject_id, x_slice, y_slice, label])
                    tr_indices.append(start_index)

    # Format data into DataFrame
    columns = (
        ['subject'] +
        [f'x{i+1}' for i in range(slice_length)] +
        [f'y{i+1}' for i in range(slice_length)] +
        ['label']
    )
    
    formatted_data = [
        [entry[0]] + list(entry[1]) + list(entry[2]) + [entry[3]]
        for entry in all_data
    ]

    df_final = pd.DataFrame(formatted_data, columns=columns)

    # Encode labels
    label_encoder = LabelEncoder()
    df_final['label_encoded'] = label_encoder.fit_transform(df_final['label'])

    print("Label Mapping:", dict(zip(label_encoder.classes_, 
                                      label_encoder.transform(label_encoder.classes_))))

    # Assign task labels
    df_final['task'] = [get_task_label(tr) for tr in tr_indices]

    return df_final, tr_indices
       
def get_task_label(tr):
    tr_in_run = tr % nTR_run
    for task, (start, end) in task_TRs.items():
        if start <= tr_in_run < end:
            return task
    return "ITI"


    def run_classfier_loo(df_task, task_name, condition, model, save_dir):
        X = df_task[feature_cols].values
        y = df_task['label_encoded'].values
        groups = df_task['subject'].values

        # Compute baseline accuracy
        most_frequent_class = np.bincount(y).argmax()
        baseline_preds = np.full_like(y, most_frequent_class)
        baseline_acc = accuracy_score(y, baseline_preds)

        logo = LeaveOneGroupOut()
        subject_accuracies = {}
        conf_matrices = {}

        for train_idx, test_idx in logo.split(X, y, groups):
            subject_left_out = groups[test_idx][0]

            X_train, X_val = X[train_idx], X[test_idx]
            y_train, y_val = y[train_idx], y[test_idx]

            clf = LogisticRegression(max_iter=1000)
            clf.fit(X_train, y_train)

            y_pred = clf.predict(X_val)
            acc = accuracy_score(y_val, y_pred)

            subject_accuracies[subject_left_out] = acc

            # Train a classifier on all of df_task_1 for plotting
            X_plot = df_task_1[feature_cols].values
            y_plot = df_task_1['label_encoded'].values

            clf_plot = LogisticRegression(max_iter=1000)
            clf_plot.fit(X_plot, y_plot)
            # Confusion Matrix (as percent)
            cm = confusion_matrix(
                y_val, y_pred, labels=np.arange(len(df_final['label_encoded'].unique()))
            )

            # Normalize by row (i.e., true label) to get percent per class
            cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
            conf_matrices[subject_left_out] = cm_percent


        # Accuracy Summary
        df_acc = pd.DataFrame(subject_accuracies.items(), columns=["Subject", f"{task_name} Accuracy"])
        mean_acc = df_acc[f"{task_name} Accuracy"].mean()

        print(f"\n==== {task_name} ====")
        print(f"Baseline Accuracy (Majority Class): {baseline_acc:.2f}")
        print(f"Mean LOO-CV Accuracy: {mean_acc:.2f}")
        print(df_acc)

        df_acc.to_csv(f"{save_dir}/accuracies_{task_name}_{condition}_{model}.csv", index=False)

        return df_acc, conf_matrices