def paired_permutation_tests(df, value_col, model_a, model_b):
    results = []

    for task, group in df.groupby("task"):
        g = group[group["model"].isin([model_a, model_b])]

        pivoted = g.pivot(index="subject", columns="model", values=value_col).dropna()

        if pivoted.shape[0] < 2:
            continue

        a = pivoted[model_a].values
        b = pivoted[model_b].values

        res = permutation_test(
            (a, b),
            statistic=lambda x, y: (x - y).mean(),
            permutation_type="samples",
            alternative="two-sided",
            n_resamples=10000,
        )

        results.append({
            "task": task,
            "model_a": model_a,
            "model_b": model_b,
            "mean_diff": (a - b).mean(),
            "p_value": res.pvalue,
            "n": len(pivoted)
        })

    return pd.DataFrame(results)




def compute_ee_summary(subjects,tasks,runs,model_name,base_path,ci95=True,):
    """
    Compute EE summary per subject and task.

    Parameters
    ----------
    subjects : list
    tasks : iterable
    runs : iterable of run labels (strings)
    model_name : str
    base_path : str
    ci95 : bool
        If True, compute 95% confidence interval of the mean.

    Returns
    -------
    DataFrame
    """
    import pandas as pd
    import numpy as np
    from pathlib import Path

    results = []

    for subject in subjects:
        for task in tasks:

            run_arrays = []

            for run in runs:
                fpath = (
                    Path(base_path)
                    / subject
                    / "eyetracking"
                    / "timeseries"
                    / f"{subject}_task-DeepMReyeCalib_subtask-{task}_run_{run}_ee_{model_name}.tsv.gz"
                )

                if not fpath.exists():
                    continue

                arr = pd.read_csv(
                    fpath,
                    compression="gzip",
                    delimiter="\t",
                    usecols=["ee"],
                ).to_numpy()

                run_arrays.append(arr)

            if len(run_arrays) == 0:
                continue

            all_ee = np.concatenate(run_arrays).ravel()

            mean_ee = np.mean(all_ee)
            perc_75 = np.percentile(all_ee, 75)

            row = {
                "subject": subject,
                "task": task,
                "model": model_name,
                "mean": mean_ee,
                "75_perc": perc_75,
                "n_samples": len(all_ee),
            }

            if ci95:
                se = np.std(all_ee, ddof=1) / np.sqrt(len(all_ee))
                ci_low = mean_ee - 1.96 * se
                ci_high = mean_ee + 1.96 * se
                row["ci95_low"] = ci_low
                row["ci95_high"] = ci_high

            results.append(row)

    return pd.DataFrame(results)
