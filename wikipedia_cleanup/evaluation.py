import itertools
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_fscore_support


def stats_to_string(
    pre_rec_f1_stat: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    num_pos_predictions: int,
    title: str,
) -> str:
    percent_data = pre_rec_f1_stat[3][1] / (
        pre_rec_f1_stat[3][0] + pre_rec_f1_stat[3][1]
    )
    percent_changes_pred = num_pos_predictions / (
        pre_rec_f1_stat[3][0] + pre_rec_f1_stat[3][1]
    )

    output = (
        f"{title} \t\t\tchanges \tno changes \n"
        f"Precision:\t\t{pre_rec_f1_stat[0][1]:.4f} \t\t{pre_rec_f1_stat[0][0]:1.4f}\n"
        f"Recall:\t\t\t{pre_rec_f1_stat[1][1]:1.4f} \t\t{pre_rec_f1_stat[1][0]:1.4f}\n"
        f"F1score:\t\t{pre_rec_f1_stat[2][1]:1.4f} \t\t{pre_rec_f1_stat[2][0]:1.4f}\n"
        f"Changes of Data:\t{percent_data:.3%}, \tTotal: {pre_rec_f1_stat[3][1]}\n"
        f"Changes of Pred:\t{percent_changes_pred:.3%}, \tTotal: {num_pos_predictions}"
    )
    return output


def evaluate_prediction(
    labels: np.ndarray, prediction: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    stats = precision_recall_fscore_support(
        labels.reshape(-1), prediction.reshape(-1), zero_division=0, labels=[0, 1]
    )
    return stats


def create_prediction_output(
    labels: np.ndarray, prediction: np.ndarray, title: str
) -> str:
    stats = evaluate_prediction(labels, prediction)
    total_positive_predictions = np.count_nonzero(prediction)
    return stats_to_string(stats, total_positive_predictions, title)


def evaluate_bucketed_predictions(
    labels: np.ndarray,
    predictions: np.ndarray,
    testing_timeframes: List[int],
    output_folder: Path,
    keys: np.ndarray,
    data: pd.DataFrame,
):
    n_changes = data.groupby("key")["value_valid_from"].count()
    bucket_limits = [0, 5, 15, 50, 100, n_changes.max() + 1]
    buckets = list(zip(bucket_limits[:-1], bucket_limits[1:]))

    stats = []
    for low, high in buckets:
        keys_in_bucket = n_changes[(n_changes >= low) & (n_changes < high)].index
        used_indices = pd.DataFrame(keys)[0].isin(keys_in_bucket).to_numpy()
        stats.extend(evaluate_key_subset(labels, predictions, used_indices))

    plot_multi_stat_data(stats, testing_timeframes, buckets, output_folder / "bucketed")


def evaluate_template_predictions(
    labels: np.ndarray,
    predictions: np.ndarray,
    testing_timeframes: List[int],
    output_folder: Path,
    keys: np.ndarray,
    data: pd.DataFrame,
):
    top_n_templates = 10
    templates = data.groupby(["template"])
    template_count = templates["value_valid_from"].count()
    template_count = template_count.sort_values(ascending=False)
    top_templates = template_count.head(top_n_templates).index.tolist()
    keys_of_template = templates["key"].apply(set)

    stats = []
    for temp in top_templates:
        current_keys = set(keys_of_template.loc[temp])
        used_indices = [key in current_keys for key in keys]
        stats.extend(evaluate_key_subset(labels, predictions, used_indices))

    plt.figure(figsize=(10, 5))
    plot_multi_stat_data(
        stats, testing_timeframes, top_templates, output_folder / "templates"
    )


def evaluate_static_dynamic(
    labels: np.ndarray,
    predictions: np.ndarray,
    testing_timeframes: List[int],
    output_folder: Path,
    keys: np.ndarray,
    data: pd.DataFrame,
):
    assert "dynamic" in data.columns
    key_groups = (
        set(data[data["dynamic"]]["key"].unique()),
        set(data[~data["dynamic"]]["key"].unique()),
    )
    stats = []
    for current_keys in key_groups:
        used_indices = [key in current_keys for key in keys]
        stats.extend(evaluate_key_subset(labels, predictions, used_indices))

    plt.figure(figsize=(10, 5))
    plot_multi_stat_data(
        stats,
        testing_timeframes,
        ["dynamic", "static"],
        output_folder / "static_dynamic",
    )


def evaluate_key_subset(
    labels: np.ndarray, predictions: np.ndarray, used_indices: List[bool]
):
    current_evaluation = []
    cur_labels = [arr[used_indices] for arr in labels]
    cur_predictions = [arr[used_indices] for arr in predictions]
    for timeframe_label, timeframe_prediction in zip(cur_labels, cur_predictions):
        current_evaluation.append(
            evaluate_prediction(timeframe_label, timeframe_prediction)
        )
    return current_evaluation


def plot_multi_stat_data(
    stats: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    testing_timeframes: List[int],
    categories: List[Any],
    output_file: Path,
) -> None:
    assert all(len(pair) == 2 for single_run in stats for pair in single_run)
    plotting_df = pd.DataFrame(
        np.array(stats)[..., :2, 1].reshape(-1, 2),
        columns=["precision", "recall"],
    )
    plotting_df["timeframe"] = list(testing_timeframes * len(categories))
    plotting_df["category"] = list(
        itertools.chain.from_iterable(
            ([([i] * len(testing_timeframes)) for i in categories])
        )
    )
    plotting_df = (
        plotting_df.set_index(["timeframe", "category"])
        .sort_index()
        .reset_index()
        .set_index(["category", "timeframe"])
    )
    plotting_df.plot(kind="bar")
    # https://stackoverflow.com/questions/4700614/how-to-put-the-legend-outside-the-plot-in-matplotlib
    plt.legend(bbox_to_anchor=(0, 1.01, 1, 0.1), loc="center", borderaxespad=0, ncol=2)

    plt.ylabel("score")
    plt.savefig(output_file, bbox_inches="tight")


def evaluate_metric_over_time(
    labels: np.ndarray,
    predictions: np.ndarray,
    testing_timeframes: List[int],
    output_folder: Path,
    *args,
    **kwargs,
):
    for i, timeframe in enumerate(testing_timeframes[:-1]):
        current_labels = labels[i]
        current_predictions = predictions[i]
        stats = [
            precision_recall_fscore_support(
                current_labels[:, i],
                current_predictions[:, i],
                labels=[1],
                zero_division=0,
            )
            for i in range(current_labels.shape[1])
        ]
        prec = np.array([stat[0][0] for stat in stats])
        rec = np.array([stat[1][0] for stat in stats])
        plt.figure()
        if timeframe == 1:
            average = 5
            prec = np.mean(np.reshape(prec, (-1, average)), axis=1)
            rec = np.mean(np.reshape(rec, (-1, average)), axis=1)
            plt.xlabel(f"time in {timeframe} day(s), averaged over 5 days")
        else:
            plt.xlabel(f"time in {timeframe} day(s)")

        plt.plot(prec, label="precision")
        # trend line
        x = list(range(len(prec)))
        multidim_pol = np.polyfit(x, prec, 1)
        simple_pol = np.poly1d(multidim_pol)
        plt.plot(x, simple_pol(x), "r--", color="grey")

        plt.plot(rec, label="recall")
        plt.ylabel("score")
        plt.ylim((-0.05, 1.05))
        plt.legend(
            bbox_to_anchor=(0, 1.01, 1, 0.1), loc="center", borderaxespad=0, ncol=2
        )

        plt.savefig(output_folder / f"over_time_{timeframe}.png", bbox_inches="tight")


ALL_EVAL_METHODS = [
    evaluate_metric_over_time,
    evaluate_bucketed_predictions,
    evaluate_template_predictions,
    evaluate_static_dynamic,
]
