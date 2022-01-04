from datetime import datetime, timedelta
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from tqdm.auto import tqdm

from wikipedia_cleanup.data_filter import (
    KeepAttributesDataFilter,
    OnlyUpdatesDataFilter,
)
from wikipedia_cleanup.data_processing import get_data
from wikipedia_cleanup.predictor import Predictor
from wikipedia_cleanup.property_correlation import PropertyCorrelationPredictor


class TrainAndPredictFramework:
    def __init__(
        self,
        predictor: Predictor,
        group_key: List[str],
        test_start_date: datetime = datetime(2018, 9, 1),
        test_duration: int = 365,
    ):
        self.test_start_date = test_start_date
        self.test_duration = test_duration
        self.group_key = group_key
        # total_time_window = timedelta(testset_duration)  # days
        # testset_end = testset_start + total_time_window
        # time_offset = timedelta(1)

        self.predictor = predictor
        own_relevant_attributes = ["value_valid_from"]
        self.relevant_attributes = list(
            set(
                own_relevant_attributes
                + predictor.get_relevant_attributes()
                + self.group_key
            )
        )
        self.data: pd.DataFrame = pd.DataFrame()

    def load_data(self, input_path: Path, n_files: int, n_jobs: int):
        filters = [
            OnlyUpdatesDataFilter(),
            KeepAttributesDataFilter(self.relevant_attributes),
        ]
        self.data = get_data(
            input_path, n_files=n_files, n_jobs=n_jobs, filters=filters  # type: ignore
        )
        self.data["value_valid_from"] = self.data["value_valid_from"].dt.tz_localize(
            None
        )
        self.data["key"] = list(
            zip(*[self.data[group_key] for group_key in self.group_key])
        )

    def fit_model(self):
        train_data = self.data[self.data["value_valid_from"] < self.test_start_date]
        self.predictor.fit(train_data.copy(), self.test_start_date)

    def test_model(self):
        page_ids = self.data["key"].unique()
        all_day_labels = []
        test_dates = [
            (self.test_start_date + timedelta(days=x)).date()
            for x in range(self.test_duration)
        ]

        testing_timeframes = [1, 7, 30, 365]
        timeframe_labels = ["day", "week", "month", "year"]
        predictions = [[] for _ in testing_timeframes]

        for page_id in tqdm(page_ids[:200]):
            days_evaluated = 0
            train_data_idx = 0
            current_page_predictions = [[] for _ in testing_timeframes]

            relevant_page_ids = self.predictor.get_relevant_ids(page_id)
            current_data = self.data[
                self.data["key"].isin(relevant_page_ids)
            ].sort_values(by=["value_valid_from"])
            test_set_timestamps = current_data["value_valid_from"].dt.date.to_numpy()

            # If the current page id is in the relevant_page_ids, this can probably be
            # sped up by looking at current_data
            current_page_id_timestamps = self.data[self.data["key"] == page_id][
                "value_valid_from"
            ].dt.date.to_numpy()
            day_labels = [date in current_page_id_timestamps for date in test_dates]
            # empty dataframe with same columns
            train_input = current_data.iloc[:0]
            for current_date in test_dates:
                # todo provide data of other page_ids for the current day.
                if len(test_set_timestamps) > 0:
                    offset = np.searchsorted(
                        test_set_timestamps, current_date, side="left"
                    )
                    test_set_timestamps = test_set_timestamps[offset:]
                    train_data_idx += offset
                    train_input = current_data.iloc[:train_data_idx]

                for i, timeframe in enumerate(testing_timeframes):
                    if days_evaluated % timeframe == 0:
                        current_page_predictions[i].append(
                            self.predictor.predict_timeframe(
                                train_input, current_date, timeframe
                            )
                        )
                days_evaluated += 1

            for i, prediction in enumerate(current_page_predictions):
                predictions[i].append(prediction)
            all_day_labels.append(day_labels)

        predictions = [
            np.array(prediction, dtype=np.bool) for prediction in predictions
        ]
        all_day_labels = np.array(all_day_labels, dtype=np.bool)
        labels = [
            self.aggregate_labels(all_day_labels, timeframe)
            for timeframe in testing_timeframes
        ]

        prediction_stats = []
        for y_true, y_hat, title in zip(labels, predictions, timeframe_labels):
            prediction_stats.append(self.evaluate_prediction(y_true, y_hat, title))
        return prediction_stats

    def aggregate_labels(self, labels: np.ndarray, n: int):
        if n == 1:
            return labels
        if self.test_duration % n != 0:
            padded_labels = np.pad(labels, ((0, 0), (0, n - self.test_duration % n)))
        else:
            padded_labels = labels
        padded_labels = padded_labels.reshape((-1, n, labels.shape[0]))
        return np.any(padded_labels, axis=1).reshape(labels.shape[0], -1)

    @staticmethod
    def print_stats(pre_rec_f1_stat, title):
        percent_data = pre_rec_f1_stat[3][1] / (
            pre_rec_f1_stat[3][0] + pre_rec_f1_stat[3][1]
        )
        print(f"{title} \t\t changes \t no changes")
        print(
            f"Precision:\t\t {pre_rec_f1_stat[0][1]:.4} \t\t {pre_rec_f1_stat[0][0]:.4}"
        )
        print(
            f"Recall:\t\t\t {pre_rec_f1_stat[1][1]:.4} \t\t {pre_rec_f1_stat[1][0]:.4}"
        )
        print(
            f"F1score:\t\t {pre_rec_f1_stat[2][1]:.4} \t\t {pre_rec_f1_stat[2][0]:.4}"
        )
        print(f"Percent of Data:\t {percent_data:.4}, \tTotal: {pre_rec_f1_stat[3][1]}")
        print()

    def evaluate_prediction(
        self, labels: np.ndarray, prediction: np.ndarray, title: str
    ):
        stats = precision_recall_fscore_support(labels.flatten(), prediction.flatten())
        self.print_stats(stats, title)
        return stats

    def run_pipeline(self):
        pass


if __name__ == "__main__":
    n_files = 40
    n_jobs = 8
    input_path = Path(
        "/run/media/secret/manjaro-home/secret/mp-data/custom-format-default-filtered"
    )
    model = PropertyCorrelationPredictor()
    # framework = TrainAndPredictFramework(model, ['infobox_key', 'property_name'])
    framework = TrainAndPredictFramework(model, ["page_id"])
    framework.load_data(input_path, n_files, n_jobs)
    framework.fit_model()
    framework.test_model()
