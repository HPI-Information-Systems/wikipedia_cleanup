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

        for page_id in tqdm(page_ids):
            current_page_predictions = [[] for _ in testing_timeframes]

            relevant_page_ids = self.predictor.get_relevant_ids(page_id).copy()

            current_data_key = self.data[self.data["key"] == page_id].sort_values(
                by=["value_valid_from"]
            )
            relevant_page_ids = list(filter((page_id).__ne__, relevant_page_ids))
            current_data = self.data[
                self.data["key"].isin(relevant_page_ids)
            ].sort_values(by=["value_valid_from"])

            test_set_timestamps = current_data["value_valid_from"].dt.date.to_numpy()
            test_set_timestamps_key = current_data_key[
                "value_valid_from"
            ].dt.date.to_numpy()

            day_labels = [date in test_set_timestamps_key for date in test_dates]
            # empty dataframe with same columns
            train_input = current_data.iloc[:0]
            train_input_key = current_data_key.iloc[:0]

            for days_evaluated, current_date in enumerate(test_dates):
                if len(test_set_timestamps_key) > 0:
                    offset = np.searchsorted(
                        test_set_timestamps_key, current_date, side="left"
                    )
                    train_input_key = current_data.iloc[:offset]

                for i, timeframe in enumerate(testing_timeframes):
                    if len(test_set_timestamps) > 0:
                        offset = np.searchsorted(
                            test_set_timestamps,
                            current_date + timedelta(days=timeframe),
                            side="left",
                        )
                        train_input = current_data.iloc[:offset]
                    if days_evaluated % timeframe == 0:
                        current_page_predictions[i].append(
                            self.predictor.predict_timeframe(
                                train_input_key, train_input, current_date, timeframe
                            )
                        )

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
        self.labels = labels
        self.predictions = predictions

        prediction_stats = []
        for y_true, y_hat, title in zip(labels, predictions, timeframe_labels):
            prediction_stats.append(self.evaluate_prediction(y_true, y_hat, title))
        return prediction_stats

    def aggregate_labels(self, labels: np.ndarray, n: int):
        if n == 1:
            return labels
        if self.test_duration % n != 0:
            padded_labels = np.pad(labels, ((0, 0), (0, (n - self.test_duration) % n)))
        else:
            padded_labels = labels
        padded_labels = padded_labels.reshape(labels.shape[0], -1, n)
        return np.any(padded_labels, axis=2)

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
