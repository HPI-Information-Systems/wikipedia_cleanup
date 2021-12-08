from datetime import datetime, timedelta
from typing import Optional
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

import numpy as np

from wikipedia_cleanup.data_filter import KeepAttributesDataFilter
from wikipedia_cleanup.data_processing import get_data


class TrainAndPredictFramework:

    def __init__(self, predictor: 'Predictor', test_start_date: datetime = datetime(2018, 9, 1),
                 test_duration: int = 365):
        self.test_start_date = test_start_date
        self.test_duration = test_duration
        # total_time_window = timedelta(testset_duration)  # days
        # testset_end = testset_start + total_time_window
        # time_offset = timedelta(1)

        self.predictor = predictor
        self.relevant_attributes = predictor.get_relevant_attributes()
        self.data: pd.DataFrame = pd.DataFrame()

    def load_data(self, input_path: Path, n_files: int, n_jobs: int):
        filters = [KeepAttributesDataFilter(self.relevant_attributes)]
        data = get_data(input_path, n_files=n_files, n_jobs=n_jobs, filters=filters)

    def fit_model(self):
        train_data = self.data[self.data['value_valid_from'] < self.test_start_date]
        self.predictor.fit(train_data, self.test_start_date)

    def test_model(self):
        page_ids = self.data['page_id'].unique()
        # property_change_history = self.data.groupby(['page_id']).agg(list)

        for page_id in page_ids:
            # for page_id, changes in tqdm(property_change_history.iteritems(), total=len(property_change_history)):
            days_evaluated = 0
            current_date = self.test_start_date
            relevant_page_ids = self.predictor.get_relevant_ids(page_id)
            current_data = self.data[self.data['page_id'].isin(relevant_page_ids)]

            # changes = np.sort(changes)
            # train_data_idx = np.searchsorted(changes, current_date, side="right")
            # day_predictions = np.empty(testset_duration)
            week_predictions = []
            month_predictions = []
            year_predictions = []
            day_labels = []
            test_dates = pd.date_range(self.test_start_date, self.test_start_date + timedelta(days=self.test_duration))
            for current_date in test_dates:
                # todo provide data of other page_ids for the current day.
                train_input = current_data[current_date['value_valid_from'] < current_date]
                # day_predictions[days_evaluated] = self.predictor.predict_day(train_input, current_date)
                # if days_evaluated % 7 == 0:
                #     week_predictions.append(self.predictor.predict_week(train_input, current_date))
                if days_evaluated % 30 == 0:
                    month_predictions.append(self.predictor.predict_month(train_input, current_date))
                if days_evaluated % 365 == 0:
                    year_predictions.append(self.predictor.predict_year(train_input, current_date))
                if train_data_idx < len(changes):
                    day_labels.append(changes[train_data_idx].date() == current_date.date())
                else:
                    day_labels.append(False)
                days_evaluated += 1
                current_date = current_dates[days_evaluated]
                while (train_data_idx < len(changes) and changes[train_data_idx] < current_date):
                    train_data_idx += 1
            all_day_predictions.append(day_predictions)
            all_week_predictions.append(week_predictions)
            all_month_predictions.append(month_predictions)
            all_year_predictions.append(year_predictions)
            all_day_labels.append(day_labels)

    def run_pipeline(self):
        pass


def next_change(previous_change_timestamps: np.ndarray) -> Optional[datetime]:
    if len(previous_change_timestamps) < 2:
        return None

    mean_time_to_change: timedelta = np.mean(
        previous_change_timestamps[1:] - previous_change_timestamps[0:-1]
    )
    return_value: datetime = previous_change_timestamps[-1] + mean_time_to_change
    return return_value


class Predictor:
    @staticmethod
    def get_relevant_attributes():
        return []

    def fit(self, train_data: pd.DataFrame, last_day: datetime):
        pass

    def predict_day(self, data: pd.DataFrame, current_day: datetime):
        return False

    def predict_week(self, data: pd.DataFrame, current_day: datetime):
        return False

    def predict_month(self, data: pd.DataFrame, current_day: datetime):
        return False

    def predict_year(self, data: pd.DataFrame, current_day: datetime):
        return False

    def get_relevant_ids(self, identifier):
        return list(identifier)
