import calendar
import itertools
import pickle
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from tqdm.auto import tqdm

from wikipedia_cleanup.predictor import CachedPredictor


class RandomForestPredictor(CachedPredictor):
    def __init__(
        self, use_cache: bool = True, padding: bool = False, classify: bool = True
    ) -> None:
        super().__init__(use_cache)
        self.regressors: dict = {}
        self.classifiers: dict = {}
        self.last_preds: dict = {}
        # contains for a given infobox_propertyname (key) a (date,pred) tuple (value)
        # date is the date of the last change and pred the days until next change
        self.padding = padding
        self.classify = classify
        self.date_mapping: Dict[date, pd.Timestamp] = {}

    def get_relevant_ids(self, identifier: Tuple) -> List[Tuple]:
        return []

    def get_relevant_attributes(self) -> List[str]:
        if self.padding:
            return [
                "value_valid_from",
                "day_of_year",
                "day_of_month",
                "day_of_week",
                "month_of_year",
                "quarter_of_year",
                "is_month_start",
                "is_month_end",
                # "is_quarter_start",
                # "is_quarter_end",
                "days_since_last_change",
                "days_until_next_change",
            ]
        else:
            return [
                "value_valid_from",
                "day_of_year",
                "day_of_month",
                "day_of_week",
                "month_of_year",
                "quarter_of_year",
                "is_month_start",
                "is_month_end",
                "is_quarter_start",
                "is_quarter_end",
                "days_since_last_change",
                "days_since_last_2_changes",
                "days_since_last_3_changes",
                "days_between_last_and_2nd_to_last_change",
                "days_until_next_change",
                "mean_change_frequency_all_previous",
                "mean_change_frequency_last_3",
            ]

    def _load_cache_file(self, file_object) -> bool:
        if self.classify:
            self.classifiers = pickle.load(file_object)
        else:
            self.regressors = pickle.load(file_object)
        return True

    def _get_cache_object(self) -> Any:
        if self.classify:
            return self.classifiers
        else:
            return self.regressors

    def _fit_classifier(
        self, train_data: pd.DataFrame, last_day: datetime, keys: List[str]
    ) -> None:
        DUMMY_TIMESTAMP = pd.Timestamp("1999-12-31")
        # used as dummy for date comparison in first prediction
        keys = train_data["key"].unique()
        columns = train_data.columns.tolist()
        key_column_idx = columns.index("key")
        key_map = {
            key: np.array(list(group))
            for key, group in itertools.groupby(
                train_data.to_numpy(), lambda x: x[key_column_idx]
            )
        }
        relevant_train_column_indexes: List[int] = [
            columns.index(relevant_attribute)
            for relevant_attribute in self.get_relevant_attributes()
        ]
        for not_needed_attribute in ["value_valid_from", "days_until_next_change"]:
            relevant_train_column_indexes.remove(columns.index(not_needed_attribute))
        # relevant_train_column_indexes.remove(days_until_next_change_column_idx)
        # Assumption: data is already sorted by keys so
        # no further sorting needs to be done
        for key in tqdm(keys):
            current_data = key_map[key]
            if len(current_data) <= 2:
                continue
            # Remove last change from training data
            sample = current_data[:-1, :].copy()
            if self.padding:
                # ranges = sample['days_since_last_change'].
                # apply(lambda x: list(range(1, x+1)))
                # add ascending counts for days since last change
                # that get reset whenever there was a change
                range_list = [
                    list(range(1, x + 1))
                    for x in sample[:, columns.index("days_since_last_change")]
                ]
                ranges = np.append([0], np.concatenate(range_list).ravel())
                # add descending counts for days until next change
                # that get reset whenever there was a change
                # revert_ranges = sample['days_since_last_change']
                # .apply(lambda x: list(range(x+1, 1, -1)))
                reverted_range_list = [
                    list(range(x, 0, -1))
                    for x in sample[:, columns.index("days_since_last_change")]
                ]
                reverted_ranges = np.append(
                    np.concatenate(reverted_range_list).ravel(), [0]
                )
                # pad all days between first and last change
                # TODO: maybe use only random subset for padding
                padded_dates = np.arange(
                    sample[:, columns.index("value_valid_from")]
                    .min()
                    .to_pydatetime()
                    .date(),
                    sample[:, columns.index("value_valid_from")]
                    .max()
                    .to_pydatetime()
                    .date()
                    + timedelta(days=1),
                    timedelta(days=1),
                ).astype(datetime)
                padded_sample = np.zeros(
                    (padded_dates.shape[0], len(self.get_relevant_attributes())),
                    dtype=object,
                )
                padded_sample[:, 0] = padded_dates
                padded_sample[:, 1] = [
                    int(padded.strftime("%-j")) for padded in padded_dates
                ]  # day of year, use # instead of - on Windows
                padded_sample[:, 2] = [
                    int(padded.strftime("%-d")) for padded in padded_dates
                ]  # day of month
                padded_sample[:, 3] = [
                    int(padded.strftime("%w")) for padded in padded_dates
                ]  # day of week
                padded_sample[:, 4] = [
                    int(padded.strftime("%-m")) for padded in padded_dates
                ]  # month of year
                # needs to be a custom function as strftime does
                # not support quarters, only pandas.datetime does
                padded_sample[:, 5] = [
                    1 if month < 4 else 2 if month < 7 else 3 if month < 10 else 4
                    for month in padded_sample[:, 4]
                ]  # quarter of year
                padded_sample[:, 6] = [
                    True if day == 1 else False for day in padded_sample[:, 2]
                ]  # start of month
                padded_sample[:, 7] = [
                    True
                    if int(padded.strftime("%-d"))
                    == calendar.monthrange(
                        int(padded.strftime("%Y")), int(padded.strftime("%-m"))
                    )[1]
                    else False
                    for padded in padded_dates
                ]  # end of month
                # leaving out start and end of quarter for now as they would
                # need a custom function too and probably won't add lots of value
                if self.classify:
                    timeframe_since_last_change = [
                        1
                        if timeframe <= 1
                        else 7
                        if timeframe <= 7
                        else 30
                        if timeframe <= 30
                        else 365
                        if timeframe <= 365
                        else 9999
                        for timeframe in ranges
                    ]
                    timeframe_til_next_change = [
                        1
                        if timeframe <= 1
                        else 7
                        if timeframe <= 7
                        else 30
                        if timeframe <= 30
                        else 365
                        if timeframe <= 365
                        else 9999
                        for timeframe in reverted_ranges
                    ]
                    padded_sample[:, 8] = timeframe_since_last_change
                    padded_sample[:, 9] = timeframe_til_next_change
                else:
                    padded_sample[:, 8] = ranges  # days since last change
                    padded_sample[:, 9] = reverted_ranges  # days until next change
                X = padded_sample[:, 1:9]
                y = padded_sample[:, 9]
            else:
                X = sample[:, relevant_train_column_indexes]
                y = sample[:, columns.index("days_until_next_change")]
            y = y.astype("int")
            if self.classify:
                clf = RandomForestClassifier(
                    random_state=0, n_estimators=10, max_features="auto"
                )
                clf.fit(X, y)
                self.classifiers[key] = clf
            else:
                reg = RandomForestRegressor(
                    random_state=0, n_estimators=10, max_features="auto"
                )
                reg.fit(X, y)
                self.regressors[key] = reg

            self.last_preds[key] = (DUMMY_TIMESTAMP, 0)

    def predict_timeframe(
        self,
        data_key: np.ndarray,
        additional_data: np.ndarray,
        columns: List[str],
        first_day_to_predict: date,
        timeframe: int,
    ) -> bool:
        if len(data_key) == 0:
            return False
        key_column_idx = columns.index("key")
        data_key_item = data_key[0, key_column_idx]
        value_valid_from_column_idx = columns.index("value_valid_from")
        if self.classify:
            if data_key_item not in self.classifiers:
                # checks if model has been trained for the key
                # (it didnt if there was no traindata)
                return False
        else:
            if data_key_item not in self.regressors:
                # checks if model has been trained for the key
                # (it didnt if there was no traindata)
                return False

        if self.padding:
            if first_day_to_predict in self.date_mapping.keys():
                first_day_to_predict_pd = self.date_mapping[first_day_to_predict]
            else:
                first_day_to_predict_pd = pd.to_datetime(first_day_to_predict)
                self.date_mapping[first_day_to_predict] = first_day_to_predict_pd

            np_date = data_key[-1, value_valid_from_column_idx]
            if np_date in self.date_mapping:
                last_change = self.date_mapping[np_date]
            else:
                last_change = pd.to_datetime(data_key[-1, value_valid_from_column_idx])
                self.date_mapping[np_date] = last_change

            sample = [
                first_day_to_predict_pd.dayofyear,
                first_day_to_predict_pd.day,
                first_day_to_predict_pd.dayofweek,
                first_day_to_predict_pd.month,
                first_day_to_predict_pd.quarter,
                first_day_to_predict_pd.is_month_start,
                first_day_to_predict_pd.is_month_end,
            ]
            days_diff = (first_day_to_predict_pd - last_change).days
            if self.classify:
                sample.append(
                    1
                    if days_diff <= 1
                    else 7
                    if days_diff <= 7
                    else 30
                    if days_diff <= 30
                    else 365
                    if days_diff <= 365
                    else -1
                )
            else:
                sample.append(days_diff)
            X_test = sample
            if self.classify:
                clf = self.classifiers[data_key_item]
                pred = clf.predict([X_test])[0]
            else:
                reg = self.regressors[data_key_item]
                pred = int(reg.predict([X_test])[0])
            self.last_preds[data_key_item] = [first_day_to_predict, pred]
        else:
            sample = data_key[-1, ...]
            date_of_last_change = sample[value_valid_from_column_idx]
            if self.last_preds[data_key_item][0] != date_of_last_change:
            # check if we already predicted next change for that day
                indices : np.ndarray = [
                    columns.index(attr)
                    for attr in self.get_relevant_attributes()
                    if not (
                        attr == "value_valid_from"
                        or attr == "days_until_next_change"
                        or attr == "is_change"
                    )
                ]
                X_test = sample[indices].reshape(1, -1)
                if self.classify:
                    clf = self.classifiers[data_key_item]
                    pred = clf.predict(X_test)[0]
                else:
                    reg = self.regressors[data_key_item]
                    pred = int(reg.predict(X_test)[0])
                self.last_preds[data_key_item] = [date_of_last_change, pred]
            else:
                pred = self.last_preds[data_key_item][1]
        if self.padding:
            return pred <= timeframe
        return (
            first_day_to_predict
            <= (date_of_last_change + timedelta(days=int(pred)))
            < first_day_to_predict + timedelta(days=timeframe))
