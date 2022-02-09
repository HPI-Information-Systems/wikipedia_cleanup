import itertools
import pickle
from datetime import date, datetime, timedelta
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from tqdm.auto import tqdm

from wikipedia_cleanup.predictor import CachedPredictor


class RandomForestPredictor(CachedPredictor):
    def __init__(self, use_cache: bool = True) -> None:
        super().__init__(use_cache)
        # contains for a given infobox_property_name (key) the regressor (value)
        self.regressors: dict = {}
        # contains for a given infobox_property_name (key) a (date,pred) tuple (value)
        # date is the date of the last change and pred the days until next change
        self.last_preds: dict = {}

    def get_relevant_ids(self, identifier: Tuple) -> List[Tuple]:
        return []

    @staticmethod
    def get_relevant_attributes() -> List[str]:
        return [
            "value_valid_from",
            "day_of_year",  # values from feature engineering
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
            # check if this really improves the prediction
            "mean_change_frequency_last_3",
        ]

    def _load_cache_file(self, file_object: Any) -> bool:
        self.regressors = pickle.load(file_object)
        return True

    def _get_cache_object(self) -> Any:
        return self.regressors

    def _fit_classifier(
        self, train_data: pd.DataFrame, last_day: datetime, keys: List[str]
    ) -> None:
        DUMMY_TIMESTAMP = pd.Timestamp("1999-12-31")
        # used as dummy for date comparison in first prediction
        keys = train_data["key"].unique()
        columns = train_data.columns.tolist()
        key_column_idx = columns.index("key")
        days_until_next_change_column_idx = columns.index("days_until_next_change")
        key_map = {
            key: np.array(list(group))
            for key, group in itertools.groupby(
                train_data.to_numpy(), lambda x: x[key_column_idx]
            )
        }
        relevant_train_column_indexes = [
            columns.index(relevant_attribute)
            for relevant_attribute in self.get_relevant_attributes()
        ]
        relevant_train_column_indexes.remove(columns.index("value_valid_from"))
        relevant_train_column_indexes.remove(days_until_next_change_column_idx)

        for key in tqdm(keys):
            current_data = key_map[key]
            if len(current_data) <= 1:
                continue
            sample = current_data[:-1, :]
            X = sample[:, relevant_train_column_indexes]
            y = sample[:, days_until_next_change_column_idx].astype(np.int64)
            reg = RandomForestClassifier(
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
        if data_key_item not in self.regressors:
            # checks if model has been trained for the key
            # (it didn't if there was no train data)
            return False

        value_valid_from_column_idx = columns.index("value_valid_from")
        sample = data_key[-1, ...]
        sample_value_valid_from = sample[value_valid_from_column_idx]
        if self.last_preds[data_key_item][0] != sample_value_valid_from:
            reg = self.regressors[data_key_item]
            indices = [
                columns.index(attr)
                for attr in self.get_relevant_attributes()
                if not (attr == "value_valid_from" or attr == "days_until_next_change")
            ]
            X_test = sample[indices].reshape(1, -1)
            pred = int(reg.predict(X_test)[0])
            self.last_preds[data_key_item] = (sample_value_valid_from, pred)

        else:
            pred = self.last_preds[data_key_item][1]

        return (
            first_day_to_predict
            <= (sample_value_valid_from + timedelta(pred))
            < first_day_to_predict + timedelta(timeframe)
        )
