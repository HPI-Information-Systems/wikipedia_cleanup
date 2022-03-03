import itertools
import pickle
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from tqdm.auto import tqdm

from wikipedia_cleanup.predictor import CachedPredictor


class RandomForestPredictor(CachedPredictor):
    def __init__(
        self,
        use_cache: bool = True,
        threshold: float = 0.6,
        min_number_changes: int = 230,
        cluster_classes: bool = False,
        return_probs: bool = False,
    ) -> None:
        super().__init__(use_cache)
        # contains for a given infobox_property_name (key) the regressor (value)
        self.classifiers: Dict[Tuple, RandomForestClassifier] = {}
        # contains for a given infobox_property_name (key) a (date,pred) tuple (value)
        # date is the date of the last change and pred the days until next change
        self.last_known_prediction: Any = None
        self.last_known_timestamp: Optional[date] = None
        self.last_known_timeframe: Optional[int] = None
        self.last_known_key: Optional[Tuple] = None

        self.threshold = threshold
        self.min_number_changes = min_number_changes
        self.return_probs = return_probs
        if return_probs:
            self.cluster_classes = False
        else:
            self.cluster_classes = cluster_classes

    def get_relevant_ids(self, identifier: Tuple) -> List[Tuple]:
        return []

    def get_relevant_attributes(self) -> List[str]:
        return [
            "value_valid_from",
            "day_of_year",
            "day_of_month",
            "day_of_week",
            "month_of_year",
            "days_since_last_change",
            "days_since_last_2_changes",
            "days_since_last_3_changes",
            "days_between_last_and_2nd_to_last_change",
            "days_until_next_change",
            "mean_change_frequency_all_previous",
            "mean_change_frequency_last_3",
        ]

    def _load_cache_file(self, file_object: Any) -> bool:
        self.classifiers = pickle.load(file_object)
        return True

    def _get_cache_object(self) -> Any:
        return self.classifiers

    def _fit_classifier(
        self, train_data: pd.DataFrame, last_day: datetime, keys: List[str]
    ) -> None:
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
            clf = RandomForestClassifier(
                random_state=0, n_estimators=10, max_features="auto"
            )
            clf.fit(X, y)
            self.classifiers[key] = clf

    @staticmethod
    def calc_sum_of_probabilites(
        classes: np.ndarray,
        pred_probs: np.ndarray,
        first_day_to_predict: date,
        date_of_last_change: date,
        timeframe: int,
    ) -> float:
        class_in_timestamp_idx = [
            i
            for i, c in enumerate(classes)
            if first_day_to_predict
            <= (date_of_last_change + timedelta(days=int(c)))
            < (first_day_to_predict + timedelta(days=timeframe))
        ]
        return pred_probs[class_in_timestamp_idx].sum()

    def predict_timeframe(
        self,
        data_key: np.ndarray,
        additional_data: np.ndarray,
        columns: List[str],
        first_day_to_predict: date,
        timeframe: int,
    ) -> Any:  # can be bool for actual predictions or float for confidence scores
        if len(data_key) == 0:
            return False
        if len(data_key) < self.min_number_changes:
            return False

        key_column_idx = columns.index("key")
        data_key_item = data_key[0, key_column_idx]
        if data_key_item not in self.classifiers:
            # checks if model has been trained for the key
            # (it didn't if there was no train data)
            return False

        value_valid_from_column_idx = columns.index("value_valid_from")
        sample = data_key[-1, ...]
        date_of_last_change = sample[value_valid_from_column_idx]
        if (
            data_key_item == self.last_known_key
            and data_key[-1, value_valid_from_column_idx] == self.last_known_timestamp
        ):
            # load cached prediction
            if self.return_probs or self.cluster_classes:
                classes = self.classifiers[data_key_item].classes_
                pred_probs = self.last_known_prediction
                sum_of_probabilites = RandomForestPredictor.calc_sum_of_probabilites(
                    classes,
                    pred_probs,
                    first_day_to_predict,
                    date_of_last_change,
                    timeframe,
                )
            else:
                pred = self.last_known_prediction
        else:
            # make prediction
            indices = [
                columns.index(attr)
                for attr in self.get_relevant_attributes()
                if not (attr == "value_valid_from" or attr == "days_until_next_change")
            ]
            X_test = sample[indices].reshape(1, -1)

            clf = self.classifiers[data_key_item]
            classes = clf.classes_
            pred_probs = clf.predict_proba(X_test)[0]

            self.last_known_key = data_key_item
            self.last_known_timestamp = date_of_last_change

            if self.return_probs or self.cluster_classes:
                sum_of_probabilites = RandomForestPredictor.calc_sum_of_probabilites(
                    classes,
                    pred_probs,
                    first_day_to_predict,
                    date_of_last_change,
                    timeframe,
                )
                self.last_known_prediction = pred_probs
                self.last_known_timeframe = timeframe
            else:
                if pred_probs.max() >= self.threshold:
                    pred = int(classes[pred_probs.argmax()])
                else:
                    pred = 9999
                self.last_known_prediction = pred

        if self.return_probs:
            return sum_of_probabilites
        elif self.cluster_classes:
            return sum_of_probabilites >= self.threshold
        else:
            return (
                first_day_to_predict
                <= (date_of_last_change + timedelta(days=pred))
                < first_day_to_predict + timedelta(days=timeframe)
            )
