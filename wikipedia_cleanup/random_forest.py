import hashlib
import pickle
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from tqdm.auto import tqdm

from wikipedia_cleanup.predictor import Predictor


class RandomForestPredictor(Predictor):
    def __init__(self, use_cache: bool = True) -> None:
        super().__init__()
        self.regressors: dict = {}
        self.last_preds: dict = {}
        self.hash_location = Path("cache") / self.__class__.__name__
        self.use_hash = use_cache

    @staticmethod
    def get_relevant_ids(identifier: Tuple) -> List[Tuple]:
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

    def _load_cache(self, possible_cached_mapping: Path) -> bool:
        try:
            if possible_cached_mapping.exists():
                print(f"Cached model found, loading from {possible_cached_mapping}")
                with open(possible_cached_mapping, "rb") as f:
                    self.related_properties_lookup = pickle.load(f)
                return True
            else:
                print("No cache found, recalculating model.")
        except EOFError:
            print("Caching failed, recalculating model.")
        return False

    def fit(
        self, train_data: pd.DataFrame, last_day: datetime, keys: List[str]
    ) -> None:
        if self.use_hash:
            possible_cached_mapping = self._calculate_cache_name(train_data)
            if self._load_cache(possible_cached_mapping):
                return
        ilocs = train_data.groupby(keys, sort=False).count()["value_valid_from"]
        # Assumption: data is already sorted by keys so
        # no further sorting needs to be done
        start = 0
        for iloc in tqdm(ilocs):
            if iloc==1: continue
            # maybe dont train if we are below a trainsize threshold
            sample = train_data.iloc[start : start + iloc]
            start += iloc
            sample = sample.drop(columns=keys).iloc[:-1]
            X = sample[self.get_relevant_attributes()].drop(
                columns=["value_valid_from", "days_until_next_change"]
            )
            y = sample["days_until_next_change"]
            reg = RandomForestClassifier(
                random_state=0, n_estimators=10, max_features="auto"
            )
            reg.fit(X, y)
            self.regressors[sample["key"].iloc[0]] = reg
            self.last_preds[sample["key"].iloc[0]] = [pd.Timestamp("1999-12-31"), 0]

        if self.use_hash:
            possible_cached_mapping.parent.mkdir(exist_ok=True, parents=True)
            with open(possible_cached_mapping, "wb") as f:
                pickle.dump(self.regressors, f)

    def _calculate_cache_name(self, data: pd.DataFrame) -> Path:
        hash_string = "".join(str(x) for x in [data.shape, data.head(2), data.tail(2)])
        hash_id = hashlib.md5(hash_string.encode("utf-8")).hexdigest()[:10]
        possible_cached_mapping = self.hash_location / hash_id
        return possible_cached_mapping

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
            # (it didnt if there was no traindata)
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
            self.last_preds[data_key_item] = [sample_value_valid_from, pred]

        else:
            pred = self.last_preds[data_key_item][1]

        return (
            first_day_to_predict
            <= (sample_value_valid_from + timedelta(pred))
            < first_day_to_predict + timedelta(timeframe)
        )
