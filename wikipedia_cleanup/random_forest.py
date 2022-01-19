import hashlib
import pickle
import re
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from wikipedia_cleanup.predictor import Predictor


class RandomForestPredictor(Predictor):

    def __init__(self, use_cache: bool = True) -> None:
        super().__init__()
        self.regressors: dict = {}
        self.hash_location = Path("cache") / self.__class__.__name__
        self.use_hash = use_cache    

    @staticmethod
    def get_relevant_ids(identifier: Tuple) -> List[Tuple]:
        return []
    
    @staticmethod
    def get_relevant_attributes() -> List[str]:
        return [
            "value_valid_from",
            "day_of_year", # values from feature engineering
            "day_of_month",
            "day_of_week", 
            "month_of_year",
            "quarter_of_year",
            "is_month_start", 
            "is_month_end",
            "is_quarter_start",
            "days_since_last_change",
            "days_since_last_2_changes",
            "days_since_last_3_changes",
            "days_between_last_and_2nd_to_last_change",
            "days_until_next_change",
            "mean_change_frequency_all_previous",
            "mean_change_frequency_last_3"
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
        ilocs = train_data.groupby(keys,sort=False).count()["value_valid_from"]
        # Assumption: data is already sorted by keys so no further sorting needs to be done
        start=0
        for iloc in tqdm(ilocs):
            sample = train_data.iloc[start:start+iloc]
            start+=iloc
            sample = sample.drop(columns=keys)
            X = sample.drop(columns=['days_until_next_change','value_valid_from', 'key'])
            y = sample['days_until_next_change']
            reg = RandomForestClassifier(random_state=0, n_estimators=100, max_features="auto")
            reg.fit(X, y)
            self.regressors[sample['key'].iloc[0]] = reg
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
        data_key: pd.DataFrame,
        additional_data: pd.DataFrame,
        current_day: date,
        timeframe: int,
    ) -> bool:
        sample = data_key.tail(1)
        reg = self.regressors[data_key['key'].iloc[0]]
        X_test = sample[self.get_relevant_attributes()].drop(columns=['value_valid_from', 'days_until_next_change'])
        pred = round(reg.predict(X_test)[0])
        return pd.to_datetime(current_day) <= (sample['value_valid_from'].iloc[0] + timedelta(days=pred)) <= pd.to_datetime(current_day) + timedelta(timeframe)
