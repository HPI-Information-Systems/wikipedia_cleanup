import itertools
import pickle
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from tqdm.auto import tqdm

from wikipedia_cleanup.predictor import CachedPredictor


class BaselineMinPrecision(CachedPredictor):
    def __init__(
        self,
        use_cache: bool = True,
    ) -> None:
        super().__init__(use_cache)
        self.classifier={}

    def get_relevant_ids(self, identifier: Tuple) -> List[Tuple]:
        return []

    def get_relevant_attributes(self) -> List[str]:
        return [
            "value_valid_from"
        ]

    def _load_cache_file(self, file_object: Any) -> bool:
        self.classifier = pickle.load(file_object)
        return True

    def _get_cache_object(self) -> Any:
        return self.classifier

    def _fit_classifier(
        self, train_data: pd.DataFrame, last_day: datetime, keys: List[str]
    ) -> None:
        
        bins = pd.date_range(
            last_day - timedelta(364),
            last_day,
        )
        total_days = len(bins)
        bins = pd.cut(train_data["value_valid_from"].astype(np.datetime64), bins, labels=False)
        train_data['bins'] = bins
        train_data = train_data[train_data['bins'].notna()]
        train_data['bins'] = train_data['bins'].astype(int)

        def _create_time_series(a, duration):
            series = np.zeros(duration)
            uniques, counts = np.unique(a, return_counts=True)

            series[uniques] = counts
            return series

        def _aggregate_labels(labels: np.ndarray, n: int) -> np.ndarray:
            if n == 1:
                return labels
            if 365 % n != 0:
                cut_labels = labels[:, : -(365 % n)]
            else:
                cut_labels = labels
            cut_labels = cut_labels.reshape((labels.shape[0], -1, n))
            return np.any(cut_labels, axis=2)

        train_data_grp = train_data.groupby(['key'])['bins'].apply(_create_time_series, duration=total_days)
        x = np.vstack(train_data_grp.to_numpy())
        
        self.classifier[1]=set(train_data_grp.index[x.sum(axis=1)>=365 * 0.85])
        self.classifier[7]=set(train_data_grp.index[_aggregate_labels(x, 7).sum(axis=1)>=365//7 * 0.85])
        self.classifier[30]=set(train_data_grp.index[_aggregate_labels(x, 30).sum(axis=1)>=365//30 * 0.85])
        self.classifier[365]=set(train_data_grp.index[_aggregate_labels(x, 365).sum(axis=1)>=1])

    def predict_timeframe(
        self,
        data_key: np.ndarray,
        additional_data: np.ndarray,
        columns: List[str],
        first_day_to_predict: date,
        timeframe: int,
    ) -> Any:  # can be bool for actual predictions or float for confidence scores
        key_col=columns.index("key")
        if len(data_key)==0:
            return False
        current_key = data_key[0, key_col]
        return current_key in self.classifier[timeframe]
