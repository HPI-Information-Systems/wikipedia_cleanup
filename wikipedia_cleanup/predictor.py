import random
from abc import ABC, abstractmethod
from datetime import date, datetime, timedelta
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


class Predictor(ABC):
    @abstractmethod
    def fit(
        self, train_data: pd.DataFrame, last_day: datetime, keys: List[str]
    ) -> None:
        raise NotImplementedError()

    def get_relevant_attributes(self) -> List[str]:
        raise NotImplementedError()

    @abstractmethod
    def predict_timeframe(
        self,
        data_key: np.ndarray,
        additional_data: np.ndarray,
        columns: List[str],
        first_day_to_predict: date,
        timeframe: int,
    ) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def get_relevant_ids(self, identifier: Tuple) -> List[Tuple]:
        raise NotImplementedError()


class StaticPredictor(Predictor, ABC):
    def fit(
        self, train_data: pd.DataFrame, last_day: datetime, keys: List[str]
    ) -> None:
        pass

    def get_relevant_ids(self, identifier: Tuple) -> List[Tuple]:
        return []

    def get_relevant_attributes(self) -> List[str]:
        return []


class ZeroPredictor(StaticPredictor):
    def predict_timeframe(
        self,
        data_key: np.ndarray,
        additional_data: np.ndarray,
        columns: List[str],
        first_day_to_predict: date,
        timeframe: int,
    ) -> bool:
        return False


class OnePredictor(StaticPredictor):
    def predict_timeframe(
        self,
        data_key: np.ndarray,
        additional_data: np.ndarray,
        columns: List[str],
        first_day_to_predict: date,
        timeframe: int,
    ) -> bool:
        return True


class RandomPredictor(StaticPredictor):
    def __init__(self, p: float = 0.5):
        self.p = p

    def predict_timeframe(
        self,
        data_key: np.ndarray,
        additional_data: np.ndarray,
        columns: List[str],
        first_day_to_predict: date,
        timeframe: int,
    ) -> bool:
        return random.random() <= self.p


class MeanPredictor(ZeroPredictor):
    def predict_timeframe(
        self,
        data_key: np.ndarray,
        additional_data: np.ndarray,
        columns: List[str],
        first_day_to_predict: date,
        timeframe: int,
    ) -> bool:
        col_idx = columns.index("value_valid_from")
        pred = self.next_change(data_key[:, col_idx])
        if pred is None:
            return False
        return (
            first_day_to_predict <= pred <= first_day_to_predict + timedelta(timeframe)
        )

    def get_relevant_ids(self, identifier: Tuple) -> List[Tuple]:
        return [identifier]

    @staticmethod
    def next_change(time_series: np.ndarray) -> Optional[date]:
        if len(time_series) < 2:
            return None

        mean_time_to_change: np.timedelta64 = np.mean(
            time_series[1:] - time_series[0:-1]
        )
        return_value: np.datetime64 = time_series[-1] + mean_time_to_change
        return pd.to_datetime(return_value).date()
