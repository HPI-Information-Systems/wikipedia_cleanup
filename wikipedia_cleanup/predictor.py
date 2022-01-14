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

    @staticmethod
    def get_relevant_attributes() -> List[str]:
        raise NotImplementedError()

    @abstractmethod
    def predict_timeframe(
        self,
        data_key: pd.DataFrame,
        additional_data: pd.DataFrame,
        current_day: date,
        timeframe: int,
    ) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def get_relevant_ids(self, identifier: Tuple) -> List[Tuple]:
        raise NotImplementedError()


class ZeroPredictor(Predictor):
    def fit(
        self, train_data: pd.DataFrame, last_day: datetime, keys: List[str]
    ) -> None:
        pass

    def predict_timeframe(
        self,
        data_key: pd.DataFrame,
        additional_data: pd.DataFrame,
        current_day: date,
        timeframe: int,
    ) -> bool:
        return False

    def get_relevant_ids(self, identifier: Tuple) -> List[Tuple]:
        return [identifier]

    @staticmethod
    def get_relevant_attributes() -> List[str]:
        return []


class OnePredictor(ZeroPredictor):
    def predict_timeframe(
        self,
        data_key: pd.DataFrame,
        additional_data: pd.DataFrame,
        current_day: date,
        timeframe: int,
    ) -> bool:
        return True


class RandomPredictor(ZeroPredictor):
    def __init__(self, p: float = 0.5):
        self.p = p

    def predict_timeframe(
        self,
        data_key: pd.DataFrame,
        additional_data: pd.DataFrame,
        current_day: date,
        timeframe: int,
    ) -> bool:
        return random.random() <= self.p


class MeanPredictor(ZeroPredictor):
    def predict_timeframe(
        self,
        data_key: pd.DataFrame,
        additional_data: pd.DataFrame,
        current_day: date,
        timeframe: int,
    ) -> bool:
        pred = self.next_change(data_key)
        if pred is None:
            return False
        return current_day <= pred <= current_day + timedelta(timeframe)

    @staticmethod
    def next_change(time_series: pd.DataFrame) -> Optional[date]:
        previous_change_timestamps = time_series["value_valid_from"].to_numpy()
        if len(previous_change_timestamps) < 2:
            return None

        mean_time_to_change: np.timedelta64 = np.mean(
            previous_change_timestamps[1:] - previous_change_timestamps[0:-1]
        )
        return_value: np.datetime64 = (
            previous_change_timestamps[-1] + mean_time_to_change
        )
        return pd.to_datetime(return_value).date()