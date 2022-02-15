import hashlib
import pickle
import random
from abc import ABC, abstractmethod
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import pandas as pd

from wikipedia_cleanup.utils import cache_directory


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


class RegressionPredictor(Predictor, ABC):
    def __init__(self):
        super().__init__()
        self.last_known_key = None
        self.last_known_timestamp = None
        self.last_known_prediction = False

    def predict_timeframe(
            self,
            data_key: np.ndarray,
            additional_data: np.ndarray,
            columns: List[str],
            first_day_to_predict: date,
            timeframe: int,
    ) -> bool:
        if not self._should_make_prediction(data_key, columns):
            return False
        key_column = columns.index("key")
        value_valid_from_idx = columns.index("value_valid_from")
        current_key = data_key[0, key_column]
        if current_key == self.last_known_key:
            if data_key[-1, value_valid_from_idx] == self.last_known_timestamp:
                return self.last_known_prediction

        pred = self._predict_next_change(data_key, columns)
        self.last_known_key = current_key
        self.last_known_timestamp = data_key[-1, value_valid_from_idx]
        self.last_known_prediction = (
                first_day_to_predict <= pred <= first_day_to_predict + timedelta(timeframe)
        )
        return self.last_known_prediction

    @abstractmethod
    def _predict_next_change(
            self, data_key: np.ndarray, columns: List[str]
    ) -> datetime:
        pass

    @abstractmethod
    def _should_make_prediction(self, data_key: np.ndarray, columns: List[str]):
        pass


class CachedPredictor(Predictor):
    def __init__(self, use_cache: bool = True) -> None:
        self.use_cache = use_cache
        self.hash_location = cache_directory() / self.__class__.__name__

    def fit(
            self, train_data: pd.DataFrame, last_day: datetime, keys: List[str]
    ) -> None:
        possible_cached_mapping = Path()
        if self.use_cache:
            possible_cached_mapping = self._calculate_cache_name(train_data)
            if self._load_cache(possible_cached_mapping):
                self._adjust_cache()
                return

        self._fit_classifier(train_data, last_day, keys)

        if self.use_cache:
            self._save_cache(possible_cached_mapping)
        self._adjust_cache()

    def _calculate_dependent_cache_name(self, data: pd.DataFrame) -> str:
        hash_string = (
            f"{data.shape},\n"
            f"{','.join([str(v) for v in data.columns])},\n"
            f"{','.join([str(v) for v in data.iloc[0]])},\n"
            f"{','.join([str(v) for v in data.iloc[-1]])},\n"
        )
        return hash_string

    def _calculate_cache_name(self, data: pd.DataFrame):
        hash_string = self._calculate_dependent_cache_name(data)
        hash_id = hashlib.md5(hash_string.encode("utf-8")).hexdigest()[:20]
        possible_cached_mapping = self.hash_location / hash_id
        return possible_cached_mapping

    @abstractmethod
    def _fit_classifier(
            self, train_data: pd.DataFrame, last_day: datetime, keys: List[str]
    ):
        pass

    @abstractmethod
    def _load_cache_file(self, file_object: Any) -> bool:
        pass

    def _load_cache(self, possible_cached_mapping: Path) -> bool:
        try:
            if possible_cached_mapping.exists():
                print(f"Cache found. Loading from {possible_cached_mapping.name}.")
                with open(possible_cached_mapping, "rb") as f:
                    return self._load_cache_file(f)
            else:
                print("No cache found, recalculating model.")
        except EOFError:
            print("Caching failed, recalculating model.")
        return False

    @abstractmethod
    def _get_cache_object(self) -> Any:
        pass

    def _save_cache(self, possible_cached_mapping: Path) -> None:
        possible_cached_mapping.parent.mkdir(exist_ok=True, parents=True)
        print(f"Saving cache to {possible_cached_mapping.name}.")
        with open(possible_cached_mapping, "wb") as f:
            pickle.dump(self._get_cache_object(), f)

    def _adjust_cache(self) -> None:
        pass


class MeanPredictor(StaticPredictor, RegressionPredictor):
    def _predict_next_change(self, data: np.ndarray, columns: List[str]) -> datetime:
        changes = data[:, columns.index("value_valid_from")]
        mean_time_to_change: datetime = np.mean(
            changes[1:] - changes[0:-1]
        )  # type: ignore
        return changes[-1] + mean_time_to_change

    def _should_make_prediction(self, data_key: np.ndarray, columns: List[str]):
        return len(data_key) >= 2

    def get_relevant_ids(self, identifier: Tuple) -> List[Tuple]:
        return [identifier]


class LastChangePredictor(MeanPredictor):
    def _predict_next_change(self, data: np.ndarray, columns: List[str]) -> datetime:
        changes = data[:, columns.index("value_valid_from")]
        return changes[-1] + (changes[-1] - changes[-2])

    def _should_make_prediction(self, data_key: np.ndarray, columns: List[str]):
        return len(data_key) >= 2

    def get_relevant_ids(self, identifier: Tuple) -> List[Tuple]:
        return [identifier]
