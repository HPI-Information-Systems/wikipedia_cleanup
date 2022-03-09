from abc import ABC
from datetime import date, datetime
from typing import Callable, List, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from wikipedia_cleanup.predictor import Predictor


class BasicEnsemble(Predictor, ABC):
    def __init__(self, predictors: List[Predictor]):
        self._predictors = predictors

    def fit(
        self, train_data: pd.DataFrame, last_day: datetime, keys: List[str]
    ) -> None:
        for predictor in tqdm(self._predictors):
            predictor.fit(train_data.copy(), last_day, keys)

    def get_relevant_attributes(self) -> List[str]:
        all_relevant_attributes = set()
        for predictor in self._predictors:
            all_relevant_attributes |= set(predictor.get_relevant_attributes())
        return list(all_relevant_attributes)

    def get_relevant_ids(self, identifier: Tuple) -> List[Tuple]:
        get_relevant_ids = set()
        for predictor in self._predictors:
            get_relevant_ids |= set(predictor.get_relevant_ids(identifier))
        return list(get_relevant_ids)

    def _make_individual_predictions(
        self,
        data_key: np.ndarray,
        additional_data: np.ndarray,
        columns: List[str],
        first_day_to_predict: date,
        timeframe: int,
    ) -> List[bool]:
        return [
            predictor.predict_timeframe(
                data_key, additional_data, columns, first_day_to_predict, timeframe
            )
            for predictor in self._predictors
        ]

    def predict_timeframe(
        self,
        data_key: np.ndarray,
        additional_data: np.ndarray,
        columns: List[str],
        first_day_to_predict: date,
        timeframe: int,
    ) -> bool:
        raise NotImplementedError


class FunctionEnsemble(BasicEnsemble):
    def __init__(self, predictors: List[Predictor], aggregate_fct: Callable):
        super().__init__(predictors)
        self.aggregation = aggregate_fct

    def predict_timeframe(
        self,
        data_key: np.ndarray,
        additional_data: np.ndarray,
        columns: List[str],
        first_day_to_predict: date,
        timeframe: int,
    ) -> bool:
        predictions = self._make_individual_predictions(
            data_key, additional_data, columns, first_day_to_predict, timeframe
        )
        return self.aggregation(predictions)


class OrEnsemble(FunctionEnsemble):
    def __init__(self, predictors: List[Predictor]):
        super().__init__(predictors, any)


class AndEnsemble(FunctionEnsemble):
    def __init__(self, predictors: List[Predictor]):
        super().__init__(predictors, all)


class AverageEnsemble(FunctionEnsemble):
    def __init__(self, predictors: List[Predictor]):
        super().__init__(predictors, lambda x: sum(x) / len(x) >= 0.5)


class MajorityEnsemble(FunctionEnsemble):
    def __init__(self, predictors: List[Predictor]):
        super().__init__(
            predictors,
            lambda x: sum(single_pred >= 0.5 for single_pred in x) / len(x) >= 0.5,
        )
