from datetime import date, datetime
from typing import List, Tuple

import numpy as np
import pandas as pd

from wikipedia_cleanup.predictor import Predictor


class EnsemblePredictor(Predictor):
    def __init__(self, predictors: List[Predictor]):
        self._predictors = predictors

    def fit(
        self, train_data: pd.DataFrame, last_day: datetime, keys: List[str]
    ) -> None:
        for predictor in self._predictors:
            predictor.fit(train_data, last_day, keys)

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

    def predict_timeframe(
        self,
        data_key: np.ndarray,
        additional_data: np.ndarray,
        columns: List[str],
        first_day_to_predict: date,
        timeframe: int,
    ) -> bool:
        predictions = [
            predictor.predict_timeframe(
                data_key, additional_data, columns, first_day_to_predict, timeframe
            )
            for predictor in self._predictors
        ]
        return any(predictions)
