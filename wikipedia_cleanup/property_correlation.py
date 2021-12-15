from datetime import date, datetime
from typing import List

import pandas as pd

from wikipedia_cleanup.baseline import Predictor


class PropertyCorrelationPredictor(Predictor):
    def fit(self, train_data: pd.DataFrame, last_day: datetime) -> None:
        raise NotImplementedError()

    @staticmethod
    def get_relevant_attributes() -> List[str]:
        return [
            "page_id",
            "infobox_key",
            "page_title",
            "property_name",
            "previous_value",
            "current_value",
            "value_valid_from",
        ]

    def predict_timeframe(
        self, data: pd.DataFrame, current_day: date, timeframe: int
    ) -> bool:
        raise NotImplementedError()

    def get_relevant_ids(self, identifier: str) -> List[str]:
        raise NotImplementedError()
