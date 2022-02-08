import collections
from datetime import date, datetime
from typing import Dict, FrozenSet, List, Tuple

import numpy as np
import pandas as pd
from efficient_apriori import apriori

from wikipedia_cleanup.predictor import Predictor


class AssociationRulesPredictor(Predictor):
    def __init__(self, min_support: float = 0.2, min_confidence: float = 0.8) -> None:
        super().__init__()
        self.min_support: float = min_support
        self.min_confidence: float = min_confidence

    def fit(
        self, train_data: pd.DataFrame, last_day: datetime, keys: List[str]
    ) -> None:
        df = train_data[["value_valid_from", "key"]].copy()
        df["value_valid_from"] = pd.to_datetime(df["value_valid_from"])
        tl = tuple(
            t
            for t in df.groupby([pd.Grouper(key="value_valid_from", freq="D")])[
                "key"
            ].apply(frozenset)
            if len(t)
        )
        _, mined_rules = apriori(
            tl,
            min_support=self.min_support,
            min_confidence=self.min_confidence,
            max_length=2,
        )
        rules = collections.defaultdict(set)
        for rule in mined_rules:
            rules[rule.rhs[0]].add(rule.lhs[0])
        self.rules: Dict[Tuple, FrozenSet[Tuple]] = dict(rules)

    def predict_timeframe(
        self,
        data_key: np.ndarray,
        additional_data: np.ndarray,
        columns: List[str],
        first_day_to_predict: date,
        timeframe: int,
    ) -> bool:
        """
        Example arguments:

        data_key = np.array(
            [
                [
                    "297391749-0",
                    1003178,
                    "label",
                    datetime.date(2009, 8, 20),
                    ("297391749-0", "label"),
                ],
                [
                    "297391749-0",
                    1003178,
                    "label",
                    datetime.date(2011, 2, 20),
                    ("297391749-0", "label"),
                ],
                [
                    "297391749-0",
                    1003178,
                    "label",
                    datetime.date(2014, 5, 9),
                    ("297391749-0", "label"),
                ],
            ]
        )

        additional_data = []

        columns = ['infobox_key', 'page_id', 'property_name', 'value_valid_from', 'key']

        first_day_to_predict = 2018-09-01

        timeframe = 1
        """
        return (
            additional_data[:, columns.index("value_valid_from")]
            >= first_day_to_predict
        ).any()

    @staticmethod
    def get_relevant_attributes() -> List[str]:
        return ["value_valid_from"]

    def get_relevant_ids(self, identifier: Tuple) -> List[Tuple]:
        return list(self.rules.get(identifier, set()))
