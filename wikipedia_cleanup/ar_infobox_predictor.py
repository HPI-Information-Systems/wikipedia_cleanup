import collections
from datetime import date, datetime
from typing import Dict, FrozenSet, List, Set, Tuple

import numpy as np
import pandas as pd
from efficient_apriori import apriori
from tqdm.auto import tqdm

from wikipedia_cleanup.predictor import Predictor

tqdm.pandas()


class AssociationRulesInfoboxPredictor(Predictor):
    def __init__(
        self,
        min_support: float = 0.2,
        min_confidence: float = 0.8,
        transaction_freq: str = "D",
    ) -> None:
        super().__init__()
        self.min_support: float = min_support
        self.min_confidence: float = min_confidence
        self.transaction_freq: str = transaction_freq

    def fit(
        self, train_data: pd.DataFrame, last_day: datetime, keys: List[str]
    ) -> None:
        self.infobox_mapping: Dict[str, FrozenSet[str]] = (
            train_data.groupby("infobox_key")["property_name"]
            .apply(frozenset)
            .to_dict()
        )
        df = train_data[["infobox_key", "value_valid_from"]].copy()
        df["value_valid_from"] = pd.to_datetime(df["value_valid_from"])
        _, mined_rules = apriori(
            tuple(
                val
                for val in df.groupby(
                    pd.Grouper(key="value_valid_from", freq=self.transaction_freq)
                )["infobox_key"].progress_apply(frozenset)
                if val
            ),
            min_support=self.min_support,
            min_confidence=self.min_confidence,
            max_length=2,
        )
        rules: Dict[str, Set[str]] = collections.defaultdict(set)
        for rule in mined_rules:
            rules[rule.rhs[0]].add(rule.lhs[0])
        self.rules: Dict[str, FrozenSet[str]] = {
            k: frozenset(v) for k, v in rules.items()
        }

    def predict_timeframe(
        self,
        data_key: np.ndarray,
        additional_data: np.ndarray,
        columns: List[str],
        first_day_to_predict: date,
        timeframe: int,
    ) -> bool:
        if not (bool(len(additional_data)) and bool(len(data_key))):
            return False
        return (
            additional_data[:, columns.index("value_valid_from")]
            >= first_day_to_predict
        ).any()

    @staticmethod
    def get_relevant_attributes() -> List[str]:
        return ["value_valid_from", "infobox_key", "property_name"]

    def get_relevant_ids(self, identifier: Tuple) -> List[Tuple]:
        return [
            (infobox_key, prop_name)
            for infobox_key in self.rules.get(identifier[0], frozenset())
            for prop_name in self.infobox_mapping.get(infobox_key, frozenset())
        ]
