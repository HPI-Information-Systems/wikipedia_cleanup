import collections
from datetime import date, datetime
from typing import Dict, FrozenSet, List, Set, Tuple

import numpy as np
import pandas as pd
from efficient_apriori import apriori
from tqdm.auto import tqdm

from wikipedia_cleanup.ar.utils import precision, train_val_split
from wikipedia_cleanup.predictor import Predictor

tqdm.pandas()


def transform_data(
    data: pd.DataFrame, transaction_freq: str
) -> Tuple[FrozenSet[Tuple[str, str]], ...]:
    return tuple(
        val
        for val in data.groupby(
            [pd.Grouper(key="value_valid_from", freq=transaction_freq)]
        )["key"].progress_apply(frozenset)
        if val
    )


class AssociationRulesPredictor(Predictor):
    def __init__(
        self,
        min_support: float = 0.2,
        min_confidence: float = 0.8,
        val_size: float = 0.2,
        val_precision: float = 0.8,
        transaction_freq: str = "D",
    ) -> None:
        super().__init__()
        self.min_support: float = min_support
        self.min_confidence: float = min_confidence
        self.val_size: float = val_size
        self.val_precision: float = val_precision
        self.transaction_freq: str = transaction_freq

    def fit(
        self, train_data: pd.DataFrame, last_day: datetime, keys: List[str]
    ) -> None:
        train_data["value_valid_from"] = pd.to_datetime(train_data["value_valid_from"])
        train_df, val_df = train_val_split(
            train_data[["value_valid_from", "key"]].sort_values("value_valid_from"),
            self.val_size,
        )
        del train_data
        train_tl = transform_data(train_df, self.transaction_freq)
        del train_df
        val_tl = transform_data(val_df, self.transaction_freq)
        del val_df
        _, mined_rules = apriori(
            train_tl,
            min_support=self.min_support,
            min_confidence=self.min_confidence,
            max_length=2,
        )
        rules: Dict[Tuple[str, str], Set[Tuple[str, str]]] = collections.defaultdict(
            set
        )
        for rule in mined_rules:
            rhs = rule.rhs[0]
            lhs = rule.lhs[0]
            if precision(val_tl, rhs, lhs) >= self.val_precision:
                rules[rhs].add(lhs)
        self.rules: Dict[Tuple[str, str], FrozenSet[Tuple[str, str]]] = {
            rhs: frozenset(lhss) for rhs, lhss in rules.items()
        }

    def predict_timeframe(
        self,
        data_key: np.ndarray,
        additional_data: np.ndarray,
        columns: List[str],
        first_day_to_predict: date,
        timeframe: int,
    ) -> bool:
        if not len(additional_data):
            return False
        return (
            additional_data[-1:, columns.index("value_valid_from")]
            >= first_day_to_predict
        )

    @staticmethod
    def get_relevant_attributes() -> List[str]:
        return ["value_valid_from", "property_name"]

    def get_relevant_ids(  # type: ignore
        self, identifier: Tuple[str, str]
    ) -> List[Tuple[str, str]]:
        return list(self.rules.get(identifier, set()))
