import collections
import math
from bisect import bisect_left
from datetime import date, datetime
from typing import Dict, FrozenSet, List, Set, Tuple

import numpy as np
import pandas as pd
from efficient_apriori import apriori
from tqdm.auto import tqdm

from wikipedia_cleanup.predictor import Predictor

tqdm.pandas()


class AssociationRulesTemplatePredictor(Predictor):
    def __init__(
        self,
        min_support: float = 0.05,
        min_confidence: float = 0.8,
        min_template_support: float = 0.001,
        val_size: float = 0.2,
        val_precision: float = 0.8,
        transaction_freq: str = "W",
    ) -> None:
        super().__init__()
        self.min_support: float = min_support
        self.min_confidence: float = min_confidence
        self.min_template_support: float = min_template_support
        self.transaction_freq: str = transaction_freq
        self.val_size: float = val_size
        self.val_precision: float = val_precision

    def transform_data(self, train_data: pd.DataFrame) -> pd.Series:
        return (
            train_data[["infobox_key", "value_valid_from", "template", "property_name"]]
            .groupby(
                [
                    "infobox_key",
                    "template",
                    pd.Grouper(key="value_valid_from", freq=self.transaction_freq),
                ]
            )["property_name"]
            .progress_apply(frozenset)
            .groupby("template")
            .progress_apply(tuple)
        )

    def precision(
        self, transactions: Tuple[FrozenSet[str], ...], rhs: str, lhs: str
    ) -> float:
        true_positives = 0
        false_positives = 0
        for transaction in transactions:
            if lhs in transaction:
                if rhs in transaction:
                    true_positives += 1
                else:
                    false_positives += 1
        denominator = true_positives + false_positives
        if not denominator:
            return float("nan")
        return true_positives / denominator

    def fit(
        self, train_data: pd.DataFrame, last_day: datetime, keys: List[str]
    ) -> None:
        self.template_mapping: Dict[str, FrozenSet[str]] = (
            train_data.groupby("infobox_key")["template"].apply(frozenset).to_dict()
        )
        train_data["value_valid_from"] = pd.to_datetime(train_data["value_valid_from"])
        train_size = math.floor(len(train_data) * (1 - self.val_size))
        train_df = self.transform_data(train_data.iloc[:train_size])
        val_df = self.transform_data(train_data.iloc[train_size:])
        del train_data
        train_df = train_df.reindex(val_df.index).dropna()
        lengths = train_df.apply(len)
        train_df = train_df[lengths >= lengths.sum() * self.min_template_support]
        del lengths
        rules: Dict[str, Dict[str, Set[str]]] = collections.defaultdict(
            lambda: collections.defaultdict(set)
        )
        for template, tl in tqdm(train_df.iteritems(), total=len(train_df)):
            _, mined_rules = apriori(
                tl,
                min_support=self.min_support,
                min_confidence=self.min_confidence,
                max_length=2,
            )
            for rule in mined_rules:
                rules[template][rule.rhs[0]].add(rule.lhs[0])
        self.rules: Dict[str, Dict[str, FrozenSet[str]]] = {
            template: {
                rhs: frozenset(
                    {
                        lhs
                        for lhs in lhss
                        if self.precision(val_df[template], rhs, lhs)
                        >= self.val_precision
                    }
                )
                for rhs, lhss in template_rules.items()
            }
            for template, template_rules in rules.items()
        }
        self.rules = {
            template: {rhs: lhss for rhs, lhss in template_rules.items() if lhss}
            for template, template_rules in self.rules.items()
        }
        self.rules = {
            template: template_rules
            for template, template_rules in self.rules.items()
            if template_rules
        }

    def predict_timeframe(
        self,
        data_key: np.ndarray,
        additional_data: np.ndarray,
        columns: List[str],
        first_day_to_predict: date,
        timeframe: int,
    ) -> bool:
        if not len(data_key) or not len(additional_data):
            return False
        template = data_key[-1, columns.index("template")]
        if template not in self.rules:
            return False
        rhs = data_key[-1, columns.index("property_name")]
        if rhs not in self.rules[template]:
            return False
        lhss = self.rules[template][rhs]
        offset = bisect_left(
            additional_data[:, columns.index("value_valid_from")], first_day_to_predict
        )
        for lhs in additional_data[offset:, columns.index("property_name")]:
            if lhs in lhss:
                return True
        return False

    @staticmethod
    def get_relevant_attributes() -> List[str]:
        return ["value_valid_from", "infobox_key", "template", "property_name"]

    def get_relevant_ids(self, identifier: Tuple) -> List[Tuple]:
        templates = self.template_mapping.get(identifier[0], frozenset())
        return [
            (identifier[0], prop_name)
            for template in templates
            for prop_name in self.rules.get(template, {}).get(
                identifier[1], frozenset()
            )
        ]
