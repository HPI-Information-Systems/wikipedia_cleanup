import collections
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
        transaction_freq: str = "W",
    ) -> None:
        super().__init__()
        self.min_support: float = min_support
        self.min_confidence: float = min_confidence
        self.min_template_support: float = min_template_support
        self.transaction_freq: str = transaction_freq

    def fit(
        self, train_data: pd.DataFrame, last_day: datetime, keys: List[str]
    ) -> None:
        self.template_mapping: Dict[str, FrozenSet[str]] = (
            train_data.groupby("infobox_key")["template"].apply(frozenset).to_dict()
        )
        df = train_data[
            ["infobox_key", "value_valid_from", "template", "property_name"]
        ].copy()
        df["value_valid_from"] = pd.to_datetime(df["value_valid_from"])
        df = (
            df.groupby(
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
        lengths = df.apply(len)
        df = df[lengths >= lengths.sum() * self.min_template_support]
        rules: Dict[str, Dict[str, Set[str]]] = collections.defaultdict(
            lambda: collections.defaultdict(set)
        )
        for template, tl in tqdm(df.iteritems(), total=len(df)):
            _, mined_rules = apriori(
                tl,
                min_support=self.min_support,
                min_confidence=self.min_confidence,
                max_length=2,
            )
            for rule in mined_rules:
                rules[template][rule.rhs[0]].add(rule.lhs[0])
        self.rules: Dict[str, Dict[str, FrozenSet[str]]] = {
            template: {key: frozenset(value) for key, value in template_rules.items()}
            for template, template_rules in rules.items()
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
        template = data_key[-1, columns.index("template")]
        prop_name = data_key[-1, columns.index("property_name")]
        if template not in self.rules or prop_name not in self.rules[template]:
            return False
        lhss = self.rules[template][prop_name]
        offset = bisect_left(
            additional_data[:, columns.index("value_valid_from")], first_day_to_predict
        )
        for prop_name in additional_data[offset:, columns.index("property_name")]:
            if prop_name in lhss:
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
