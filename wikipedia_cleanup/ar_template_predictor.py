import collections
from datetime import date, datetime
from typing import Dict, FrozenSet, List, Set, Tuple

import numpy as np
import pandas as pd
from efficient_apriori import apriori
from tqdm.auto import tqdm

from wikipedia_cleanup.predictor import Predictor


class AssociationRulesTemplatePredictor(Predictor):
    def __init__(
        self,
        min_support: float = 0.1,
        min_confidence: float = 0.9,
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
            .apply(frozenset)
            .groupby("template")
            .apply(tuple)
        )
        rules: Dict[str, Dict[str, Set[str]]] = collections.defaultdict(
            lambda: collections.defaultdict(set)
        )
        for template, tl in tqdm(df.iteritems(), total=len(df)):
            if len(tl) <= df.apply(len).sum() * self.min_template_support:
                continue
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
        if not (bool(len(data_key)) and bool(len(additional_data))):
            return False
        template = data_key[-1, columns.index("template")]
        prop_name = data_key[-1, columns.index("property_name")]
        lhss = self.rules.get(template, {}).get(prop_name, frozenset())
        if not lhss:
            return False
        relevant_rows = additional_data[
            additional_data[:, columns.index("value_valid_from")]
            >= first_day_to_predict
        ]
        return np.isin(
            relevant_rows[:, columns.index("property_name")], tuple(lhss)
        ).any()

    @staticmethod
    def get_relevant_attributes() -> List[str]:
        return ["value_valid_from", "template", "property_name"]

    def get_relevant_ids(self, identifier: Tuple) -> List[Tuple]:
        templates = self.template_mapping.get(identifier[0], frozenset())
        return [
            (identifier[0], prop_name)
            for template in templates
            for prop_name in self.rules.get(template, {}).get(
                identifier[1], frozenset()
            )
        ]
