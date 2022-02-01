import collections
import hashlib
import pickle
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from efficient_apriori import apriori
from scipy.sparse import csr_matrix

from wikipedia_cleanup.predictor import Predictor
from wikipedia_cleanup.utils import cache_directory


class AssociationRulesCorrelationPredictor(Predictor):
    NUM_REQUIRED_CHANGES = 5
    MAX_ALLOWED_PROPERTIES = 55  # Set via boxplot whisker for all links (53)
    # TODO justify choice
    PERCENT_ALLOWED_MISMATCHES = 0.05

    def __init__(self, allowed_change_delay: int = 3, use_cache: bool = True) -> None:
        super().__init__()
        self.related_properties_lookup: dict = {}
        self.hash_location = cache_directory() / self.__class__.__name__

        self.use_hash = use_cache
        # TODO justify choice
        self.delay_range = allowed_change_delay

    @staticmethod
    def _get_links(train_data: pd.DataFrame) -> Dict[str, List[str]]:
        regex_str = '\\[\\[((?:\\w+:)?[^<>\\[\\]"\\|]+)(?:\\|[^\\n\\]]+)?\\]\\]'
        regex = re.compile(regex_str)

        related_page_index = {}

        grouped_infoboxes = train_data.groupby("page_title")["current_value"].unique()

        for key, row in grouped_infoboxes.iteritems():
            related_page_index[key] = list(
                set(
                    match.groups()[0].strip()
                    for value in row
                    if value
                    for match in regex.finditer(value)
                    if not match.groups()[0].startswith(("Image:", "File:"))
                )
            )

        return related_page_index

    @staticmethod
    def _create_time_series(a: Any, duration: int) -> csr_matrix:
        series = np.zeros(duration)
        uniques, counts = np.unique(a, return_counts=True)
        series[uniques] = counts
        return csr_matrix(series)

    def _load_cache(self, possible_cached_mapping: Path) -> bool:
        try:
            if possible_cached_mapping.exists():
                print(f"Cached model found, loading from {possible_cached_mapping}")
                with open(possible_cached_mapping, "rb") as f:
                    self.related_properties_lookup = pickle.load(f)
                return True
            else:
                print("No cache found, recalculating model.")
        except EOFError:
            print("Caching failed, recalculating model.")
        return False

    def fit(
        self, train_data: pd.DataFrame, last_day: datetime, keys: List[str]
    ) -> None:
        if self.use_hash:
            possible_cached_mapping = self._calculate_cache_name(train_data)
            if self._load_cache(possible_cached_mapping):
                return

        df = train_data[["value_valid_from", "infobox_key", "key"]]
        rules = collections.defaultdict(set)
        transaction_list = df.groupby(
            ["infobox_key", pd.Grouper(key="value_valid_from", freq="D")]
        )["key"].apply(frozenset)
        for key in transaction_list.index.get_level_values(0).unique():
            tl = tuple(transaction_list.loc[key].tolist())
            _, mined_rules = apriori(
                tl,
                min_support=0.1,  # usually min_support = 0.2
                min_confidence=0.9,  # usually min_confidence = 0.8
                max_length=2,
            )
            for rule in mined_rules:
                rules[rule.rhs[0]].add(rule.lhs[0])

        self.related_properties_lookup = dict(rules)

        """ this runs slow:
        grouped = df.groupby(["infobox_key"])
        def get_transactions(ibk):
            return tuple(
                s
                for s in grouped.get_group(ibk)
                .groupby(pd.Grouper(key="value_valid_from", freq="D"))["key"]
                .apply(frozenset)
                .tolist()
                if len(s)
            )
        for key in tqdm(df["infobox_key"].unique()):
            tl = get_transactions(key)
            _, mined_rules = apriori(
                tl, min_support=0.1, min_confidence=0.9, max_length=2
            )
            for rule in mined_rules:
                rules[rule.rhs[0]].add(rule.lhs[0])
        """

        if self.use_hash:
            possible_cached_mapping.parent.mkdir(exist_ok=True, parents=True)
            print(f"Saving cache to {possible_cached_mapping.name}.")
            with open(possible_cached_mapping, "wb") as f:
                pickle.dump(self.related_properties_lookup, f)

    @staticmethod
    def _get_related_page_mapping(links, related_page_index):
        page_to_related_pages = {}
        for page_title, related_pages in related_page_index.items():
            found_related_pages = []
            for related_page in related_pages:
                if (
                    related_page in links
                    and page_title in related_page_index[related_page]
                    and related_pages != page_title
                ):
                    found_related_pages.append(related_page)
            page_to_related_pages[page_title] = found_related_pages
        return page_to_related_pages

    @staticmethod
    def _find_working_links(min_support_groups, related_page_index):
        links = set()
        for related_pages in related_page_index.values():
            for related_page in related_pages:
                links.add(related_page)
        pandas_links = pd.Series(list(links))
        links_found = set(
            pandas_links[
                pandas_links.isin(
                    min_support_groups.reset_index("page_title")["page_title"]
                )
            ]
        )
        return links_found

    def _calculate_cache_name(self, data: pd.DataFrame) -> Path:
        hash_string = (
            f"{data.shape},\n"
            f"{','.join([str(v) for v in data.columns])},\n"
            f"{','.join([str(v) for v in data.iloc[0]])},\n"
            f"{','.join([str(v) for v in data.iloc[-1]])},\n"
            f"{self.NUM_REQUIRED_CHANGES},\n"
            f"{self.MAX_ALLOWED_PROPERTIES},\n"
            f"{self.PERCENT_ALLOWED_MISMATCHES},\n"
            f"{self.delay_range}"
        )
        hash_id = hashlib.md5(hash_string.encode("utf-8")).hexdigest()[:20]
        possible_cached_mapping = self.hash_location / hash_id
        return possible_cached_mapping

    def predict_timeframe(
        self,
        data_key: np.ndarray,
        additional_data: np.ndarray,
        columns: List[str],
        first_day_to_predict: date,
        timeframe: int,
    ) -> bool:
        # pass in which data point we are supposed to predict

        # We can't really deal with daily predictions
        #  or timeframes that are less than the self.DELAY_RANGE

        # Check how many changes/ any changes we have inside the
        # (timeframe - delay range) area. If yes, return true, else false
        # Maybe decrease the delay range here or see how many related
        #  properties have changes here to increase precision, has to be tested
        if len(additional_data) == 0:
            return False
        col_idx = columns.index("value_valid_from")
        return additional_data[-1:, col_idx] >= first_day_to_predict

    @staticmethod
    def get_relevant_attributes() -> List[str]:
        return [
            "page_id",
            "infobox_key",
            "page_title",
            "property_name",
            "value_valid_from",
            "current_value",
        ]

    def get_relevant_ids(self, identifier: Tuple) -> List[Tuple]:
        if identifier not in self.related_properties_lookup.keys():
            return [identifier]
        return self.related_properties_lookup[identifier]
