import hashlib
import pickle
import re
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from line_profiler_pycharm import profile
from scipy.sparse import csr_matrix, vstack
from sklearn.neighbors import NearestNeighbors
from tqdm.auto import tqdm

from wikipedia_cleanup.predictor import Predictor


class PropertyCorrelationPredictor(Predictor):
    def __init__(self, allowed_change_delay: int = 3, use_cache: bool = True) -> None:
        # TODO justify the 3 here
        self.DELAY_RANGE = allowed_change_delay
        self.related_properties_lookup = {}
        self.use_hash = use_cache
        self.hash_location = Path("")
        super().__init__()

    @staticmethod
    def _get_links(train_data: pd.DataFrame):
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
    def _create_time_series(a, duration):
        series = np.zeros(duration)
        uniques, counts = np.unique(a, return_counts=True)
        series[uniques] = counts
        return csr_matrix(series)

    @staticmethod
    def _sparse_time_series_conversion(train_data: pd.DataFrame):
        bins = pd.date_range(
            train_data["value_valid_from"].min().date(),
            train_data["value_valid_from"].max().date() + timedelta(1),
        )
        total_days = len(bins)
        bins = pd.cut(train_data["value_valid_from"], bins, labels=False)
        train_data["bin_idx"] = bins

        num_required_changes = 5
        groups = train_data.groupby(["infobox_key", "property_name"])
        min_support_groups = train_data[
            groups["bin_idx"].transform("count") > num_required_changes
        ].groupby(["infobox_key", "page_id", "page_title", "property_name"])
        min_support_groups = min_support_groups["bin_idx"].apply(
            PropertyCorrelationPredictor._create_time_series, duration=total_days
        )
        return min_support_groups

    @staticmethod
    def calculate_hash(data: pd.DataFrame) -> str:
        hash_string = "".join(str(x) for x in [data.shape, data.head(2), data.tail(2)])
        hash_id = hashlib.md5(hash_string.encode("utf-8")).hexdigest()[:10]
        return hash_id

    def fit(self, train_data: pd.DataFrame, last_day: datetime) -> None:
        if self.use_hash:
            try:
                hash_string = "".join(
                    str(x)
                    for x in [train_data.shape, train_data.head(2), train_data.tail(2)]
                )
                hash_id = hashlib.md5(hash_string.encode("utf-8")).hexdigest()[:10]
                possible_cached_mapping = self.hash_location / hash_id
                if (possible_cached_mapping).exists():
                    print(f"Cached model found, loading from {possible_cached_mapping}")
                    with open(possible_cached_mapping, "rb") as f:
                        self.related_properties_lookup = pickle.load(f)
                    return
                else:
                    print("No cache found, recalculating model.")
            except:
                print("Caching failed, recalculating model.")

        related_page_index = self._get_links(train_data)

        def percentage_manhatten_adaptive_time_lag(arr1, arr2):
            arr1 = arr1.toarray()
            arr2 = arr2.toarray()
            max_changes = arr1.sum()
            mask = np.nonzero(arr1)
            error = 0

            for idx in mask[1]:
                needed_num_changes = arr1[0, idx]
                for off in range(
                    -min(self.DELAY_RANGE, idx),
                    min(self.DELAY_RANGE, arr2.shape[1] - idx),
                ):
                    used_changes = min(needed_num_changes, arr2[0, idx + off])
                    arr2[0, idx + off] -= used_changes
                    needed_num_changes -= used_changes
                error += needed_num_changes

            return error / max_changes

        def percentage_manhatten_adaptive_time_lag_symmetric(arr1, arr2):
            return max(
                percentage_manhatten_adaptive_time_lag(arr1, arr2),
                percentage_manhatten_adaptive_time_lag(arr2, arr1),
            )

        min_support_groups = self._sparse_time_series_conversion(train_data)

        page_id_groups = min_support_groups.reset_index()
        page_id_groups = page_id_groups.groupby(["page_title"])[
            ["property_name", "bin_idx", "infobox_key"]
        ].agg(list)

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

        filtered_related_page_index = {}
        for page_title, related_pages in related_page_index.items():
            found_related_pages = []
            for related_page in related_pages:
                if (
                    related_page in links_found
                    and page_title in related_page_index[related_page]
                    and related_pages != page_title
                ):
                    found_related_pages.append(related_page)
            filtered_related_page_index[page_title] = found_related_pages

        max_dist = 0.05

        @profile
        def hihi():
            same_infoboxes = []
            matches = {}
            for key, row in tqdm(page_id_groups.iterrows(), total=len(page_id_groups)):
                if len(row[1]) > 1:
                    for related_key, related_row in page_id_groups.loc[
                        filtered_related_page_index[key]
                    ].iterrows():
                        row[0].extend(related_row[0])
                        row[1].extend(related_row[1])
                        row[2].extend(related_row[2])
                    # TODO cap the number of entities that we add to a page to 50 total entities or 50 entities that get linked?
                    input_data = vstack(row[1])
                    neigh = NearestNeighbors(
                        radius=max_dist,
                        metric=percentage_manhatten_adaptive_time_lag_symmetric,
                    )
                    neigh.fit(input_data)
                    neighbor_indices = neigh.radius_neighbors(return_distance=False)
                    for i, neighbors in enumerate(neighbor_indices):
                        infobox = row[2][i]
                        if len(neighbors) > 0:
                            infobox_keys = np.array(row[2])[neighbors]
                            same_infobox = infobox_keys == infobox
                            same_infoboxes.append(same_infobox)

                            property_names = np.array(row[0])[neighbors]
                            match = list(zip(infobox_keys, property_names))
                            match.append((infobox, row[0][i]))
                            matches[(infobox, row[0][i])] = match
            return matches

        self.related_properties_lookup = hihi()
        if self.use_hash:
            with open(possible_cached_mapping, "wb") as f:
                pickle.dump(self.related_properties_lookup, f)

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

    def predict_timeframe(
        self, data: pd.DataFrame, current_day: date, timeframe: int
    ) -> bool:
        # pass in which data point we are supposed to predict
        unique = data["property_name"].nunique()
        if unique > 2:
            print(unique)
        # print(data['property_name'].nunique())
        self.test = (data, current_day, timeframe)

        return True

        # We can't really deal with daily predictions or timeframes that are less than the self.DELAY_RANGE

        # Check how many changes/ any changes we have inside the (timeframe - delay range) area. If yes, return true, else false
        # Maybe decrease the delay range here or see how many related properties have changes here to increase precision, has to be tested
        self.test = (data, current_day, timeframe)
        future_data = data[data["value_valid_from"] > np.datetime64(current_day)]
        return len(future_data) != 0

    def get_relevant_ids(self, identifier: str) -> List[str]:
        if identifier not in self.related_properties_lookup.keys():
            return [identifier]
        return self.related_properties_lookup[identifier]
