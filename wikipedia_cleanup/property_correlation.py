import re
from datetime import date, datetime, timedelta
from typing import List

import numpy as np
import pandas as pd
from line_profiler_pycharm import profile
from scipy.sparse import csr_matrix, vstack
from sklearn.neighbors import NearestNeighbors
from tqdm.auto import tqdm

from wikipedia_cleanup.predictor import Predictor


class PropertyCorrelationPredictor(Predictor):
    @staticmethod
    def _get_links(train_data: pd.DataFrame):
        regex_str = '\\[\\[((?:\\w+:)?[^<>\\[\\]"\\|]+)(?:\\|[^\\n\\]]+)?\\]\\]'
        regex = re.compile(regex_str)

        infobox_key_to_related_page_titles = {}

        grouped_infoboxes = train_data.groupby("infobox_key")["current_value"].unique()

        for key, row in grouped_infoboxes.iteritems():
            infobox_key_to_related_page_titles[key] = list(
                set(
                    match.groups()[0].strip()
                    for value in row
                    if value
                    for match in regex.finditer(value)
                    if not match.groups()[0].startswith(("Image:", "File:"))
                )
            )

        return infobox_key_to_related_page_titles

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

    def fit(self, train_data: pd.DataFrame, last_day: datetime) -> None:

        infobox_key_to_related_page_titles = self._get_links(train_data)

        def percentage_manhatten_adaptive_time_lag(arr1, arr2):
            DELAY_RANGE = 3

            arr1 = arr1.toarray()
            arr2 = arr2.toarray()
            max_changes = arr1.sum()
            mask = np.nonzero(arr1)
            error = 0

            for idx in mask[1]:
                needed_num_changes = arr1[0, idx]
                for off in range(
                    -min(DELAY_RANGE, idx), min(DELAY_RANGE, arr2.shape[1] - idx)
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

        infobox_key_to_related_page_titles_filtered = {}

        links = set()

        for infobox_key, related_pages in infobox_key_to_related_page_titles.items():
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

        for infobox_key, related_pages in tqdm(
            infobox_key_to_related_page_titles.items(),
            total=len(infobox_key_to_related_page_titles),
        ):
            found_related_pages = []
            for related_page in related_pages:
                if related_page in links_found:
                    found_related_pages.append(related_page)
            infobox_key_to_related_page_titles_filtered[
                infobox_key
            ] = found_related_pages

        max_dist = 0.05

        @profile
        def hihi():
            same_infoboxes = []
            matches = []
            for key, row in tqdm(page_id_groups.iterrows(), total=len(page_id_groups)):
                if len(row[1]) > 1:
                    infobox_keys_on_page = set(row[2])
                    related_page_names = set()
                    for infobox_key in infobox_keys_on_page:
                        related_page_names.update(
                            infobox_key_to_related_page_titles_filtered[infobox_key]
                        )
                    related_page_names.discard(key)
                    for related_key, related_row in page_id_groups.loc[
                        list(related_page_names)
                    ].iterrows():
                        row[0].extend(related_row[0])
                        row[1].extend(related_row[1])
                        row[2].extend(related_row[2])
                        pass
                    input_data = vstack(row[1])
                    neigh = NearestNeighbors(
                        radius=max_dist,
                        metric=percentage_manhatten_adaptive_time_lag_symmetric,
                    )
                    print(input_data.count_nonzero(), key)
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
                            matches.append(match)

        hihi()

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
        raise NotImplementedError()

    def get_relevant_ids(self, identifier: str) -> List[str]:
        raise NotImplementedError()
