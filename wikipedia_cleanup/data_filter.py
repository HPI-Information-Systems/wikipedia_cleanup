from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional

from wikipedia_cleanup.schema import InfoboxChange

INITIAL_STATS_VALUE = -1


class FilterStats:
    def __init__(self) -> None:
        self.initial_num_changes = INITIAL_STATS_VALUE
        self.input_num_changes = INITIAL_STATS_VALUE
        self.output_num_changes = INITIAL_STATS_VALUE

    def reset(self) -> None:
        self.initial_num_changes = INITIAL_STATS_VALUE
        self.input_num_changes = INITIAL_STATS_VALUE
        self.output_num_changes = INITIAL_STATS_VALUE

    def __str__(self) -> str:
        num_total_deletions = self.initial_num_changes - self.output_num_changes
        num_self_deletions = self.input_num_changes - self.output_num_changes
        return (
            f"Initial Number of Changes: \t {self.initial_num_changes}\n"
            f"Input Number of Changes: \t {self.input_num_changes}\n"
            f"Output Number of Changes: \t {self.output_num_changes}\n\n"
            f"Filtered Total: \t\t\t\t {num_total_deletions} \t "
            f"{num_total_deletions / self.initial_num_changes * 100} %\n"
            f"Filtered By current Filter: \t\t {num_self_deletions} "
            f"\t current:\t {num_self_deletions / self.input_num_changes * 100} %"
            f"\t total:\t {num_self_deletions / self.initial_num_changes * 100} %\n"
        )

    def add_stats(self, other_stats: "FilterStats") -> None:
        self.initial_num_changes += other_stats.initial_num_changes
        self.input_num_changes += other_stats.input_num_changes
        self.output_num_changes += other_stats.output_num_changes


class AbstractDataFilter(ABC):
    _filter_stats: FilterStats

    def __init__(self) -> None:
        self._filter_stats = FilterStats()

    def filter(
        self, changes: List[InfoboxChange], initial_num_changes: int
    ) -> List[InfoboxChange]:
        if self._filter_stats.initial_num_changes != INITIAL_STATS_VALUE:
            print(
                "WARNING: Using a filter whose stats are not reset. "
                "Thus the stats will be overwritten."
            )
        self._filter_stats.initial_num_changes = initial_num_changes
        self._filter_stats.input_num_changes = len(changes)
        filtered_changes = []
        start_idx = 0
        for end_idx in range(len(changes)):
            if (
                changes[start_idx].infobox_key != changes[end_idx].infobox_key
                or changes[start_idx].property_name != changes[end_idx].property_name
            ):
                filtered_changes.extend(
                    self._filter_for_property(changes[start_idx:end_idx])
                )
                start_idx = end_idx
        self._filter_stats.output_num_changes = len(filtered_changes)
        return filtered_changes

    @abstractmethod
    def _filter_for_property(self, changes: List[InfoboxChange]) -> List[InfoboxChange]:
        pass

    @property
    def filter_stats(self) -> FilterStats:
        return self._filter_stats

    def __str__(self) -> str:
        base_print_width = 30
        return (
            f'{"+" * base_print_width}\n'
            f"{self.__class__.__name__}\n"
            f'{"+" * base_print_width}\n' + str(self.filter_stats)
        )


class DataFilterMinNumChanges(AbstractDataFilter):
    _min_number_of_changes: int

    def __init__(self, min_number_of_changes: int = 5):
        super().__init__()
        self._min_number_of_changes = min_number_of_changes

    @property
    def min_number_of_changes(self) -> int:
        return self._min_number_of_changes

    def _filter_for_property(self, changes: List[InfoboxChange]) -> List[InfoboxChange]:
        return [] if len(changes) < self._min_number_of_changes else changes


class DataFilterMajorityValuePerDay(AbstractDataFilter):
    """
    This filter needs to process each tuple (infobox, property_name, day).
    Therefore the method is overwritten / slightly changes.
    """

    def filter(
        self, changes: List[InfoboxChange], initial_num_changes: int
    ) -> List[InfoboxChange]:
        self._filter_stats.initial_num_changes = initial_num_changes
        self._filter_stats.input_num_changes = len(changes)
        filtered_changes = []
        start_idx = 0
        for end_idx in range(len(changes)):
            if (
                (
                    changes[start_idx].value_valid_from.date()
                    != changes[end_idx].value_valid_from.date()
                )
                or changes[start_idx].infobox_key != changes[end_idx].infobox_key
                or changes[start_idx].property_name != changes[end_idx].property_name
            ):
                filtered_changes.extend(
                    self._filter_for_property(changes[start_idx:end_idx])
                )
                start_idx = end_idx
        self._filter_stats.output_num_changes = len(filtered_changes)
        return filtered_changes

    def _filter_for_property(self, changes: List[InfoboxChange]) -> List[InfoboxChange]:
        if len(changes) == 1:
            return [changes[0]]
        values_to_occurrences: Dict[Optional[str], int] = {}
        for change in changes:
            if change.current_value in values_to_occurrences.keys():
                values_to_occurrences[change.current_value] += 1
            else:
                values_to_occurrences[change.current_value] = 1
        max_occurrence = max(
            values_to_occurrences.iteritems(), key=lambda val_occ: val_occ[1]
        )[1]
        representative_change = deepcopy(
            next(
                filter(
                    lambda change: values_to_occurrences[change.current_value]
                    >= max_occurrence,
                    reversed(changes),
                )
            )
        )
        representative_change.value_valid_to = deepcopy(changes[-1].value_valid_to)
        representative_change.value_valid_from = deepcopy(changes[0].value_valid_from)
        representative_change.num_changes = len(changes)
        return [representative_change]


class DataFilterBotReverts(AbstractDataFilter):
    def _filter_for_property(self, changes: List[InfoboxChange]) -> List[InfoboxChange]:
        def change_pair_is_bot_revert(
            reverted_change: InfoboxChange, bot_revert_change: InfoboxChange
        ) -> bool:
            return (
                reverted_change.current_value == bot_revert_change.previous_value
                and reverted_change.previous_value == bot_revert_change.current_value
                and reverted_change.value_valid_from
                == bot_revert_change.value_valid_from
                and reverted_change.value_valid_to == reverted_change.value_valid_to
                and reverted_change.value_valid_to == bot_revert_change.value_valid_from
            )

        def change_might_be_reverted(change: InfoboxChange) -> bool:
            return change.value_valid_from == change.value_valid_to

        filtered_change_indices = []
        for idx in range(len(changes)):
            curr_change = changes[idx]
            if change_might_be_reverted(curr_change):
                # only look back for the reverting bot change if the index exists
                if idx > 0:
                    related_change = changes[idx - 1]
                    if change_pair_is_bot_revert(curr_change, related_change):
                        # Adjust the valid time if a previous change exists
                        if idx > 1:
                            changes[
                                idx - 2
                            ].value_valid_to = related_change.value_valid_to
                        filtered_change_indices.extend([idx, idx - 1])
                # only look ahead for the reverting bot change if the index exists
                if idx < len(changes) - 1:
                    related_change = changes[idx + 1]
                    if change_pair_is_bot_revert(curr_change, related_change):
                        # Adjust the valid time if a previous change exists
                        if idx > 0:
                            changes[
                                idx - 1
                            ].value_valid_to = related_change.value_valid_to
                        filtered_change_indices.extend([idx, idx + 1])
        filtered_changes = [
            change
            for idx, change in enumerate(changes)
            if idx not in filtered_change_indices
        ]
        return filtered_changes


def generate_default_filters() -> List[AbstractDataFilter]:
    return [
        DataFilterBotReverts(),
        DataFilterMajorityValuePerDay(),
        DataFilterMinNumChanges(),
    ]


def filter_changes_with(
    changes: List[InfoboxChange], filters: List[AbstractDataFilter]
) -> List[InfoboxChange]:
    if not filters:
        return changes
    initial_num_changes = len(changes)
    for data_filter in filters:
        changes = data_filter.filter(changes, initial_num_changes)
    return changes


def merge_filter_stats(
    list_of_filters: List[List[AbstractDataFilter]],
) -> List[AbstractDataFilter]:
    if len(list_of_filters) == 0:
        return []
    merged_filters = deepcopy(list_of_filters[0])
    for filters in list_of_filters[1:]:
        for idx in range(len(filters)):
            filter_name = filters[idx].__class__.__name__
            if filter_name != merged_filters[idx].__class__.__name__:
                raise ValueError("Expected all filters to have the same order.")
            merged_filters[idx].filter_stats.add_stats(filters[idx].filter_stats)
    return merged_filters


def get_stats_from_filters(filters: List[AbstractDataFilter]) -> str:
    if len(filters) == 0:
        return ""
    result = ""
    initial_num_changes = filters[0].filter_stats.initial_num_changes
    if any(
        [
            data_filter.filter_stats.initial_num_changes != initial_num_changes
            for data_filter in filters
        ]
    ):
        result += (
            "WARNING: Initial number of changes mismatch for the given filters. "
            "Filters were probably not used in the same context.\n\n"
        )
    result += "\n".join([str(data_filter) for data_filter in filters])
    return result


def write_filter_stats_to_file(
    filters: List[AbstractDataFilter], output_folder: Path
) -> None:
    with open(output_folder.joinpath("filter-stats.txt"), "wt") as out_file:
        out_file.write(get_stats_from_filters(filters))
