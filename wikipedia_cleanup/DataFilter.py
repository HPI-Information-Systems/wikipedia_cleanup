from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Dict, List, Optional

from wikipedia_cleanup.schema import InfoboxChange


class AbstractDataFilter(ABC):
    def filter(self, changes: List[InfoboxChange]) -> List[InfoboxChange]:
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
        return filtered_changes

    @abstractmethod
    def _filter_for_property(self, changes: List[InfoboxChange]) -> List[InfoboxChange]:
        pass


class DataFilterMinNumChanges(AbstractDataFilter):
    _min_number_of_changes: int

    def __init__(self, min_number_of_changes: int = 5):
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

    def filter(self, changes: List[InfoboxChange]) -> List[InfoboxChange]:
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
            values_to_occurrences.items(), key=lambda val_occ: val_occ[1]
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
    for data_filter in filters:
        changes = data_filter.filter(changes)
    return changes
