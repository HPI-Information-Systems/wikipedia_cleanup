from copy import deepcopy
from datetime import datetime
from typing import List

from wikipedia_cleanup.schema import InfoboxChange


def get_representative_value(changes: List[InfoboxChange]) -> InfoboxChange:
    values_to_duration = {
        change.current_value: [0.0, int(i)] for i, change in enumerate(changes)
    }
    for change in changes:
        tzinfo = None
        if change.value_valid_from and change.value_valid_from.tzinfo:
            tzinfo = change.value_valid_from.tzinfo
        elif change.value_valid_to and change.value_valid_to.tzinfo:
            tzinfo = change.value_valid_to.tzinfo
        start_date = change.value_valid_from
        if not start_date:
            assert change.value_valid_to, "Expected other value to be defined"
            start_date = datetime.combine(
                change.value_valid_to.date(), datetime.min.time()
            )
        end_date = change.value_valid_to
        if not end_date:
            assert change.value_valid_from, "Expected other value to be defined"
            start_date = datetime.combine(
                change.value_valid_from.date(), datetime.min.time()
            )
        start_date = start_date.replace(tzinfo=tzinfo)
        end_date = end_date.replace(tzinfo=tzinfo)  # type: ignore
        values_to_duration[change.current_value][0] += (
            end_date - start_date
        ).total_seconds()
    max_item = max(
        values_to_duration.items(), key=lambda key_and_val: key_and_val[1][0]
    )
    representative_change = deepcopy(changes[int(max_item[1][1])])
    representative_change.value_valid_to = deepcopy(changes[-1].value_valid_to)
    print(representative_change)
    return representative_change


# expects the changes to bes sorted.
def filter_to_only_one_value_per_day(
    changes: List[InfoboxChange],
) -> List[InfoboxChange]:
    filtered_changes = []
    start_idx = 0
    for end_idx in range(len(changes)):
        if (
            changes[start_idx].value_valid_from.date()
            != changes[end_idx].value_valid_from.date()
            or changes[start_idx].infobox_key != changes[end_idx].infobox_key
            or changes[start_idx].property_name != changes[end_idx].property_name
        ):
            filtered_changes.append(
                get_representative_value(changes[start_idx:end_idx])
            )
            start_idx = end_idx
    return filtered_changes
