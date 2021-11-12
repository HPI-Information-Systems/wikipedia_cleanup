from copy import deepcopy
from typing import Any, Dict, List, Optional

from wikipedia_cleanup.schema import InfoboxChange


# This selects the most present value and the last one if
# there are multiple values with same number of occurrences.
def get_representative_value_for_day(
    changes: List[InfoboxChange], **kwargs: Any
) -> List[InfoboxChange]:
    if len(changes) == 1:
        return [changes[0]]
    values_to_occurrences: Dict[Optional[str], int] = {}
    for change in changes:
        if change.current_value in values_to_occurrences.keys():
            values_to_occurrences[change.current_value] += 1
        else:
            values_to_occurrences[change.current_value] = 1
    max_occurrence = max(values_to_occurrences.items(), key=lambda val_occ: val_occ[1])[
        1
    ]
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
