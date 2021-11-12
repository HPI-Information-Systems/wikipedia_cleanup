from typing import Any, List

from wikipedia_cleanup.schema import InfoboxChange


def filter_properties_with_low_number_of_changes(
    changes: List[InfoboxChange], **kwargs: Any
) -> List[InfoboxChange]:
    min_num_changes = kwargs["min_num_changes"]
    return [] if min_num_changes < len(changes) else changes
