from typing import Any, List

from wikipedia_cleanup.schema import InfoboxChange


def change_pair_is_bot_revert(
    reverted_change: InfoboxChange, bot_revert_change: InfoboxChange
) -> bool:
    return (
        reverted_change.current_value == bot_revert_change.previous_value
        and reverted_change.previous_value == bot_revert_change.current_value
        and reverted_change.value_valid_from == bot_revert_change.value_valid_from
        and reverted_change.value_valid_to == reverted_change.value_valid_to
        and reverted_change.value_valid_to == bot_revert_change.value_valid_from
    )


# expects the changes to be sorted.
def filter_bot_reverts(
    changes: List[InfoboxChange], **kwargs: Any
) -> List[InfoboxChange]:
    filtered_change_indices = []
    for idx in range(len(changes)):
        curr_change = changes[idx]
        if curr_change.value_valid_from == curr_change.value_valid_to:
            if idx > 0:
                related_change = changes[idx - 1]
                if change_pair_is_bot_revert(curr_change, related_change):
                    if idx > 1:
                        changes[idx - 2].value_valid_to = related_change.value_valid_to
                    filtered_change_indices.extend([idx, idx - 1])
            if idx < len(changes) - 1:
                related_change = changes[idx + 1]
                if change_pair_is_bot_revert(curr_change, related_change):
                    if idx > 0:
                        changes[idx - 1].value_valid_to = related_change.value_valid_to
                    filtered_change_indices.extend([idx, idx + 1])
    filtered_changes = [
        change
        for idx, change in enumerate(changes)
        if idx not in filtered_change_indices
    ]
    return filtered_changes
