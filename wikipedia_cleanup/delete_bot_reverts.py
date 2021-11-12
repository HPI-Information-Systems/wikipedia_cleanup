from typing import List

from wikipedia_cleanup.schema import InfoboxChange


# expects the changes to be sorted.
def filter_bot_reverts(changes: List[InfoboxChange]) -> List[InfoboxChange]:
    # filtered_changes = []
    for idx in range(len(changes) - 1):
        curr_change = changes[idx]
        next_change = changes[idx + 1]
        if (
            curr_change.current_value == next_change.previous_value
            or curr_change.previous_value == next_change.current_value
        ):
            print(curr_change.value_valid_from, next_change.value_valid_from)
            print(curr_change.value_valid_to, next_change.value_valid_to)

    return changes
