import numpy as np


def next_change(previous_change_timestamps):
    if len(previous_change_timestamps) < 2:
        return None
    mean_time_to_change = np.mean(
        previous_change_timestamps[1:] - previous_change_timestamps[0:-1]
    )
    return previous_change_timestamps[-1] + mean_time_to_change
