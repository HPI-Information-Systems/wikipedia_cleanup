from datetime import datetime, timedelta
from typing import Optional

import numpy as np


def next_change(previous_change_timestamps: np.array[datetime]) -> Optional[datetime]:
    if len(previous_change_timestamps) < 2:
        return None

    mean_time_to_change: timedelta = np.mean(
        previous_change_timestamps[1:] - previous_change_timestamps[0:-1]
    )
    return_value: datetime = previous_change_timestamps[-1] + mean_time_to_change
    return return_value
