import math
from typing import FrozenSet, Tuple

import pandas as pd


def train_val_split(
    train_data: pd.DataFrame, val_size: float
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    assert 0 <= val_size <= 1
    cutoff = math.floor(len(train_data) * (1 - val_size))
    return train_data.iloc[:cutoff], train_data.iloc[cutoff:]


def precision(transactions: Tuple[FrozenSet[str], ...], rhs: str, lhs: str) -> float:
    true_positives = 0
    false_positives = 0
    for transaction in transactions:
        if lhs in transaction:
            if rhs in transaction:
                true_positives += 1
            else:
                false_positives += 1
    denominator = true_positives + false_positives
    if not denominator:
        return float("nan")
    return true_positives / denominator
