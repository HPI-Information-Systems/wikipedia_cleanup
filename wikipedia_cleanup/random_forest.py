import hashlib
import pickle
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from tqdm.auto import tqdm

from wikipedia_cleanup.predictor import Predictor


class RandomForestPredictor(Predictor):
    def __init__(self, use_cache: bool = True, padding: bool = False, classify: bool = True, binary: bool=False) -> None:
        super().__init__()
        self.regressors: dict = {}
        self.classifiers: dict = {}
        self.last_preds: dict = {}
        # contains for a given infobox_propertyname (key) a (date,pred) tuple (value)
        # date is the date of the last change and pred the days until next change
        self.hash_location = Path("cache") / self.__class__.__name__
        self.use_hash = use_cache
        self.padding = padding
        self.classify = classify
        self.binary = binary

    @staticmethod
    def get_relevant_ids(identifier: Tuple) -> List[Tuple]:
        return []

    def get_relevant_attributes(self) -> List[str]:
        if self.padding:
            return [
                "value_valid_from",
                "day_of_year",  
                "day_of_month",
                "day_of_week",
                "month_of_year",
                "quarter_of_year",
                "is_month_start",
                "is_month_end",
                "is_quarter_start",
                "is_quarter_end",
                "days_since_last_change",
                "days_until_next_change",
                "is_change"
            ]
        else:
            return [
                "value_valid_from",
                "day_of_year",  
                "day_of_month",
                "day_of_week",
                "month_of_year",
                "quarter_of_year",
                "is_month_start",
                "is_month_end",
                "is_quarter_start",
                "is_quarter_end",
                "days_since_last_change",
                "days_since_last_2_changes",
                "days_since_last_3_changes",
                "days_between_last_and_2nd_to_last_change",
                "days_until_next_change",
                "mean_change_frequency_all_previous",
                "mean_change_frequency_last_3",
                "is_change"
            ]


    def _load_cache(self, possible_cached_mapping: Path) -> bool:
        try:
            if possible_cached_mapping.exists():
                print(f"Cached model found, loading from {possible_cached_mapping}")
                with open(possible_cached_mapping, "rb") as f:
                    if self.classify:
                        self.classifers = pickle.load(f)
                    else:
                        self.regressors = pickle.load(f)
                return True
            else:
                print("No cache found, recalculating model.")
        except EOFError:
            print("Caching failed, recalculating model.")
        return False

    def fit(
        self, train_data: pd.DataFrame, last_day: datetime, keys: List[str]
    ) -> None:
        if (not self.classify and self.binary) or (not self.padding and self.binary):
            raise NotImplementedError
        if self.use_hash:
            possible_cached_mapping = self._calculate_cache_name(train_data)
            if self._load_cache(possible_cached_mapping):
                return
        DUMMY_TIMESTAMP = pd.Timestamp("1999-12-31")
        # used as dummy for date comparison in first prediction
        key_change_counts = train_data.groupby(keys, sort=False).count()[
            "value_valid_from"
        ]
        # Assumption: data is already sorted by keys so
        # no further sorting needs to be done
        start = 0
        for key_change_count in tqdm(key_change_counts):
            if key_change_count == 1:
                start += key_change_count
                continue
            # maybe dont train if we are below a trainsize threshold
            if self.padding: 
                sample = train_data.iloc[start : start + key_change_count].copy
                sample['is_change'] = True
                ranges = sample['days_since_last_change'].apply(lambda x: list(range(1, x+1)))
                ranges = np.append([0],np.concatenate(ranges.values).ravel())
                revert_ranges = sample['days_since_last_change'].apply(lambda x: list(range(x+1, 1, -1)))
                revert_ranges = np.append(np.concatenate(revert_ranges.values).ravel(), [0])
                # pad all days between first and last change
                # TODO: maybe use only random subset for padding
                sample  = sample.set_index('value_valid_from')\
                    .groupby(['infobox_key', 'property_name']).apply(lambda x: \
                        x.reindex(pd.date_range(x.index.min(), x.index.max(), freq='D', name='value_valid_from')))\
                        .drop(['infobox_key', 'property_name'], axis=1).reset_index()
                sample.fillna(False, inplace=True)
                sample["day_of_year"] = sample["value_valid_from"].dt.dayofyear
                sample["day_of_month"] = sample["value_valid_from"].dt.day
                sample["day_of_week"] = sample["value_valid_from"].dt.dayofweek
                sample["month_of_year"] = sample["value_valid_from"].dt.month
                sample["quarter_of_year"] = sample["value_valid_from"].dt.quarter
                sample["is_month_start"] = sample["value_valid_from"].dt.is_month_start
                sample["is_month_end"] = sample["value_valid_from"].dt.is_month_end
                sample["is_quarter_start"] = sample["value_valid_from"].dt.is_quarter_start
                sample["is_quarter_end"] = sample["value_valid_from"].dt.is_quarter_end
                sample["days_since_last_change"] = ranges
                sample["days_until_next_change"] = revert_ranges

            else:
                sample = train_data.iloc[start : start + key_change_count-1]
                sample['is_change'] = True

            start += key_change_count

            X = sample[self.get_relevant_attributes()].drop(
                    columns=["value_valid_from","days_until_next_change", "is_change"]
                )
            if self.binary:                
                y = sample["is_change"]
    
            else:
                y = sample["days_until_next_change"]
        
            if self.classify:
                clf = RandomForestClassifier(
                    random_state=0, n_estimators=10, max_features="auto"
                )
                clf.fit(X, y)
                self.classifiers[sample["key"].iloc[0]] = clf
            else:
                reg = RandomForestRegressor(
                    random_state=0, n_estimators=10, max_features="auto"
                )
                reg.fit(X, y)
                self.regressors[sample["key"].iloc[0]] = reg

            self.last_preds[sample["key"].iloc[0]] = [DUMMY_TIMESTAMP, 0]

        if self.use_hash:
            possible_cached_mapping.parent.mkdir(exist_ok=True, parents=True)
            with open(possible_cached_mapping, "wb") as f:
                if self.classify:
                    pickle.dump(self.classifiers, f)
                else:
                    pickle.dump(self.regressors, f)

    def _calculate_cache_name(self, data: pd.DataFrame) -> Path:
        hash_string = "".join(str(x) for x in [data.shape, data.head(2), data.tail(2)])
        hash_id = hashlib.md5(hash_string.encode("utf-8")).hexdigest()[:10]
        possible_cached_mapping = self.hash_location / hash_id
        return possible_cached_mapping

    def predict_timeframe(
        self,
        data_key: np.ndarray,
        additional_data: np.ndarray,
        columns: List[str],
        first_day_to_predict: date,
        timeframe: int,
    ) -> bool:
        if len(data_key) == 0:
            return False
        key_column_idx = columns.index("key")
        data_key_item = data_key[0, key_column_idx]
        value_valid_from_column_idx = columns.index("value_valid_from")
        if self.classify:
            if data_key_item not in self.classifiers:
                # checks if model has been trained for the key
                # (it didnt if there was no traindata)
                return False  
        else:
            if data_key_item not in self.regressors:
                # checks if model has been trained for the key
                # (it didnt if there was no traindata)
                return False        
        if self.padding:
            first_day_to_predict = pd.to_datetime(first_day_to_predict)
            sample  = []
            last_change = pd.to_datetime(data_key[-1, value_valid_from_column_idx])
            sample.append(first_day_to_predict.dayofyear)
            sample.append(first_day_to_predict.day)
            sample.append(first_day_to_predict.dayofweek)
            sample.append(first_day_to_predict.month)
            sample.append(first_day_to_predict.quarter)
            sample.append(first_day_to_predict.is_month_start)
            sample.append(first_day_to_predict.is_month_end)
            sample.append(first_day_to_predict.is_quarter_start)
            sample.append(first_day_to_predict.is_quarter_end)
            sample.append((first_day_to_predict - last_change).days)
            sample_value_valid_from = sample[value_valid_from_column_idx]
            X_test = sample
            if self.classify:
                clf = self.classifiers[data_key_item]
                pred = clf.predict([X_test])[0]
                if self.binary:
                    return pred
            else:
                reg = self.regressors[data_key_item]
                pred = int(reg.predict([X_test])[0])
            self.last_preds[data_key_item] = [first_day_to_predict, pred]
        else:
            sample = data_key[-1, ...]
            sample_value_valid_from = sample[value_valid_from_column_idx]
            if self.last_preds[data_key_item][0] != sample_value_valid_from:
            # check if we already predicted next change for that day
                indices = [
                    columns.index(attr)
                    for attr in self.get_relevant_attributes()
                    if not (attr == "value_valid_from" or attr == "days_until_next_change" 
                            or attr == "is_change")
                ]
                X_test = sample[indices].reshape(1, -1)
                if self.classify:
                    clf = self.classifiers[data_key_item]
                    pred = clf.predict(X_test)[0]   
                else:
                    reg = self.regressors[data_key_item]
                    pred = int(reg.predict(X_test)[0])
                self.last_preds[data_key_item] = [sample_value_valid_from, pred]
            else:
                pred = self.last_preds[data_key_item][1]
        return (
            first_day_to_predict
            <= (first_day_to_predict + timedelta(pred))
            < first_day_to_predict + timedelta(timeframe))
