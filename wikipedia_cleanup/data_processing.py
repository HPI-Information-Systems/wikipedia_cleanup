import itertools
import json
import pickle
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map

from wikipedia_cleanup.data_filter import (
    AbstractDataFilter,
    EditWarRevertsDataFilter,
    KeepAttributesDataFilter,
    StaticInfoboxTemplateDataFilter,
    filter_changes_with,
    generate_default_filters,
    get_stats_from_filters,
    merge_filter_stats_into,
)
from wikipedia_cleanup.schema import EditType, InfoboxChange, PropertyType


def json_to_infobox_changes(json_obj: Dict[Any, Any]) -> List[InfoboxChange]:
    changes: List[InfoboxChange] = []
    for change in json_obj["changes"]:
        changes.append(
            InfoboxChange(
                page_id=json_obj["pageID"],
                property_name=change["property"]["name"],
                value_valid_from=json_obj["validFrom"],
                value_valid_to=change.get("valueValidTo", None),
                current_value=change.get("currentValue", None),
                previous_value=change.get("previousValue", None),
                page_title=json_obj["pageTitle"],
                revision_id=json_obj["revisionId"],
                edit_type=EditType[json_obj["type"].upper()],
                property_type=PropertyType[change["property"]["type"].upper()],
                comment=json_obj.get("comment", None),
                infobox_key=json_obj["key"],
                username=json_obj["user"].get("username", None)
                if "user" in json_obj.keys()
                else None,
                user_id=json_obj["user"].get("id", None)
                if "user" in json_obj.keys()
                else None,
                position=json_obj.get("position"),
                template=json_obj.get("template"),
                revision_valid_to=json_obj.get("validTo", None),
                num_changes=1,
            )
        )
    return changes


def read_json_file(file_path: Path) -> List[InfoboxChange]:
    changes = []
    with open(file_path) as f:
        for line in f:
            revision_obj = json.loads(line)
            changes.extend(json_to_infobox_changes(revision_obj))
    return changes


def read_pickle_file(file_path: Path) -> List[InfoboxChange]:
    with open(file_path, "rb") as file:
        return pickle.load(file)  # type: ignore


def sort_changes(changes: List[InfoboxChange]) -> List[InfoboxChange]:
    changes.sort(
        key=lambda change: (
            change.page_id,
            change.infobox_key,
            change.property_name,
            change.value_valid_from,
        )
    )
    return changes


def read_file_sorted(file_path: Path) -> List[InfoboxChange]:
    if file_path.suffix == ".pickle":
        # pickle is already sorted
        return read_pickle_file(file_path)
    elif file_path.suffix == ".json":
        return sort_changes(read_json_file(file_path))
    else:
        raise ValueError("Expected a pickle or json file")


def read_and_filter_file(
    file_path: Path, filters: List[AbstractDataFilter]
) -> Tuple[List[InfoboxChange], List[AbstractDataFilter]]:
    changes = read_file_sorted(file_path)
    return filter_changes_with(changes, filters), filters


def get_data(
    input_path: Path,
    n_files: Optional[int] = None,
    n_jobs: int = 0,
    filters: Optional[List[AbstractDataFilter]] = None,
) -> pd.DataFrame:
    """
    Reads the data into a pd.DataFrame from all files in parallel
    and applies the given filters on the fly.
    The dataframe is guaranteed to be sorted for
    all changes of a page after the priority:
    infobox_key, property_name, value_valid_from.
    The returned filters contain the accumulated stats of the read.

    Example usage:
    ```
    filters = get_default_filters()
    data_frame = get_data(file_path, filters=filters, n_jobs=5)
    ```

    :param input_path: :pathlib.Path:
    Path to the input folder containing decompressed jsons or pickle files.
    :param n_files :Optional[int]:
    Number of files to read from the input folder. None means using all files.
    :param n_jobs: int:
    Number of jobs / processes used for parallel reads.
    :param filters: Optional[List[AbstractDataFilter]]:
    (Ordered) List of filters
    that should be applied on the fly when loading.
    The FilterStats will be written into them.
    Consider using: `get_default_filters()`
    :return: All change items from all read files
    where each all files are sorted after
    (page_id, infobox_key, property_name, value_valid_from)
    """
    if not Path(input_path).is_dir():
        raise AttributeError("Expected to get a valid input path.")
    if filters is None:
        filters = []
    files = [x for x in Path(input_path).rglob("*.output.json") if x.is_file()]
    files.extend([x for x in Path(input_path).rglob("*.pickle") if x.is_file()])
    files = sorted(files)
    files = files[slice(n_files)]
    n_jobs = min(n_jobs, len(files))
    if n_jobs > 1:
        all_changes, mapped_filters = zip(
            *process_map(
                read_and_filter_file,
                files,
                itertools.repeat(filters),
                max_workers=n_jobs,
                total=len(files),
            )
        )
    else:
        all_changes = []
        mapped_filters = []
        for file, data_filters in tqdm(
            zip(files, [deepcopy(filters) for _ in range(len(files))]), total=len(files)
        ):
            changes_and_filters = read_and_filter_file(file, data_filters)
            all_changes.append(changes_and_filters[0])
            mapped_filters.append(changes_and_filters[1])
    all_changes = itertools.chain.from_iterable(all_changes)
    merge_filter_stats_into(mapped_filters, filters)
    res_df = pd.DataFrame(change.__dict__ for change in all_changes)
    return res_df.sort_index(axis=1)


def get_data_single(
    file: Path,
    data_filters: Optional[List[AbstractDataFilter]] = None,
) -> pd.DataFrame:
    if data_filters is None:
        data_filters = []
    changes_and_filters = read_and_filter_file(file, data_filters)
    all_changes = [changes_and_filters[0]]
    mapped_filters = [changes_and_filters[1]]
    all_changes_iterator = itertools.chain.from_iterable(all_changes)
    merge_filter_stats_into(mapped_filters, data_filters)
    return pd.DataFrame([change.__dict__ for change in all_changes_iterator])


# local test
if __name__ == "__main__":
    get_data(
        Path("/home/secret/uni/Masterprojekt/data/test_case_data/output-infobox"),
        1000,
        3,
    )
    filters = generate_default_filters()
    filters.append(
        StaticInfoboxTemplateDataFilter(
            Path("/run/media/secret/manjaro-home/secret/mp-data/avg_dynamic.csv")
        )
    )
    get_data(
        Path(
            "/run/media/secret/manjaro-home/secret/mp-data/"
            "costum-format-filtered-dayly/costum-format-filtered-dayly"
        ),
        5,
        3,
        filters=filters,
    )
    print(get_stats_from_filters(filters))
    filters = generate_default_filters()
    filters.append(EditWarRevertsDataFilter())
    get_data(
        Path(
            "/run/media/secret/manjaro-home/secret/mp-data/"
            "costum-format-filtered-dayly/costum-format-filtered-dayly"
        ),
        5,
        1,
        filters=filters,
    )
    print(get_stats_from_filters(filters))

    filters = generate_default_filters()
    filters.append(KeepAttributesDataFilter(["page_id", "property_name"]))
    print(
        get_data(
            Path("/home/secret/uni/Masterprojekt/data/test_case_data/output-infobox"),
            5,
            1,
            filters=filters,
        ).head()
    )
    print(get_stats_from_filters(filters))
