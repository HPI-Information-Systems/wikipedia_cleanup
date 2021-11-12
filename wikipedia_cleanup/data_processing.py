import itertools
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

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


def read_file(file_path: Path) -> List[InfoboxChange]:
    if "pickle" in str(file_path):
        return read_pickle_file(file_path)
    elif "json" in str(file_path):
        return read_json_file(file_path)
    raise ValueError("Expected a pickle or json file")


def get_data(
    input_path: Path, n_files: Optional[int] = None, n_jobs: int = 0
) -> pd.DataFrame:
    files = [x for x in Path(input_path).rglob("*.output.json") if x.is_file()]
    files.extend([x for x in Path(input_path).rglob("*.pickle") if x.is_file()])
    if n_files is not None:
        n_jobs = min(n_jobs, n_files)
    n_jobs = min(n_jobs, len(files))
    files = files[slice(n_files)]
    if n_jobs > 1:
        all_changes = process_map(read_file, files, max_workers=n_jobs)
    else:
        all_changes = []
        for file in tqdm(files):
            all_changes.append(read_file(file))
    all_changes = itertools.chain.from_iterable(all_changes)
    return pd.DataFrame([change.__dict__ for change in all_changes])


# test
if __name__ == "__main__":
    get_data(
        Path("/home/secret/uni/Masterprojekt/data/test_case_data/output-infobox"),
        1000,
        3,
    )
    get_data(
        Path(
            "/run/media/secret/manjaro-home/secret/mp-data/"
            "costum-format-filtered-dayly/costum-format-filtered-dayly"
        ),
        5,
        3,
    )
