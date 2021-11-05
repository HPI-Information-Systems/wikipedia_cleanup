import argparse
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import libarchive.public
from tqdm.contrib.concurrent import process_map

from wikipedia_cleanup.majority_value_per_day import filter_to_only_one_value_per_day
from wikipedia_cleanup.schema import EditType, InfoboxChange, PropertyType

parser = argparse.ArgumentParser(
    description="Transform raw json data to our internal format."
)
parser.add_argument(
    "--input_folder",
    type=str,
    required=True,
    help="Location of the raw 7z compressed json files.",
)
parser.add_argument(
    "--output_folder",
    type=str,
    required=True,
    help="Location of the output folder to put the pickle files.",
)
parser.add_argument(
    "--test",
    default=False,
    action="store_true",
    help="Test the script by executing it on only one file.",
)
parser.add_argument(
    "--max_workers",
    type=int,
    default=2,
    help="Max number of workers for parallelization.",
)


def json_to_infobox_change(json_obj: Dict[Any, Any], idx: int) -> InfoboxChange:
    change_obj = json_obj["changes"][idx]
    return InfoboxChange(
        page_id=json_obj["pageID"],
        property_name=change_obj["property"]["name"],
        value_valid_from=json_obj["validFrom"],
        value_valid_to=datetime.strptime(
            change_obj["valueValidTo"], "%Y-%m-%dT%H:%M:%SZ"
        )
        if "valueValidTo" in change_obj
        else None,
        current_value=change_obj.get("currentValue", None),
        previous_value=change_obj.get("previousValue", None),
        page_title=json_obj["pageTitle"],
        revision_id=json_obj["revisionId"],
        edit_type=EditType[json_obj["type"].upper()],
        property_type=PropertyType[change_obj["property"]["type"].upper()],
        comment=json_obj.get("comment", None),
        infobox_key=json_obj["key"],
        username=json_obj["user"].get("username", None)
        if "user" in json_obj.keys()
        else None,
        user_id=json_obj["user"].get("id", None) if "user" in json_obj.keys() else None,
        position=json_obj.get("position"),
        template=json_obj.get("template"),
        revision_valid_to=json_obj.get("validTo", None),
    )


def calculate_output_path(change: InfoboxChange, output_folder: Path) -> Path:
    return output_folder.joinpath(f"{change.page_id}.pickle")


def process_json_file(input_and_output_path: Tuple[Path, Path]) -> None:
    input_file, output_folder = input_and_output_path
    changes: List[InfoboxChange] = []
    with libarchive.public.file_reader(str(input_file)) as archive:
        for entry in archive:
            content_bytes = bytearray("", encoding="utf_8")
            for block in entry.get_blocks():
                content_bytes += block
            content = content_bytes.decode(encoding="utf_8")
            json_objs = content.split("\n")
            # load all changes into map
            for jsonObj in filter(lambda x: x, json_objs):
                obj = json.loads(jsonObj)
                for idx in range(len(obj["changes"])):
                    changes.append(json_to_infobox_change(obj, 0))
            # sort changes after infobox_key, property_name, change.timestamp
            changes.sort(
                key=lambda change: (
                    change.page_id,
                    change.infobox_key,
                    change.property_name,
                    change.value_valid_from,
                )
            )
    changes = filter_to_only_one_value_per_day(changes)
    with open(calculate_output_path(changes[0], output_folder), "wb") as out_file:
        pickle.dump(changes, out_file)


if __name__ == "__main__":
    args = parser.parse_args()
    input_files = list(Path(args.input_folder).rglob("*.7z"))
    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    if args.test:
        input_files = [input_files[0]]
    process_map(
        process_json_file,
        zip(input_files, [output_folder] * len(input_files)),
        max_workers=args.max_workers,
    )
