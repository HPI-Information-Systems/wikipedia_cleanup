import argparse
import json
import pickle
from pathlib import Path
from typing import List, Tuple

import libarchive.public
from tqdm.contrib.concurrent import process_map

from wikipedia_cleanup.data_processing import json_to_infobox_changes
from wikipedia_cleanup.majority_value_per_day import filter_to_only_one_value_per_day
from wikipedia_cleanup.schema import InfoboxChange

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


def calculate_output_path(changes: List[InfoboxChange], output_folder: Path) -> Path:
    page_ids = [change.page_id for change in changes]
    return output_folder.joinpath(f"{min(page_ids)}-{max(page_ids)}-.pickle")


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
                changes.extend(json_to_infobox_changes(json.loads(jsonObj)))
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
    with open(calculate_output_path(changes, output_folder), "wb") as out_file:
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
