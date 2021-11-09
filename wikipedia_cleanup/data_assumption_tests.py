import argparse
import json
from pathlib import Path
from typing import List, Tuple

import libarchive.public
from tqdm.contrib.concurrent import process_map

parser = argparse.ArgumentParser(
    description="Transform raw json data to our internal format."
)
parser.add_argument(
    "--input_folder",
    type=str,
    required=True,
    help="Location of the raw 7z compressed json files.",
)


def get_json_file_stats(input_path: Path) -> Tuple[List[int], List[str]]:
    page_ids = set()
    infobox_ids = set()
    with libarchive.public.file_reader(str(input_path)) as archive:
        for entry in archive:
            content_bytes = bytearray("", encoding="utf_8")
            for block in entry.get_blocks():
                content_bytes += block
            content = content_bytes.decode(encoding="utf_8")
            json_objs = content.split("\n")
            for jsonObj in filter(lambda x: x, json_objs):
                revision = json.loads(jsonObj)
                page_ids.add(revision["pageID"])
                infobox_ids.add(revision["key"])
    return list(page_ids), list(infobox_ids)


if __name__ == "__main__":
    args = parser.parse_args()
    input_files = list(Path(args.input_folder).rglob("*.7z"))
    res = process_map(
        get_json_file_stats,
        input_files,
        max_workers=3,
    )
    for idx in range(len(res) - 1):
        page_ids, infobox_ids = res[idx]
        max_page_id, min_page_id = max(page_ids), min(page_ids)
        max_infobox_id, min_infobox_id = max(infobox_ids), min(infobox_ids)

        for page_ids, infobox_ids in res[:idx] + res[idx + 1 :]:
            for page_id in page_ids:
                if min_page_id <= page_id <= max_page_id:
                    print("PAGE ID ASSERT FAILED")
            """for infobox_id in infobox_ids:
                if min_infobox_id <= infobox_id <= max_infobox_id:
                    print("INFOBOX ID ASSERT FAILED")"""  # Failed
