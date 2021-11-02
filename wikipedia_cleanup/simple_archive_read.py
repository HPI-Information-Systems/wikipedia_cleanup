import json
import pickle
from datetime import datetime
from pathlib import Path

import Levenshtein
import libarchive.public
import numpy as np
import pandas as pd
from tqdm.contrib.concurrent import process_map

raw_data_folder = Path("/home/secret/uni/Masterprojekt/data/matched-infoboxes-raw")


def analyse_string_numeric(archive_path):
    numeric = 0
    string = 0
    numeric_to_string = 0
    string_to_numeric = 0
    with libarchive.public.file_reader(str(archive_path)) as archive:
        for entry in archive:
            content_bytes = bytearray("", encoding="utf_8")
            for block in entry.get_blocks():
                content_bytes += block
            content = content_bytes.decode(encoding="utf_8")
            jsonObjs = content.split("\n")
            for jsonObj in filter(lambda x: x, jsonObjs):
                obj = json.loads(jsonObj)
                changes = obj["changes"]
                for change in changes:
                    curr_val_number = None
                    prev_val_number = None
                    if "previousValue" in change:
                        try:
                            float(change["previousValue"])
                            prev_val_number = True
                        except ValueError:
                            prev_val_number = False
                    if "currentValue" in change:
                        try:
                            float(change["currentValue"])
                            curr_val_number = True
                        except ValueError:
                            curr_val_number = False
                if curr_val_number is not None:
                    numeric += int(curr_val_number)
                    string += int(not curr_val_number)
                    if prev_val_number is not None:
                        if curr_val_number and not prev_val_number:
                            string_to_numeric += 1
                        if not curr_val_number and prev_val_number:
                            numeric_to_string += 1
    return numeric, string, numeric_to_string, string_to_numeric


def assert_every_change_changes_the_value(archive_path):
    with libarchive.public.file_reader(str(archive_path)) as archive:
        for entry in archive:
            content_bytes = bytearray("", encoding="utf_8")
            for block in entry.get_blocks():
                content_bytes += block
            content = content_bytes.decode(encoding="utf_8")
            jsonObjs = content.split("\n")
            for jsonObj in filter(lambda x: x, jsonObjs):
                obj = json.loads(jsonObj)
                changes = obj["changes"]
                for change in changes:
                    if (
                        "previousValue" in change.keys()
                        and "currentValue" in change.keys()
                    ):
                        assert change["previousValue"] != change["currentValue"]


def get_every_change_size(archive_path):
    change_sizes = []
    with libarchive.public.file_reader(str(archive_path)) as archive:
        for entry in archive:
            content_bytes = bytearray("", encoding="utf_8")
            for block in entry.get_blocks():
                content_bytes += block
            content = content_bytes.decode(encoding="utf_8")
            jsonObjs = content.split("\n")
            for jsonObj in filter(lambda x: x, jsonObjs):
                obj = json.loads(jsonObj)
                changes = obj["changes"]
                for change in changes:
                    if (
                        "previousValue" in change.keys()
                        and "currentValue" in change.keys()
                    ):
                        assert change["previousValue"] != change["currentValue"]
                        change_sizes.append(
                            Levenshtein.distance(
                                change["previousValue"], change["currentValue"]
                            )
                        )
    return change_sizes


# TODO test bot reverts on local dataset.


def count_creation_and_deletion_numbers(archive_path):
    creations = 0
    deletions = 0
    edits = 0
    with libarchive.public.file_reader(str(archive_path)) as archive:
        for entry in archive:
            content_bytes = bytearray("", encoding="utf_8")
            for block in entry.get_blocks():
                content_bytes += block
            content = content_bytes.decode(encoding="utf_8")
            jsonObjs = content.split("\n")
            for jsonObj in filter(lambda x: x, jsonObjs):
                obj = json.loads(jsonObj)
                changes = obj["changes"]
                for change in changes:
                    edits += 1
                    if (
                        "previousValue" in change.keys()
                        and "currentValue" not in change.keys()
                    ):
                        deletions += 1
                    if (
                        "previousValue" not in change.keys()
                        and "currentValue" in change.keys()
                    ):
                        creations += 1
    return creations, deletions, edits


def get_all_valid_times(archive_path):
    valid_times = []
    with libarchive.public.file_reader(str(archive_path)) as archive:
        for entry in archive:
            content_bytes = bytearray("", encoding="utf_8")
            for block in entry.get_blocks():
                content_bytes += block
            content = content_bytes.decode(encoding="utf_8")
            jsonObjs = content.split("\n")
            for jsonObj in filter(lambda x: x, jsonObjs):
                obj = json.loads(jsonObj)
                changes = obj["changes"]
                valid_from = obj["validFrom"]
                for change in changes:
                    if "valueValidTo" in change.keys():
                        delta = (
                            datetime.strptime(
                                change["valueValidTo"], "%Y-%m-%dT%H:%M:%SZ"
                            )
                            - datetime.strptime(valid_from, "%Y-%m-%dT%H:%M:%SZ")
                        ).total_seconds()
                        valid_times.append(delta)
    return valid_times


if __name__ == "__main__":
    input_files = list(raw_data_folder.rglob("*.7z"))

    """res = process_map(analyse_string_numeric, input_files, max_workers=3)
    numeric = 0
    string = 0
    numeric_to_string = 0
    string_to_numeric = 0

    for (curr_numeric, curr_string, curr_numeric_to_string, curr_string_to_numeric) in res:
        numeric += curr_numeric
        string += curr_string
        numeric_to_string += curr_numeric_to_string
        string_to_numeric += curr_string_to_numeric

    print(f'numeric: {numeric} \t\t % {numeric / (numeric + string)}')
    print(f'string: {string} \t\t % {string / (numeric + string)}')
    print('\n\nType Changes\n\n')
    print(f'numeric to string: {numeric_to_string} \t\t % {numeric_to_string / (numeric + string)}')
    print(f'string to numeric: {string_to_numeric} \t\t % {string_to_numeric / (numeric + string)}')"""

    """numeric: 4749733 		 % 0.0811687487193842
        string: 53767037 		 % 0.9188312512806158
        
        
        Type Changes
        
        
        numeric to string: 622079 		 % 0.01063078156911258
        string to numeric: 849259 		 % 0.014513087444847007"""

    # process_map(assert_every_change_changes_the_value, input_files, max_workers=4)
    # asserts match

    """res = process_map(get_every_change_size, input_files, max_workers=3)
    all_change_sizes = []
    for change_sizes in res:
        all_change_sizes.extend(change_sizes)

    with open(raw_data_folder.joinpath('change_sizes.pickle'), 'wb') as file:
        pickle.dump(all_change_sizes, file)

    plt.hist(np.array(all_change_sizes), bins=list(range(40)))
    plt.title("Size of changes")
    plt.ylabel("#Occurances")"""

    """res = process_map(count_creation_and_deletion_numbers, input_files, max_workers=3)
    all_creations = 0
    all_deletions = 0
    all_edits = 0
    for (creations, deletions, edits) in res:
        all_creations += creations
        all_deletions += deletions
        all_edits += edits

    print(f'Total Edits: {all_edits}')
    print(f'creations: {all_creations} \t\t % {all_creations / all_edits}')
    print(f'deletions: {all_deletions} \t\t % {all_deletions / all_edits}')"""

    """Total Edits: 282713251
    creations: 143045982 		 % 0.5059755122691437
    deletions: 57379699 		 % 0.20296076960326137"""

    res = process_map(get_all_valid_times, input_files, max_workers=1)

    all_times = []
    for a in res:
        all_times.extend(a)
    del res
    with open(raw_data_folder.joinpath("all_times.pickle"), "wb") as file:
        pickle.dump(all_times, file)
