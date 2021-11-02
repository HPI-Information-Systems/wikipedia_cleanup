import itertools
import json
from pathlib import Path
from typing import Any, Iterable, Optional, Tuple

import pandas as pd
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from .schema import InfoboxRevision


def read_file(file_path: Path) -> Iterable[Tuple[Any, ...]]:
    changes = []
    with open(file_path) as f:
        for line in f:
            revision = InfoboxRevision(**json.loads(line))
            for change in revision.changes:
                changes.append(
                    (
                        revision.key,
                        revision.revisionId,
                        revision.pageID,
                        revision.pageTitle,
                        change.property.name,
                        change.previousValue,
                        change.currentValue,
                        revision.validFrom,
                        change.valueValidTo,
                    )
                )
    return changes


def get_data(
    input_path: Path, n_files: Optional[int] = None, n_jobs: int = 0
) -> pd.DataFrame:
    files = [x for x in Path(input_path).rglob("*.output.json") if x.is_file()]
    if n_jobs > n_files or n_jobs > len(files):
        n_jobs = min(n_files, len(files))
    files = files[slice(n_files)]
    if n_jobs > 1:
        all_changes = process_map(read_file, files, max_workers=n_jobs)
    else:
        all_changes = []
        for file in tqdm(files):
            all_changes.extend(read_file(file))
    all_changes = itertools.chain.from_iterable(all_changes)
    return pd.DataFrame(
        all_changes,
        columns=[
            "key",
            "revisionId",
            "pageID",
            "pageTitle",
            "property.name",
            "previousValue",
            "currentValue",
            "validFrom",
            "validTo",
        ],
    )
