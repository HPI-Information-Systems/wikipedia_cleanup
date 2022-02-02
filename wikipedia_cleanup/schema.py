from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel


class EditType(Enum):
    CREATE = 0
    DELETE = 1
    UPDATE = 2


class PropertyType(Enum):
    ATTRIBUTE = 0
    META = 1


class InfoboxChange(BaseModel):
    page_id: int  # wikipedia pageID: Page key but no infobox id
    property_name: str
    value_valid_to: Optional[datetime] = None
    value_valid_from: datetime  # timestamp of the revision
    current_value: Optional[str] = None
    previous_value: Optional[str] = None
    num_changes: int = 1

    page_title: str  # self-explanatory: can change, no identifier
    revision_id: int  # ChangeID of a page
    edit_type: EditType
    property_type: PropertyType
    comment: Optional[str] = None
    infobox_key: str  # infobox-key, unrelated to PageID and revisionId, globally unique
    username: Optional[str] = None
    user_id: Optional[str] = None
    position: Optional[int] = None  # i-th infobox of the page in this revision
    template: Optional[
        str
    ] = None  # name of the used template / category e.g. "infobox person"
    revision_valid_to: Optional[
        datetime
    ] = None  # date of the next revision for that infobox

    # => revision_valid_to <= value_valid_to

    def __str__(self) -> str:
        return str(self.__dict__)


class InfoboxChangeWithFeatures(InfoboxChange):
    day_of_year: int
    day_of_month: int
    day_of_week: int
    month_of_year: int
    quarter_of_year: int
    is_month_start: bool
    is_month_end: bool
    is_quarter_start: bool
    is_quarter_end: bool
    days_since_last_change: int
    days_since_last_2_changes: int
    days_since_last_3_changes: int
    days_until_next_change: int
    days_between_last_and_2nd_to_last_change: int
    mean_change_frequency_all_previous: float
    mean_change_frequency_last_3: float


class SparseInfoboxChange:
    pass


# Knowledge over json-files:
# Each file consists of the revisions / changes of one or more pages.
# All revisions of a page are in one file.
# Since a page can have multiple infoboxes, one json-file
# can contain all changes of multiple infoboxes.

# TODO: check: InfoboxRevisionHistory.key is globally unique
#  and all changes are written in one file. = true

# TODO look for InfoboxProperty.propertyType == "meta" and
#  look if these are present in source code of wikipedia.

# TODO check: valueValidTo exists if next change exists to that
#  property. and the date matches.

# TODO check: revision types

# TODO test bot reverts on local dataset.

# Further info can be googled: https://en.wikipedia.org/wiki/Wikipedia:Revision_id
# https://www.mediawiki.org/wiki/Manual:Database_layout
