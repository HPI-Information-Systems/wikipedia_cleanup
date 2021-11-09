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
