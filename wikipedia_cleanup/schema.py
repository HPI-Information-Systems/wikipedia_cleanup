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
    page_id: int
    property_name: str
    value_valid_to: Optional[datetime] = None
    value_valid_from: datetime
    current_value: Optional[str] = None
    previous_value: Optional[str] = None

    page_title: str
    revision_id: int
    edit_type: EditType
    property_type: PropertyType
    comment: Optional[str] = None
    infobox_key: str
    username: Optional[str] = None
    user_id: Optional[str] = None
    position: Optional[int] = None
    template: Optional[str] = None
    revision_valid_to: Optional[datetime] = None


# Knowledge over json-files:
# Each file consists of the revisions / changes of one or more pages.
# All revisions of a page are in one file.
# Since a page can have multiple infoboxes, one json-file
# can contain all changes of multiple infoboxes.


"""class InfoboxProperty(BaseModel):
    propertyType: Optional[str]  # attribute, meta
    name: str


class InfoboxChange(BaseModel):
    property: InfoboxProperty
    valueValidTo: Optional[datetime] = None
    currentValue: Optional[str] = None
    previousValue: Optional[str] = None


class User(BaseModel):
    username: Optional[str]
    id: Optional[int]


class InfoboxRevision(BaseModel):
    revisionId: int  # ChangeID of a page
    # since a page can contain multiple infoboxes it is not unique.
    pageTitle: str  # self-explanatory: can change, no identifier
    changes: Sequence[InfoboxChange]
    validFrom: datetime  # timestamp of the revision
    attributes: Optional[
        Dict[str, str]
    ]  # snapshot after revision of all properties to value mappings
    pageID: int  # wikipedia pageID: Page key but no infobox id
    revisionType: Optional[str]  # "CREATE", probably: "DELETE", "UPDATE"
    key: str  # infobox-key, unrelated to PageID and revisionId
    template: Optional[
        str
    ] = None  # name of the used template / category e.g. "infobox person"
    position: Optional[int] = None  # i-th infobox of the page in this revision
    user: Optional[User] = None
    validTo: Optional[datetime] = None  # date of the next revision"""


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
