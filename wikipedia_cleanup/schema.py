from datetime import datetime
from typing import Dict, Optional, Sequence

from pydantic import BaseModel


class InfoboxProperty(BaseModel):
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
    revisionType: Optional[str]  # "CREATE", probably: "DELETE" ""
    key: str  # infobox-key, unrelated to PageID and revisionId
    template: Optional[
        str
    ] = None  # name of the used template / category e.g. "infobox person"
    position: Optional[int] = None  # i-th infobox of the page in this revision
    user: Optional[User] = None
    validTo: Optional[datetime] = None  # date of the next revision


# TODO: check: InfoboxRevisionHistory.key is globally unique
#  and all changes are written in one file.

# TODO look for InfoboxProperty.propertyType == "meta" and
#  look if these are present in source code of wikipedia.

# TODO check: valueValidTo exists if next change exists to that
#  property. and the date matches.

# TODO check: revision types

# TODO test bot reverts on local dataset.

# Further info can be googled: https://en.wikipedia.org/wiki/Wikipedia:Revision_id
# https://www.mediawiki.org/wiki/Manual:Database_layout
