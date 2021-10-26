from dataclasses import dataclass
from typing import Dict, Optional, Sequence


@dataclass(frozen=True)
class InfoboxProperty:
    propertyType: str
    name: str


@dataclass(frozen=True)
class InfoboxChange:
    property: InfoboxProperty
    valueValidTo: Optional[str] = None
    currentValue: Optional[str] = None
    previousValue: Optional[str] = None


@dataclass(frozen=True)
class User:
    username: str
    id: int


@dataclass(frozen=True)
class InfoboxRevision:
    revisionId: int
    pageTitle: str
    changes: Sequence[InfoboxChange]
    validFrom: str
    attributes: Optional[Dict[str, str]]
    pageID: int
    revisionType: Optional[str]
    key: str
    template: Optional[str] = None
    position: Optional[int] = None
    user: Optional[User] = None
    validTo: Optional[str] = None


@dataclass(frozen=True)
class InfoboxRevisionHistory:
    key: str
    revisions: Sequence[InfoboxRevision]
