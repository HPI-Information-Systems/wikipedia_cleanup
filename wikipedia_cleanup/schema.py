from datetime import datetime
from typing import Dict, Optional, Sequence

from pydantic import BaseModel


class InfoboxProperty(BaseModel):
    propertyType: Optional[str]
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
    revisionId: int
    pageTitle: str
    changes: Sequence[InfoboxChange]
    validFrom: datetime
    attributes: Optional[Dict[str, str]]
    pageID: int
    revisionType: Optional[str]
    key: str
    template: Optional[str] = None
    position: Optional[int] = None
    user: Optional[User] = None
    validTo: Optional[datetime] = None


class InfoboxRevisionHistory(BaseModel):
    key: str
    revisions: Sequence[InfoboxRevision]
