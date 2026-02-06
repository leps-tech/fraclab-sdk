"""Algorithm manifest specification (FracLabAlgorithmManifestV1)."""

from __future__ import annotations

import re
from typing import Annotated, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, StringConstraints, model_validator

ManifestVersion = Literal["1"]

_SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+([-+][0-9A-Za-z.-]+)?$")

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=256)]
IdStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=128)]
UrlStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2048)]
EmailStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=3, max_length=320)]


class Author(BaseModel):
    """Author information."""

    model_config = ConfigDict(extra="ignore")

    name: NonEmptyStr
    email: Optional[EmailStr] = None
    organization: Optional[NonEmptyStr] = None


class Compatibility(BaseModel):
    """Compatibility gates."""

    model_config = ConfigDict(extra="ignore")

    sdk: Optional[NonEmptyStr] = None
    core: Optional[NonEmptyStr] = None

    @model_validator(mode="after")
    def _validate_semver_like(self) -> "Compatibility":
        for label, v in (("sdk", self.sdk), ("core", self.core)):
            if v is not None and not _SEMVER_RE.match(v):
                raise ValueError(f"requires.{label} must be semver-like (e.g. 1.2.3), got: {v}")
        return self


class FracLabAlgorithmManifestV1(BaseModel):
    """Minimal but complete algorithm manifest."""

    model_config = ConfigDict(extra="allow")

    manifestVersion: ManifestVersion

    algorithmId: IdStr
    name: NonEmptyStr
    summary: NonEmptyStr

    notes: Optional[str] = None
    tags: Optional[List[NonEmptyStr]] = None

    authors: List[Author]

    contractVersion: NonEmptyStr
    codeVersion: NonEmptyStr

    requires: Optional[Compatibility] = None

    repository: Optional[UrlStr] = None
    homepage: Optional[UrlStr] = None
    license: Optional[NonEmptyStr] = None

    @model_validator(mode="after")
    def _validate_minimal(self) -> "FracLabAlgorithmManifestV1":
        if len(self.authors) == 0:
            raise ValueError("authors must contain at least one author")

        if not _SEMVER_RE.match(self.contractVersion):
            raise ValueError(
                f"contractVersion must be semver-like (e.g. 1.2.3), got: {self.contractVersion}"
            )

        return self


__all__ = [
    "ManifestVersion",
    "FracLabAlgorithmManifestV1",
    "Author",
    "Compatibility",
]
