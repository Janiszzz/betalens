"""Portable inputs and validation shared by archival adapters."""
from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
from typing import Any


class ArchiveError(ValueError):
    """An experiment cannot be represented or a database is incompatible."""


class ConflictError(ArchiveError):
    """An immutable identity already exists with different contents."""


class MissingArtifactError(ArchiveError):
    """The requested original data were not captured."""


@dataclasses.dataclass
class ExperimentInput:
    """One completed result plus the context absent from runtime objects."""

    result: Any
    factor_id: str
    config: dict = dataclasses.field(default_factory=dict)
    parameters: dict = dataclasses.field(default_factory=dict)
    metadata: dict = dataclasses.field(default_factory=dict)
    source_id: str | None = None
    tags: list[str] = dataclasses.field(default_factory=list)


def json_value(value):
    """Normalize supported scientific scalars without arbitrary object execution."""
    if type(value).__name__ in {"NAType", "NaTType"}:
        return None
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (dt.datetime, dt.date, dt.time)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return json_value(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    # NumPy/pandas objects are intentionally recognized without importing them
    # on the lightweight package import path.
    if type(value).__module__.split(".")[0] in {"numpy", "pandas"}:
        if type(value).__name__ in {"NAType", "NaTType"}:
            return None
        if hasattr(value, "tolist"):
            return json_value(value.tolist())
        if hasattr(value, "item"):
            return json_value(value.item())
        if hasattr(value, "isoformat"):
            return value.isoformat()
    raise ArchiveError(f"Unsupported metadata type: {type(value).__name__}")


def dumps(value) -> bytes:
    return json.dumps(json_value(value), ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def loads(value):
    return json.loads(bytes(value)) if value is not None else None


def digest(value) -> str:
    return hashlib.sha256(dumps(value)).hexdigest()


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


_MISSING = object()


def lookup(config, path):
    value = config
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return _MISSING
        value = value[part]
    return value


def matches(config, locks):
    return all(lookup(config, path) is not _MISSING and lookup(config, path) == value
               for path, value in locks.items())
