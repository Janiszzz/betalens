"""Data provider abstractions.

Only file providers are implemented in v1. ``ApiProvider`` exists so the GUI
and job records already have a place to route future network updates.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SourceRef:
    source_kind: str
    path: str | None = None
    api_name: str | None = None


class FileProvider:
    source_kind = "file"

    def resolve(self, path: str | Path) -> SourceRef:
        return SourceRef(source_kind=self.source_kind, path=str(Path(path)))


class ApiProvider:
    source_kind = "api"

    def fetch(self, *args, **kwargs):
        raise NotImplementedError("联网 API 更新为预留能力，第一版未实现")

