from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


RunStatus = Literal["queued", "running", "completed", "failed"]


class FactorSummary(BaseModel):
    factor_class: str
    name: str
    formula: str = ""
    logic: str = ""
    source: str = ""
    inputs: dict[str, str] = Field(default_factory=dict)
    defaults: dict[str, Any] = Field(default_factory=dict)


class FactorDetail(FactorSummary):
    doc: str = ""
    compute_kwargs: dict[str, Any] = Field(default_factory=dict)
    script_path: str
    factor_dir: str


class RunRequest(BaseModel):
    factor_class: str
    name: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    compute_kwargs: dict[str, Any] = Field(default_factory=dict)


class RunCreated(BaseModel):
    run_id: str


class RunState(BaseModel):
    run_id: str
    status: RunStatus
    factor_class: str
    name: str
    started_at: str | None = None
    finished_at: str | None = None
    elapsed_seconds: float = 0
    error: str | None = None
    log_size: int = 0


class DownloadInfo(BaseModel):
    kind: str
    path: str | None
    exists: bool
