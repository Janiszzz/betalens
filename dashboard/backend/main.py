from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

from .factors import clear_factor_cache, discover_factors, get_factor_detail
from .runs import manager
from .schemas import FactorDetail, FactorSummary, RunCreated, RunRequest, RunState
from .serialization import build_downloads


app = FastAPI(title="betalens dashboard", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/factors", response_model=list[FactorSummary])
def list_factors(refresh: bool = False):
    if refresh:
        clear_factor_cache()
    return list(discover_factors())


@app.get("/api/factors/{factor_class}/{name}", response_model=FactorDetail)
def factor_detail(factor_class: str, name: str):
    try:
        return get_factor_detail(factor_class, name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/runs", response_model=RunCreated)
def create_run(request: RunRequest):
    try:
        run = manager.create(request)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return RunCreated(run_id=run.run_id)


@app.get("/api/runs/{run_id}", response_model=RunState)
def run_state(run_id: str):
    try:
        return manager.get(run_id).to_state()
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="run not found") from exc


@app.get("/api/runs/{run_id}/result")
def run_result(run_id: str):
    try:
        run = manager.get(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="run not found") from exc
    if run.status != "completed":
        raise HTTPException(status_code=409, detail=f"run is {run.status}")
    try:
        return manager.serialize_result(run_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/runs/{run_id}/logs")
async def run_logs(run_id: str):
    try:
        run = manager.get(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="run not found") from exc

    async def event_stream():
        cursor = 0
        while True:
            current = run.log
            if len(current) > cursor:
                chunk = current[cursor:]
                cursor = len(current)
                payload = json.dumps({"chunk": chunk}, ensure_ascii=False)
                yield f"event: log\ndata: {payload}\n\n"
            if run.status in ("completed", "failed") and cursor >= len(run.log):
                break
            await asyncio.sleep(0.5)
        payload = json.dumps({"status": run.status}, ensure_ascii=False)
        yield f"event: close\ndata: {payload}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/api/runs/{run_id}/download/{kind}")
def download(run_id: str, kind: str):
    try:
        run = manager.get(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="run not found") from exc
    if not run.factor_dir:
        raise HTTPException(status_code=404, detail="no output directory for this run")
    downloads = build_downloads(Path(run.factor_dir), run.name)
    item = downloads.get(kind)
    if item is None:
        raise HTTPException(status_code=404, detail="unknown download kind")
    path = Path(item["path"])
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"{kind} file not found")
    return FileResponse(path, filename=path.name)
