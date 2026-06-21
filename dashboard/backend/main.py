from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

from .eventstudy_dashboard import discover_event_files, run_event_study
from .factors import clear_factor_cache, discover_factors, get_factor_detail
from .runs import manager
from .schemas import (
    EventStudyRequest,
    FactorDetail,
    FactorSummary,
    RunCreated,
    RunRequest,
    RunState,
)
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


@app.get("/api/eventstudy/files")
def eventstudy_files():
    return discover_event_files()


@app.post("/api/eventstudy/run")
def eventstudy_run(request: EventStudyRequest):
    try:
        return run_event_study(request.model_dump())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


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


@app.get("/api/runs/{run_id}/table/{kind}")
def run_table(
    run_id: str,
    kind: str,
    request: Request,
    page: int = Query(1, ge=1),
    size: int = Query(50, ge=1, le=500),
    query: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
):
    if kind not in ("trades", "positions"):
        raise HTTPException(status_code=404, detail="unknown table kind")
    # 形如 ?filter.direction=buy&filter.date=2024-01-02 的列过滤
    filters = {
        key[len("filter.") :]: value
        for key, value in request.query_params.items()
        if key.startswith("filter.") and value
    }
    try:
        return manager.table_page(
            run_id,
            kind,
            page=page,
            size=size,
            query=query,
            filters=filters or None,
            date_from=date_from,
            date_to=date_to,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="run not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


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
