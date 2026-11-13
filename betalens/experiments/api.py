"""Read-only HTTP views over an existing experiment archive."""
from __future__ import annotations

from contextlib import asynccontextmanager
import base64
import io
import json
import re

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from .models import ArchiveError, MissingArtifactError, json_value
from .store import ExperimentStore
from .catalog import factor_catalog, factor_history
from .presentation import ArchiveViews


def _json_object(value):
    try:
        result = json.loads(value) if value else {}
    except (ValueError, TypeError) as exc:
        raise ArchiveError("Filter must be a JSON object") from exc
    if not isinstance(result, dict):
        raise ArchiveError("Filter must be a JSON object")
    return result


def _frame(value):
    frame = value.to_frame(name=value.name or "value") if isinstance(value, pd.Series) else value.copy()
    # Include original row labels without colliding with a column of the same name.
    names = [str(n) if n is not None else f"index_{i}" for i, n in enumerate(frame.index.names)]
    used = set(map(str, frame.columns))
    for i, name in enumerate(names):
        while name in used:
            name = "index_" + name
        names[i] = name
        used.add(name)
    frame.index.names = names
    frame = frame.reset_index()
    frame.columns = [str(c) for c in frame.columns]
    return frame


def _points(series):
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0] if len(series.columns) else pd.Series(dtype=float)
    return [{"time": json_value(i), "value": json_value(v)} for i, v in series.items()]


def _all_records(store, **filters):
    page = 1
    while True:
        response = store.query(**filters, page=page, page_size=1000)
        yield from response["items"]
        if page * 1000 >= response["total"]:
            break
        page += 1


def _artifact_preview(value):
    if isinstance(value, (pd.DataFrame, pd.Series)):
        return {"type": "table", "rows": json_value(_frame(value.head(100)).to_dict("records")), "total": len(value)}
    if isinstance(value, bytes):
        return {"type": "image", "data": base64.b64encode(value).decode()}
    if isinstance(value, dict):
        return {str(k): _artifact_preview(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_artifact_preview(v) for v in value]
    return json_value(value)


def create_app(path):
    @asynccontextmanager
    async def lifespan(app):
        with ExperimentStore(path, readonly=True) as store:
            app.state.store = store
            app.state.views = ArchiveViews(store)
            yield

    app = FastAPI(title="Betalens 实验档案", version="1.0.0", lifespan=lifespan)

    @app.exception_handler(KeyError)
    async def missing_record(request, exc):
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(ArchiveError)
    async def invalid_archive(request, exc):
        return JSONResponse(status_code=404 if isinstance(exc, MissingArtifactError) else 400, content={"detail": str(exc)})

    @app.get("/api/health")
    def health():
        return {"status": "ok", "readonly": True}

    @app.get("/api/factors")
    def factors(request: Request, search: str = ""):
        return request.app.state.store.factors(search=search)

    @app.get("/api/factors/{factor_id}")
    def factor(request: Request, factor_id: str):
        return request.app.state.store.get_factor(factor_id)

    @app.get("/api/catalog")
    def catalog(request: Request, basis: str = "latest_full", search: str = "", category: str = "",
                strategy_type: str = "", entry_type: str = "", frequency: str = "", universe: str = "",
                metrics: str = "", sort_by: str = "name", descending: bool = False, page: int = 1, page_size: int = 50):
        return factor_catalog(request.app.state.store, basis=basis, search=search, category=category,
            strategy_type=strategy_type, entry_type=entry_type, frequency=frequency, universe=universe,
            metrics=_json_object(metrics), sort_by=sort_by, descending=descending, page=page, page_size=page_size)

    @app.get("/api/factors/{factor_id}/history")
    def history(request: Request, factor_id: str, page: int = 1, page_size: int = 200):
        return factor_history(request.app.state.store, factor_id, page=page, page_size=page_size)

    @app.get("/api/batches")
    def batches(request: Request):
        return request.app.state.store.batches()

    @app.get("/api/batches/{batch_id}")
    def batch(request: Request, batch_id: str):
        return request.app.state.store.get_batch(batch_id)

    @app.get("/api/experiments")
    def experiments(request: Request, factor_id: str | None = None, batch_id: str | None = None,
                    kind: str | None = None, parameters: str = "", config: str = "", metrics: str = "",
                    tags: str = "", has_raw: bool | None = None, start: str | None = None,
                    end: str | None = None, search: str = "", page: int = 1, page_size: int = 50,
                    sort_by: str = "created_at", descending: bool = True):
        return request.app.state.store.query(factor_id=factor_id, batch_id=batch_id, kind=kind,
            parameters=_json_object(parameters), config=_json_object(config), metrics=_json_object(metrics),
            tags=[s.strip() for s in tags.split(",") if s.strip()], has_raw=has_raw, start=start, end=end,
            search=search, page=page, page_size=page_size, sort_by=sort_by or "created_at", descending=descending)

    @app.get("/api/experiments/{experiment_id}")
    def experiment(request: Request, experiment_id: str):
        return request.app.state.store.get(experiment_id)

    @app.get("/api/experiments/{experiment_id}/result")
    def dashboard_result(request: Request, experiment_id: str, scope: str = "total"):
        return request.app.state.views.get(experiment_id, scope).payload()

    @app.get("/api/experiments/{experiment_id}/profiling")
    def dashboard_profile(request: Request, experiment_id: str, scope: str = "total", date_from: str | None = None, date_to: str | None = None):
        return request.app.state.views.get(experiment_id, scope).profile(date_from, date_to)

    @app.get("/api/experiments/{experiment_id}/result-table/{kind}")
    def dashboard_table(request: Request, experiment_id: str, kind: str, scope: str = "total", page: int = 1,
                        size: int = 50, query: str = "", date_from: str | None = None, date_to: str | None = None):
        filters = {key[7:]: value for key, value in request.query_params.items() if key.startswith("filter.") and value}
        return request.app.state.views.get(experiment_id, scope).table(kind, page=page, size=size,
            query=query, filters=filters, date_from=date_from, date_to=date_to)

    @app.get("/api/experiments/{experiment_id}/result-download/{kind}")
    def dashboard_download(request: Request, experiment_id: str, kind: str, scope: str = "total"):
        body, media, suffix = request.app.state.views.get(experiment_id, scope).download(kind)
        return Response(body, media_type=media, headers={"Content-Disposition": f'attachment; filename="{experiment_id}_{kind}.{suffix}"'})

    @app.get("/api/experiments/{experiment_id}/charts")
    def charts(request: Request, experiment_id: str, scope: str = "total"):
        store = request.app.state.store
        doc = store.get(experiment_id)
        prefix = "" if scope == "total" else f"scopes/{scope}/"
        def read(name):
            for candidate in (prefix + name, prefix + "analysis/" + name):
                if candidate in doc["tables"]:
                    return store.read_table(experiment_id, candidate)
            return None
        nav = read("nav")
        if nav is None:
            raise MissingArtifactError(f"No NAV captured for scope {scope!r}")
        if isinstance(nav, pd.DataFrame):
            nav = nav.iloc[:, 0]
        returns = read("daily_returns")
        if returns is None:
            returns = nav.pct_change(fill_method=None).dropna()
        monthly = returns.groupby(pd.to_datetime(returns.index).strftime("%Y-%m")).apply(lambda x: (1 + x).prod() - 1)
        weight = read("daily_weight")
        if weight is None:
            weight = read("weight")
        holdings = []
        if weight is not None and not weight.empty:
            last = weight.iloc[-1].dropna()
            last = last.loc[last.abs().sort_values(ascending=False).index].head(20)
            holdings = [{"code": str(k), "weight": json_value(v)} for k, v in last.items()]
        metrics = doc["metadata"].get("scope_metrics", {}).get(scope, doc["metrics"])
        return {"scope": scope, "metrics": metrics, "nav": _points(nav),
                "drawdown": _points(nav / nav.cummax() - 1), "returns": _points(returns),
                "monthly": _points(monthly), "holdings": holdings,
                "metric_basis": "captured backtest" if doc["metadata"].get("scope_metrics") else "archived analysis"}

    @app.get("/api/experiments/{experiment_id}/table")
    def table(request: Request, experiment_id: str, name: str, page: int = Query(1, ge=1),
              page_size: int = Query(50, ge=1, le=500), search: str = "", filters: str = "",
              columns: str = "", start: str | None = None, end: str | None = None):
        store = request.app.state.store
        doc = store.get(experiment_id)
        if name not in doc["tables"]:
            raise MissingArtifactError(name)
        conditions = _json_object(filters)
        projection = [c for c in columns.split(",") if c] if columns else None
        if search or conditions or start or end:
            frame = _frame(store.read_table(experiment_id, name, columns=projection, start=start, end=end))
            if search:
                frame = frame.loc[frame.astype(str).apply(lambda col: col.str.contains(search, regex=False)).any(axis=1)]
            for key, target in conditions.items():
                if key not in frame:
                    raise ArchiveError(f"Unknown table filter column: {key}")
                frame = frame.loc[frame[key] == target]
            total = len(frame)
            frame = frame.iloc[(page - 1) * page_size:page * page_size]
        else:
            total = doc["tables"][name]["rows"]
            frame = _frame(store.read_table(experiment_id, name, offset=(page - 1) * page_size, limit=page_size, columns=projection))
        return {"columns": list(frame.columns), "rows": json_value(frame.to_dict("records")),
                "total": total, "page": page, "page_size": page_size}

    @app.get("/api/experiments/{experiment_id}/download")
    def download(request: Request, experiment_id: str, name: str, format: str = "csv"):
        value = request.app.state.store.read_table(experiment_id, name)
        frame = value.to_frame() if isinstance(value, pd.Series) else value
        if format == "csv":
            body, media = frame.to_csv().encode("utf-8-sig"), "text/csv; charset=utf-8"
        elif format == "parquet":
            buffer = io.BytesIO()
            frame.to_parquet(buffer, index=True)
            body, media = buffer.getvalue(), "application/vnd.apache.parquet"
        else:
            raise ArchiveError("Download format must be csv or parquet")
        filename = experiment_id + "_" + re.sub(r"[^a-zA-Z0-9_-]", "_", name) + "." + format
        return Response(body, media_type=media, headers={"Content-Disposition": f'attachment; filename="{filename}"'})

    @app.get("/api/experiments/{experiment_id}/extras")
    def extras(request: Request, experiment_id: str):
        return _artifact_preview(request.app.state.store.read_extras(experiment_id))

    @app.get("/api/compare")
    def compare(request: Request, ids: str, scope: str = "total"):
        identities = list(dict.fromkeys(s for s in ids.split(",") if s))
        if not 1 <= len(identities) <= 20:
            raise ArchiveError("Select between 1 and 20 experiments")
        result = []
        for identity in identities:
            doc = request.app.state.store.get(identity)
            factor = request.app.state.store.get_factor(doc["factor_id"])
            try:
                chart = charts(request, identity, scope)
            except MissingArtifactError:
                chart = None
            result.append({"experiment": doc, "factor_name": factor["name"], "charts": chart})
        return result

    @app.get("/api/mining/heatmap")
    def heatmap(request: Request, factor_id: str | None = None, batch_id: str | None = None,
                x: str = "", y: str = "", metric: str = "sharpe", stage: str = "", window_scheme: str = ""):
        rows, parameter_names = [], set()
        for doc in _all_records(request.app.state.store, factor_id=factor_id, batch_id=batch_id, kind="mining"):
            if "mining/window_results" not in doc["tables"]:
                continue
            frame = request.app.state.store.read_table(doc["id"], "mining/window_results")
            parameter_names.update(doc["parameters"])
            for row in frame.to_dict("records"):
                row["_parameters"] = doc["parameters"]
                rows.append(row)
        params = sorted(parameter_names)
        if x and x not in params or y and y != "__all__" and y not in params:
            raise ArchiveError("Unknown heatmap parameter")
        x = x or (params[0] if params else "")
        y = "" if y == "__all__" else y or (params[1] if len(params) > 1 else "")
        stages = sorted({str(r.get("stage", "")) for r in rows})
        schemes = sorted({"/".join(str(r.get("window_id", "")).split("/")[:2]) for r in rows})
        cells = []
        def axis(value):
            return json.dumps(json_value(value), ensure_ascii=False, sort_keys=True) if isinstance(value, (dict, list, tuple)) else value
        for row in rows:
            if row.get("error") is not None and pd.notna(row.get("error")):
                continue
            if stage and row.get("stage") != stage:
                continue
            if window_scheme and "/".join(str(row.get("window_id", "")).split("/")[:2]) != window_scheme:
                continue
            value = row.get(metric)
            if value is None or pd.isna(value):
                continue
            cells.append({"x": axis(row["_parameters"].get(x)) if x else "全部",
                          "y": axis(row["_parameters"].get(y)) if y else "全部", "value": float(value)})
        frame = pd.DataFrame(cells, columns=["x", "y", "value"])
        if frame.empty:
            xs, ys, z, counts = [], [], [], []
        else:
            grouped = frame.groupby(["y", "x"], dropna=False)["value"].agg(["mean", "count"])
            values = grouped["mean"].unstack("x")
            count_frame = grouped["count"].unstack("x").reindex_like(values)
            xs, ys = list(values.columns), list(values.index)
            z, counts = values.to_numpy().tolist(), count_frame.to_numpy().tolist()
        return json_value({"parameters": params, "x_parameter": x, "y_parameter": y,
            "x": xs, "y": ys, "z": z, "counts": counts, "stages": stages,
            "window_schemes": schemes, "metric": metric, "aggregate": "mean",
            "basis": "各搜索阶段及窗口原始指标的算术平均；保留重复评价记录"})

    from pathlib import Path
    web = Path(__file__).with_name("web")
    if (web / "index.html").exists():
        app.mount("/", StaticFiles(directory=web, html=True), name="viewer")
    return app
