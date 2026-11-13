"""Factor-level metric rows from archive summaries, without loading raw tables."""
from __future__ import annotations

import math

from .models import ArchiveError, lookup, matches, _MISSING
from .store import _metric_match


METRICS = [
    ("ic", "IC均值", "number", ("IC均值", "IC", "ic_mean")),
    ("rank_ic", "Rank IC", "number", ("mean_rank_ic_median", "rank_ic_median", "mean_rank_ic", "rank_ic")),
    ("ic_ir", "ICIR", "number", ("ICIR", "ic_ir_median", "ic_ir")),
    ("annual_return", "年化收益", "percent", ("年化收益", "策略年化收益", "ann_ret_median", "ann_ret")),
    ("sharpe", "夏普比率", "number", ("夏普比率", "sharpe_median", "sharpe")),
    ("max_drawdown", "最大回撤", "percent", ("最大回撤", "max_dd_median", "max_dd")),
    ("turnover", "年化单边换手", "percent", ("单边换手率(年化)", "turnover_median", "turnover")),
    ("win_rate", "交易胜率", "percent", ("交易胜率", "win_rate_median", "win_rate")),
    ("calmar", "卡玛比率", "number", ("卡玛比率", "calmar_median", "calmar")),
    ("total_return", "累计收益", "percent", ("累计收益", "策略收益", "total_ret_median", "total_ret")),
    ("information_ratio", "信息比率", "number", ("信息比率",)),
    ("excess_annual_return", "超额年化收益", "percent", ("超额年化收益",)),
]


def all_experiments(store):
    page = 1
    while True:
        result = store.query(page=page, page_size=1000)
        yield from result["items"]
        if page * 1000 >= result["total"]:
            break
        page += 1


def run_setting(doc, key):
    paths = {"frequency": ("run.rebal_freq", "run.rebalance_frequency"),
             "universe": ("factor_spec.index_code", "run.index_code")}
    for path in paths[key]:
        value = lookup(doc["config"], path)
        if value is not _MISSING and value is not None:
            return str(value)
    return ""


def period(doc):
    config = doc.get("config", {})
    run = config.get("run", {})
    span = run.get("span") or []
    metadata = doc.get("metadata", {})
    start = run.get("start_date") or (span[0] if len(span) == 2 else None) or metadata.get("start")
    end = run.get("end_date") or (span[1] if len(span) == 2 else None) or metadata.get("end")
    return {"start": str(start)[:10] if start else None, "end": str(end)[:10] if end else None}


def summary_values(doc):
    full = doc.get("metadata", {}).get("scope_metrics", {}).get("total")
    source = full if full is not None else doc.get("metrics", {})
    values, keys = {}, {}
    for key, _, _, aliases in METRICS:
        name = next((name for name in aliases if isinstance(source.get(name), (float, int))
                     and not isinstance(source.get(name), bool) and math.isfinite(source[name])), None)
        values[key] = float(source[name]) if name else None
        keys[key] = name
    basis = "完整回测指标" if full is not None or doc.get("kind") != "mining" else "窗口聚合指标"
    return values, keys, basis


def factor_catalog(store, *, basis="latest_full", search="", category="", strategy_type="", entry_type="",
                   frequency="", universe="", metrics=None, sort_by="name", descending=False,
                   page=1, page_size=50):
    if basis not in {"latest_full", "latest"}:
        raise ArchiveError("Metric basis must be latest_full or latest")
    if page < 1 or not 1 <= page_size <= 500:
        raise ArchiveError("Invalid catalog pagination")
    metric_keys = {key for key, *_ in METRICS}
    if set(metrics or {}) - metric_keys or sort_by not in metric_keys | {"name", "category", "experiment_count", "created_at"}:
        raise ArchiveError("Unknown catalog metric or sort field")
    groups, frequencies, universes = {}, set(), set()
    for doc in all_experiments(store):
        freq, pool = run_setting(doc, "frequency"), run_setting(doc, "universe")
        if freq:
            frequencies.add(freq)
        if pool:
            universes.add(pool)
        if frequency and frequency != freq or universe and universe != pool:
            continue
        groups.setdefault(doc["factor_id"], []).append(doc)
    rows = []
    for factor in store.factors(search=search):
        if category and factor["category"] != category or strategy_type and factor["strategy_type"] != strategy_type:
            continue
        if entry_type == "root" and factor["parent_id"] or entry_type == "derived" and not factor["parent_id"]:
            continue
        candidates = [doc for doc in groups.get(factor["parent_id"] or factor["id"], [])
                      if matches({**doc["config"], "parameters": doc["parameters"]}, factor["locks"])]
        # Store.query already orders each root's records newest first.
        full = [doc for doc in candidates if doc["has_raw"] and not doc.get("capture_error")
                and doc["metadata"].get("compute_status", "complete") == "complete"]
        chosen = (full if basis == "latest_full" and full else candidates)
        source = chosen[0] if chosen else None
        values, keys, metric_basis = summary_values(source) if source else ({key: None for key in metric_keys}, {}, "无实验")
        if not all(_metric_match(values.get(key), rule) for key, rule in (metrics or {}).items()):
            continue
        rows.append({**factor, "experiment_count": len(candidates), "metrics": values,
                     "metric_keys": keys, "metric_basis": metric_basis,
                     "source": None if source is None else {"experiment_id": source["id"],
                         "created_at": source["created_at"], "has_raw": source["has_raw"], "kind": source["kind"],
                         "parameters": source["parameters"], "period": period(source),
                         "frequency": run_setting(source, "frequency"), "universe": run_setting(source, "universe")}})
    def value(row):
        if sort_by in metric_keys:
            return row["metrics"][sort_by]
        if sort_by == "created_at":
            return row["source"]["created_at"] if row["source"] else None
        return row[sort_by]
    present, missing = [r for r in rows if value(r) is not None], [r for r in rows if value(r) is None]
    present.sort(key=lambda r: (value(r), r["key"]), reverse=descending)
    rows = present + sorted(missing, key=lambda r: r["name"])
    start = (page - 1) * page_size
    return {"items": rows[start:start + page_size], "total": len(rows), "page": page, "page_size": page_size,
            "basis": basis, "metrics": [{"key": key, "label": label, "format": format} for key, label, format, _ in METRICS],
            "frequencies": sorted(frequencies), "universes": sorted(universes)}


def factor_history(store, factor_id, *, page=1, page_size=200):
    records = store.query(factor_id=factor_id, page=page, page_size=page_size)
    return {**records, "items": [{"id": doc["id"], "created_at": doc["created_at"], "kind": doc["kind"],
        "parameters": doc["parameters"], "has_raw": doc["has_raw"], "capture_error": doc.get("capture_error"),
        "period": period(doc), "available_scopes": doc["metadata"].get("available_scopes", ["total"] if doc["has_raw"] else [])}
        for doc in records["items"]]}
