"""Offline adapter for the canonical Dashboard result contract."""
from __future__ import annotations

from collections import OrderedDict
from itertools import islice
import io
import threading
from types import SimpleNamespace

import pandas as pd

from betalens.analyst import presentation as shared
from .catalog import METRICS, summary_values
from .models import ArchiveError, json_value


_FIELDS = (
    "nav", "daily_returns", "daily_pnl_total", "daily_position_value", "daily_amount",
    "daily_weight", "actual_weight", "weight", "position", "equivalent_position", "daily_pnl",
    "daily_raw_price", "daily_price", "cost_price", "cost_ret", "rebalance_log", "trade_cycles",
    "risk_events", "synthetic_settlements", "research_group_nav",
)


class ArchiveView:
    def __init__(self, store, identity, scope="total"):
        self.store, self.identity, self.scope = store, identity, scope
        self.doc = store.get(identity)
        self.prefix = "" if scope == "total" else f"scopes/{scope}/"
        available = self.doc["metadata"].get("available_scopes", [])
        if scope != "total" and scope not in available:
            raise ArchiveError(f"Unknown captured scope: {scope}")
        self.metadata = self.doc["metadata"].get("scope_metadata", {}).get(scope, self.doc["metadata"])
        self.names = self.metadata.get("name_map", {})
        self.bt = SimpleNamespace(**{key: value for key, value in self.metadata.items() if key not in _FIELDS})
        for field in _FIELDS:
            value = self.read(field)
            if value is not None:
                setattr(self.bt, field, value)
        # Analyst-only snapshots use daily weights rather than target weights.
        if not hasattr(self.bt, "daily_weight"):
            value = self.read("analysis/weight")
            if value is not None:
                self.bt.daily_weight = value
        if not hasattr(self.bt, "daily_pnl_total") and getattr(self.bt, "daily_pnl", None) is not None:
            self.bt.daily_pnl_total = self.bt.daily_pnl.sum(axis=1)
        self.bt.calculation_version = self.metadata.get("calculation_version", "archive")
        self.factor_values = self.read("factor_values")
        self.extras = store.read_extras(identity) if self.doc.get("extras_available") else {}
        if self.prefix:
            self.extras = self.extras.get(scope, {})
        self.trade_rows = shared.build_trade_table(self.bt)
        self._payload = None

    def read(self, name):
        for candidate in (self.prefix + name, self.prefix + "analysis/" + name):
            if candidate in self.doc["tables"]:
                return self.store.read_table(self.identity, candidate)
        return None

    @property
    def has_nav(self):
        return getattr(self.bt, "nav", None) is not None

    def _position_count(self, bt=None):
        bt = bt or self.bt
        frames = [getattr(bt, name, None) for name in ("position", "daily_position_value", "daily_pnl")]
        frames = [frame for frame in frames if frame is not None and not frame.empty]
        if not frames:
            return 0
        dates = sorted(set().union(*(set(frame.index) for frame in frames)))
        codes = sorted(set().union(*(set(frame.columns) for frame in frames)))
        position = getattr(bt, "position", None)
        if position is None or position.empty:
            return len(dates) * len(codes)
        return int(position.reindex(index=dates, columns=codes).ne(0).to_numpy().sum())

    def _table_metas(self):
        with shared.archived_names(self.names):
            first = next(shared.iter_position_records(self.bt), None)
        return {"trades": shared._table_meta(self.trade_rows),
                "positions": {"total": self._position_count(), "columns": list(first) if first else []}}

    def payload(self):
        if self._payload is not None:
            return self._payload
        root = self.doc.get("factor_snapshot") or self.store.get_factor(self.doc["factor_id"])
        config, metadata = self.doc["config"], self.doc["metadata"]
        parameters = {**config.get("run", {}), **self.doc["parameters"]}
        state = {"run_id": self.identity, "status": "completed", "factor_class": root["category"],
                 "name": root["name"], "started_at": self.doc["created_at"], "finished_at": self.doc["created_at"],
                 "elapsed_seconds": 0, "error": None, "log_size": 0}
        original = metadata.get("scope_metrics", {}).get(self.scope, self.doc["metrics"])
        normalized = dict(original)
        values, _, basis = summary_values({**self.doc, "metrics": original,
            "metadata": {"scope_metrics": {"total": original}} if self.has_nav else {}})
        for key, label in (("ic", "IC均值"), ("ic_ir", "ICIR"), ("annual_return", "年化收益"),
                ("sharpe", "夏普比率"), ("max_drawdown", "最大回撤"), ("turnover", "单边换手率(年化)"),
                ("win_rate", "交易胜率"), ("calmar", "卡玛比率"), ("total_return", "累计收益")):
            if label not in normalized:
                normalized[label] = values[key]
        run = SimpleNamespace(backtest=self.bt, analyst=SimpleNamespace(an=SimpleNamespace(summary=lambda: normalized)),
            factor_class=root["category"], name=root["name"], parameters=parameters,
            compute_kwargs=config.get("factor_spec", {}).get("compute_kwargs", {}),
            to_state=lambda: SimpleNamespace(model_dump=lambda: state))
        with shared.archived_names(self.names):
            payload = shared.build_result_payload(run, self._table_metas(), self.factor_values,
                self.read("pit_validation"), self.read("neutralize_stats"), self.extras.get("chart_data"))
        if not self.has_nav:
            # Preserve window aggregation labels for candidates without raw data.
            payload["metrics"] = [{"label": key, "value": json_value(value),
                "format": "percent" if key in shared.PERCENT_METRICS else "number", "group": "raw"}
                for key, value in original.items()]
        payload["downloads"] = {"config": {"path": None, "exists": True},
                                "dump": {"path": None, "exists": self.has_nav},
                                "report": {"path": None, "exists": self.has_nav},
                                "html": {"path": None, "exists": self.has_nav}}
        payload["availability"] = {"raw": self.has_nav, "metric_basis": basis,
                                   "capture_error": self.doc.get("capture_error")}
        self._payload = json_value(payload)
        return self._payload

    def profile(self, date_from=None, date_to=None):
        return shared.build_factor_profile_payload(self.factor_values, date_from, date_to)

    def table(self, kind, *, page=1, size=50, query="", filters=None, date_from=None, date_to=None):
        if page < 1 or not 1 <= size <= 500:
            raise ArchiveError("Invalid result-table pagination")
        if kind == "trades":
            return shared.paginate_table(self.trade_rows, page=page, size=size, query=query, filters=filters)
        if kind != "positions":
            raise ArchiveError("Table kind must be trades or positions")
        bt = self.bt
        if date_from or date_to:
            bt = SimpleNamespace(**vars(self.bt))
            for field in _FIELDS:
                frame = getattr(bt, field, None)
                if isinstance(frame, (pd.Series, pd.DataFrame)) and isinstance(frame.index, pd.DatetimeIndex):
                    dates = frame.index.strftime("%Y-%m-%d")
                    mask = pd.Series(True, index=frame.index)
                    if date_from:
                        mask &= dates >= date_from
                    if date_to:
                        mask &= dates <= date_to
                    setattr(bt, field, frame.loc[mask.to_numpy()])
        start, stop = (page - 1) * size, page * size
        with shared.archived_names(self.names):
            records = shared.iter_position_records(bt)
            if not query and not filters:
                total = self._position_count(bt)
                rows = list(islice(records, start, stop))
            else:
                rows, total = [], 0
                for row in records:
                    if any(str(row.get(key, "")) != str(value) for key, value in (filters or {}).items()):
                        continue
                    if query and query.lower() not in " ".join(str(value) for value in row.values()).lower():
                        continue
                    if start <= total < stop:
                        rows.append(row)
                    total += 1
        return {"rows": rows, "total": total, "page": page, "size": size, "pages": (total + size - 1) // size}

    def download(self, kind):
        if kind == "config":
            import yaml
            return yaml.safe_dump(self.doc["config"], allow_unicode=True).encode(), "application/yaml", "yaml"
        if not self.has_nav:
            raise ArchiveError("No captured backtest is available for this export")
        buffer = io.BytesIO()
        if kind == "report":
            self.store.to_analyst(self.identity, scope=self.scope).to_excel(buffer)
        elif kind == "dump":
            with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                pd.DataFrame({"key": list(self.metadata), "value": [str(v) for v in self.metadata.values()]}).to_excel(writer, sheet_name="meta", index=False)
                for name in self.doc["tables"]:
                    if not name.startswith(self.prefix):
                        continue
                    local = name[len(self.prefix):]
                    if "/" in local:
                        continue
                    frame = self.store.read_table(self.identity, name)
                    frame.to_excel(writer, sheet_name=local[:31])
        elif kind == "html":
            from html import escape
            analyst = self.store.to_analyst(self.identity, scope=self.scope)
            html = "<!doctype html><meta charset='utf-8'><h1>" + escape(analyst.name) + "</h1>" + analyst.summary_df().to_html(index=False)
            for index, (title, figure) in enumerate(analyst.interactive_plots().items()):
                html += "<h2>" + escape(title) + "</h2>" + figure.to_html(full_html=False, include_plotlyjs=True if index == 0 else False)
            return html.encode(), "text/html; charset=utf-8", "html"
        else:
            raise ArchiveError("Unknown result export")
        return buffer.getvalue(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "xlsx"


class ArchiveViews:
    """One immutable snapshot cached per server, bounded independently of run count."""
    def __init__(self, store):
        self.store, self.lock = store, threading.RLock()
        self.cache = OrderedDict()

    def get(self, identity, scope="total"):
        with self.lock:
            key = (identity, scope)
            if key not in self.cache:
                self.cache.clear()
                self.cache[key] = ArchiveView(self.store, identity, scope)
            return self.cache[key]
