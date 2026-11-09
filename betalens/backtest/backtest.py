"""Target-weight backtesting with drift, paired prices and close-observed risk."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from datetime import date, datetime
import json
import re
import warnings

import numpy as np
import pandas as pd

from betalens.datafeed import Datafeed
from .execution import round_order, share_rule
from .risk import RiskConfig, RiskManager


class BacktestDataError(ValueError):
    """A backtest cannot satisfy its input/data contract."""


class DateMismatchError(BacktestDataError):
    pass


class CodeMismatchError(BacktestDataError):
    pass


def _local(value):
    ts = pd.Timestamp(value)
    if pd.isna(ts):
        raise BacktestDataError("时间不能为空")
    return ts.tz_convert("Asia/Shanghai").tz_localize(None) if ts.tzinfo else ts


def _end_time(value):
    ts = _local(value)
    if (isinstance(value, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", value)) or (
        isinstance(value, date) and not isinstance(value, (datetime, pd.Timestamp))
    ):
        return ts.normalize() + pd.Timedelta(hours=15)
    return ts


def validate_weight_input(weight):
    if not isinstance(weight, pd.DataFrame) or weight.empty:
        raise BacktestDataError("weight 必须是非空 DataFrame")
    if weight.columns.has_duplicates or any(not isinstance(c, str) for c in weight):
        raise BacktestDataError("weight 列名必须是唯一的证券代码字符串")
    if pd.api.types.is_numeric_dtype(weight.index):
        raise BacktestDataError("weight.index 必须为带日期时间的索引，不能为数字")
    try:
        values = weight.to_numpy(dtype=float)
    except (ValueError, TypeError) as exc:
        raise BacktestDataError("weight 必须包含数值权重") from exc
    if not np.isfinite(values).all():
        raise BacktestDataError("weight 不允许 NaN 或 Inf")


class BacktestBase:
    """Execute dated targets; carry holdings between targets without rebalancing.

    ``daily_weight`` is the post-fee closing weight path. ``actual_weight`` is
    the post-execution snapshot path. ``position`` is explicitly an equivalent
    reinvested share count, not a broker share ledger.
    """

    calculation_version = "backtest-v2-drift-risk"

    def __init__(self, weight, symbol, amount, ftc=0.0, ptc=0.0, verbose=True,
                 metric="close", table_name="market_daily_post_adjusted",
                 check_trade_status=True, trade_status_mode="to_cash",
                 trade_status_table="stock_suspension_intervals_versions", *,
                 end_date=None, price_time_map=None, risk=None, **removed):
        if removed:
            raise BacktestDataError(
                f"已移除的/未知回测参数: {', '.join(sorted(removed))}；"
                "删除 time_tolerance、preloaded_*、lot_size；数量规则改为按股票板块确定"
            )
        validate_weight_input(weight)
        self.weight = weight.astype(float).copy()
        try:
            self.weight.index = pd.DatetimeIndex([_local(ts) for ts in weight.index])
        except (ValueError, TypeError) as exc:
            raise BacktestDataError("weight.index 必须能转换为时间戳") from exc
        if self.weight.index.has_duplicates:
            raise BacktestDataError("weight 时间戳不能重复")
        self.weight = self.weight.sort_index()
        self.codes = [c for c in self.weight if c != "cash"]
        self.active_codes = [c for c in self.codes if self.weight[c].ne(0).any()]
        self.initial_amount = float(amount)
        self.ftc, self.ptc = float(ftc), float(ptc)
        if not np.isfinite(self.initial_amount) or self.initial_amount <= 0:
            raise BacktestDataError("amount 必须为有限正数")
        if not all(np.isfinite(x) and x >= 0 for x in (self.ftc, self.ptc)):
            raise BacktestDataError("ftc、ptc 必须为有限非负数")
        self.symbol, self.metric, self.table_name = symbol, metric, table_name
        self.verbose = verbose
        self.check_trade_status, self.trade_status_mode = check_trade_status, trade_status_mode
        self.trade_status_table = trade_status_table
        if trade_status_mode not in {"to_cash", "hold", "redistribute", "as_normal", "report_only"}:
            raise BacktestDataError("无效 trade_status_mode: " + str(trade_status_mode))
        self.risk_config = RiskConfig.parse(risk)
        self._risk = RiskManager(self.risk_config)
        self.price_time_map = {"open": "09:30", "close": "15:00", **(price_time_map or {})}
        self._clocks = {}
        for name, clock in self.price_time_map.items():
            if not isinstance(name, str) or not re.fullmatch(r"[a-z_][a-z0-9_]*", name):
                raise BacktestDataError("价格字段必须使用数据库物理字段名")
            if not isinstance(clock, str) or not re.fullmatch(r"\d{2}:\d{2}(?::\d{2})?", clock):
                raise BacktestDataError("price_time_map 时点必须为 HH:MM[:SS]")
            try:
                parsed = datetime.strptime(clock, "%H:%M:%S" if len(clock) == 8 else "%H:%M")
            except ValueError as exc:
                raise BacktestDataError("无效价格时点: " + clock) from exc
            self._clocks[name] = pd.Timedelta(hours=parsed.hour, minutes=parsed.minute, seconds=parsed.second)
        self._metrics = list(dict.fromkeys([metric, "close"] + ([self.risk_config.execution_metric] if self.risk_config.enabled else [])))
        if any(m not in self._clocks for m in self._metrics):
            raise BacktestDataError("未定义价格字段时点；请在 price_time_map 中显式配置")
        self.start = self.weight.index[0]
        self._explicit_end = _end_time(end_date) if end_date is not None else None
        if self._explicit_end is not None and self._explicit_end < self.start:
            raise DateMismatchError("end_date 不能早于首个目标时间")
        self.data_sources = {"prices": "database", "trade_status": "database" if check_trade_status else "disabled"}
        self.get_trade_status()
        self._load_data()
        self._run()

    def get_trade_status(self):
        """Query once, preserving target timestamps; this only changes targets."""
        self.effective_target_weight = self.weight.copy()
        self.trade_status = None
        self.trade_status_matrix = None
        if self.check_trade_status and self.active_codes:
            db = None
            try:
                db = Datafeed(self.trade_status_table)
                self.trade_status = db.query_trade_status({"codes": self.active_codes,
                    "dates": [ts.strftime("%Y-%m-%d %H:%M:%S.%f") for ts in self.weight.index]})
            except Exception as exc:
                warnings.warn(f"交易状态查询失败，保留目标权重并标记未知: {exc}", UserWarning)
            finally:
                if db is not None:
                    db.close()
            if self.trade_status is not None and not self.trade_status.empty:
                status = self.trade_status.copy()
                status["datetime"] = status["datetime"].map(_local)
                self.trade_status_matrix = status.pivot_table(index="datetime", columns="code", values="value", aggfunc="first").reindex(index=self.weight.index, columns=self.codes)
                self._apply_trade_status()
        self.effective_target_weight["cash"] = 1.0 - self.effective_target_weight[self.codes].sum(axis=1)
        return self.trade_status

    def _apply_trade_status(self):
        if self.trade_status_mode in {"as_normal", "report_only"}:
            return
        for pos, ts in enumerate(self.weight.index):
            suspended = [c for c in self.active_codes
                         if pd.notna(self.trade_status_matrix.at[ts, c])
                         and self.trade_status_matrix.at[ts, c] == 0
                         and self.effective_target_weight.at[ts, c] != 0]
            if not suspended:
                continue
            row = self.effective_target_weight.loc[ts].copy()
            if self.trade_status_mode == "hold" and pos:
                self.effective_target_weight.loc[ts] = self.effective_target_weight.iloc[pos - 1]
                continue
            for code in suspended:
                self.effective_target_weight.at[ts, code] = 0.0
            if self.trade_status_mode == "redistribute":
                for side in (1, -1):
                    side_codes = [c for c in self.codes if row[c] * side > 0]
                    remaining = [c for c in side_codes if c not in suspended]
                    old_total = sum(abs(row[c]) for c in side_codes)
                    new_total = sum(abs(row[c]) for c in remaining)
                    if new_total:
                        for c in remaining:
                            self.effective_target_weight.at[ts, c] *= old_total / new_total

    def _load_data(self):
        db = Datafeed(self.table_name)
        try:
            self.data_sources["pit_policy"] = getattr(db, "pit_policy", "research")
            self.market_bars = db.query_backtest_bars(self.codes, self.start.normalize(), self._explicit_end, self._metrics) if self.codes else pd.DataFrame()
            self._validate_bars()
            self._quote_times = {c: [] for c in self.active_codes}
            for row in self.market_bars.to_dict("records"):
                if row["code"] in self._quote_times and pd.notna(row.get(self.metric)):
                    self._quote_times[row["code"]].append(row["trade_date"] + self._clocks[self.metric])
            if self._explicit_end is None:
                candidates = [self.weight.index[-1].normalize() + self._clocks["close"]]
                for code in self.active_codes:
                    future = [t for t in self._quote_times[code] if t >= self.weight.index[-1]]
                    if future:
                        candidates.append(max(future[0], future[0].normalize() + self._clocks["close"]))
                self.end = max(candidates)
            else:
                self.end = self._explicit_end
            self.trade_days = pd.DatetimeIndex(db.query_backtest_calendar(self.start.normalize(), self.end.normalize())).normalize().sort_values().unique()
        finally:
            db.close()
        self._markets = {r["code"]: r.get("market") for r in self.market_bars.to_dict("records")}
        self.price_audit = self.market_bars.copy()

    def _validate_bars(self):
        if not self.codes and self.market_bars.empty:
            return
        required = {"code", "trade_date", "market", "adjustment_factor"}
        required.update(key for m in self._metrics for key in (m, "raw_" + m))
        if not required.issubset(self.market_bars.columns):
            raise BacktestDataError(f"成对行情缺少字段: {sorted(required - set(self.market_bars.columns))}")
        self.market_bars = self.market_bars.copy()
        self.market_bars["trade_date"] = self.market_bars["trade_date"].map(lambda t: _local(t).normalize())
        if self.market_bars.duplicated(["code", "trade_date"]).any():
            raise BacktestDataError("行情存在重复的证券交易日")
        self.market_bars = self.market_bars.sort_values(["trade_date", "code"])
        for row in self.market_bars.to_dict("records"):
            factor = float(row["adjustment_factor"])
            if not np.isfinite(factor) or factor <= 0:
                raise BacktestDataError("复权因子必须为有限正数")
            for metric in self._metrics:
                adjusted, raw = row[metric], row["raw_" + metric]
                if pd.isna(adjusted) and pd.isna(raw):
                    continue
                if not (pd.notna(adjusted) and pd.notna(raw) and np.isfinite(adjusted) and np.isfinite(raw) and adjusted > 0 and raw > 0):
                    raise BacktestDataError("原始与后复权行情必须为成对的有限正数")
                if not np.isclose(adjusted, raw * factor, rtol=1e-8, atol=1e-10):
                    raise BacktestDataError("原始价、后复权价和复权因子不一致")

    def _new_order(self, code, ts, weight, metric, source="strategy", event_id=None, cash_weight=0.0):
        order = dict(order_id=len(self._order_rows) + 1, code=code, target_ts=ts,
                     target_weight=float(weight), metric=metric, source=source,
                     risk_event_id=event_id, cash_weight=cash_weight, status="pending",
                     execution_ts=None)
        self._order_rows.append(order)
        return order

    def _mark(self, code, quote):
        adjusted, raw, factor = quote
        old_price = self._marks[code]
        if old_price > 0:
            before = self._values[code]
            self._values[code] *= adjusted / old_price
            self._pnl[code] += self._values[code] - before
        self._marks[code], self._raw_marks[code], self._factors[code] = adjusted, raw, factor

    def _equity(self):
        return self._cash + sum(self._values.values())

    def _fee(self, quantity, raw):
        return self.ftc + self.ptc * abs(quantity) * raw if quantity else 0.0

    def _execute(self, code, order, quote, quantity, fraction, ts):
        adjusted, raw, factor = quote
        before = self._values[code]
        before_shares = before / raw
        fee = self._fee(quantity, raw)
        transferred = (quantity + fraction) * raw
        self._values[code] += transferred
        if abs(self._values[code]) < 1e-7:
            self._values[code] = 0.0
        self._cash -= transferred + fee
        self._pnl[code] -= fee
        self._fees[code] += fee
        if fraction:
            self._settlement_rows.append(dict(datetime=ts, code=code, shares=fraction,
                price=raw, cash_change=-fraction * raw, source=order["source"],
                order_id=order["order_id"], adjustment_factor=factor,
                adjusted_units=fraction / factor, adjusted_price=adjusted,
                risk_event_id=order["risk_event_id"], reason="synthetic_fraction_redemption"))
        if quantity:
            self._trade_rows.append(dict(datetime=ts, target_ts=order["target_ts"], code=code,
                order_id=order["order_id"], holding_units=self._values[code] / adjusted,
                direction="buy" if quantity > 0 else "sell", source=order["source"],
                risk_event_id=order["risk_event_id"], target_weight=order["target_weight"],
                actual_weight=0.0, price=raw, adjusted_price=adjusted, adjustment_factor=factor,
                shares=abs(quantity), signed_shares=quantity,
                lots=abs(quantity) / share_rule(self._markets[code]).minimum,
                value=abs(quantity) * raw, adjusted_units=quantity / factor, fee=fee,
                holding_shares=self._values[code] / raw, holding_value=self._values[code],
                equity_before=self._execution_equity))
        self._risk.on_fill(code, before_shares, self._values[code] / raw, adjusted, ts)
        order["execution_ts"] = ts

    def _process_orders(self, ts, quotes):
        eligible = {}
        for code, order in list(self._pending.items()):
            quote = quotes.get((code, order["metric"]))
            later = ts > order["target_ts"] if order["source"] == "risk" else ts >= order["target_ts"]
            if quote is not None and later:
                eligible[code] = (order, quote)
                self._mark(code, quote)
        if not eligible:
            return
        equity = self._equity()
        if equity <= 0:
            raise BacktestDataError(f"账户净资产不为正，无法继续分配: {ts}")
        self._execution_equity = equity
        proposals = {}
        for code, (order, quote) in eligible.items():
            raw = quote[1]
            holding = self._values[code] / raw
            delta = (order["target_weight"] * equity - self._values[code]) / raw
            rule = share_rule(self._markets.get(code)) if abs(delta) > 1e-8 else None
            quantity, fraction = round_order(delta, holding, rule) if rule else (0, 0.0)
            proposals[code] = (delta, holding, rule, quantity, fraction)
        for code in sorted(proposals):
            _, _, _, q, fraction = proposals[code]
            if q < 0 or (q == 0 and fraction < 0):
                self._execute(code, *eligible[code], q, fraction, ts)
        buyers = [c for c, p in proposals.items() if p[3] > 0 or (p[3] == 0 and p[4] > 0)]
        cash_floor = min([0.0] + [float(o[0]["cash_weight"]) * equity for o in eligible.values()])
        available_cash = max(0.0, self._cash - cash_floor)

        def scaled(scale):
            result = {}
            for code in buyers:
                delta, holding, rule, _, _ = proposals[code]
                result[code] = round_order(delta * scale, holding, rule)
            return result

        def cost(items):
            return sum((q + f) * eligible[c][1][1] + self._fee(q, eligible[c][1][1]) for c, (q, f) in items.items())

        buy_orders = scaled(1.0)
        if cost(buy_orders) > available_cash + 1e-8:
            lo, hi = 0.0, 1.0
            for _ in range(64):
                mid = (lo + hi) / 2
                if cost(scaled(mid)) <= available_cash + 1e-9:
                    lo = mid
                else:
                    hi = mid
            buy_orders = scaled(lo)
        for code in sorted(buy_orders):
            q, fraction = buy_orders[code]
            if q or fraction:
                self._execute(code, *eligible[code], q, fraction, ts)
        for code, (order, _) in eligible.items():
            if order["source"] == "risk" and abs(self._values[code]) > 1e-7:
                continue
            self._pending.pop(code, None)
            order["status"] = "applied" if order["execution_ts"] is not None else "no_trade"
            if order["execution_ts"] is None:
                order["execution_ts"] = ts
            if order["source"] == "strategy":
                self.actual_datetime.at[order["target_ts"], code] = ts
        for row in reversed(self._trade_rows):
            if row["datetime"] != ts:
                break
            row["actual_weight"] = self._values[row["code"]] / self._equity()
        self._rebalance_snapshots[ts] = (equity, dict(self._values), self._cash, dict(self._marks), dict(self._raw_marks))

    def _snapshot(self, ts, fresh_close):
        equity = self._equity()
        if equity <= 0:
            raise BacktestDataError(f"账户净资产不为正: {ts}")
        values = {**self._values, "cash": self._cash}
        self._daily_values[ts] = values
        self._daily_weights[ts] = {c: v / equity for c, v in values.items()}
        self._daily_pnls[ts] = {**self._pnl, "cash": 0.0}
        self._daily_fees[ts] = {**self._fees, "cash": 0.0}
        self._daily_prices[ts] = {**{c: self._marks[c] if self._marks[c] > 0 else np.nan for c in self.codes}, "cash": 1.0}
        self._daily_raw[ts] = {**{c: self._raw_marks[c] if self._raw_marks[c] > 0 else np.nan for c in self.codes}, "cash": 1.0}
        self._daily_positions[ts] = {**{c: self._values[c] / self._raw_marks[c] if self._raw_marks[c] > 0 else 0.0 for c in self.codes}, "cash": self._cash}
        self._daily_units[ts] = {**{c: self._values[c] / self._marks[c] if self._marks[c] > 0 else 0.0 for c in self.codes}, "cash": self._cash}
        self._valuation_rows.extend(dict(datetime=ts, code=c, stale=c not in fresh_close,
                                         missing_risk_observation=self.risk_config.enabled and c not in fresh_close)
                                    for c in self.codes if self._values[c] != 0)
        self._pnl = dict.fromkeys(self.codes, 0.0)
        self._fees = dict.fromkeys(self.codes, 0.0)

    def _run(self):
        self._cash = self.initial_amount
        self._values = dict.fromkeys(self.codes, 0.0)
        self._marks = dict.fromkeys(self.codes, 0.0)
        self._raw_marks = dict.fromkeys(self.codes, 0.0)
        self._factors = dict.fromkeys(self.codes, 1.0)
        self._pnl = dict.fromkeys(self.codes, 0.0)
        self._fees = dict.fromkeys(self.codes, 0.0)
        self._pending, self._order_rows, self._trade_rows = {}, [], []
        self._settlement_rows, self._valuation_rows = [], []
        self._rebalance_snapshots = {}
        self._daily_values, self._daily_weights, self._daily_pnls = {}, {}, {}
        self._daily_prices, self._daily_raw, self._daily_positions = {}, {}, {}
        self._daily_units, self._daily_fees = {}, {}
        self.actual_datetime = pd.DataFrame(pd.NaT, index=self.weight.index, columns=self.codes)
        events = defaultdict(lambda: dict(signals=[], quotes={}, snapshot=False))
        for ts, row in self.effective_target_weight.iterrows():
            if ts <= self.end:
                events[ts]["signals"].append(row)
        for row in self.market_bars.to_dict("records"):
            for metric in self._metrics:
                ts = row["trade_date"] + self._clocks[metric]
                if self.start.normalize() <= ts <= self.end and pd.notna(row[metric]):
                    events[ts]["quotes"][(row["code"], metric)] = (float(row[metric]), float(row["raw_" + metric]), float(row["adjustment_factor"]))
        for day in self.trade_days:
            ts = day + self._clocks["close"]
            if ts <= self.end:
                events[ts]["snapshot"] = True
        if self.end not in events or not events[self.end]["snapshot"]:
            events[self.end]["snapshot"] = True
        for ts in sorted(events):
            event = events[ts]
            quotes = event["quotes"]
            marks = {}
            for (code, metric), quote in quotes.items():
                if code not in marks or metric == "close":
                    marks[code] = quote
            for code, quote in marks.items():
                self._mark(code, quote)
            for row in event["signals"]:
                for code in self.active_codes:
                    order = self._new_order(code, ts, row[code], self.metric, cash_weight=row["cash"])
                    if not self._risk.accepts(code, ts):
                        order["status"] = "risk_blocked"
                        continue
                    if code in self._pending:
                        self._pending[code]["status"] = "superseded"
                    self._pending[code] = order
            self._process_orders(ts, quotes)
            # A selected proxy price (e.g. high at 15:00) is distinct from close valuation.
            fresh_close = {c: q for (c, m), q in quotes.items() if m == "close"}
            for code, quote in fresh_close.items():
                self._mark(code, quote)
            if event["snapshot"]:
                self._snapshot(ts, fresh_close)
                for code, quote in fresh_close.items():
                    risk_event = self._risk.observe(code, quote[0], ts)
                    if risk_event:
                        if code in self._pending:
                            self._pending[code]["status"] = "risk_cancelled"
                        self._pending[code] = self._new_order(code, ts, 0.0,
                            self.risk_config.execution_metric, source="risk", event_id=risk_event["event_id"])
        self._finish()

    def _finish(self):
        columns = self.codes + ["cash"]
        def frame(rows):
            result = pd.DataFrame.from_dict(rows, orient="index").reindex(columns=columns)
            result.index = pd.DatetimeIndex(result.index, name="datetime")
            return result.sort_index().astype(float)
        self.daily_position_value = frame(self._daily_values)
        self.daily_weight = frame(self._daily_weights)
        self.daily_pnl = frame(self._daily_pnls)
        self.daily_fees = frame(self._daily_fees)
        self.daily_price = frame(self._daily_prices)
        self.daily_raw_price = frame(self._daily_raw)
        self.position = frame(self._daily_positions)
        self.equivalent_position = self.position.copy()
        self.position_units = frame(self._daily_units)
        self.daily_amount = self.daily_position_value.sum(axis=1).rename("amount")
        self.daily_pnl_total = self.daily_pnl.sum(axis=1).rename("daily_pnl")
        self.daily_returns = self.daily_amount.pct_change(fill_method=None)
        if len(self.daily_returns):
            self.daily_returns.iloc[0] = self.daily_amount.iloc[0] / self.initial_amount - 1.0
        baseline = self.start.normalize()
        if len(self.daily_amount) and baseline >= self.daily_amount.index[0]:
            baseline = self.daily_amount.index[0] - pd.Timedelta(microseconds=1)
        self.nav = pd.concat([pd.Series([1.0], index=pd.DatetimeIndex([baseline], name="datetime")), self.daily_amount / self.initial_amount]).rename("nav")
        aw, cp, raw_cp, amounts = {}, {}, {}, {}
        for ts, (amount, values, cash, marks, raw_marks) in self._rebalance_snapshots.items():
            total = cash + sum(values.values())
            aw[ts] = {**{c: values[c] / total for c in self.codes}, "cash": cash / total}
            cp[ts] = {**marks, "cash": 1.0}
            raw_cp[ts] = {**raw_marks, "cash": 1.0}
            amounts[ts] = amount
        self.actual_weight, self.cost_price, self.raw_cost_price = frame(aw), frame(cp), frame(raw_cp)
        self.cost_ret = self.cost_price.pct_change(fill_method=None).fillna(0.0)
        self.cost_ret.attrs['index_role'] = 'execution_events'
        self.cost_price.attrs['index_role'] = 'execution_events'
        self.amount = pd.Series(amounts, dtype=float, name="amount")
        self.amount.index = pd.DatetimeIndex(self.amount.index, name="datetime")
        trade_columns = ["datetime", "target_ts", "order_id", "code", "direction", "source", "risk_event_id", "target_weight", "actual_weight", "price", "adjusted_price", "adjustment_factor", "shares", "signed_shares", "lots", "value", "adjusted_units", "holding_units", "fee", "holding_shares", "holding_value", "equity_before"]
        self.rebalance_log = pd.DataFrame(self._trade_rows, columns=trade_columns)
        self.order_log = pd.DataFrame(self._order_rows)
        self.synthetic_settlements = pd.DataFrame(self._settlement_rows, columns=["datetime", "order_id", "code", "shares", "price", "adjusted_price", "adjustment_factor", "adjusted_units", "cash_change", "source", "risk_event_id", "reason"])
        self.valuation_audit = pd.DataFrame(self._valuation_rows, columns=["datetime", "code", "stale", "missing_risk_observation"])
        risk_columns = ["event_id", "code", "reason", "trigger_ts", "entry_ts", "entry_price", "observed_price", "trigger_return", "side", "execution_metric", "status", "execution_ts"]
        self.risk_events = pd.DataFrame(self._risk.events, columns=risk_columns)
        self.trades = len(self.rebalance_log)
        self.units = 0
        self.nav.attrs.update(initial_nav=1.0, calculation_version=self.calculation_version)
        from betalens.analyst.trades import build_trade_cycles
        self.trade_cycles = build_trade_cycles(self.rebalance_log, self.synthetic_settlements, self.daily_position_value, self.daily_amount.index)

    def dump_to_excel(self, filepath):
        attrs = ["weight", "effective_target_weight", "actual_weight", "daily_weight", "cost_price", "raw_cost_price", "actual_datetime", "cost_ret", "amount", "position", "position_units", "daily_price", "daily_raw_price", "daily_amount", "nav", "daily_returns", "daily_position_value", "daily_pnl", "daily_pnl_total", "daily_fees", "rebalance_log", "order_log", "risk_events", "synthetic_settlements", "valuation_audit", "price_audit", "trade_status", "trade_status_matrix"]
        metadata = {k: getattr(self, k) for k in ["symbol", "metric", "table_name", "initial_amount", "ftc", "ptc", "start", "end", "trades", "calculation_version"]}
        attrs.extend(["trade_cycles", "research_schedule", "research_forward_returns", "research_group_nav", "research_group_coverage", "factor_values"])
        metadata.update(risk=json.dumps(asdict(self.risk_config), ensure_ascii=False),
                        price_time_map=json.dumps(self.price_time_map, ensure_ascii=False),
                        data_sources=json.dumps(self.data_sources, ensure_ascii=False),
                        position_semantics="dividend-reinvested equivalent shares")
        with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
            pd.DataFrame(metadata.items(), columns=["key", "value"]).to_excel(writer, sheet_name="meta", index=False)
            for name in attrs:
                value = getattr(self, name, None)
                if isinstance(value, pd.Series):
                    value = value.to_frame(name=name)
                if isinstance(value, pd.DataFrame):
                    value.to_excel(writer, sheet_name=name[:31], index=isinstance(value.index, pd.DatetimeIndex))
        return str(filepath)
