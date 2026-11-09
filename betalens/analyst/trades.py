"""Complete directional position cycles, using adjusted units and real cash flows."""
from __future__ import annotations

import numpy as np
import pandas as pd


COLUMNS = ["cycle_id", "code", "side", "entry_ts", "exit_ts", "status", "entry_value",
           "entry_price", "exit_price", "fees", "pnl_net", "return_net", "holding_days",
           "exit_source", "risk_event_id", "accounting_status", "buy_date", "sell_date",
           "buy_price", "sell_price", "return"]


def _legacy(log):
    records = []
    for code, frame in log.groupby("code", sort=False):
        pending = []
        for row in frame.sort_values("datetime").to_dict("records"):
            if row.get("direction") == "buy":
                pending.append(row)
            elif row.get("direction") == "sell" and pending:
                entry = pending.pop(0)
                price_key = "adjusted_price" if "adjusted_price" in row else "price"
                start, end = float(entry[price_key]), float(row[price_key])
                records.append(dict(cycle_id=len(records) + 1, code=code, side="long",
                    entry_ts=pd.Timestamp(entry["datetime"]), exit_ts=pd.Timestamp(row["datetime"]),
                    status="closed", entry_value=np.nan, entry_price=start, exit_price=end,
                    fees=np.nan, pnl_net=np.nan, return_net=np.nan, holding_days=np.nan,
                    accounting_status="legacy_incomplete", buy_date=pd.Timestamp(entry["datetime"]),
                    sell_date=pd.Timestamp(row["datetime"]), buy_price=start, sell_price=end,
                    **{"return": (end - start) / start if start else np.nan}))
    return pd.DataFrame(records, columns=COLUMNS)


def build_trade_cycles(rebalance_log, synthetic_settlements=None, daily_position_value=None,
                       trade_days=None):
    if rebalance_log is None or rebalance_log.empty:
        return pd.DataFrame(columns=COLUMNS)
    required = {"signed_shares", "adjustment_factor", "fee", "adjusted_price", "price"}
    if not required.issubset(rebalance_log.columns):
        return _legacy(rebalance_log)
    numeric = rebalance_log[list(required)].apply(pd.to_numeric, errors='coerce')
    if not np.isfinite(numeric.to_numpy()).all() or (numeric['adjustment_factor'] <= 0).any():
        return _legacy(rebalance_log)
    operations = []
    for row in rebalance_log.to_dict("records"):
        factor = float(row["adjustment_factor"])
        operations.append(dict(code=str(row["code"]), ts=pd.Timestamp(row["datetime"]),
            order_id=row.get("order_id", 0), priority=1,
            units=float(row["signed_shares"]) / factor,
            cash=-float(row["signed_shares"]) * float(row["price"]), fee=float(row["fee"]),
            price=float(row["adjusted_price"]), source=row.get("source", "strategy"),
            risk_event_id=row.get("risk_event_id")))
    if synthetic_settlements is not None and not synthetic_settlements.empty:
        for row in synthetic_settlements.to_dict("records"):
            factor = row.get("adjustment_factor")
            if factor is None or pd.isna(factor):
                matched = rebalance_log.loc[(rebalance_log.code == row["code"]) &
                    (pd.to_datetime(rebalance_log.datetime) == pd.Timestamp(row["datetime"]))]
                if matched.empty:
                    raise ValueError("再投余份缺少复权因子，不能重建净交易周期")
                factor = matched.iloc[0].adjustment_factor
            operations.append(dict(code=str(row["code"]), ts=pd.Timestamp(row["datetime"]),
                order_id=row.get("order_id", 0), priority=0,
                units=float(row["shares"]) / float(factor), cash=float(row["cash_change"]), fee=0.,
                price=float(row["price"]) * float(factor), source=row.get("source", "strategy"),
                risk_event_id=row.get("risk_event_id")))
    operations.sort(key=lambda r: (r["ts"], r["order_id"], r["priority"]))
    calendar = pd.DatetimeIndex(trade_days if trade_days is not None else (
        daily_position_value.index if daily_position_value is not None else [])).normalize().unique().sort_values()
    records, current, units = [], {}, {}

    def finish(code, ts=None, price=np.nan, source=None, risk_id=None):
        cycle = current.pop(code)
        closed = ts is not None
        marked = 0.0
        if not closed:
            marked = np.nan
            if daily_position_value is not None and code in daily_position_value and len(daily_position_value):
                marked = float(daily_position_value[code].iloc[-1])
        pnl = cycle.pop("cash_flow") + marked
        entry_ts = cycle["entry_ts"]
        duration_end = ts if closed else (calendar[-1] if len(calendar) else entry_ts)
        if len(calendar):
            duration = max(1, int(calendar.searchsorted(pd.Timestamp(duration_end).normalize()) - calendar.searchsorted(entry_ts.normalize())))
        else:
            duration = max(1, int(np.busday_count(entry_ts.date(), pd.Timestamp(duration_end).date())))
        cycle.update(exit_ts=ts if closed else pd.NaT, status="closed" if closed else "open",
            exit_price=price, pnl_net=pnl, return_net=pnl / cycle["entry_value"] if cycle["entry_value"] else np.nan,
            holding_days=duration, exit_source=source, risk_event_id=risk_id,
            accounting_status="complete", buy_date=entry_ts, sell_date=ts if closed else pd.NaT,
            buy_price=cycle["entry_price"], sell_price=price)
        cycle["return"] = cycle["return_net"]
        records.append(cycle)

    next_id = 1
    for op in operations:
        code, delta = op["code"], op["units"]
        if abs(delta) < 1e-12:
            continue
        before = units.get(code, 0.0)
        chunks = []
        if before * delta < 0:
            closing = min(abs(before), abs(delta))
            chunks.append((np.sign(delta) * closing, False))
            if abs(delta) > closing + 1e-9:
                chunks.append((np.sign(delta) * (abs(delta) - closing), True))
        else:
            chunks.append((delta, True))
        for piece, opening in chunks:
            fraction = abs(piece / delta)
            if code not in current:
                current[code] = dict(cycle_id=next_id, code=code, side="long" if piece > 0 else "short",
                    entry_ts=op["ts"], entry_price=op["price"], entry_value=0., fees=0., cash_flow=0.)
                next_id += 1
            cycle = current[code]
            if opening:
                cycle["entry_value"] += abs(op["cash"] * fraction)
            cycle["cash_flow"] += (op["cash"] - op["fee"]) * fraction
            cycle["fees"] += op["fee"] * fraction
            units[code] = units.get(code, 0.0) + piece
            if abs(units[code]) <= 1e-8:
                units[code] = 0.0
                finish(code, op["ts"], op["price"], op["source"], op["risk_event_id"])
    for code in list(current):
        finish(code)
    return pd.DataFrame(records, columns=COLUMNS).sort_values("cycle_id").reset_index(drop=True)


def trade_cycle_stats(cycles):
    closed = cycles.loc[cycles.status.eq('closed') & cycles.accounting_status.eq('complete')]
    gains = closed.return_net.dropna()
    wins, losses = gains[gains > 0], gains[gains <= 0]
    loss = abs(float(losses.mean())) if len(losses) else 0.
    odds = float(wins.mean() / loss) if len(wins) and loss else (float('inf') if len(wins) else 0.)
    return dict(trade_count=len(closed), win_rate=float((gains > 0).mean()) if len(gains) else 0.,
                odds=odds, avg_holding_days=float(closed.holding_days.mean()) if len(closed) else np.nan)
