"""Transactional LMDB archives with independent, immutable experiment records."""
from __future__ import annotations

import dataclasses
import hashlib
import os
from pathlib import Path
import threading
import uuid
from typing import Any, Iterable

import pandas as pd

from .codec import decode_chunks, encode_table, pack_tree, unpack_tree
from .models import (ArchiveError, ConflictError, ExperimentInput, MissingArtifactError,
                     digest, dumps, json_value, loads, lookup, matches, now, _MISSING)
from .snapshot import Snapshot, capture_analyst, capture_backtest, capture_run


_ENVIRONMENTS = {}
_ENV_LOCK = threading.RLock()
_DATABASES = ("factors", "aliases", "records", "artifacts", "indexes", "sources", "batches")


def _after_fork():
    global _ENV_LOCK
    for shared in _ENVIRONMENTS.values():
        shared["env"].close()
    _ENVIRONMENTS.clear()
    _ENV_LOCK = threading.RLock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork)


def _validate_locks(locks):
    locks = json_value(locks)
    for path in locks:
        if not path or any(not part for part in path.split(".")):
            raise ArchiveError(f"Invalid locked field: {path!r}")
    pairs = list(locks.items())
    for i, (path, value) in enumerate(pairs):
        for other, target in pairs[i + 1:]:
            if other.startswith(path + ".") and lookup(value, other[len(path) + 1:]) != target:
                raise ConflictError(f"Conflicting locked fields: {path}, {other}")
            if path.startswith(other + ".") and lookup(target, path[len(other) + 1:]) != value:
                raise ConflictError(f"Conflicting locked fields: {other}, {path}")
    return locks


def _metric_match(actual, rule):
    if not isinstance(rule, dict):
        return actual == rule
    operators = {"gt": lambda a, b: a > b, "gte": lambda a, b: a >= b,
                 "lt": lambda a, b: a < b, "lte": lambda a, b: a <= b,
                 "eq": lambda a, b: a == b}
    for op, target in rule.items():
        if op not in operators:
            raise ArchiveError(f"Unknown metric operator: {op}")
        if actual is None:
            return False
        try:
            if not operators[op](actual, target):
                return False
        except TypeError:
            return False
    return True


class ExperimentStore:
    """A local archive. ``query`` returns ``items/total/page/page_size``.

    ``source_id`` identifies an immutable upstream run within its factor/batch.
    Retrying identical data returns the same ID; changed data raise ConflictError.
    Readonly handles never create a database. Close all handles before resizing
    a database; concurrent automatic map resizing is deliberately unsupported.
    """

    def __init__(self, path, *, readonly=False, map_size=128 * 1024**3, chunk_rows=2048):
        if chunk_rows < 1:
            raise ArchiveError("chunk_rows must be positive")
        self.path = Path(path).expanduser().resolve()
        self.readonly, self.map_size, self.chunk_rows = readonly, map_size, chunk_rows
        self._shared = None
        self._pid = None
        self._closed = False
        self._open()

    def _open(self):
        import lmdb
        with _ENV_LOCK:
            key = (os.getpid(), str(self.path))
            shared = _ENVIRONMENTS.get(key)
            if shared and shared["readonly"] and not self.readonly:
                raise ArchiveError("Close readonly handles before opening a writer in this process")
            if shared is None:
                if self.readonly and not (self.path / "data.mdb").is_file():
                    raise FileNotFoundError(self.path)
                if not self.readonly:
                    self.path.mkdir(parents=True, exist_ok=True)
                env = lmdb.open(str(self.path), readonly=self.readonly, create=not self.readonly,
                                map_size=self.map_size, max_dbs=8, max_readers=256,
                                sync=True, metasync=True, writemap=False, lock=True, max_spare_txns=0)
                try:
                    with env.begin() as txn:
                        version = loads(txn.get(b"schema_version"))
                        if version is None and txn.stat()["entries"]:
                            raise ArchiveError("Not a Betalens experiment archive")
                        if version not in (None, 1):
                            raise ArchiveError(f"Unsupported archive schema version: {version}")
                    if version is None:
                        if self.readonly:
                            raise ArchiveError("Archive has not been initialized")
                        with env.begin(write=True) as txn:
                            txn.put(b"schema_version", b"1")
                    dbs = {name: env.open_db(name.encode(), create=not self.readonly) for name in _DATABASES}
                except BaseException:
                    env.close()
                    raise
                shared = {"env": env, "dbs": dbs, "readonly": self.readonly, "refs": 0}
                _ENVIRONMENTS[key] = shared
            shared["refs"] += 1
            self._shared, self._pid = shared, os.getpid()

    def _ensure(self, write=False):
        if self._closed:
            raise ArchiveError("ExperimentStore is closed")
        if write and self.readonly:
            raise PermissionError("The experiment archive is read only")
        if self._pid != os.getpid():
            self._open()
        return self._shared["env"], self._shared["dbs"]

    def close(self):
        with _ENV_LOCK:
            if self._closed:
                return
            if self._pid == os.getpid():
                self._shared["refs"] -= 1
                if self._shared["refs"] == 0:
                    self._shared["env"].close()
                    _ENVIRONMENTS.pop((self._pid, str(self.path)), None)
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def register_factor(self, *, key, name, category="", description="", locks=None,
                        strategy_type="cross_sectional", metadata=None):
        return self._register(key, name, category, description, locks or {}, strategy_type, metadata or {})

    def _register(self, key, name, category, description, locks, strategy_type, metadata, parent_id=None):
        if not str(key).strip() or not str(name).strip():
            raise ArchiveError("Factor key and name must be nonempty")
        locks = _validate_locks(locks)
        env, dbs = self._ensure(write=True)
        alias = digest(str(key)).encode()
        record = dict(key=str(key), name=str(name), category=str(category), description=str(description),
                      locks=locks, strategy_type=strategy_type, metadata=json_value(metadata), parent_id=parent_id)
        with env.begin(write=True) as txn:
            existing_id = txn.get(alias, db=dbs["aliases"])
            if existing_id:
                existing = loads(txn.get(existing_id, db=dbs["factors"]))
                if any(existing[k] != v for k, v in record.items()):
                    raise ConflictError(f"Factor key already registered with different contents: {key}")
                return existing["id"]
            record.update(id=uuid.uuid4().hex, created_at=now())
            txn.put(record["id"].encode(), dumps(record), db=dbs["factors"])
            txn.put(alias, record["id"].encode(), db=dbs["aliases"])
        return record["id"]

    def get_factor(self, factor_id):
        env, dbs = self._ensure()
        with env.begin() as txn:
            record = loads(txn.get(str(factor_id).encode(), db=dbs["factors"]))
        if record is None:
            raise KeyError(f"Unknown factor: {factor_id}")
        return record

    def factors(self, *, search=""):
        env, dbs = self._ensure()
        with env.begin() as txn:
            records = [loads(value) for _, value in txn.cursor(dbs["factors"])]
        term = search.casefold()
        return sorted([r for r in records if term in " ".join((r["key"], r["name"], r["category"])).casefold()],
                      key=lambda r: (r["category"], r["parent_id"] or r["id"], bool(r["parent_id"]), r["name"]))

    def derive_factor(self, parent_id, *, key, name, locks, description="", category=None):
        parent = self.get_factor(parent_id)
        if parent["parent_id"]:
            raise ArchiveError("Derived factors cannot have children; derive from the mother factor")
        new_locks = json_value(locks)
        for path in parent["locks"].keys() & new_locks.keys():
            if parent["locks"][path] != new_locks[path]:
                raise ConflictError(f"Cannot override locked field: {path}")
        return self._register(key, name, parent["category"] if category is None else category,
                              description, {**parent["locks"], **new_locks}, parent["strategy_type"], {}, parent_id)

    def update_factor(self, factor_id, **changes):
        if set(changes) - {"name", "description", "category"}:
            raise ArchiveError("Only name, description and category are editable")
        if "name" in changes and not str(changes["name"]).strip():
            raise ArchiveError("Factor name must be nonempty")
        env, dbs = self._ensure(write=True)
        with env.begin(write=True) as txn:
            record = loads(txn.get(str(factor_id).encode(), db=dbs["factors"]))
            if record is None:
                raise KeyError(factor_id)
            record.update({k: str(v) for k, v in changes.items()})
            txn.put(str(factor_id).encode(), dumps(record), db=dbs["factors"])
        return record

    def save_backtest(self, bt, *, analyst=None, **context):
        return self._save_snapshot(capture_backtest(bt, analyst), **context)

    def save_analyst(self, analyst, **context):
        return self._save_snapshot(capture_analyst(analyst), **context)

    def save_run(self, result, **context):
        return self._save_snapshot(capture_run(result), **context)

    def _save_result(self, result, **context):
        if isinstance(result, Snapshot):
            return self._save_snapshot(result, **context)
        if hasattr(result, "backtest"):
            return self.save_run(result, **context)
        if hasattr(result, "an"):
            return self.save_analyst(result, **context)
        if hasattr(result, "nav"):
            return self.save_backtest(result, **context)
        raise ArchiveError(f"Unsupported completed result: {type(result).__name__}")

    def _save_snapshot(self, snapshot, *, factor_id, config=None, parameters=None, metadata=None,
                       source_id=None, batch_id=None, kind="manual", tags=(), capture_error=None):
        env, dbs = self._ensure(write=True)
        factor = self.get_factor(factor_id)
        root = self.get_factor(factor["parent_id"]) if factor["parent_id"] else factor
        config = json_value(config or {})
        if parameters is None:
            parameters = config.get("parameters")
            if parameters is None:
                inferred = lookup(config, "factor_spec.compute_kwargs")
                parameters = {} if inferred is _MISSING else inferred
        parameters = json_value(parameters)
        fields = {**config, "parameters": parameters}
        if not matches(fields, factor["locks"]):
            raise ArchiveError("Effective configuration does not satisfy the factor's locked fields")
        tables = dict(snapshot.tables)
        extras = dumps(pack_tree(snapshot.extras, tables))
        attributes = {}
        # Weight hooks attach factor/event DataFrames to attrs. Store those as
        # artifacts too, keeping all original data out of experiment summaries.
        pending = list(tables)
        for name in pending:
            if len(pending) > 10000 or name.count("attributes/") > 32:
                raise ArchiveError("Cyclic or excessively nested table attributes")
            if not isinstance(tables[name], (pd.Series, pd.DataFrame)):
                raise ArchiveError(f"Artifact {name} must be a pandas object")
            before = set(tables)
            attributes[name] = dumps(pack_tree(tables[name].attrs, tables, "attributes/" + name))
            pending.extend(sorted(set(tables) - before))
        encoded, descriptors = {}, {}
        for name, value in sorted(tables.items()):
            if not isinstance(value, (pd.Series, pd.DataFrame)):
                raise ArchiveError(f"Artifact {name} must be a pandas object")
            detached = value.copy(deep=False)
            detached.attrs = {}
            descriptors[name], encoded[name] = encode_table(detached, self.chunk_rows)
        doc = dict(schema_version=1, factor_id=root["id"], batch_id=batch_id, kind=kind,
                   config=config, parameters=parameters, metadata={**json_value(snapshot.metadata), **json_value(metadata or {})},
                   metrics=json_value(snapshot.metrics), tags=sorted(set(map(str, tags))),
                   source_id=source_id, tables=descriptors, has_raw=any(k == "nav" or k.endswith("/nav") for k in tables),
                   capture_error=json_value(capture_error), extras_available=bool(snapshot.extras))
        # Source aliases describe provenance, not scientific contents. Inside
        # a batch identical effective parameters refer to one experiment.
        checksum = hashlib.sha256(dumps({k: v for k, v in doc.items() if k != "source_id"}) + extras)
        for name, blocks in encoded.items():
            checksum.update(name.encode())
            checksum.update(attributes[name])
            for block in blocks:
                checksum.update(block)
        content_hash = checksum.hexdigest()
        identity = digest([root["id"], batch_id, str(source_id)]) if source_id is not None else None
        group_identity = digest(["group", root["id"], batch_id, config, parameters]) if batch_id else None
        experiment_id = uuid.uuid4().hex
        doc.update(id=experiment_id, created_at=now(), content_hash=content_hash, factor_snapshot=root)
        try:
            with env.begin(write=True) as txn:
                for lookup_identity in (identity, group_identity):
                    if not lookup_identity:
                        continue
                    previous = loads(txn.get(lookup_identity.encode(), db=dbs["sources"]))
                    if previous:
                        if previous["content_hash"] != content_hash:
                            raise ConflictError(f"Source or batch parameter group {source_id!r} already contains different data; declare distinct effective parameters for independent runs")
                        if identity:
                            txn.put(identity.encode(), dumps(previous), db=dbs["sources"])
                        return previous["id"]
                if batch_id and txn.get(str(batch_id).encode(), db=dbs["batches"]) is None:
                    raise ArchiveError(f"Unknown batch: {batch_id}")
                txn.put(experiment_id.encode(), dumps(doc), db=dbs["records"])
                for name, blocks in encoded.items():
                    for index, block in enumerate(blocks):
                        txn.put(self._artifact_key(experiment_id, name, index), block, db=dbs["artifacts"])
                    txn.put(self._artifact_key(experiment_id, "@attributes/" + name, 0), attributes[name], db=dbs["artifacts"])
                txn.put(self._artifact_key(experiment_id, "@extras", 0), extras, db=dbs["artifacts"])
                for prefix in ("f:" + root["id"], "b:" + batch_id if batch_id else None):
                    if prefix:
                        txn.put(f"{prefix}:{doc['created_at']}:{experiment_id}".encode(), experiment_id.encode(), db=dbs["indexes"])
                for lookup_identity in (identity, group_identity):
                    if lookup_identity:
                        txn.put(lookup_identity.encode(), dumps({"id": experiment_id, "content_hash": content_hash}), db=dbs["sources"])
                if batch_id:
                    batch = loads(txn.get(batch_id.encode(), db=dbs["batches"]))
                    if experiment_id not in batch["experiment_ids"]:
                        batch["experiment_ids"].append(experiment_id)
                    txn.put(batch_id.encode(), dumps(batch), db=dbs["batches"])
        except Exception as exc:
            import lmdb
            if isinstance(exc, lmdb.MapFullError):
                raise ArchiveError("Archive capacity exceeded; close all handles and reopen with a larger map_size") from exc
            raise
        return experiment_id

    @staticmethod
    def _artifact_key(experiment_id, name, index):
        return f"{experiment_id}:{hashlib.sha256(name.encode()).hexdigest()}:{index:08d}".encode()

    def get(self, experiment_id):
        env, dbs = self._ensure()
        with env.begin() as txn:
            doc = loads(txn.get(str(experiment_id).encode(), db=dbs["records"]))
        if doc is None:
            raise KeyError(f"Unknown experiment: {experiment_id}")
        return doc

    def query(self, *, factor_id=None, batch_id=None, kind=None, parameters=None, config=None,
              metrics=None, tags=(), has_raw=None, start=None, end=None, search="", page=1, page_size=50,
              sort_by="created_at", descending=True):
        if page < 1 or not 1 <= page_size <= 1000:
            raise ArchiveError("page must be >= 1 and page_size must be between 1 and 1000")
        factor = self.get_factor(factor_id) if factor_id else None
        root_id = (factor["parent_id"] or factor["id"]) if factor else None
        locks = factor["locks"] if factor else {}
        env, dbs = self._ensure()
        found = []
        with env.begin() as txn:
            prefix = ("f:" + root_id + ":" if root_id else "b:" + batch_id + ":" if batch_id else None)
            if prefix:
                cursor = txn.cursor(dbs["indexes"])
                ids = []
                if cursor.set_range(prefix.encode()):
                    for key, value in cursor:
                        if not key.startswith(prefix.encode()):
                            break
                        ids.append(bytes(value))
                records = (loads(txn.get(identity, db=dbs["records"])) for identity in ids)
            else:
                records = (loads(value) for _, value in txn.cursor(dbs["records"]))
            for doc in records:
                if batch_id and doc["batch_id"] != batch_id or kind and doc["kind"] != kind:
                    continue
                if has_raw is not None and doc["has_raw"] != has_raw:
                    continue
                if start and doc["created_at"] < start or end and doc["created_at"] > end:
                    continue
                if not set(tags).issubset(doc["tags"]) or not matches(doc["parameters"], parameters or {}):
                    continue
                if not matches({**doc["config"], "parameters": doc["parameters"]}, locks):
                    continue
                if not matches(doc["config"], config or {}):
                    continue
                if not all(_metric_match(doc["metrics"].get(k), rule) for k, rule in (metrics or {}).items()):
                    continue
                if search.casefold() not in dumps({k: doc[k] for k in ("id", "parameters", "metadata", "tags")}).decode().casefold():
                    continue
                found.append(doc)
        if sort_by == "created_at":
            found.sort(key=lambda doc: (doc["created_at"], doc["id"]), reverse=descending)
        else:
            valid = [doc for doc in found if isinstance(doc["metrics"].get(sort_by), (float, int))]
            invalid = [doc for doc in found if not isinstance(doc["metrics"].get(sort_by), (float, int))]
            def score(doc):
                value = doc["metrics"][sort_by]
                if sort_by == "objective" and doc["metadata"].get("objective", {}).get("direction") == "minimize":
                    value = -value
                return value
            valid.sort(key=score, reverse=descending)
            found = valid + invalid
        offset = (page - 1) * page_size
        return dict(items=found[offset:offset + page_size], total=len(found), page=page, page_size=page_size)

    def read_table(self, experiment_id, name, *, offset=0, limit=None, columns=None, start=None, end=None):
        if offset < 0 or limit is not None and limit < 0:
            raise ArchiveError("offset/limit must be nonnegative")
        doc = self.get(experiment_id)
        if name not in doc["tables"]:
            raise MissingArtifactError(f"Experiment {experiment_id} has no captured table {name!r}")
        meta = doc["tables"][name]
        # Date filtering requires the original index before applying pagination.
        first = 0 if start or end else min(offset // meta["chunk_rows"], meta["chunks"] - 1)
        stop = meta["chunks"] if start or end or limit is None else min(meta["chunks"], max(first + 1, (offset + limit + meta["chunk_rows"] - 1) // meta["chunk_rows"]))
        env, dbs = self._ensure()
        with env.begin() as txn:
            blocks = [bytes(txn.get(self._artifact_key(experiment_id, name, i), db=dbs["artifacts"])) for i in range(first, stop)]
            attributes = loads(txn.get(self._artifact_key(experiment_id, "@attributes/" + name, 0), db=dbs["artifacts"]))
        value = decode_chunks(blocks, meta)
        if attributes:
            value.attrs = unpack_tree(attributes, lambda key: self.read_table(experiment_id, key))
        if start or end:
            if not isinstance(value.index, pd.DatetimeIndex):
                raise ArchiveError("Date filtering requires a DatetimeIndex")
            if start:
                value = value.loc[value.index >= pd.Timestamp(start)]
            if end:
                value = value.loc[value.index <= pd.Timestamp(end)]
        local_offset = offset if start or end else offset - first * meta["chunk_rows"]
        value = value.iloc[local_offset:None if limit is None else local_offset + limit]
        if columns is not None:
            if isinstance(value, pd.Series):
                raise ArchiveError("Column projection requires a DataFrame")
            value = value.loc[:, columns]
        return value

    def read_extras(self, experiment_id):
        self.get(experiment_id)
        env, dbs = self._ensure()
        with env.begin() as txn:
            tree = loads(txn.get(self._artifact_key(experiment_id, "@extras", 0), db=dbs["artifacts"]))
        return unpack_tree(tree, lambda name: self.read_table(experiment_id, name))

    def to_analyst(self, experiment_id, *, scope="total", **overrides):
        from betalens.analyst import Analyst
        from betalens.analyst.analyst import PortfolioAnalyzer
        doc = self.get(experiment_id)
        prefix = "" if scope == "total" else f"scopes/{scope}/"
        metadata = doc["metadata"].get("scope_metadata", {}).get(scope, doc["metadata"])
        def read(name, fallback=None):
            for candidate in (prefix + "analysis/" + name, prefix + (fallback or name)):
                if candidate in doc["tables"]:
                    return self.read_table(experiment_id, candidate)
            return None
        nav = read("nav")
        if nav is None:
            raise MissingArtifactError(f"No analysis snapshot for scope {scope!r}; select a captured scope")
        arguments = {"nav_series": nav, **metadata.get("analysis", {})}
        for name, fallback in {"weight": "daily_weight", "daily_position_value": None, "daily_pnl": None,
                "rebalance_log": None, "benchmark": None, "cost_price": None, "factor_values": None,
                "daily_returns": None, "daily_amount": None, "synthetic_settlements": None,
                "risk_events": None, "forward_returns": "research_forward_returns", "research_schedule": None,
                "trade_cycles": None}.items():
            arguments[name] = read(name, fallback)
        name = overrides.pop("name", metadata.get("name", "组合"))
        arguments.update(overrides)
        analyst = Analyst(PortfolioAnalyzer(**arguments), name=name)
        analyst.an.metadata.update(metadata.get("analysis_metadata", {}))
        if not {"factor_values", "forward_returns", "cost_price"}.intersection(overrides):
            saved_ic = read("ic_series")
            if saved_ic is not None:
                analyst.an.ic_series = saved_ic
        analyst._name_map = metadata.get("name_map", {})
        return analyst

    def _begin_batch(self, *, source_id=None, kind="batch", metadata=None):
        env, dbs = self._ensure(write=True)
        batch_id = uuid.uuid5(uuid.NAMESPACE_URL, "betalens:batch:" + source_id).hex if source_id else uuid.uuid4().hex
        with env.begin(write=True) as txn:
            existing = loads(txn.get(batch_id.encode(), db=dbs["batches"]))
            if existing and (existing["metadata"] != json_value(metadata or {}) or existing["kind"] != kind):
                raise ConflictError("Batch source already registered with different metadata")
            if not existing:
                txn.put(batch_id.encode(), dumps(dict(id=batch_id, source_id=source_id, kind=kind,
                    metadata=metadata or {}, created_at=now(), status="saving", experiment_ids=[], failures=[])), db=dbs["batches"])
        return batch_id

    def _finish_batch(self, batch_id, failures):
        env, dbs = self._ensure(write=True)
        with env.begin(write=True) as txn:
            batch = loads(txn.get(batch_id.encode(), db=dbs["batches"]))
            batch.update(failures=failures, finished_at=now(), status=("partial" if batch["experiment_ids"] else "failed") if failures else "complete")
            txn.put(batch_id.encode(), dumps(batch), db=dbs["batches"])
        return batch

    def batches(self):
        env, dbs = self._ensure()
        with env.begin() as txn:
            return sorted([loads(v) for _, v in txn.cursor(dbs["batches"])], key=lambda b: b["created_at"], reverse=True)

    def get_batch(self, batch_id):
        env, dbs = self._ensure()
        with env.begin() as txn:
            batch = loads(txn.get(str(batch_id).encode(), db=dbs["batches"]))
        if batch is None:
            raise KeyError(batch_id)
        return batch

    def save_batch(self, results: Iterable, *, source_id=None, kind="batch", metadata=None, factor_id=None, config=None):
        batch_id = self._begin_batch(source_id=source_id, kind=kind, metadata=metadata)
        failures = []
        iterator = iter(results.items() if isinstance(results, dict) else enumerate(results))
        while True:
            try:
                key, entry = next(iterator)
            except StopIteration:
                break
            except Exception as exc:
                failures.append({"source_id": "iterator", "error": f"{type(exc).__name__}: {exc}"})
                break
            try:
                if isinstance(entry, ExperimentInput):
                    context = {f.name: getattr(entry, f.name) for f in dataclasses.fields(entry) if f.name != "result"}
                    result = entry.result
                elif isinstance(entry, dict) and "result" in entry:
                    context = {k: v for k, v in entry.items() if k != "result"}
                    result = entry["result"]
                else:
                    result, context = entry, {"factor_id": factor_id, "config": config or {}}
                context["source_id"] = context.get("source_id") or str(key)
                self._save_result(result, batch_id=batch_id, kind=kind, **context)
            except Exception as exc:
                failures.append({"source_id": str(key), "error": f"{type(exc).__name__}: {exc}"})
        return self._finish_batch(batch_id, failures)

    def save_mining(self, result, *, factor_ids=None):
        """Archive returned mining objects only; no file discovery or backtests."""
        batch_id = self._begin_batch(source_id="mining:" + result.launch_id, kind="mining")
        failures = []
        for run in result.factor_runs:
            try:
                context = getattr(run, "archive_context", {})
                category = context.get("factor_class", "")
                strategy = context.get("strategy_type", "cross_sectional")
                root_id = (factor_ids or {}).get(run.factor_id)
                if root_id is None:
                    key = f"{category}:{run.factor_id}:{strategy}"
                    existing = next((f for f in self.factors() if f["key"] == key), None)
                    root_id = existing["id"] if existing else self.register_factor(key=key, name=context.get("name", run.factor_id),
                        category=category, strategy_type=strategy, metadata=context.get("factor_metadata", {}))
                frames = run.stage_window_results or {"coarse": run.coarse_window_results, "fine": run.fine_window_results,
                                                       "stability": run.stability_window_results}
                summaries = run.stage_summaries or {"coarse": run.coarse_summary, "fine": run.fine_summary,
                                                    "stability": run.stability_summary}
                def combined(values):
                    parts = []
                    for stage, frame in values.items():
                        if frame is not None and not frame.empty:
                            part = frame.copy()
                            part["stage"] = stage
                            parts.append(part)
                    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
                windows, summary = combined(frames), combined(summaries)
                selected = run.selected_candidates
                ids = set()
                for frame in (windows, summary, selected):
                    if "candidate_id" in frame:
                        ids.update(frame.candidate_id.dropna().astype(str))
                for candidate_id in sorted(ids):
                    try:
                        def subset(frame):
                            return frame.loc[frame.candidate_id.astype(str) == candidate_id].copy() if "candidate_id" in frame else frame.iloc[:0].copy()
                        win, summ, chosen = subset(windows), subset(summary), subset(selected)
                        source = next((f.iloc[0] for f in (chosen, summ, win) if not f.empty), None)
                        import json
                        parameters = json.loads(source["params_json"]) if source is not None and pd.notna(source.get("params_json")) else {}
                        captured = getattr(run, "selected_results", {}).get(candidate_id, {})
                        tables = {"mining/window_results": win, "mining/summaries": summ, "mining/selected": chosen}
                        rank_ic = getattr(run, "rank_ic_series", pd.DataFrame())
                        if not rank_ic.empty:
                            tables["mining/rank_ic"] = subset(rank_ic)
                        scope_metadata, scope_metrics = {}, {}
                        scope_extras = dict(captured.get("extras", {}))
                        for scope, snapshot in captured.get("scopes", {}).items():
                            prefix = "" if scope == "total" else f"scopes/{scope}/"
                            tables.update({prefix + k: v for k, v in snapshot.tables.items()})
                            scope_metadata[scope] = snapshot.metadata
                            scope_metrics[scope] = snapshot.metrics
                            if snapshot.extras:
                                # Each scope's auxiliary artifacts remain independently addressable.
                                scope_extras[scope] = snapshot.extras
                        if "window_metrics" in captured:
                            tables["capture/window_metrics"] = captured["window_metrics"]
                        error = getattr(run, "capture_errors", {}).get(candidate_id)
                        row = chosen.iloc[0] if not chosen.empty else summ.iloc[-1] if not summ.empty else None
                        metric = {str(k): v for k, v in row.items() if k not in parameters
                                  and (isinstance(v, (float, int)) or type(v).__module__.startswith("numpy"))} if row is not None else {}
                        config = captured.get("config") or getattr(run, "candidate_configs", {}).get(candidate_id) or {"parameters": parameters}
                        if config.get("archive_context_error"):
                            failures.append({"factor_id": run.factor_id, "candidate_id": candidate_id,
                                             "stage": "configuration", "error": config["archive_context_error"]})
                        snapshot = Snapshot(tables=tables, metrics=metric,
                            metadata={"scope_metadata": scope_metadata, "scope_metrics": scope_metrics, "available_scopes": list(scope_metadata),
                                      "selected": not chosen.empty, "candidate_id": candidate_id, "run_id": run.run_id,
                                      "compute_status": run.status, "window_refs": captured.get("window_refs", {}),
                                      "objective": context.get("selection", {}).get("objective", {}),
                                      "metric_basis": "mining window aggregation", "archive_context": context},
                            extras=scope_extras)
                        self._save_snapshot(snapshot, factor_id=root_id, config=config, parameters=parameters,
                            source_id=run.run_id + ":" + candidate_id, batch_id=batch_id, kind="mining", capture_error=error)
                        if error:
                            failures.append({"factor_id": run.factor_id, "candidate_id": candidate_id, "stage": "capture", "error": error})
                    except Exception as exc:
                        failures.append({"factor_id": run.factor_id, "candidate_id": candidate_id, "error": f"{type(exc).__name__}: {exc}"})
                if not ids and run.status != "complete":
                    failures.append({"factor_id": run.factor_id, "error": "Upstream mining did not complete"})
            except Exception as exc:
                failures.append({"factor_id": run.factor_id, "error": f"{type(exc).__name__}: {exc}"})
        return self._finish_batch(batch_id, failures)
