"""Chunked Arrow IPC and explicit nested artifact envelopes; never pickle."""
from __future__ import annotations

import base64
import datetime as dt
import io

import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc

from .models import ArchiveError, dumps, json_value, loads


def encode_table(value, chunk_rows=2048):
    series = isinstance(value, pd.Series)
    frame = value.to_frame(name="__value__") if series else value
    metadata = {"kind": "series" if series else "frame", "name": pack_tree(value.name, {}) if series else None,
                "rows": len(frame), "columns": [str(c) for c in frame.columns],
                "attrs": json_value(value.attrs), "index_freq": getattr(frame.index, "freqstr", None)}
    chunks = []
    for start in range(0, max(1, len(frame)), chunk_rows):
        part = frame.iloc[start:start + chunk_rows].copy()
        # Arrow cannot infer heterogeneous object columns. Preserve each cell
        # with a typed JSON envelope rather than converting all values to str.
        encoded_columns = []
        for column in part:
            if part[column].dtype == object:
                try:
                    pa.array(part[column], from_pandas=True)
                except (pa.ArrowInvalid, pa.ArrowTypeError):
                    part[column] = part[column].map(lambda x: dumps(x).decode())
                    encoded_columns.append(str(column))
        part.attrs = {}
        table = pa.Table.from_pandas(part, preserve_index=True)
        meta = dict(table.schema.metadata or {})
        meta[b"betalens_json_columns"] = dumps(encoded_columns)
        table = table.replace_schema_metadata(meta)
        sink = pa.BufferOutputStream()
        with ipc.new_file(sink, table.schema, options=ipc.IpcWriteOptions(compression="lz4")) as writer:
            writer.write_table(table)
        chunks.append(sink.getvalue().to_pybytes())
    metadata["chunks"] = len(chunks)
    metadata["chunk_rows"] = chunk_rows
    metadata["bytes"] = sum(map(len, chunks))
    return metadata, chunks


def decode_chunks(chunks, metadata):
    frames = []
    for value in chunks:
        with ipc.open_file(pa.BufferReader(value)) as reader:
            table = reader.read_all()
            frame = table.to_pandas()
            for column in loads((table.schema.metadata or {}).get(b"betalens_json_columns", b"[]")):
                frame[column] = frame[column].map(lambda x: loads(x.encode()))
            frames.append(frame)
    result = pd.concat(frames) if len(frames) > 1 else frames[0]
    if metadata["kind"] == "series":
        result = result.iloc[:, 0].rename(unpack_tree(metadata["name"], lambda _: None))
    result.attrs = metadata["attrs"]
    freq = metadata.get("index_freq")
    if freq and isinstance(result.index, pd.DatetimeIndex):
        try:
            result.index.freq = freq
        except ValueError:
            pass
    return result


def pack_tree(value, tables, path="extras"):
    if isinstance(value, (pd.DataFrame, pd.Series)):
        tables[path] = value
        return {"type": "table", "name": path}
    if isinstance(value, bytes):
        return {"type": "bytes", "value": base64.b64encode(value).decode()}
    if isinstance(value, pd.Timestamp):
        return {"type": "timestamp", "value": value.isoformat()}
    if isinstance(value, pd.Timedelta):
        return {"type": "timedelta", "value": value.value}
    if type(value).__name__ in {"NAType", "NaTType"}:
        return {"type": "nat" if type(value).__name__ == "NaTType" else "na"}
    if isinstance(value, (dt.datetime, dt.date)):
        return {"type": "datetime" if isinstance(value, dt.datetime) else "date", "value": value.isoformat()}
    if type(value).__module__.split(".")[0] == "numpy" and hasattr(value, "shape") and value.shape:
        tables[path] = pd.Series(value.reshape(-1))
        return {"type": "ndarray", "name": path, "shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, dict):
        return {"type": "dict", "items": {str(k): pack_tree(v, tables, path + "/" + str(k)) for k, v in value.items()}}
    if isinstance(value, (list, tuple)):
        return {"type": "tuple" if isinstance(value, tuple) else "list",
                "items": [pack_tree(v, tables, path + "/" + str(i)) for i, v in enumerate(value)]}
    if hasattr(value, "to_plotly_json"):
        return {"type": "plotly", "value": json_value(value.to_plotly_json())}
    if hasattr(value, "savefig"):
        buffer = io.BytesIO()
        value.savefig(buffer, format="png")
        return {"type": "bytes", "value": base64.b64encode(buffer.getvalue()).decode()}
    return {"type": "scalar", "value": json_value(value)}


def unpack_tree(node, read_table):
    kind = node["type"]
    if kind == "table":
        return read_table(node["name"])
    if kind == "bytes":
        return base64.b64decode(node["value"])
    if kind == "timestamp":
        return pd.Timestamp(node["value"])
    if kind == "timedelta":
        return pd.Timedelta(node["value"], unit="ns")
    if kind == "datetime":
        return dt.datetime.fromisoformat(node["value"])
    if kind == "date":
        return dt.date.fromisoformat(node["value"])
    if kind in {"nat", "na"}:
        return pd.NaT if kind == "nat" else pd.NA
    if kind == "ndarray":
        return read_table(node["name"]).to_numpy(dtype=node["dtype"]).reshape(node["shape"])
    if kind == "dict":
        return {k: unpack_tree(v, read_table) for k, v in node["items"].items()}
    if kind in {"tuple", "list"}:
        values = [unpack_tree(v, read_table) for v in node["items"]]
        return tuple(values) if kind == "tuple" else values
    return node["value"]
