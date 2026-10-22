from __future__ import annotations

import hashlib
import json
import os
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def encode(value: Any) -> Any:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Decimal):
        text = format(value, "f")
        return (text.rstrip("0").rstrip(".") if "." in text else text) if value else "0"
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def dumps(value: Any) -> str:
    def safe(item):
        if isinstance(item,int) and not isinstance(item,bool) and abs(item)>9007199254740991:return str(item)
        if isinstance(item,dict):return {key:safe(value) for key,value in item.items()}
        if isinstance(item,(list,tuple)):return [safe(value) for value in item]
        return item
    return json.dumps(safe(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=encode, allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(dumps(value).encode()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(dumps(value), encoding="utf-8")
    tmp.replace(path)


def parse_date(value: Any) -> date:
    text = str(value)
    if len(text) == 8 and text.isdigit():
        return datetime.strptime(text, "%Y%m%d").date()
    return date.fromisoformat(text[:10])


def identifier(value: str) -> str:
    import re
    if not isinstance(value,str) or len(value)>63 or not re.fullmatch(r"[a-z_][a-z0-9_]*", value):
        raise ValueError("标识符只能包含小写字母、数字与下划线")
    return value

