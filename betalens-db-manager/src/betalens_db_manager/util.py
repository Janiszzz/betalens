from __future__ import annotations

import hashlib
import json
import os
from datetime import date, datetime, timezone
from decimal import Decimal
from functools import lru_cache
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
        # Audit scopes can contain hundreds of thousands of ordinary JSON
        # objects. Copy only branches that actually contain unsafe integers.
        if isinstance(item,dict):
            result=None
            for key,value in item.items():
                converted=safe(value)
                if converted is not value:
                    if result is None:result=item.copy()
                    result[key]=converted
            return item if result is None else result
        if isinstance(item,(list,tuple)):
            result=None
            for index,value in enumerate(item):
                converted=safe(value)
                if converted is not value:
                    if result is None:result=list(item)
                    result[index]=converted
            return item if result is None else result
        return item
    return json.dumps(safe(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=encode, allow_nan=False)


def digest(value: Any) -> str:
    text=dumps(value)
    if len(text)<=65536:return hashlib.sha256(text.encode()).hexdigest()
    result=hashlib.sha256()
    for start in range(0,len(text),65536):result.update(text[start:start+65536].encode())
    return result.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(dumps(value), encoding="utf-8")
    tmp.replace(path)


@lru_cache(maxsize=8192)
def _parse_date_text(text: str) -> date:
    if len(text) == 8 and text.isascii() and text.isdigit():
        # date validates month/day/leap years without strptime's regex work.
        return date(int(text[:4]), int(text[4:6]), int(text[6:]))
    return date.fromisoformat(text[:10])


def parse_date(value: Any) -> date:
    if isinstance(value,date) and not isinstance(value,datetime):return value
    return _parse_date_text(str(value))


def identifier(value: str) -> str:
    import re
    if not isinstance(value,str) or len(value)>63 or not re.fullmatch(r"[a-z_][a-z0-9_]*", value):
        raise ValueError("标识符只能包含小写字母、数字与下划线")
    return value

