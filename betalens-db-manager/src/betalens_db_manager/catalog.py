from __future__ import annotations

import json
from dataclasses import dataclass, replace
from functools import lru_cache
from importlib.resources import files

from .util import digest, identifier


def optional_source_fields(endpoint: str) -> frozenset[str]:
    # Announcement dates are publication evidence, not a mandatory NAV identity.
    # Keep the field in the source key so distinct dated revisions stay distinct.
    return frozenset({'ann_date'}) if endpoint=='fund_nav' else frozenset()


@dataclass(frozen=True)
class Dataset:
    name: str
    fields: tuple[dict, ...]
    keys: tuple[str, ...]
    date_field: str | None
    strategy: str
    limit: int
    versioned: bool
    document: str
    input_fields: tuple[str, ...]
    filter_date_field: str | None = None

    @property
    def hash(self) -> str:
        return digest(self.as_dict())

    def as_dict(self) -> dict:
        value=dict(name=self.name, fields=self.fields, keys=self.keys, date_field=self.date_field,
                    strategy=self.strategy, limit=self.limit, versioned=self.versioned,
                    document=self.document, input_fields=self.input_fields)
        if self.filter_date_field:value['filter_date_field']=self.filter_date_field
        return value


@lru_cache(maxsize=1)
def load_catalog() -> dict[str, Dataset]:
    raw = json.loads(files(__package__).joinpath("catalog.json").read_text(encoding="utf-8"))
    result = {}
    for item in raw:
        for name in [item["name"], *[f["name"] for f in item["fields"]]]:
            identifier(name)
        item["fields"] = tuple(item["fields"])
        item["keys"] = tuple(item["keys"])
        item["input_fields"] = tuple(item["input_fields"])
        result[item["name"]] = Dataset(**item)
    # The current Tushare documentation exposes CITIC membership separately
    # from the historical catalog snapshot shipped with the package.
    result.setdefault("ci_index_member", Dataset(
        name="ci_index_member",
        fields=tuple({"name": name, "type": "date" if name in {"in_date", "out_date"} else "text",
                      "description": "中信行业成分字段", "multiplier": "1", "unit": ""} for name in (
                          "l1_code", "l1_name", "l2_code", "l2_name", "l3_code", "l3_name",
                          "ts_code", "name", "in_date", "out_date", "is_new")),
        keys=("ts_code", "l1_code", "in_date"), date_field="in_date", strategy="snapshot", limit=5000,
        versioned=False, document="https://tushare.pro/document/2?doc_id=373",
        input_fields=("l1_code", "l2_code", "l3_code", "ts_code", "is_new"),
    ))
    # Official quarterly bulk endpoints share their ordinary endpoint fields
    # and document the same filters, with a 5,000-point permission requirement.
    # Their documents do not publish offset/limit support or a 1,000-row cap:
    # 1,000 is only a client guard. Every task must prove advancing, disjoint
    # offset pages and a terminal empty page before publishing a snapshot.
    for name in ("income", "balancesheet", "cashflow", "fina_indicator"):
        base = result[name]
        result[name + "_vip"] = replace(base, name=name + "_vip", strategy="periodic", limit=1000,
                                        input_fields=tuple(dict.fromkeys((*base.input_fields, "offset", "limit"))))
    return result


def dataset(name: str) -> Dataset:
    try:
        return load_catalog()[name]
    except KeyError:
        raise ValueError(f"未登记的数据集：{name}") from None

