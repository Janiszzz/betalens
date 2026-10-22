"""Refresh field contracts from official output tables; development tool only.

Requires requests and beautifulsoup4. Review generated changes before release.
"""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re

import requests
from bs4 import BeautifulSoup


# endpoint: document, business key, date partition, request strategy, response cap
SPECS = {
    "stock_basic": (25, "ts_code", None, "stock_basic", 6000),
    "trade_cal": (26, "exchange cal_date", "cal_date", "calendar", 6000),
    "namechange": (100, "ts_code name start_date", "start_date", "security", 5000),
    "bse_mapping": (375, "o_code n_code", None, "snapshot", 6000),
    "daily": (27, "ts_code trade_date", "trade_date", "daily", 6000),
    "daily_basic": (32, "ts_code trade_date", "trade_date", "daily", 6000),
    "adj_factor": (28, "ts_code trade_date", "trade_date", "daily", 6000),
    "income": (33, "ts_code end_date report_type comp_type ann_date f_ann_date", "end_date", "security", 5000),
    "balancesheet": (36, "ts_code end_date report_type comp_type ann_date f_ann_date", "end_date", "security", 5000),
    "cashflow": (44, "ts_code end_date report_type comp_type ann_date f_ann_date", "end_date", "security", 5000),
    "fina_indicator": (79, "ts_code end_date ann_date", "end_date", "security", 100),
    "dividend": (103, "ts_code end_date ann_date div_proc", "end_date", "security", 2000),
    "fund_basic": (19, "ts_code", None, "snapshot", 15000),
    "fund_daily": (127, "ts_code trade_date", "trade_date", "daily", 2000),
    "fund_adj": (199, "ts_code trade_date", "trade_date", "daily", 2000),
    "fund_nav": (119, "ts_code nav_date ann_date", "nav_date", "security", 800),
    "fund_share": (207, "ts_code trade_date", "trade_date", "security", 2000),
    "index_basic": (94, "ts_code", None, "snapshot", 5000),
    "index_daily": (95, "ts_code trade_date", "trade_date", "security", 8000),
    "index_weight": (96, "index_code con_code trade_date", "trade_date", "index", 5000),
    "index_classify": (181, "index_code", None, "snapshot", 2000),
    "index_member_all": (335, "l3_code ts_code in_date out_date", "in_date", "snapshot", 2000),
    "sw_daily": (327, "ts_code trade_date", "trade_date", "security", 4000),
    "ci_index_member": (373, "ts_code l3_code in_date out_date", "in_date", "snapshot", 5000),
    "suspend_d": (214, "ts_code trade_date suspend_timing suspend_type", "trade_date", "daily", 5000),
}


def build(pair):
    name, (doc, keys, date_field, strategy, limit) = pair
    url = f"https://tushare.pro/document/2?doc_id={doc}"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    tables = soup.find_all("table")
    parameter_tables = [t for t in tables if "名称 类型" in t.find("tr").get_text(" ", strip=True)]
    output = parameter_tables[1]
    fields = []
    for row in output.find_all("tr")[1:]:
        cells = [c.get_text(" ", strip=True) for c in row.find_all(["td", "th"])]
        if len(cells) < 4 or not re.fullmatch("[a-z_][a-z0-9_]*", cells[0]):
            continue
        col, kind, default_display, description = cells[:4]
        typ = "numeric" if kind in {"float", "double", "int", "int64"} else "text"
        if col.endswith("_date") or col in {"cal_date", "nav_date", "purc_startdate", "redm_startdate", "div_listdate"}:
            typ = "date"
        scale, unit = "1", description
        if name in {"daily", "fund_daily", "index_daily"}:
            if col in {"vol", "ah_vol"}: scale, unit = "100", "股（基金为份）"
            if col in {"amount", "ah_amount"}: scale, unit = "1000", "元"
        if name == "sw_daily" and col in {"vol", "amount"}:
            scale, unit = "10000", "股" if col == "vol" else "元"
        if name == "daily_basic" and col in {"total_share", "float_share", "free_share", "total_mv", "circ_mv"}:
            scale, unit = "10000", "元" if col.endswith("mv") else "股"
        source_unit = '手' if name in {'daily','fund_daily','index_daily'} and col in {'vol','ah_vol'} else '千元' if name in {'daily','fund_daily','index_daily'} and col in {'amount','ah_amount'} else '万股' if name=='sw_daily' and col=='vol' else '万元' if name=='sw_daily' and col in {'amount','float_mv','total_mv'} else '万股' if name=='daily_basic' and col in {'total_share','float_share','free_share'} else '万元' if name=='daily_basic' and col in {'total_mv','circ_mv'} else ''
        if name=='sw_daily' and col in {'float_mv','total_mv'}:scale,unit='10000','元'
        if name=='fund_share' and col=='fd_share':scale,unit,source_unit='10000','份','万份'
        if name=='fund_basic' and col=='issue_amount':scale,unit,source_unit='100000000','份','亿份'
        if name=='fund_basic' and col=='min_amount':scale,unit,source_unit='10000','元','万元'
        if name=='dividend' and col=='base_share':scale,unit,source_unit='10000','股','万股'
        fields.append(dict(name=col, type=typ, source_type=kind, default_display=default_display=='Y',description=description, multiplier=scale, unit=unit,source_unit=source_unit))
    inputs = parameter_tables[0]
    input_fields = []
    for row in inputs.find_all("tr")[1:]:
        cells = row.find_all(["td", "th"])
        if cells:
            col = cells[0].get_text(strip=True)
            if re.fullmatch("[a-z_][a-z0-9_]*", col): input_fields.append(col)
    present = {f["name"] for f in fields}
    missing = set(keys.split()) - present
    if missing: raise ValueError((name, missing))
    return dict(name=name, fields=fields, keys=keys.split(), date_field=date_field, strategy=strategy,
                limit=limit, versioned=name in {"income", "balancesheet", "cashflow", "fina_indicator", "dividend", "fund_nav"},
                document=url, input_fields=input_fields)


if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=4) as pool:
        catalog = list(pool.map(build, SPECS.items()))
    destination = Path(__file__).parents[1] / "src/betalens_db_manager/catalog.json"
    destination.write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print({x["name"]: len(x["fields"]) for x in catalog})
