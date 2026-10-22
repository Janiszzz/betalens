"""Versioned source-to-canonical field mappings.

Mappings are data, not positional code.  They are exposed by the API and
written to ``meta_source_mapping`` during schema installation.
"""
from __future__ import annotations

from .catalog import load_catalog


VERSION = "canonical-v2"


FIELD_MAP = {
    "stock_basic": {
        "ts_code": ("security_dim", "security_code"), "name": ("security_dim", "name"),
        "exchange": ("security_dim", "exchange"), "list_date": ("security_dim", "list_date"),
        "delist_date": ("security_dim", "delist_date"), "list_status": ("security_dim", "status"),
    },
    "trade_cal": {
        "exchange": ("trade_calendar", "exchange"), "cal_date": ("trade_calendar", "trade_date"),
        "is_open": ("trade_calendar", "is_open"), "pretrade_date": ("trade_calendar", "previous_trade_date"),
    },
    "daily": {
        "ts_code": ("market_daily_raw", "security_code"), "trade_date": ("market_daily_raw", "trade_date"),
        "open": ("market_daily_raw", "open"), "high": ("market_daily_raw", "high"),
        "low": ("market_daily_raw", "low"), "close": ("market_daily_raw", "close"),
        "pre_close": ("market_daily_raw", "previous_close"), "change": ("market_daily_raw", "change"),
        "pct_chg": ("market_daily_raw", "pct_change"), "vol": ("market_daily_raw", "volume"),
        "amount": ("market_daily_raw", "amount"),
    },
    "fund_daily": {
        "ts_code": ("market_daily_raw", "security_code"), "trade_date": ("market_daily_raw", "trade_date"),
        "open": ("market_daily_raw", "open"), "high": ("market_daily_raw", "high"),
        "low": ("market_daily_raw", "low"), "close": ("market_daily_raw", "close"),
        "pre_close": ("market_daily_raw", "previous_close"), "change": ("market_daily_raw", "change"),
        "pct_chg": ("market_daily_raw", "pct_change"), "vol": ("market_daily_raw", "volume"),
        "amount": ("market_daily_raw", "amount"),
    },
    "index_daily": {
        "ts_code": ("market_daily_raw", "security_code"), "trade_date": ("market_daily_raw", "trade_date"),
        "open": ("market_daily_raw", "open"), "high": ("market_daily_raw", "high"),
        "low": ("market_daily_raw", "low"), "close": ("market_daily_raw", "close"),
        "pre_close": ("market_daily_raw", "previous_close"), "change": ("market_daily_raw", "change"),
        "pct_chg": ("market_daily_raw", "pct_change"), "vol": ("market_daily_raw", "volume"),
        "amount": ("market_daily_raw", "amount"),
    },
    "daily_basic": {
        "ts_code": ("market_daily_raw", "security_code"), "trade_date": ("market_daily_raw", "trade_date"),
        "turnover_rate": ("market_daily_raw", "turnover_rate"), "volume_ratio": ("market_daily_raw", "volume_ratio"),
        "pe": ("market_daily_raw", "pe"), "pb": ("market_daily_raw", "pb"),
        "ps": ("market_daily_raw", "ps"), "pcf": ("market_daily_raw", "pcf"),
        "total_share": ("market_daily_raw", "total_shares"), "float_share": ("market_daily_raw", "float_shares"),
        "free_share": ("market_daily_raw", "free_shares"), "total_mv": ("market_daily_raw", "total_market_value"),
        "circ_mv": ("market_daily_raw", "circulating_market_value"),
    },
    "adj_factor": {"ts_code": ("market_daily_post_adjusted", "security_code"), "trade_date": ("market_daily_post_adjusted", "trade_date"), "adj_factor": ("market_daily_post_adjusted", "adjustment_factor")},
    "fund_adj": {"ts_code": ("market_daily_post_adjusted", "security_code"), "trade_date": ("market_daily_post_adjusted", "trade_date"), "adj_factor": ("market_daily_post_adjusted", "adjustment_factor")},
    "suspend_d": {
        "ts_code": ("trade_status_daily", "security_code"), "trade_date": ("trade_status_daily", "trade_date"),
        "suspend_type": ("trade_status_daily", "status_code"),
    },
    "index_weight": {
        "index_code": ("index_membership", "index_code"), "con_code": ("index_membership", "security_code"),
        "trade_date": ("index_membership", "trade_date"), "weight": ("index_membership", "weight"),
    },
    "index_classify": {
        "index_code": ("industry_dim", "industry_code"), "industry_name": ("industry_dim", "industry_name"),
        "parent_code": ("industry_dim", "parent_code"), "level": ("industry_dim", "level"),
    },
    "index_member_all": {
        "ts_code": ("security_industry_membership", "security_code"), "l3_code": ("security_industry_membership", "industry_code"),
        "in_date": ("security_industry_membership", "valid_from"), "out_date": ("security_industry_membership", "valid_to"),
    },
    "ci_index_member": {
        "ts_code": ("security_industry_membership", "security_code"), "l1_code": ("security_industry_membership", "industry_code"),
        "in_date": ("security_industry_membership", "valid_from"), "out_date": ("security_industry_membership", "valid_to"),
    },
    "namechange": {
        "ts_code": ("corporate_action", "security_code"), "start_date": ("corporate_action", "event_date"),
        "end_date": ("corporate_action", "effective_date"), "name": ("corporate_action", "new_name"),
    },
    "dividend": {
        "ts_code": ("corporate_action", "security_code"), "end_date": ("corporate_action", "event_date"),
    },
}

FIELD_MAP['sw_daily']={**FIELD_MAP['index_daily'],'name':('market_daily_raw','security_name'),'pct_change':('market_daily_raw','pct_change'),
                       'pe':('market_daily_raw','pe'),'pb':('market_daily_raw','pb'),'float_mv':('market_daily_raw','circulating_market_value'),'total_mv':('market_daily_raw','total_market_value')}
FIELD_MAP['fund_nav']={f:('fund_nav_daily',{'ts_code':'security_code','ann_date':'announcement_date'}.get(f,f)) for f in ('ts_code','nav_date','ann_date','unit_nav','accum_nav','accum_div','net_asset','total_netasset','adj_nav')}
FIELD_MAP['fund_share']={'ts_code':('fund_shares_daily','security_code'),'trade_date':('fund_shares_daily','trade_date'),'fd_share':('fund_shares_daily','shares')}
FIELD_MAP['bse_mapping']={f:('security_code_history',{'o_code':'previous_code','n_code':'security_code','name':'security_name','list_date':'list_date'}[f]) for f in ('o_code','n_code','name','list_date')}

for endpoint, asset in (("fund_basic","fund"),("index_basic","index")):
    FIELD_MAP[endpoint] = {source:("security_dim",target) for source,target in
                          (("ts_code","security_code"),("name","name"),("list_date","list_date"),("delist_date","delist_date"),("status","status"))}
for source,target in (("close","close"),("turnover_rate_f","turnover_rate_free"),("pe_ttm","pe_ttm"),("ps_ttm","ps_ttm"),("dv_ratio","dividend_yield"),("dv_ttm","dividend_yield_ttm"),("limit_status","limit_status")):
    FIELD_MAP["daily_basic"][source] = ("market_daily_raw",target)
for source,target in (("ah_vol","ah_volume"),("ah_amount","ah_amount")):
    FIELD_MAP["daily"][source] = ("market_daily_raw",target)
from .canonical import FINANCIAL_ALIASES
for endpoint in ("income", "balancesheet", "cashflow", "fina_indicator"):
    FIELD_MAP[endpoint] = {"ts_code": ("fundamental_periodic", "security_code"), "end_date": ("fundamental_periodic", "period_end"), "ann_date": ("fundamental_periodic", "announcement_date"), "f_ann_date": ("fundamental_periodic", "announcement_date"), "report_type": ("fundamental_periodic", "report_type"), "comp_type": ("fundamental_periodic", "company_type"), "update_flag": ("fundamental_periodic", "update_flag")}
    for field in load_catalog()[endpoint].fields:
        if field["type"] == "numeric":
            FIELD_MAP[endpoint][field["name"]] = ("fundamental_periodic",FINANCIAL_ALIASES.get(field["name"],field["name"]))


for endpoint in ('income','balancesheet','cashflow','fina_indicator'):
    FIELD_MAP[endpoint+'_vip']=dict(FIELD_MAP[endpoint])


def all_source_mappings() -> list[dict]:
    rows = []
    from .canonical import load_canonical_catalog
    canonical = load_canonical_catalog()
    catalog = load_catalog()
    for endpoint, spec in catalog.items():
        for field in spec.fields:
            source_field = field["name"]
            target = FIELD_MAP.get(endpoint, {}).get(source_field)
            if endpoint.removesuffix("_vip") in {"income","balancesheet","cashflow","fina_indicator"}:
                if source_field=="f_ann_date":target=("fundamental_periodic","actual_announcement_date")
                elif source_field=="end_type":target=("fundamental_periodic","end_type")
            table, column = target if target else ("__raw_archive_only__", "__unmapped__")
            multiplier = field.get("multiplier", "1")
            target_field = next((f for f in canonical[table].fields if f["name"] == column), {}) if table in canonical else {}
            source_unit = "万份" if endpoint=='fund_share' and source_field=='fd_share' else "万股" if endpoint=='sw_daily' and source_field=='vol' else '万元' if endpoint=='sw_daily' and source_field in {'amount','float_mv','total_mv'} else "手" if source_field in {"vol","ah_vol"} and endpoint in {"daily","fund_daily","index_daily"} else "千元" if source_field in {"amount","ah_amount"} and endpoint in {"daily","fund_daily","index_daily"} else "万股" if endpoint == "daily_basic" and source_field in {"total_share","float_share","free_share"} else "万元" if endpoint == "daily_basic" and source_field in {"total_mv","circ_mv"} else field.get("unit", "")
            transform = f"Decimal(value)*{multiplier}" if field["type"] == "numeric" and str(multiplier) != "1" else "identity"
            rows.append({"source_name": "tushare", "source_endpoint": endpoint, "source_field": source_field,
                         "canonical_table": table, "canonical_column": column, "source_type": field["type"],
                         "target_type": target_field.get("type", field["type"]), "source_unit": source_unit, "target_unit": target_field.get("unit", ""),
                         "transform_expression": transform, "date_role": "announcement_date" if source_field in {'ann_date','f_ann_date'} else 'valid_to' if source_field=='out_date' else "event_date" if source_field in {"trade_date", "cal_date", "end_date", "in_date", "start_date", "nav_date"} else "",
                         "required": 1 if target else 0, "mapping_version": VERSION,
                         "description": field.get("description", "")})
    for endpoint in ('index_member_all','ci_index_member'):
        for level in (1,2,3):
            for source_field,column,table in ((f'l{level}_code','industry_code','security_industry_membership'),(f'l{level}_code','industry_code','industry_dim'),(f'l{level}_name','industry_name','industry_dim')):
                base=next(r for r in rows if r['source_endpoint']==endpoint and r['source_field']==source_field)
                if base['canonical_table']=='__raw_archive_only__':
                    base.update(canonical_table=table,canonical_column=column,required=1,target_type='text')
                else:rows.append({**base,'canonical_table':table,'canonical_column':column,'required':1,'target_type':'text'})
    for endpoint in ('adj_factor','fund_adj'):
        for source_field,column in (('ts_code','security_code'),('trade_date','event_date'),('adj_factor','adjustment_factor')):
            base=next(r for r in rows if r['source_endpoint']==endpoint and r['source_field']==source_field)
            rows.append({**base,'canonical_table':'corporate_action','canonical_column':column})
    for endpoint in ('daily','fund_daily','index_daily','sw_daily'):
        for column in ('open','high','low','close','previous_close','change'):
            source_field='pre_close' if column=='previous_close' else column
            base=next((r for r in rows if r['source_endpoint']==endpoint and r['source_field']==source_field),None)
            if not base:continue
            rows.append({**base,'canonical_table':'market_daily_post_adjusted','canonical_column':'raw_'+column,'transform_expression':'identity'})
            rows.append({**base,'canonical_table':'market_daily_post_adjusted','canonical_column':column,'transform_expression':'native_index_value' if endpoint in ('index_daily','sw_daily') else 'Decimal(raw_price)*Decimal(adjustment_factor)'})
    unique={tuple(row[key] for key in ('source_name','source_endpoint','source_field','canonical_table','canonical_column')):row for row in rows}
    return list(unique.values())


def source_capabilities() -> list[dict]:
    return [
        *[{'source_name':'tushare','endpoint':endpoint+'_vip','capability':'bulk_financial','status':'probe_required','permission_note':'官方按季度全市场接口；5000积分；需验证分页和完整字段合同'} for endpoint in ('income','balancesheet','cashflow','fina_indicator')],
        {"source_name": "tushare", "endpoint": "stock_basic", "capability": "security_dim", "status": "available", "permission_note": "股票列表"},
        {"source_name": "tushare", "endpoint": "trade_cal", "capability": "trade_calendar", "status": "available", "permission_note": "交易日历"},
        {"source_name": "tushare", "endpoint": "daily", "capability": "stock_raw_daily", "status": "probe_required", "permission_note": "日线行情"},
        {"source_name": "tushare", "endpoint": "daily_basic", "capability": "daily_valuation", "status": "probe_required", "permission_note": "每日指标"},
        {"source_name": "tushare", "endpoint": "adj_factor", "capability": "stock_post_adjusted", "status": "probe_required", "permission_note": "股票复权因子"},
        {"source_name": "tushare", "endpoint": "fund_daily", "capability": "fund_raw_daily", "status": "probe_required", "permission_note": "基金日线行情"},
        {"source_name": "tushare", "endpoint": "fund_adj", "capability": "fund_post_adjusted", "status": "probe_required", "permission_note": "基金复权因子"},
        {"source_name": "tushare", "endpoint": "index_daily", "capability": "index_daily", "status": "probe_required", "permission_note": "指数日线行情；原生指数值"},
        {"source_name": "tushare", "endpoint": "suspend_d", "capability": "stock_trade_status", "status": "probe_required", "permission_note": "停复牌"},
        {"source_name": "tushare", "endpoint": "index_member_all", "capability": "sw_industry", "status": "probe_required", "permission_note": "申万行业成分，需验证返回字段"},
        {"source_name": "tushare", "endpoint": "ci_index_member", "capability": "citic_industry", "status": "permission_required", "permission_note": "中信行业成分需要积分权限"},
        {"source_name":"tushare","endpoint":"daily_basic.pcf","capability":"daily_pcf","status":"not_found","permission_note":"daily_basic 没有市现率字段；pcf 保持 NULL，不推测或错用其他指标"},
        {"source_name": "tushare", "endpoint": "fund_trade_status", "capability": "fund_trade_status", "status": "not_found", "permission_note": "未发现权威逐基金停牌接口，将仅支持派生状态"},
        {"source_name": "tushare", "endpoint": "index_trade_status", "capability": "index_trade_status", "status": "not_found", "permission_note": "未发现权威逐指数停牌接口，将仅支持派生状态"},
        {"source_name": "wind", "endpoint": "*", "capability": "all", "status": "not_configured", "permission_note": "Wind 适配器接口已登记，需配置 Wind SDK"},
        {"source_name": "file", "endpoint": "explicit_mapping", "capability": "all", "status": "available", "permission_note": "CSV/XLSX/Parquet 必须显式映射"},
    ]

