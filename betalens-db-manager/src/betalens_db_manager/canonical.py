"""Canonical, source-neutral Betalens infrastructure contracts.

The contracts in this module are deliberately independent of the upstream API
names.  Every source adapter produces these rows before a target database is
written.  PostgreSQL and ClickHouse consume the same logical columns.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from importlib.resources import files

from .util import digest, identifier


@dataclass(frozen=True)
class CanonicalTable:
    name: str
    zh_name: str
    description: str
    grain: str
    fields: tuple[dict, ...]
    keys: tuple[str, ...]
    domain: str = "market"

    @property
    def hash(self) -> str:
        return digest(self.as_dict())

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "zh_name": self.zh_name,
            "description": self.description,
            "grain": self.grain,
            "domain": self.domain,
            "fields": self.fields,
            "keys": self.keys,
        }


def _f(name: str, typ: str = "text", zh: str | None = None, unit: str = "", nullable: bool = True, description: str = "") -> dict:
    if zh is None:
        zh, typ = typ, "text"
    return {"name": name, "type": typ, "zh_name": zh, "unit": unit, "nullable": nullable,
            "description": description or zh}


COMMON_DAILY = [
    _f("event_date", "date", "数据发生日期", nullable=False),
    _f("available_at", "timestamp", "数据实际可得时间", description="公告时间、源端时间或推断的可得时间"),
    _f("downloaded_at", "timestamp", "数据下载时间", nullable=False),
    _f("availability_basis", "text", "可得时间依据"),
    _f("import_task_id", "text", "导入任务 ID", nullable=False),
    _f("source_name", "text", "数据来源", nullable=False),
    _f("source_endpoint", "text", "源接口", nullable=False),
    _f("source_row_no", "integer", "源数据行号", nullable=False),
    _f("source_record_hash", "text", "源记录哈希", nullable=False),
    _f("mapping_version", "text", "字段映射版本", nullable=False),
    _f("data_quality_status", "text", "数据质量状态", nullable=False),
]


MARKET_FIELDS = [
    _f("asset_type", "text", "资产类型", nullable=False),
    _f("security_code", "text", "证券代码", nullable=False),
    _f("security_name", "证券名称"),
    _f("trade_date", "date", "交易日期", nullable=False),
    _f("open", "numeric", "开盘价", "元/指数点"),
    _f("high", "numeric", "最高价", "元/指数点"),
    _f("low", "numeric", "最低价", "元/指数点"),
    _f("close", "numeric", "收盘价", "元/指数点"),
    _f("previous_close", "numeric", "前收盘价", "元/指数点"),
    _f("change", "numeric", "涨跌", "元/指数点"),
    _f("pct_change", "numeric", "涨跌幅", "%"),
    _f("volume", "numeric", "成交量", "股/份"),
    _f("amount", "numeric", "成交额", "元"),
    _f("average_price", "numeric", "均价", "元"),
    _f("turnover_rate", "numeric", "换手率", "%"),
    _f("volume_ratio", "numeric", "量比"),
    _f("pe", "numeric", "市盈率"),
    _f("pb", "numeric", "市净率"),
    _f("ps", "numeric", "市销率"),
    _f("pcf", "numeric", "市现率"),
    _f("total_shares", "numeric", "总股本", "股"),
    _f("float_shares", "numeric", "流通股本", "股"),
    _f("free_shares", "numeric", "自由流通股本", "股"),
    _f("total_market_value", "numeric", "总市值", "元"),
    _f("circulating_market_value", "numeric", "流通市值", "元"),
    _f("adjustment_factor", "numeric", "复权因子"),
    _f("adjustment_method", "text", "复权方式"),
    _f("adjustment_formula", "text", "复权公式"),
    _f("adjustment_version", "text", "复权计算版本"),
    *[_f("raw_" + name, "numeric", "原始" + label, "元/指数点") for name, label in (("open","开盘价"),("high","最高价"),("low","最低价"),("close","收盘价"),("previous_close","前收盘价"),("change","涨跌额"))],
    _f("turnover_rate_free", "numeric", "自由流通换手率", "%"),
    _f("pe_ttm", "numeric", "滚动市盈率"), _f("ps_ttm", "numeric", "滚动市销率"),
    _f("dividend_yield", "numeric", "股息率", "%"), _f("dividend_yield_ttm", "numeric", "滚动股息率", "%"),
    _f("limit_status", "text", "涨跌停状态"),
    _f("ah_volume", "numeric", "盘后成交量", "股/份"), _f("ah_amount", "numeric", "盘后成交额", "元"),
    *COMMON_DAILY,
]


_TABLES = [
    CanonicalTable("security_dim", "证券基本信息", "股票、基金和指数统一证券标识及生命周期", "每个证券一行", (
        _f("security_code", "text", "证券代码", nullable=False), _f("asset_type", "text", "资产类型", nullable=False),
        _f("name", "text", "证券名称"), _f("exchange", "text", "交易所"), _f("list_date", "date", "上市日期"),
        _f("delist_date", "date", "退市日期"), _f("status", "text", "当前状态"), _f("source_name", "text", "数据来源", nullable=False),
        _f("downloaded_at", "timestamp", "数据下载时间", nullable=False), _f("import_task_id", "text", "导入任务 ID", nullable=False),
    ), ("security_code", "asset_type")),
    CanonicalTable("trade_calendar", "交易日历", "各交易所交易日和前一交易日", "交易所和日期一行", (
        _f("exchange", "text", "交易所", nullable=False), _f("trade_date", "date", "交易日期", nullable=False),
        _f("is_open", "integer", "是否交易", nullable=False), _f("previous_trade_date", "date", "上一交易日"),
        _f("event_date", "date", "数据发生日期", nullable=False), _f("available_at", "timestamp", "数据实际可得时间"),
        _f("downloaded_at", "timestamp", "数据下载时间", nullable=False), _f("availability_basis", "text", "可得时间依据"),
        _f("import_task_id", "text", "导入任务 ID", nullable=False), _f("source_name", "text", "数据来源", nullable=False),
        _f("source_endpoint", "text", "源接口", nullable=False), _f("source_row_no", "integer", "源数据行号", nullable=False),
        _f("source_record_hash", "text", "源记录哈希", nullable=False), _f("mapping_version", "text", "字段映射版本", nullable=False),
        _f("data_quality_status", "text", "数据质量状态", nullable=False),
    ), ("exchange", "trade_date")),
    CanonicalTable("market_daily_raw", "原始日频行情", "股票、基金和指数的未复权日频宽表", "资产和交易日一行", tuple(MARKET_FIELDS), ("asset_type", "security_code", "trade_date")),
    CanonicalTable("market_daily_post_adjusted", "后复权日频行情", "保存价格后复权值和复权计算依据的日频宽表", "资产和交易日一行", tuple(MARKET_FIELDS), ("asset_type", "security_code", "trade_date")),
    CanonicalTable("trade_status_daily", "每日可交易状态", "证券在每个交易日的可交易状态及来源可信度", "资产和交易日一行", (
        _f("asset_type", "text", "资产类型", nullable=False), _f("security_code", "text", "证券代码", nullable=False),
        _f("trade_date", "date", "交易日期", nullable=False), _f("is_tradable", "integer", "是否可交易"),
        _f("status_code", "text", "状态代码"), _f("status_name", "text", "状态名称"), _f("status_source", "text", "状态来源", nullable=False),
        _f("confidence", "text", "状态可信度", nullable=False), *COMMON_DAILY,
    ), ("asset_type", "security_code", "trade_date")),
    CanonicalTable("industry_scheme_dim", "行业分类体系", "申万、中信等行业分类体系", "每个分类体系一行", (
        _f("scheme_code", "text", "体系代码", nullable=False), _f("scheme_name", "text", "体系名称", nullable=False),
        _f("description", "text", "说明"), _f("source_name", "text", "数据来源", nullable=False),
        _f("downloaded_at", "timestamp", "数据下载时间", nullable=False), _f("import_task_id", "text", "导入任务 ID", nullable=False),
    ), ("scheme_code",)),
    CanonicalTable("industry_dim", "行业分类", "行业代码、名称、层级和父子关系", "体系和行业代码一行", (
        _f("scheme_code", "text", "体系代码", nullable=False), _f("industry_code", "text", "行业代码", nullable=False),
        _f("industry_name", "text", "行业名称", nullable=False), _f("parent_code", "text", "父行业代码"), _f("level", "text", "行业层级"),
        _f("source_name", "text", "数据来源", nullable=False), _f("downloaded_at", "timestamp", "数据下载时间", nullable=False),
        _f("import_task_id", "text", "导入任务 ID", nullable=False),
    ), ("scheme_code", "industry_code")),
    CanonicalTable("security_industry_membership", "证券行业归属", "证券在申万或中信行业中的历史归属", "体系、证券、行业和有效期一行", (
        _f("scheme_code", "text", "体系代码", nullable=False), _f("security_code", "text", "证券代码", nullable=False),
        _f("industry_code", "text", "行业代码", nullable=False), _f("valid_from", "date", "纳入日期", nullable=False),
        _f("valid_to", "date", "剔除日期"), _f("membership_status", "text", "归属状态", nullable=False),
        _f("event_date", "date", "数据发生日期", nullable=False), _f("available_at", "timestamp", "数据实际可得时间"),
        _f("downloaded_at", "timestamp", "数据下载时间", nullable=False), _f("availability_basis", "text", "可得时间依据"),
        _f("import_task_id", "text", "导入任务 ID", nullable=False), _f("source_name", "text", "数据来源", nullable=False),
        _f("source_endpoint", "text", "源接口", nullable=False), _f("source_row_no", "integer", "源数据行号", nullable=False),
        _f("source_record_hash", "text", "源记录哈希", nullable=False), _f("mapping_version", "text", "字段映射版本", nullable=False),
        _f("data_quality_status", "text", "数据质量状态", nullable=False),
    ), ("scheme_code", "security_code", "industry_code", "valid_from")),
    CanonicalTable("index_membership", "指数成分", "指数成分及历史权重", "指数、成分证券和日期一行", (
        _f("index_code", "text", "指数代码", nullable=False), _f("security_code", "text", "成分证券代码", nullable=False),
        _f("trade_date", "date", "交易日期", nullable=False), _f("weight", "numeric", "权重", "%"), *COMMON_DAILY,
    ), ("index_code", "security_code", "trade_date")),
    CanonicalTable("fundamental_periodic", "定期财务数据", "利润表、资产负债表、现金流量表和财务指标的统一宽表", "证券、报告期、报告类型和版本一行", (
        _f("security_code", "text", "证券代码", nullable=False), _f("period_end", "date", "报告期末", nullable=False),
        _f("announcement_date", "date", "公告日期"), _f("report_type", "text", "报告类型", nullable=False),
        _f("statement_type", "text", "报表种类", nullable=False), _f("version_id", "text", "报表版本标识", nullable=False), _f("company_type", "text", "公司类型"),
        _f("update_flag", "text", "修订标记"), _f("revenue", "numeric", "营业收入", "元"), _f("net_income", "numeric", "净利润", "元"),
        _f("total_assets", "numeric", "总资产", "元"), _f("total_liabilities", "numeric", "总负债", "元"),
        _f("operating_cashflow", "numeric", "经营活动现金流", "元"), _f("roe", "numeric", "净资产收益率", "%"),
        _f("event_date", "date", "数据发生日期", nullable=False), _f("available_at", "timestamp", "数据实际可得时间"),
        _f("downloaded_at", "timestamp", "数据下载时间", nullable=False), _f("availability_basis", "text", "可得时间依据"),
        _f("import_task_id", "text", "导入任务 ID", nullable=False), _f("source_name", "text", "数据来源", nullable=False),
        _f("source_endpoint", "text", "源接口", nullable=False), _f("source_row_no", "integer", "源数据行号", nullable=False),
        _f("source_record_hash", "text", "源记录哈希", nullable=False), _f("mapping_version", "text", "字段映射版本", nullable=False),
        _f("data_quality_status", "text", "数据质量状态", nullable=False),
    ), ("security_code", "period_end", "statement_type", "report_type", "version_id"), domain="fundamental"),
    CanonicalTable("corporate_action", "证券事件", "分红、改名和复权因子等证券事件", "证券和事件日期一行", (
        _f("security_code", "text", "证券代码", nullable=False), _f("event_type", "text", "事件类型", nullable=False),
        _f("event_date", "date", "数据发生日期", nullable=False), _f("effective_date", "date", "生效日期"),
        _f("adjustment_factor", "numeric", "复权因子"), _f("old_name", "text", "旧名称"), _f("new_name", "text", "新名称"),
        _f("cash_dividend", "numeric", "现金分红", "元"), _f("share_dividend", "numeric", "送股比例"),
        _f("available_at", "timestamp", "数据实际可得时间"), _f("downloaded_at", "timestamp", "数据下载时间", nullable=False),
        _f("availability_basis", "text", "可得时间依据"), _f("import_task_id", "text", "导入任务 ID", nullable=False),
        _f("source_name", "text", "数据来源", nullable=False), _f("source_endpoint", "text", "源接口", nullable=False),
        _f("source_row_no", "integer", "源数据行号", nullable=False), _f("source_record_hash", "text", "源记录哈希", nullable=False),
        _f("mapping_version", "text", "字段映射版本", nullable=False), _f("data_quality_status", "text", "数据质量状态", nullable=False),
    ), ("security_code", "event_type", "event_date"), domain="event"),
]


_TABLES.extend([
    CanonicalTable('fund_nav_daily','基金日净值','基金单位净值、累计净值和资产净值','基金、净值日期和公告日期一行',(
        _f('security_code','text','基金代码',nullable=False),_f('nav_date','date','净值日期',nullable=False),
        _f('announcement_date','date','公告日期'),
        *[_f(name,'numeric',label,'元') for name,label in [('unit_nav','单位净值'),('accum_nav','累计净值'),('accum_div','累计分红'),('net_asset','资产净值'),('total_netasset','合计资产净值'),('adj_nav','复权单位净值')]],
        *COMMON_DAILY),('security_code','nav_date')),
    CanonicalTable('fund_shares_daily','基金份额','基金每日份额规模','基金和变动日期一行',(
        _f('security_code','text','基金代码',nullable=False),_f('trade_date','date','变动日期',nullable=False),_f('shares','numeric','基金份额','份'),*COMMON_DAILY),('security_code','trade_date')),
    CanonicalTable('security_code_history','证券代码对照','证券新旧代码关系；上市日期不代表代码变更生效日','旧代码和新代码一行',(
        _f('previous_code','text','旧代码',nullable=False),_f('security_code','text','新代码',nullable=False),_f('security_name','text','证券名称'),
        _f('list_date','date','上市日期'),_f('valid_from','date','代码变更生效日期'),
        _f('source_name','text','数据来源',nullable=False),_f('downloaded_at','timestamp','下载时间',nullable=False),_f('import_task_id','text','任务ID',nullable=False)
    ),('previous_code','security_code')),
])


FINANCIAL_ALIASES = {"total_revenue":"total_revenue", "n_income":"net_income", "n_income_attr_p":"net_income_attributable",
                     "total_liab":"total_liabilities", "n_cashflow_act":"operating_cashflow"}


@lru_cache(maxsize=1)
def builtin_catalog() -> dict[str, CanonicalTable]:
    result = {}
    for spec in _TABLES:
        identifier(spec.name)
        for field in spec.fields:
            identifier(field["name"])
        result[spec.name] = spec
    from .catalog import load_catalog
    catalog = load_catalog()
    financial = result["fundamental_periodic"]
    known = {f["name"] for f in financial.fields}
    added = []
    for endpoint in ("income","balancesheet","cashflow","fina_indicator"):
        for field in catalog[endpoint].fields:
            name = FINANCIAL_ALIASES.get(field["name"], field["name"])
            if field["type"] != "numeric" or name in known: continue
            known.add(name)
            added.append(_f(name, "numeric", field["description"], field.get("unit", ""), description=field["description"]))
    result["fundamental_periodic"] = replace(financial, fields=financial.fields + tuple(added))
    # A query's complete fields stay together at their original business grain.
    contracts = [
        ('stock_basic','stock_profile','股票档案',('security_code',)),
        ('fund_basic','fund_profile','基金档案',('security_code',)),
        ('index_basic','index_profile','指数档案',('security_code',)),
        ('dividend','dividend_event','分红方案',('security_code','record_version')),
        ('suspend_d','suspension_event','停复牌事件',('security_code','record_version')),
        ('namechange','security_name_history','证券名称历史',('security_code','record_version')),
        ('fund_nav','fund_nav_observation','基金净值公告记录',('security_code','record_version')),
        ('index_member_all','industry_membership_record','行业归属原始记录',('scheme_code','security_code','record_version')),
        ('ci_index_member','industry_membership_record','行业归属原始记录',('scheme_code','security_code','record_version')),
    ]
    for endpoint,name,label,keys in contracts:
        fields=list(result[name].fields) if name in result else []
        known={f['name'] for f in fields}
        for field in catalog[endpoint].fields:
            column='security_code' if field['name']=='ts_code' else field['name']
            if column not in known:
                fields.append(_f(column,field['type'],field['description'],field.get('unit','')));known.add(column)
        for key in keys:
            if key not in known:fields.append(_f(key,'text',key,nullable=False));known.add(key)
        if 'event_date' not in known:
            fields.extend([dict(f,nullable=True) for f in COMMON_DAILY])
        result[name]=CanonicalTable(name,label,label+'；完整保存查询字段和版本','源记录业务粒度',tuple(fields),keys)
    financial=result['fundamental_periodic']
    result['fundamental_periodic']=replace(financial,fields=financial.fields+(
        _f('actual_announcement_date','date','实际公告日期'),_f('end_type','text','报告期类型'),))
    industry=result['industry_dim']
    missing=[f for f in catalog['index_classify'].fields if f['name'] not in {'index_code','industry_name','parent_code','level'}]
    result['industry_dim']=replace(industry,fields=industry.fields+tuple(_f('classification_code' if f['name']=='industry_code' else f['name'],f['type'],f['description'],f.get('unit','')) for f in missing))
    return result


def canonical(name: str) -> CanonicalTable:
    try:
        return load_canonical_catalog()[name]
    except KeyError:
        registry=_active_registry.get()
        if registry and registry.record(name):return registry.spec(name)
        raise ValueError(f"未登记的本地表：{name}") from None


def canonical_metadata() -> list[dict]:
    rows = []
    for spec in load_canonical_catalog().values():
        for ordinal, field in enumerate(spec.fields, 1):
            rows.append({"table_name": spec.name, "table_zh_name": spec.zh_name, "table_description": spec.description,
                         "grain": spec.grain, "column_name": field["name"], "column_zh_name": field["zh_name"],
                         "description": field["description"], "unit": field["unit"], "data_type": field["type"],
                         "ordinal": ordinal})
    return rows



from contextvars import ContextVar
from contextlib import contextmanager
_active_registry = ContextVar('table_registry', default=None)

@contextmanager
def registry_context(registry):
    token = _active_registry.set(registry)
    try: yield
    finally: _active_registry.reset(token)

def load_canonical_catalog() -> dict[str, CanonicalTable]:
    registry = _active_registry.get()
    return registry.catalog() if registry else builtin_catalog()
