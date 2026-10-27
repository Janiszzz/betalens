"""Append-only business reference contracts, independent of legacy business APIs.

The physical key includes a revision. Logical scope keys live in the publication
manifest and must never be inferred from a row's mere presence in storage.
"""
from .canonical import CanonicalTable, _f
from .util import digest

VERSION = 'reference-domains-v1'

COMMON = [
    _f('revision_id', 'text', '修订编号', nullable=False),
    _f('scope_id', 'text', '覆盖范围编号', nullable=False),
    _f('record_id', 'text', '范围内记录编号', nullable=False),
    _f('import_task_id', 'text', '导入任务编号', nullable=False),
    _f('attempt_id', 'text', '执行尝试编号', nullable=False),
    _f('ingestion_seq', 'integer', '目标库单调入库序号', nullable=False),
    _f('ingested_at', 'timestamp', '目标库实际入库时间', nullable=False),
    _f('event_date', 'date', '业务发生日期'),
    _f('available_at', 'timestamp', '实际可得时间'),
    _f('downloaded_at', 'timestamp', '下载时间', nullable=False),
    _f('availability_basis', 'text', '可得时间依据'),
    _f('source_name', 'text', '来源类型', nullable=False),
    _f('source_id', 'text', '订阅源实例', nullable=False),
    _f('source_endpoint', 'text', '源接口', nullable=False),
    _f('source_row_no', 'integer', '源记录行号'),
    _f('source_record_hash', 'text', '源记录哈希'),
    _f('mapping_version', 'text', '映射版本', nullable=False),
    _f('schema_version', 'text', '结构版本', nullable=False),
    _f('data_quality_status', 'text', '数据质量状态', nullable=False),
]

def _table(name, zh, grain, fields):
    return CanonicalTable(name, zh, '追加版本；仅可信发布范围可供业务读取。'+grain,
                          grain, tuple(fields+COMMON), ('revision_id','scope_id','record_id'), 'reference')

TABLES = {}
for spec in [
    _table('stock_industry_membership_versions', '股票历史行业分类版本',
           '股票×分类体系×层级完整历史快照中的一个有效区间', [
        _f('security_code','text','股票代码',nullable=False),
        _f('scheme_code','text','分类体系',nullable=False),
        _f('classification_version','text','分类版本'),
        _f('level','integer','行业层级',nullable=False),
        _f('industry_code','text','行业代码'), _f('industry_name','text','行业名称'),
        _f('valid_from','date','生效日'), _f('valid_to','date','失效日（不含）'),
        _f('boundary_basis','text','区间边界依据'),
        _f('membership_status','text','归属状态'), _f('raw_record','text','源记录'),
    ]),
    _table('trade_calendar_versions','交易日历版本','交易所×自然日期', [
        _f('exchange','text','交易所',nullable=False), _f('trade_date','date','自然日期',nullable=False),
        _f('is_open','integer','是否开市'), _f('previous_trade_date','date','上一交易日'),
        _f('reference_exchange','text','参考交易所'), _f('calendar_basis','text','日历依据'),
    ]),
    _table('stock_suspension_intervals_versions','股票停牌区间版本',
           '股票×业务日期完整事件集中的一条停复牌事件', [
        _f('security_code','text','股票代码',nullable=False), _f('trade_date','date','业务日期',nullable=False),
        _f('suspend_type','text','停复牌类型'), _f('raw_timing','text','原始停牌时段'),
        _f('start_at','timestamp','停牌开始（含）'), _f('end_at','timestamp','停牌结束（不含）'),
        _f('is_full_day','integer','是否整日'), _f('time_precision','text','时间精度'),
        _f('parse_status','text','时段解析状态'), _f('parse_reason','text','解析说明'),
        _f('raw_record','text','源记录'),
    ]),
]:
    TABLES[spec.name] = spec

MODULE_TABLE = {'industry':'stock_industry_membership_versions',
                'calendar':'trade_calendar_versions',
                'suspension':'stock_suspension_intervals_versions'}

LOGICAL_KEYS = {'calendar':['exchange','trade_date'],
                'industry':['security_code','scheme_code','level','industry_code','valid_from','valid_to'],
                'suspension':['security_code','trade_date','source_record_hash','start_at','end_at']}
# Declarative provenance for the fixed, versioned reference processors. Source
# fields are selected by name; raw_record preserves the complete source event.
SOURCE_MAPPINGS = {
    'calendar': [{'endpoint':'trade_cal','fields':{'exchange':'exchange','cal_date':'trade_date',
                 'is_open':'is_open','pretrade_date':'previous_trade_date'},
                 'rule':'全部自然日；BSE 显式参考 SSE，缺日不推断休市'}],
    'industry': [{'endpoint':endpoint,'fields':{'ts_code':'security_code','in_date':'valid_from','out_date':'valid_to',
                  **{f'l{i}_code':'industry_code (level='+str(i)+')' for i in (1,2,3)},
                  **{f'l{i}_name':'industry_name (level='+str(i)+')' for i in (1,2,3)}},
                  'rule':'Y+N 完整历史；体系及层级分别发布；区间边界不明保留未知'}
                  for endpoint in ('index_member_all','ci_index_member')],
    'suspension':[{'endpoint':'suspend_d','fields':{'ts_code':'security_code','trade_date':'trade_date',
                    'suspend_type':'suspend_type','suspend_timing':'raw_timing'},
                    'rule':'上海时区解析为 UTC 左闭右开；无法解析保留原文及未知状态'}],
}
CONTRACT_HASH = digest({name:spec.as_dict() for name,spec in TABLES.items()})
