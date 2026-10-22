"""Read-only inspection of pre-existing infrastructure, including non-Tushare tables."""
from psycopg2 import sql
from psycopg2.extras import RealDictCursor

from .storage import Postgres


def tables(pg: Postgres) -> list[dict]:
    with pg.connect(readonly=True) as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""SELECT n.nspname AS schema,c.relname AS name,c.reltuples::bigint AS estimated_rows,
                       pg_total_relation_size(c.oid) AS bytes,c.relkind AS kind
                       FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace
                       WHERE c.relkind IN ('r','p','v') AND NOT c.relispartition
                       AND n.nspname NOT IN ('pg_catalog','information_schema') AND n.nspname NOT LIKE 'pg_toast%'
                       ORDER BY n.nspname,c.relname""")
        return [dict(r) for r in cur.fetchall()]


def sample(pg: Postgres, schema: str, table: str, limit: int = 100) -> dict:
    registered = {(r["schema"], r["name"]) for r in tables(pg)}
    if (schema, table) not in registered: raise ValueError("表未登记或属于系统对象")
    with pg.connect(readonly=True) as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("SELECT column_name,data_type,is_nullable FROM information_schema.columns WHERE table_schema=%s AND table_name=%s ORDER BY ordinal_position", (schema, table))
        cols = [dict(r) for r in cur.fetchall()]
        cur.execute(sql.SQL("SELECT * FROM {} LIMIT %s").format(sql.Identifier(schema, table)), (max(1,min(limit,1000)),))
        return dict(columns=cols, rows=[dict(r) for r in cur.fetchall()], limited=True)


def coverage(pg: Postgres) -> list[dict]:
    from .canonical import load_canonical_catalog
    from .storage import relation
    result = []
    with pg.connect(readonly=True) as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        for name, spec in load_canonical_catalog().items():
            date_col = "event_date" if "event_date" in {f["name"] for f in spec.fields} else "trade_date" if "trade_date" in {f["name"] for f in spec.fields} else None
            if not date_col:
                cur.execute(sql.SQL("SELECT count(*) AS rows FROM {}").format(relation(pg.settings, name)))
                row = dict(cur.fetchone()); row.update(first_date=None, last_date=None)
            else:
                cur.execute(sql.SQL("SELECT min({}) AS first_date,max({}) AS last_date,count(*) AS rows FROM {}").format(sql.Identifier(date_col), sql.Identifier(date_col), relation(pg.settings, name)))
                row = dict(cur.fetchone())
            result.append(dict(dataset=name, **row))
    return result


def daily_gaps(target, table: str, code: str, start: str, end: str) -> dict:
    """Calendar coverage for an explicit security window, without inferring halts."""
    from .util import parse_date
    if table not in {'market_daily_raw','market_daily_post_adjusted','trade_status_daily'}:
        raise ValueError('交易日缺口检查只支持日行情和每日可交易状态表')
    if not code:raise ValueError('检查缺口需要指定证券代码和日期范围')
    first,last=parse_date(start),parse_date(end)
    if first>last:raise ValueError('日期范围无效')

    def read(name, **filters):
        result=[];after=None
        while True:
            page=target.query(name,limit=1000,after=after,**filters)
            result.extend(page['rows']);after=page['next']
            if not after:return result

    securities=read('security_dim',code=code)
    if not securities:raise ValueError('证券基本信息缺失，无法确认交易所及生命周期')
    quotes=read(table,code=code,start=start,end=end)
    calendar=read('trade_calendar',start=start,end=end)
    summary=[];gaps=[]
    for security in securities:
        exchange=security['exchange']
        exchange='SSE' if security['asset_type']=='index' and exchange not in {'SSE','SZSE','BSE'} else exchange
        exchange_calendar=[row for row in calendar if row['exchange']==exchange]
        days={parse_date(row['trade_date']) for row in exchange_calendar if int(row['is_open'])==1}
        if not exchange_calendar:
            summary.append({'security_code':code,'asset_type':security['asset_type'],'status':'calendar_missing','exchange':exchange,
                            'note':'所选范围缺少交易日历，不能判定缺口'})
            continue
        listed=security.get('list_date');delisted=security.get('delist_date')
        expected={day for day in days if (not listed or day>=parse_date(listed)) and (not delisted or day<=parse_date(delisted))}
        present={parse_date(row['trade_date']) for row in quotes if row['asset_type']==security['asset_type']}
        missing=expected-present
        unknown=sum(row.get('is_tradable') is None for row in quotes if row['asset_type']==security['asset_type']) if table=='trade_status_daily' else 0
        unavailable=sum(row.get('close') is None for row in quotes if row['asset_type']==security['asset_type']) if table=='market_daily_post_adjusted' else 0
        note='缺行仅表示覆盖缺口，不能据此认定停牌'
        if not listed:note+='；上市/起始日期未知，边界未经核实'
        summary.append({'security_code':code,'asset_type':security['asset_type'],'table':table,'exchange':exchange,
                        'start_date':first,'end_date':last,'calendar_trading_days':len(days),'expected_rows':len(expected),
                        'present_rows':len(expected&present),'missing_rows':len(missing),'unknown_status_rows':unknown,
                        'unavailable_adjusted_price_rows':unavailable,'note':note})
        gaps.extend({'security_code':code,'asset_type':security['asset_type'],'trade_date':day,'reason':'交易日内无记录，需核查源端或补拉'} for day in sorted(missing))
    return {'summary':summary,'gaps':gaps}

