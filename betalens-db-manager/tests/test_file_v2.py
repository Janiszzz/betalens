"""File tasks share v2 normalization, receipts, manual commit and inverses."""
from decimal import Decimal
from pathlib import Path

import pytest

from betalens_db_manager.canonical import registry_context
from betalens_db_manager.files import import_file,verify_file_cache
from betalens_db_manager.pipeline import Pipeline
from test_manager import live_manager,manager,table_definition


def write_source(path):
    rows=[{'code':'a','day':'2026-10-13','price':'12.34567890'},
          {'code':'b','day':'2026-10-14','price':None}]
    if path.suffix=='.csv':path.write_text('code,day,price\na,2026-10-13,12.34567890\nb,2026-10-14,\n')
    elif path.suffix=='.xlsx':
        from openpyxl import Workbook
        book=Workbook();sheet=book.active;sheet.append(list(rows[0]))
        for row in rows:sheet.append(list(row.values()))
        book.save(path);book.close()
    else:
        import pyarrow as pa
        import pyarrow.parquet as pq
        pq.write_table(pa.Table.from_pylist(rows),path)
    return {'code':'code','day':'business_date','price':'price'}


@pytest.mark.parametrize('suffix',['.csv','.xlsx','.parquet'])
def test_file_v2_manual_commit_audit_and_exact_rollback(live_manager,tmp_path,suffix):
    s=live_manager;s.registry.save(table_definition('file_quotes'))
    with registry_context(s.registry):assert s.schema.apply(s.schema.plan('file_quotes')['token'])['status']=='completed'
    path=tmp_path/('quotes'+suffix);mapping=write_source(path)
    job=import_file(s,'file_quotes',path,mapping)
    assert job['plan']['pipeline_version']==2 and not job['plan']['request']['auto_commit']
    result=Pipeline(s).run(job['id']);assert result['status']=='ready_to_commit',result.get('error')
    with registry_context(s.registry):assert s.pg.query('file_quotes')['rows']==s.ch.query('file_quotes')['rows']==[]
    result=Pipeline(s).commit(job['id']);assert result['status']=='completed',result.get('error')
    with registry_context(s.registry):
        for target in (s.pg,s.ch):
            rows=target.query('file_quotes')['rows']
            assert [(r['code'],r['price']) for r in rows]==[('a',Decimal('12.34567890')),('b',None)]
    assert {r['target'] for r in result['pipeline']['receipts']}=={'postgres','clickhouse'}
    assert Pipeline(s).rollback(job['id'])['status']=='rolled_back'
    with registry_context(s.registry):assert s.pg.query('file_quotes')['rows']==s.ch.query('file_quotes')['rows']==[]


@pytest.mark.parametrize('damage',['original','slice','metadata'])
def test_file_cache_reuse_still_rejects_corrupt_evidence(tmp_path,damage):
    s=manager(tmp_path);s.registry.save(table_definition('file_quotes'))
    path=tmp_path/'quotes.csv';job=import_file(s,'file_quotes',path,write_source(path))
    verify_file_cache(s,job)
    proof=s.state.ledger_get('file_ingestion',job['id']);part=job['slices'][0]
    damaged=Path(proof['source_path']) if damage=='original' else Path(part['archive']) if damage=='slice' else Path(part['archive']).with_suffix('.meta.json')
    damaged.write_bytes(b'{}')
    with pytest.raises((ValueError,OSError)):verify_file_cache(s,job)
