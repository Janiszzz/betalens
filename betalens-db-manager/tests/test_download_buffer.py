from threading import Event,Thread,Lock,Barrier
from types import SimpleNamespace
from pathlib import Path
import time

import pytest

from betalens_db_manager.downloader import TaskDownloader
from betalens_db_manager.pipeline import Pipeline
from betalens_db_manager.pipeline_download import DownloadBuffer
from betalens_db_manager.pipeline_options import options
from betalens_db_manager.source import TushareSource
from test_retrieval import task


def setup(tmp_path,count=20,workers=4,high=8*1024**3):
    def response(endpoint,params):
        return [] if params.get('offset',0) else [{'ts_code':params['ts_code'],'trade_date':'20261020','adj_factor':'1'}]
    service,source,item=task(tmp_path,'fund_adj',{'ts_code':'000001.SZ','trade_date':'20261020'},response,mode='offset')
    root=item['plan']['requests'][0]
    groups={str(i):[{**root,'parameters':{**root['parameters'],'ts_code':f'{i:06d}.SZ'}}] for i in range(count)}
    request={**options({}),'targets':['postgres'],'download_workers':workers,'backlog_high_bytes':high,'backlog_low_bytes':0}
    pipeline=Pipeline(service);pipeline.failed_targets=set()
    return pipeline,TaskDownloader(service,item),groups,request,source


def test_downloads_finish_while_downstream_consumes_nothing(tmp_path):
    pipeline,downloader,groups,request,source=setup(tmp_path)
    with downloader,DownloadBuffer(pipeline,downloader,groups,request) as buffer:
        assert buffer.done.wait(8),'downloads were blocked by an idle downstream'
        assert buffer.error is None
        assert len(source.calls)==2*len(groups)
        assert buffer.queue.qsize()==len(groups) and buffer.bytes>0
        assert [key for key,errors in buffer if not errors]==list(groups)
        archives=list(pipeline.s.settings.state_dir.rglob('*.json.gz'))
        for key in groups:buffer.written(key,'postgres')
        assert buffer.bytes==0 and all(path.exists() for path in archives)
        progress=pipeline.s.state.ledger_get('pipeline_progress',downloader.job)
        assert progress['download_status']=='completed' and progress['buffered_source_bytes']==0


def test_byte_watermark_waits_for_ack_then_resumes_without_losing_pages(tmp_path):
    pipeline,downloader,groups,request,source=setup(tmp_path,count=3,workers=1,high=1)
    with downloader,DownloadBuffer(pipeline,downloader,groups,request) as buffer:
        iterator=iter(buffer);key,errors=next(iterator)
        assert key=='0' and not errors
        time.sleep(.15)
        assert buffer.paused and len(source.calls)==2
        buffer.written(key,'postgres')
        for key,errors in iterator:
            assert not errors
            buffer.written(key,'postgres')
        assert buffer.done.is_set() and buffer.error is None
        assert len(source.calls)==6 and buffer.bytes==0


def test_local_stop_interrupts_a_producer_waiting_for_byte_budget(tmp_path):
    pipeline,downloader,groups,request,source=setup(tmp_path,count=4,workers=1,high=1)
    buffer=DownloadBuffer(pipeline,downloader,groups,request)
    with downloader:
        buffer.__enter__()
        next(iter(buffer))
        start=time.monotonic();buffer.close()
        assert time.monotonic()-start<2 and not buffer.thread.is_alive()
        assert pipeline.s.state.job_header(downloader.job)['cancel']==0


def test_later_groups_keep_downloading_behind_a_slow_head(tmp_path,monkeypatch):
    pipeline,downloader,groups,request,source=setup(tmp_path,count=20,workers=2)
    slow=Event();fast=Event();seen=[];result=[]
    def retrieve(root):
        code=root['parameters']['ts_code']
        if code=='000000.SZ':assert slow.wait(8)
        else:
            seen.append(code)
            if len(seen)==19:fast.set()
        return []
    monkeypatch.setattr(downloader,'_retrieve',retrieve)
    flow=SimpleNamespace(admit=lambda *_:True,archived=lambda *_:None)
    thread=Thread(target=lambda:result.extend(key for key,_,_ in downloader.run_groups(groups,2,lookahead=2,flow=flow)))
    with downloader:
        thread.start()
        try:assert fast.wait(5),'completed groups incorrectly consumed the network lookahead'
        finally:slow.set();thread.join(8)
    assert not thread.is_alive() and result==list(groups)


def test_shared_archive_occupies_the_byte_budget_until_all_owners_finish(tmp_path):
    pipeline,downloader,groups,request,source=setup(tmp_path,count=1)
    root=groups['0'][0]
    with downloader:
        downloader.run([root],workers=1)
        buffer=DownloadBuffer(pipeline,downloader,groups,request)
        buffer.archived('a',root);first=buffer.bytes
        buffer.archived('b',root)
        assert first>0 and buffer.bytes==first
        buffer.written('a','postgres');assert buffer.bytes==first
        buffer.written('b','postgres');assert buffer.bytes==0


def test_source_rate_wait_observes_producer_local_stop(tmp_path):
    pipeline,downloader,groups,request,source=setup(tmp_path,count=1)
    real=TushareSource(pipeline.s.settings,pipeline.s.state)
    real.active_job=downloader.job;real._download_stop=Event();real._download_stop.set()
    with pytest.raises(InterruptedError):real._wait(60)


def test_target_failure_keeps_original_error_in_backpressure_message(tmp_path):
    pipeline,downloader,groups,request,source=setup(tmp_path,count=1,high=1)
    pipeline.failed_targets={'postgres'}
    pipeline.target_errors={'postgres':'fund_nav_daily: write unavailable'}
    with downloader:
        buffer=DownloadBuffer(pipeline,downloader,groups,request);buffer.bytes=1
        with pytest.raises(RuntimeError,match='PostgreSQL: fund_nav_daily: write unavailable'):
            buffer.admit('0','0')


def test_pipeline_records_target_failure_before_backlog_limit_is_reached(tmp_path,monkeypatch):
    from concurrent.futures import Future
    import json
    from test_manager import manager
    from test_retrieval import ArchivedSource
    class Pool:
        def __init__(self,**_):pass
        def __enter__(self):return self
        def __exit__(self,*_):pass
        def submit(self,*_):
            value=Future();value.set_result({'tables':{},'rejected':0});return value
    def response(endpoint,params):
        if endpoint=='daily':return [{'ts_code':'000001.SZ','trade_date':params['trade_date'],'close':'10'}]
        if endpoint=='trade_cal':return [{'exchange':params['exchange'],'cal_date':'20261020','is_open':'1'}]
        if endpoint=='stock_basic' and params.get('list_status')=='L':return [{'ts_code':'000001.SZ','name':'test','list_date':'20000101'}]
        return []
    service=manager(tmp_path);service.source=ArchivedSource(response)
    job=service.plan({'tables':['market_daily_raw'],'source_id':'tushare','endpoints':['daily'],
        'asset_types':['stock'],'start_date':'2026-10-20','end_date':'2026-10-20','targets':['postgres'],'auto_commit':True})
    pipeline=Pipeline(service)
    def fail(*_):raise RuntimeError('write unavailable')
    monkeypatch.setattr('betalens_db_manager.pipeline.ProcessPoolExecutor',Pool)
    monkeypatch.setattr(pipeline,'_write_many',fail)
    monkeypatch.setattr(pipeline,'reconcile',lambda *_:[])
    result=pipeline.run(job['id'])
    assert result['status']=='partial'
    assert pipeline.target_errors=={'postgres':'write unavailable'}
    with service.state.connect() as c:
        event=json.loads(c.execute("SELECT value FROM event_log WHERE job=? AND event='pipeline_target_failed'",(job['id'],)).fetchone()[0])
    assert event['target']=='postgres' and event['error']=='write unavailable' and event['child_ids']


def test_resume_skips_completed_groups_and_releases_after_only_missing_target_finishes(tmp_path):
    pipeline,downloader,groups,request,source=setup(tmp_path,count=2)
    request['targets']=['postgres','clickhouse']
    for group in groups:
        child='child-'+group
        pipeline.store.put(downloader.job,group,'normalize',{})
        pipeline.store.finish(downloader.job,group,{'child_id':child})
        for target in request['targets']:
            if group=='1' and target=='clickhouse':continue
            key='stored:'+child+':'+target
            pipeline.store.put(downloader.job,key,'stored',{'child_id':child,'target':target})
            pipeline.store.finish(downloader.job,key,{'status':'completed'})
    with downloader,DownloadBuffer(pipeline,downloader,groups,request) as buffer:
        assert buffer.pending_targets('0')==[]
        assert buffer.pending_targets('1')==['clickhouse']
        assert buffer.done.wait(8)
        assert [key for key,errors in buffer if not errors]==['1']
        assert len(source.calls)==2
        assert buffer.bytes>0 and buffer.completed==len(groups)
        buffer.written('1','clickhouse')
        assert buffer.bytes==0 and buffer.pending_targets('1')==[]
        progress=pipeline.s.state.ledger_get('pipeline_progress',downloader.job)
        assert progress['completed_groups']==progress['total_groups']==2


def test_bootstrap_directory_filters_run_in_parallel(tmp_path):
    paired=Barrier(2)
    def response(endpoint,params):
        if params['list_status'] in {'L','D'}:
            paired.wait(timeout=3)
            return [{'ts_code':'000001.SZ' if params['list_status']=='L' else '000002.SZ'}]
        return []
    service,source,item=task(tmp_path,'daily',{'trade_date':'20261020'},response)
    item['plan']['request']['download_workers']=2
    with TaskDownloader(service,item) as downloader:
        assert downloader.complete_universe('stock')==['000001.SZ','000002.SZ']


def test_pipeline_keeps_downloading_when_all_conversion_slots_are_blocked(tmp_path,monkeypatch):
    from concurrent.futures import Future
    from datetime import date,timedelta
    from test_manager import manager
    from test_retrieval import ArchivedSource
    released=Event();lock=Lock();pending=[];outcome=[]
    class Pool:
        def __init__(self,**_):pass
        def __enter__(self):return self
        def __exit__(self,*_):pass
        def submit(self,*_):
            future=Future()
            with lock:
                pending.append(future)
                if released.is_set():future.set_result({'tables':{},'rejected':0})
            return future
    def response(endpoint,params):
        if endpoint=='daily':return [{'ts_code':'000001.SZ','trade_date':params['trade_date'],'open':'10','high':'11','low':'9','close':'10','vol':'2','amount':'1'}]
        if endpoint=='trade_cal':return [{'exchange':params['exchange'],'cal_date':(date(2026,10,1)+timedelta(days=i)).strftime('%Y%m%d'),'is_open':'1'} for i in range(20)]
        if endpoint=='stock_basic' and params.get('list_status')=='L':return [{'ts_code':'000001.SZ','name':'test','list_date':'20000101'}]
        return []
    service=manager(tmp_path);service.source=ArchivedSource(response)
    job=service.plan({'tables':['market_daily_raw'],'source_id':'tushare','endpoints':['daily'],'asset_types':['stock'],
        'start_date':'2026-10-01','end_date':'2026-10-20','targets':['postgres'],'auto_commit':False})
    monkeypatch.setattr('betalens_db_manager.pipeline.ProcessPoolExecutor',Pool)
    def run():
        try:outcome.append(Pipeline(service).run(job['id']))
        except BaseException as exc:outcome.append(exc)
    thread=Thread(target=run);thread.start()
    try:
        deadline=time.monotonic()+10
        while time.monotonic()<deadline:
            progress=service.state.ledger_get('pipeline_progress',job['id']) or {}
            if progress.get('download_status')=='completed':break
            if outcome:raise AssertionError(str(outcome[0]))
            time.sleep(.05)
        else:pytest.fail('conversion backpressure stopped the network producer')
        assert sum(p['dataset']=='daily' for p in service.state.slices(job['id']))==20
        with lock:assert len(pending)<=6 and all(not f.done() for f in pending)
    finally:
        with lock:
            released.set()
            for future in pending:
                if not future.done():future.set_result({'tables':{},'rejected':0})
        thread.join(10)
    assert not thread.is_alive() and outcome[0]['status']=='ready_to_commit'
