from copy import deepcopy
from test_retrieval import task
from betalens_db_manager.downloader import TaskDownloader
from betalens_db_manager.pipeline_store import PipelineStore
from betalens_db_manager.task_progress import freeze,summary


def test_pages_do_not_change_frozen_request_or_batch_totals(tmp_path):
    def response(_,p):return [{'ts_code':str(p['offset']),'trade_date':'20261013','adj_factor':'1'}] if p['offset']<3 else []
    s,source,item=task(tmp_path,'fund_adj',{'trade_date':'20261013'},response,cap=2,mode='offset')
    store=PipelineStore(s.state);downloader=TaskDownloader(s,item);groups={'2026-10-13':item['plan']['requests']}
    frozen=freeze(s,item,groups,downloader)
    before=summary(s,s.state.job(item['id']),store.units(item['id']))
    assert before['requests']=={'total':1,'processed':0,'failed':0}
    assert before['batches']['total']==1
    downloader.run()
    after=summary(s,s.state.job(item['id']),store.units(item['id']))
    assert len(s.state.job(item['id'])['slices'])==4
    assert after['requests']=={'total':1,'processed':1,'failed':0}
    assert after['additional_requests']==3 and after['batches']['total']==1
    assert freeze(s,item,groups,downloader)['version']==frozen['version']


def test_failed_root_and_rejected_batch_are_processed_but_not_written(tmp_path):
    s,_,item=task(tmp_path,'fund_adj',{'trade_date':'20261013'},lambda *_:[{'ts_code':'a','trade_date':'20261014'}],mode='offset')
    store=PipelineStore(s.state);d=TaskDownloader(s,item);freeze(s,item,{'one':item['plan']['requests']},d);d.run()
    store.finish(item['id'],'one',{'rejected':1},error='拒收')
    value=summary(s,s.state.job(item['id']),store.units(item['id']))
    assert value['requests']=={'total':1,'processed':1,'failed':1}
    assert value['batches']=={'total':1,'processed':1,'passed':0,'rejected':1}
    assert value['writes']['postgres']=={'total':1,'processed':1,'written':0,'rejected':1}


def test_unknown_scope_is_planning_and_historical_total_is_fixed(tmp_path):
    s,_,item=task(tmp_path,'daily',{'trade_date':'20261013'},lambda *_:[])
    store=PipelineStore(s.state)
    assert summary(s,item,[])['phase']=='planning'
    TaskDownloader(s,item).run();s.state.update_job(item['id'],status='partial')
    value=summary(s,s.state.job(item['id']),[]);assert value['basis']=='historical_archives'
    assert value['requests']['total']==1
    # Reopening the page reads the persisted denominator, rather than max(done, planned).
    assert summary(s,s.state.job(item['id']),[])['version']==value['version']
