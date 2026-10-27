from betalens_db_manager.time_scope import row_in_scope
from betalens_db_manager.response_validation import validate_response
from betalens_db_manager.catalog import dataset
from test_manager import manager


def test_namechange_request_dates_filter_announcements_not_effective_bounds():
    contract=dataset('namechange').as_dict();fields=[f['name'] for f in contract['fields']]
    source=dict(ts_code='000001.SZ',name='平安银行',start_date='20120802',end_date=None,ann_date='20120120')
    request=dict(source_id='tushare',endpoint='namechange',parameters={'ts_code':'000001.SZ','start_date':'20120101','end_date':'20121231'})
    data=lambda:{'fields':fields,'items':[[source.get(f) for f in fields]]}
    assert not validate_response({'request':{'codes':['000001.SZ']}},request,contract,data())
    source['ann_date']='20130120'
    assert any(e['code']=='date_scope' for e in validate_response({'request':{'codes':['000001.SZ']}},request,contract,data()))


def test_name_history_fetches_one_security_history_and_declares_local_filter(tmp_path):
    service=manager(tmp_path)
    plan=service.compile(dict(tables=['security_name_history'],endpoints=['namechange'],codes=['000001.SZ'],start_date='2004-01-01',end_date='2026-10-23'))
    assert [r['parameters'] for r in plan['requests']]==[{'ts_code':'000001.SZ'}]
    assert any('公告日期' in w['note'] for w in plan['warnings'])


def test_membership_selection_keeps_active_interval_started_before_requested_window():
    assert row_in_scope('security_industry_membership',{'valid_from':'1991-04-03','valid_to':None},'2004-01-01','2026-10-23','valid_from')
    assert row_in_scope('industry_membership_record',{'in_date':'1991-04-03','out_date':None},'2004-01-01','2026-10-23','event_date')
    assert not row_in_scope('security_industry_membership',{'valid_from':'1991-04-03','valid_to':'2003-12-31'},'2004-01-01','2026-10-23','valid_from')
    assert not row_in_scope('industry_membership_record',{'in_date':'2027-01-01','out_date':None},'2004-01-01','2026-10-23','event_date')
    assert row_in_scope('security_name_history',{'start_date':'1991-04-03','end_date':'2006-10-08'},'2004-01-01','2004-01-01','event_date')
    assert not row_in_scope('corporate_action',{'event_date':'1991-04-03'},'2004-01-01','2026-10-23','event_date')


def test_index_membership_filters_index_not_each_constituent():
    contract=dataset('index_weight').as_dict()
    request=dict(source_id='tushare',endpoint='index_weight',parameters={'index_code':'399300.SZ','start_date':'20261001','end_date':'20261023'},validation_scope={'security':['index_code','con_code'],'dates':[{'field':'trade_date'}],'parameters':{}})
    data={'fields':['index_code','con_code','trade_date','weight'],'items':[['399300.SZ','000001.SZ','20261016','0.5']]}
    assert not validate_response({'request':{'codes':['399300.SZ']}},request,contract,data)
    data['items'][0][0]='000016.SH'
    assert any(e['code']=='security_scope' for e in validate_response({'request':{'codes':['399300.SZ']}},request,contract,data))
