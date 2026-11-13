from copy import deepcopy
from datetime import datetime,timezone

import pytest

from betalens_db_manager.canonical import canonical
from betalens_db_manager.comparison import typed,compare
from betalens_db_manager.commit_protocol import remaining_changes,NeedsReconciliation
from betalens_db_manager.nav_semantics import NavPublicationConflict
from betalens_db_manager.nav_repair import approved_nav_replacement
from betalens_db_manager.pipeline_cache import save_journal,load_journal
from test_nav_comparison import row


def test_approved_replacement_is_atomic_replayable_and_preserves_rollback_images(tmp_path):
    spec=canonical('fund_nav_daily');before=row();incoming=row(unit_nav='1.21',net_asset=None)
    pair={'before':before,'incoming':incoming};original=deepcopy(pair)
    intent=approved_nav_replacement(spec,[pair],ingested_at=datetime.now(timezone.utc))
    assert pair==original
    assert len(remaining_changes(spec,intent,[before]))==1
    change=intent['changes'][0]
    assert change['before']==typed(spec,before)
    assert change['after']['net_asset'] is None
    assert remaining_changes(spec,intent,[change['after']])==[]
    journal=save_journal(tmp_path/'repair',spec,intent,[change['after']])
    saved=load_journal(journal,spec)
    assert saved['changes']==intent['changes']
    with pytest.raises(NeedsReconciliation):
        remaining_changes(spec,saved,[row(unit_nav='1.19')])
    with pytest.raises(NavPublicationConflict):
        compare(spec,[incoming],[before],'refresh_source')


@pytest.mark.parametrize('damage',['key','unknown_date','different_date','identical','duplicate'])
def test_approval_intent_rejects_unreviewed_shapes(damage):
    spec=canonical('fund_nav_daily');pair={'before':row(),'incoming':row(unit_nav='1.21')}
    if damage=='key':pair['incoming']['security_code']='510050.SH'
    elif damage=='unknown_date':pair['incoming']['announcement_date']=None
    elif damage=='different_date':pair['incoming']['announcement_date']='2026-10-21'
    elif damage=='identical':pair['incoming']=row()
    pairs=[pair,pair] if damage=='duplicate' else [pair]
    with pytest.raises(ValueError):approved_nav_replacement(spec,pairs,ingested_at=datetime.now(timezone.utc))
