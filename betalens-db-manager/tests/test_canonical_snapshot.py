from betalens_db_manager.canonical import CanonicalTable,canonical,catalog_snapshot,registry_context


class Registry:
    def __init__(self,label):self.label=label;self.calls=0
    def catalog(self):
        self.calls+=1
        return {'example':CanonicalTable('example',self.label,'test','one row',
            ({'name':'code','type':'text','nullable':False},),('code',))}


def test_normalization_snapshot_reads_registry_once_and_nested_calls_share_it():
    registry=Registry('first')
    with registry_context(registry),catalog_snapshot(registry):
        for _ in range(100):assert canonical('example').zh_name=='first'
        with registry_context(registry),catalog_snapshot(registry):
            assert canonical('example').zh_name=='first'
    assert registry.calls==1


def test_snapshot_is_local_to_registry_and_nested_context_restores_parent():
    first,second=Registry('first'),Registry('second')
    with registry_context(first),catalog_snapshot(first):
        with registry_context(second),catalog_snapshot(second):
            assert canonical('example').zh_name=='second'
        assert canonical('example').zh_name=='first'
    assert first.calls==second.calls==1


def test_next_operation_observes_registry_change_and_exception_releases_snapshot():
    registry=Registry('first')
    with registry_context(registry):
        try:
            with catalog_snapshot(registry):
                registry.label='second'
                assert canonical('example').zh_name=='first'
                raise RuntimeError('interrupted')
        except RuntimeError:pass
        with catalog_snapshot(registry):assert canonical('example').zh_name=='second'
        registry.label='third'
        assert canonical('example').zh_name=='third'
