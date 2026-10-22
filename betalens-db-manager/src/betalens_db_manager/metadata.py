"""Metadata records use the same logical contracts in both databases."""
from .canonical import canonical_metadata,load_canonical_catalog
from .mapping import VERSION,all_source_mappings,source_capabilities
from .storage import CONTROL_DDL,control_spec
from .util import now


def records():
    result={'meta_table':[],'meta_column':[],'meta_source_mapping':all_source_mappings(),'meta_source_capability':source_capabilities()}
    specs=[*load_canonical_catalog().values(),*[control_spec(name) for name in CONTROL_DDL]]
    for spec in specs:
        result['meta_table'].append({'table_name':spec.name,'table_zh_name':spec.zh_name,'description':spec.description,'grain':spec.grain,'domain':spec.domain,'mapping_version':VERSION,'updated_at':now()})
        for index,field in enumerate(spec.fields,1):
            result['meta_column'].append({'table_name':spec.name,'column_name':field['name'],'column_zh_name':field['zh_name'],'description':field['description'],
                                          'unit':field['unit'],'data_type':field['type'],'ordinal':index,'mapping_version':VERSION,'updated_at':now(),
                                          'precision':field.get('precision'),'scale':field.get('scale'),'nullable':int(field.get('nullable',True)),'system_generated':int(field.get('system',False)),'date_role':'observation_date' if field['name'] in {'event_date','trade_date','period_end','nav_date','valid_from'} else 'available_at' if field['name']=='available_at' else 'ingested_at' if field['name']=='ingested_at' else 'downloaded_at' if field['name']=='downloaded_at' else 'announcement_date' if field['name']=='announcement_date' else None})
    from .canonical import _active_registry
    registry=_active_registry.get()
    if registry:
        from .mapping_store import mapping_rows
        result['meta_source_mapping']=mapping_rows(registry)
    registered=registry.catalog() if registry else {}
    active_mappings={m['table']:m for m in registry.state.ledger_rows('mapping_active')} if registry else {}
    for row in result['meta_table']:
        name=row['table_name']
        row.update(schema_version=next(s.hash for s in specs if s.name==name),business_keys=','.join(next(s.keys for s in specs if s.name==name)))
        if registry and name in registered:
            definition=registry.get(name);row.update(time_field=definition.get('time_field'),temporal_rule=definition.get('temporal_rule'))
            mapping=active_mappings.get(name,{})
            row['sources']=','.join(sorted({b['source_id']+':'+b['endpoint'] for b in mapping.get('bindings',[])}))
    return result
