"""Render the reviewed shared WSL budget without changing the running deployment."""
import argparse
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import yaml


PG = {
    'shared_buffers': "'3584MB'", 'effective_cache_size': "'10GB'", 'work_mem': "'32MB'",
    'hash_mem_multiplier': '2.0', 'maintenance_work_mem': "'1GB'", 'autovacuum_work_mem': "'256MB'",
    'max_worker_processes': '8', 'max_parallel_workers': '3', 'max_parallel_workers_per_gather': '2',
    'max_parallel_maintenance_workers': '1', 'autovacuum_max_workers': '2',
    'fsync': 'on', 'full_page_writes': 'on', 'synchronous_commit': 'on', 'jit': 'off',
}
CH_SERVER = {
    'max_server_memory_usage': 15*1024**3, 'mark_cache_size': 1024**3, 'uncompressed_cache_size': 0,
    'concurrent_threads_soft_limit_num': 4, 'max_concurrent_queries': 12,
    'background_pool_size': 6, 'background_merges_mutations_concurrency_ratio': 2,
    'merges_mutations_memory_usage_soft_limit': 2*1024**3,
}
CH_USER = {
    'max_threads': 3, 'max_insert_threads': 2, 'use_concurrency_control': 1,
    'max_memory_usage': 4*1024**3, 'max_memory_usage_for_user': 10*1024**3,
    'max_concurrent_queries_for_user': 6,
    'max_bytes_before_external_group_by': 1024**3, 'max_bytes_before_external_sort': 1024**3,
    'max_bytes_in_join': 1024**3,
}


def update_xml(source, values, base=''):
    tree=ET.fromstring(source)
    for name,value in values.items():
        node=tree
        for part in (base+name).split('/'):
            found=node.find(part)
            node=found if found is not None else ET.SubElement(node,part)
        node.text=str(value)
    ET.indent(tree,space='    ')
    return ET.tostring(tree,encoding='unicode')+'\n'


def prepare(infra, output, manager_root):
    output.mkdir(parents=True,exist_ok=True,mode=0o700)
    compose=(infra/'compose.yaml').read_text()
    parsed=yaml.safe_load(compose)
    # Preserve credentials, interpolation, mounts, image tags and all unrelated
    # settings byte-for-byte; do not serialize the whole private Compose file.
    for name,cpu,memory in [('postgres','3.0','14g'),('clickhouse','4.0','20g')]:
        assert name in parsed['services']
        pattern=r'(^  '+name+r':\n)(.*?)(?=^  [A-Za-z_][^\n]*:\n|^\S|\Z)'
        match=re.search(pattern,compose,re.M|re.S)
        if match is None:raise ValueError('Unsupported Compose indentation')
        block=match[2]
        for key,value in [('cpus',cpu),('mem_limit',memory),('memswap_limit',memory)]:
            replacement='    '+key+': "'+value+'"'
            if re.search(r'^    '+key+r':',block,re.M):block=re.sub(r'^    '+key+r':[^\n]*',replacement,block,flags=re.M)
            else:block+=replacement+'\n'
        compose=compose[:match.start(2)]+block+compose[match.end(2):]
    pg=(infra/'postgres/postgresql.conf').read_text()
    pg=re.sub(r'^#.*(?:only|twelve|only one database).*\n','',pg,flags=re.M)
    pg='# PG + CH + manager share 10.8 CPUs / 42 GiB; PostgreSQL: 3 CPUs / 14 GiB.\n'+pg
    for name,value in PG.items():
        if re.search(r'^'+name+r'\s*=',pg,re.M):pg=re.sub(r'^'+name+r'\s*=.*',name+' = '+value,pg,flags=re.M)
        else:pg+=name+' = '+value+'\n'
    server=update_xml((infra/'clickhouse/config.d/99-performance.xml').read_text(),CH_SERVER)
    server=update_xml(server,{name:3 for name in (
        'number_of_free_entries_in_pool_to_lower_max_size_of_merge',
        'number_of_free_entries_in_pool_to_execute_mutation',
        'number_of_free_entries_in_pool_to_execute_optimize_entire_partition')},'merge_tree/')
    user=update_xml((infra/'clickhouse/users.d/99-performance.xml').read_text(),CH_USER,'profiles/default/')
    files={'compose.yaml':compose,'postgres/postgresql.conf':pg,'clickhouse/config.d/99-performance.xml':server,'clickhouse/users.d/99-performance.xml':user}
    for name,data in files.items():
        dest=output/name;dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_text(data);dest.chmod(0o600)
    unit=f'''[Unit]
Description=Betalens database manager

[Service]
Type=simple
WorkingDirectory={manager_root}
ExecStart={manager_root}/.venv/bin/python -m betalens_db_manager --env {infra}/tushare.private.env serve --port 8010
Restart=on-failure
RestartSec=5
TimeoutStopSec=80
KillMode=mixed
MemoryAccounting=yes
CPUQuota=380%
MemoryHigh=7G
MemoryMax=8G
MemorySwapMax=0
TasksMax=256
UMask=0077

[Install]
WantedBy=default.target
'''
    (output/'betalens-db-manager.service').write_text(unit)
    summary={'profile':'throughput_90','cpu_total':10.8,'memory_gib_total':42,'postgres':PG,'clickhouse_server':CH_SERVER,'clickhouse_user':CH_USER}
    (output/'profile.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({'rendered':list(files)+['betalens-db-manager.service'],'output':str(output),'cpu_total':10.8,'memory_gib_total':42}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--infra',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--manager-root',type=Path,required=True)
    args=parser.parse_args()
    prepare(args.infra.resolve(),args.output.resolve(),args.manager_root.resolve())
