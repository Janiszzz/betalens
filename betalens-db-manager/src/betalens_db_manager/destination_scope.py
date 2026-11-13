"""Explicitly narrow a stopped task's remaining destinations, retaining evidence."""
import json
from pathlib import Path

from .private import atomic_private
from .util import dumps, digest, now, identifier


def remove_destination(state, job, table, backup_path):
    identifier(table)
    with state.connect() as connection:
        connection.execute('BEGIN IMMEDIATE')
        parent=connection.execute('SELECT id,status,plan FROM jobs WHERE id=?',(job,)).fetchone()
        if parent is None:raise ValueError('任务不存在')
        plan=json.loads(parent['plan'])
        if plan.get('revision')!='manager-v1' or plan.get('pipeline_version')!=2 or plan.get('pipeline_parent'):
            raise ValueError('仅支持普通表新版主任务')
        if connection.execute("SELECT 1 FROM tasks WHERE job=? AND status IN ('running','queued')",(job,)).fetchone():
            raise ValueError('请先保存检查点并等待当前执行器停止')
        tables=plan['request']['tables']
        if table not in tables:
            previous=connection.execute("SELECT value FROM ledger WHERE table_name='destination_removal' AND row_key=?",(job+':'+table,)).fetchone()
            if previous:return json.loads(previous[0])
            raise ValueError('该表不属于当前任务')
        if len(tables)==1:raise ValueError('没有其他目标表，请直接取消该任务')
        targets=plan['request']['targets']
        # A stored marker covers the whole mixed cohort for that target. Do not
        # change completed children or the immutable manifests behind receipts.
        children=connection.execute('''SELECT j.id,j.plan FROM jobs j
            WHERE json_extract(j.plan,'$.pipeline_parent')=? AND EXISTS(
                SELECT 1 FROM json_each(json_extract(j.plan,'$.request.tables')) WHERE value=?)
            AND EXISTS(SELECT 1 FROM json_each(?) t WHERE NOT EXISTS(
                SELECT 1 FROM pipeline_units u WHERE u.job=? AND u.id='stored:'||j.id||':'||t.value AND u.status='completed'))''',
            (job,table,dumps(targets),job)).fetchall()
        units=connection.execute('''SELECT id,result FROM pipeline_units u WHERE job=? AND kind='normalize'
            AND json_type(result,?) IS NOT NULL AND EXISTS(
                SELECT 1 FROM json_each(?) t WHERE NOT EXISTS(
                    SELECT 1 FROM pipeline_units s WHERE s.job=u.job AND s.id='stored:'||json_extract(u.result,'$.child_id')||':'||t.value AND s.status='completed'))''',
            (job,'$.tables.'+table,dumps(targets))).fetchall()
        receipts=connection.execute("SELECT * FROM pipeline_receipts WHERE job=? AND json_extract(manifest,'$.table')=? AND status NOT IN ('completed','cancelled')",(job,table)).fetchall()
        backup={'job':job,'table':table,'created_at':now(),
                'plans':[dict(parent),*[dict(c) for c in children]],
                'units':[dict(u) for u in units],'receipts':[dict(r) for r in receipts]}
        path=Path(backup_path)
        if path.exists():raise ValueError('备份路径已存在，不能覆盖')
        atomic_private(path,dumps(backup))
        stamp=now()
        for record in [parent,*children]:
            value=json.loads(record['plan'])
            value['request']['tables']=[name for name in value['request']['tables'] if name!=table]
            connection.execute('UPDATE jobs SET plan=?,updated=? WHERE id=?',(dumps(value),stamp,record['id']))
        for unit in units:
            value=json.loads(unit['result']);value['tables'].pop(table,None)
            connection.execute('UPDATE pipeline_units SET result=?,updated=? WHERE job=? AND id=?',(dumps(value),stamp,job,unit['id']))
        for receipt in receipts:
            connection.execute("UPDATE pipeline_receipts SET status='cancelled',error=?,updated=? WHERE job=? AND batch=? AND target=?",
                ('用户移除剩余目标表；保留原始写入镜像',stamp,job,receipt['batch'],receipt['target']))
            connection.execute('UPDATE pipeline_sync SET revision=revision+1 WHERE job=? AND batch=? AND target=?',(job,receipt['batch'],receipt['target']))
        result={'task_id':job,'table':table,'removed_at':stamp,'remaining_tables':len(tables)-1,
                'children_changed':len(children),'cached_cohorts_changed':len(units),'pending_receipts_cancelled':len(receipts),
                'backup_path':str(path),'backup_hash':digest(backup),'completed_receipts_unchanged':True,
                'frozen_contract_snapshots_unchanged':True}
        connection.execute("INSERT INTO ledger VALUES('destination_removal',?,?)",(job+':'+table,dumps(result)))
        connection.execute('INSERT INTO event_log(job,created,event,value) VALUES(?,?,?,?)',(job,stamp,'destination_removed',dumps(result)))
    return result
