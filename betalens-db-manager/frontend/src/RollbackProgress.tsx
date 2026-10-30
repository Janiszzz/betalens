import React from 'react';
import {Table,targetNames,type R} from './ui';

/** Count immutable write receipts, never the growing list of inverse receipts. */
export function rollbackProgress(job:R){
 const receipts:R[]=job.pipeline?.receipts||[];
 const operations:R[]=(job.tasks||[]).filter((t:R)=>t.operation==='rollback');
 const operation=operations[operations.length-1];
 const originals=new Map(receipts.filter(r=>!r.batch.startsWith('rollback:')).map(r=>[r.target+':'+r.batch,r]));
 const inverses=new Map(receipts.filter(r=>r.batch.startsWith('rollback:')).map(r=>[r.target+':'+r.batch.slice(9),r]));
 if(!operation&&!inverses.size&&!['rolled_back','partially_rolled_back'].includes(job.status))return;
 const targets=[...new Set<string>([...(job.plan?.request?.targets||[]),...receipts.map(r=>r.target)])];
 const running=operation?.status==='running';
 const results=targets.map(target=>{
  const writes=[...originals.values()].filter(r=>r.target===target);
  const undo=writes.map(r=>inverses.get(target+':'+r.batch)).filter((r):r is R=>!!r);
  const completed=undo.filter(r=>r.status==='completed').length;
  const pending=undo.filter(r=>r.status!=='completed').map(r=>{
   const original=originals.get(target+':'+r.batch.slice(9));
   return {target,table:r.table||original?.table||original?.result?.table||r.batch.split(':').slice(-2,-1)[0],
    batch:r.batch,status:r.status==='failed'?'failed':running?'running':'pending',error:r.error||undefined};
  });
  return {target,total:writes.length,completed,remaining:writes.length-completed,
   failed:undo.filter(r=>r.status==='failed').length,
   applied:undo.filter(r=>r.status==='completed').reduce((n,r)=>n+Number(r.result?.applied||0),0),pending};
 });
 const total=results.reduce((n,r)=>n+r.total,0),completed=results.reduce((n,r)=>n+r.completed,0);
 const status=operation?.status||(['rolled_back','partially_rolled_back'].includes(job.status)?'completed':'interrupted');
 const label=status==='running'?(total===completed?'正在完成审计与收尾':'回退执行中'):
  status==='queued'?'回退排队中':status==='failed'?'回退失败':status==='cancelled'?'回退已停止':
  status==='completed'?(total===completed?'回退完成':'本轮回退完成，任务仍有未回退批次'):'回退已中断';
 return {total,completed,remaining:total-completed,percent:total?Math.round(completed/total*1000)/10:0,
  results,status,label,error:operation?.error};
}

export default function RollbackProgress({job}:{job:R}){
 if(job.plan?.pipeline_version!==2||job.plan?.revision==='reference-v1')return null;
 const p=rollbackProgress(job);if(!p)return null;
 const pending=p.results.flatMap(r=>r.pending);
 return <div className="rollback-progress batch-preview" role="region" aria-label="回退进度">
  <div className="heading"><h3>回退进度</h3><span className={'status-pill status-'+p.status}>{p.label}</span></div>
  <div className="phase-grid">
   <div><small>本任务累计回退批次</small><strong>{p.completed} / {p.total} · {p.percent}%</strong>
    <progress aria-label="回退总进度" value={p.completed} max={p.total||1}/><small>剩余 {p.remaining} 批</small></div>
   {p.results.map(r=><div key={r.target} aria-label={targetNames[r.target]+' 回退详情'}><small>{targetNames[r.target]} 回退批次</small>
    <strong>{r.completed} / {r.total}</strong><progress aria-label={targetNames[r.target]+' 回退进度'} value={r.completed} max={r.total||1}/>
    <small>剩余 {r.remaining} · 失败 {r.failed} · 实际逆操作 {r.applied.toLocaleString('zh-CN')} 行</small>
    {r.pending.length>0&&<small>{p.status==='running'?'当前处理':'未完成'}：{r.pending.map(v=>v.table).join('、')}</small>}
   </div>)}
  </div>
  {p.error&&<p role="alert" className="error">回退操作异常：{p.error}</p>}
  {pending.length>0&&<details open={!!p.error}><summary>当前批次与待核验项 · {pending.length} 批</summary><Table rows={pending}/></details>}
  {p.status==='running'&&!pending.length&&<p className="muted">{p.remaining?'正在准备逆操作或同步审计回执。':'批次逆操作已完成，等待审计同步和任务收尾。'}</p>}
  <p className="muted">每 3 秒自动刷新。总量按原始写入回执固定，包含本任务的人工放行和修补写入；仅已完成的逆操作计入进度，失败和待核验批次不计完成。各数据库累计显示，单库回退后另一库会保留未回退数量。</p>
 </div>;
}
