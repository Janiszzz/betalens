import React,{useEffect,useRef,useState} from 'react';
import {api,Status,Table,Targets,Time,type R} from './ui';

export default function RejectedReview({taskId,onBack}:{taskId?:string,onBack:()=>void}){
 const [listing,setListing]=useState<R>(),[detail,setDetail]=useState<R>(),[selected,setSelected]=useState(''),[offset,setOffset]=useState(0),[filter,setFilter]=useState('');
 const [note,setNote]=useState(''),[targets,setTargets]=useState<string[]>([]),[repair,setRepair]=useState('{}'),[includeCached,setIncludeCached]=useState(true),[mode,setMode]=useState('release');
 const [preview,setPreview]=useState<R>(),[error,setError]=useState(''),[busy,setBusy]=useState(false),[accepted,setAccepted]=useState(false);
 const [acceptCoverage,setAcceptCoverage]=useState(false);
 const generation=useRef(0),selectedRef=useRef(selected),detailGeneration=useRef(0);selectedRef.current=selected;
 const base='/tasks/'+taskId+'/reviews';
 const loadDetail=async(batch:string)=>{const n=++detailGeneration.current;const d=await api(base+'/'+encodeURIComponent(batch));if(n===detailGeneration.current&&selectedRef.current===batch)setDetail(d);};
 const load=async()=>{if(!taskId)return;const n=++generation.current;try{const value=await api(base+'?limit=20&offset='+offset+'&status='+filter);if(n!==generation.current)return;setListing(value);if(!selectedRef.current&&value.rows.length)setSelected(value.rows[0].batch);if(selectedRef.current)await loadDetail(selectedRef.current);}catch(e:any){if(n===generation.current)setError(e.message);}};
 useEffect(()=>{setListing(undefined);setSelected('');setDetail(undefined);setOffset(0);setError('');},[taskId]);
 useEffect(()=>{load();const t=setInterval(load,5000);return()=>{clearInterval(t);generation.current++;detailGeneration.current++;};},[taskId,offset,filter]);
 useEffect(()=>{setDetail(undefined);setNote('');setPreview(undefined);setAccepted(false);setAcceptCoverage(false);setError('');if(selected){loadDetail(selected).then(()=>{}).catch(e=>setError(e.message));}},[selected]);
 useEffect(()=>{if(!detail)return;setTargets(detail.targets||[]);setRepair(JSON.stringify(Object.fromEntries(detail.tables.map((t:R)=>[t.name,[]])),null,2));},[detail?.batch]);
 useEffect(()=>{setPreview(undefined);setAccepted(false);},[note,JSON.stringify(targets),repair,includeCached,mode,acceptCoverage]);
 const run=async(fn:()=>Promise<any>)=>{setBusy(true);setError('');try{await fn();await load();}catch(e:any){setError(e.message);}finally{setBusy(false);}};
 const reviewPreview=async(action:string)=>{setMode(action);const parsed=action==='repair'?JSON.parse(repair):undefined;const result=await api(base+'/'+selected+'/preview',{action,note,targets,version:detail?.version,...(detail?.coverage_acceptance_available?{accept_coverage:acceptCoverage}:{}),...(action==='repair'?{rows:parsed.rows||parsed,include_cached:includeCached}:{})});setPreview(result);setAccepted(false);};
 const locked=busy||!listing?.target_matches||listing?.active||!!detail?.decision;
 const importFile=async(file:File)=>{if(file.size>4*1024*1024){setError('修补文件不能超过 4 MiB');return;}try{const value=JSON.parse(await file.text());setRepair(JSON.stringify(value.rows||value,null,2));setMode('repair');setError('');}catch{setError('请选择有效的 JSON 修补文件');}};
 if(!taskId)return <section className="workspace-page"><p>请先选择需要审计的任务。</p><button onClick={onBack}>返回任务清单</button></section>;
 return <div className="workspace-page review-workspace"><section><div className="heading"><div><h2>任务拒收批次人工审计</h2><p className="muted">任务 {taskId}</p></div><button onClick={onBack}>返回任务详情</button></div>
 {error&&<p role="alert" className="error">{error}</p>}
 <p>原始响应和拒收记录保留。人工放行及修补导入会记录说明、目标库和独立写入回执，可随原任务回退。</p>
 {listing?.active&&<p className="notice">任务仍在执行，可以查看拒收记录；等待任务完成或停止后再提交人工决定。</p>}
 {listing&&!listing.target_matches&&<p className="notice">当前数据库连接与原任务不同，仅可查看和导出。</p>}
 {listing&&<div className="phase-grid" aria-label="人工审计进度"><div><small>拒收批次总量</small><strong>{listing.counts.total}</strong></div><div><small>待审计</small><strong>{listing.counts.pending}</strong></div><div><small>执行中</small><strong>{listing.counts.queued}</strong></div><div><small>已放行 / 已修补</small><strong>{listing.counts.released} / {listing.counts.repaired}</strong></div><div><small>已拒绝</small><strong>{listing.counts.rejected}</strong></div><div><small>执行失败</small><strong>{listing.counts.failed}</strong></div></div>}
 <div className="workspace-toolbar"><label>审计状态<select aria-label="筛选人工审计状态" value={filter} onChange={e=>{setFilter(e.target.value);setOffset(0);}}><option value="">全部状态</option>{[['pending','待审计'],['queued','排队中'],['running','执行中'],['released','已放行'],['repaired','已修补'],['rejected','已拒绝'],['failed','执行失败']].map(([v,l])=><option key={v} value={v}>{l}</option>)}</select></label><button disabled={busy} onClick={()=>run(load)}>刷新拒收清单</button></div>
 <div className="scroll"><table><thead><tr><th>批次</th><th>审计状态</th><th>拒收原因</th><th>可放行缓存</th><th>操作</th></tr></thead><tbody>{listing?.rows.map((r:R)=><tr key={r.batch} className={r.batch===selected?'selected-task-row':''}><td><button onClick={()=>setSelected(r.batch)}>{r.batch}</button></td><td><Status value={r.status}/></td><td>{r.reasons.slice(0,3).join('；')||r.error}</td><td>{r.cached_rows} 行</td><td><button onClick={()=>setSelected(r.batch)}>审计此批次</button></td></tr>)}</tbody></table></div>
 {listing&&!listing.rows.length&&<p className="muted">没有符合条件的拒收批次。</p>}
 <div className="actions"><span>共 {listing?.total||0} 个批次</span><button disabled={!offset} onClick={()=>setOffset(Math.max(0,offset-20))}>上一页拒收</button><button disabled={!listing||offset+20>=listing.total} onClick={()=>setOffset(offset+20)}>下一页拒收</button></div></section>
 {detail&&<section aria-label="拒收批次审计详情"><div className="heading"><h2>批次 {detail.batch}</h2><Status value={detail.status}/></div><p>{detail.reasons.join('；')||detail.error}</p>
 <h3>拒收记录样例</h3><Table rows={detail.samples}/>{detail.samples_are_partial&&<p className="muted">这里只保留前 100 条样例，请结合原始响应核对完整范围。</p>}
 <details><summary>原始响应与归档证据 · {detail.sources.length} 份</summary>{detail.sources.map((s:R)=><div key={s.slice_id}><a href={s.url}>{s.endpoint} · {s.parameters.ts_code||s.parameters.index_code||s.parameters.trade_date||s.slice_id} · {s.rows} 行 · 下载原始归档</a><Status value={s.status}/></div>)}</details>
 <details><summary>本地表、字段合同及可入库样本</summary>{detail.tables.map((t:R)=><div key={t.name}><h3>{t.zh_name} · {t.name}</h3><p>主键：{t.keys.join('、')} · 可入库缓存 {t.cached_rows} 行</p><Table rows={t.fields}/><Table rows={t.preview}/></div>)}</details>
 <div className="actions"><a href={'/api'+base+'/'+selected+'/export'}>导出本批次审计元信息</a><a href={'/api'+base+'/'+selected+'/template'}>下载手工修补模板</a></div>
 {detail.decision?<div className="batch-preview"><h3>已记录的人工决定</h3><Table rows={[detail.decision]}/>{detail.decision.status==='failed'&&<button disabled={busy||listing?.active||!listing?.target_matches} onClick={()=>run(()=>api(base+'/'+selected+'/retry',{}))}>重试原决定</button>}{['failed','rejected'].includes(detail.decision.status)&&<button disabled={busy||listing?.active||!listing?.target_matches} onClick={()=>run(()=>api(base+'/'+selected+'/reopen',{}))}>重新审计</button>}</div>:<>
 <label>人工审计说明<textarea aria-label="人工审计说明" rows={3} value={note} onChange={e=>setNote(e.target.value)} placeholder="说明核对依据、接受哪些记录，或填写拒绝原因"/></label><Targets value={targets} onChange={setTargets}/>
 {detail.coverage_acceptance_available&&<label className="notice"><input type="checkbox" aria-label="人工接受现有归档的覆盖范围" checked={acceptCoverage} onChange={e=>setAcceptCoverage(e.target.checked)}/>我已核对原始归档，接受当前来源覆盖范围；完整性异常仍保留，只放行满足存储合同的记录</label>}
 <div className="actions"><button disabled={locked||!note.trim()||!targets.length||!detail.can_release||(detail.coverage_acceptance_available&&!acceptCoverage)} onClick={()=>run(()=>reviewPreview('release'))}>确认放行 · 预览可入库记录</button><button disabled={locked||!note.trim()} onClick={()=>run(async()=>{const r=await api(base+'/'+selected+'/reject',{note,version:detail.version});const link=document.createElement('a');link.href=r.export_url;link.download='rejected-'+selected+'.json';link.click();})}>拒绝并导出元信息</button><button disabled={locked} onClick={()=>setMode('repair')}>手工修补后导入</button></div>
 {!detail.can_release&&<p className="notice">{detail.release_reason}</p>}
 {mode==='repair'&&<div className="batch-preview" role="region" aria-label="手工修补编辑器"><h3>填写修补后的本地表记录</h3><p>按表名填写 JSON 数组，可只填写业务主键及需修补的字段。主键、类型和原任务范围必须有效；系统审计字段自动生成。时间戳请包含时区。</p><input aria-label="导入修补 JSON 文件" type="file" accept=".json,application/json" onChange={e=>{const f=e.target.files?.[0];if(f)importFile(f);}}/><textarea aria-label="修补记录 JSON" rows={16} value={repair} onChange={e=>setRepair(e.target.value)}/><label><input type="checkbox" checked={includeCached} onChange={e=>setIncludeCached(e.target.checked)}/>同时导入此批次已有的合格缓存记录</label><button disabled={locked||!note.trim()||!targets.length} onClick={()=>run(()=>reviewPreview('repair'))}>预览修补后导入</button></div>}
 {preview&&<div className="batch-preview" role="region" aria-label="人工审计写入预览"><h3>{preview.action==='repair'?'修补后导入预览':'确认放行预览'}</h3><Table rows={preview.rows}/><p>{preview.notice}</p><label><input type="checkbox" checked={accepted} onChange={e=>setAccepted(e.target.checked)}/>已核对本次范围、目标库及剩余拒收记录，确认执行所选处理</label><button className="primary" disabled={locked||!accepted} onClick={()=>run(async()=>{await api(base+'/'+selected+'/confirm',{token:preview.token});setPreview(undefined);})}>确认执行人工审计决定</button></div>}
 </>}
 </section>}
 {busy&&<p role="status" className="busy">正在处理人工审计…</p>}
 </div>;
}
