"""Independent read-only monitoring; never initialize state or a data worker."""
from __future__ import annotations

from contextlib import closing
from datetime import datetime, timezone
from types import SimpleNamespace
import json
import re
import sqlite3
import threading
import time

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .security import public_data


class ReadOnlyMonitor:
    def __init__(self, settings):
        self.settings=settings
        self.path=settings.state_dir/'tasks.sqlite3'
        self.cache={}
        self.lock=threading.Lock()

    def connect(self):
        # mode=ro also prevents accidentally creating a database at a wrong path.
        conn=sqlite3.connect(self.path.as_uri()+'?mode=ro',uri=True,timeout=3)
        conn.row_factory=sqlite3.Row
        return conn

    @staticmethod
    def validate(job):
        if not re.fullmatch('[a-f0-9]{32}',job):raise ValueError('任务编号格式无效')

    def logs(self,job,limit=500):
        self.validate(job)
        with closing(self.connect()) as conn:
            exists=conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='event_log'").fetchone()
            rows=list(conn.execute('SELECT id,created,event,value FROM event_log WHERE job=? ORDER BY id DESC LIMIT ?',
                                   (job,max(1,min(limit,500))))) if exists else []
        return public_data([{'id':r['id'],'created':r['created'],'event':r['event'],**json.loads(r['value'])} for r in reversed(rows)],self.settings)

    def snapshot(self,job):
        self.validate(job)
        with self.lock:
            cached=self.cache.get(job)
            if cached and time.monotonic()-cached[0]<3:return cached[1]
            result=self._snapshot(job)
            # Bound memory when viewing different history tasks.
            if len(self.cache)>=16:self.cache.clear()
            self.cache[job]=(time.monotonic(),result)
            return result

    def _snapshot(self,job):
        with closing(self.connect()) as conn:
            conn.execute('BEGIN')
            row=conn.execute("""SELECT id,created,updated,status,substr(error,1,2000) AS error,
                json_extract(plan,'$.request.tables') AS tables,
                json_extract(plan,'$.request.targets') AS targets,
                json_extract(plan,'$.request.start_date') AS start_date,
                json_extract(plan,'$.request.end_date') AS end_date FROM jobs WHERE id=?""",(job,)).fetchone()
            if row is None:raise ValueError('任务不存在')
            result=dict(row)
            for name in ('tables','targets'):result[name]=json.loads(result[name] or '[]')
            tables={r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            frozen=conn.execute("SELECT value FROM ledger WHERE table_name='pipeline_work_plan' AND row_key=?",(job,)).fetchone()
            plan=json.loads(frozen[0]) if frozen else None
            units=[]
            if 'pipeline_units' in tables:
                for r in conn.execute("""SELECT id,kind,status,json_extract(payload,'$.child_id') AS child,
                    json_extract(payload,'$.target') AS target,json_extract(result,'$.child_id') AS result_child
                    FROM pipeline_units WHERE job=?""",(job,)):
                    units.append({'id':r['id'],'kind':r['kind'],'status':r['status'],
                                  'payload':{'child_id':r['child'],'target':r['target']},'result':{'child_id':r['result_child']}})
            parts=[{'id':r['id'],'status':r['status'],
                    'result':{'children':json.loads(r['children'])} if r['children'] else None}
                   for r in conn.execute("SELECT id,status,json_extract(result,'$.children') AS children FROM slices WHERE job=?",(job,))]
            if plan:
                from .task_progress import summary
                class FrozenState:
                    def ledger_get(self,*_):return plan
                    def ledger_put(self,*_):raise AssertionError('只读监控不得写入任务状态')
                item={'id':job,'status':row['status'],'slices':parts,'tasks':[],
                      'plan':{'request':{'targets':result['targets']}}}
                result['progress']=summary(SimpleNamespace(state=FrozenState()),item,units)
            else:result['progress']={'phase':'planning'}
            result['writes']=[dict(r) for r in conn.execute("""SELECT target,status,count(*) AS batches,
                sum(CAST(json_extract(result,'$.applied') AS INTEGER)) AS applied,max(updated) AS updated
                FROM pipeline_receipts WHERE job=? AND batch NOT LIKE 'rollback:%' GROUP BY target,status""",(job,))] if 'pipeline_receipts' in tables else []
            result['audit']=[dict(r) for r in conn.execute('SELECT target,sum(revision>acknowledged) AS pending FROM pipeline_sync WHERE job=? GROUP BY target',(job,))] if 'pipeline_sync' in tables else []
            latest=conn.execute('SELECT created FROM event_log WHERE job=? ORDER BY id DESC LIMIT 1',(job,)).fetchone() if 'event_log' in tables else None
            result['last_log_at']=latest[0] if latest else None
        try:
            beat=json.loads((self.settings.state_dir/'worker-heartbeat.json').read_text())
            age=(datetime.now(timezone.utc)-datetime.fromisoformat(beat['at'])).total_seconds()
            result['worker']={k:beat.get(k) for k in ('status','job','at')}
            result['worker']['fresh']=0<=age<30
        except (OSError,ValueError,KeyError):result['worker']={'fresh':False}
        result['observed_at']=datetime.now(timezone.utc).isoformat()
        return public_data(result,self.settings)


def create_monitor_app(settings):
    monitor=ReadOnlyMonitor(settings)
    app=FastAPI(title='Betalens 只读任务监控',docs_url=None,redoc_url=None,openapi_url=None)
    app.state.monitor=monitor
    app.add_middleware(TrustedHostMiddleware,allowed_hosts=['127.0.0.1','localhost','[::1]','testserver'])

    @app.exception_handler(ValueError)
    async def invalid(_,exc):return JSONResponse({'detail':str(exc)},status_code=404,headers={'Cache-Control':'no-store'})

    @app.exception_handler(sqlite3.Error)
    async def unavailable(_,exc):return JSONResponse({'detail':'任务状态文件暂不可读，正在重试'},status_code=503,headers={'Cache-Control':'no-store'})

    @app.get('/api/health')
    def health():
        with closing(monitor.connect()) as conn:conn.execute('SELECT 1').fetchone()
        return {'status':'ok','read_only':True}

    @app.get('/api/tasks/{job}/status')
    def status(job:str):return JSONResponse(monitor.snapshot(job),headers={'Cache-Control':'no-store'})

    @app.get('/api/tasks/{job}/logs')
    def logs(job:str,limit:int=500):return JSONResponse(monitor.logs(job,limit),headers={'Cache-Control':'no-store'})

    @app.get('/')
    @app.get('/tasks')
    def page():return HTMLResponse(PAGE,headers={'Cache-Control':'no-store'})

    return app


PAGE='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Betalens · 只读任务监控</title><style>
body{font-family:system-ui,"Microsoft YaHei",sans-serif;background:#f3f6fa;color:#203044;margin:0;padding:24px;line-height:1.65}main{max-width:1280px;margin:auto}h1{font-size:25px;margin:0}header,section{background:white;border:1px solid #dde5ef;border-radius:12px;padding:20px;margin-bottom:18px}small,.muted{color:#627187}input[type=text]{padding:9px;width:min(580px,70%);border:1px solid #bccbdc;border-radius:6px}button{padding:8px 18px;background:#195fad;color:white;border:0;border-radius:6px;cursor:pointer}a{color:#195fad}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:16px;margin-top:16px}.card{background:#f5f8fd;border-radius:8px;padding:14px}.card strong{display:block;font-size:21px}progress{width:100%;accent-color:#2567ad}.error{color:#a82828;white-space:pre-wrap}.controls{display:flex;gap:16px;flex-wrap:wrap;margin-bottom:12px}.stream{max-height:65vh;overflow:auto;font-size:13px}.line{display:grid;grid-template-columns:175px 220px 130px 1fr;gap:12px;padding:8px;border-bottom:1px solid #edf0f5}.line.bad{color:#ad3030;background:#fff6f4}.line time{color:#66758a}.line details{grid-column:2/-1}pre{white-space:pre-wrap;word-break:break-all;background:#f4f6f9;padding:10px}@media(max-width:800px){body{padding:12px}.line{display:block}.line span,.line time{display:block}}
</style><main><header><h1>只读任务监控</h1><p class="muted">实时查看任务进度与最新日志。此页面不提供启动、停止或写入操作。</p><input id="task" type="text" aria-label="任务编号" placeholder="输入完整任务编号"><button id="view">查看</button> <a id="manager">返回管理器</a><p id="identity"></p><div id="statusError" class="error" role="alert"></div><div id="facts" class="muted"></div><div id="cards" class="cards"></div></header>
<section><div class="controls"><strong>最新任务日志</strong><label><input id="follow" type="checkbox" checked>自动滚动</label><label><input id="errors" type="checkbox">仅错误</label><input id="filter" type="text" aria-label="筛选日志" placeholder="接口、事件或关键词"></div><div id="logError" class="error" role="alert"></div><div id="logs" class="stream" aria-label="实时任务日志"></div><small id="stamp">每 3 秒刷新，直接读取最新 500 条，不追赶历史积压。</small></section></main>
<script>
const $=id=>document.getElementById(id),names={running:'执行中',queued:'排队中',partial:'部分完成',completed:'已完成',paused:'已暂停',interrupted:'已中断',cancelled:'已停止',rolled_back:'已回退',idle:'空闲',postgres:'PostgreSQL',clickhouse:'ClickHouse'};
const events={request_started:'发起请求',http_response:'收到响应',response_archived:'响应已归档',response_reused:'复用归档',pipeline_write_group_completed:'合并写入完成',pipeline_target_batch_completed:'目标批次完成',execution_started:'开始执行',execution_failed:'执行失败',rate_limit_wait:'等待请求配额',rate_limit_cooldown:'限频冷却',source_retry_scheduled:'安排自动重试',archive_missing_retry:'补拉缺失归档',stop_requested:'收到停止请求'};
let id=new URLSearchParams(location.search).get('task')||'',rows=[],generation=0,busyStatus=false,busyLogs=false;
const time=v=>v?new Date(v).toLocaleString('zh-CN',{timeZone:'Asia/Shanghai',hour12:false}):'—';
const num=v=>Number(v||0).toLocaleString('zh-CN');
function text(tag,value){const e=document.createElement(tag);e.textContent=value;return e;}
async function get(path){const c=new AbortController(),timer=setTimeout(()=>c.abort(),8000);try{const r=await fetch(path,{signal:c.signal,cache:'no-store'}),v=await r.json();if(!r.ok)throw Error(v.detail||'请求失败');return v;}finally{clearTimeout(timer);}}
function card(label,done,total,note){const e=text('div','');e.className='card';e.append(text('small',label),text('strong',num(done)+' / '+num(total)));const p=document.createElement('progress');p.value=done;p.max=total||1;e.append(p,text('small',note));$('cards').append(e);}
async function loadStatus(){if(!id||busyStatus)return;busyStatus=true;const current=generation;try{const v=await get('/api/tasks/'+id+'/status');if(current!==generation)return;$('statusError').textContent='';$('identity').textContent='任务 '+v.id+' · '+(names[v.status]||v.status);$('facts').textContent=v.start_date+' — '+v.end_date+' · '+v.tables.length+' 张表 · worker '+(v.worker.fresh?(names[v.worker.status]||v.worker.status):'心跳待确认')+' · 最新日志 '+time(v.last_log_at);$('cards').replaceChildren();const p=v.progress;if(p.phase!=='planning'){card('逻辑请求处理',p.requests.processed,p.requests.total,'异常 '+num(p.requests.failed)+' 项');card('转换批次处理',p.batches.processed,p.batches.total,'通过 '+num(p.batches.passed)+' · 拒收 '+num(p.batches.rejected));for(const [target,w] of Object.entries(p.writes||{})){const receipt=v.writes.find(r=>r.target===target&&r.status==='completed');card(names[target]||target,w.written,w.total,'已完整写入批次 · 累计处理 '+num(receipt?.applied)+' 行次');}}$('stamp').textContent='状态更新 '+time(v.observed_at)+'（北京时间）· 每 3 秒刷新，始终获取最新 500 条日志。';}catch(e){if(current===generation)$('statusError').textContent=e.name==='AbortError'?'状态请求超时，正在重试；已有日志继续保留':e.message;}finally{busyStatus=false;}}
function render(){const stream=$('logs'),query=$('filter').value.toLowerCase();stream.replaceChildren();for(const r of rows){const bad=!!r.error||Number(r.http_status)>=400||/failed|blocked/i.test(r.event);if($('errors').checked&&!bad)continue;const detail=JSON.stringify(r);if(query&&!detail.toLowerCase().includes(query))continue;const line=text('div','');line.className='line'+(bad?' bad':'');const phase=r.target?(names[r.target]||r.target)+' 写入':/request|response|source|rate|archive/.test(r.event)?'下载':'任务';const result=r.error||r.reason||r.message||(r.http_status?'HTTP '+r.http_status:r.status||'');line.append(text('time',time(r.created)),text('strong',phase+' · '+(events[r.event]||r.event)),text('span',r.endpoint||r.dataset||''),text('span',result+(r.rows!==undefined?' · '+num(r.rows)+' 行':'')+(r.elapsed_ms!==undefined?' · '+num(r.elapsed_ms)+' ms':'')));const more=document.createElement('details');more.append(text('summary','明细'),text('pre',JSON.stringify(r,null,2)));line.append(more);stream.append(line);}if(!stream.childElementCount)stream.append(text('p','暂无匹配日志'));if($('follow').checked)stream.scrollTop=stream.scrollHeight;}
async function loadLogs(){if(!id||busyLogs)return;busyLogs=true;const current=generation;try{const v=await get('/api/tasks/'+id+'/logs?limit=500');if(current!==generation)return;rows=v;$('logError').textContent='';render();}catch(e){if(current===generation)$('logError').textContent=e.name==='AbortError'?'日志请求超时，正在重试':e.message;}finally{busyLogs=false;}}
function view(){const value=$('task').value.trim();if(!/^[a-f0-9]{32}$/.test(value)){$('statusError').textContent='请输入完整的 32 位任务编号';return;}id=value;generation++;rows=[];render();history.replaceState(null,'','/tasks?task='+id);$('manager').href='http://'+location.hostname+':8010/tasks?task='+id;void loadStatus();void loadLogs();}
$('task').value=id;$('view').onclick=view;$('filter').oninput=render;$('errors').onchange=render;$('follow').onchange=render;if(id)view();setInterval(()=>{void loadStatus();void loadLogs();},3000);document.addEventListener('visibilitychange',()=>{if(!document.hidden){void loadStatus();void loadLogs();}});window.addEventListener('online',()=>{void loadStatus();void loadLogs();});
</script></html>'''
