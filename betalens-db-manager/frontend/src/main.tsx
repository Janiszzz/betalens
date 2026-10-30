import React,{useEffect,useState} from 'react';
import {createRoot} from 'react-dom/client';
import '@fontsource/noto-sans-sc/400.css';
import './style.css';
import './workspace.css';
import IntegrityWorkbench from './IntegrityWorkbench';
import LegacyWorkspace from './LegacyWorkspace';
import DatabaseWorkspace from './DatabaseWorkspace';
import UpdateRequest from './UpdateRequest';
import TaskWorkspace from './TaskWorkspace';
import RejectedReview from './RejectedReview';
import {api} from './ui';
import {pages,pageUrl,useManagerRoute} from './navigation';
function App(){
 const {route,navigate}=useManagerRoute();const [table,setTable]=useState(route.table||'market_daily_raw');
 useEffect(()=>{if(route.table)setTable(route.table);},[route.table]);
 useEffect(()=>{let sid='',closed=false;api('/sessions',{}).then(s=>{sid=s.id;if(closed)navigator.sendBeacon('/api/sessions/'+sid+'/close');}).catch(()=>{});const heartbeat=setInterval(()=>{if(sid)api('/sessions/'+sid+'/heartbeat',{}).catch(()=>{});},15000);const before=(e:BeforeUnloadEvent)=>{e.preventDefault();e.returnValue='后台任务会继续执行，旧版前台任务在所有页面关闭后停止';};const leave=()=>{closed=true;if(sid)navigator.sendBeacon('/api/sessions/'+sid+'/close');};window.addEventListener('beforeunload',before);window.addEventListener('pagehide',leave);return()=>{clearInterval(heartbeat);window.removeEventListener('beforeunload',before);window.removeEventListener('pagehide',leave);leave();};},[]);
 const open=(page:string,name?:string)=>{if(name)setTable(name);navigate(page,name);};
 return <div className="app"><aside><div className="brand">Betalens<span>DB MANAGER</span></div><p className="muted">本地数据基础设施</p>{pages.map(({label})=><a key={label} href={pageUrl(label,['字段映射','更新与审计'].includes(label)?table:undefined)} className={(route.page===label||route.page==='拒收批次审计'&&label==='任务清单与回退')?'active':''} aria-current={(route.page===label||route.page==='拒收批次审计'&&label==='任务清单与回退')?'page':undefined} onClick={e=>{if(e.button!==0||e.ctrlKey||e.metaKey||e.shiftKey||e.altKey)return;e.preventDefault();open(label,['字段映射','更新与审计'].includes(label)?table:undefined);}}>{label}</a>)}<small>后台任务关闭网页后继续<br/>请保持管理器服务运行</small></aside><main><header><div><h1>{route.page}</h1><p>数据库配置 · 多表更新 · 任务执行与回退</p></div><span className="badge">本地运行</span></header>
 {route.removedSchedule&&<div className="notice">自动更新功能已移除。历史任务保留，请手动创建和启动更新任务。</div>}
 {route.page==='本地表与配置'&&<DatabaseWorkspace onNavigate={open}/>}
 {route.page==='更新与审计'&&<UpdateRequest initialTable={route.table} onCreated={id=>navigate('任务清单与回退',undefined,id)} onNavigate={open}/>}
 {route.page==='任务清单与回退'&&<TaskWorkspace selectedId={route.task} onSelect={id=>navigate('任务清单与回退',undefined,id)} onNavigate={open} onReview={id=>navigate('拒收批次审计',undefined,id)}/>}
 {route.page==='拒收批次审计'&&<RejectedReview taskId={route.task} onBack={()=>navigate('任务清单与回退',undefined,route.task)}/>}
 {route.page==='业务完整性'&&<IntegrityWorkbench api={api}/>}
 {route.page==='字段映射'&&<LegacyWorkspace kind="mapping" initialTable={route.table||table} onNavigate={open}/>}
 {route.page==='订阅源'&&<LegacyWorkspace kind="sources" onNavigate={open}/>}
 <footer>Betalens DB Manager · 时间显示为北京时间 · 所有写入可追溯至任务</footer></main></div>;
}
createRoot(document.getElementById('root')!).render(<App/>);
