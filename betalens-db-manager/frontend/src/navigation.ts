import {useCallback,useEffect,useState} from 'react';
export const pages=[
 {label:'本地表与配置',path:'/tables'},
 {label:'更新与审计',path:'/updates'},
 {label:'任务清单与回退',path:'/tasks'},
 {label:'业务完整性',path:'/integrity'},
 {label:'字段映射',path:'/mappings'},
 {label:'订阅源',path:'/sources'},
];
export function pageUrl(label:string,table?:string,task?:string){
 const path=label==='拒收批次审计'?'/tasks/review':pages.find(p=>p.label===label)?.path||'/tables',params=new URLSearchParams();
 if(table&&['字段映射','更新与审计'].includes(label))params.set('table',table);
 if(task&&['任务清单与回退','拒收批次审计'].includes(label))params.set('task',task);
 return path+(params.size?'?'+params:'');
}
function canonicalUrl(){
 const url=new URL(location.href),path=url.pathname.replace(/\/+$/,'')||'/';
 if(path==='/settings')return '/tables?section=connections';
 if(path==='/schedules')return '/tasks?removed=schedules';
 if(path==='/tasks'&&url.searchParams.has('table')&&!url.searchParams.has('task'))return '/updates'+url.search;
 return (path==='/'?'/tables':path)+url.search;
}
function readRoute(){const url=new URL(canonicalUrl(),location.origin);return {page:url.pathname==='/tasks/review'?'拒收批次审计':pages.find(p=>p.path===url.pathname)?.label||'本地表与配置',table:url.searchParams.get('table')||undefined,task:url.searchParams.get('task')||undefined,removedSchedule:url.searchParams.get('removed')==='schedules'};}
export function useManagerRoute(){
 const [route,setRoute]=useState(readRoute);
 useEffect(()=>{if(location.pathname+location.search!==canonicalUrl())history.replaceState(null,'',canonicalUrl());const changed=()=>{if(location.pathname+location.search!==canonicalUrl())history.replaceState(null,'',canonicalUrl());setRoute(readRoute());};window.addEventListener('popstate',changed);return()=>window.removeEventListener('popstate',changed);},[]);
 const navigate=useCallback((page:string,table?:string,task?:string)=>{const url=pageUrl(page,table,task);if(location.pathname+location.search!==url)history.pushState(null,'',url);setRoute(readRoute());},[]);
 return {route,navigate};
}
