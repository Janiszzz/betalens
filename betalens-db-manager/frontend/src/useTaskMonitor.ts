import {useEffect,useRef,useState} from 'react';
import {api,type R} from './ui';

const LOG_LIMIT=500;
const REQUEST_TIMEOUT=15000;

export function useTaskMonitor(id?:string){
 const [job,setJob]=useState<R>(),[logs,setLogs]=useState<R[]>([]),[error,setError]=useState(''),[connected,setConnected]=useState(false);
 const refresh=useRef<()=>Promise<void>>(async()=>{});
 useEffect(()=>{
  let active=true,socket:WebSocket|undefined,timer:ReturnType<typeof setTimeout>|undefined;
  let delay=1000,loadingJob=false,loadingLogs=false,lastMessage=Date.now();
  const requests=new Set<AbortController>();
  setJob(undefined);setLogs([]);setError('');setConnected(false);

  const request=async(path:string)=>{
   const controller=new AbortController();requests.add(controller);
   const timeout=setTimeout(()=>controller.abort(),REQUEST_TIMEOUT);
   try{return await api(path,undefined,controller.signal);}
   finally{clearTimeout(timeout);requests.delete(controller);}
  };
  const append=(rows:R[])=>{
   if(!active||!rows.length)return;
   // A slower HTTP snapshot must not overwrite newer socket records.
   setLogs(prev=>[...new Map([...prev,...rows].map(r=>[String(r.id),r])).values()]
    .sort((a,b)=>Number(a.id)-Number(b.id)).slice(-LOG_LIMIT));
  };
  const loadJob=async()=>{
   if(!id||!active||loadingJob)return;
   loadingJob=true;
   try{const value=await request('/task-history/'+id);if(active){setJob(value);setError('');}}
   catch(e:any){if(active)setError(e.name==='AbortError'?'任务状态请求超时，正在重试':e.message);}
   finally{loadingJob=false;}
  };
  const loadLogs=async()=>{
   if(!id||!active||loadingLogs)return;
   loadingLogs=true;
   try{append(await request('/tasks/'+id+'/logs?latest=true&limit='+LOG_LIMIT));}
   catch(e:any){if(active)setError(e.name==='AbortError'?'最新日志请求超时，正在重试':e.message);}
   finally{loadingLogs=false;}
  };
  const load=async()=>{
   await Promise.all([loadJob(),socket?.readyState===WebSocket.OPEN?Promise.resolve():loadLogs()]);
  };
  refresh.current=load;

  const connect=()=>{
   if(!id||!active)return;
   clearTimeout(timer);
   if(socket){socket.onclose=null;socket.onmessage=null;socket.onerror=null;socket.onopen=null;socket.close();}
   setConnected(false);
   const current=new WebSocket(`${location.protocol==='https:'?'wss':'ws'}://${location.host}/api/live/${id}?latest=true&limit=${LOG_LIMIT}`);
   socket=current;lastMessage=Date.now();
   current.onopen=()=>{if(active&&socket===current){setConnected(true);delay=1000;lastMessage=Date.now();}};
   current.onmessage=e=>{
    if(!active||socket!==current)return;
    lastMessage=Date.now();
    const value=JSON.parse(e.data);
    if(value.type==='logs')append(value.records);
    else if(value.type==='log')append([value.record]);
    else if(value.type==='task')void loadJob();
   };
   current.onclose=()=>{
    if(!active||socket!==current)return;
    socket=undefined;setConnected(false);void loadLogs();
    timer=setTimeout(connect,delay);delay=Math.min(15000,delay*2);
   };
   current.onerror=()=>current.close();
  };
  const wake=()=>{
   if(document.visibilityState==='hidden')return;
   void loadJob();void loadLogs();connect();
  };
  void load();connect();
  const poll=setInterval(()=>{
   if(socket?.readyState===WebSocket.OPEN&&Date.now()-lastMessage>REQUEST_TIMEOUT)connect();
   void load();
  },3000);
  document.addEventListener('visibilitychange',wake);
  window.addEventListener('pageshow',wake);
  window.addEventListener('online',wake);
  return()=>{
   active=false;clearInterval(poll);clearTimeout(timer);
   document.removeEventListener('visibilitychange',wake);
   window.removeEventListener('pageshow',wake);
   window.removeEventListener('online',wake);
   for(const controller of requests)controller.abort();
   if(socket){socket.onclose=null;socket.close();}
  };
 },[id]);
 return {job,logs,error,connected,refresh:()=>refresh.current()};
}
