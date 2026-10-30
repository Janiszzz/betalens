import {test,expect,type APIRequestContext} from '@playwright/test';

async function createTask(request:APIRequestContext){
 const r=await request.post('/api/tasks',{data:{pipeline_version:2,tables:['security_dim'],targets:['postgres','clickhouse'],start_date:'2026-10-13',end_date:'2026-10-13'}});
 expect(r.ok(),await r.text()).toBeTruthy();return r.json();
}
const originals=()=>['postgres','clickhouse'].flatMap(target=>['first','manual-repair'].map(id=>({batch:id+':security_dim:0',target,status:'completed',result:{table:'security_dim',applied:5}})));
const inverse=(target:string,id:string,status:string,applied=0)=>({batch:'rollback:'+id+':security_dim:0',target,status,result:status==='completed'?{applied,verified:true}:null,error:status==='failed'?'后续修改发生冲突':null});

test('回退排队、执行、冲突与恢复：固定分母且只统计完成回执',async({page,request},info)=>{
 page.on('dialog',d=>d.accept());const errors:string[]=[];page.on('pageerror',e=>errors.push(e.message));
 const job=await createTask(request);let socket:any;
 let state:any={...job,status:'partial',target_matches:true,tasks:[{id:'rollback-op',operation:'rollback',status:'queued'}],pipeline:{receipts:originals(),reviews:{pending:2}}};
 await page.route('**/api/task-history/'+job.id,route=>route.fulfill({json:state}));
 await page.route('**/api/tasks/'+job.id+'/logs?*',route=>route.fulfill({json:[]}));
 await page.routeWebSocket('**/api/live/'+job.id+'?*',ws=>{socket=ws;});
 await page.goto('/tasks?task='+job.id);
 const card=page.getByRole('region',{name:'回退进度'}),total=card.getByRole('progressbar',{name:'回退总进度',exact:true});
 await expect(card).toContainText('回退排队中');await expect(total).toHaveAttribute('max','4');await expect(total).toHaveAttribute('value','0');
 state.tasks[0].status='running';state.pipeline.receipts=[...originals(),inverse('postgres','first','completed',5),inverse('clickhouse','first','prepared')];socket.send(JSON.stringify({type:'task'}));
 await expect(card).toContainText('回退执行中');await expect(total).toHaveAttribute('value','1');await expect(total).toHaveAttribute('max','4');await expect(card).toContainText('25%');
 const ch=card.getByLabel('ClickHouse 回退详情');await expect(ch).toContainText('当前处理：security_dim');await expect(ch.getByRole('progressbar')).toHaveAttribute('value','0');
 await expect(card.getByLabel('PostgreSQL 回退详情')).toContainText('实际逆操作 5 行');
 state.tasks[0]={...state.tasks[0],status:'failed',error:'后续修改发生冲突'};state.pipeline.receipts[5]=inverse('clickhouse','first','failed');socket.send(JSON.stringify({type:'task'}));
 await expect(card).toContainText('回退失败');await expect(card.getByRole('alert')).toContainText('后续修改发生冲突');await expect(total).toHaveAttribute('value','1');await expect(ch).toContainText('失败 1');
 // Reloading a failed rollback must keep its progress visible, even with rejected batches pending review.
 await page.reload();await expect(card).toContainText('回退失败');await expect(page).toHaveURL(new RegExp('/tasks\\?task='+job.id+'$'));
 state.tasks.push({id:'retry-op',operation:'rollback',status:'running'});state.pipeline.receipts=[...originals(),...['postgres','clickhouse'].flatMap(t=>['first','manual-repair'].map(id=>inverse(t,id,'completed',5)))];socket.send(JSON.stringify({type:'task'}));
 await expect(total).toHaveAttribute('value','4');await expect(total).toHaveAttribute('max','4');await expect(card).toContainText('正在完成审计与收尾');
 state.status='rolled_back';state.tasks[1].status='completed';socket.send(JSON.stringify({type:'task'}));await expect(card).toContainText('回退完成');await expect(card).toContainText('100%');
 await page.screenshot({path:info.outputPath('rollback-completed.png'),fullPage:true});
 expect(errors).toEqual([]);
});

test('单库回退显示双库累计量，切换任务和重载不串进度',async({page,request})=>{
 page.on('dialog',d=>d.accept());const first=await createTask(request),second=await createTask(request);
 const value={...first,status:'partially_rolled_back',target_matches:true,tasks:[{id:'single',operation:'rollback',status:'completed'}],pipeline:{receipts:[...originals(),inverse('postgres','first','completed',5),inverse('postgres','manual-repair','completed',5)]}};
 await page.route('**/api/task-history/'+first.id,route=>route.fulfill({json:value}));
 await page.route('**/api/task-history/'+second.id,route=>route.fulfill({json:{...second,target_matches:true,pipeline:{receipts:[]}}}));
 await page.goto('/tasks?task='+first.id);const card=page.getByRole('region',{name:'回退进度'});
 await expect(card).toContainText('本轮回退完成，任务仍有未回退批次');await expect(card.getByLabel('PostgreSQL 回退进度',{exact:true})).toHaveAttribute('value','2');
 await expect(card.getByLabel('ClickHouse 回退进度',{exact:true})).toHaveAttribute('value','0');await expect(card.getByLabel('回退总进度',{exact:true})).toHaveAttribute('max','4');
 await page.getByRole('button',{name:second.id.slice(0,12),exact:true}).click();await expect(card).toHaveCount(0);
 await page.goBack();await expect(card).toContainText('50%');await page.reload();await expect(card).toContainText('50%');
 await expect(card.getByLabel('回退总进度',{exact:true})).toHaveAttribute('max','4');
});
