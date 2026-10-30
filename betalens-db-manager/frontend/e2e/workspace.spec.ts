import {test,expect,type APIRequestContext} from '@playwright/test';
const first='workspace_ui_quotes',second='workspace_ui_prices';
async function prepare(request:APIRequestContext){
 for(const name of [first,second]){
  const definition={name,zh_name:name===first?'工作台行情':'工作台价格',description:'隔离浏览器表',grain:'证券与日期一行',keys:['code','business_date'],time_field:'business_date',temporal_rule:'calendar_day',fields:[{name:'code',type:'text'},{name:'business_date',type:'date'},{name:'price',type:'numeric',precision:28,scale:8}]};
  const r=await request.post('/api/tables',{data:{definition}});expect(r.ok(),await r.text()).toBeTruthy();
 }
 const source={source_id:'workspace_ui_http',provider:'http',base_url:'http://127.0.0.1:18082',endpoints:{prices:{path:'/prices',strategy:'range',keys:['ticker','day'],date_field:'day',date_format:'%Y/%m/%d',fields:[{name:'ticker',type:'text'},{name:'day',type:'date'},{name:'price_milli',type:'numeric'}]}}};
 const r=await request.post('/api/sources',{data:source});expect(r.ok(),await r.text()).toBeTruthy();
 for(const name of [first,second]){
  const mapping={format_version:1,table:name,bindings:[{source_id:'workspace_ui_http',endpoint:'prices',fields:[{source_field:'ticker',column:'code'},{source_field:'day',column:'business_date',date_format:'%Y/%m/%d'},{source_field:'price_milli',column:'price',multiplier:'0.001'}]}]};
  const saved=await request.post('/api/tables/'+name+'/mapping',{data:{mapping}});expect(saved.ok(),await saved.text()).toBeTruthy();
 }
}
test('数据库分组多选建表 → 多表任务 → 跳转后手动启动 → 监听与单次回退',async({page,request},info)=>{
 page.on('dialog',d=>d.accept());const errors:string[]=[];page.on('pageerror',e=>errors.push(e.message));await prepare(request);
 await page.goto('/tables');await expect(page.getByRole('heading',{name:'本地表与配置',exact:true})).toBeVisible();
 await expect(page.getByRole('link',{name:'自动更新',exact:true})).toHaveCount(0);await expect(page.getByRole('button',{name:'查看数据样本',exact:true})).toHaveCount(0);
 for(const target of ['PostgreSQL','ClickHouse']){
  await page.getByLabel(target+' 搜索表',{exact:true}).fill('workspace_ui');
  await page.getByLabel(target+' 选择 '+first,{exact:true}).check();await page.getByLabel(target+' 选择 '+second,{exact:true}).check();
 }
 await page.getByRole('button',{name:'一键创建缺失表',exact:true}).click();
 const preview=page.getByRole('region',{name:'批量建表预览'});await expect(preview).toContainText('自动加入依赖');
 await preview.getByRole('button',{name:'确认一键建表',exact:true}).click();
 await expect(page.getByRole('region',{name:'批量建表结果'})).toContainText('已完成',{timeout:30000});
 await page.screenshot({path:info.outputPath('database-groups.png'),fullPage:true});
 await page.getByRole('link',{name:'更新与审计',exact:true}).click();await expect(page).toHaveURL(/\/updates/);
 await page.getByRole('button',{name:'清空选择',exact:true}).click();await page.getByLabel('搜索更新表',{exact:true}).fill('workspace_ui');
 await page.getByLabel('更新表 '+first,{exact:true}).check();await page.getByLabel('更新表 '+second,{exact:true}).check();
 await page.getByLabel('任务起始日期').fill('2026-10-13');await page.getByLabel('任务结束日期').fill('2026-10-13');
 // Leave the source on automatic matching; both selected tables share one HTTP request.
 await page.getByRole('button',{name:'匹配并预览 API 请求',exact:true}).click();
 await expect(page.getByRole('region',{name:'多表请求预览'})).toContainText('共 1 个初始请求');
 await page.getByRole('button',{name:'创建任务并查看详情',exact:true}).click();await expect(page).toHaveURL(/\/tasks\?task=[a-f0-9]{32}$/);
 const id=new URL(page.url()).searchParams.get('task')!;const before=await (await request.get('/api/tasks/'+id)).json();expect(before.status).toBe('planned');expect(before.tasks).toHaveLength(0);expect(before.plan.request.tables).toEqual([first,second]);
 const detail=page.getByRole('region',{name:'任务执行详情'});await detail.getByRole('button',{name:'启动异步采集并自动入库',exact:true}).click();
 await expect.poll(async()=>(await (await request.get('/api/tasks/'+id)).json()).status,{timeout:60000}).toBe('completed');
 await expect(detail.getByRole('heading',{name:'在线日志',exact:true})).toBeVisible();await expect(detail.getByLabel('实时任务日志')).toContainText('目标批次完成',{timeout:10000});
 await page.screenshot({path:info.outputPath('multi-table-task.png'),fullPage:true});
 await page.reload();await expect(page.getByRole('region',{name:'任务执行详情'})).toContainText(id);await detail.getByRole('button',{name:'回退本任务',exact:true}).click();
 await detail.getByRole('button',{name:'确认回退所选数据库',exact:true}).click();
 await expect.poll(async()=>(await (await request.get('/api/tasks/'+id)).json()).status,{timeout:60000}).toBe('rolled_back');
 const rollbackProgress=detail.getByRole('region',{name:'回退进度'});await expect(rollbackProgress).toContainText('回退完成');
 await expect(rollbackProgress.getByLabel('回退总进度',{exact:true})).toHaveAttribute('max','4');
 await expect(rollbackProgress.getByLabel('回退总进度',{exact:true})).toHaveAttribute('value','4');
 await expect(detail.getByLabel('分库回退结果')).toContainText('PostgreSQL');await expect(detail.getByLabel('分库回退结果')).toContainText('ClickHouse');
 await detail.getByRole('tab',{name:'缓存与写入审计',exact:true}).click();
 await detail.getByRole('button',{name:'优化请求并建立关联任务',exact:true}).click();
 await expect(page).not.toHaveURL(new RegExp('task='+id+'$'));
 const linkedId=new URL(page.url()).searchParams.get('task')!;
 const linked=await (await request.get('/api/tasks/'+linkedId)).json();
 expect(linked.status).toBe('planned');expect(linked.tasks).toHaveLength(0);
 expect(linked.plan.request).toEqual(before.plan.request);expect(linked.plan.parent_task_id).toBe(id);
 expect(linked.plan.retrieval_replan.reused_slices).toBeGreaterThan(0);
 await expect(detail).toContainText('这是优化请求后的关联任务');
 await detail.getByRole('button',{name:'启动异步采集并自动入库',exact:true}).click();
 await expect.poll(async()=>(await (await request.get('/api/tasks/'+linkedId)).json()).status,{timeout:60000}).toBe('completed');
 const logs=await (await request.get('/api/tasks/'+linkedId+'/logs?after=0&limit=1000')).json();
 expect(logs.filter((r:any)=>r.event==='request_started')).toHaveLength(0);
 await detail.getByRole('button',{name:'回退本任务',exact:true}).click();await detail.getByRole('button',{name:'确认回退所选数据库',exact:true}).click();
 await expect.poll(async()=>(await (await request.get('/api/tasks/'+linkedId)).json()).status,{timeout:60000}).toBe('rolled_back');
 expect(errors).toEqual([]);
});

test('旧链接转向、六页导航、配置另存提示与表元信息展示',async({page})=>{
 page.on('dialog',d=>d.accept());await page.goto('/settings');await expect(page).toHaveURL(/\/tables\?section=connections$/);
 await expect(page.getByLabel('凭据配置文件保存位置')).toHaveValue(/\.env$/);
 const current=await page.getByLabel('凭据配置文件保存位置').inputValue();expect(current).toContain('.env');
 await page.route('**/api/config',async route=>{
  if(route.request().method()!=='POST')return route.continue();
  const data=route.request().postDataJSON();expect(data.env_file).toBe('/tmp/workspace-ui-copy.env');
  await route.fulfill({json:{saved:true,path:data.env_file,active_path:current,restart_required:true,config:{},restart_command:'python -m betalens_db_manager --env /tmp/workspace-ui-copy.env serve --port 8010'}});
 });
 await page.getByLabel('凭据配置文件保存位置').fill('/tmp/workspace-ui-copy.env');await page.getByRole('button',{name:'保存私有配置',exact:true}).click();await expect(page.getByText('待重启切换：/tmp/workspace-ui-copy.env',{exact:true})).toBeVisible();
 await page.goto('/tasks?table=market_daily_raw');await expect(page).toHaveURL(/\/updates\?table=market_daily_raw$/);
 await expect(page.getByLabel('更新表 market_daily_raw',{exact:true})).toBeChecked();await page.getByRole('link',{name:'任务清单与回退',exact:true}).click();await expect(page).toHaveURL(/\/tasks$/);
 await page.goBack();await expect(page.getByRole('heading',{name:'1 · 指定请求参数',exact:true})).toBeVisible();
 await page.goto('/schedules');await expect(page).toHaveURL(/\/tasks\?removed=schedules$/);await expect(page.getByText('自动更新功能已移除。历史任务保留，请手动创建和启动更新任务。',{exact:true})).toBeVisible();
});

test('当前任务日志去重、限制保留量、重连直达最新日志与切换订阅',async({page,request})=>{
 page.on('dialog',d=>d.accept());
 const ids:string[]=[];
 for(let n=0;n<2;n++){
  const response=await request.post('/api/tasks',{data:{tables:['security_dim'],targets:['postgres'],assets:['stock'],start_date:'2026-10-13',end_date:'2026-10-13'}});
  expect(response.ok()).toBeTruthy();ids.push((await response.json()).id);
 }
 const opened:{id:string,latest:string|null,after:string|null}[]=[],closed:string[]=[];
 let firstSocket:any;
 const record=(id:number,endpoint:string)=>({id,created:'2026-10-28T00:00:00+00:00',event:'request_started',endpoint});
 await page.route('**/api/tasks/*/logs?*',async route=>{
  const url=new URL(route.request().url());expect(url.searchParams.get('latest')).toBe('true');expect(url.searchParams.get('limit')).toBe('500');
  await route.fulfill({json:[]});
 });
 await page.routeWebSocket(/\/api\/live\//,ws=>{
  const url=new URL(ws.url()),id=url.pathname.split('/').pop()!;
  opened.push({id,latest:url.searchParams.get('latest'),after:url.searchParams.get('after')});
  ws.onClose(()=>closed.push(id));
  if(id===ids[0]&&opened.filter(x=>x.id===id).length===1){
   firstSocket=ws;
   for(let n=1;n<=600;n++)ws.send(JSON.stringify({type:'log',record:record(n,'first_'+n)}));
   ws.send(JSON.stringify({type:'log',record:record(600,'first_600')}));
  }else if(id===ids[0]){
   // A long disconnection must jump over the backlog immediately.
   ws.send(JSON.stringify({type:'logs',records:Array.from({length:500},(_,n)=>record(9501+n,'latest_'+(9501+n)))}));
   ws.send(JSON.stringify({type:'log',record:record(10000,'latest_10000')}));
   ws.send(JSON.stringify({type:'log',record:record(10001,'reconnected_10001')}));
  }else ws.send(JSON.stringify({type:'log',record:record(900,'second_only')}));
 });
 await page.goto('/tasks?task='+ids[0]);
 const stream=page.getByLabel('实时任务日志');
 await expect(stream.locator('.log-line')).toHaveCount(500);
 await expect(stream).toContainText('first_600');await expect(stream).not.toContainText('first_100');
 firstSocket.close({code:1012,reason:'reconnect test'});
 await expect(stream).toContainText('reconnected_10001');
 expect(opened.find((x,i)=>i>0&&x.id===ids[0])?.latest).toBe('true');
 expect(opened.every(x=>x.after===null)).toBeTruthy();
 await expect(stream).not.toContainText('first_');
 await expect(stream.locator('.log-line')).toHaveCount(500);
 await page.getByRole('button',{name:ids[1].slice(0,12),exact:true}).click();
 await expect(page).toHaveURL(new RegExp('task='+ids[1]+'$'));
 await expect(stream).toContainText('second_only');await expect(stream.locator('.log-line')).toHaveCount(1);
 await expect(stream).not.toContainText('first_');await expect(stream).not.toContainText('reconnected_10001');
 await expect.poll(()=>closed.includes(ids[0])).toBeTruthy();
});

test('日志轮询直接读取最新窗口，唤醒重连且迟到快照不覆盖新日志',async({page,request})=>{
 page.on('dialog',d=>d.accept());
 const response=await request.post('/api/tasks',{data:{tables:['security_dim'],targets:['postgres'],start_date:'2026-10-13',end_date:'2026-10-13'}});
 expect(response.ok()).toBeTruthy();const id=(await response.json()).id;
 const record=(n:number)=>({id:n,created:'2026-10-29T00:00:00Z',event:'request_started',endpoint:'latest_'+n});
 let last=1000,opened=0,socket:any,hold=false,release:()=>void=()=>{};
 const requested:string[]=[];
 await page.route('**/api/tasks/'+id+'/logs?*',async route=>{
  requested.push(route.request().url());const n=last;
  if(hold){hold=false;await new Promise<void>(resolve=>{release=resolve;});}
  await route.fulfill({json:Array.from({length:500},(_,i)=>record(n-499+i))});
 });
 await page.routeWebSocket('**/api/live/'+id+'?*',ws=>{socket=ws;opened++;});
 await page.goto('/tasks?task='+id);const stream=page.getByLabel('实时任务日志');
 await expect(stream).toContainText('latest_1000');await expect(stream.locator('.log-line')).toHaveCount(500);
 // HTTP fallback while reconnecting also skips thousands of old records.
 last=10000;socket.close({code:1012});await expect(stream).toContainText('latest_10000');
 await expect(stream.getByText('latest_1000',{exact:true})).toHaveCount(0);
 // A sleeping page can retain an apparently open socket; waking replaces it.
 hold=true;last=11000;const before=opened;
 await page.evaluate(()=>document.dispatchEvent(new Event('visibilitychange')));
 await expect.poll(()=>opened).toBeGreaterThan(before);
 socket.send(JSON.stringify({type:'log',record:record(12000)}));
 await expect(stream).toContainText('latest_12000');release();
 await expect(stream).toContainText('latest_11000');await expect(stream).toContainText('latest_12000');
 await expect(stream.locator('.log-line')).toHaveCount(500);
 // Detect a half-open connection without waiting for a browser close event.
 await page.clock.install();last=13000;const previous=opened;
 await page.clock.fastForward(18000);
 await expect.poll(()=>opened).toBeGreaterThan(previous);
 await expect(stream).toContainText('latest_13000');
 expect(requested.every(url=>new URL(url).searchParams.get('latest')==='true')).toBeTruthy();
});

test('参考域历史显示发布与撤销摘要，回退使用原可信发布接口',async({page,request})=>{
 page.on('dialog',d=>d.accept());
 const response=await request.post('/api/reference/tasks',{data:{modules:['calendar'],targets:['postgres'],exchanges:['SSE'],start_date:'2026-10-13',end_date:'2026-10-13'}});
 expect(response.ok()).toBeTruthy();const task=await response.json();
 let rolledBack=false,rollbackCalls=0;
 await page.route('**/api/task-history/'+task.id,async route=>route.fulfill({json:{...task,target_matches:true,status:rolledBack?'rolled_back':'completed',tasks:[],result:{results:[{target:'postgres',module:'calendar',status:'published'}]},rollbacks:rolledBack?[{rollback_task_id:'reference-revoke',created_at:'2026-10-28T00:00:00Z',results:[{target:'postgres',module:'calendar',status:'revoked'}]}]:[]}}));
 await page.route('**/api/reference/tasks/'+task.id+'/rollback',async route=>{
  expect(route.request().postDataJSON().targets).toEqual(['postgres']);rollbackCalls++;rolledBack=true;await route.fulfill({json:{task:'reference-revoke'}});
 });
 await page.goto('/tasks?task='+task.id);
 const detail=page.getByRole('region',{name:'任务执行详情'});
 await expect(detail.getByRole('heading',{name:'参考域分库发布结果',exact:true})).toBeVisible();
 await expect(detail.getByRole('link',{name:'打开业务完整性任务',exact:true})).toHaveAttribute('href','/integrity?task='+task.id);
 await detail.getByRole('button',{name:'回退本任务',exact:true}).click();await detail.getByRole('button',{name:'确认回退所选数据库',exact:true}).click();
 await expect.poll(()=>rollbackCalls).toBe(1);await expect(detail).toContainText('已撤销');await expect(detail.getByRole('button',{name:'回退本任务',exact:true})).toBeDisabled();
});
