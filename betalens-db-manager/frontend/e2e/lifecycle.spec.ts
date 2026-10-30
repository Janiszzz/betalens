import {test,expect,type Page,type APIRequestContext} from '@playwright/test';
async function selectMapping(page:Page,field:string,column:string){const select=page.getByLabel('映射 '+field,{exact:true});await select.focus();await select.selectOption(column);}
async function waitForTaskIdle(page:Page,request:APIRequestContext,status="completed"){const heading=await page.getByRole('heading',{name:/^任务 [a-f0-9]{32}$/}).innerText();const id=heading.split(' ')[1];await expect.poll(async()=>{const job=await (await request.get('/api/tasks/'+id)).json();return job.status===status&&!job.tasks.some((t:any)=>['queued','running'].includes(t.status));}).toBeTruthy();}
test('更新任务可设置每分钟请求次数并固定获取策略',async({page,request},info)=>{
 page.on('dialog',d=>d.accept());
 await page.goto('/tasks?table=security_dim');
 const rpm=page.getByLabel('Tushare 每分钟请求次数',{exact:true});
 await expect(rpm).toBeVisible();await rpm.fill('200');
 await page.getByLabel('基金',{exact:true}).uncheck();await page.getByLabel('指数',{exact:true}).uncheck();
 await page.getByRole('button',{name:'匹配并预览 API 请求',exact:true}).click();
 await expect(page.getByRole('heading',{name:'接口限频与完整获取策略',exact:true})).toBeVisible();
 await page.getByRole('button',{name:'创建任务并查看详情',exact:true}).click();
 const heading=page.getByRole('heading',{name:/^任务 [a-f0-9]{32}$/});await expect(heading).toBeVisible();
 const id=(await heading.innerText()).split(' ')[1];
 const task=await (await request.get('/api/tasks/'+id)).json();
 expect(task.plan.request.requests_per_minute).toBe(200);
 expect(task.plan.retrieval_policies['tushare:stock_basic'].effective_rpm).toBe(50);
 expect(task.slices).toHaveLength(0);expect(task.status).toBe('planned');
 await page.screenshot({path:info.outputPath('request-rate-preview.png'),fullPage:true});
});
test('手工建表 → HTTP 映射 → 缺口补数 → 双库确认写入 → 回退',async({page,request},info)=>{
 page.on('dialog',d=>d.accept());
 const errors:string[]=[];page.on('pageerror',e=>errors.push(e.message));
 await page.goto('/');await expect(page.getByRole('heading',{name:'本地表与配置',exact:true})).toBeVisible();
 await page.getByText('高级管理 · 手工定义、结构迁移与未登记表',{exact:true}).click();
 await page.getByRole('button',{name:'手工建表',exact:true}).click();
 for(const [label,value] of [['英文表名','browser_prices'],['中文名称','浏览器验收行情'],['说明','完整用户流程验收'],['一行代表什么','证券和业务日期一行']])await page.getByLabel(label,{exact:true}).fill(value);
 await page.getByLabel('字段名0',{exact:true}).fill('code');await page.getByLabel('主键0',{exact:true}).check();
 await page.getByRole('button',{name:'增加业务字段',exact:true}).click();await page.getByLabel('字段名1',{exact:true}).fill('business_date');await page.getByLabel('字段类型1',{exact:true}).selectOption('date');await page.getByLabel('主键1',{exact:true}).check();
 await page.getByRole('button',{name:'增加业务字段',exact:true}).click();await page.getByLabel('字段名2',{exact:true}).fill('price');await page.getByLabel('字段类型2',{exact:true}).selectOption('numeric');await page.getByLabel('精度2',{exact:true}).fill('28');await page.getByLabel('小数位2',{exact:true}).fill('8');
 await page.getByLabel('业务日期字段',{exact:true}).selectOption('business_date');await page.getByLabel('时间规则',{exact:true}).selectOption('calendar_day');
 await page.getByRole('button',{name:'保存表定义',exact:true}).click();await expect(page.getByRole('button',{name:'预览两库结构操作'})).toBeEnabled();
 await page.getByRole('button',{name:'预览两库结构操作'}).click();await page.getByRole('button',{name:'确认执行结构计划'}).click();await expect(page.getByText('completed',{exact:true}).first()).toBeVisible({timeout:30000});
 await page.getByRole('link',{name:'订阅源',exact:true}).click();
 for(const [label,value] of [['来源实例标识','browser_vendor'],['来源中文名','浏览器测试 API'],['API 基础地址','http://127.0.0.1:18082'],['查询名称','prices'],['查询路径','prices'],['源业务键（逗号分隔）','ticker,day'],['源业务日期字段','day']])await page.getByLabel(label,{exact:true}).fill(value);
 for(const [i,name,kind] of [[0,'ticker','text'],[1,'day','date'],[2,'price_milli','numeric']] as const){if(i)await page.getByRole('button',{name:'增加源字段',exact:true}).click();await page.getByLabel('源字段名'+i,{exact:true}).fill(name);await page.getByLabel('源字段类型'+i,{exact:true}).selectOption(kind);}
 await page.getByRole('button',{name:'保存查询源',exact:true}).click();await expect(page.getByText('来源已登记；可在字段映射页绑定')).toBeVisible();
 await page.getByRole('link',{name:'字段映射',exact:true}).click();await page.getByText('选择 API 查询源（已选 0）',{exact:true}).click();await page.getByLabel('查询源 browser_vendor · prices',{exact:true}).check();await page.getByText('选择 API 查询源（已选 1）',{exact:true}).click();
 await page.getByTestId('source-ticker').dragTo(page.getByTestId('target-code'));await expect(page.getByLabel('映射 ticker',{exact:true})).toHaveValue('code');
 await selectMapping(page,'day','business_date');await selectMapping(page,'price_milli','price');await page.getByLabel('乘数 price_milli',{exact:true}).fill('0.001');
 await page.getByTestId('source-day').click();await page.getByLabel('日期格式 day',{exact:true}).fill('%Y/%m/%d');await page.getByRole('button',{name:'保存映射规则',exact:true}).click();
 await expect(page.getByText('映射已写入本地 YAML，新任务使用此版本')).toBeVisible();
 const seeded=await (await request.get('http://127.0.0.1:18082/seed')).json();expect(seeded).toEqual({seeded:true});
 await page.goto('/updates?table=browser_prices');
 await page.getByLabel('任务起始日期').fill('2026-10-13');await page.getByLabel('任务结束日期').fill('2026-10-13');await page.getByLabel('异步后台采集，轻量检查后自动入库',{exact:true}).uncheck();await page.getByRole('button',{name:'匹配并预览 API 请求'}).click();await page.getByRole('button',{name:'创建任务并查看详情',exact:true}).click();
 await page.getByRole('button',{name:'开始拉取',exact:true}).click();
 // A separately supervised worker is started by the test fixture's server.
 await page.getByRole('tab',{name:'缓存与写入审计',exact:true}).click();
 await expect(page.getByRole('button',{name:'确认增量入库'})).toBeVisible({timeout:30000});
 expect(await (await request.get('http://127.0.0.1:18082/assert-empty')).json()).toEqual({pg:0,ch:0});
 await page.getByText('原始响应缓存与下载',{exact:true}).click();
 await page.getByRole('button',{name:'查看审计缓存'}).click();await expect(page.getByText('12345.678901',{exact:true}).first()).toBeVisible();
 await page.screenshot({path:info.outputPath('audit-before-confirm.png'),fullPage:true});
 await page.getByRole('button',{name:'确认增量入库'}).click();await expect.poll(async()=>await (await request.get('http://127.0.0.1:18082/assert-empty')).json()).toEqual({pg:1,ch:1});
 await page.getByRole('button',{name:'查看写入结果'}).click();await page.screenshot({path:info.outputPath('write-results.png'),fullPage:true});
 await waitForTaskIdle(page,request);await page.getByRole('button',{name:'回退本任务',exact:true}).click();await page.getByRole('button',{name:'确认回退所选数据库',exact:true}).click();await expect.poll(async()=>await (await request.get('http://127.0.0.1:18082/assert-empty')).json()).toEqual({pg:0,ch:0});
 // A second provider has different names, units, date format and JSON shape.
 const registered=await request.post('/api/sources',{data:{source_id:'alternate_vendor',provider:'http',label:'第二订阅源',base_url:'http://127.0.0.1:18082',endpoints:{alternate:{path:'alternate',method:'GET',strategy:'range',date_field:'observed',keys:['symbol','observed'],fields:[{name:'symbol',type:'text'},{name:'observed',type:'date'},{name:'close_yuan',type:'numeric'}]}}}});expect(registered.ok()).toBeTruthy();
 const doc=await (await request.get('/api/tables/browser_prices/mapping')).json();doc.mapping.bindings.push({source_id:'alternate_vendor',endpoint:'alternate',priority:10,join_keys:[{source_field:'symbol',column:'code'},{source_field:'observed',column:'business_date'}],fields:[{source_field:'symbol',column:'code'},{source_field:'observed',column:'business_date',date_format:'%Y/%m/%d'},{source_field:'close_yuan',column:'price',multiplier:'1'}]});
 expect((await request.post('/api/tables/browser_prices/mapping',{data:{mapping:doc.mapping,file_hash:doc.file_hash}})).ok()).toBeTruthy();
 await page.goto('/updates?table=browser_prices');
 await page.getByLabel('任务起始日期').fill('2026-10-13');await page.getByLabel('任务结束日期').fill('2026-10-13');await page.getByLabel('更新来源',{exact:true}).selectOption('alternate_vendor');
 await page.getByLabel('异步后台采集，轻量检查后自动入库',{exact:true}).uncheck();await page.getByRole('button',{name:'匹配并预览 API 请求'}).click();await page.getByRole('button',{name:'创建任务并查看详情',exact:true}).click();await page.getByRole('button',{name:'开始拉取',exact:true}).click();
 await page.getByRole('tab',{name:'缓存与写入审计',exact:true}).click();
 await expect(page.getByRole('button',{name:'确认增量入库'})).toBeVisible({timeout:30000});await page.getByRole('button',{name:'确认增量入库'}).click();
 await expect.poll(async()=>await (await request.get('http://127.0.0.1:18082/assert-empty')).json()).toEqual({pg:1,ch:1});
 const secondRows=await (await request.get('/api/query/browser_prices?target=clickhouse&start=2026-10-13&end=2026-10-13')).json();expect(secondRows.rows[0].source_id).toBe('alternate_vendor');expect(secondRows.rows[0].price).toBe('12.3456789');
 expect(errors).toEqual([]);
 await page.close({runBeforeUnload:true});await expect.poll(()=>page.isClosed()).toBeTruthy();
});

test('关闭警告、多个页面、最后页面停止任务、重新打开不自动恢复',async({page,context,request})=>{
 const source={source_id:'slow_vendor',provider:'http',label:'慢速接口',base_url:'http://127.0.0.1:18082',endpoints:{slow:{path:'slow',method:'GET',strategy:'range',date_field:'day',keys:['ticker','day'],fields:[{name:'ticker',type:'text'},{name:'day',type:'date'},{name:'price_milli',type:'numeric'}]}}};
 expect((await request.post('/api/sources',{data:source})).ok()).toBeTruthy();
 const doc=await (await request.get('/api/tables/browser_prices/mapping')).json();doc.mapping.bindings.push({source_id:'slow_vendor',endpoint:'slow',priority:20,join_keys:[{source_field:'ticker',column:'code'},{source_field:'day',column:'business_date'}],fields:[{source_field:'ticker',column:'code'},{source_field:'day',column:'business_date',date_format:'%Y/%m/%d'},{source_field:'price_milli',column:'price',multiplier:'0.001'}]});
 expect((await request.post('/api/tables/browser_prices/mapping',{data:{mapping:doc.mapping,file_hash:doc.file_hash}})).ok()).toBeTruthy();
 const task=await (await request.post('/api/tasks',{data:{pipeline_version:1,tables:['browser_prices'],source_id:'slow_vendor',start_date:'2026-10-15',end_date:'2026-10-15'}})).json();
 await page.goto('/tasks?task='+task.id);
 const second=await context.newPage();await second.goto('/');await second.getByRole('link',{name:'更新与审计',exact:true}).click();
 await page.getByRole('button',{name:'开始拉取',exact:true}).click();await expect.poll(async()=>{const j=await (await request.get('/api/tasks/'+task.id)).json();return j.status;}).toBe('downloading');
 const dismissed=page.waitForEvent('dialog');await page.close({runBeforeUnload:true});await (await dismissed).dismiss();expect(page.isClosed()).toBeFalsy();
 const accepted=page.waitForEvent('dialog');await page.close({runBeforeUnload:true});await (await accepted).accept();await expect.poll(()=>page.isClosed()).toBeTruthy();
 await second.waitForTimeout(17000);const stillRunning=await (await request.get('/api/tasks/'+task.id)).json();expect(stillRunning.cancel).toBe(0);expect(stillRunning.status).toBe('downloading');
 const lastDialog=second.waitForEvent('dialog');await second.close({runBeforeUnload:true});await (await lastDialog).accept();
 await expect.poll(async()=>(await (await request.get('/api/tasks/'+task.id)).json()).cancel,{timeout:55000}).toBe(1);
 await expect.poll(async()=>(await (await request.get('/api/tasks/'+task.id)).json()).status,{timeout:45000}).toMatch(/cancelled|interrupted/);
 const reopened=await context.newPage();await reopened.goto('/tasks');await expect(reopened.getByRole('button',{name:new RegExp(task.id.slice(0,12))})).toBeVisible();
 const final=await (await request.get('/api/tasks/'+task.id)).json();expect(final.tasks.some((t:any)=>['queued','running'].includes(t.status))).toBeFalsy();
});

test('查询源多选去重与移除 → 可视连线 → 联合键多源拼接 → 双库写入和回退',async({page,request},info)=>{
 page.on('dialog',d=>d.accept());
 const definition={name:'joined_quotes',zh_name:'多源行情',description:'价格和成交量按代码与日期拼接',grain:'代码和日期一行',domain:'custom',keys:['code','business_date'],time_field:'business_date',temporal_rule:'calendar_day',fields:[{name:'code',type:'text',nullable:false},{name:'business_date',type:'date',nullable:false},{name:'price',type:'numeric',precision:28,scale:8,nullable:false},{name:'volume',type:'numeric',nullable:false}]};
 expect((await request.post('/api/tables',{data:{definition}})).ok()).toBeTruthy();
 const plan=await (await request.post('/api/tables/joined_quotes/schema-plan',{data:{targets:['postgres','clickhouse']}})).json();
 expect((await (await request.post('/api/tables/schema-apply',{data:{token:plan.token}})).json()).status).toBe('completed');
 const source={source_id:'join_vendor',provider:'http',label:'多源拼接测试',base_url:'http://127.0.0.1:18082',endpoints:{prices:{path:'prices',strategy:'range',date_field:'day',keys:['ticker','day'],fields:[{name:'ticker',type:'text'},{name:'day',type:'date'},{name:'price_milli',type:'numeric'}]},volumes:{path:'volumes',strategy:'range',date_field:'observed',keys:['instrument','observed'],fields:[{name:'instrument',type:'text'},{name:'quantity_lots',type:'numeric'},{name:'observed',type:'date'},{name:'note',type:'text'}]}}};
 expect((await request.post('/api/sources',{data:source})).ok()).toBeTruthy();
 await page.route('**/api/tables/market_daily_raw/mapping',async route=>{await new Promise(resolve=>setTimeout(resolve,1200));await route.continue();});
 await page.goto('/');await page.getByRole('link',{name:'字段映射',exact:true}).click();await page.getByLabel('选择本地表',{exact:true}).selectOption('joined_quotes');
 await page.getByText('选择 API 查询源（已选 0）',{exact:true}).click();
 await page.getByLabel('查询源 join_vendor · prices',{exact:true}).check();await page.getByLabel('查询源 join_vendor · volumes',{exact:true}).check();
 await expect(page.getByLabel('查询源 join_vendor · prices',{exact:true})).toBeChecked();
 await page.getByLabel('查询源 join_vendor · volumes',{exact:true}).uncheck();await expect(page.getByRole('button',{name:'移除查询源 join_vendor:volumes',exact:true})).toHaveCount(0);
 await page.getByLabel('查询源 join_vendor · volumes',{exact:true}).check();await expect(page.getByRole('button',{name:'移除查询源 join_vendor:volumes',exact:true})).toHaveCount(1);
 await page.getByText('选择 API 查询源（已选 2）',{exact:true}).click();
 await page.getByTestId('source-ticker').dragTo(page.getByTestId('target-code'));
 await selectMapping(page,'day','business_date');await page.getByLabel('日期格式 day',{exact:true}).fill('%Y/%m/%d');
 await selectMapping(page,'price_milli','price');await page.getByLabel('乘数 price_milli',{exact:true}).fill('0.001');
 await selectMapping(page,'instrument','code');
 await selectMapping(page,'observed','business_date');await page.getByLabel('日期格式 observed',{exact:true}).fill('%Y%m%d');
 await selectMapping(page,'quantity_lots','volume');await page.getByLabel('乘数 quantity_lots',{exact:true}).fill('100');
 await selectMapping(page,'note','__ignore__');await page.getByLabel('忽略原因',{exact:true}).fill('备注不属于本表业务字段');
 await expect(page.getByRole('heading',{name:'完整性检查 · 无阻断项',exact:true})).toBeVisible();
 await expect(page.locator('[data-testid="mapping-edge"][data-source="ticker"][data-target="code"]')).toHaveCount(1);
 await page.getByRole('button',{name:'保存映射规则',exact:true}).click();await expect(page.getByText('映射已写入本地 YAML，新任务使用此版本')).toBeVisible();
 const mapping=await (await request.get('/api/tables/joined_quotes/mapping')).json();expect(mapping.mapping.bindings).toHaveLength(2);
 expect(mapping.mapping.bindings.every((b:any)=>b.join_keys.length===2)).toBeTruthy();expect(mapping.mapping.format_version).toBe(2);
 await page.screenshot({path:info.outputPath('mapping-graph.png'),fullPage:true});
 await page.reload();await page.getByRole('link',{name:'字段映射',exact:true}).click();await page.getByLabel('选择本地表',{exact:true}).selectOption('joined_quotes');
 await page.getByText('选择 API 查询源（已选 2）',{exact:true}).click();await expect(page.getByLabel('查询源 join_vendor · prices',{exact:true})).toBeChecked();await expect(page.getByLabel('查询源 join_vendor · volumes',{exact:true})).toBeChecked();await page.getByText('选择 API 查询源（已选 2）',{exact:true}).click();
 await page.getByRole('link',{name:'更新与审计',exact:true}).click();await page.getByLabel('任务起始日期').fill('2026-10-13');await page.getByLabel('任务结束日期').fill('2026-10-13');
 await page.getByLabel('异步后台采集，轻量检查后自动入库',{exact:true}).uncheck();await page.getByRole('button',{name:'匹配并预览 API 请求'}).click();await page.getByRole('button',{name:'创建任务并查看详情',exact:true}).click();await page.getByRole('button',{name:'开始拉取',exact:true}).click();
 await page.getByRole('tab',{name:'缓存与写入审计',exact:true}).click();
 await expect(page.getByRole('button',{name:'确认增量入库'})).toBeVisible({timeout:30000});
 for(const target of ['postgres','clickhouse'])expect((await (await request.get('/api/query/joined_quotes?target='+target)).json()).rows).toHaveLength(0);
 await page.getByRole('button',{name:'确认增量入库'}).click();
 for(const target of ['postgres','clickhouse'])await expect.poll(async()=>{const r=await (await request.get('/api/query/joined_quotes?target='+target)).json();return r.rows.map((v:any)=>[v.code,v.business_date,v.price,v.volume]);}).toEqual([['a','2026-10-13','12.3456789','200']]);
 await waitForTaskIdle(page,request);await page.getByRole('button',{name:'回退本任务',exact:true}).click();await page.getByRole('button',{name:'确认回退所选数据库',exact:true}).click();
 for(const target of ['postgres','clickhouse'])await expect.poll(async()=>(await (await request.get('/api/query/joined_quotes?target='+target)).json()).rows).toHaveLength(0);
 await page.close({runBeforeUnload:true});await expect.poll(()=>page.isClosed()).toBeTruthy();
});

test('各页面独立路径、直接访问、刷新和历史导航；已绘制连线后切换全部本地表不白屏',async({page,request},info)=>{
 page.on('dialog',d=>d.accept());const errors:string[]=[];page.on('pageerror',e=>errors.push(e.message));
 const routes=[['本地表与配置','/tables'],['更新与审计','/updates'],['任务清单与回退','/tasks'],['业务完整性','/integrity'],['字段映射','/mappings'],['订阅源','/sources']];
 for(const [name,path] of routes){
   await page.goto(path);await expect(page.getByRole('heading',{name,exact:true})).toBeVisible();
   await expect(page.locator('aside a[aria-current="page"]')).toHaveText(name);
   await page.reload();await expect(page.getByRole('heading',{name,exact:true})).toBeVisible();
 }
 await page.getByRole('link',{name:'字段映射',exact:true}).click();await expect(page).toHaveURL(/\/mappings(?:\?|$)/);
 await page.getByRole('link',{name:'本地表与配置',exact:true}).click();await expect(page).toHaveURL(/\/tables$/);
 await page.goBack();await expect(page.getByRole('heading',{name:'字段映射',exact:true})).toBeVisible();
 await page.goForward();await expect(page.getByRole('heading',{name:'本地表与配置',exact:true})).toBeVisible();
 await page.goto('/mappings?table=market_daily_raw');
 await expect.poll(()=>page.getByTestId('mapping-edge').count()).toBeGreaterThan(0);
 // The original crash used the old edge index while the next response was in flight.
 await page.route('**/api/tables/dividend_event/mapping',async route=>{await new Promise(r=>setTimeout(r,400));await route.continue();});
 await page.getByLabel('选择本地表',{exact:true}).selectOption('dividend_event');
 await expect(page.getByRole('button',{name:'选择 API 查询源（已选 1）',exact:true})).toBeEnabled();
 await expect(page).toHaveURL(/\/mappings\?table=dividend_event$/);
 await expect.poll(()=>page.getByTestId('mapping-edge').count()).toBeGreaterThan(0);
 await page.reload();await expect(page.getByLabel('选择本地表',{exact:true})).toHaveValue('dividend_event');
 await expect(page.getByRole('button',{name:'选择 API 查询源（已选 1）',exact:true})).toBeEnabled();
 const definition={name:'empty_mapping',zh_name:'空映射验收',description:'未绑定查询源',grain:'代码一行',keys:['code'],fields:[{name:'code',type:'text'}]};
 expect((await request.post('/api/tables',{data:{definition}})).ok()).toBeTruthy();
 // Refresh the catalog, then exercise populated → empty → populated graphs.
 await page.reload();const tables=await (await request.get('/api/catalog')).json();
 for(const table of [...tables.filter((t:any)=>t.name!=='market_daily_raw'),{name:'market_daily_raw'}]){
   await page.getByLabel('选择本地表',{exact:true}).selectOption(table.name);
   await expect(page.locator('.query-picker>button')).toBeEnabled();
   await expect(page.locator('.target-node .node-heading strong')).toHaveText(table.name);
   if(table.name==='fundamental_periodic'){const select=page.locator('.source-node select').first();await select.focus();await expect.poll(()=>select.locator('option').count()).toBeGreaterThan(100);await select.evaluate((el:HTMLSelectElement)=>el.blur());await expect.poll(()=>page.locator('.source-node select option').count()).toBeLessThan(15000);}
   await expect(page.getByRole('heading',{name:'字段映射',exact:true})).toBeVisible();
 }
 await expect.poll(()=>page.getByTestId('mapping-edge').count()).toBeGreaterThan(0);
 // Removing a source also invalidates old edge indices. No mapping is saved here.
 while(await page.locator('.binding-chip').count()){
   await page.locator('.binding-chip').last().getByRole('button',{name:/移除查询源/}).click();
   await expect(page.getByRole('heading',{name:'字段映射',exact:true})).toBeVisible();
 }
 // Derived declarations remain visible even after all API bindings are removed.
 await expect(page.locator('[data-testid="mapping-edge"][data-kind="direct"]')).toHaveCount(0);
 await page.getByLabel('选择本地表',{exact:true}).selectOption('dividend_event');await expect.poll(()=>page.getByTestId('mapping-edge').count()).toBeGreaterThan(0);
 await page.screenshot({path:info.outputPath('mapping-table-switch.png'),fullPage:true});
 expect(errors).toEqual([]);
 await page.close({runBeforeUnload:true});await expect.poll(()=>page.isClosed()).toBeTruthy();
});

test('数据写入后审计同步失败 → 网页补同步 → 不重复业务写入',async({page,request},info)=>{
 page.on('dialog',d=>d.accept());
 const created=await request.post('/api/tasks',{data:{pipeline_version:1,tables:['browser_prices'],source_id:'browser_vendor',targets:['postgres'],start_date:'2026-10-16',end_date:'2026-10-16'}});
 expect(created.ok()).toBeTruthy();const task=await created.json();
 await request.get('http://127.0.0.1:18082/audit-sync-fault?enabled=true');
 try{
   await page.goto('/tasks?task='+task.id);
   await page.getByRole('button',{name:'开始拉取',exact:true}).click();
   await page.getByRole('tab',{name:'缓存与写入审计',exact:true}).click();
   await expect(page.getByRole('button',{name:'确认增量入库',exact:true})).toBeVisible({timeout:30000});
   await page.getByRole('button',{name:'确认增量入库',exact:true}).click();
   await expect.poll(async()=>(await (await request.get('/api/tasks/'+task.id)).json()).status,{timeout:30000}).toBe('audit_pending');
   await waitForTaskIdle(page,request,'audit_pending');
   const url='/api/query/browser_prices?target=postgres&start=2026-10-16&end=2026-10-16';
   const before=(await (await request.get(url)).json()).rows;expect(before).toHaveLength(1);
   await page.getByRole('button',{name:'查看写入结果',exact:true}).click();
   await page.screenshot({path:info.outputPath('audit-pending.png'),fullPage:true});
   await request.get('http://127.0.0.1:18082/audit-sync-fault?enabled=false');
   await page.getByRole('button',{name:'重试同步审计',exact:true}).click();
   await expect.poll(async()=>(await (await request.get('/api/tasks/'+task.id)).json()).status).toBe('completed');
   expect((await (await request.get(url)).json()).rows).toEqual(before);
   const result=await (await request.get('/api/tasks/'+task.id)).json();
   expect(result.audit_sync_status).toBe('synced');
   await info.attach('audit-sync-result',{body:JSON.stringify({task_id:task.id,status:result.status,audit_sync_status:result.audit_sync_status,rows:before.length,business_rows_unchanged:true}),contentType:'application/json'});
   await page.screenshot({path:info.outputPath('audit-synced.png'),fullPage:true});
 }finally{await request.get('http://127.0.0.1:18082/audit-sync-fault?enabled=false');}
 await page.close({runBeforeUnload:true});await expect.poll(()=>page.isClosed()).toBeTruthy();
});
