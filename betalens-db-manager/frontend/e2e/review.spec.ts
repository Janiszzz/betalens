import {test,expect,type APIRequestContext} from '@playwright/test';

async function rejectedTask(request:APIRequestContext,name:string){
 const definition={name,zh_name:'人工审计验收表',description:'隔离库拒收测试',grain:'证券日期一行',keys:['code','business_date'],time_field:'business_date',temporal_rule:'calendar_day',fields:[{name:'code',type:'text'},{name:'business_date',type:'date'},{name:'price',type:'numeric',precision:28,scale:8}]};
 let r=await request.post('/api/tables',{data:{definition}});expect(r.ok(),await r.text()).toBeTruthy();
 const source={source_id:'review_browser_http',provider:'http',base_url:'http://127.0.0.1:18082',endpoints:{prices:{path:'/review-prices',strategy:'range',keys:['ticker','day'],date_field:'day',date_format:'%Y/%m/%d',fields:[{name:'ticker',type:'text'},{name:'day',type:'date'},{name:'price_milli',type:'numeric'}]}}};
 r=await request.post('/api/sources',{data:source});expect(r.ok(),await r.text()).toBeTruthy();
 const mapping={format_version:1,table:name,bindings:[{source_id:'review_browser_http',endpoint:'prices',fields:[{source_field:'ticker',column:'code'},{source_field:'day',column:'business_date',date_format:'%Y/%m/%d'},{source_field:'price_milli',column:'price',multiplier:'0.001'}]}]};
 r=await request.post('/api/tables/'+name+'/mapping',{data:{mapping}});expect(r.ok(),await r.text()).toBeTruthy();
 const schema=await(await request.post('/api/tables/'+name+'/schema-plan',{data:{}})).json();
 r=await request.post('/api/tables/schema-apply',{data:{token:schema.token}});expect(r.ok(),await r.text()).toBeTruthy();
 r=await request.post('/api/tasks',{data:{pipeline_version:2,tables:[name],source_id:'review_browser_http',start_date:'2026-10-13',end_date:'2026-10-13',min_free_bytes:0}});expect(r.ok(),await r.text()).toBeTruthy();
 const job=await r.json();r=await request.post('/api/tasks/'+job.id+'/start',{data:{}});expect(r.ok(),await r.text()).toBeTruthy();return job;
}

test('拒收任务自动进入审计页，放行合格缓存，原任务回退包含人工写入',async({page,request},info)=>{
 page.on('dialog',d=>d.accept());const errors:string[]=[];page.on('pageerror',e=>errors.push(e.message));
 const job=await rejectedTask(request,'review_browser_release');await page.goto('/tasks?task='+job.id);
 await expect(page).toHaveURL(new RegExp('/tasks/review\\?task='+job.id+'$'),{timeout:60000});
 const panel=page.getByRole('region',{name:'拒收批次审计详情'});await expect(panel).toContainText('共同键');
 await panel.getByLabel('人工审计说明').fill('已核对合格记录，接受该批次的合格缓存');
 await panel.getByRole('button',{name:'确认放行 · 预览可入库记录',exact:true}).click();
 const preview=page.getByRole('region',{name:'人工审计写入预览'});await expect(preview).toContainText('PostgreSQL');await expect(preview).toContainText('ClickHouse');
 await preview.getByLabel('已核对本次范围、目标库及剩余拒收记录，确认执行所选处理').check();
 await preview.getByRole('button',{name:'确认执行人工审计决定',exact:true}).click();
 await expect.poll(async()=>(await(await request.get('/api/tasks/'+job.id+'/reviews')).json()).counts.released,{timeout:60000}).toBe(1);
 await page.screenshot({path:info.outputPath('manual-release.png'),fullPage:true});
 await page.getByRole('button',{name:'返回任务详情',exact:true}).click();await expect(page).toHaveURL(new RegExp('/tasks\\?task='+job.id+'$'));
 const progress=page.getByLabel('固定任务进度');await expect(progress.getByRole('progressbar',{name:'请求处理进度',exact:true})).toHaveAttribute('max','1');
 await expect(progress.getByRole('progressbar',{name:'转换批次进度',exact:true})).toHaveAttribute('max','1');
 const detail=page.getByRole('region',{name:'任务执行详情'});await detail.getByRole('button',{name:'回退本任务',exact:true}).click();await detail.getByRole('button',{name:'确认回退所选数据库',exact:true}).click();
 await expect.poll(async()=>(await(await request.get('/api/tasks/'+job.id)).json()).status,{timeout:60000}).toBe('rolled_back');expect(errors).toEqual([]);
});

test('拒绝批次并导出审计元信息，拒绝操作不产生写入回执',async({page,request})=>{
 page.on('dialog',d=>d.accept());const job=await rejectedTask(request,'review_browser_reject');
 await page.goto('/tasks?task='+job.id);await expect(page).toHaveURL(new RegExp('/tasks/review\\?task='+job.id+'$'),{timeout:60000});
 const panel=page.getByRole('region',{name:'拒收批次审计详情'});await panel.getByLabel('人工审计说明').fill('拒绝缺少证券代码的批次，保留来源证据');
 const download=page.waitForEvent('download');await panel.getByRole('button',{name:'拒绝并导出元信息',exact:true}).click();
 expect((await download).suggestedFilename()).toMatch(/^rejected-.*\.json$/);
 await expect.poll(async()=>(await(await request.get('/api/tasks/'+job.id+'/reviews')).json()).counts.rejected).toBe(1);
 const audit=await(await request.get('/api/tasks/'+job.id+'/pipeline')).json();expect(audit.pipeline.receipts).toHaveLength(0);
});

test('JSON 手工修补后预览导入，刷新深链接保留决定且可回退',async({page,request},info)=>{
 page.on('dialog',d=>d.accept());const name='review_browser_repair',job=await rejectedTask(request,name);
 await page.goto('/tasks?task='+job.id);await expect(page).toHaveURL(new RegExp('/tasks/review\\?task='+job.id+'$'),{timeout:60000});
 const panel=page.getByRole('region',{name:'拒收批次审计详情'});await panel.getByRole('button',{name:'手工修补后导入',exact:true}).click();
 await panel.getByLabel('人工审计说明').fill('已查明缺失证券代码，补充 b 并确认价格');
 await panel.getByLabel('导入修补 JSON 文件').setInputFiles({name:'repair.json',mimeType:'application/json',buffer:Buffer.from(JSON.stringify({rows:{[name]:[{code:'b',business_date:'2026-10-13',price:'9.12345678'}]}}))});
 await expect(panel.getByLabel('修补记录 JSON')).toHaveValue(/9\.12345678/);
 await panel.getByRole('button',{name:'预览修补后导入',exact:true}).click();const preview=page.getByRole('region',{name:'人工审计写入预览'});await expect(preview).toBeVisible();
 await preview.getByLabel('已核对本次范围、目标库及剩余拒收记录，确认执行所选处理').check();await preview.getByRole('button',{name:'确认执行人工审计决定',exact:true}).click();
 await expect.poll(async()=>(await(await request.get('/api/tasks/'+job.id+'/reviews')).json()).counts.repaired,{timeout:60000}).toBe(1);
 await page.reload();await expect(page.getByRole('heading',{name:'已记录的人工决定',exact:true})).toBeVisible();
 await page.screenshot({path:info.outputPath('manual-repair.png'),fullPage:true});
 const audit=await(await request.get('/api/tasks/'+job.id+'/audit')).json();expect(audit.changes).toHaveLength(4);
 const rollback=await request.post('/api/tasks/'+job.id+'/rollback',{data:{targets:['postgres','clickhouse']}});expect(rollback.ok(),await rollback.text()).toBeTruthy();
 await expect.poll(async()=>(await(await request.get('/api/tasks/'+job.id)).json()).status,{timeout:60000}).toBe('rolled_back');
});

test('来源覆盖异常需显式接受，放行后原拒收证据仍保留',async({page,request})=>{
 page.on('dialog',d=>d.accept());const response=await request.get('http://127.0.0.1:18082/seed-review-coverage');expect(response.ok(),await response.text()).toBeTruthy();const job=await response.json();
 await page.goto('/tasks?task='+job.id);await expect(page).toHaveURL(new RegExp('/tasks/review\\?task='+job.id+'$'),{timeout:60000});
 const panel=page.getByRole('region',{name:'拒收批次审计详情'});await panel.getByLabel('人工审计说明').fill('已核对原始归档，接受当前可得范围并保留完整性异常');
 await expect(panel.getByRole('button',{name:'确认放行 · 预览可入库记录',exact:true})).toBeDisabled();
 await panel.getByLabel('人工接受现有归档的覆盖范围').check();await panel.getByRole('button',{name:'确认放行 · 预览可入库记录',exact:true}).click();
 const preview=page.getByRole('region',{name:'人工审计写入预览'});await expect(preview).toContainText('本次人工接受现有归档覆盖',{timeout:30000});
 await preview.getByLabel('已核对本次范围、目标库及剩余拒收记录，确认执行所选处理').check();await preview.getByRole('button',{name:'确认执行人工审计决定',exact:true}).click();
 await expect.poll(async()=>(await(await request.get('/api/tasks/'+job.id+'/reviews')).json()).counts.released,{timeout:60000}).toBe(1);
 const detail=await(await request.get('/api/tasks/'+job.id+'/reviews/history_batch_0')).json();expect(detail.decision.result.coverage_accepted).toBe(true);expect(detail.decision.result.original_rejections_retained).toBe(true);
 const audit=await(await request.get('/api/tasks/'+job.id+'/pipeline')).json();expect(audit.pipeline.units.find((u:any)=>u.id==='history_batch_0').status).toBe('failed');
 const rolled=await request.post('/api/tasks/'+job.id+'/rollback',{data:{targets:['postgres','clickhouse']}});expect(rolled.ok(),await rolled.text()).toBeTruthy();
 await expect.poll(async()=>(await(await request.get('/api/tasks/'+job.id)).json()).status,{timeout:60000}).toBe('rolled_back');
});
