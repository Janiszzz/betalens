import {test, expect, APIRequestContext} from '@playwright/test';

const mockUrl = 'http://127.0.0.1:18086';
async function fixtureState(request: APIRequestContext) {
  const response = await request.get(mockUrl + '/fixture/state');
  expect(response.ok()).toBeTruthy();
  return response.json();
}

test('真实后端和双库：发现接续缺日→HTTP分页→缓存审计→确认发布→读取→追加撤销', async ({page, request}, info) => {
  page.on('dialog', dialog => dialog.accept());
  const browserErrors: string[] = [];page.on('pageerror', error => browserErrors.push(error.message));
  const evidence: any = {flow: 'reference-live-v1', mock_source: mockUrl};
  await page.goto('/integrity');
  await expect(page.getByRole('heading', {name: '业务完整性', exact: true})).toBeVisible();
  await page.getByLabel('参考表更新来源', {exact: true}).selectOption('reference_http');
  const modules = page.getByRole('group', {name: '业务模块', exact: true});
  await modules.getByLabel('股票历史行业', {exact: true}).uncheck();
  await modules.getByLabel('股票停牌区间', {exact: true}).uncheck();
  await page.getByLabel('完整性股票代码', {exact: true}).fill('');
  await page.getByLabel('完整性起始日期', {exact: true}).fill('2026-10-12');
  await page.getByLabel('完整性结束日期', {exact: true}).fill('2026-10-14');
  const exchanges = page.getByRole('group', {name: '日历交易所', exact: true});
  await exchanges.getByLabel('SZSE', {exact: true}).uncheck();
  await exchanges.getByLabel('BSE', {exact: true}).uncheck();
  // The user reviews actual DDL for both independent targets before any table
  // exists. Creating tables cannot trigger a source request or data import.
  await page.getByRole('button', {name: '预览参考表建表计划', exact: true}).click();
  await expect(page.locator('.reference-schema')).toContainText('CREATE TABLE IF NOT EXISTS');
  await expect(page.locator('.reference-schema')).toContainText('trade_calendar_versions');
  await page.getByRole('button', {name: '确认新增参考表结构', exact: true}).click();
  await expect(page.getByText('installed', {exact: true})).toHaveCount(2);
  let state = await fixtureState(request);
  expect(state.targets.postgres.trusted_rows).toBe(0);
  expect(state.targets.clickhouse.trusted_rows).toBe(0);
  expect(state.source_requests).toHaveLength(0);
  const seeded = await request.post(mockUrl + '/fixture/seed');
  expect(seeded.ok(), await seeded.text()).toBeTruthy();
  evidence.seed = await seeded.json();
  // The middle natural day is missing despite two separately published tasks.
  await page.getByRole('button', {name: '检查业务完整性', exact: true}).click();
  for (const target of ['postgres', 'clickhouse']) {
    const table = page.getByTestId('integrity-rows-' + target);
    await expect(table.locator(':scope > table > tbody > tr')).toHaveCount(3);
    const middle = table.locator(':scope > table > tbody > tr[data-business-date="2026-10-13"]');
    await expect(middle).toContainText('not_requested');
    await expect(table.locator(':scope > table > tbody > tr[data-business-date="2026-10-12"]')).toContainText('complete');
    await expect(table.locator(':scope > table > tbody > tr[data-business-date="2026-10-14"]')).toContainText('complete');
  }
  await page.getByRole('button', {name: '按此范围准备任务 · calendar · SSE · 2026-10-13', exact: true}).first().click();
  await expect(page.getByLabel('参考表请求起始日期', {exact: true})).toHaveValue('2026-10-13');
  await expect(page.getByLabel('参考表请求结束日期', {exact: true})).toHaveValue('2026-10-13');
  await expect(page.getByLabel('参考表请求来源', {exact: true})).toHaveValue('reference_http');
  for (const label of ['PostgreSQL', 'ClickHouse']) await page.getByRole('group', {name: '请求目标库', exact: true}).getByLabel(label, {exact: true}).check();
  await page.getByRole('button', {name: '登记参考表更新任务', exact: true}).click();
  await expect(page.getByTestId('reference-task')).toContainText('planned');
  const job = new URL(page.url()).searchParams.get('task')!;
  expect(job).toBeTruthy(); evidence.task_id = job;
  state = await fixtureState(request);
  expect(state.targets.postgres.history_rows).toBe(2);
  expect(state.targets.clickhouse.history_rows).toBe(2);
  // Real source HTTP requests are archived and paginated to the empty page.
  await page.getByRole('button', {name: '开始拉取参考数据', exact: true}).click();
  await expect.poll(async () => (await request.get('/api/reference/tasks/' + job)).json().then(item => item.status), {timeout: 45000}).toBe('previewed');
  await expect(page.getByTestId('reference-audit')).toBeVisible();
  await expect(page.getByTestId('reference-task-logs')).toContainText('http_response');
  await expect(page.getByTestId('reference-task-logs')).toContainText('200');
  await expect(page.getByTestId('reference-audit')).toContainText('SSE');
  await expect(page.getByTestId('reference-audit')).toContainText('2026-10-13');
  const audit = await (await request.get('/api/reference/tasks/' + job)).json();
  expect(audit.preview.blocking).toBe(false);
  expect(audit.preview.rejected).toBe(0);
  evidence.audit = {status: audit.status, preview: audit.preview, attempts: audit.attempts, logs: audit.logs};
  state = await fixtureState(request);
  for (const target of ['postgres', 'clickhouse']) {
    expect(state.targets[target].history_rows).toBe(2);
    expect(state.targets[target].trusted_rows).toBe(2);
  }
  const calendarRequests = state.source_requests.filter((item: any) => item.endpoint === 'trade_cal' && item.parameters.start_date === '20261013');
  expect(calendarRequests.some((item: any) => item.parameters.offset === '0')).toBeTruthy();
  expect(calendarRequests.some((item: any) => item.parameters.offset === '1')).toBeTruthy();
  evidence.before_confirm = state;
  await page.screenshot({path: info.outputPath('reference-live-audit-before-confirm.png'), fullPage: true});
  // Confirm once, then verify the business read and physical append history.
  await page.getByRole('button', {name: '确认追加并核验发布', exact: true}).click();
  await expect.poll(async () => (await request.get('/api/reference/tasks/' + job)).json().then(item => item.status), {timeout: 45000}).toBe('completed');
  await expect(page.getByTestId('reference-publish-results')).toContainText('published');
  await expect(page.getByTestId('integrity-rows-postgres').locator(':scope > table > tbody > tr[data-business-date="2026-10-13"]')).toContainText('complete');
  await expect(page.getByTestId('integrity-rows-clickhouse').locator(':scope > table > tbody > tr[data-business-date="2026-10-13"]')).toContainText('complete');
  state = await fixtureState(request);
  for (const target of ['postgres', 'clickhouse']) {
    expect(state.targets[target].history_rows).toBe(3);
    expect(state.targets[target].trusted_rows).toBe(3);
    expect(state.targets[target].trusted_dates).toEqual(['2026-10-12', '2026-10-13', '2026-10-14']);
    const visible = await (await request.get('/api/reference/versions?target=' + target)).json();
    expect(visible.rows.filter((row: any) => row.import_task_id === job)).toHaveLength(1);
  }
  evidence.after_publish = state;
  await page.screenshot({path: info.outputPath('reference-live-published-both-targets.png'), fullPage: true});
  await page.getByRole('button', {name: '撤销本任务可信发布', exact: true}).click();
  await expect.poll(async () => (await request.get('/api/reference/tasks/' + job)).json().then(item => item.status), {timeout: 45000}).toBe('rolled_back');
  await expect(page.getByTestId('reference-rollback-results')).toContainText('rollback_task_id');
  for (const target of ['postgres', 'clickhouse']) {
    const middle = page.getByTestId('integrity-rows-' + target).locator(':scope > table > tbody > tr[data-business-date="2026-10-13"]');
    await expect(middle).toContainText('revoked');
    await expect(middle).toContainText('unknown');
  }
  state = await fixtureState(request);
  for (const target of ['postgres', 'clickhouse']) {
    expect(state.targets[target].history_rows).toBe(3);
    expect(state.targets[target].trusted_rows).toBe(2);
    expect(state.targets[target].events).toBe(4);
    expect(state.targets[target].trusted_dates).toEqual(['2026-10-12', '2026-10-14']);
  }
  evidence.after_revoke = state;
  evidence.browser_errors = browserErrors;
  expect(browserErrors).toEqual([]);
  await info.attach('reference-live-evidence', {body: JSON.stringify(evidence, null, 2), contentType: 'application/json'});
  await page.screenshot({path: info.outputPath('reference-live-revoked-gap-preserved.png'), fullPage: true});
});
