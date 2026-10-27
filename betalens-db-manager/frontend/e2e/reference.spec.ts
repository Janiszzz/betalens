import {test, expect, Page} from '@playwright/test';

async function baseRoutes(page: Page) {
  page.on('dialog', dialog => dialog.accept());
  await page.route('**/api/catalog', route => route.fulfill({json: []}));
  await page.route('**/api/sources', route => route.fulfill({json: []}));
  await page.route('**/api/tasks', route => route.fulfill({json: []}));
  await page.route('**/api/config', route => route.fulfill({json: {requests_per_minute: 40}}));
  await page.route('**/api/sessions', route => route.fulfill({json: {id: 'reference-ui-session'}}));
  await page.route('**/api/sessions/reference-ui-session/**', route => route.fulfill({json: {ok: true}}));
  await page.route('**/api/reference/tables', route => route.fulfill({json: [
    {table: 'trade_calendar_versions', target: 'postgres', status: 'not_deployed'},
    {table: 'trade_calendar_versions', target: 'clickhouse', status: 'not_deployed'},
  ]}));
}

test('完整性缺口生成可审查请求，显式拉取和审计后才允许追加发布，撤销保留历史', async ({page}, info) => {
  await baseRoutes(page);
  const errors: string[] = []; page.on('pageerror', error => errors.push(error.message));
  let task: any; const calls: string[] = []; let request: any; let publishBody: any; let checks = 0;
  await page.route('**/api/integrity/check', async route => {
    checks++; const body = route.request().postDataJSON();
    await route.fulfill({json: {results: Object.fromEntries(body.targets.map((target: string) => [target, {
      summary: {requested: 3, unknown: task?.status === 'completed' ? 0 : 1},
      rows: [{module: 'suspension', security_code: '000001.SZ', date: '2026-10-13', status: task?.status === 'completed' ? 'complete' : 'unknown', reason: task?.status === 'completed' ? '未报告停牌' : '未请求', revision_id: task?.status === 'completed' ? 'revision-1' : null}],
    }]))}});
  });
  await page.route('**/api/reference/tasks', async route => {
    if (route.request().method() === 'POST') {
      request = route.request().postDataJSON(); calls.push('create');
      task = {id: 'reference-ui-task', status: 'planned', request, attempts: [], logs: []};
      await route.fulfill({json: task});
    } else await route.fulfill({json: task ? [task] : []});
  });
  await page.route('**/api/reference/tasks/reference-ui-task', route => route.fulfill({json: task}));
  await page.route('**/api/reference/tasks/reference-ui-task/download', async route => {
    calls.push('download'); task = {...task, status: 'downloaded', attempts: [{attempt_id: 'attempt-1', status: 'completed'}], logs: [{event: 'request', source_endpoint: 'suspend_d', http_status: 200, rows: 0}]};
    await route.fulfill({json: task});
  });
  await page.route('**/api/reference/tasks/reference-ui-task/preview', async route => {
    calls.push('preview'); task = {...task, status: 'previewed', preview: {token: 'review-token', blocking: false, raw_sample: [{security_code: '000001.SZ', suspend_timing: null}], normalized_sample: [{security_code: '000001.SZ', available_at: null, downloaded_at: '2026-10-25T00:00:00+00:00'}], scopes: [{scope_id: 'scope-1', reason: '完整响应无事件'}], heads: [{target: 'postgres', revision_id: 'old-revision'}], changes: [{before: {revision_id: 'old-revision'}, after: {revision_id: 'revision-1', status: '未报告停牌'}}]}};
    await route.fulfill({json: task});
  });
  await page.route('**/api/reference/tasks/reference-ui-task/commit', async route => {
    calls.push('commit'); publishBody = route.request().postDataJSON();
    task = {...task, status: 'completed', result: {targets: [{target: 'postgres', module: 'suspension', status: 'published', revision_id: 'revision-1'}]}};
    await route.fulfill({json: task});
  });
  await page.route('**/api/reference/tasks/reference-ui-task/rollback', async route => {
    calls.push('rollback'); task = {...task, status: 'revoked', result: {targets: [{target: 'postgres', status: 'revoked', rollback_task_id: 'rollback-1'}]}};
    await route.fulfill({json: task});
  });
  await page.goto('/integrity');
  await expect(page.getByRole('heading', {name: '业务完整性', exact: true})).toBeVisible();
  await expect(page.getByRole('link', {name: '业务完整性', exact: true})).toHaveAttribute('href', '/integrity');
  const targetOptions = page.getByRole('group', {name: '检查目标库', exact: true});
  await expect(targetOptions.getByLabel('PostgreSQL')).toBeChecked();
  await expect(targetOptions.getByLabel('ClickHouse')).toBeChecked();
  await page.getByLabel('完整性起始日期').fill('2026-10-12');
  await page.getByLabel('完整性结束日期').fill('2026-10-14');
  await page.getByRole('button', {name: '检查业务完整性', exact: true}).click();
  await expect(page.getByTestId('integrity-rows-postgres')).toContainText('未请求');
  await expect(page.getByTestId('integrity-rows-clickhouse')).toContainText('000001.SZ');
  expect(calls).toEqual([]);
  await page.getByRole('button', {name: '按此范围准备任务 · suspension · 000001.SZ · 2026-10-13', exact: true}).first().click();
  await expect(page.getByTestId('reference-request-draft')).toBeVisible();
  await expect(page.getByLabel('参考表请求起始日期')).toHaveValue('2026-10-13');
  await expect(page.getByLabel('参考表请求结束日期')).toHaveValue('2026-10-13');
  expect(calls).toEqual([]);
  await page.getByRole('button', {name: '登记参考表更新任务', exact: true}).click();
  await expect(page.getByTestId('reference-task')).toContainText('reference-ui-task');
  expect(request.targets).toEqual(['postgres']); expect(request.modules).toEqual(['suspension']);
  expect(request.security_codes).toEqual(['000001.SZ']); expect(calls).toEqual(['create']);
  await expect(page).toHaveURL(/\/integrity\?task=reference-ui-task$/);
  await expect(page.getByRole('button', {name: '确认追加并核验发布', exact: true})).toHaveCount(0);
  await page.getByRole('button', {name: '开始拉取参考数据', exact: true}).click();
  await expect(page.getByTestId('reference-task-logs')).toContainText('200');
  expect(calls).toEqual(['create', 'download']);
  await page.getByRole('button', {name: '检查参考数据缓存与差异', exact: true}).click();
  await expect(page.getByTestId('reference-audit')).toContainText('原始响应缓存');
  await expect(page.getByTestId('reference-audit')).toContainText('完整响应无事件');
  expect(calls).toEqual(['create', 'download', 'preview']);
  await page.getByRole('button', {name: '确认追加并核验发布', exact: true}).click();
  await expect(page.getByTestId('reference-publish-results')).toContainText('published');
  expect(publishBody).toEqual({token: 'review-token', accept_partial: false});
  await expect(page.getByTestId('integrity-rows-postgres')).toContainText('未报告停牌');
  await page.getByRole('button', {name: '撤销本任务可信发布', exact: true}).click();
  await expect(page.getByTestId('reference-publish-results')).toContainText('rollback-1');
  await expect(page.getByTestId('integrity-rows-postgres')).toContainText('未请求');
  expect(calls).toEqual(['create', 'download', 'preview', 'commit', 'rollback']);
  expect(checks).toBeGreaterThanOrEqual(3); expect(errors).toEqual([]);
  await page.screenshot({path: info.outputPath('reference-task-revocation.png'), fullPage: true});
});

test('参考表建表预览不写库，过期审计拒绝写入，版本审计明确包含未发布数据', async ({page}) => {
  await baseRoutes(page);
  let applies = 0; let versionRequest = ''; let commitCalls = 0;
  const task = {id: 'blocked-reference', status: 'previewed', request: {targets: ['postgres']}, preview: {token: 'expired', blocking: true, errors: [{reason: '版本头已改变，需要重新审计'}]}};
  await page.route('**/api/reference/tasks', route => route.fulfill({json: [task]}));
  await page.route('**/api/reference/tasks/blocked-reference', route => route.fulfill({json: task}));
  await page.route('**/api/reference/tasks/blocked-reference/commit', route => {commitCalls++;return route.fulfill({status: 409, json: {detail: 'stale confirmation'}});});
  await page.route('**/api/reference/schema/preview', route => route.fulfill({json: {token: 'schema-review', targets: [{target: 'postgres', ddl: ['CREATE TABLE betalens.trade_calendar_versions (...)']}, {target: 'clickhouse', ddl: ['CREATE TABLE betalens.trade_calendar_versions (...)']}]}}));
  await page.route('**/api/reference/schema/apply', route => {applies++;return route.fulfill({json: {results: [{target: 'postgres', status: 'completed'}, {target: 'clickhouse', status: 'failed', reason: '模拟单库失败'}]}});});
  await page.route('**/api/reference/versions?*', route => {versionRequest = route.request().url();return route.fulfill({json: {revisions: [{revision_id: 'unpublished', status: 'unpublished'}], scopes: [], rows: [{revision_id: 'unpublished', security_code: '000001.SZ'}], events: [{revision_id: 'old', event: 'revoked'}]}});});
  await page.goto('/integrity?task=blocked-reference');
  await expect(page.getByRole('button', {name: '确认追加并核验发布', exact: true})).toBeDisabled();
  await expect(page.getByTestId('reference-audit')).toContainText('版本头已改变，需要重新审计');
  expect(commitCalls).toBe(0);
  await page.getByRole('button', {name: '预览参考表建表计划', exact: true}).click();
  await expect(page.getByRole('button', {name: '确认新增参考表结构', exact: true})).toBeVisible();
  expect(applies).toBe(0);
  await page.getByRole('button', {name: '确认新增参考表结构', exact: true}).click();
  await expect(page.getByText('模拟单库失败', {exact: true})).toBeVisible(); expect(applies).toBe(1);
  await page.getByLabel('审计模式：包含全部版本、未发布记录和撤销历史').check();
  await page.getByRole('button', {name: '查询参考表版本', exact: true}).click();
  await expect(page.getByTestId('reference-versions')).toContainText('unpublished');
  expect(versionRequest).toContain('audit=true');
  await page.getByLabel('筛选修订编号').fill('does-not-exist');
  await expect(page.getByTestId('reference-versions').getByText('000001.SZ', {exact: true})).toHaveCount(0);
});
