import {test, expect, type Page, type Route} from '@playwright/test';

const taskId = 'a'.repeat(32);
const task = {
  id: taskId, status: 'planned', target_matches: true, created: '2026-10-20T00:00:00Z', tasks: [], slices: [],
  plan: {pipeline_version: 2, request: {tables: ['market_daily_raw'], targets: ['postgres'], start_date: '2026-10-20', end_date: '2026-10-20'}},
  pipeline: {datasets: ['daily', 'weekly'], receipts: [], reviews: {pending: 0}},
};

async function mockApi(page: Page, handle: (route: Route, url: URL) => Promise<boolean>) {
  page.on('dialog', dialog => dialog.accept());
  await page.routeWebSocket('**/api/live/**', () => {});
  await page.route('**/api/**', async route => {
    const url = new URL(route.request().url());
    if (await handle(route, url)) return;
    const defaults: Record<string, unknown> = {
      '/api/sessions': {id: 'pagination-session'},
      '/api/config': {requests_per_minute: 450},
      '/api/catalog': [{name: 'market_daily_raw', zh_name: '行情'}],
      '/api/sources': [],
      '/api/task-history': {rows: [], total: 0, offset: 0},
      ['/api/task-history/' + taskId]: task,
      ['/api/tasks/' + taskId + '/logs']: [],
    };
    await route.fulfill({json: defaults[url.pathname] ?? {}});
  });
}

test('历史任务显示总页数、任意跳转、边界校验，并随筛选重置', async ({page}) => {
  let total = 45;
  const offsets: number[] = [];
  await mockApi(page, async (route, url) => {
    if (url.pathname !== '/api/task-history') return false;
    const offset = Number(url.searchParams.get('offset'));
    const count = url.searchParams.get('search') ? 0 : total;
    offsets.push(offset);
    await route.fulfill({json: {total: count, offset, rows: Array.from({length: Math.max(0, Math.min(20, count - offset))}, (_, i) => ({
      id: 'task-' + String(offset + i + 1).padStart(4, '0'), status: 'planned', tables: [], targets: ['postgres'], target_matches: true,
    }))}});
    return true;
  });
  await page.goto('/tasks');
  const pager = page.getByRole('navigation', {name: '历史任务分页'});
  await expect(pager).toContainText('共 45 项 · 第 1 / 3 页');
  await expect(pager.getByRole('button', {name: '上一页任务', exact: true})).toBeDisabled();
  const input = pager.getByRole('spinbutton');
  for (const invalid of ['', '0', '-1', '1.5', '4']) {
    await input.fill(invalid);
    await expect(pager.getByRole('button', {name: '跳转', exact: true})).toBeDisabled();
  }
  expect(offsets).toEqual([0]);
  await input.fill('3');await input.press('Enter');
  await expect(pager).toContainText('第 3 / 3 页');
  await expect(page.getByRole('button', {name: 'task-0041', exact: true})).toBeVisible();
  await expect(pager.getByRole('button', {name: '下一页任务', exact: true})).toBeDisabled();
  await pager.getByRole('button', {name: '上一页任务', exact: true}).click();
  await expect(pager).toContainText('第 2 / 3 页');
  await pager.getByRole('button', {name: '末页', exact: true}).click();
  await expect(pager).toContainText('第 3 / 3 页');
  // A shrinking list cannot strand the user beyond its last page.
  total = 1;await page.getByRole('button', {name: '刷新清单', exact: true}).click();
  await expect(pager).toContainText('共 1 项 · 第 1 / 1 页');
  await page.getByRole('textbox', {name: '筛选任务编号', exact: true}).fill('missing');
  await expect(pager).toContainText('共 0 项 · 第 1 / 1 页');
  await expect(input).toBeDisabled();
  await expect(page.getByText('没有匹配的任务。', {exact: true})).toBeVisible();
});

test('请求预览可跳到末页，修改参数后重新从首页预览', async ({page}) => {
  let count = 205;
  await mockApi(page, async (route, url) => {
    if (url.pathname !== '/api/tasks/compile') return false;
    await route.fulfill({json: {requests: Array.from({length: count}, (_, i) => ({source_id: 'mock', endpoint: 'daily', parameters: {index: 'request-' + (i + 1)}}))}});
    return true;
  });
  await page.goto('/updates');
  await page.getByRole('button', {name: '匹配并预览 API 请求', exact: true}).click();
  await page.getByText('实际请求明细 · 每页 100 条', {exact: true}).click();
  const pager = page.getByRole('navigation', {name: '请求预览分页'});
  await expect(pager).toContainText('共 205 条 · 第 1 / 3 页');
  await pager.getByRole('spinbutton').fill('3');
  await pager.getByRole('button', {name: '跳转', exact: true}).click();
  await expect(pager).toContainText('第 3 / 3 页');
  await expect(page.getByText('request-201', {exact: true})).toHaveCount(1);
  await expect(page.getByText('request-1', {exact: true})).toHaveCount(0);
  await page.getByRole('textbox', {name: '证券代码', exact: true}).fill('000001.SZ');
  await expect(pager).toHaveCount(0);
  count = 1;await page.getByRole('button', {name: '匹配并预览 API 请求', exact: true}).click();
  await page.getByText('实际请求明细 · 每页 100 条', {exact: true}).click();
  await expect(pager).toContainText('共 1 条 · 第 1 / 1 页');
});

test('缓存与逐行变更独立跳转，保留接口筛选，失败时保留当前页', async ({page}) => {
  const auditOffsets: number[] = [];
  let failAudit = false;
  await mockApi(page, async (route, url) => {
    const offset = Number(url.searchParams.get('offset'));
    if (url.pathname.endsWith('/cache')) {
      const dataset = url.searchParams.get('dataset'), total = dataset === 'weekly' ? 1 : 205;
      await route.fulfill({json: {total, offset, rows: [{value: (dataset || 'all') + '-cache-' + (offset + 1)}]}});
    } else if (url.pathname.endsWith('/audit')) {
      auditOffsets.push(offset);
      expect(url.searchParams.get('after')).toBeNull();
      if (failAudit) await route.fulfill({status: 409, json: {detail: '审计读取失败'}});
      else await route.fulfill({json: {total: 205, offset, changes: [{value: 'change-' + (offset + 1)}], receipts: [], rollbacks: []}});
    } else return false;
    return true;
  });
  await page.goto('/tasks?task=' + taskId);
  await page.getByRole('tab', {name: '缓存与写入审计', exact: true}).click();
  await page.getByText('原始响应缓存与下载', {exact: true}).click();
  await page.getByRole('button', {name: '查看审计缓存', exact: true}).click();
  const cache = page.getByRole('navigation', {name: '原始缓存分页'});
  await expect(cache).toContainText('第 1 / 3 页');
  await page.getByRole('combobox', {name: '缓存接口', exact: true}).selectOption('daily');
  await page.getByRole('button', {name: '查看审计缓存', exact: true}).click();
  await cache.getByRole('spinbutton').fill('3');await cache.getByRole('spinbutton').press('Enter');
  await expect(cache).toContainText('第 3 / 3 页');
  await expect(page.getByText('daily-cache-201', {exact: true})).toBeVisible();
  await cache.getByRole('button', {name: '上一页缓存', exact: true}).click();
  await expect(cache).toContainText('第 2 / 3 页');
  await page.getByRole('button', {name: '查看写入结果', exact: true}).click();
  await page.getByText('逐行变更与核验结果', {exact: true}).click();
  const audit = page.getByRole('navigation', {name: '逐行变更分页'});
  await expect(audit).toContainText('第 1 / 3 页');
  await audit.getByRole('button', {name: '末页', exact: true}).click();
  await expect(audit).toContainText('第 3 / 3 页');
  await expect(page.getByText('change-201', {exact: true})).toBeVisible();
  await expect(audit.getByRole('button', {name: '下一页变更', exact: true})).toBeDisabled();
  await expect(cache).toContainText('第 2 / 3 页');
  failAudit = true;await audit.getByRole('button', {name: '上一页变更', exact: true}).click();
  await expect(page.getByRole('alert')).toContainText('审计读取失败');
  await expect(audit).toContainText('第 3 / 3 页');
  failAudit = false;await audit.getByRole('button', {name: '首页', exact: true}).click();
  await expect(audit).toContainText('第 1 / 3 页');
  expect(auditOffsets).toEqual([0, 200, 100, 0]);
  await page.getByRole('combobox', {name: '缓存接口', exact: true}).selectOption('weekly');
  await expect(cache).toHaveCount(0);
  await page.getByRole('button', {name: '查看审计缓存', exact: true}).click();
  await expect(cache).toContainText('共 1 行 · 第 1 / 1 页');
});

test('拒收批次任意跳转后筛选回到首页', async ({page}) => {
  const offsets: number[] = [];
  await mockApi(page, async (route, url) => {
    if (url.pathname.startsWith('/api/tasks/' + taskId + '/reviews/')) {
      await route.fulfill({json: {batch: url.pathname.split('/').at(-1), status: 'pending', reasons: ['示例拒收原因'], tables: [], sources: [], targets: ['postgres']}});
      return true;
    }
    if (url.pathname !== '/api/tasks/' + taskId + '/reviews') return false;
    const offset = Number(url.searchParams.get('offset'));
    offsets.push(offset);
    const total = url.searchParams.get('status') ? 0 : 41;
    const rows = Array.from({length: Math.max(0, Math.min(20, total - offset))}, (_, i) => ({batch: 'batch-' + (offset + i + 1), status: 'pending', reasons: ['示例拒收原因'], cached_rows: 1}));
    await route.fulfill({json: {rows, total, offset, counts: {total: 41, pending: 41}, target_matches: true, active: false}});
    return true;
  });
  await page.goto('/tasks/review?task=' + taskId);
  const pager = page.getByRole('navigation', {name: '拒收批次分页'});
  await expect(pager).toContainText('共 41 个批次 · 第 1 / 3 页');
  await pager.getByRole('spinbutton').fill('3');await pager.getByRole('button', {name: '跳转', exact: true}).click();
  await expect(pager).toContainText('第 3 / 3 页');
  await expect(page.getByRole('button', {name: 'batch-41', exact: true})).toBeVisible();
  expect(offsets).toEqual([0, 40]);
  await pager.getByRole('button', {name: '上一页拒收', exact: true}).click();
  await expect(pager).toContainText('第 2 / 3 页');
  await page.getByRole('combobox', {name: '筛选人工审计状态', exact: true}).selectOption('rejected');
  await expect(pager).toContainText('共 0 个批次 · 第 1 / 1 页');
  await expect(pager.getByRole('spinbutton')).toBeDisabled();
  expect(offsets.at(-1)).toBe(0);
});
