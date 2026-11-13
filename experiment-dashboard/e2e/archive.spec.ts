import { test, expect } from '@playwright/test';

test.beforeEach(async ({ page }) => {
  await page.route('**/*', route => {
    const host = new URL(route.request().url()).hostname;
    return host === '127.0.0.1' || host === 'localhost' ? route.continue() : route.abort();
  });
});

test('metric catalog and historical Dashboard results stay read only', async ({ page }) => {
  const errors: string[] = [];
  const writes: string[] = [];
  page.on('pageerror', e => errors.push(e.message));
  page.on('request', request => { if (request.url().includes('/api/') && request.method() !== 'GET') writes.push(request.method()); });
  await page.goto('/');
  await expect(page.getByRole('heading', { name: '因子库', exact: true })).toBeVisible();
  await page.getByLabel('搜索因子').fill('20日');
  await expect(page.locator('.factor-metrics-table tbody tr')).toHaveCount(1);
  await page.getByRole('button', { name: '20日动量', exact: true }).click();
  await expect(page.getByRole('heading', { name: '20日动量', exact: true })).toBeVisible();
  const history = page.getByLabel('历次实验');
  await expect(history.locator('option')).toHaveCount(2);
  const initial = await history.inputValue();
  await expect(page.locator('.dashboard-result-surface').locator('.metric-block-title')).toHaveText(['裸指标', '超额指标']);
  await expect(page.getByText('分组累计收益曲线', { exact: true })).toBeVisible();
  await page.getByRole('button', { name: '每日持仓&收益', exact: true }).click();
  await expect(page.locator('.dashboard-result-surface').locator('tbody tr')).toHaveCount(16);
  const metricOnly = await history.locator('option').evaluateAll(options => options.find(option => option.textContent?.includes('仅指标'))?.getAttribute('value'));
  await history.selectOption(metricOnly!);
  await expect(page.locator('.archive-notice')).toBeVisible();
  await history.selectOption(initial);
  await page.getByRole('button', { name: '原始数据与配置', exact: true }).click();
  await expect(page.getByRole('dialog')).toBeVisible();
  await page.getByRole('button', { name: '参数与来源', exact: true }).click();
  await expect(page.getByRole('heading', { name: '完整有效配置' })).toBeVisible();
  await page.getByRole('button', { name: '原始数据', exact: true }).click();
  await expect(page.getByRole('link', { name: 'PARQUET' })).toBeVisible();
  const download = page.waitForEvent('download');
  await page.getByRole('link', { name: 'CSV', exact: true }).click();
  await expect((await download).suggestedFilename()).toContain('.csv');
  await page.getByLabel('关闭详情').click();
  expect(errors).toEqual([]);
  expect(writes).toEqual([]);
});

test('factor metrics support sorting, filtering, visible columns, and CSV export', async ({ page }) => {
  await page.goto('/');
  await expect(page.locator('.factor-metrics-table tbody tr')).toHaveCount(3);
  await page.getByRole('button', { name: '夏普比率', exact: true }).click();
  await expect(page.locator('th[aria-sort="descending"]')).toContainText('夏普比率');
  await page.getByRole('button', { name: '夏普比率', exact: true }).click();
  await expect(page.locator('th[aria-sort="ascending"]')).toContainText('夏普比率');
  await page.getByLabel('最小夏普').fill('999999');
  await page.getByRole('button', { name: '应用筛选', exact: true }).click();
  await expect(page.locator('.factor-metrics-table tbody tr')).toHaveCount(0);
  await page.getByRole('button', { name: '重置', exact: true }).click();
  await expect(page.locator('.factor-metrics-table tbody tr')).toHaveCount(3);
  await page.getByRole('button', { name: '显示列', exact: true }).click();
  await page.getByRole('checkbox', { name: '卡玛比率', exact: true }).check();
  await expect(page.getByRole('button', { name: '卡玛比率', exact: true })).toBeVisible();
  const download = page.waitForEvent('download');
  await page.getByRole('button', { name: '导出本页 CSV', exact: true }).click();
  expect((await download).suggestedFilename()).toBe('因子指标.csv');
});

test('comparison includes multiple experiments and available NAV paths', async ({ page }) => {
  await page.goto('/');
  await page.getByRole('button', { name: '实验记录', exact: true }).click();
  await expect(page.locator('.experiments-table tbody tr')).toHaveCount(5);
  await page.locator('.experiments-table input[type=checkbox]').nth(0).check();
  await page.locator('.experiments-table input[type=checkbox]').nth(1).check();
  await page.getByRole('button', { name: '比较实验', exact: true }).click();
  await expect(page.getByRole('heading', { name: '归档指标对比' })).toBeVisible();
  await expect(page.locator('.comparison-card')).toHaveCount(2);
  await expect(page.locator('.js-plotly-plot').first()).toBeVisible();
});

test('mining heatmap, filters, and narrow viewport work without internet', async ({ page }) => {
  await page.goto('/');
  await page.getByRole('button', { name: '参数挖掘', exact: true }).click();
  await expect(page.locator('.heatmaplayer')).toBeVisible();
  await expect(page.getByRole('heading', { name: '候选排名', exact: false })).toBeVisible();
  await page.setViewportSize({ width: 390, height: 844 });
  await expect(page.getByRole('heading', { name: '参数挖掘', exact: true })).toBeVisible();
  await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
});

test('timing factor uses the shared result view on a narrow screen', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto('/');
  await expect(page.locator('.factor-metrics-table tbody tr')).toHaveCount(3);
  await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.getByRole('button', { name: '趋势择时', exact: true }).click();
  await expect(page.getByRole('button', { name: '择时概览', exact: true })).toBeVisible();
  await expect(page.locator('.dashboard-result-surface').locator('.js-plotly-plot').first()).toBeVisible();
  await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  expect(errors).toEqual([]);
});
