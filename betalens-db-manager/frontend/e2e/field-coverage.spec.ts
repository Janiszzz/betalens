import {test, expect} from '@playwright/test';

test('字段覆盖展示生产节点、留空依据、目标库证据；保存直接规则保留派生声明', async ({page}, info) => {
  page.on('dialog', dialog => dialog.accept());
  const errors: string[] = [];page.on('pageerror', error => errors.push(error.message));
  const fields = [
    {name: 'code', type: 'text', nullable: false}, {name: 'raw_close', type: 'numeric'},
    {name: 'adjusted_close', type: 'numeric'}, {name: 'asset_type', type: 'text'},
    {name: 'pcf', type: 'numeric'}, {name: 'raw_open', type: 'numeric'},
    {name: 'missing_value', type: 'numeric'}, {name: 'ingested_at', type: 'datetime', system: true},
  ];
  const producers = [
    {column: 'adjusted_close', kind: 'computed', processor: 'post_adjusted_v1', inputs: ['prices.close', 'adjustment.factor'], reason: '原始价格乘复权因子', scope: {asset_type: 'stock'}},
    {column: 'asset_type', kind: 'constant', value: 'stock', inputs: [], reason: '本来源固定股票', scope: {}},
    {column: 'pcf', kind: 'unavailable', inputs: [], reason: '日线接口未提供市现率，保留空值', scope: {}},
    {column: 'raw_open', kind: 'not_applicable', inputs: [], reason: '原始行情无需复权辅助列', scope: {}},
  ];
  let document: any = {mapping: {table: 'ui_coverage', bindings: [{source_id: 'ui_vendor', endpoint: 'prices', mapping_mode: 'direct', fields: [{source_field: 'ticker', column: 'code'}, {source_field: 'close', column: 'raw_close'}], join_keys: [{source_field: 'ticker', column: 'code'}]}], producers}, protected_columns: ['ingested_at'], builtin_table: false, file_hash: 'first', yaml: 'table: ui_coverage'};
  let savedMapping: any;let coverageCalls = 0;
  const columns = fields.map(field => ({column: field.name, ...field, producers: producers.filter(producer => producer.column === field.name), evidence: field.name === 'raw_close' ? [{target: 'postgres', task_id: 'task', status: 'source_null', applicable_count: 1, source_null_count: 1, verified_count: 0}] : [], status: field.name === 'missing_value' ? 'unmapped' : 'configured'}));
  await page.route('**/api/catalog', route => route.fulfill({json: [{name: 'ui_coverage', zh_name: '覆盖界面验收', grain: '代码一行', keys: ['code'], fields}]}));
  await page.route('**/api/sources', route => route.fulfill({json: [{source_id: 'ui_vendor', label: '界面测试来源', configured: true, endpoints: {prices: {fields: [{name: 'ticker', type: 'text'}, {name: 'close', type: 'numeric'}]}}}]}));
  await page.route('**/api/tables/ui_coverage/mapping', async route => {
    if (route.request().method() === 'POST') {savedMapping = route.request().postDataJSON().mapping;document = {...document, mapping: savedMapping, file_hash: 'second'};}
    await route.fulfill({json: document});
  });
  await page.route('**/api/tables/ui_coverage/mapping/validate', route => route.fulfill({json: {diagnostics: []}}));
  await page.route('**/api/tables/ui_coverage/field-coverage', route => {coverageCalls++;return route.fulfill({json: {table: 'ui_coverage', columns, summary: {}}});});
  await page.goto('/mappings?table=ui_coverage');
  await expect(page.getByRole('heading', {name: '本地列生产规则与实证', exact: true})).toBeVisible();
  await expect(page.getByTestId('mapping-producer-node')).toHaveCount(2);
  await expect(page.locator('[data-testid="mapping-edge"][data-kind="derived"]')).toHaveCount(2);
  await expect(page.locator('[data-testid="mapping-edge"][data-kind="dependency"]')).toHaveCount(2);
  await expect(page.getByTestId('target-adjusted_close')).toContainText('计算');
  await page.getByLabel('本地字段筛选', {exact: true}).selectOption('unmapped');
  await expect(page.locator('.target-field')).toHaveCount(1);await expect(page.getByTestId('target-missing_value')).toBeVisible();
  await page.getByLabel('本地字段筛选', {exact: true}).selectOption('not_applicable');
  await expect(page.locator('.target-field')).toHaveCount(1);await expect(page.getByTestId('target-raw_open')).toContainText('不适用');
  await page.getByLabel('本地字段筛选', {exact: true}).selectOption('source_null');
  await expect(page.locator('.target-field')).toHaveCount(1);await expect(page.getByTestId('target-raw_close')).toBeVisible();
  await page.getByLabel('实证目标库', {exact: true}).selectOption('clickhouse');
  await expect(page.locator('.target-field')).toHaveCount(0);
  await page.getByLabel('实证目标库', {exact: true}).selectOption('postgres');
  await page.getByLabel('本地字段筛选', {exact: true}).selectOption('all');
  await page.getByText('逐列生产规则、空值原因与目标库证据', {exact: true}).click();
  await expect(page.getByTestId('coverage-pcf')).toContainText('日线接口未提供市现率');
  await expect(page.getByTestId('coverage-adjusted_close')).toContainText('未验证');
  await expect(page.getByTestId('coverage-raw_close')).toContainText('source_null');
  await page.getByLabel('覆盖表包含系统列', {exact: true}).check();
  await expect(page.getByTestId('coverage-ingested_at')).toContainText('系统生成');
  await page.getByTestId('source-close').click();
  await page.getByLabel('乘数 close', {exact: true}).fill('0.01');
  const before = coverageCalls;
  await page.getByRole('button', {name: '保存映射规则', exact: true}).click();
  await expect.poll(() => coverageCalls).toBeGreaterThan(before);
  expect(savedMapping.producers).toEqual(producers);
  expect(savedMapping.bindings[0].fields.find((field: any) => field.source_field === 'close').multiplier).toBe('0.01');
  expect(errors).toEqual([]);
  await page.screenshot({path: info.outputPath('column-coverage-and-producers.png'), fullPage: true});
});

test('字段覆盖请求失败仍可编辑映射，切换表不接受旧响应', async ({page}) => {
  page.on('dialog', dialog => dialog.accept());
  const errors: string[] = [];page.on('pageerror', error => errors.push(error.message));
  const fields = [{name: 'code', type: 'text', nullable: false}];
  await page.route('**/api/catalog', route => route.fulfill({json: ['ui_first', 'ui_second'].map(name => ({name, zh_name: name, grain: '一行', keys: ['code'], fields}))}));
  await page.route(/\/api\/tables\/ui_(first|second)\/mapping$/, route => route.fulfill({json: {mapping: {table: route.request().url().includes('ui_first') ? 'ui_first' : 'ui_second', bindings: []}, builtin_table: false, protected_columns: [], file_hash: 'test', yaml: ''}}));
  await page.route('**/api/tables/ui_first/field-coverage', async route => {await new Promise(resolve => setTimeout(resolve, 600));await route.fulfill({json: {table: 'ui_first', columns: [{column: 'code', producers: [{kind: 'constant', value: 'old', reason: '旧表不应泄入新表'}], evidence: []}], summary: {}}});});
  await page.route('**/api/tables/ui_second/field-coverage', route => route.fulfill({status: 503, json: {detail: '覆盖服务暂不可用'}}));
  await page.goto('/mappings?table=ui_first');
  await page.getByLabel('选择本地表', {exact: true}).selectOption('ui_second');
  await expect(page.getByText(/字段覆盖无法加载：覆盖服务暂不可用/)).toBeVisible();
  await expect(page.getByRole('button', {name: '保存映射规则', exact: true})).toBeEnabled();
  await expect(page.getByText('旧表不应泄入新表', {exact: false})).toHaveCount(0);
  expect(errors).toEqual([]);
});
