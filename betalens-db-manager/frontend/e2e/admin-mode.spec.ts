import {test, expect} from '@playwright/test';

test('配置页明确选择三种CH写入模式并显示管理员旁路边界', async ({page}) => {
  page.on('dialog', dialog => dialog.accept());
  await page.route('**/api/config', async route => {
    const response = await route.fetch();
    const config = await response.json();
    await route.fulfill({json: {...config, clickhouse_writer_mode: 'local_admin'}});
  });
  await page.goto('/settings');
  const mode = page.getByLabel('ClickHouse 写入模式', {exact: true});
  await expect(mode).toHaveValue('local_admin');
  await expect(mode.locator('option')).toHaveCount(3);
  await expect(mode.locator('option[value="local_admin"]')).toHaveText('local_admin · 本机现有管理员受控写入');
  await expect(page.getByText(/管理员模式不能阻止其他客户端直接使用管理员凭据/)).toBeVisible();
  await mode.selectOption('disabled');await expect(mode).toHaveValue('disabled');
  await mode.selectOption('local_dedicated');await expect(mode).toHaveValue('local_dedicated');
  await mode.selectOption('local_admin');await expect(mode).toHaveValue('local_admin');
  // No save, grant, connection mutation or database operation is needed to select a policy.
});
