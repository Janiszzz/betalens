import {defineConfig} from '@playwright/test';

export default defineConfig({
  testDir: './e2e',
  testMatch: 'pagination.spec.ts',
  timeout: 30000,
  workers: 1,
  use: {baseURL: 'http://127.0.0.1:18083', headless: true, viewport: {width: 1440, height: 1100}, screenshot: 'only-on-failure'},
  webServer: {command: 'npm run dev -- --port 18083 --strictPort', url: 'http://127.0.0.1:18083', reuseExistingServer: false},
  reporter: 'list',
});
