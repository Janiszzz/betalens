import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: './e2e',
  timeout: 30000,
  workers: 1,
  use: { baseURL: 'http://127.0.0.1:8021', browserName: 'chromium', screenshot: 'only-on-failure' },
  webServer: {
    command: '../.venv/bin/python -m betalens.experiments serve --db ../outputs/experiments-e2e --port 8021',
    url: 'http://127.0.0.1:8021/api/health',
    reuseExistingServer: false,
    timeout: 30000,
  },
});
