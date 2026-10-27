import {defineConfig} from '@playwright/test';
import os from 'node:os';
import path from 'node:path';
const artifacts = process.env.BETALENS_REFERENCE_ARTIFACTS || path.join(os.homedir(), '.cache', 'betalens-reference-browser');
export default defineConfig({
  testDir: './e2e', testMatch: 'reference-live.spec.ts', timeout: 180000, workers: 1,
  outputDir: path.join(artifacts, 'browser-artifacts'),
  use: {baseURL: 'http://127.0.0.1:18084', headless: true, viewport: {width: 1440, height: 1100}, trace: 'retain-on-failure', screenshot: 'only-on-failure'},
  webServer: {command: '../.venv/bin/python ../tests/reference_e2e_server.py', url: 'http://127.0.0.1:18084/api/config', timeout: 30000, reuseExistingServer: false, gracefulShutdown: {signal: 'SIGTERM', timeout: 30000}},
  reporter: [['list'], ['html', {open: 'never', outputFolder: path.join(artifacts, 'html-report')}]],
});
