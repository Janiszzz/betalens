import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { spawn, type ChildProcess } from 'node:child_process';
import { existsSync } from 'node:fs';
import { resolve } from 'node:path';

const repoRoot = resolve(__dirname, '../..');
const backendUrl = 'http://127.0.0.1:8000';

const sleep = (ms: number) => new Promise((resolveSleep) => setTimeout(resolveSleep, ms));

async function waitForBackend(timeoutMs = 30000) {
  const started = Date.now();
  while (Date.now() - started < timeoutMs) {
    try {
      const response = await fetch(`${backendUrl}/api/health`);
      if (response.ok) return true;
    } catch {
      // Retry until timeout.
    }
    await sleep(500);
  }
  return false;
}

function dashboardBackendPlugin() {
  let child: ChildProcess | null = null;
  return {
    name: 'betalens-dashboard-backend',
    async configureServer() {
      if (await waitForBackend(12000)) return;

      const venvPython = resolve(repoRoot, '.venv/Scripts/python.exe');
      const python = existsSync(venvPython) ? venvPython : 'python';
      child = spawn(
        python,
        ['-m', 'uvicorn', 'dashboard.backend.main:app', '--host', '127.0.0.1', '--port', '8000'],
        { cwd: repoRoot, stdio: 'inherit', windowsHide: true }
      );
      await waitForBackend();

      const stop = () => {
        if (child && !child.killed) child.kill();
      };
      process.once('exit', stop);
      process.once('SIGINT', () => {
        stop();
        process.exit();
      });
      process.once('SIGTERM', () => {
        stop();
        process.exit();
      });
    }
  };
}

export default defineConfig({
  plugins: [react(), dashboardBackendPlugin()],
  server: {
    proxy: {
      '/api': backendUrl
    }
  }
});
