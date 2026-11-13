// A Dashboard result change must reach both shipped browser applications.
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const online = fileURLToPath(new URL('../', import.meta.url));
const offline = fileURLToPath(new URL('../../../experiment-dashboard/', import.meta.url));
const npmCli = process.env.npm_execpath;

for (const [cwd, script] of [[online, 'build:local'], [offline, 'build']]) {
  const command = npmCli ? process.execPath : process.platform === 'win32' ? 'npm.cmd' : 'npm';
  const args = npmCli ? [npmCli, 'run', script] : ['run', script];
  const result = spawnSync(command, args, { cwd, stdio: 'inherit', shell: !npmCli && process.platform === 'win32' });
  if (result.error) throw result.error;
  if (result.status !== 0) process.exit(result.status || 1);
}
