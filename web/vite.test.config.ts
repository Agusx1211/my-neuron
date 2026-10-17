// Static dev server for long-running test pages: no HMR, no file watching, so
// edits elsewhere never reload a test in progress. Serves web/ on :8658.
import { defineConfig } from 'vite';

export default defineConfig({
  root: '.',
  server: { host: '0.0.0.0', port: 8658, strictPort: true, hmr: false, watch: { ignored: ['**/*'] }, fs: { allow: ['..'] } },
  worker: { format: 'es' },
});
