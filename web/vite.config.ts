import { defineConfig } from 'vite';
import basicSsl from '@vitejs/plugin-basic-ssl';

// WebGPU and OPFS need a secure context, so LAN/Tailscale previews are served
// over HTTPS with a self-signed certificate (HTTPS=1).
const https = process.env.HTTPS === '1';

export default defineConfig({
  plugins: https ? [basicSsl()] : [],
  // dev only: lets the engine test page stream local GGUF files via /@fs/
  server: { host: '0.0.0.0', port: 8655, strictPort: true, allowedHosts: true, fs: { allow: ['..'] } },
  preview: { host: '0.0.0.0', port: 8655, strictPort: true, allowedHosts: true },
  worker: { format: 'es' },
  build: { target: 'es2022' },
});
