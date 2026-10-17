// Drive dev/engine-test.html in headless Chrome (WebGPU) and save activations.
import { chromium } from '../web/node_modules/playwright-core/index.mjs';
import fs from 'node:fs';
const [file, layers, out] = process.argv.slice(2);
const browser = await chromium.launch({
  executablePath: '/usr/bin/google-chrome',
  args: (process.env.CHROME_ARGS ?? '--enable-unsafe-webgpu --use-webgpu-adapter=swiftshader --enable-unsafe-swiftshader --ignore-gpu-blocklist').split(' '),
});
const page = await browser.newPage();
page.on('console', (m) => console.log('[page]', m.text().slice(0, 400)));
page.on('pageerror', (e) => console.log('[pageerror]', e.message));
await page.goto(`http://localhost:8658/dev/engine-test.html?file=${encodeURIComponent('/@fs' + file)}&layers=${layers}`);
await page.waitForFunction(() => (window).__result, null, { timeout: 4 * 60 * 60 * 1000, polling: 1000 });
const r = await page.evaluate(() => (window).__result);
fs.writeFileSync(out, JSON.stringify(r));
await browser.close();
