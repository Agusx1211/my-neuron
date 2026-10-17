// End-to-end run of the real app (worker + WASM + WebGPU + OPFS) in headless Chrome.
// usage (inside the container): node dev/e2e.mjs <outdir> <model-id> <local gguf path> [text]
import { chromium } from '../web/node_modules/playwright-core/index.mjs';

const [out, modelId, gguf, text = 'Lionel Messi'] = process.argv.slice(2);
const base = process.env.URL ?? 'http://localhost:8657/';
const browser = await chromium.launch({
  executablePath: '/usr/bin/google-chrome',
  args: ['--enable-unsafe-webgpu', '--use-webgpu-adapter=swiftshader', '--enable-unsafe-swiftshader',
    '--use-angle=swiftshader', '--ignore-gpu-blocklist', '--enable-webgl'],
});
const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
const t0 = Date.now();
const log = (...a) => console.log(`[${((Date.now() - t0) / 1000).toFixed(0)}s]`, ...a);
page.on('console', (m) => log('[console]', m.type(), m.text().slice(0, 300)));
page.on('pageerror', (e) => log('[pageerror]', e.message));
const shot = async (name) => {
  try { await page.screenshot({ path: `${out}/e2e-${name}.png`, timeout: 600000 }); log('shot', name); } catch (e) { log('shot failed', name, e.message.split('\n')[0]); }
};
const LONG = 60 * 60 * 1000;

await page.goto(`${base}?gguf=${encodeURIComponent(gguf)}`, { waitUntil: 'load' });
await page.waitForTimeout(3000);
await page.click(`label:has(input[name=model][value="${modelId}"])`);
await page.click('#load-btn');
await page.waitForTimeout(20000);
await shot('1-loading');
await page.waitForSelector('#prompt:not([hidden])', { timeout: LONG });
log('model ready');
await page.waitForTimeout(2500);
await shot('2-ready');
await page.fill('#prompt-input', text);
await page.keyboard.press('Enter');
await page.waitForSelector('#card:not([hidden])', { timeout: LONG });
log('analysis shown');
await page.waitForTimeout(4000);
await shot('3-focused');
const card = await page.evaluate(() => document.getElementById('card')?.innerText);
log('CARD:\n' + card);
await page.click('#explore-btn');
await page.waitForFunction(() => /Verified|verified/.test(document.getElementById('verify-status')?.textContent ?? ''), null, { timeout: LONG, polling: 2000 }).catch((e) => log('verify wait failed', e.message));
await page.waitForTimeout(3000);
await shot('4-constellation');
log('VERIFY:\n' + await page.evaluate(() => document.getElementById('verify-section')?.innerText));
// second visit: the weights should now come from the OPFS cache
await page.reload({ waitUntil: 'load', timeout: LONG });
await page.waitForTimeout(4000);
log('PICKER after reload:\n' + await page.evaluate(() => document.getElementById('model-picker')?.innerText));
await page.click(`label:has(input[name=model][value="${modelId}"])`);
const t1 = Date.now();
await page.click('#load-btn');
await page.waitForTimeout(15000);
log('PHASE during reload: ' + await page.evaluate(() => document.getElementById('load-phase')?.textContent));
await page.waitForSelector('#prompt:not([hidden])', { timeout: LONG });
log(`cached load took ${((Date.now() - t1) / 1000).toFixed(0)}s`);
await browser.close();
