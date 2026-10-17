// Drive the mock UI in headless Chrome (SwiftShader WebGL) and capture screenshots.
// usage (inside the container): node dev/shots.mjs <outdir> [width] [height]
import { chromium } from '../web/node_modules/playwright-core/index.mjs';

const out = process.argv[2] ?? 'dev/shots';
const W = Number(process.argv[3] ?? 1440), H = Number(process.argv[4] ?? 900);
const url = process.env.URL ?? 'http://localhost:8655/?mock';
const tag = `${W}x${H}`;

const browser = await chromium.launch({
  executablePath: '/usr/bin/google-chrome',
  args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist', '--enable-webgl'],
});
const page = await browser.newPage({ viewport: { width: W, height: H }, deviceScaleFactor: 1, reducedMotion: 'no-preference' });
page.on('console', (m) => { if (m.type() === 'error' || m.type() === 'warning') console.log('[console]', m.type(), m.text()); });
page.on('pageerror', (e) => console.log('[pageerror]', e.message));
const shot = async (name) => { await page.screenshot({ path: `${out}/${tag}-${name}.png` }); console.log('shot', name); };
const wait = (ms) => page.waitForTimeout(ms);

await page.goto(url, { waitUntil: 'load', timeout: 90000 });
await page.evaluate(() => localStorage.clear());
await page.reload({ waitUntil: 'load', timeout: 90000 });
await wait(3500);
await shot('1-landing');
await page.click('#load-btn');
await wait(3000);
await shot('2-loading');
await page.waitForSelector('#prompt:not([hidden])', { timeout: 60000 });
await wait(2800);
await shot('3-ready');
await page.fill('#prompt-input', process.env.TEXT ?? 'Lionel Messi');
await page.keyboard.press('Enter');
await wait(1800);
await shot('4-analyzing');
for (const k of ['a', 'b', 'c', 'd', 'e']) { await wait(900); await shot(`4${k}-pulse`); }
await page.waitForSelector('#card:not([hidden])', { timeout: 90000 });
await wait(2500);
await shot('5-focused');
await page.click('#explore-btn');
await wait(6500);
await shot('6-constellation');
await page.fill('#test-input', 'tango');
await page.keyboard.press('Enter');
await wait(2500);
await shot('7-tested');
await page.click('#runners li:nth-child(3) button');
await wait(4000);
await shot('8-runner');
await page.click('#again-btn');
await wait(4500);
await shot('9-again');
await page.click('#howto-open');
await wait(800);
await shot('10-howto');
await page.keyboard.press('Escape');
const fps = await page.evaluate(() => new Promise((r) => { let n = 0; const t0 = performance.now(); const f = () => { n++; if (performance.now() - t0 < 2000) requestAnimationFrame(f); else r(n / 2); }; requestAnimationFrame(f); }));
console.log('fps (swiftshader)', fps);
await browser.close();
