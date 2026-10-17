// Run on the M2: full flow on the live site via a shared link, checking analytics
// traffic and that the typed text never reaches Google Analytics.
import { chromium } from 'playwright-core';
const url = process.argv[2];
const secret = 'Lionel Messi';
const browser = await chromium.launchPersistentContext(`${process.env.HOME}/my-neuron-gpu/profile`, {
  executablePath: `${process.env.HOME}/my-neuron-gpu/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing`,
  args: ['--enable-unsafe-webgpu'],
  viewport: { width: 1440, height: 900 },
  permissions: ['clipboard-read', 'clipboard-write'],
});
const page = browser.pages()[0] ?? await browser.newPage();
const t0 = Date.now();
const log = (...a) => console.log(`[${((Date.now() - t0) / 1000).toFixed(1)}s]`, ...a);
const ga = [];
let leaked = 0;
page.on('request', (r) => {
  const u = r.url();
  if (!/google-analytics\.com|googletagmanager\.com/.test(u)) return;
  const body = r.postData() ?? '';
  const all = decodeURIComponent(u + ' ' + body).toLowerCase();
  if (all.includes(secret.toLowerCase()) || all.includes('lionel')) leaked++;
  const events = [...(u + '\n' + body).matchAll(/(?:^|[?&\n])en=([^&\n]+)/g)].map((m) => decodeURIComponent(m[1]));
  ga.push(events.length ? events.join(',') : new URL(u).pathname);
});
page.on('pageerror', (e) => log('[pageerror]', e.message));
page.on('console', (m) => { if (m.type() === 'error') log('[console error]', m.text().slice(0, 200)); });

await page.goto(`${url}?q=${encodeURIComponent(secret)}`, { waitUntil: 'load' });
await page.waitForTimeout(2500);
log('address bar:', page.url());
log('note:', await page.textContent('#shared-note'));
await page.click('#load-btn');
await page.waitForSelector('#prompt:not([hidden])', { timeout: 600000 });
log('model ready');
await page.waitForSelector('#card:not([hidden])', { timeout: 120000 });
log('auto-analyzed:', (await page.textContent('#card-title')).replace(/\s+/g, ' '), '|', await page.textContent('#card-theme'));
await page.waitForTimeout(2500);
await page.click('#explore-btn');
await page.waitForFunction(() => /Verified/i.test(document.getElementById('verify-status')?.textContent ?? ''), null, { timeout: 120000 });
await page.waitForTimeout(1500);
log('verify:', (await page.textContent('#concept-list')).replace(/\s+/g, ' ').slice(0, 200));
await page.click('#share-copy');
await page.waitForTimeout(500);
log('share text:', await page.evaluate(() => navigator.clipboard.readText()));
await page.waitForTimeout(4000);
log('GA requests:', ga.length, '| events:', [...new Set(ga.join(',').split(','))].filter(Boolean).join(' '));
log('requests containing the typed text:', leaked);
await browser.close();
