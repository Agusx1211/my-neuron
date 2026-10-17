// Clickable tokens: analyze a phrase, read it at one token, then go back to the whole text.
// usage: node dev/token-check.mjs <outdir> [base-url] [text]   (base-url defaults to the mock)
// On the M2 (real GPU, cached model): PW=playwright-core CHROME=<chrome> CHROME_ARGS=--enable-unsafe-webgpu PROFILE=<dir>
const { chromium } = await import(process.env.PW ?? '../web/node_modules/playwright-core/index.mjs');

const [out, base = 'https://localhost:8655/?mock', text = 'Europe vs USA'] = process.argv.slice(2);
const opts = {
  executablePath: process.env.CHROME ?? '/usr/bin/google-chrome',
  args: (process.env.CHROME_ARGS ?? '--use-angle=swiftshader --enable-unsafe-swiftshader --ignore-gpu-blocklist').split(' '),
};
const view = { viewport: { width: 1440, height: 900 }, ignoreHTTPSErrors: true };
const browser = process.env.PROFILE
  ? await chromium.launchPersistentContext(process.env.PROFILE, { ...opts, ...view })
  : await chromium.launch(opts);
const page = process.env.PROFILE ? (browser.pages()[0] ?? await browser.newPage()) : await browser.newPage(view);
page.on('pageerror', (e) => console.log('[pageerror]', e.message));
page.on('console', (m) => { if (m.type() === 'error') console.log('[console]', m.text().slice(0, 200)); });
const card = () => page.evaluate(() => ({
  eyebrow: document.getElementById('card-eyebrow').textContent,
  title: document.getElementById('card-title').textContent,
  theme: document.getElementById('card-theme')?.textContent,
  z: document.getElementById('stat-z').textContent,
  tokens: [...document.querySelectorAll('#token-strip .tok')].map((t) => (t.classList.contains('reading') ? `[${t.textContent}]` : t.textContent)).join(' '),
  note: document.getElementById('token-note').textContent,
}));
await page.goto(base, { waitUntil: 'load', timeout: 600000 });
await page.waitForTimeout(3000);
if (!base.includes('mock')) await page.click('label:has(input[name=model][value="qwen3.5-4b"])');
await page.click('#load-btn');
await page.waitForSelector('#prompt:not([hidden])', { timeout: 3600000 });
await page.waitForTimeout(2500);
await page.fill('#prompt-input', text);
await page.keyboard.press('Enter');
await page.waitForSelector('#card:not([hidden])', { timeout: 3600000 });
await page.waitForTimeout(3000);
console.log('default', await card());
await page.screenshot({ path: `${out}/tok-1-default.png` });
const n = await page.locator('#token-strip .tok').count();
for (let j = 0; j < n; j++) {
  const t0 = Date.now();
  await page.locator('#token-strip .tok').nth(j).click();
  await page.waitForFunction((j) => document.querySelectorAll('#token-strip .tok')[j]?.getAttribute('aria-pressed') === 'true', j, { timeout: 600000 });
  console.log(`token ${j} (${Date.now() - t0} ms)`, await card());
  if (j === n - 2) {
    await page.waitForTimeout(2500);
    await page.screenshot({ path: `${out}/tok-2-token.png` });
  }
}
await page.click('#token-note .link-btn');
await page.waitForFunction(() => !document.querySelector('#token-strip .tok[aria-pressed="true"]'), null, { timeout: 600000 });
await page.waitForTimeout(2500);
console.log('back', await card());
await page.screenshot({ path: `${out}/tok-3-back.png` });
await browser.close();
