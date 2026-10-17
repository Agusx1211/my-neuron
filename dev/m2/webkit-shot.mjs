// Run on the M2: screenshot a page in WebKit (Safari's engine), optionally after loading the mock model.
import { webkit } from 'playwright-core';
const [url, out, flow = 'landing'] = process.argv.slice(2);
const browser = await webkit.launch();
const page = await browser.newPage({ viewport: { width: 1487, height: 928 }, deviceScaleFactor: 2, ignoreHTTPSErrors: true });
page.on('pageerror', (e) => console.log('[pageerror]', e.message));
page.on('console', (m) => { if (m.type() === 'error' || m.type() === 'warning') console.log('[console]', m.type(), m.text().slice(0, 200)); });
await page.goto(url, { waitUntil: 'load' });
await page.waitForTimeout(3000);
console.log(JSON.stringify(await page.evaluate(() => {
  const c = document.querySelector('#stage canvas');
  const s = document.getElementById('stage');
  return { inner: [innerWidth, innerHeight], dpr: devicePixelRatio, stage: [s.clientWidth, s.clientHeight], canvasCss: [c.clientWidth, c.clientHeight], canvasBuf: [c.width, c.height], gl: !!c.getContext && 'webgl' };
})));
await page.screenshot({ path: `${out}-1.png` });
if (flow === 'mock') {
  await page.click('#load-btn');
  await page.waitForSelector('#prompt:not([hidden])', { timeout: 60000 });
  await page.waitForTimeout(2500);
  await page.screenshot({ path: `${out}-2.png` });
  await page.fill('#prompt-input', 'pizza');
  await page.keyboard.press('Enter');
  await page.waitForSelector('#card:not([hidden])', { timeout: 60000 });
  await page.waitForTimeout(3000);
  await page.screenshot({ path: `${out}-3.png` });
}
await browser.close();
