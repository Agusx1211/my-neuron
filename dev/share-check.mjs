// Mock-mode check of shared links (?q=) and the share text.
import { chromium } from '../web/node_modules/playwright-core/index.mjs';
const browser = await chromium.launch({ executablePath: '/usr/bin/google-chrome', args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader'] });
const ctx = await browser.newContext({ viewport: { width: 1280, height: 800 }, permissions: ['clipboard-read', 'clipboard-write'] });
const page = await ctx.newPage();
page.on('pageerror', (e) => console.log('[pageerror]', e.message));
await page.goto('http://localhost:8659/?mock&q=pizza', { waitUntil: 'load' });
await page.waitForTimeout(1500);
console.log('url after load:', page.url());
console.log('note:', await page.textContent('#shared-note'));
await page.click('#load-btn');
await page.waitForSelector('#card:not([hidden])', { timeout: 180000 });
console.log('auto-analyzed card:', (await page.textContent('#card-title')).trim());
await page.waitForTimeout(1000);
await page.click('#share-copy');
await page.waitForTimeout(500);
console.log('clipboard:', await page.evaluate(() => navigator.clipboard.readText()));
console.log('toast:', await page.textContent('#toast'));
await browser.close();
