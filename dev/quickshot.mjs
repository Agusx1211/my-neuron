// Quick mock screenshots of a couple of states (landing, ready, focused).
import { chromium } from '../web/node_modules/playwright-core/index.mjs';
const out = process.argv[2];
const browser = await chromium.launch({ executablePath: '/usr/bin/google-chrome', args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] });
const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
page.on('pageerror', (e) => console.log('[pageerror]', e.message));
await page.goto('http://localhost:8655/?mock', { waitUntil: 'load' });
await page.waitForTimeout(5000);
await page.screenshot({ path: `${out}/q-1-landing.png` });
await page.click('#load-btn');
await page.waitForSelector('#prompt:not([hidden])', { timeout: 120000 });
await page.waitForTimeout(3500);
await page.screenshot({ path: `${out}/q-2-ready.png` });
await page.fill('#prompt-input', 'pizza');
await page.keyboard.press('Enter');
await page.waitForSelector('#card:not([hidden])', { timeout: 120000 });
await page.waitForTimeout(4000);
await page.screenshot({ path: `${out}/q-3-focused.png` });
await browser.close();
