// Render PNG icons from the SVGs in web/public.
import { chromium } from '../web/node_modules/playwright-core/index.mjs';
import fs from 'node:fs';
const dir = 'web/public';
const jobs = [['favicon.svg', 'favicon-32.png', 32], ['icon-full.svg', 'apple-touch-icon.png', 180], ['icon-full.svg', 'icon-192.png', 192], ['icon-full.svg', 'icon-512.png', 512]];
const browser = await chromium.launch({ executablePath: '/usr/bin/google-chrome' });
for (const [src, dst, size] of jobs) {
  const page = await browser.newPage({ viewport: { width: size, height: size } });
  const svg = fs.readFileSync(`${dir}/${src}`, 'utf8');
  await page.setContent(`<html><body style="margin:0;background:transparent">${svg.replace('<svg ', `<svg width="${size}" height="${size}" `)}</body></html>`);
  await page.screenshot({ path: `${dir}/${dst}`, omitBackground: true });
  await page.close();
  console.log(dst, fs.statSync(`${dir}/${dst}`).size);
}
await browser.close();
