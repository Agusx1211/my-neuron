// Run on the M2: confirm headless Chrome exposes the Apple GPU through WebGPU.
import { chromium } from 'playwright-core';
const browser = await chromium.launch({
  executablePath: `${process.env.HOME}/my-neuron-gpu/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing`,
  args: ['--enable-unsafe-webgpu'],
});
const page = await browser.newPage({ ignoreHTTPSErrors: true });
await page.goto(process.argv[2], { waitUntil: 'load' });
console.log(await page.evaluate(async () => {
  const a = await navigator.gpu?.requestAdapter();
  return a ? JSON.stringify({ ...a.info, maxBuf: a.limits.maxBufferSize, maxBind: a.limits.maxStorageBufferBindingSize }) : 'no adapter';
}));
await browser.close();
