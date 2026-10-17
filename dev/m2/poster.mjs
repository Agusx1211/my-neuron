// Drive the real app (4B model, real atlas) to capture README screenshots and
// the 1200x630 social preview image.
// M2 variant: real Apple GPU via Metal, persistent profile (OPFS cache), 2x screenshots.
// usage (on the M2, in ~/my-neuron-gpu): node poster.mjs <outdir> <base-url> [text]
import { chromium } from 'playwright-core';

const [out, base, text = 'Lionel Messi'] = process.argv.slice(2);
const browser = await chromium.launchPersistentContext(`${process.env.HOME}/my-neuron-gpu/profile`, {
  executablePath: `${process.env.HOME}/my-neuron-gpu/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing`,
  args: ['--enable-unsafe-webgpu'],
  viewport: { width: 1600, height: 900 },
  deviceScaleFactor: 2,
  ignoreHTTPSErrors: true,
});
const page = browser.pages()[0] ?? await browser.newPage();
const t0 = Date.now();
const log = (...a) => console.log(`[${((Date.now() - t0) / 1000).toFixed(0)}s]`, ...a);
page.on('pageerror', (e) => log('[pageerror]', e.message));
page.on('console', (m) => { if (m.type() === 'error') log('[console]', m.text().slice(0, 200)); });
const LONG = 3 * 60 * 60 * 1000;
const shot = async (name, opts = {}) => {
  try { await page.screenshot({ path: `${out}/${name}`, timeout: 900000, ...opts }); log('shot', name); }
  catch (e) { log('shot failed', name, e.message.split('\n')[0]); }
};

await page.goto(base, { waitUntil: 'load', timeout: LONG });
await page.waitForTimeout(3000);
await page.click('label:has(input[name=model][value="qwen3.5-4b"])');
const tLoad = Date.now();
await page.click('#load-btn');
await page.waitForSelector('#prompt:not([hidden])', { timeout: LONG });
log(`model ready in ${((Date.now() - tLoad) / 1000).toFixed(1)}s`, await page.evaluate(() => document.getElementById('model-badge')?.title));
await page.waitForTimeout(3000);
await shot('ready.png');
await page.fill('#prompt-input', text);
await page.keyboard.press('Enter');
await page.waitForSelector('#card:not([hidden])', { timeout: LONG });
log('card shown');
await page.waitForTimeout(6000);
log('CARD:\n' + await page.evaluate(() => document.getElementById('card').innerText));
await shot('focused.png');
await page.click('#explore-btn');
await page.waitForFunction(() => /Verified/i.test(document.getElementById('verify-status')?.textContent ?? ''), null, { timeout: LONG, polling: 2000 });
await page.waitForTimeout(5000);
await shot('constellation.png');
log('VERIFY:\n' + await page.evaluate(() => document.getElementById('verify-section')?.innerText));
// timings straight from the engine
log('TIMING', await page.evaluate(async () => {
  const e = window.__neuron.engine;
  let t = performance.now();
  const a = await e.analyze('Lionel Messi');
  const analyzeMs = performance.now() - t;
  const atlas = await e.atlas(a.top[0].layer, a.top[0].index);
  t = performance.now();
  const r = await e.probe(a.top[0].layer, a.top[0].index, 1, atlas.positive.slice(0, 24).map((c) => c.text));
  return JSON.stringify({ analyzeMs: Math.round(analyzeMs), forwardMs: Math.round(a.ms), probe24Ms: Math.round(performance.now() - t), sample: r.slice(0, 4).map((x) => [x.text, +x.act.toFixed(2)]) });
}));

// social preview: hide the app chrome, center the subject, add a title overlay
await page.addStyleTag({ content: `
  .topbar, #card, .legend, #status, #toast, #prompt { display: none !important; }
  #og { position: fixed; inset: 0; z-index: 999; pointer-events: none; font-family: Inter, system-ui, sans-serif; color: #eceefb; }
  #og .shade { position: absolute; inset: 0; background:
      linear-gradient(180deg, rgba(4,5,10,0) 52%, rgba(4,5,10,0.78) 82%, rgba(4,5,10,0.92) 100%),
      radial-gradient(120% 90% at 50% 40%, rgba(4,5,10,0) 55%, rgba(4,5,10,0.55) 100%); }
  #og .tag { position: absolute; top: 34px; left: 40px; display: flex; gap: 10px; align-items: center;
      font: 600 15px/1 'JetBrains Mono', monospace; letter-spacing: 0.08em; color: #b9bddb; text-transform: uppercase; }
  #og .tag i { width: 10px; height: 10px; border-radius: 50%; background: #ffb547; box-shadow: 0 0 14px #ffb547; }
  #og .find { position: absolute; top: 30px; right: 40px; padding: 9px 14px; border-radius: 999px;
      background: rgba(9,11,22,0.78); border: 1px solid rgba(255,181,71,0.45); font: 500 15px/1.2 Inter, sans-serif; }
  #og .find b { color: #ffd88a; font-weight: 600; }
  #og .title { position: absolute; left: 0; right: 0; bottom: 92px; text-align: center; font-weight: 800;
      font-size: 92px; letter-spacing: -0.035em; line-height: 1; }
  #og .title span { background: linear-gradient(90deg, #ffd27a, #ffb547 40%, #c9a7ff); -webkit-background-clip: text; color: transparent; }
  #og .sub { position: absolute; left: 0; right: 0; bottom: 44px; text-align: center; font-size: 25px; color: #d4d7f0; }
  #og .sub em { color: #ffd88a; font-style: normal; font-weight: 600; }
` });
await page.evaluate((text) => {
  const card = document.getElementById('card-title')?.textContent ?? '';
  const el = document.createElement('div');
  el.id = 'og';
  el.innerHTML = `<div class="shade"></div>
    <div class="tag"><i></i>Qwen3.5-4B · 294,912 neurons · in your browser</div>
    <div class="find">“${text}” → <b>${card.replace('·', ' · ')}</b></div>
    <div class="title">My <span>Neuron</span></div>
    <div class="sub">Find the neuron inside a real AI that lights up for <em>you</em></div>`;
  document.body.appendChild(el);
}, text);
await page.setViewportSize({ width: 1200, height: 630 });
await page.waitForTimeout(3000);
await page.evaluate(() => {
  const st = window.__neuron.stage;
  // centered and lifted above the title; freeze the app's own layout shifts
  st.setShift(0, -95, true);
  st.setShift = () => {};
  const tgt = st.controls.target;
  st.camera.position.sub(tgt).multiplyScalar(1.22).add(tgt);
  st.controls.update();
});
await page.waitForTimeout(6000);
await shot('og.png');
await browser.close();
log('done');
