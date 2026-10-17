// Record the app's real audio output while driving the mock flow, and log levels.
// usage (inside the container): node dev/audio-check.mjs <out.webm>
import { chromium } from '../web/node_modules/playwright-core/index.mjs';
import fs from 'node:fs';

const out = process.argv[2];
const browser = await chromium.launch({
  executablePath: '/usr/bin/google-chrome',
  args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'],
});
const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });
page.on('pageerror', (e) => console.log('[pageerror]', e.message));
page.on('console', (m) => { if (m.type() === 'error') console.log('[console]', m.text()); });
await page.goto('http://localhost:8659/?mock', { waitUntil: 'load' });
await page.waitForTimeout(2000);
await page.mouse.click(5, 400); // first gesture: unlocks audio
await page.waitForTimeout(500);
await page.evaluate(() => {
  const s = window.__neuron.sound;
  const ctx = s.ctx;
  const an = ctx.createAnalyser(); an.fftSize = 2048;
  const dest = ctx.createMediaStreamDestination();
  s.master.connect(an); s.master.connect(dest);
  const rec = new MediaRecorder(dest.stream, { mimeType: 'audio/webm;codecs=opus', audioBitsPerSecond: 192000 });
  const chunks = [];
  rec.ondataavailable = (e) => chunks.push(e.data);
  rec.start(250);
  const buf = new Float32Array(an.fftSize);
  const levels = [];
  const iv = setInterval(() => {
    an.getFloatTimeDomainData(buf);
    let pk = 0, ss = 0;
    for (const v of buf) { pk = Math.max(pk, Math.abs(v)); ss += v * v; }
    levels.push([performance.now(), pk, Math.sqrt(ss / buf.length), document.getElementById('app').dataset.state]);
  }, 100);
  window.__rec = { rec, chunks, levels, iv, state: ctx.state };
});
const mark = (m) => page.evaluate((m) => window.__rec.levels.push([performance.now(), -1, -1, m]), m);
await page.waitForTimeout(6000); await mark('idle-music');
await page.click('#load-btn');
await page.waitForSelector('#prompt:not([hidden])', { timeout: 120000 });
await mark('loaded');
await page.waitForTimeout(3000);
await page.fill('#prompt-input', 'pizza');
await page.keyboard.press('Enter');
await page.waitForSelector('#card:not([hidden])', { timeout: 120000 });
await mark('focused');
await page.waitForTimeout(3000);
await page.click('#explore-btn');
await page.waitForTimeout(9000);
await mark('constellation');
await page.fill('#test-input', 'tango');
await page.keyboard.press('Enter');
await page.waitForTimeout(3000);
await page.click('#again-btn');
await page.waitForTimeout(5000);
await mark('again');
await page.click('#sound-toggle');
await page.waitForTimeout(1500);
await mark('muted');
const res = await page.evaluate(async () => {
  const r = window.__rec;
  clearInterval(r.iv);
  r.rec.stop();
  await new Promise((ok) => setTimeout(ok, 600));
  const blob = new Blob(r.chunks, { type: 'audio/webm' });
  const b = new Uint8Array(await blob.arrayBuffer());
  let s = ''; for (let i = 0; i < b.length; i += 0x8000) s += String.fromCharCode(...b.subarray(i, i + 0x8000));
  return { webm: btoa(s), levels: r.levels, state: r.state };
});
fs.writeFileSync(out, Buffer.from(res.webm, 'base64'));
console.log('ctx state at tap:', res.state, 'bytes:', fs.statSync(out).size);
// per-phase stats (peak and RMS in dBFS) between marks
let seg = [], name = 'start';
const db = (x) => (x > 0 ? (20 * Math.log10(x)).toFixed(1) : '-inf');
const flush = () => {
  if (!seg.length) return;
  const pk = Math.max(...seg.map((l) => l[1]));
  const rms = Math.sqrt(seg.reduce((a, l) => a + l[2] * l[2], 0) / seg.length);
  console.log(`${name.padEnd(14)} peak ${db(pk).padStart(6)} dBFS   rms ${db(rms).padStart(6)} dBFS   (${seg.length} samples)`);
};
for (const l of res.levels) {
  if (l[1] === -1) { name = `→ ${l[3]}`; flush(); seg = []; continue; }
  seg.push(l);
}
flush();
await browser.close();
