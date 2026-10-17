// Compare live probes (worker path) on a linear and a full-attention layer.
import { chromium } from '../web/node_modules/playwright-core/index.mjs';
import fs from 'node:fs';
const out = process.argv[2];
const browser = await chromium.launch({ executablePath: '/usr/bin/google-chrome', args: ['--enable-unsafe-webgpu', '--use-webgpu-adapter=swiftshader', '--enable-unsafe-swiftshader', '--use-angle=swiftshader'] });
const page = await browser.newPage({ viewport: { width: 1000, height: 700 } });
page.on('pageerror', (e) => console.log('[pageerror]', e.message));
await page.goto('http://localhost:8657/?gguf=%2Fdev-models%2Fq08-Q4_K_M.gguf', { waitUntil: 'load' });
await page.waitForTimeout(3000);
await page.click('label:has(input[name=model][value="qwen3.5-0.8b"])');
await page.click('#load-btn');
await page.waitForSelector('#prompt:not([hidden])', { timeout: 3600000 });
console.log('loaded');
const texts = ['Lionel Messi', 'pizza', 'nostalgia', 'Buenos Aires', 'a rainy Sunday afternoon', 'Diego Maradona', 'tango', 'mate', 'empanadas', 'Patagonia', 'Jorge Luis Borges', 'River Plate', 'Boca Juniors', 'dulce de leche', 'asado', 'Mendoza', 'Rosario', 'Córdoba', 'Malbec', 'gaucho', 'Evita', 'Iguazú Falls', 'La Boca', 'Ushuaia'];
const res = await page.evaluate(async (texts) => {
  const e = window.__neuron.engine;
  const r = {};
  // same order as the app: a full analysis first, then probes
  const a = await e.analyze('Lionel Messi');
  r.analysisTop = [a.top[0].layer, a.top[0].index, a.top[0].act];
  for (const [l, i] of [[3, 100], [3, 2000], [4, 100], [r.analysisTop[0], r.analysisTop[1]]]) r[`${l}:${i}`] = (await e.probe(l, i, 1, texts)).map((x) => [x.text, x.act]);
  return r;
}, texts);
fs.writeFileSync(out, JSON.stringify({ texts, res }));
console.log(JSON.stringify(res));
await browser.close();
