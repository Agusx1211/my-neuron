import fs from 'node:fs';
import { initSync, selectNeurons, f16ToF32 } from '../../web/src/wasm-pkg/neuron_core.js';
initSync({ module: fs.readFileSync(new URL('../../web/src/wasm-pkg/neuron_core_bg.wasm', import.meta.url)) });
const atlas = process.argv[2];
const meta = JSON.parse(fs.readFileSync(`${atlas}/meta.json`));
const lex = JSON.parse(fs.readFileSync(`${atlas}/lexicon.json`));
const stats = f16ToF32(fs.readFileSync(`${atlas}/stats.bin`));
const t8 = fs.readFileSync(`${atlas}/top8.bin`);
const top8 = new Uint16Array(t8.buffer, t8.byteOffset, t8.byteLength / 2);
const sem = f16ToF32(fs.readFileSync(`${atlas}/sem.bin`));
const E = sem.subarray(0, meta.concepts * meta.semDim);
for (let i = 0; fs.existsSync(`dev/sel/ext${i}.bin`); i++) {
  const rd = (f) => { const b = fs.readFileSync(f); return new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength)); };
  const ext = rd(`dev/sel/ext${i}.bin`);
  const q = rd(`dev/sel/q${i}.bin`);
  const t0 = performance.now();
  const r = selectNeurons(ext, stats, meta.neurons, new Float32Array(meta.floorPos), new Float32Array(meta.floorNeg), top8, E, q, meta.minLayer, 6, true, 3, 3);
  const ms = performance.now() - t0;
  const rows = [];
  for (let j = 0; j < r.length; j += 7) rows.push(`L${r[j]} N${r[j + 1]} ${r[j + 2] > 0 ? '+' : '-'} score=${r[j + 3].toFixed(2)} z=${r[j + 4].toFixed(1)}`);
  console.log(i, `${ms.toFixed(0)}ms`, rows.join(' | '));
}
