import './styles/main.css';
import * as THREE from 'three';
import { MODELS, type Analysis, type Engine, type ModelInfo, type ModelSpec, type NeuronHit } from './engine/api';
import { MockEngine } from './engine/mock';
import { WorkerEngine } from './engine/client';
import { Stage } from './viz/scene';
import { Tower, type TowerDims } from './viz/tower';
import { Hero } from './viz/hero';
import { Constellation } from './viz/constellation';
import { flyTo, type Shot } from './viz/camera';
import { Landing, PREVIEW_DIMS, defaultModel } from './ui/landing';
import { Prompt } from './ui/prompt';
import { Card } from './ui/card';
import { flyTokens } from './ui/flight';
import { $, conceal, fmtNum, reveal, sleep, toast } from './ui/dom';
import { sound } from './audio/sound';
import { initAnalytics, track } from './analytics';

// a shared link (?q=...) replays someone's neuron; strip the text from the URL
// before analytics loads so it never gets reported
const sharedQuery = new URLSearchParams(location.search).get('q')?.trim().slice(0, 160) || null;
if (sharedQuery) {
  const u = new URL(location.href);
  u.searchParams.delete('q');
  history.replaceState(null, '', u);
}
initAnalytics();

// `?mock` runs the whole experience against fake data (no GPU, no download).
const engine: Engine = new URLSearchParams(location.search).has('mock') ? new MockEngine() : new WorkerEngine();

type State = 'landing' | 'loading' | 'ready' | 'analyzing' | 'focused';

const app = $('app');
const stage = new Stage($('stage'));
if (stage.reducedMotion) stage.tweens.speed = 0.35;

const dimsFor = (m: ModelSpec): TowerDims => {
  const d = PREVIEW_DIMS[m.id] ?? PREVIEW_DIMS['qwen3.5-4b'];
  return { layers: d.layers, neurons: d.neurons, layerTypes: Array.from({ length: d.layers }, (_, i) => ((i + 1) % 4 === 0 ? 'full' : 'linear')) };
};

const tower = new Tower(stage, dimsFor(defaultModel(MODELS)));
const hero = new Hero(stage);
const constellation = new Constellation(stage);

let state: State = 'landing';
let info: ModelInfo | null = null;
let analysis: Analysis | null = null;
let hitIdx = 0;
let probeGen = 0;

function setState(s: State) {
  state = s;
  app.dataset.state = s;
}
// read through a function so TS doesn't keep stale narrowing across awaits
const now = () => state;

const isNarrow = () => innerWidth <= 760;

/** Where the subject should sit on screen for each state. */
function layoutShift(instant = false) {
  const { w, h } = stage.size;
  if (state === 'landing' || state === 'loading') {
    stage.setShift(isNarrow() ? 0 : w * 0.2, isNarrow() ? -h * 0.18 : 0, instant);
  } else if (state === 'focused') {
    if (isNarrow()) stage.setShift(0, -h * 0.25, instant);
    else stage.setShift(-(Math.min(400, w - 40) + 20) / 2, 0, instant);
  } else {
    stage.setShift(0, isNarrow() ? -h * 0.06 : -h * 0.04, instant);
  }
}
window.addEventListener('resize', () => layoutShift());

function overviewShot(): Shot {
  const aspect = stage.size.w / stage.size.h;
  const dist = (tower.height * 1.15 + 22) * 1.1 * Math.max(1, 0.62 / aspect);
  const az = Math.atan2(stage.camera.position.x, stage.camera.position.z) || 0.7;
  const el = 0.36;
  return {
    position: new THREE.Vector3(Math.sin(az) * Math.cos(el) * dist, Math.sin(el) * dist, Math.cos(az) * Math.cos(el) * dist),
    target: new THREE.Vector3(0, 0, 0),
    fov: 45,
  };
}

function focusShot(p: THREE.Vector3, back = 1): Shot {
  if (back === 1 && isNarrow()) back = 1.35;
  const radial = new THREE.Vector3(p.x, 0, p.z);
  if (radial.lengthSq() < 0.5) radial.set(stage.camera.position.x, 0, stage.camera.position.z);
  radial.normalize();
  const tangent = new THREE.Vector3(-radial.z, 0, radial.x);
  const pos = p.clone()
    .addScaledVector(radial, 2.8 * back)
    .addScaledVector(tangent, 0.9 * back)
    .add(new THREE.Vector3(0, 3.0 * back, 0));
  return { position: pos, target: p.clone(), fov: back > 1 ? 42 : 38 };
}

/** Final world position of a neuron once its layer is the focused one. */
function focusedPosition(hit: NeuronHit) {
  return tower.group.localToWorld(tower.localPosition(hit.layer, hit.index, 0));
}

// ---------- landing & loading ----------

/** Keep the "How it works" numbers in sync with the chosen model. */
function describeModel(name: string, layers: number, neurons: number) {
  const params: Record<string, string> = { 'Qwen3.5-4B': '4-billion', 'Qwen3.5-0.8B': '0.8-billion' };
  const set = (k: string, v: string) => document.querySelectorAll(`[data-hw="${k}"]`).forEach((el) => { el.textContent = v; });
  set('name', name);
  set('params', params[name] ?? 'multi-billion');
  set('layers', String(layers));
  set('per', fmtNum(neurons));
  set('total', fmtNum(layers * neurons));
}

const landing = new Landing(
  MODELS,
  (m) => {
    if (state !== 'landing') return;
    const d = dimsFor(m);
    tower.build(d);
    describeModel(m.name, d.layers, d.neurons);
  },
  (m) => void loadModel(m),
);

async function loadModel(m: ModelSpec) {
  if (state !== 'landing') return;
  setState('loading');
  landing.showLoading(m);
  const layers = dimsFor(m).layers;
  tower.setLayersReady(0);
  sound.whoosh('up', 2.2);
  sound.setMood('loading');
  let heard = 0;
  let fromCache = false;
  const t0 = performance.now();
  track('model_load_start', { model: m.id });
  try {
    info = await engine.load(m, (p) => {
      landing.progress(p, layers);
      tower.setLayersReady(p.layersReady);
      if (p.phase === 'cached') fromCache = true;
      sound.progress(p.total ? p.loaded / p.total : 0);
      // every layer that lands on the GPU plays its note
      for (; heard < Math.min(p.layersReady, layers); heard++) sound.layer(heard, layers, (heard + 1) % 4 === 0);
    });
  } catch (e) {
    console.error(e);
    toast(`Couldn't load the model: ${(e as Error).message ?? e}`);
    track('model_load_error', { model: m.id, message: String((e as Error).message ?? e).slice(0, 100) });
    sound.error();
    sound.setMood('idle');
    setState('landing');
    $('landing-body').hidden = false;
    $('loading-body').hidden = true;
    return;
  }
  if (info.layers !== tower.dims.layers || info.neurons !== tower.dims.neurons) {
    tower.build({ layers: info.layers, neurons: info.neurons, layerTypes: info.layerTypes });
  }
  tower.setLayersReady(info.layers);
  describeModel(info.name, info.layers, info.neurons);
  const badge = $('model-badge');
  badge.textContent = `${info.name} · ${fmtNum(info.layers * info.neurons)} neurons`;
  badge.title = info.adapter;
  badge.hidden = false;
  sound.bloom();
  sound.setMood('ready');
  track('model_load_complete', { model: m.id, seconds: Math.round((performance.now() - t0) / 1000), from_cache: fromCache });
  await sleep(700);
  await landing.hide();
  setState('ready');
  layoutShift();
  flyTo(stage.tweens, stage.camera, stage.controls, overviewShot(), { duration: 1800 }).then(() => {
    if (now() === 'ready') stage.controls.autoRotate = !stage.reducedMotion;
  });
  prompt.show();
  if (sharedQuery) {
    // replay the shared neuron once the camera settles
    await sleep(1900);
    if (now() === 'ready') {
      $<HTMLInputElement>('prompt-input').value = sharedQuery;
      track('shared_open');
      void analyze(sharedQuery);
    }
  }
}

// ---------- analysis ----------

const prompt = new Prompt((text) => void analyze(text));
const status = $('status');
const say = (msg: string) => { status.textContent = msg; status.hidden = false; };

async function analyze(text: string) {
  if (state !== 'ready' || !info) return;
  setState('analyzing');
  sound.setMood('analyzing');
  const from = prompt.inputRect;
  const pending = engine.analyze(text);
  await prompt.hide();
  say(`Running “${text.length > 28 ? text.slice(0, 27) + '…' : text}” through ${info.layers} layers on your GPU…`);
  let a: Analysis;
  try {
    a = await pending;
  } catch (e) {
    console.error(e);
    status.hidden = true;
    toast(`Analysis failed: ${(e as Error).message ?? e}`);
    track('analyze_error', { model: info.id, message: String((e as Error).message ?? e).slice(0, 100) });
    sound.error();
    sound.setMood('ready');
    setState('ready');
    prompt.show();
    return;
  }
  if (!a.top.length) {
    status.hidden = true;
    toast('No neuron stood out for that text, try something else.');
    track('analyze_no_hit', { model: info.id, tokens: a.tokens.length });
    sound.error();
    sound.setMood('ready');
    setState('ready');
    prompt.show();
    return;
  }
  analysis = a;
  stage.controls.autoRotate = false;
  const base = stage.toScreen(tower.basePosition());
  sound.tokens(Math.min(a.tokens.length, 24), stage.reducedMotion ? 0 : 55);
  await flyTokens(a.tokens.map((t) => t.text), from, base, stage.reducedMotion);

  tower.setHeat(a.heat);
  tower.setHits(a.top.slice(0, 24));
  say(`Scoring ${fmtNum(info.layers * info.neurons)} neurons against their baselines…`);
  sound.riser(1.9, info.layers);
  await tower.pulse(1900);

  const win = a.top[0];
  // counts and timings only: the text itself never leaves the device
  track('analyze', { model: info.id, tokens: a.tokens.length, forward_ms: Math.round(a.ms), layer: win.layer, pole: win.pole, z: Math.round(win.z) });
  say(`Found it: layer ${win.layer}, neuron #${fmtNum(win.index)} (${win.z.toFixed(1)}σ)`);
  tower.showHits(true);
  hitIdx = 0;
  const p = focusedPosition(win);
  tower.focus(win.layer, win.index, 2000);
  setState('focused');
  layoutShift();
  sound.whoosh('up', 2.6);
  await flyTo(stage.tweens, stage.camera, stage.controls, focusShot(p), { duration: 2600, arc: 0.3 });
  sound.impact(win.pole);
  sound.humOn(win.layer, win.index, win.pole);
  sound.setMood('focused', { dark: win.pole < 0 });
  await hero.show(p, win.pole);
  status.hidden = true;
  card.show(a, info);
  reveal($('legend'));
}

// ---------- card interactions ----------

const card = new Card({
  onExplore: () => void explore(),
  onAgain: () => void again(),
  onRunner: (i) => void selectHit(i),
  onTest: (text) => void testConcept(text),
});

async function selectHit(i: number) {
  if (!analysis || state !== 'focused') return;
  probeGen++;
  hitIdx = i;
  track('runner_select', { rank: i });
  const hit = analysis.top[i];
  card.setHit(i);
  constellation.hide();
  const p = focusedPosition(hit);
  tower.focus(hit.layer, hit.index, 1500);
  hero.moveTo(p, hit.pole, 2000);
  sound.whoosh('side', 1.8);
  sound.humOn(hit.layer, hit.index, hit.pole);
  sound.setMood('focused', { dark: hit.pole < 0 });
  await flyTo(stage.tweens, stage.camera, stage.controls, focusShot(p), { duration: 2000, arc: 0.25 });
}

async function explore() {
  if (!analysis || state !== 'focused' || constellation.isOpen) return;
  const gen = ++probeGen;
  const hit = analysis.top[hitIdx];
  card.exploring('loading');
  track('explore', { layer: hit.layer, rank: hitIdx });
  let atlas;
  try {
    atlas = await engine.atlas(hit.layer, hit.index);
  } catch (e) {
    toast(`Couldn't read the atlas: ${(e as Error).message ?? e}`);
    card.setHit(hitIdx);
    return;
  }
  if (gen !== probeGen) return;
  // the user's own text is shown as the "you" node, so drop it from the atlas lists
  const mine = analysis.text.trim().toLowerCase();
  const notMine = (c: { text: string }) => c.text.toLowerCase() !== mine;
  const primary = (hit.pole > 0 ? atlas.positive : atlas.negative).filter(notMine);
  const opposite = (hit.pole > 0 ? atlas.negative : atlas.positive).filter(notMine);
  const center = hero.position.clone();
  // pull back far enough that the whole constellation fits the visible area
  const aspect = stage.size.w / stage.size.h;
  const back = 2.6 * Math.pow(Math.max(1, 1.25 / aspect), 0.8);
  await flyTo(stage.tweens, stage.camera, stage.controls, focusShot(center, back), { duration: 1500, arc: 0.1 });
  if (gen !== probeGen) return;
  constellation.show({ center, pole: hit.pole, primary: primary.slice(0, 24), opposite: opposite.slice(0, 10), you: { text: analysis.text, act: hit.act } });
  // each concept chimes as it flies in (same stagger as the constellation), stronger = higher
  sound.setMood('constellation', { dark: hit.pole < 0 });
  const peak = Math.max(1e-6, ...primary.slice(0, 24).map((c) => Math.abs(c.act)));
  primary.slice(0, 24).forEach((c, i) => sound.concept(Math.abs(c.act) / peak, i * 70 + 120));
  opposite.slice(0, 10).forEach((_, i) => sound.concept(0.05, 900 + i * 40 + 120));
  card.exploring('open');
  card.startVerify(primary, hit.pole, { text: analysis.text, act: hit.act });

  const texts = primary.slice(0, 24).map((c) => c.text);
  const listed = Math.min(16, texts.length);
  let done = 0;
  await engine.probe(hit.layer, hit.index, hit.pole, texts, (r) => {
    if (gen !== probeGen) return;
    constellation.setLive(r.text, r.act);
    card.liveResult(r);
    sound.verify(Math.abs(r.act) / peak);
    if (texts.indexOf(r.text) < listed) card.verifyStatus(++done, listed);
  });
  if (gen !== probeGen) return;
  const oppTexts = opposite.slice(0, 10).map((c) => c.text);
  await engine.probe(hit.layer, hit.index, hit.pole > 0 ? -1 : 1, oppTexts, (r) => {
    if (gen === probeGen) constellation.setLive(r.text, r.act);
  });
}

async function testConcept(text: string) {
  if (!analysis || state !== 'focused') return;
  const hit = analysis.top[hitIdx];
  const gen = probeGen;
  if (!constellation.isOpen) void explore();
  const [r] = await engine.probe(hit.layer, hit.index, hit.pole, [text]);
  if (!r) return;
  card.testResult(r);
  sound.result(r.z >= 3);
  track('test_concept', { fires: r.z >= 3 });
  // wait for the constellation if it is still opening
  for (let i = 0; i < 40 && !constellation.isOpen; i++) await sleep(100);
  if (gen === probeGen || constellation.isOpen) constellation.addUser(r.text, r.act, hit.pole);
}

async function again() {
  if (state !== 'focused') return;
  probeGen++;
  setState('ready');
  sound.whoosh('down', 2);
  sound.humOff();
  track('try_another');
  sound.setMood('ready');
  constellation.hide();
  card.hide();
  conceal($('legend'), 400);
  hero.hide();
  tower.showHits(false, 600);
  tower.unfocus(1400);
  tower.clearHeat(1200);
  layoutShift();
  prompt.show();
  await flyTo(stage.tweens, stage.camera, stage.controls, overviewShot(), { duration: 2200, arc: 0.2 });
  if (now() === 'ready') stage.controls.autoRotate = !stage.reducedMotion;
}

$('brand').addEventListener('click', () => void again());

// ---------- how it works ----------

const howto = $('howto');
const openHowto = () => { track('howto_open'); howto.hidden = false; howto.querySelector<HTMLButtonElement>('.modal-close')?.focus(); };
const closeHowto = () => { howto.hidden = true; $('howto-open').focus(); };
$('howto-open').addEventListener('click', openHowto);
howto.querySelectorAll('[data-close]').forEach((el) => el.addEventListener('click', closeHowto));
window.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && !howto.hidden) closeHowto();
});

// ---------- sharing ----------

function shareData() {
  if (!analysis || !info) return null;
  const hit = analysis.top[hitIdx];
  const mine = analysis.text.trim().toLowerCase();
  const also = (hit.preview ?? []).filter((p) => p.toLowerCase() !== mine).slice(0, 3);
  const list = also.length > 1 ? `${also.slice(0, -1).join(', ')} and ${also[also.length - 1]}` : also[0];
  const url = `${location.origin}${location.pathname}?q=${encodeURIComponent(analysis.text)}`;
  const text = `“${analysis.text}” lights up layer ${hit.layer}, neuron #${fmtNum(hit.index)} of ${info.name}`
    + (list ? `. It also fires for ${list}.` : '.')
    + ' Find yours, it runs entirely in your browser:';
  return { text, url };
}

const nativeShare = typeof navigator.share === 'function' && matchMedia('(pointer: coarse)').matches;
$('share-native').hidden = !nativeShare;
$('share-x').hidden = nativeShare;
$('share-copy').hidden = nativeShare;
$('share-native').addEventListener('click', () => {
  const d = shareData();
  if (!d) return;
  track('share', { method: 'native' });
  navigator.share({ title: 'My Neuron', text: d.text, url: d.url }).catch(() => {});
});
$('share-x').addEventListener('click', () => {
  const d = shareData();
  if (!d) return;
  track('share', { method: 'x' });
  open(`https://x.com/intent/post?text=${encodeURIComponent(d.text)}&url=${encodeURIComponent(d.url)}`, '_blank', 'noopener,width=600,height=520');
});
$('share-copy').addEventListener('click', async () => {
  const d = shareData();
  if (!d) return;
  track('share', { method: 'copy' });
  try {
    await navigator.clipboard.writeText(`${d.text} ${d.url}`);
    toast('Copied, paste it anywhere');
  } catch {
    toast(d.url);
  }
});

if (sharedQuery) {
  const note = $('shared-note');
  note.hidden = false;
  note.append('Someone shared their neuron for ', Object.assign(document.createElement('b'), { textContent: `“${sharedQuery}”` }), '. Load the model to see it light up.');
}

// ---------- sound ----------

const soundBtn = $<HTMLButtonElement>('sound-toggle');
const syncSoundBtn = (on: boolean) => {
  soundBtn.setAttribute('aria-pressed', String(on));
  soundBtn.setAttribute('aria-label', on ? 'Mute sound' : 'Unmute sound');
};
syncSoundBtn(sound.on);
sound.onChange(syncSoundBtn);
soundBtn.addEventListener('click', () => { sound.toggle(); track('sound_toggle', { on: sound.on }); });
// browsers only allow audio after a gesture: the first click or key starts the soundtrack
for (const ev of ['pointerdown', 'keydown'] as const) window.addEventListener(ev, () => sound.unlock(), { capture: true });
document.addEventListener('click', (e) => {
  const b = (e.target as HTMLElement).closest('button');
  if (b && b !== soundBtn) sound.tick();
});
window.addEventListener('keydown', (e) => {
  const typing = e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement;
  if (!typing && !e.metaKey && !e.ctrlKey && !e.altKey && e.key.toLowerCase() === 'm') sound.toggle();
});

// ---------- boot ----------

async function boot() {
  setState('landing');
  const shot = overviewShot();
  stage.camera.position.copy(shot.position);
  stage.controls.target.copy(shot.target);
  stage.controls.update();
  layoutShift(true);
  const support = await engine.init();
  if (!support.webgpu) landing.noWebGPU(support.reason ?? '');
  track('webgpu_check', { supported: support.webgpu, adapter: (support.adapter ?? support.reason ?? '').slice(0, 80) });
  for (const m of MODELS) engine.isCached(m).then((c) => landing.setCached(m.id, c)).catch(() => {});
}

void boot();

// handy for debugging from the console
Object.assign(window as unknown as Record<string, unknown>, { __neuron: { stage, tower, hero, constellation, engine, sound } });
