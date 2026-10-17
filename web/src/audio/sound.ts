// Generative soundtrack + sound effects, synthesized live with Web Audio (no
// audio files). Everything is tuned to one harmonic world centered on D: a lush
// lydian/major palette by default and a darker minor one for negative poles.

export type Mood = 'idle' | 'loading' | 'ready' | 'analyzing' | 'focused' | 'constellation';

const mtof = (m: number) => 440 * Math.pow(2, (m - 69) / 12);
const rand = (a: number, b: number) => a + Math.random() * (b - a);

// pad chords (MIDI), voiced low and open
const BRIGHT = [
  [50, 57, 61, 64, 66], // Dmaj9
  [47, 54, 57, 62, 64], // Bm11
  [43, 50, 54, 59, 61], // Gmaj7#11
  [45, 52, 59, 61, 66], // A6/9
];
const DARK = [
  [50, 57, 60, 64, 65], // Dm9
  [46, 53, 57, 62, 65], // Bbmaj7
  [43, 50, 53, 58, 62], // Gm7
  [45, 52, 55, 60, 62], // Am7(11)
];
// pentatonic melodies over those palettes
const PENTA_BRIGHT = [0, 2, 4, 7, 9];  // D E F# A B
const PENTA_DARK = [0, 3, 5, 7, 10];   // D F G A C
const scaleNote = (dark: boolean, degree: number, base = 62) => {
  const p = dark ? PENTA_DARK : PENTA_BRIGHT;
  const oct = Math.floor(degree / p.length);
  return base + 12 * oct + p[((degree % p.length) + p.length) % p.length];
};

interface MoodParams { pad: number; cutoff: number; sparkle: number; sub: number; wind: number; chordSec: number }
const MOODS: Record<Mood, MoodParams> = {
  idle:          { pad: 0.55, cutoff: 650,  sparkle: 0.22, sub: 0.03, wind: 0.020, chordSec: 14 },
  loading:       { pad: 0.55, cutoff: 800,  sparkle: 0.15, sub: 0.03, wind: 0.026, chordSec: 12 },
  ready:         { pad: 0.62, cutoff: 950,  sparkle: 0.30, sub: 0.03, wind: 0.020, chordSec: 12 },
  analyzing:     { pad: 0.75, cutoff: 1700, sparkle: 0.05, sub: 0.04, wind: 0.035, chordSec: 9 },
  focused:       { pad: 0.62, cutoff: 1150, sparkle: 0.35, sub: 0.04, wind: 0.018, chordSec: 11 },
  constellation: { pad: 0.58, cutoff: 1350, sparkle: 0.65, sub: 0.03, wind: 0.018, chordSec: 10 },
};

const STORE_KEY = 'my-neuron:sound';
const MASTER = 1.8; // ≈ -20 LUFS integrated over a session (measured with dev/audio-check.mjs)

interface PadVoice { gain: GainNode; nodes: AudioScheduledSourceNode[] }
interface Hum { oscs: OscillatorNode[]; gain: GainNode; lfo: OscillatorNode }

class Sound {
  private ctx: AudioContext | null = null;
  private master!: GainNode;
  private music!: GainNode;
  private sfx!: GainNode;
  private reverbIn!: GainNode;
  private delayIn!: GainNode;
  private padBus!: GainNode;
  private padFilter!: BiquadFilterNode;
  private sub!: OscillatorNode;
  private subGain!: GainNode;
  private windGain!: GainNode;
  private noise!: AudioBuffer;
  private pads: PadVoice[] = [];
  private hum: Hum | null = null;
  private chordIdx = 0;
  private chordTimer = 0;
  private sparkleTimer = 0;
  private mood: Mood = 'idle';
  private dark = false;
  private nextLayerAt = 0;
  private nextVerifyAt = 0;
  private enabled = true;
  private listeners = new Set<(on: boolean) => void>();

  constructor() {
    try { this.enabled = localStorage.getItem(STORE_KEY) !== 'off'; } catch { /* storage blocked */ }
    document.addEventListener('visibilitychange', () => {
      if (!this.ctx) return;
      if (document.hidden) void this.ctx.suspend();
      else if (this.enabled) void this.ctx.resume();
    });
  }

  get on() { return this.enabled; }

  onChange(fn: (on: boolean) => void) { this.listeners.add(fn); }

  /** Call from any user gesture: creates/resumes the audio graph (autoplay policy). */
  unlock() {
    if (!this.enabled) return;
    if (!this.ctx) this.build();
    if (this.ctx!.state !== 'running') void this.ctx!.resume();
  }

  toggle() {
    this.enabled = !this.enabled;
    try { localStorage.setItem(STORE_KEY, this.enabled ? 'on' : 'off'); } catch { /* ignore */ }
    if (this.enabled) {
      this.unlock();
      const t = this.ctx!.currentTime;
      this.master.gain.cancelScheduledValues(t);
      this.master.gain.setTargetAtTime(MASTER, t, 0.3);
      this.bell(mtof(scaleNote(this.dark, 7)), t + 0.05, 0.05, 2.5);
    } else if (this.ctx) {
      const ctx = this.ctx;
      this.master.gain.setTargetAtTime(0, ctx.currentTime, 0.08);
      setTimeout(() => { if (!this.enabled) void ctx.suspend(); }, 500);
    }
    this.listeners.forEach((fn) => fn(this.enabled));
  }

  private get live() { return !!this.ctx && this.enabled && this.ctx.state === 'running'; }

  // --- graph -------------------------------------------------------------------

  private build() {
    const ctx = new AudioContext({ latencyHint: 'interactive' });
    this.ctx = ctx;
    const comp = ctx.createDynamicsCompressor();
    comp.threshold.value = -16; comp.knee.value = 20; comp.ratio.value = 4; comp.attack.value = 0.01; comp.release.value = 0.35;
    comp.connect(ctx.destination);
    this.master = ctx.createGain();
    this.master.gain.value = 0;
    this.master.gain.setTargetAtTime(MASTER, ctx.currentTime, 0.8);
    this.master.connect(comp);
    this.music = ctx.createGain(); this.music.gain.value = 0.8; this.music.connect(this.master);
    this.sfx = ctx.createGain(); this.sfx.gain.value = 1.0; this.sfx.connect(this.master);

    // long dark reverb from a synthesized impulse response
    const reverb = ctx.createConvolver();
    reverb.buffer = this.impulse(4.8);
    const wetTone = ctx.createBiquadFilter(); wetTone.type = 'lowpass'; wetTone.frequency.value = 4200;
    const wet = ctx.createGain(); wet.gain.value = 0.55;
    this.reverbIn = ctx.createGain();
    this.reverbIn.connect(reverb); reverb.connect(wetTone); wetTone.connect(wet); wet.connect(this.master);

    // ping-pong delay with a darkening feedback loop
    this.delayIn = ctx.createGain(); this.delayIn.gain.value = 0.5;
    const dl = ctx.createDelay(2); dl.delayTime.value = 0.43;
    const dr = ctx.createDelay(2); dr.delayTime.value = 0.29;
    const fb = ctx.createGain(); fb.gain.value = 0.42;
    const fbTone = ctx.createBiquadFilter(); fbTone.type = 'lowpass'; fbTone.frequency.value = 2600;
    const pl = ctx.createStereoPanner(); pl.pan.value = -0.65;
    const pr = ctx.createStereoPanner(); pr.pan.value = 0.65;
    this.delayIn.connect(dl); dl.connect(pl); pl.connect(this.master); pl.connect(this.reverbIn);
    dl.connect(fbTone); fbTone.connect(fb); fb.connect(dr); dr.connect(pr); pr.connect(this.master); dr.connect(dl);

    this.noise = ctx.createBuffer(1, ctx.sampleRate * 2, ctx.sampleRate);
    const nd = this.noise.getChannelData(0);
    for (let i = 0; i < nd.length; i++) nd[i] = Math.random() * 2 - 1;

    // pad bus: warm lowpass that breathes slowly
    this.padFilter = ctx.createBiquadFilter(); this.padFilter.type = 'lowpass'; this.padFilter.Q.value = 0.6;
    this.padFilter.frequency.value = MOODS.idle.cutoff;
    const lfo = ctx.createOscillator(); lfo.frequency.value = 0.045;
    const lfoDepth = ctx.createGain(); lfoDepth.gain.value = 260;
    lfo.connect(lfoDepth); lfoDepth.connect(this.padFilter.frequency); lfo.start();
    this.padBus = ctx.createGain(); this.padBus.gain.value = MOODS.idle.pad;
    const padSend = ctx.createGain(); padSend.gain.value = 0.6;
    this.padFilter.connect(this.padBus); this.padBus.connect(this.music); this.padBus.connect(padSend); padSend.connect(this.reverbIn);

    // sub drone that follows the chord root
    this.sub = ctx.createOscillator(); this.sub.type = 'sine';
    this.subGain = ctx.createGain(); this.subGain.gain.value = MOODS.idle.sub;
    this.sub.connect(this.subGain); this.subGain.connect(this.music); this.sub.start();

    // cosmic wind: band-passed noise with a wandering center
    const wind = ctx.createBufferSource(); wind.buffer = this.noise; wind.loop = true;
    const bp = ctx.createBiquadFilter(); bp.type = 'bandpass'; bp.Q.value = 0.9; bp.frequency.value = 700;
    const wl = ctx.createOscillator(); wl.frequency.value = 0.031;
    const wd = ctx.createGain(); wd.gain.value = 450;
    wl.connect(wd); wd.connect(bp.frequency); wl.start();
    this.windGain = ctx.createGain(); this.windGain.gain.value = MOODS.idle.wind;
    wind.connect(bp); bp.connect(this.windGain); this.windGain.connect(this.music); this.windGain.connect(this.reverbIn);
    wind.start();

    this.nextChord(true);
    this.scheduleSparkle();
  }

  private impulse(seconds: number) {
    const ctx = this.ctx!;
    const len = Math.floor(ctx.sampleRate * seconds);
    const buf = ctx.createBuffer(2, len, ctx.sampleRate);
    for (let c = 0; c < 2; c++) {
      const d = buf.getChannelData(c);
      for (let i = 0; i < len; i++) {
        const t = i / len;
        d[i] = (Math.random() * 2 - 1) * Math.pow(1 - t, 2.8) * (i < 400 ? i / 400 : 1);
      }
    }
    return buf;
  }

  // --- music --------------------------------------------------------------------

  private nextChord(first = false) {
    if (!this.ctx) return;
    const ctx = this.ctx;
    clearTimeout(this.chordTimer);
    if (!first && !this.live) {
      // suspended clock: don't stack voices whose stop times never arrive
      this.chordTimer = window.setTimeout(() => this.nextChord(), 4000);
      return;
    }
    const palette = this.dark ? DARK : BRIGHT;
    if (!first) this.chordIdx = (this.chordIdx + 1 + (Math.random() < 0.2 ? 1 : 0)) % palette.length;
    const chord = palette[this.chordIdx];
    const t = ctx.currentTime;
    const attack = first ? 4 : 3.2;
    // fade out the previous chord
    for (const v of this.pads) {
      v.gain.gain.cancelScheduledValues(t);
      v.gain.gain.setTargetAtTime(0, t, 1.6);
      v.nodes.forEach((n) => n.stop(t + 8));
    }
    this.pads = [];
    const voice: PadVoice = { gain: ctx.createGain(), nodes: [] };
    voice.gain.gain.value = 0;
    voice.gain.gain.setTargetAtTime(1, t, attack / 3);
    voice.gain.connect(this.padFilter);
    chord.forEach((m, i) => {
      const pan = ctx.createStereoPanner();
      pan.pan.value = (i / (chord.length - 1) - 0.5) * 0.9;
      const g = ctx.createGain(); g.gain.value = 0.055 / (1 + i * 0.12);
      g.connect(pan); pan.connect(voice.gain);
      for (const det of [-7, 6]) {
        const o = ctx.createOscillator();
        o.type = 'sawtooth';
        o.frequency.value = mtof(m);
        o.detune.value = det + rand(-2, 2);
        o.connect(g); o.start(t); voice.nodes.push(o);
      }
    });
    this.pads.push(voice);
    if (first) this.sub.frequency.setValueAtTime(mtof(chord[0] - 12), t);
    else this.sub.frequency.setTargetAtTime(mtof(chord[0] - 12), t, 1.2);
    this.chordTimer = window.setTimeout(() => this.nextChord(), MOODS[this.mood].chordSec * 1000 * rand(0.85, 1.15));
  }

  private scheduleSparkle() {
    clearTimeout(this.sparkleTimer);
    const rate = MOODS[this.mood].sparkle;
    const wait = -Math.log(1 - Math.random()) / Math.max(rate, 0.01);
    this.sparkleTimer = window.setTimeout(() => {
      if (this.live) {
        const deg = Math.floor(rand(5, 14));
        const t = this.ctx!.currentTime + 0.02;
        this.bell(mtof(scaleNote(this.dark, deg)), t, rand(0.018, 0.04), rand(2, 3.5), rand(-0.8, 0.8), 0.7);
        if (Math.random() < 0.3) this.bell(mtof(scaleNote(this.dark, deg + 2)), t + rand(0.12, 0.3), rand(0.012, 0.025), 2.5, rand(-0.8, 0.8), 0.7);
      }
      this.scheduleSparkle();
    }, Math.min(wait, 12) * 1000);
  }

  setMood(mood: Mood, opts: { dark?: boolean } = {}) {
    const wasDark = this.dark;
    this.mood = mood;
    if (opts.dark !== undefined) this.dark = opts.dark;
    else if (mood === 'idle' || mood === 'ready' || mood === 'loading') this.dark = false;
    if (!this.ctx) return;
    const t = this.ctx.currentTime;
    const p = MOODS[mood];
    this.padBus.gain.setTargetAtTime(p.pad, t, 1.2);
    this.padFilter.frequency.setTargetAtTime(p.cutoff, t, mood === 'analyzing' ? 0.6 : 1.5);
    this.subGain.gain.setTargetAtTime(p.sub, t, 1.5);
    this.windGain.gain.setTargetAtTime(p.wind, t, 1.5);
    if (wasDark !== this.dark) this.nextChord();
    this.scheduleSparkle();
  }

  /** Load progress opens the filter little by little. */
  progress(frac: number) {
    if (!this.ctx || this.mood !== 'loading') return;
    this.padFilter.frequency.setTargetAtTime(MOODS.loading.cutoff + 900 * frac, this.ctx.currentTime, 1);
  }

  // --- instruments --------------------------------------------------------------

  /** Output strip for one event: pan, then dry + reverb/delay sends. */
  private out(pan = 0, reverb = 0.4, delay = 0) {
    const ctx = this.ctx!;
    const p = ctx.createStereoPanner(); p.pan.value = pan;
    p.connect(this.sfx);
    if (reverb) { const s = ctx.createGain(); s.gain.value = reverb; p.connect(s); s.connect(this.reverbIn); }
    if (delay) { const s = ctx.createGain(); s.gain.value = delay; p.connect(s); s.connect(this.delayIn); }
    return p;
  }

  /** FM-ish bell: fundamental + inharmonic partials. */
  private bell(f: number, t: number, gain: number, decay: number, pan = 0, reverb = 0.6, delay = 0.35) {
    const ctx = this.ctx!;
    const g = this.out(pan, reverb, delay);
    const partials: [number, number, number][] = [[1, 1, 1], [2.756, 0.28, 0.45], [5.404, 0.08, 0.2]];
    for (const [ratio, amp, dk] of partials) {
      const o = ctx.createOscillator(); o.type = 'sine'; o.frequency.value = f * ratio;
      const e = ctx.createGain(); e.gain.value = 0;
      e.gain.setValueAtTime(0, t);
      e.gain.linearRampToValueAtTime(gain * amp, t + 0.006);
      e.gain.exponentialRampToValueAtTime(0.0001, t + decay * dk);
      o.connect(e); e.connect(g); o.start(t); o.stop(t + decay + 0.1);
    }
  }

  /** Soft plucked string: triangle + saw through a closing lowpass. */
  private pluck(f: number, t: number, gain: number, decay = 1.4, pan = 0, reverb = 0.5, delay = 0.2) {
    const ctx = this.ctx!;
    const g = this.out(pan, reverb, delay);
    const lp = ctx.createBiquadFilter(); lp.type = 'lowpass'; lp.Q.value = 2;
    lp.frequency.setValueAtTime(Math.min(f * 8, 12000), t);
    lp.frequency.exponentialRampToValueAtTime(Math.max(f * 1.2, 200), t + 0.5);
    const e = ctx.createGain(); e.gain.value = 0;
    e.gain.setValueAtTime(0, t);
    e.gain.linearRampToValueAtTime(gain, t + 0.004);
    e.gain.exponentialRampToValueAtTime(0.0001, t + decay);
    for (const [type, amp] of [['triangle', 1], ['sawtooth', 0.25]] as const) {
      const o = ctx.createOscillator(); o.type = type; o.frequency.value = f;
      const a = ctx.createGain(); a.gain.value = amp;
      o.connect(a); a.connect(lp); o.start(t); o.stop(t + decay + 0.1);
    }
    lp.connect(e); e.connect(g);
  }

  private noiseSweep(t: number, dur: number, f0: number, f1: number, peak: number, q = 1.1, pan0 = 0, pan1 = 0, reverb = 0.5) {
    const ctx = this.ctx!;
    const src = ctx.createBufferSource(); src.buffer = this.noise; src.loop = true;
    const bp = ctx.createBiquadFilter(); bp.type = 'bandpass'; bp.Q.value = q;
    bp.frequency.setValueAtTime(f0, t);
    bp.frequency.exponentialRampToValueAtTime(f1, t + dur);
    const e = ctx.createGain(); e.gain.value = 0;
    e.gain.setValueAtTime(0.0001, t);
    e.gain.exponentialRampToValueAtTime(peak, t + dur * 0.62);
    e.gain.exponentialRampToValueAtTime(0.0001, t + dur);
    const p = ctx.createStereoPanner();
    p.pan.setValueAtTime(pan0, t); p.pan.linearRampToValueAtTime(pan1, t + dur);
    src.connect(bp); bp.connect(e); e.connect(p); p.connect(this.sfx);
    if (reverb) { const s = ctx.createGain(); s.gain.value = reverb; p.connect(s); s.connect(this.reverbIn); }
    src.start(t, Math.random()); src.stop(t + dur + 0.05);
  }

  // --- effects (public) ------------------------------------------------------------

  /** Quiet UI tick for buttons. */
  tick() {
    if (!this.live) return;
    const t = this.ctx!.currentTime;
    this.bell(mtof(scaleNote(this.dark, 12)), t, 0.012, 0.35, 0, 0.2, 0);
  }

  whoosh(dir: 'up' | 'down' | 'side', dur = 1.6) {
    if (!this.live) return;
    const t = this.ctx!.currentTime;
    if (dir === 'up') this.noiseSweep(t, dur, 250, 3200, 0.16, 1.1, -0.5, 0.5);
    else if (dir === 'down') this.noiseSweep(t, dur, 3000, 220, 0.14, 1.1, 0.5, -0.5);
    else this.noiseSweep(t, dur, 600, 1800, 0.1, 1.4, -0.7, 0.7);
  }

  /** A transformer layer finished uploading: the tower plays itself, low to high. */
  layer(i: number, n: number, full: boolean) {
    if (!this.live) return;
    const ctx = this.ctx!;
    const t = Math.max(ctx.currentTime + 0.01, this.nextLayerAt);
    this.nextLayerAt = t + 0.075; // bursts (cache loads) become a glissando
    const deg = Math.round((i / Math.max(1, n - 1)) * 14);
    const f = mtof(scaleNote(false, deg, 50));
    const pan = Math.sin(i * 2.4) * 0.6;
    this.pluck(f, t, 0.075, 1.6, pan, 0.45, 0.25);
    if (full) this.bell(f * 2, t + 0.01, 0.03, 2.2, -pan, 0.6, 0.3);
  }

  /** Model ready: a slow chord bloom. */
  bloom() {
    if (!this.live) return;
    const t = this.ctx!.currentTime + 0.05;
    const chord = (this.dark ? DARK : BRIGHT)[this.chordIdx];
    chord.forEach((m, i) => this.bell(mtof(m + 24), t + i * 0.11, 0.035, 4, (i / 4 - 0.5) * 1.2, 0.8, 0.4));
  }

  /** Token chips leaving the input, one blip each (matches the chip stagger). */
  tokens(n: number, staggerMs = 55) {
    if (!this.live) return;
    const t = this.ctx!.currentTime + 0.02;
    for (let i = 0; i < Math.min(n, 24); i++) {
      this.bell(mtof(scaleNote(this.dark, 7 + (i % 8), 62)), t + (i * staggerMs) / 1000, 0.022, 0.6, (i / Math.max(1, n - 1) - 0.5) * 0.8, 0.3, 0.15);
    }
  }

  /** The signal climbing the tower: riser + a tick for every few layers. */
  riser(seconds: number, layers: number) {
    if (!this.live) return;
    const ctx = this.ctx!;
    const t = ctx.currentTime + 0.02;
    this.noiseSweep(t, seconds + 0.25, 180, 5200, 0.12, 0.9, 0, 0, 0.6);
    const o = ctx.createOscillator(); o.type = 'sawtooth';
    o.frequency.setValueAtTime(mtof(38), t);
    o.frequency.exponentialRampToValueAtTime(mtof(74), t + seconds);
    const lp = ctx.createBiquadFilter(); lp.type = 'lowpass'; lp.Q.value = 4;
    lp.frequency.setValueAtTime(180, t); lp.frequency.exponentialRampToValueAtTime(3800, t + seconds);
    const e = ctx.createGain();
    e.gain.setValueAtTime(0.0001, t);
    e.gain.exponentialRampToValueAtTime(0.05, t + seconds * 0.9);
    e.gain.exponentialRampToValueAtTime(0.0001, t + seconds + 0.3);
    o.connect(lp); lp.connect(e); e.connect(this.sfx);
    const s = ctx.createGain(); s.gain.value = 0.5; e.connect(s); s.connect(this.reverbIn);
    o.start(t); o.stop(t + seconds + 0.4);
    const steps = Math.min(16, layers);
    for (let k = 0; k < steps; k++) {
      // ease-in-out timing like the visual pulse
      const x = (1 - Math.cos(Math.PI * (k / steps))) / 2;
      this.bell(mtof(scaleNote(this.dark, 3 + k, 62)), t + x * seconds, 0.012 + 0.012 * (k / steps), 0.5, Math.sin(k * 1.7) * 0.5, 0.35, 0.1);
    }
  }

  /** Arrival at the winning neuron: low boom, air crack, chord bloom. */
  impact(pole: 1 | -1) {
    if (!this.live) return;
    const ctx = this.ctx!;
    const t = ctx.currentTime + 0.01;
    const o = ctx.createOscillator(); o.type = 'sine';
    o.frequency.setValueAtTime(120, t); o.frequency.exponentialRampToValueAtTime(36, t + 1.1);
    const e = ctx.createGain();
    e.gain.setValueAtTime(0.0001, t);
    e.gain.exponentialRampToValueAtTime(0.42, t + 0.02);
    e.gain.exponentialRampToValueAtTime(0.0001, t + 2.2);
    o.connect(e); e.connect(this.sfx); o.start(t); o.stop(t + 2.3);
    this.noiseSweep(t, 0.5, 5000, 1800, 0.05, 0.7, 0, 0, 0.9);
    const chord = (pole < 0 ? DARK : BRIGHT)[0];
    chord.forEach((m, i) => this.bell(mtof(m + 24), t + 0.08 + i * 0.07, 0.04, 4.5, (i / 4 - 0.5) * 1.4, 0.9, 0.45));
  }

  /** The focused neuron's own voice: pitch is derived from its address. */
  humOn(layer: number, index: number, pole: 1 | -1) {
    if (!this.ctx) return;
    const ctx = this.ctx;
    const t = ctx.currentTime;
    const h = Math.abs((layer * 9973 + index * 7919) % 15);
    const f = mtof(scaleNote(pole < 0, h, 38));
    if (this.hum) {
      for (const [i, o] of this.hum.oscs.entries()) o.frequency.setTargetAtTime(f * [1, 1.0035, 2][i], t, 0.35);
      return;
    }
    const gain = ctx.createGain(); gain.gain.value = 0;
    const lp = ctx.createBiquadFilter(); lp.type = 'lowpass'; lp.frequency.value = 900;
    const trem = ctx.createGain(); trem.gain.value = 1;
    const lfo = ctx.createOscillator(); lfo.frequency.value = 0.23;
    const depth = ctx.createGain(); depth.gain.value = 0.35;
    lfo.connect(depth); depth.connect(trem.gain); lfo.start(t);
    const oscs = (['sine', 'sine', 'triangle'] as const).map((type, i) => {
      const o = ctx.createOscillator(); o.type = type; o.frequency.value = f * [1, 1.0035, 2][i];
      const a = ctx.createGain(); a.gain.value = [1, 0.8, 0.12][i];
      o.connect(a); a.connect(lp); o.start(t);
      return o;
    });
    lp.connect(trem); trem.connect(gain); gain.connect(this.sfx);
    const s = ctx.createGain(); s.gain.value = 0.5; gain.connect(s); s.connect(this.reverbIn);
    gain.gain.setTargetAtTime(0.035, t, 0.8);
    this.hum = { oscs, gain, lfo };
  }

  humOff() {
    if (!this.ctx || !this.hum) return;
    const t = this.ctx.currentTime;
    const h = this.hum;
    this.hum = null;
    h.gain.gain.setTargetAtTime(0, t, 0.5);
    h.oscs.forEach((o) => o.stop(t + 3));
    h.lfo.stop(t + 3);
  }

  /** A concept entering the constellation; stronger = higher and brighter. */
  concept(strength: number, delayMs = 0) {
    if (!this.live) return;
    const t = this.ctx!.currentTime + 0.02 + delayMs / 1000;
    const s = Math.max(0, Math.min(1, strength));
    this.bell(mtof(scaleNote(this.dark, 4 + Math.round(s * 9), 62)), t, 0.012 + 0.03 * s, 1.6 + s, rand(-0.9, 0.9), 0.6, 0.35);
  }

  /** A live measurement came back from the GPU (batches arrive together, so stagger them). */
  verify(strength: number) {
    if (!this.live) return;
    const t = Math.max(this.ctx!.currentTime + 0.01, this.nextVerifyAt);
    this.nextVerifyAt = t + 0.06;
    const s = Math.max(0, Math.min(1, strength));
    this.pluck(mtof(scaleNote(this.dark, 7 + Math.round(s * 7), 62)), t, 0.025 + 0.03 * s, 0.9, rand(-0.6, 0.6), 0.4, 0.3);
  }

  /** Result of testing the user's own concept against the neuron. */
  result(fires: boolean) {
    if (!this.live) return;
    const t = this.ctx!.currentTime + 0.01;
    if (fires) {
      this.pluck(mtof(scaleNote(this.dark, 7, 62)), t, 0.06, 1.5, -0.2);
      this.bell(mtof(scaleNote(this.dark, 11, 62)), t + 0.12, 0.05, 3, 0.2, 0.7, 0.4);
    } else {
      this.pluck(mtof(scaleNote(this.dark, 2, 50)), t, 0.05, 0.8, 0, 0.3, 0);
    }
  }

  error() {
    if (!this.live) return;
    const t = this.ctx!.currentTime + 0.01;
    this.pluck(mtof(scaleNote(true, 4, 50)), t, 0.05, 1, 0, 0.3, 0);
    this.pluck(mtof(scaleNote(true, 2, 50)), t + 0.16, 0.05, 1.2, 0, 0.3, 0);
  }
}

export const sound = new Sound();
