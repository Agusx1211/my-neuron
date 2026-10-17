// The model as a tower: one phyllotaxis disk of points per transformer layer,
// one point per MLP neuron. All per-point animation happens in the shader.

import * as THREE from 'three';
import type { Stage } from './scene';
import { ease } from './camera';

export interface TowerDims {
  layers: number;
  neurons: number;
  layerTypes: ('linear' | 'full')[];
}

export const COLORS = {
  base: new THREE.Color('#2b3fa0'),
  baseTop: new THREE.Color('#5a2f9e'),
  ghost: new THREE.Color('#5560c4'),
  pos: new THREE.Color('#ffb547'),
  posHi: new THREE.Color('#ffe3a3'),
  neg: new THREE.Color('#3fb8ff'),
  negHi: new THREE.Color('#bfeaff'),
  pulse: new THREE.Color('#c9bcff'),
  ring: new THREE.Color('#8f7bff'),
};

const GOLDEN = Math.PI * (3 - Math.sqrt(5));

const vert = /* glsl */ `
  attribute float aHeat;
  attribute float aSeed;
  attribute float aLayer;
  uniform float uTime, uLayersReady, uGhost, uRevealY, uPulse, uHeatMix, uDim, uFocusOn;
  uniform float uSplitLayer, uSplit, uSize, uScale, uPR, uLayers;
  uniform vec3 uFocus, uBase, uBaseTop, uGhostC, uPos, uPosHi, uNeg, uNegHi, uPulseC;
  varying vec3 vColor;
  varying float vAlpha;
  varying float vCore;

  void main() {
    vec3 p = position;
    float layerY = p.y;

    // assembly: a layer sparkles as its weights arrive
    float appear = clamp(uLayersReady - aLayer, 0.0, 1.0);
    float flare = pow(sin(appear * 3.14159), 2.0);
    float sparkle = flare * step(0.82, fract(aSeed * 17.0));
    p.y += flare * (aSeed - 0.5) * 0.5;

    // faint galaxy arms so each disk reads as a structure, not a flat plate
    float rr = length(p.xz) / 10.0;
    float th = atan(p.z, p.x);
    float arms = 0.72 + 0.28 * sin(2.0 * th - 7.0 * rr + aLayer * 0.7);
    float frac = aLayer / max(1.0, uLayers - 1.0);

    // accordion split around the focused layer
    float d = aLayer - uSplitLayer;
    p.y += (step(0.5, d) - step(0.5, -d)) * uSplit;

    // heat, revealed below the travelling pulse front
    float revealed = smoothstep(uRevealY + 0.35, uRevealY - 0.35, layerY);
    float h = aHeat * uHeatMix * revealed;
    float a = abs(h);
    vec3 hot = h > 0.0 ? mix(uPos, uPosHi, smoothstep(0.75, 1.0, a)) : mix(uNeg, uNegHi, smoothstep(0.75, 1.0, a));
    vec3 idle = mix(uGhostC, mix(uBase, uBaseTop, frac), appear);
    float heatK = smoothstep(0.12, 0.55, a);
    vec3 col = mix(idle, hot, heatK);
    float alpha = mix(uGhost, 0.5, appear) * arms * (0.85 + 0.3 * (1.0 - rr));
    alpha *= 1.0 - 0.5 * uHeatMix * revealed; // calm neurons recede so the hot ones pop
    alpha = mix(alpha, 1.0, heatK);

    // pulse band
    float band = exp(-pow((layerY - uRevealY) * 1.6, 2.0)) * uPulse;
    col = mix(col, uPulseC, band * 0.7);
    alpha += band * 0.6;

    // sparkle while assembling + gentle twinkle
    col = mix(col, uPulseC, flare * 0.5);
    col = mix(col, vec3(1.0), sparkle * 0.7);
    alpha += flare * 0.18 + sparkle * 0.8;
    alpha *= 0.82 + 0.18 * sin(uTime * (0.8 + aSeed * 1.7) + aSeed * 50.0);

    // dim everything but the focus neighbourhood
    vec4 world = modelMatrix * vec4(p, 1.0);
    float near = 1.0 - smoothstep(0.4, 3.2, distance(world.xyz, uFocus));
    float sameLayer = 1.0 - step(0.5, abs(d));
    float keep = max(near * uFocusOn, sameLayer * uFocusOn * 0.22);
    alpha *= mix(1.0, 0.07 + 0.93 * keep, uDim);

    vec4 mv = viewMatrix * world;
    float size = uSize * (0.85 + 1.9 * heatK + 1.6 * sparkle + band * 0.8);
    float px = size * uScale / -mv.z;
    float maxPx = (9.0 + 12.0 * heatK) * uPR;
    px = min(px, maxPx);
    // sub-pixel points: keep a pixel but fade them so dense disks stay soft
    float minPx = 1.2 * uPR;
    alpha *= clamp(px / minPx, 0.15, 1.0);
    gl_PointSize = max(px, minPx);
    gl_Position = projectionMatrix * mv;
    vColor = col;
    vAlpha = alpha;
    vCore = heatK;
  }
`;

const frag = /* glsl */ `
  varying vec3 vColor;
  varying float vAlpha;
  varying float vCore;
  void main() {
    float d = length(gl_PointCoord - 0.5);
    float m = smoothstep(0.5, 0.05, d);
    m *= m;
    float core = smoothstep(0.16, 0.0, d) * (0.4 + vCore);
    gl_FragColor = vec4(vColor * (1.0 + core), vAlpha * m);
  }
`;

const hitVert = /* glsl */ `
  attribute float aPole;
  attribute float aRank;
  uniform float uTime, uOn, uScale, uPR;
  varying vec3 vColor;
  varying float vAlpha;
  uniform vec3 uPos, uNeg;
  void main() {
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    float pulse = 0.75 + 0.25 * sin(uTime * 2.4 - aRank * 0.7);
    float size = (0.42 - aRank * 0.008) * pulse;
    gl_PointSize = clamp(size * uScale / -mv.z, 3.0 * uPR, 64.0 * uPR);
    gl_Position = projectionMatrix * mv;
    vColor = aPole > 0.0 ? uPos : uNeg;
    vAlpha = uOn * (1.0 - aRank * 0.025);
  }
`;

const hitFrag = /* glsl */ `
  varying vec3 vColor;
  varying float vAlpha;
  void main() {
    float d = length(gl_PointCoord - 0.5);
    float glow = smoothstep(0.5, 0.0, d);
    float ring = smoothstep(0.03, 0.0, abs(d - 0.36)) * 0.8;
    float core = smoothstep(0.12, 0.0, d) * 1.5;
    vec3 c = vColor * (glow * glow * 1.4 + ring) + vec3(1.0) * core;
    gl_FragColor = vec4(c, vAlpha * clamp(glow + ring, 0.0, 1.0));
  }
`;

export interface HitMark { layer: number; index: number; pole: 1 | -1 }

export class Tower {
  readonly group = new THREE.Group();
  readonly R = 10;
  readonly spacing = 1.1;
  dims: TowerDims;
  private points!: THREE.Points;
  private geo!: THREE.BufferGeometry;
  private mat: THREE.ShaderMaterial;
  private heatAttr!: THREE.BufferAttribute;
  private rings: THREE.LineLoop[] = [];
  private ringMat: THREE.LineBasicMaterial;
  private hits: THREE.Points;
  private hitList: HitMark[] = [];
  private ready = 0;
  private readyTarget = 0;
  private splitLayer = -100;
  private split = 0;

  constructor(private stage: Stage, dims: TowerDims) {
    this.dims = dims;
    this.mat = new THREE.ShaderMaterial({
      uniforms: {
        uTime: { value: 0 },
        uLayersReady: { value: 0 },
        uGhost: { value: 0.3 },
        uRevealY: { value: -100 },
        uPulse: { value: 0 },
        uHeatMix: { value: 0 },
        uDim: { value: 0 },
        uFocusOn: { value: 0 },
        uFocus: { value: new THREE.Vector3() },
        uSplitLayer: { value: -100 },
        uSplit: { value: 0 },
        uSize: { value: 0.1 },
        uScale: { value: 500 },
        uPR: { value: stage.renderer.getPixelRatio() },
        uLayers: { value: dims.layers },
        uBase: { value: COLORS.base },
        uBaseTop: { value: COLORS.baseTop },
        uGhostC: { value: COLORS.ghost },
        uPos: { value: COLORS.pos },
        uPosHi: { value: COLORS.posHi },
        uNeg: { value: COLORS.neg },
        uNegHi: { value: COLORS.negHi },
        uPulseC: { value: COLORS.pulse },
      },
      vertexShader: vert,
      fragmentShader: frag,
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });
    this.ringMat = new THREE.LineBasicMaterial({
      color: COLORS.ring, transparent: true, opacity: 0.0, blending: THREE.AdditiveBlending, depthWrite: false,
    });

    const hg = new THREE.BufferGeometry();
    hg.setAttribute('position', new THREE.BufferAttribute(new Float32Array(64 * 3), 3));
    hg.setAttribute('aPole', new THREE.BufferAttribute(new Float32Array(64), 1));
    hg.setAttribute('aRank', new THREE.BufferAttribute(new Float32Array(64), 1));
    hg.setDrawRange(0, 0);
    this.hits = new THREE.Points(
      hg,
      new THREE.ShaderMaterial({
        uniforms: {
          uTime: { value: 0 }, uOn: { value: 0 }, uScale: { value: 500 }, uPR: { value: stage.renderer.getPixelRatio() },
          uPos: { value: COLORS.pos }, uNeg: { value: COLORS.neg },
        },
        vertexShader: hitVert,
        fragmentShader: hitFrag,
        transparent: true,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
      }),
    );
    this.hits.frustumCulled = false;
    this.group.add(this.hits);

    this.build(dims);
    stage.scene.add(this.group);
    stage.onTick((t, dt) => this.tick(t, dt));
  }

  get height() {
    return (this.dims.layers - 1) * this.spacing;
  }

  layerY(layer: number) {
    return (layer - (this.dims.layers - 1) / 2) * this.spacing;
  }

  /** Rebuild geometry for a model's dimensions. */
  build(dims: TowerDims) {
    this.dims = dims;
    this.mat.uniforms.uLayers.value = dims.layers;
    if (this.points) {
      this.group.remove(this.points);
      this.geo.dispose();
    }
    for (const r of this.rings) { this.group.remove(r); r.geometry.dispose(); }
    this.rings = [];

    const { layers: L, neurons: N } = dims;
    const n = L * N;
    const pos = new Float32Array(n * 3);
    const seed = new Float32Array(n);
    const layer = new Float32Array(n);
    let s = 1234567;
    const rnd = () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296);
    for (let l = 0; l < L; l++) {
      const y = this.layerY(l);
      const twist = l * 0.21;
      for (let i = 0; i < N; i++) {
        const k = l * N + i;
        const r = this.R * Math.sqrt((i + 0.5) / N) * (1 + (rnd() - 0.5) * 0.012);
        const th = i * GOLDEN + twist;
        pos[k * 3] = r * Math.cos(th);
        pos[k * 3 + 1] = y + (rnd() - 0.5) * 0.05;
        pos[k * 3 + 2] = r * Math.sin(th);
        seed[k] = rnd();
        layer[k] = l;
      }
    }
    this.geo = new THREE.BufferGeometry();
    this.geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    this.geo.setAttribute('aSeed', new THREE.BufferAttribute(seed, 1));
    this.geo.setAttribute('aLayer', new THREE.BufferAttribute(layer, 1));
    this.heatAttr = new THREE.BufferAttribute(new Float32Array(n), 1);
    this.heatAttr.setUsage(THREE.DynamicDrawUsage);
    this.geo.setAttribute('aHeat', this.heatAttr);
    this.geo.boundingSphere = new THREE.Sphere(new THREE.Vector3(), this.R + this.height);
    this.points = new THREE.Points(this.geo, this.mat);
    this.points.frustumCulled = false;
    this.group.add(this.points);

    const pts: THREE.Vector3[] = [];
    for (let i = 0; i < 256; i++) {
      const a = (i / 256) * Math.PI * 2;
      pts.push(new THREE.Vector3(Math.cos(a) * (this.R + 0.45), 0, Math.sin(a) * (this.R + 0.45)));
    }
    const ringGeo = new THREE.BufferGeometry().setFromPoints(pts);
    dims.layerTypes.forEach((t, l) => {
      if (t !== 'full') return;
      const ring = new THREE.LineLoop(ringGeo, this.ringMat.clone());
      ring.position.y = this.layerY(l);
      ring.userData.layer = l;
      this.rings.push(ring);
      this.group.add(ring);
    });
    this.ready = Math.min(this.ready, L);
  }

  /** Smoothly bring layers [0, n) to full brightness. */
  setLayersReady(n: number, instant = false) {
    this.readyTarget = n;
    if (instant) this.ready = n;
  }

  /** Signed per-neuron z-scores ([layers*neurons]); null clears. */
  setHeat(heat: Float32Array | null) {
    const arr = this.heatAttr.array as Float32Array;
    if (!heat) arr.fill(0);
    else for (let i = 0; i < arr.length; i++) arr[i] = Math.max(-1, Math.min(1, (heat[i] ?? 0) / 8));
    this.heatAttr.needsUpdate = true;
  }

  /** Run the forward-pass pulse up the tower, revealing heat as it passes. */
  async pulse(duration = 1900) {
    const u = this.mat.uniforms;
    const y0 = this.layerY(0) - 1.2, y1 = this.layerY(this.dims.layers - 1) + 1.2;
    u.uHeatMix.value = 1;
    u.uPulse.value = 1;
    await this.stage.tweens.run(duration, (k) => { u.uRevealY.value = y0 + (y1 - y0) * k; }, { ease: ease.inOutSine, key: 'pulse' });
    await this.stage.tweens.run(500, (k) => { u.uPulse.value = 1 - k; }, { key: 'pulse-fade' });
  }

  /** Fade heat out (e.g. when starting over). */
  clearHeat(duration = 600) {
    const u = this.mat.uniforms;
    const from = u.uHeatMix.value as number;
    return this.stage.tweens.run(duration, (k) => { u.uHeatMix.value = from * (1 - k); }, { key: 'heat' }).then(() => {
      u.uRevealY.value = -100;
    });
  }

  setHits(hits: HitMark[]) {
    this.hitList = hits.slice(0, 64);
    const g = this.hits.geometry;
    const pole = g.getAttribute('aPole') as THREE.BufferAttribute;
    const rank = g.getAttribute('aRank') as THREE.BufferAttribute;
    this.hitList.forEach((h, i) => { pole.setX(i, h.pole); rank.setX(i, i); });
    pole.needsUpdate = rank.needsUpdate = true;
    g.setDrawRange(0, this.hitList.length);
    this.updateHitPositions();
  }

  showHits(on: boolean, duration = 800) {
    const u = (this.hits.material as THREE.ShaderMaterial).uniforms;
    const from = u.uOn.value as number;
    return this.stage.tweens.run(duration, (k) => { u.uOn.value = from + ((on ? 1 : 0) - from) * k; }, { key: 'hits' });
  }

  /** Dim the tower around a focused neuron and open the stack at its layer. */
  async focus(layer: number, index: number, duration = 1600) {
    const u = this.mat.uniforms;
    if (this.splitLayer !== layer && this.split > 0.01) {
      // close the old gap before opening a new one
      const from = this.split;
      await this.stage.tweens.run(450, (k) => {
        this.split = from * (1 - k);
        u.uSplit.value = this.split;
        this.updateHitPositions();
        this.updateRings();
      }, { key: 'focus' });
    }
    const fromSplit = this.splitLayer === layer ? this.split : 0;
    const fromDim = u.uDim.value as number;
    this.splitLayer = layer;
    u.uSplitLayer.value = layer;
    u.uFocusOn.value = 1;
    u.uFocus.value.copy(this.group.localToWorld(this.localPosition(layer, index, 0)));
    await this.stage.tweens.run(
      duration,
      (k) => {
        this.split = fromSplit + (2.6 - fromSplit) * k;
        u.uSplit.value = this.split;
        u.uDim.value = fromDim + (1 - fromDim) * k;
        this.updateHitPositions();
        this.updateRings();
      },
      { ease: ease.inOutCubic, key: 'focus' },
    );
  }

  unfocus(duration = 1400) {
    const u = this.mat.uniforms;
    const fromSplit = this.split;
    const fromDim = u.uDim.value as number;
    return this.stage.tweens.run(
      duration,
      (k) => {
        this.split = fromSplit * (1 - k);
        u.uSplit.value = this.split;
        u.uDim.value = fromDim * (1 - k);
        this.updateHitPositions();
        this.updateRings();
      },
      { ease: ease.inOutCubic, key: 'focus' },
    ).then(() => {
      u.uFocusOn.value = 0;
      this.splitLayer = -100;
      u.uSplitLayer.value = -100;
    });
  }

  /** Local (group-space) position of a neuron, including the current split. */
  localPosition(layer: number, index: number, split = this.split, out = new THREE.Vector3()) {
    const N = this.dims.neurons;
    const r = this.R * Math.sqrt((index + 0.5) / N);
    const th = index * GOLDEN + layer * 0.21;
    const d = layer - this.splitLayer;
    const off = d > 0.5 ? split : d < -0.5 ? -split : 0;
    return out.set(r * Math.cos(th), this.layerY(layer) + off, r * Math.sin(th));
  }

  /** World position of a neuron (what cameras and labels should target). */
  neuronPosition(layer: number, index: number, out = new THREE.Vector3()) {
    this.localPosition(layer, index, this.split, out);
    return this.group.localToWorld(out);
  }

  /** World position of the bottom of the tower (where tokens enter). */
  basePosition(out = new THREE.Vector3()) {
    out.set(0, this.layerY(0) - 0.6, 0);
    return this.group.localToWorld(out);
  }

  private updateHitPositions() {
    const pos = this.hits.geometry.getAttribute('position') as THREE.BufferAttribute;
    const v = new THREE.Vector3();
    this.hitList.forEach((h, i) => {
      this.localPosition(h.layer, h.index, this.split, v);
      pos.setXYZ(i, v.x, v.y, v.z);
    });
    pos.needsUpdate = true;
  }

  private updateRings() {
    for (const r of this.rings) {
      const l = r.userData.layer as number;
      const d = l - this.splitLayer;
      r.position.y = this.layerY(l) + (d > 0.5 ? this.split : d < -0.5 ? -this.split : 0);
    }
  }

  private tick(t: number, dt: number) {
    // layers assemble at a bounded pace so each gets its flare
    if (this.ready !== this.readyTarget) {
      const step = Math.max(dt * 6, Math.abs(this.readyTarget - this.ready) * (1 - Math.exp(-dt * 2.5)));
      this.ready = this.ready < this.readyTarget ? Math.min(this.readyTarget, this.ready + step) : Math.max(this.readyTarget, this.ready - step * 3);
    }
    const u = this.mat.uniforms;
    u.uTime.value = t;
    u.uLayersReady.value = this.ready;
    const cam = this.stage.camera;
    const pr = this.stage.renderer.getPixelRatio();
    const scale = (this.stage.size.h * pr) / (2 * Math.tan((cam.fov * Math.PI) / 360));
    u.uScale.value = scale;
    u.uPR.value = pr;
    const hu = (this.hits.material as THREE.ShaderMaterial).uniforms;
    hu.uTime.value = t;
    hu.uScale.value = scale;
    hu.uPR.value = pr;
    const dim = u.uDim.value as number;
    for (const r of this.rings) {
      const l = r.userData.layer as number;
      const appear = Math.max(0, Math.min(1, this.ready - l));
      (r.material as THREE.LineBasicMaterial).opacity = (0.12 + 0.38 * appear) * (1 - dim * 0.93);
    }
  }
}
