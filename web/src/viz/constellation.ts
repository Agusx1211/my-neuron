// Concepts orbiting the focused neuron. Stronger activation = closer and
// brighter; the opposite pole's concepts hang in an outer, dimmer band.

import * as THREE from 'three';
import { CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';
import type { Stage } from './scene';
import { ease } from './camera';
import { COLORS } from './tower';

type Kind = 'primary' | 'opposite' | 'you' | 'added';

interface Node {
  text: string;
  kind: Kind;
  blue: boolean;
  angle: number;
  depth: number;
  r: number;        // current radius
  rTarget: number;
  enter: number;    // 0..1 fly-in progress
  strength: number; // 0..1 brightness
  live: boolean;
  vis: number;      // 0..1 label visibility after decluttering
  w: number;
  h: number;
  el: HTMLDivElement;
  valueEl: HTMLSpanElement;
  label: CSS2DObject;
}

const GOLDEN = Math.PI * (3 - Math.sqrt(5));
const MAX = 72;

export class Constellation {
  readonly group = new THREE.Group();
  private nodes: Node[] = [];
  private points: THREE.Points;
  private lines: THREE.LineSegments;
  private right = new THREE.Vector3(1, 0, 0);
  private up = new THREE.Vector3(0, 1, 0);
  private fwd = new THREE.Vector3(0, 0, 1);
  private scale = 1;     // |act| of the strongest primary concept
  private oppScale = 1;
  private opacity = 0;
  private open = false;

  constructor(private stage: Stage) {
    const pg = new THREE.BufferGeometry();
    pg.setAttribute('position', new THREE.BufferAttribute(new Float32Array(MAX * 3), 3));
    pg.setAttribute('color', new THREE.BufferAttribute(new Float32Array(MAX * 3), 3));
    pg.setAttribute('aSize', new THREE.BufferAttribute(new Float32Array(MAX), 1));
    this.points = new THREE.Points(pg, new THREE.ShaderMaterial({
      uniforms: { uScale: { value: 500 }, uOpacity: { value: 0 } },
      vertexShader: /* glsl */ `
        attribute vec3 color; attribute float aSize;
        uniform float uScale; varying vec3 vC;
        void main() {
          vec4 mv = modelViewMatrix * vec4(position, 1.0);
          gl_Position = projectionMatrix * mv;
          gl_PointSize = clamp(aSize * uScale / -mv.z, 2.0, 48.0);
          vC = color;
        }`,
      fragmentShader: /* glsl */ `
        uniform float uOpacity; varying vec3 vC;
        void main() {
          float d = length(gl_PointCoord - 0.5);
          float g = smoothstep(0.5, 0.0, d);
          float core = smoothstep(0.14, 0.0, d);
          gl_FragColor = vec4(vC * (g * g + core * 1.5), uOpacity * g);
        }`,
      transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    }));
    this.points.frustumCulled = false;
    const lg = new THREE.BufferGeometry();
    lg.setAttribute('position', new THREE.BufferAttribute(new Float32Array(MAX * 6), 3));
    lg.setAttribute('color', new THREE.BufferAttribute(new Float32Array(MAX * 6), 3));
    this.lines = new THREE.LineSegments(lg, new THREE.LineBasicMaterial({
      vertexColors: true, transparent: true, opacity: 0, blending: THREE.AdditiveBlending, depthWrite: false,
    }));
    this.lines.frustumCulled = false;
    this.group.add(this.lines, this.points);
    stage.scene.add(this.group);
    stage.onTick((_, dt) => this.tick(dt));
  }

  get isOpen() {
    return this.open;
  }

  /** Lay the concepts out in the plane facing the camera, around `center`. */
  show(opts: {
    center: THREE.Vector3;
    pole: 1 | -1;
    primary: { text: string; act: number }[];
    opposite: { text: string; act: number }[];
    you?: { text: string; act: number };
  }) {
    this.clearNodes();
    this.open = true;
    this.group.position.copy(opts.center);
    const cam = this.stage.camera;
    cam.updateMatrixWorld();
    this.fwd.copy(cam.position).sub(opts.center).normalize();
    this.right.setFromMatrixColumn(cam.matrixWorld, 0).normalize();
    this.up.crossVectors(this.fwd, this.right).normalize();
    this.scale = Math.max(1e-3, Math.abs(opts.primary[0]?.act ?? 1));
    this.oppScale = Math.max(1e-3, Math.abs(opts.opposite[0]?.act ?? 1));
    const primaryBlue = opts.pole < 0;

    opts.primary.slice(0, 30).forEach((c, k) => {
      this.addNode(c.text, 'primary', primaryBlue, k * GOLDEN + 0.6, this.radiusFor(c.act), k);
    });
    const opp = opts.opposite.slice(0, 12);
    opp.forEach((c, k) => {
      const ang = -Math.PI * 0.15 - (k / Math.max(1, opp.length - 1)) * Math.PI * 0.7; // lower arc
      const r = 3.55 + (1 - Math.min(1, Math.abs(c.act) / this.oppScale)) * 0.6 + (k % 2) * 0.25;
      this.addNode(c.text, 'opposite', !primaryBlue, ang, r, 8 + k, -0.8);
    });
    if (opts.you) {
      this.addNode(opts.you.text, 'you', primaryBlue, Math.PI * 0.5 + 0.35, this.radiusFor(opts.you.act), 0, 0.4);
      this.setValue(this.nodes[this.nodes.length - 1], opts.you.act);
    }
    this.stage.tweens.run(700, (k) => { this.opacity = k; }, { key: 'const-op' });
  }

  /** Live-measured activation arrived for a concept: snap it to its true radius. */
  setLive(text: string, act: number) {
    for (const n of this.nodes) {
      if (n.text !== text || n.kind === 'you') continue;
      n.live = true;
      n.el.classList.add('live');
      this.setValue(n, act);
      if (n.kind === 'opposite') {
        n.rTarget = 3.55 + (1 - Math.min(1, Math.abs(act) / this.oppScale)) * 0.6;
      } else {
        n.rTarget = this.radiusFor(act);
        n.strength = Math.min(1, Math.abs(act) / this.scale);
      }
      // a quick flash on the label
      n.el.animate([{ filter: 'brightness(2.2)' }, { filter: 'brightness(1)' }], { duration: 600, easing: 'ease-out' });
    }
  }

  /** A concept typed by the user, measured live. */
  addUser(text: string, act: number, pole: 1 | -1) {
    const existing = this.nodes.find((n) => n.text.toLowerCase() === text.toLowerCase() && n.kind === 'added');
    if (existing) { this.setLive(existing.text, act); return; }
    const sameSign = Math.sign(act) === pole || act === 0;
    const blue = sameSign ? pole < 0 : pole > 0;
    const k = this.nodes.filter((n) => n.kind === 'added').length;
    const r = sameSign ? this.radiusFor(act) : 3.8;
    this.addNode(text, 'added', blue, Math.PI * 0.5 - 0.5 - k * 0.9, r, 0, 0.6);
    const n = this.nodes[this.nodes.length - 1];
    n.live = true;
    n.el.classList.add('live');
    this.setValue(n, act);
  }

  hide() {
    if (!this.open) return Promise.resolve();
    this.open = false;
    return this.stage.tweens.run(500, (k) => { this.opacity = 1 - k; }, { key: 'const-op' }).then(() => this.clearNodes());
  }

  private radiusFor(act: number) {
    const s = Math.abs(act) / this.scale;
    return 0.95 + 2.25 * Math.pow(Math.max(0, 1 - Math.min(1.15, s)), 0.85);
  }

  private setValue(n: Node, act: number) {
    n.valueEl.textContent = (act >= 0 ? '+' : '') + act.toFixed(2);
  }

  private addNode(text: string, kind: Kind, blue: boolean, angle: number, r: number, order: number, depth = 0) {
    if (this.nodes.length >= MAX) return;
    const el = document.createElement('div');
    el.className = `clabel ${blue ? 'neg' : ''} ${kind === 'you' ? 'you' : ''} ${kind === 'added' ? 'added' : ''} ${kind === 'opposite' ? 'dim' : ''}`;
    const t = document.createElement('span');
    t.textContent = text;
    const v = document.createElement('span');
    v.className = 'cv';
    el.append(t, v);
    el.style.opacity = '0';
    const label = new CSS2DObject(el);
    label.center.set(0.5, 1.35);
    this.group.add(label);
    const node: Node = {
      text, kind, blue, angle, depth: depth + ((order * 0.37) % 1 - 0.5) * 0.9,
      r, rTarget: r, enter: 0,
      strength: kind === 'opposite' ? 0.45 : kind === 'you' || kind === 'added' ? 1 : Math.max(0.25, 1 - (r - 0.95) / 2.25),
      live: false, vis: 1, w: 0, h: 0, el, valueEl: v, label,
    };
    this.nodes.push(node);
    this.stage.tweens.run(1200, (k) => { node.enter = k; }, { ease: ease.outQuint, delay: kind === 'opposite' ? 900 + order * 40 : order * 70 });
  }

  private clearNodes() {
    for (const n of this.nodes) {
      this.group.remove(n.label);
      n.el.remove();
    }
    this.nodes = [];
    this.points.geometry.setDrawRange(0, 0);
    this.lines.geometry.setDrawRange(0, 0);
  }

  /** Stronger labels claim screen space first; overlapped ones fade out. */
  private declutter(dt: number) {
    const prio = (n: Node) => (n.kind === 'you' ? 1000 : n.kind === 'added' ? 900 : n.kind === 'primary' ? 100 + n.strength * 50 : n.strength * 10);
    const order = [...this.nodes].sort((a, b) => prio(b) - prio(a));
    const placed: [number, number, number, number][] = [];
    const w = new THREE.Vector3();
    const k = 1 - Math.exp(-dt * 8);
    for (const n of order) {
      if (!n.w) { n.w = n.el.offsetWidth; n.h = n.el.offsetHeight; }
      w.copy(n.label.position).add(this.group.position);
      const p = this.stage.toScreen(w);
      const x0 = p.x - n.w / 2 - 3, x1 = p.x + n.w / 2 + 3;
      const y0 = p.y - n.h * 1.35 - 2, y1 = p.y - n.h * 0.35 + 2;
      const hit = n.enter > 0.5 && placed.some(([a, b, c, d]) => x0 < c && x1 > a && y0 < d && y1 > b);
      n.vis += ((hit ? 0 : 1) - n.vis) * k;
      if (!hit && n.enter > 0.5) placed.push([x0, y0, x1, y1]);
      n.el.style.opacity = String(this.opacity * n.enter * n.vis * (n.kind === 'opposite' ? 0.75 : 1));
      n.el.style.pointerEvents = n.vis > 0.5 ? 'auto' : 'none';
    }
  }

  private tick(dt: number) {
    const pm = this.points.material as THREE.ShaderMaterial;
    const cam = this.stage.camera;
    pm.uniforms.uScale.value = (this.stage.size.h * this.stage.renderer.getPixelRatio()) / (2 * Math.tan((cam.fov * Math.PI) / 360));
    pm.uniforms.uOpacity.value = this.opacity;
    (this.lines.material as THREE.LineBasicMaterial).opacity = this.opacity * 0.9;
    if (!this.nodes.length) return;
    const pos = this.points.geometry.getAttribute('position') as THREE.BufferAttribute;
    const col = this.points.geometry.getAttribute('color') as THREE.BufferAttribute;
    const size = this.points.geometry.getAttribute('aSize') as THREE.BufferAttribute;
    const lpos = this.lines.geometry.getAttribute('position') as THREE.BufferAttribute;
    const lcol = this.lines.geometry.getAttribute('color') as THREE.BufferAttribute;
    const v = new THREE.Vector3();
    const c = new THREE.Color();
    const t = performance.now() / 1000;
    this.nodes.forEach((n, i) => {
      n.r += (n.rTarget - n.r) * (1 - Math.exp(-dt * 3.2));
      const r = 11 + (n.r - 11) * n.enter;
      const wob = Math.sin(t * 0.5 + i) * 0.04;
      const a = n.angle + wob * 0.3;
      v.copy(this.right).multiplyScalar(Math.cos(a) * r)
        .addScaledVector(this.up, Math.sin(a) * r * 0.78)
        .addScaledVector(this.fwd, n.depth + wob);
      pos.setXYZ(i, v.x, v.y, v.z);
      n.label.position.copy(v);
      const base = n.kind === 'you' ? COLORS.posHi : n.blue ? COLORS.neg : COLORS.pos;
      const k = (0.35 + 0.65 * n.strength) * n.enter;
      c.copy(base).multiplyScalar(k * (n.kind === 'you' ? 1.6 : 1));
      col.setXYZ(i, c.r, c.g, c.b);
      size.setX(i, (n.kind === 'you' ? 0.32 : 0.14 + 0.14 * n.strength) * n.enter);
      lpos.setXYZ(i * 2, 0, 0, 0);
      lpos.setXYZ(i * 2 + 1, v.x, v.y, v.z);
      const lk = n.kind === 'opposite' ? 0.06 : 0.1 + 0.25 * n.strength;
      lcol.setXYZ(i * 2, c.r * 0.05, c.g * 0.05, c.b * 0.05);
      lcol.setXYZ(i * 2 + 1, base.r * lk * n.enter, base.g * lk * n.enter, base.b * lk * n.enter);
    });
    pos.needsUpdate = col.needsUpdate = size.needsUpdate = lpos.needsUpdate = lcol.needsUpdate = true;
    this.declutter(dt);
    this.points.geometry.setDrawRange(0, this.nodes.length);
    this.lines.geometry.setDrawRange(0, this.nodes.length * 2);
  }
}
