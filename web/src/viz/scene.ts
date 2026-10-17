// Renderer, camera, bloom, CSS2D label layer, orbit controls and the frame loop.

import * as THREE from 'three';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { ShaderPass } from 'three/addons/postprocessing/ShaderPass.js';
import { CSS2DRenderer } from 'three/addons/renderers/CSS2DRenderer.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { Tweens } from './camera';

export type TickFn = (t: number, dt: number) => void;

export class Stage {
  readonly renderer: THREE.WebGLRenderer;
  readonly labels: CSS2DRenderer;
  readonly scene = new THREE.Scene();
  readonly camera: THREE.PerspectiveCamera;
  readonly composer: EffectComposer;
  readonly bloom: UnrealBloomPass;
  readonly controls: OrbitControls;
  readonly tweens = new Tweens();
  /** Shift of the rendered image in px (positive x = content moves right, positive y = down). */
  readonly shift = new THREE.Vector2();
  private shiftTarget = new THREE.Vector2();
  private ticks: TickFn[] = [];
  private last = performance.now();
  private width = 1;
  private height = 1;
  readonly reducedMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;

  constructor(private host: HTMLElement) {
    this.renderer = new THREE.WebGLRenderer({ antialias: false, powerPreference: 'high-performance', alpha: false });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.renderer.setClearColor(0x04050a, 1);
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.0;
    host.appendChild(this.renderer.domElement);

    this.labels = new CSS2DRenderer();
    this.labels.domElement.className = 'labels';
    host.appendChild(this.labels.domElement);

    this.camera = new THREE.PerspectiveCamera(45, 1, 0.05, 2000);
    this.camera.position.set(34, 12, 40);

    this.composer = new EffectComposer(this.renderer);
    this.composer.addPass(new RenderPass(this.scene, this.camera));
    this.bloom = new UnrealBloomPass(new THREE.Vector2(256, 256), 0.9, 0.42, 0.14);
    this.composer.addPass(this.bloom);
    this.composer.addPass(new OutputPass());
    // a whisper of noise after tone mapping hides 8-bit banding in the dark glows
    const dither = new ShaderPass({
      uniforms: { tDiffuse: { value: null }, uTime: { value: 0 } },
      vertexShader: /* glsl */ `varying vec2 vUv; void main() { vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
      fragmentShader: /* glsl */ `
        uniform sampler2D tDiffuse; uniform float uTime; varying vec2 vUv;
        float rand(vec2 co) { return fract(sin(dot(co, vec2(12.9898, 78.233))) * 43758.5453); }
        void main() {
          vec4 c = texture2D(tDiffuse, vUv);
          c.rgb += (rand(vUv * 1000.0 + fract(uTime)) - 0.5) * (2.0 / 255.0);
          gl_FragColor = c;
        }`,
    });
    this.composer.addPass(dither);
    this.onTick((t) => { dither.uniforms.uTime.value = t; });

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.06;
    this.controls.rotateSpeed = 0.55;
    this.controls.zoomSpeed = 0.7;
    this.controls.enablePan = false;
    this.controls.minDistance = 1.2;
    this.controls.maxDistance = 140;
    this.controls.autoRotate = !this.reducedMotion;
    this.controls.autoRotateSpeed = 0.35;

    this.addStars();
    // follow the host's real size: Safari can run this before the stylesheet
    // applies (the host is then only as tall as a default 300px canvas), and
    // no window resize event follows once the CSS lands
    new ResizeObserver(() => this.resize()).observe(host);
    window.addEventListener('resize', () => this.resize());
    this.resize();
    this.renderer.setAnimationLoop(() => this.frame());
  }

  onTick(fn: TickFn) {
    this.ticks.push(fn);
  }

  /** Slide the whole render so the subject sits beside (or above) a panel. */
  setShift(x: number, y = 0, instant = false) {
    this.shiftTarget.set(x, y);
    if (instant) {
      this.shift.set(x, y);
      this.applyShift();
    }
  }

  get size() {
    return { w: this.width, h: this.height };
  }

  /** Project a world point to CSS pixels in the stage (view offset included). */
  toScreen(p: THREE.Vector3): { x: number; y: number; visible: boolean } {
    const v = p.clone().project(this.camera);
    return {
      x: (v.x * 0.5 + 0.5) * this.width,
      y: (-v.y * 0.5 + 0.5) * this.height,
      visible: v.z < 1,
    };
  }

  private addStars() {
    const n = 2600;
    const pos = new Float32Array(n * 3);
    const seed = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const u = Math.random() * 2 - 1;
      const th = Math.random() * Math.PI * 2;
      const r = 260 + Math.random() * 380;
      const s = Math.sqrt(1 - u * u);
      pos[i * 3] = r * s * Math.cos(th);
      pos[i * 3 + 1] = r * u;
      pos[i * 3 + 2] = r * s * Math.sin(th);
      seed[i] = Math.random();
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    g.setAttribute('aSeed', new THREE.BufferAttribute(seed, 1));
    const m = new THREE.ShaderMaterial({
      uniforms: { uTime: { value: 0 }, uPR: { value: this.renderer.getPixelRatio() } },
      vertexShader: /* glsl */ `
        attribute float aSeed;
        uniform float uTime; uniform float uPR;
        varying float vA; varying vec3 vC;
        void main() {
          vec4 mv = modelViewMatrix * vec4(position, 1.0);
          gl_Position = projectionMatrix * mv;
          float tw = 0.55 + 0.45 * sin(uTime * (0.4 + aSeed) + aSeed * 60.0);
          vA = (0.18 + 0.5 * aSeed * aSeed) * tw;
          vC = mix(vec3(0.62, 0.66, 1.0), vec3(1.0, 0.86, 0.7), step(0.82, aSeed));
          gl_PointSize = (0.8 + 1.6 * aSeed * aSeed) * uPR;
        }`,
      fragmentShader: /* glsl */ `
        varying float vA; varying vec3 vC;
        void main() {
          float d = length(gl_PointCoord - 0.5);
          gl_FragColor = vec4(vC, vA * smoothstep(0.5, 0.0, d));
        }`,
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });
    const stars = new THREE.Points(g, m);
    stars.frustumCulled = false;
    this.scene.add(stars);
    this.onTick((t) => { m.uniforms.uTime.value = t; });
  }

  private resizeListeners = new Set<() => void>();

  /** Called after the stage changes size (window resize or layout change). */
  onResize(fn: () => void) {
    this.resizeListeners.add(fn);
  }

  private resize() {
    const w = this.host.clientWidth || innerWidth;
    const h = this.host.clientHeight || innerHeight;
    if (w === this.width && h === this.height) return;
    this.width = w;
    this.height = h;
    this.renderer.setSize(w, h);
    this.labels.setSize(w, h);
    this.composer.setSize(w, h);
    // bloom at reduced resolution: cheap and softer
    const pr = this.renderer.getPixelRatio();
    this.bloom.setSize(Math.round(w * pr * 0.5), Math.round(h * pr * 0.5));
    this.camera.aspect = w / h;
    this.applyShift();
    this.resizeListeners.forEach((fn) => fn());
  }

  private applyShift() {
    const { width: w, height: h } = this;
    if (this.shift.lengthSq() < 0.25) this.camera.clearViewOffset();
    else this.camera.setViewOffset(w, h, -this.shift.x, -this.shift.y, w, h);
    // the projection matrix carries the offset, so CSS2D labels follow automatically
    this.camera.updateProjectionMatrix();
  }

  private frame() {
    const now = performance.now();
    const dt = Math.min(0.25, (now - this.last) / 1000);
    this.last = now;
    const t = now / 1000;
    this.tweens.update(now);
    if (this.shift.distanceTo(this.shiftTarget) > 0.3) {
      this.shift.lerp(this.shiftTarget, 1 - Math.exp(-dt * 3.5));
      this.applyShift();
    } else if (!this.shift.equals(this.shiftTarget)) {
      this.shift.copy(this.shiftTarget);
      this.applyShift();
    }
    for (const fn of this.ticks) fn(t, dt);
    if (this.controls.enabled) this.controls.update(dt);
    this.composer.render(dt);
    this.labels.render(this.scene, this.camera);
  }
}
