// The focused neuron, rendered up close: a living core with fresnel glow,
// orbiting rings, a soft halo and slow light shafts.

import * as THREE from 'three';
import type { Stage } from './scene';
import { ease } from './camera';
import { COLORS } from './tower';

const noise = /* glsl */ `
  vec3 mod289(vec3 x){return x-floor(x*(1.0/289.0))*289.0;}
  vec4 mod289(vec4 x){return x-floor(x*(1.0/289.0))*289.0;}
  vec4 permute(vec4 x){return mod289(((x*34.0)+1.0)*x);}
  vec4 taylorInvSqrt(vec4 r){return 1.79284291400159-0.85373472095314*r;}
  float snoise(vec3 v){
    const vec2 C=vec2(1.0/6.0,1.0/3.0); const vec4 D=vec4(0.0,0.5,1.0,2.0);
    vec3 i=floor(v+dot(v,C.yyy)); vec3 x0=v-i+dot(i,C.xxx);
    vec3 g=step(x0.yzx,x0.xyz); vec3 l=1.0-g; vec3 i1=min(g.xyz,l.zxy); vec3 i2=max(g.xyz,l.zxy);
    vec3 x1=x0-i1+C.xxx; vec3 x2=x0-i2+C.yyy; vec3 x3=x0-D.yyy;
    i=mod289(i);
    vec4 p=permute(permute(permute(i.z+vec4(0.0,i1.z,i2.z,1.0))+i.y+vec4(0.0,i1.y,i2.y,1.0))+i.x+vec4(0.0,i1.x,i2.x,1.0));
    float n_=0.142857142857; vec3 ns=n_*D.wyz-D.xzx;
    vec4 j=p-49.0*floor(p*ns.z*ns.z); vec4 x_=floor(j*ns.z); vec4 y_=floor(j-7.0*x_);
    vec4 x=x_*ns.x+ns.yyyy; vec4 y=y_*ns.x+ns.yyyy; vec4 h=1.0-abs(x)-abs(y);
    vec4 b0=vec4(x.xy,y.xy); vec4 b1=vec4(x.zw,y.zw);
    vec4 s0=floor(b0)*2.0+1.0; vec4 s1=floor(b1)*2.0+1.0; vec4 sh=-step(h,vec4(0.0));
    vec4 a0=b0.xzyw+s0.xzyw*sh.xxyy; vec4 a1=b1.xzyw+s1.xzyw*sh.zzww;
    vec3 p0=vec3(a0.xy,h.x); vec3 p1=vec3(a0.zw,h.y); vec3 p2=vec3(a1.xy,h.z); vec3 p3=vec3(a1.zw,h.w);
    vec4 norm=taylorInvSqrt(vec4(dot(p0,p0),dot(p1,p1),dot(p2,p2),dot(p3,p3)));
    p0*=norm.x; p1*=norm.y; p2*=norm.z; p3*=norm.w;
    vec4 m=max(0.6-vec4(dot(x0,x0),dot(x1,x1),dot(x2,x2),dot(x3,x3)),0.0); m=m*m;
    return 42.0*dot(m*m,vec4(dot(p0,x0),dot(p1,x1),dot(p2,x2),dot(p3,x3)));
  }
`;

function radialTexture(stops: [number, string][], size = 256): THREE.Texture {
  const c = document.createElement('canvas');
  c.width = c.height = size;
  const g = c.getContext('2d')!;
  const grad = g.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
  for (const [o, col] of stops) grad.addColorStop(o, col);
  g.fillStyle = grad;
  g.fillRect(0, 0, size, size);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

function shaftTexture(size = 512): THREE.Texture {
  const c = document.createElement('canvas');
  c.width = c.height = size;
  const g = c.getContext('2d')!;
  g.translate(size / 2, size / 2);
  g.globalCompositeOperation = 'lighter';
  let seed = 7;
  const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
  for (let i = 0; i < 26; i++) {
    const a = rnd() * Math.PI * 2;
    const w = 0.012 + rnd() * 0.05;
    const len = size * (0.28 + rnd() * 0.22);
    const grad = g.createLinearGradient(0, 0, Math.cos(a) * len, Math.sin(a) * len);
    grad.addColorStop(0, `rgba(255,255,255,${0.10 + rnd() * 0.16})`);
    grad.addColorStop(1, 'rgba(255,255,255,0)');
    g.fillStyle = grad;
    g.beginPath();
    g.moveTo(0, 0);
    g.arc(0, 0, len, a - w, a + w);
    g.closePath();
    g.fill();
  }
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

export class Hero {
  readonly group = new THREE.Group();
  private core: THREE.Mesh;
  private coreMat: THREE.ShaderMaterial;
  private halo: THREE.Sprite;
  private shafts: THREE.Sprite;
  private shafts2: THREE.Sprite;
  private rings: THREE.Mesh[] = [];
  private sparks: THREE.Points;
  private visible = 0;

  constructor(private stage: Stage) {
    this.coreMat = new THREE.ShaderMaterial({
      uniforms: { uTime: { value: 0 }, uColor: { value: COLORS.pos.clone() }, uHot: { value: COLORS.posHi.clone() } },
      vertexShader: noise + /* glsl */ `
        uniform float uTime;
        varying vec3 vN; varying vec3 vV; varying float vNoise;
        void main() {
          float n = snoise(normal * 2.2 + vec3(0.0, uTime * 0.6, 0.0));
          vNoise = n;
          vec3 p = position * (1.0 + 0.06 * n);
          vec4 mv = modelViewMatrix * vec4(p, 1.0);
          vN = normalize(normalMatrix * normal);
          vV = normalize(-mv.xyz);
          gl_Position = projectionMatrix * mv;
        }`,
      fragmentShader: /* glsl */ `
        uniform vec3 uColor; uniform vec3 uHot;
        varying vec3 vN; varying vec3 vV; varying float vNoise;
        void main() {
          float f = pow(1.0 - max(dot(vN, vV), 0.0), 2.2);
          float bands = 0.5 + 0.5 * sin(vNoise * 6.0);
          vec3 c = mix(uColor * 0.65, uHot * 0.95, 0.25 + 0.45 * bands);
          c += uColor * f * 1.5;
          gl_FragColor = vec4(c, 1.0);
        }`,
    });
    this.core = new THREE.Mesh(new THREE.IcosahedronGeometry(0.15, 24), this.coreMat);
    this.group.add(this.core);

    this.halo = new THREE.Sprite(new THREE.SpriteMaterial({
      map: radialTexture([[0, 'rgba(255,255,255,0.9)'], [0.12, 'rgba(255,255,255,0.45)'], [0.4, 'rgba(255,255,255,0.08)'], [1, 'rgba(255,255,255,0)']]),
      color: COLORS.pos.clone(), transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    }));
    this.halo.scale.setScalar(1.1);
    this.halo.material.opacity = 0.7;
    this.group.add(this.halo);

    const shaftMap = shaftTexture();
    this.shafts = new THREE.Sprite(new THREE.SpriteMaterial({
      map: shaftMap, color: COLORS.pos.clone(), transparent: true, opacity: 0.22, depthWrite: false, blending: THREE.AdditiveBlending,
    }));
    this.shafts.scale.setScalar(2.6);
    this.group.add(this.shafts);
    this.shafts2 = new THREE.Sprite(this.shafts.material.clone());
    this.shafts2.material.opacity = 0.14;
    this.shafts2.scale.setScalar(1.9);
    this.group.add(this.shafts2);

    [0.3, 0.44, 0.62].forEach((r, i) => {
      const ring = new THREE.Mesh(
        new THREE.TorusGeometry(r, 0.0045 + i * 0.001, 6, 160),
        new THREE.MeshBasicMaterial({ color: COLORS.pos.clone(), transparent: true, opacity: 0.8 - i * 0.18, blending: THREE.AdditiveBlending, depthWrite: false }),
      );
      ring.rotation.set(Math.PI / 2 + (i - 1) * 0.5, i * 0.8, 0);
      ring.userData.speed = (i % 2 ? -1 : 1) * (0.35 + i * 0.12);
      this.rings.push(ring);
      this.group.add(ring);
    });

    const n = 90;
    const sp = new Float32Array(n * 3);
    const sd = new Float32Array(n);
    for (let i = 0; i < n; i++) sd[i] = Math.random();
    const sg = new THREE.BufferGeometry();
    sg.setAttribute('position', new THREE.BufferAttribute(sp, 3));
    sg.setAttribute('aSeed', new THREE.BufferAttribute(sd, 1));
    this.sparks = new THREE.Points(sg, new THREE.ShaderMaterial({
      uniforms: { uTime: { value: 0 }, uColor: { value: COLORS.posHi.clone() }, uPR: { value: stage.renderer.getPixelRatio() } },
      vertexShader: /* glsl */ `
        attribute float aSeed; uniform float uTime; uniform float uPR; varying float vA;
        void main() {
          float a = aSeed * 6.2831 + uTime * (0.25 + aSeed * 0.5) * (aSeed > 0.5 ? 1.0 : -1.0);
          float r = 0.3 + aSeed * 0.55;
          float tilt = (fract(aSeed * 7.13) - 0.5) * 1.6;
          vec3 p = vec3(cos(a) * r, sin(a) * r * tilt, sin(a) * r);
          vec4 mv = modelViewMatrix * vec4(p, 1.0);
          gl_Position = projectionMatrix * mv;
          gl_PointSize = (1.5 + 2.5 * fract(aSeed * 13.7)) * uPR;
          vA = 0.5 + 0.5 * sin(uTime * 3.0 + aSeed * 40.0);
        }`,
      fragmentShader: /* glsl */ `
        uniform vec3 uColor; varying float vA;
        void main() { float d = length(gl_PointCoord - 0.5); gl_FragColor = vec4(uColor * 2.0, vA * smoothstep(0.5, 0.0, d)); }`,
      transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    }));
    this.sparks.frustumCulled = false;
    this.group.add(this.sparks);

    this.group.visible = false;
    this.group.scale.setScalar(0.001);
    stage.scene.add(this.group);
    stage.onTick((t, dt) => this.tick(t, dt));
  }

  setPole(pole: 1 | -1) {
    const c = pole > 0 ? COLORS.pos : COLORS.neg;
    const hot = pole > 0 ? COLORS.posHi : COLORS.negHi;
    this.coreMat.uniforms.uColor.value.copy(c);
    this.coreMat.uniforms.uHot.value.copy(hot);
    this.halo.material.color.copy(c);
    this.shafts.material.color.copy(c);
    this.shafts2.material.color.copy(hot);
    for (const r of this.rings) (r.material as THREE.MeshBasicMaterial).color.copy(c);
    ((this.sparks.material as THREE.ShaderMaterial).uniforms.uColor.value as THREE.Color).copy(hot);
  }

  /** Bloom into existence at a world position. */
  show(at: THREE.Vector3, pole: 1 | -1) {
    this.setPole(pole);
    this.group.position.copy(at);
    this.group.visible = true;
    const from = this.visible;
    return this.stage.tweens.run(1100, (k) => {
      this.visible = from + (1 - from) * k;
      this.group.scale.setScalar(Math.max(0.001, this.visible));
    }, { ease: ease.outBack, key: 'hero' });
  }

  /** Glide to a new neuron (runner-up clicked). */
  moveTo(at: THREE.Vector3, pole: 1 | -1, duration = 1800) {
    this.setPole(pole);
    const from = this.group.position.clone();
    this.group.visible = true;
    return this.stage.tweens.run(duration, (k) => {
      this.group.position.lerpVectors(from, at, k);
      const s = 1 - Math.sin(k * Math.PI) * 0.6;
      this.group.scale.setScalar(Math.max(0.001, s));
    }, { ease: ease.inOutCubic, key: 'hero' }).then(() => { this.visible = 1; });
  }

  hide(duration = 600) {
    const from = this.visible;
    return this.stage.tweens.run(duration, (k) => {
      this.visible = from * (1 - k);
      this.group.scale.setScalar(Math.max(0.001, this.visible));
    }, { ease: ease.outCubic, key: 'hero' }).then(() => { this.group.visible = false; });
  }

  get position() {
    return this.group.position;
  }

  private tick(t: number, dt: number) {
    if (!this.group.visible) return;
    this.coreMat.uniforms.uTime.value = t;
    (this.sparks.material as THREE.ShaderMaterial).uniforms.uTime.value = t;
    for (const r of this.rings) r.rotation.z += dt * (r.userData.speed as number);
    this.shafts.material.rotation += dt * 0.05;
    this.shafts2.material.rotation -= dt * 0.08;
    const breathe = 1 + Math.sin(t * 1.6) * 0.06;
    this.halo.scale.setScalar(1.1 * breathe);
  }
}
