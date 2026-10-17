// Tiny tween engine (driven by the stage's frame loop) and eased camera flights.

import * as THREE from 'three';
import type { OrbitControls } from 'three/addons/controls/OrbitControls.js';

export const ease = {
  linear: (t: number) => t,
  inOutCubic: (t: number) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2),
  outCubic: (t: number) => 1 - Math.pow(1 - t, 3),
  outQuint: (t: number) => 1 - Math.pow(1 - t, 5),
  inOutSine: (t: number) => -(Math.cos(Math.PI * t) - 1) / 2,
  outBack: (t: number) => {
    const c1 = 1.70158, c3 = c1 + 1;
    return 1 + c3 * Math.pow(t - 1, 3) + c1 * Math.pow(t - 1, 2);
  },
};

interface Tween {
  start: number;
  duration: number;
  ease: (t: number) => number;
  update: (k: number) => void;
  resolve: () => void;
  key?: string;
  cancelled?: boolean;
}

export class Tweens {
  private list: Tween[] = [];
  private now = performance.now();
  /** Global multiplier for durations (reduced motion). */
  speed = 1;

  /** Animate k from 0→1. A tween with the same key cancels the previous one. */
  run(duration: number, update: (k: number) => void, opts: { ease?: (t: number) => number; key?: string; delay?: number } = {}): Promise<void> {
    if (opts.key) {
      for (const t of this.list) if (t.key === opts.key) t.cancelled = true;
    }
    return new Promise((resolve) => {
      this.list.push({
        start: this.now + (opts.delay ?? 0) * this.speed,
        duration: Math.max(1, duration * this.speed),
        ease: opts.ease ?? ease.inOutCubic,
        update,
        resolve,
        key: opts.key,
      });
    });
  }

  update(now: number) {
    this.now = now;
    const keep: Tween[] = [];
    for (const t of this.list) {
      if (t.cancelled) { t.resolve(); continue; }
      if (now < t.start) { keep.push(t); continue; }
      const k = Math.min(1, (now - t.start) / t.duration);
      t.update(t.ease(k));
      if (k < 1) keep.push(t);
      else t.resolve();
    }
    this.list = keep;
  }
}

let flightId = 0;

export interface Shot {
  position: THREE.Vector3;
  target: THREE.Vector3;
  fov?: number;
}

/**
 * Fly the camera along a quadratic bezier that bows away from the straight
 * line (so it swoops instead of sliding), easing target and FOV together.
 */
export function flyTo(
  tweens: Tweens,
  camera: THREE.PerspectiveCamera,
  controls: OrbitControls,
  shot: Shot,
  opts: { duration?: number; arc?: number } = {},
): Promise<void> {
  const p0 = camera.position.clone();
  const t0 = controls.target.clone();
  const f0 = camera.fov;
  const p1 = shot.position.clone();
  const t1 = shot.target.clone();
  const f1 = shot.fov ?? f0;
  const dist = p0.distanceTo(p1);
  const mid = p0.clone().lerp(p1, 0.5);
  // bow outward from the tower axis and upward a little
  const out = new THREE.Vector3(mid.x, 0, mid.z);
  if (out.lengthSq() < 1e-4) out.set(1, 0, 0);
  out.normalize().multiplyScalar(dist * (opts.arc ?? 0.22));
  const ctrl = mid.add(out).add(new THREE.Vector3(0, dist * 0.08, 0));
  const tmp = new THREE.Vector3();
  const id = ++flightId;
  // the caller decides whether auto-orbit resumes afterwards
  controls.enabled = false;
  controls.autoRotate = false;
  return tweens
    .run(
      opts.duration ?? 2200,
      (k) => {
        const a = (1 - k) * (1 - k), b = 2 * (1 - k) * k, c = k * k;
        camera.position.set(
          a * p0.x + b * ctrl.x + c * p1.x,
          a * p0.y + b * ctrl.y + c * p1.y,
          a * p0.z + b * ctrl.z + c * p1.z,
        );
        tmp.copy(t0).lerp(t1, ease.inOutSine(k));
        controls.target.copy(tmp);
        camera.lookAt(tmp);
        camera.fov = f0 + (f1 - f0) * k;
        camera.updateProjectionMatrix();
      },
      { ease: ease.inOutCubic, key: 'camera' },
    )
    .then(() => {
      if (id !== flightId) return; // superseded by a newer flight
      controls.target.copy(t1);
      controls.enabled = true;
      controls.update();
    });
}
