// Token chips that leave the input box and dive into the base of the tower.

import { $, h, tokenLabel } from './dom';

export async function flyTokens(
  tokens: string[],
  from: DOMRect,
  target: { x: number; y: number },
  reduced: boolean,
): Promise<void> {
  const layer = $('flight-layer');
  const chips = tokens.slice(0, 24).map((t) => h('div', { class: 'fly-chip', text: tokenLabel(t) }));
  chips.forEach((c) => layer.append(c));
  // lay chips out along the input, then measure them
  const widths = chips.map((c) => c.getBoundingClientRect().width);
  const gap = 6;
  const total = widths.reduce((a, b) => a + b, 0) + gap * (chips.length - 1);
  let x = from.left + from.width / 2 - total / 2;
  const y = from.top + from.height / 2 - 14;
  const anims = chips.map((c, i) => {
    const sx = x;
    x += widths[i] + gap;
    const dx = target.x - (sx + widths[i] / 2);
    const dy = target.y - (y + 14);
    const lift = -60 - Math.random() * 60;
    const dur = reduced ? 300 : 1100 + i * 40;
    return c.animate(
      [
        { transform: `translate(${sx}px, ${y}px) scale(0.6)`, opacity: 0 },
        { transform: `translate(${sx}px, ${y - 8}px) scale(1)`, opacity: 1, offset: 0.18 },
        { transform: `translate(${sx + dx * 0.45}px, ${y + dy * 0.35 + lift}px) scale(0.85)`, opacity: 1, offset: 0.55 },
        { transform: `translate(${sx + dx}px, ${y + dy}px) scale(0.15)`, opacity: 0 },
      ],
      { duration: dur, delay: reduced ? 0 : i * 55, easing: 'cubic-bezier(.45,.05,.35,1)', fill: 'both' },
    ).finished;
  });
  await Promise.all(anims);
  chips.forEach((c) => c.remove());
}
