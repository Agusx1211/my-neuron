// The big "type anything" input and its example chips.

import { $, conceal, h, reveal } from './dom';

const EXAMPLES = ['Lionel Messi', 'pizza', 'nostalgia', 'Tokyo', 'agusx1211', 'black holes', 'Taylor Swift'];

export class Prompt {
  private root = $('prompt');
  private input = $<HTMLInputElement>('prompt-input');
  private go = $<HTMLButtonElement>('prompt-go');

  constructor(onSubmit: (text: string) => void) {
    $('prompt-form').addEventListener('submit', (e) => {
      e.preventDefault();
      const text = this.input.value.trim();
      if (text) onSubmit(text);
      else this.input.focus();
    });
    const ex = $('examples');
    for (const t of EXAMPLES) {
      const b = h('button', { class: 'chip', text: t, attrs: { type: 'button' } });
      b.addEventListener('click', () => { this.input.value = t; onSubmit(t); });
      ex.append(b);
    }
  }

  get inputRect() {
    return this.input.getBoundingClientRect();
  }

  get value() {
    return this.input.value;
  }

  show() {
    this.input.placeholder = innerWidth <= 760 ? 'Your name, a nickname, anything…' : 'Type your name, a nickname, anything…';
    reveal(this.root);
    this.go.disabled = false;
    setTimeout(() => this.input.focus({ preventScroll: true }), 300);
  }

  hide() {
    this.go.disabled = true;
    return conceal(this.root, 500);
  }
}
