/**
 * <compare-slider>: before/after comparison on the same image, two-up style.
 * Children: mark the result layer with `data-layer="after"`; anything else is "before".
 * (Astro strips `slot` attributes from markup, so slots are assigned here at runtime.)
 * The after layer is clipped at --pos (0–100%).
 * Drag with pointer, or focus the handle and use arrow keys / Home / End.
 */
const TEMPLATE = `
<style>
  :host {
    position: relative;
    display: block;
    overflow: hidden;
    background: var(--compare-bg, #000);
    user-select: none;
    touch-action: none;
    --pos: 50%;
    --line: var(--compare-line, #f2f2ee);
  }
  .layer {
    position: absolute;
    inset: 0;
  }
  slot {
    display: block;
    width: 100%;
    height: 100%;
  }
  /* !important: slotted rules lose to light-DOM author styles (Tailwind's img { height: auto }). */
  ::slotted(*) {
    width: 100% !important;
    height: 100% !important;
    max-width: 100% !important;
    display: block !important;
    object-fit: contain !important;
  }
  .layer.after {
    clip-path: inset(0 0 0 var(--pos));
  }
  .label {
    position: absolute;
    top: 10px;
    font: 11px/1 var(--font-pixel, monospace);
    letter-spacing: 0.02em;
    text-transform: uppercase;
    color: #f2f2ee;
    background: rgb(10 10 11 / 0.7);
    padding: 5px 7px;
    pointer-events: none;
  }
  .label.before { left: 10px; }
  .label.after { right: 10px; }
  .handle {
    position: absolute;
    top: 0;
    bottom: 0;
    left: var(--pos);
    width: 0;
    cursor: ew-resize;
    outline: none;
  }
  .handle::before {
    content: '';
    position: absolute;
    inset: 0 auto 0 -1px;
    width: 2px;
    background: var(--line);
    box-shadow: 0 0 0 1px rgb(0 0 0 / 0.4);
  }
  .grip {
    position: absolute;
    top: 50%;
    left: 50%;
    width: 28px;
    height: 28px;
    transform: translate(-50%, -50%);
    background: var(--line);
    display: grid;
    place-items: center;
    color: #0a0a0b;
    box-shadow: 0 2px 10px rgb(0 0 0 / 0.5);
  }
  .handle:focus-visible .grip {
    outline: 2px solid var(--compare-accent, #ffb340);
    outline-offset: 2px;
  }
  :host([data-single]) .after,
  :host([data-single]) .handle,
  :host([data-single]) .label.after { display: none; }
</style>
<div class="layer before"><slot name="before"></slot></div>
<div class="layer after"><slot name="after"></slot></div>
<span class="label before">Before</span>
<span class="label after">After</span>
<div class="handle" tabindex="0" role="slider" aria-label="Compare before and after" aria-valuemin="0" aria-valuemax="100" aria-valuenow="50">
  <div class="grip" aria-hidden="true">
    <svg width="14" height="10" viewBox="0 0 14 10" fill="currentColor" shape-rendering="crispEdges">
      <path d="M4 0v10H2V0zM0 4h2v2H0zM12 4h2v2h-2zM12 0v10h-2V0z"/>
    </svg>
  </div>
</div>`

export class CompareSlider extends HTMLElement {
  #pos = 50
  #handle!: HTMLElement
  #dragging = false

  constructor() {
    super()
    const root = this.attachShadow({ mode: 'open' })
    root.innerHTML = TEMPLATE
    this.#handle = root.querySelector('.handle')!
  }

  connectedCallback() {
    for (const child of Array.from(this.children)) {
      child.slot = (child as HTMLElement).dataset.layer === 'after' ? 'after' : 'before'
    }
    this.addEventListener('pointerdown', this.#down)
    this.addEventListener('pointermove', this.#move)
    this.addEventListener('pointerup', this.#up)
    this.addEventListener('pointercancel', this.#up)
    this.#handle.addEventListener('keydown', this.#key)
    this.position = Number(this.getAttribute('position') ?? 50)
  }

  disconnectedCallback() {
    this.removeEventListener('pointerdown', this.#down)
    this.removeEventListener('pointermove', this.#move)
    this.removeEventListener('pointerup', this.#up)
    this.removeEventListener('pointercancel', this.#up)
    this.#handle.removeEventListener('keydown', this.#key)
  }

  /** 0–100, percentage of the width revealed as "after". */
  get position() {
    return this.#pos
  }
  set position(v: number) {
    this.#pos = Math.max(0, Math.min(100, v))
    this.style.setProperty('--pos', `${this.#pos}%`)
    this.#handle.setAttribute('aria-valuenow', String(Math.round(this.#pos)))
  }

  /** Show only the before layer (no result yet). */
  set single(on: boolean) {
    this.toggleAttribute('data-single', on)
  }

  /** Animate the handle to a position. */
  reveal(to = 50, ms = 500) {
    const from = this.#pos
    const t0 = performance.now()
    const step = (now: number) => {
      const k = Math.min(1, (now - t0) / ms)
      const e = 1 - Math.pow(1 - k, 3)
      this.position = from + (to - from) * e
      if (k < 1) requestAnimationFrame(step)
    }
    requestAnimationFrame(step)
  }

  #setFromEvent(e: PointerEvent) {
    const r = this.getBoundingClientRect()
    this.position = ((e.clientX - r.left) / r.width) * 100
  }
  #down = (e: PointerEvent) => {
    if (this.hasAttribute('data-single')) return
    this.#dragging = true
    this.setPointerCapture(e.pointerId)
    this.#setFromEvent(e)
    this.#handle.focus({ preventScroll: true })
  }
  #move = (e: PointerEvent) => {
    if (this.#dragging) this.#setFromEvent(e)
  }
  #up = () => {
    this.#dragging = false
  }
  #key = (e: KeyboardEvent) => {
    const step = e.shiftKey ? 10 : 2
    if (e.key === 'ArrowLeft') this.position = this.#pos - step
    else if (e.key === 'ArrowRight') this.position = this.#pos + step
    else if (e.key === 'Home') this.position = 0
    else if (e.key === 'End') this.position = 100
    else return
    e.preventDefault()
  }
}

if (!customElements.get('compare-slider')) {
  customElements.define('compare-slider', CompareSlider)
}

/**
 * Size the host to the image: same aspect ratio, capped at `maxHeightRem`, never wider than
 * its container, centered. The box then matches the picture exactly, so the checkerboard and the
 * clip line sit on the image and not on letterbox space.
 */
export function fitCompareToImage(el: CompareSlider, img: HTMLImageElement, maxHeightRem = 34) {
  const apply = () => {
    if (img.naturalWidth && img.naturalHeight) {
      const ratio = img.naturalWidth / img.naturalHeight
      el.style.aspectRatio = `${img.naturalWidth} / ${img.naturalHeight}`
      el.style.width = `min(100%, calc(${maxHeightRem}rem * ${ratio.toFixed(4)}))`
      el.style.marginInline = 'auto'
    }
  }
  if (img.complete) apply()
  else img.addEventListener('load', apply, { once: true })
}
