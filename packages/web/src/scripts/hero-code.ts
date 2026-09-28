/**
 * Rotates the hero code panel through one snippet per model.
 * Chips jump to a model and restart the clock; hovering or focusing pauses it.
 * With reduced motion the panel stays put and only responds to chips.
 */
import { onPage } from './lifecycle'

onPage(() => {
const root = document.querySelector<HTMLElement>('[data-hero-code]')

if (root) {
  const slides = [...root.querySelectorAll<HTMLElement>('[data-hero-code-slide]')]
  const chips = [...root.querySelectorAll<HTMLButtonElement>('[data-hero-code-chip]')]
  const title = root.querySelector<HTMLElement>('[data-hero-code-title]')
  const index = root.querySelector<HTMLElement>('[data-hero-code-index]')
  const progress = root.querySelector<HTMLElement>('.hero-code-progress')
  const interval = Number(root.dataset.interval) || 5000
  const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches
  const total = slides.length
  let current = slides.findIndex((s) => s.hasAttribute('data-on'))
  if (current < 0) current = 0
  let timer: ReturnType<typeof setTimeout> | undefined
  let paused = false

  root.style.setProperty('--interval', `${interval}ms`)

  function show(next: number) {
    const i = (next + total) % total
    // Already painted (SSR or this exact chip). Re-applying data-on
    // restarts the typewriter and makes the first snippet play twice.
    if (i === current && slides[i]?.hasAttribute('data-on')) return
    current = i
    slides.forEach((s, j) => {
      const on = j === current
      s.toggleAttribute('data-on', on)
      s.toggleAttribute('data-typed', on)
      if (on) s.removeAttribute('aria-hidden')
      else s.setAttribute('aria-hidden', 'true')
    })
    chips.forEach((c, j) => c.setAttribute('aria-pressed', j === current ? 'true' : 'false'))
    const chip = chips[current]
    if (chip) {
      root!.style.setProperty('--hue', chip.dataset.hue ?? '')
      if (title) title.textContent = `${chip.textContent?.trim()} · ${chip.title}`
    }
    if (index) index.textContent = `${String(current + 1).padStart(2, '0')} / ${String(total).padStart(2, '0')}`
  }

  function restartProgress() {
    if (!progress) return
    root!.removeAttribute('data-running')
    // Force a reflow so the animation restarts from zero.
    void progress.offsetWidth
    root!.setAttribute('data-running', '')
  }

  function schedule() {
    clearTimeout(timer)
    if (reduce || document.hidden) return
    restartProgress()
    timer = setTimeout(() => {
      show(current + 1)
      schedule()
    }, interval)
  }

  chips.forEach((chip, i) =>
    chip.addEventListener('click', () => {
      show(i)
      schedule()
    }),
  )

  const pause = () => {
    if (paused) return
    paused = true
    clearTimeout(timer)
    root!.setAttribute('data-paused', '')
  }
  const resume = () => {
    if (!paused) return
    paused = false
    root!.removeAttribute('data-paused')
    schedule()
  }
  root.addEventListener('pointerenter', pause)
  root.addEventListener('pointerleave', resume)
  root.addEventListener('focusin', pause)
  root.addEventListener('focusout', resume)
  const onVisibility = () => {
    if (document.hidden) clearTimeout(timer)
    else schedule()
  }
  document.addEventListener('visibilitychange', onVisibility)

  show(current)
  schedule()
  return () => {
    clearTimeout(timer)
    document.removeEventListener('visibilitychange', onVisibility)
  }
}
})
