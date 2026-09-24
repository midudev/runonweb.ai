/**
 * Hero background: an ordered-dither field computed every frame by the browser.
 * Low-res canvas scaled up with image-rendering: pixelated, so the pixels match Geist Pixel.
 */
import { onPage } from './lifecycle'

onPage(() => {
const canvas = document.querySelector<HTMLCanvasElement>('[data-dither]')

if (canvas) {
  const ctx = canvas.getContext('2d', { alpha: true })!
  const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches
  const accent = getComputedStyle(document.documentElement).getPropertyValue('--color-accent').trim() || '#ffb340'

  const bayer = [
    [0, 8, 2, 10],
    [12, 4, 14, 6],
    [3, 11, 1, 9],
    [15, 7, 13, 5],
  ].map((row) => row.map((v) => (v + 0.5) / 16))

  let W = 0
  let H = 0
  let img: ImageData
  let px: Uint8ClampedArray
  const target = { x: 0.7, y: 0.4 }
  const pointer = { x: 0.7, y: 0.4 }
  const [ar, ag, ab] = hexToRgb(accent)

  function resize() {
    const rect = canvas!.getBoundingClientRect()
    const scale = 6 // CSS px per canvas px
    W = Math.max(32, Math.floor(rect.width / scale))
    H = Math.max(24, Math.floor(rect.height / scale))
    canvas!.width = W
    canvas!.height = H
    img = ctx.createImageData(W, H)
    px = img.data
  }

  function field(x: number, y: number, t: number) {
    // Soft ridge + pointer glow, all cheap trig.
    const nx = x / W
    const ny = y / H
    const dx = nx - pointer.x
    const dy = (ny - pointer.y) * (H / W)
    const glow = Math.exp(-(dx * dx + dy * dy) * 10)
    const wave = 0.5 + 0.5 * Math.sin(nx * 7 + t * 0.6 + Math.sin(ny * 5 - t * 0.4) * 1.3)
    const fade = Math.pow(nx, 1.6) * (1 - ny * 0.35)
    return Math.min(1, wave * fade * 0.5 + glow * 0.55)
  }

  let raf = 0
  function frame(now: number) {
    const t = now / 1000
    pointer.x += (target.x - pointer.x) * 0.08
    pointer.y += (target.y - pointer.y) * 0.08
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const v = field(x, y, t)
        const on = v > bayer[y & 3]![x & 3]!
        const i = (y * W + x) * 4
        px[i] = ar
        px[i + 1] = ag
        px[i + 2] = ab
        px[i + 3] = on ? 255 * Math.min(1, 0.3 + v * 0.5) : 0
      }
    }
    ctx.putImageData(img, 0, 0)
    if (!reduce) raf = requestAnimationFrame(frame)
  }

  function hexToRgb(hex: string): [number, number, number] {
    const h = hex.replace('#', '')
    const n = parseInt(h.length === 3 ? h.split('').map((c) => c + c).join('') : h, 16)
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255]
  }

  resize()
  addEventListener('resize', resize, { passive: true })
  const host = canvas.parentElement ?? canvas
  host.addEventListener(
    'pointermove',
    (e) => {
      const r = canvas!.getBoundingClientRect()
      target.x = (e.clientX - r.left) / r.width
      target.y = (e.clientY - r.top) / r.height
    },
    { passive: true }
  )
  raf = requestAnimationFrame(frame)
  return () => {
    cancelAnimationFrame(raf)
    removeEventListener('resize', resize)
  }
}
})
