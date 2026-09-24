/**
 * Pixel-dithered waveform player. Same Bayer field as the homepage hero,
 * driven by the audio envelope instead of a pointer glow.
 */
const BAYER = [
  [0, 8, 2, 10],
  [12, 4, 14, 6],
  [3, 11, 1, 9],
  [15, 7, 13, 5],
].map((row) => row.map((v) => (v + 0.5) / 16))

const SCALE = 5

export type WaveformPlayer = {
  load(src: string, opts?: { play?: boolean }): Promise<void>
  loadBlob(blob: Blob, opts?: { play?: boolean }): Promise<void>
  play(): void
  pause(): void
  setPlaying(play: boolean): void
  live(analyser: AnalyserNode | null): void
  dispose(): void
  get currentTime(): number
  get duration(): number
  readonly audio: HTMLAudioElement
}

export function createWaveform(opts: {
  canvas: HTMLCanvasElement
  audio: HTMLAudioElement
  playBtn?: HTMLButtonElement | null
  timeEl?: HTMLElement | null
}): WaveformPlayer {
  const { canvas, audio, playBtn, timeEl } = opts
  const ctx = canvas.getContext('2d', { alpha: true })
  if (!ctx) throw new Error('Could not get 2d context')

  const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches
  let W = 0
  let H = 0
  let img: ImageData
  let px: Uint8ClampedArray
  let source: Float32Array<ArrayBufferLike> = new Float32Array(0)
  let peaks: Float32Array<ArrayBufferLike> = new Float32Array(0)
  let livePeaks: number[] = []
  let analyser: AnalyserNode | null = null
  let analyserBuf: Float32Array<ArrayBuffer> | null = null
  let raf = 0
  let disposed = false
  const hue = { r: 94, g: 231, b: 255 }

  function readHue() {
    const raw =
      getComputedStyle(canvas).getPropertyValue('--hue').trim() ||
      getComputedStyle(document.documentElement).getPropertyValue('--color-accent').trim() ||
      '#5ee7ff'
    const [r, g, b] = hexToRgb(raw)
    hue.r = r
    hue.g = g
    hue.b = b
  }

  function resize() {
    const rect = canvas.getBoundingClientRect()
    W = Math.max(48, Math.floor(rect.width / SCALE))
    H = Math.max(10, Math.floor(rect.height / SCALE))
    canvas.width = W
    canvas.height = H
    if (!ctx) return
    img = ctx.createImageData(W, H)
    px = img.data
    peaks = source.length ? resample(source, W) : new Float32Array(W)
  }

  function draw(now = performance.now()) {
    if (!px || !W || !H) return
    const dur = audio.duration
    const progress =
      analyser ? 1 : Number.isFinite(dur) && dur > 0 ? audio.currentTime / dur : 0
    const playhead = Math.round(progress * (W - 1))
    const t = now / 1000
    const amp = analyser ? livePeaks : peaks
    const mid = (H - 1) / 2

    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const i = (y * W + x) * 4
        const a = amp[x] ?? 0.08
        const half = Math.max(1.1, a * mid * 0.95)
        const dist = Math.abs(y - mid)
        const inside = dist <= half
        const edge = 1 - Math.min(1, dist / Math.max(0.5, half))
        const played = x <= playhead
        let v = inside ? 0.22 + edge * (played ? 0.85 : 0.4) : played ? 0.06 : 0.02
        if (x === playhead) v = Math.max(v, 0.95)
        if (!reduce && audio.paused === false && inside) {
          v += 0.08 * Math.sin(t * 6 + x * 0.35)
        }
        const on = v > (BAYER[y & 3]![x & 3] ?? 0.5)
        px[i] = hue.r
        px[i + 1] = hue.g
        px[i + 2] = hue.b
        px[i + 3] = on ? Math.round(255 * Math.min(1, 0.25 + v * 0.75)) : 0
      }
    }
    if (!ctx || !img) return
    ctx.putImageData(img, 0, 0)
  }

  function tick(now: number) {
    if (disposed) return
    if (analyser && analyserBuf) {
      analyser.getFloatTimeDomainData(analyserBuf)
      let rms = 0
      for (let i = 0; i < analyserBuf.length; i++) rms += analyserBuf[i]! * analyserBuf[i]!
      livePeaks.push(Math.min(1, Math.sqrt(rms / analyserBuf.length) * 3.2))
      if (livePeaks.length > W) livePeaks.shift()
    }
    draw(now)
    paintTime()
    if (!reduce && (!audio.paused || analyser)) raf = requestAnimationFrame(tick)
    else raf = 0
  }

  function startLoop() {
    if (raf || disposed) return
    raf = requestAnimationFrame(tick)
  }

  function paintTime() {
    if (!timeEl) return
    const dur = Number.isFinite(audio.duration) ? audio.duration : 0
    timeEl.textContent = `${fmt(audio.currentTime)} / ${fmt(dur)}`
  }

  function syncPlayBtn() {
    if (playBtn) playBtn.textContent = audio.paused ? 'Play' : 'Pause'
  }

  async function decodePeaks(buffer: ArrayBuffer) {
    const AC =
      globalThis.AudioContext ??
      (globalThis as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext
    const ac = new AC()
    try {
      const decoded = await ac.decodeAudioData(buffer.slice(0))
      const channel = decoded.getChannelData(0)
      source = normalize(downsample(channel, 320))
      peaks = resample(source, W || 160)
    } finally {
      await ac.close()
    }
  }

  async function load(src: string, options?: { play?: boolean }) {
    analyser = null
    livePeaks = []
    resize()
    const next = new URL(src, location.href).href
    if (audio.src !== next) audio.src = src
    const res = await fetch(src)
    if (!res.ok) throw new Error(`Could not load audio (${res.status})`)
    await decodePeaks(await res.arrayBuffer())
    paintTime()
    draw()
    if (options?.play) {
      audio.currentTime = 0
      await audio.play().catch(() => {})
    }
  }

  async function loadBlob(blob: Blob, options?: { play?: boolean }) {
    analyser = null
    livePeaks = []
    const url = URL.createObjectURL(blob)
    audio.src = url
    await decodePeaks(await blob.arrayBuffer())
    paintTime()
    draw()
    if (options?.play) {
      audio.currentTime = 0
      await audio.play().catch(() => {})
    }
  }

  function seek(clientX: number) {
    const rect = canvas.getBoundingClientRect()
    const dur = audio.duration
    if (!Number.isFinite(dur) || dur <= 0) return
    audio.currentTime = Math.min(dur, Math.max(0, ((clientX - rect.left) / rect.width) * dur))
    draw()
    paintTime()
  }

  readHue()
  resize()
  draw()

  const onResize = () => {
    resize()
    draw()
  }
  addEventListener('resize', onResize, { passive: true })
  const ro = new ResizeObserver(onResize)
  ro.observe(canvas)
  audio.addEventListener('play', () => {
    syncPlayBtn()
    startLoop()
  })
  audio.addEventListener('pause', () => {
    syncPlayBtn()
    draw()
    paintTime()
  })
  audio.addEventListener('ended', () => {
    syncPlayBtn()
    draw()
    paintTime()
  })
  audio.addEventListener('timeupdate', () => {
    if (audio.paused) {
      draw()
      paintTime()
    }
  })
  audio.addEventListener('loadedmetadata', paintTime)
  canvas.addEventListener('click', (e) => seek(e.clientX))
  playBtn?.addEventListener('click', () => {
    if (audio.paused) void audio.play()
    else audio.pause()
  })

  return {
    load,
    loadBlob,
    play: () => void audio.play(),
    pause: () => audio.pause(),
    setPlaying(play) {
      if (play) void audio.play()
      else audio.pause()
    },
    live(node) {
      analyser = node
      if (node) {
        analyserBuf = new Float32Array(node.fftSize) as Float32Array<ArrayBuffer>
        livePeaks = []
        startLoop()
      } else {
        analyserBuf = null
      }
    },
    dispose() {
      disposed = true
      cancelAnimationFrame(raf)
      removeEventListener('resize', onResize)
      ro.disconnect()
    },
    get currentTime() {
      return audio.currentTime
    },
    get duration() {
      return Number.isFinite(audio.duration) ? audio.duration : 0
    },
    audio,
  }
}

function normalize(peaks: Float32Array): Float32Array {
  let max = 0
  for (let i = 0; i < peaks.length; i++) {
    const v = peaks[i] ?? 0
    if (v > max) max = v
  }
  if (max <= 0) return peaks
  const out = new Float32Array(peaks.length)
  const gain = 1 / max
  for (let i = 0; i < peaks.length; i++) out[i] = (peaks[i] ?? 0) * gain
  return out
}

function downsample(channel: Float32Array, columns: number): Float32Array {
  const out = new Float32Array(columns)
  const step = channel.length / columns
  for (let i = 0; i < columns; i++) {
    const start = Math.floor(i * step)
    const end = Math.max(start + 1, Math.floor((i + 1) * step))
    let peak = 0
    for (let s = start; s < end; s++) {
      const v = Math.abs(channel[s] ?? 0)
      if (v > peak) peak = v
    }
    out[i] = peak
  }
  return out
}

function resample(src: Float32Array, columns: number): Float32Array {
  if (src.length === columns) return src
  const out = new Float32Array(columns)
  for (let i = 0; i < columns; i++) {
    out[i] = src[Math.floor((i / columns) * src.length)] ?? 0
  }
  return out
}

function fmt(sec: number): string {
  if (!Number.isFinite(sec) || sec < 0) return '0:00'
  const s = Math.floor(sec)
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`
}

function hexToRgb(hex: string): [number, number, number] {
  const h = hex.replace('#', '')
  const n = parseInt(h.length === 3 ? h.split('').map((c) => c + c).join('') : h, 16)
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255]
}
