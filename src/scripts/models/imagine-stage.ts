/**
 * Pixel studio that plays while Imagine is generating.
 * Low-res buffer, scaled up with image-rendering: pixelated.
 */

const GRID = 64
const SCALE = 2

type RGB = [number, number, number]

const PAL: Record<string, RGB> = {
  w: [16, 14, 12],
  W: [28, 24, 20],
  s: [8, 7, 6],
  f: [36, 30, 24],
  F: [58, 48, 36],
  n: [14, 26, 36],
  N: [32, 52, 66],
  m: [232, 228, 214],
  e: [92, 72, 48],
  E: [148, 116, 76],
  c: [236, 228, 210],
  C: [196, 184, 164],
  p: [196, 176, 150],
  P: [122, 102, 82],
  t: [74, 50, 32],
  T: [128, 88, 52],
  l: [24, 78, 52],
  L: [52, 138, 78],
  H: [176, 210, 102],
  a: [255, 179, 64],
  A: [255, 214, 140],
  k: [36, 34, 42],
  K: [72, 68, 80],
  i: [242, 238, 228],
  o: [212, 120, 52],
  O: [140, 72, 32],
  u: [64, 118, 140],
  U: [28, 64, 84],
  v: [118, 148, 168],
  b: [36, 32, 48],
  B: [255, 92, 128],
  g: [90, 86, 96],
  d: [48, 56, 64],
  y: [255, 196, 72],
}

const BAYER = [
  [0, 8, 2, 10],
  [12, 4, 14, 6],
  [3, 11, 1, 9],
  [15, 7, 13, 5],
].map((row) => row.map((v) => (v + 0.5) / 16))

/** 22×18 paintings. `.` is bare canvas. */
const PAINTINGS = [
  [
    '......................',
    '......................',
    '.......HHHHH..........',
    '.....HHLHLHHH.........',
    '....HLHHHtHHHHH.......',
    '.....HHHHtTHHH........',
    '.......HHtTH..........',
    '........HtT...........',
    '........tT............',
    '.......tT.............',
    '......tTT.............',
    '.....PPPPPPPP.........',
    '....PppppppppP........',
    '....PppppppppP........',
    '.....PPPPPPPP.........',
    '......P....P..........',
    '......................',
    '......................',
  ],
  [
    '......................',
    '......................',
    '......OO....OO........',
    '.....OooO..OooO.......',
    '.....OOOOOOOOOO.......',
    '....OOiiOOiiOOO.......',
    '....OOOOOOOOOOO.......',
    '.....OOOaaOOO.........',
    '....OOOOOOOOOO........',
    '....OOOOOOOOOO........',
    '...OOOOOOOOOOOO.......',
    '....OOOOOOOOOO........',
    '.....OOO..OOO.........',
    '.....OO....OO.........',
    '......................',
    '......................',
    '......................',
    '......................',
  ],
  [
    'vvvvvvvvvvvvvvvvvvvvvv',
    'vvvvvvvvvvmmvvvvvvvvvv',
    'vvvvvvvvmmmmmmvvvvvvvv',
    'vvvvvvvvvmmmmvvvvvvvvv',
    'vvvvvddddddddddddvvvvv',
    'vvvdddddddddddddddvvvv',
    'dddddddddddddddddddddd',
    'uuuuuuuuuuuuuuuuuuuuuu',
    'uUuUuUuUuUuUuUuUuUuUuU',
    'uuuuummuuuuuuuuuuuuuuu',
    'uuuummmmuuuuuuuuuuuuuu',
    'uUUuUUuUUuUUuUUuUUuUUu',
    'uuuuuuuuuuuuuuuuuuuuuu',
    'uUuUuUuUuUuUuUuUuUuUuU',
    'uuuuuuuuuuuuuuuuuuuuuu',
    'HHHHHHHHHHHHHHHHHHHHHH',
    'HHLHLHHHLHHHLHLHHHLHLH',
    'llllllllllllllllllllll',
  ],
  [
    'bbbbbbbbbbbbbbbbbbbbbb',
    'bbByybbbByybbbByybbbbb',
    'bbbbbbbbbbbbbbbbbbbbbb',
    'bbByBbbbByBbbbByBbbbbb',
    'bbbbbbbbbbbbbbbbbbbbbb',
    'bByybbbbByybbbByybbbbb',
    'bbbbbbbbbbbbbbbbbbbbbb',
    'bbbbbbbbbbbbbbbbbbbbbb',
    'aaaaaaaaaaaaaaaaaaaaaa',
    'BBByyBBByyBBByyBBByyBB',
    'bbbbbbbbbbbbbbbbbbbbbb',
    'bbByybbbByybbbByybbbbb',
    'bbbbbbbbbbbbbbbbbbbbbb',
    'aaaaaaaaaaaaaaaaaaaaaa',
    '......................',
    '......................',
    '......................',
    '......................',
  ],
]

const ROBOT = [
  '.....KKKKKK.....',
  '....KkkkkkkK....',
  '...Kkiiiiiiik...',
  '...KkiKiiKiik...',
  '...Kkiiiiiiik...',
  '....KkkkkkkK....',
  '.....KKKKKK.....',
  '.......KK.......',
  '.....aaaaaa.....',
  '....akkiiiika...',
  '...aakkiiiikaaa.',
  '.....KK....aa...',
  '.....KK....E....',
  '....KKK...KK....',
  '...iiii...iii...',
]

const ROBOT_BLINK = [
  '.....KKKKKK.....',
  '....KkkkkkkK....',
  '...Kkiiiiiiik...',
  '...Kkiiiiiiik...',
  '...Kkiiiiiiik...',
  '....KkkkkkkK....',
  '.....KKKKKK.....',
  '.......KK.......',
  '.....aaaaaa.....',
  '....akkiiiika...',
  '...aakkiiiikaaa.',
  '.....KK....aa...',
  '.....KK....E....',
  '....KKK...KK....',
  '...iiii...iii...',
]

const PX = GRID * GRID

export type ImagineStage = {
  start(): void
  stop(): void
  setProgress(value: number): void
}

export function createImagineStage(canvas: HTMLCanvasElement): ImagineStage {
  const ctx = canvas.getContext('2d', { alpha: false })!
  canvas.width = GRID * SCALE
  canvas.height = GRID * SCALE
  const img = ctx.createImageData(canvas.width, canvas.height)
  const out = img.data
  const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches

  const base = new Uint8Array(PX)
  paintRoom(base)

  let progress = 0
  let raf = 0
  let running = false
  let last = 0

  function draw(now: number) {
    const t = reduce ? 1.2 : now / 1000
    const cells = base.slice()
    const blink = !reduce && Math.floor(t * 0.35) % 9 === 0
    const bob = !reduce && Math.floor(t * 2) % 2 === 0 ? 1 : 0
    const scanY = 16 + Math.floor((Math.sin(t * 1.6) * 0.5 + 0.5) * 16)
    paintCanvas(cells, t, scanY)
    stamp(cells, blink ? ROBOT_BLINK : ROBOT, 8, 35 + bob)
    paintBrush(cells, 22, 44 + bob, 34, scanY)
    paintMotes(cells, t)
    blit(cells)
    ctx.putImageData(img, 0, 0)
  }

  function paintCanvas(cells: Uint8Array, t: number, scanY: number) {
    const ox = 34
    const oy = 16
    const pw = 22
    const ph = 18
    const hold = 3.6
    const slot = Math.floor(t / hold)
    const local = (t % hold) / hold
    const index = progress > 0 ? Math.min(PAINTINGS.length - 1, Math.floor(progress * PAINTINGS.length)) : slot % PAINTINGS.length
    const from = PAINTINGS[index]!
    const to = PAINTINGS[(index + 1) % PAINTINGS.length]!
    const dissolving = progress === 0 && local > 0.8
    const wipe = dissolving ? (local - 0.8) / 0.2 : 0

    for (let y = 0; y < ph; y++) {
      for (let x = 0; x < pw; x++) {
        const chA = from[y]?.[x] ?? '.'
        const chB = to[y]?.[x] ?? '.'
        const flip = dissolving && hash(x, y, slot) < wipe + (BAYER[y & 3]![x & 3]! - 0.5) * 0.25
        let ch = flip ? chB : chA
        const onScan = oy + y === scanY && hash(x, y, Math.floor(t * 14)) > 0.62
        if (onScan) ch = 'a'
        const rgb = ch === '.' ? PAL.c! : PAL[ch] ?? PAL.c!
        cells[(oy + y) * GRID + (ox + x)] = rgbKey(rgb)
      }
    }
  }

  function paintBrush(cells: Uint8Array, x0: number, y0: number, x1: number, y1: number) {
    let x = x0
    let y = y0
    const dx = Math.abs(x1 - x0)
    const dy = Math.abs(y1 - y0)
    const sx = x0 < x1 ? 1 : -1
    const sy = y0 < y1 ? 1 : -1
    let err = dx - dy
    for (let n = 0; n < 24; n++) {
      plot(cells, x, y, n > 18 ? 'A' : 'E')
      if (x === x1 && y === y1) break
      const e2 = 2 * err
      if (e2 > -dy) {
        err -= dy
        x += sx
      }
      if (e2 < dx) {
        err += dx
        y += sy
      }
    }
    plot(cells, x1, y1, 'y')
    plot(cells, x1 + 1, y1, 'A')
  }

  function paintMotes(cells: Uint8Array, t: number) {
    if (reduce) return
    for (let i = 0; i < 5; i++) {
      const phase = (t * 0.15 + i * 0.17) % 1
      const x = 40 + ((i * 7 + Math.floor(phase * 18)) % 16)
      const y = 8 + Math.floor((1 - phase) * 28)
      if (hash(i, y, 3) > 0.4) plot(cells, x, y, i % 2 ? 'A' : 'y')
    }
    const flicker = Math.floor(t * 3) % 5 !== 0
    if (flicker) {
      for (const [sx, sy] of STARS) {
        if (hash(sx, Math.floor(t * 2), sy) > 0.35) plot(cells, sx, sy, 'A')
      }
    }
  }

  function blit(cells: Uint8Array) {
    for (let y = 0; y < GRID; y++) {
      for (let x = 0; x < GRID; x++) {
        const rgb = RGBS[cells[y * GRID + x]!] ?? PAL.w!
        for (let dy = 0; dy < SCALE; dy++) {
          for (let dx = 0; dx < SCALE; dx++) {
            const i = ((y * SCALE + dy) * GRID * SCALE + (x * SCALE + dx)) * 4
            out[i] = rgb[0]
            out[i + 1] = rgb[1]
            out[i + 2] = rgb[2]
            out[i + 3] = 255
          }
        }
      }
    }
  }

  function frame(now: number) {
    if (now - last > 70 || last === 0) {
      last = now
      draw(now)
    }
    if (running && !reduce) raf = requestAnimationFrame(frame)
  }

  return {
    start() {
      if (running) return
      running = true
      progress = 0
      last = 0
      raf = requestAnimationFrame(frame)
    },
    stop() {
      running = false
      cancelAnimationFrame(raf)
    },
    setProgress(value: number) {
      progress = Math.max(0, Math.min(1, value))
    },
  }
}

const STARS: Array<[number, number]> = [
  [8, 8],
  [14, 11],
  [6, 14],
  [18, 9],
  [11, 16],
]

const RGBS: RGB[] = []
const KEY = new Map<string, number>()

function rgbKey(rgb: RGB): number {
  const id = `${rgb[0]},${rgb[1]},${rgb[2]}`
  let k = KEY.get(id)
  if (k == null) {
    k = RGBS.length
    RGBS.push(rgb)
    KEY.set(id, k)
  }
  return k
}

function hash(x: number, y: number, n: number): number {
  const h = Math.imul(x + n * 13, 374761393) + Math.imul(y + 17, 668265263)
  return ((h ^ (h >>> 13)) >>> 0) / 4294967295
}

function plot(cells: Uint8Array, x: number, y: number, ch: string) {
  if (x < 0 || y < 0 || x >= GRID || y >= GRID) return
  const rgb = PAL[ch]
  if (!rgb) return
  cells[y * GRID + x] = rgbKey(rgb)
}

function fill(cells: Uint8Array, x: number, y: number, w: number, h: number, ch: string) {
  for (let yy = y; yy < y + h; yy++) {
    for (let xx = x; xx < x + w; xx++) plot(cells, xx, yy, ch)
  }
}

function stamp(cells: Uint8Array, rows: string[], ox: number, oy: number) {
  rows.forEach((row, y) => {
    ;[...row].forEach((ch, x) => {
      if (ch !== '.' && ch !== ' ') plot(cells, ox + x, oy + y, ch)
    })
  })
}

function paintRoom(cells: Uint8Array) {
  fill(cells, 0, 0, GRID, GRID, 'w')
  fill(cells, 0, 50, GRID, 14, 'f')
  fill(cells, 0, 50, GRID, 1, 'F')
  fill(cells, 0, 0, GRID, 3, 's')

  fill(cells, 4, 6, 18, 14, 'F')
  fill(cells, 5, 7, 16, 12, 'n')
  fill(cells, 9, 9, 6, 5, 'N')
  fill(cells, 10, 10, 4, 3, 'm')

  fill(cells, 32, 14, 26, 22, 'e')
  fill(cells, 33, 15, 24, 20, 'E')
  fill(cells, 34, 16, 22, 18, 'c')
  fill(cells, 43, 36, 2, 14, 'e')
  fill(cells, 44, 36, 2, 14, 'E')
  fill(cells, 36, 49, 16, 2, 'E')
  fill(cells, 34, 51, 3, 2, 'e')
  fill(cells, 51, 51, 3, 2, 'e')

  fill(cells, 42, 1, 2, 3, 'F')
  fill(cells, 38, 4, 10, 2, 'a')
  fill(cells, 39, 6, 8, 2, 'A')
  fill(cells, 40, 8, 6, 1, 'y')

  for (let y = 8; y < 22; y++) {
    for (let x = 36; x < 54; x++) {
      const dx = (x - 43) / 12
      const dy = (y - 8) / 14
      const v = 1 - (dx * dx + dy * dy)
      if (v > BAYER[y & 3]![x & 3]! * 1.2) plot(cells, x, y, 'A')
    }
  }
}
