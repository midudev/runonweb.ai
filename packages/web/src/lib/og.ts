/**
 * Open Graph image renderer. Runs at build time only (static endpoint), so it can
 * read font files from disk and use satori + resvg without shipping them.
 * Satori only parses TTF/OTF/WOFF (no WOFF2), so the .woff copies live in
 * src/assets/fonts (OFL, see LICENSE there).
 */
import { readFile } from 'node:fs/promises'
import path from 'node:path'
import satori from 'satori'
import { Resvg } from '@resvg/resvg-js'
import { SITE } from '../data/site'
import { models, shortSize, type Model } from '../data/models'

const FONT_DIR = path.join(process.cwd(), 'src/assets/fonts')
const fontFile = (file: string) => readFile(path.join(FONT_DIR, file))

let fontsPromise: Promise<Array<{ name: string; data: Buffer; weight: 400 | 600; style: 'normal' }>> | undefined
function fonts() {
  fontsPromise ??= Promise.all([
    fontFile('geist-latin-400-normal.woff').then((data) => ({ name: 'Geist', data, weight: 400 as const, style: 'normal' as const })),
    fontFile('geist-latin-600-normal.woff').then((data) => ({ name: 'Geist', data, weight: 600 as const, style: 'normal' as const })),
    fontFile('geist-pixel-latin-400-normal.woff').then((data) => ({ name: 'Geist Pixel', data, weight: 400 as const, style: 'normal' as const })),
  ])
  return fontsPromise
}

type Node = { type: string; props: Record<string, unknown> }
const h = (type: string, style: Record<string, unknown>, children?: unknown): Node => ({
  type,
  props: { style, children },
})

const BG = '#0a0a0b'
const BG2 = '#111113'
const FG = '#f2f2ee'
const FG2 = '#a3a3a8'
const FG3 = '#66666d'
const LINE = 'rgba(255,255,255,0.12)'
const ACCENT = '#ffb340'

const readout = (text: string, color = FG2, size = 22) =>
  h('div', { display: 'flex', fontFamily: 'Geist Pixel', fontSize: size, letterSpacing: 1, textTransform: 'uppercase', color }, text)

const led = (hue: string, size = 14) =>
  h('div', { width: size, height: size, borderRadius: 3, background: hue, boxShadow: `0 0 16px ${hue}` })

/** Brand mark: 12 dots around an accent core, same geometry as Logo.astro. */
function logo(size = 44) {
  const u = size / 24
  const dots = [
    [10, 2], [18, 2], [14, 6], [18, 10], [14, 14], [18, 18], [10, 18], [2, 18], [6, 14], [2, 10], [6, 6], [2, 2],
  ]
  return h('div', { display: 'flex', position: 'relative', width: size, height: size }, [
    ...dots.map(([x, y]) => h('div', { position: 'absolute', left: x * u, top: y * u, width: 4 * u, height: 4 * u, background: FG })),
    h('div', { position: 'absolute', left: 10 * u, top: 10 * u, width: 4 * u, height: 4 * u, background: ACCENT }),
  ])
}

/** 12×8 pixel glyph from Model.icon. */
function pixelIcon(rows: string[], hue: string, cell = 22) {
  const alpha: Record<string, number> = { '#': 1, '+': 0.6, '-': 0.3, '~': 1 }
  const cells: Node[] = []
  rows.forEach((row, y) => {
    ;[...row].forEach((ch, x) => {
      const a = alpha[ch]
      if (!a) return
      cells.push(h('div', { position: 'absolute', left: x * cell, top: y * cell, width: cell, height: cell, background: ch === '~' ? FG3 : hue, opacity: a }))
    })
  })
  return h('div', { display: 'flex', position: 'relative', width: 12 * cell, height: rows.length * cell }, cells)
}

function frame(children: Node[]) {
  return h(
    'div',
    {
      display: 'flex',
      flexDirection: 'column',
      width: 1200,
      height: 630,
      padding: '56px 64px',
      background: BG,
      backgroundImage: `radial-gradient(900px 500px at 80% -10%, rgba(255,179,64,0.10), transparent 60%)`,
      color: FG,
      fontFamily: 'Geist',
    },
    children,
  )
}

function header(right: string) {
  return h('div', { display: 'flex', alignItems: 'center', justifyContent: 'space-between', width: '100%' }, [
    h('div', { display: 'flex', alignItems: 'center', gap: 14 }, [
      logo(40),
      h('div', { display: 'flex', fontSize: 26, fontWeight: 600, letterSpacing: -0.5 }, SITE.name),
    ]),
    readout(right, FG3, 20),
  ])
}

function footer(items: string[]) {
  return h('div', { display: 'flex', alignItems: 'center', gap: 28, marginTop: 'auto', paddingTop: 26, borderTop: `2px solid ${LINE}` }, [
    led(ACCENT, 12),
    ...items.map((t) => readout(t, FG2, 20)),
  ])
}

export type OgSpec =
  | { kind: 'site' }
  | { kind: 'page'; eyebrow: string; title: string; description: string }
  | { kind: 'model'; model: Model; index: number }

function tree(spec: OgSpec): Node {
  if (spec.kind === 'model') {
    const { model, index } = spec
    const num = String(index + 1).padStart(2, '0')
    const backend = model.webgpu === 'none' ? 'WASM' : model.webgpu === 'required' ? 'WebGPU' : 'WebGPU + WASM'
    return frame([
      header(`runonweb.ai/models/${model.slug}`),
      h('div', { display: 'flex', flex: 1, alignItems: 'flex-end', justifyContent: 'space-between', gap: 40, marginTop: 40, paddingBottom: 36 }, [
        h('div', { display: 'flex', flexDirection: 'column', maxWidth: 760 }, [
          h('div', { display: 'flex', alignItems: 'center', gap: 14 }, [led(model.hue), readout(`${num} / ${model.task}`, FG2, 24)]),
          h('div', { display: 'flex', fontSize: 148, fontWeight: 600, letterSpacing: -6, lineHeight: 0.95, marginTop: 18 }, model.name),
          h('div', { display: 'flex', fontSize: 34, color: FG2, lineHeight: 1.25, marginTop: 18 }, model.tagline),
        ]),
        h('div', { display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: 22 }, [
          pixelIcon(model.icon, model.hue, 20),
          h('div', { display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: 8 }, [
            readout(model.weights.baseModel, FG, 20),
            readout(`${model.weights.author} · ${model.weights.license}`, FG3, 18),
          ]),
        ]),
      ]),
      footer([`Download ${shortSize(model)}`, backend, 'In the browser', '$0']),
    ])
  }

  if (spec.kind === 'page') {
    return frame([
      header('runonweb.ai'),
      h('div', { display: 'flex', flexDirection: 'column', flex: 1, justifyContent: 'flex-end', marginTop: 40, paddingBottom: 36 }, [
        h('div', { display: 'flex', alignItems: 'center', gap: 14 }, [led(ACCENT), readout(spec.eyebrow, FG2, 24)]),
        h('div', { display: 'flex', fontSize: 104, fontWeight: 600, letterSpacing: -4, lineHeight: 0.95, marginTop: 18 }, spec.title),
        h('div', { display: 'flex', fontSize: 30, color: FG2, lineHeight: 1.3, marginTop: 22, maxWidth: 900 }, spec.description),
      ]),
      footer(['Free', 'MIT', 'WebGPU + WASM', 'Nothing leaves the device']),
    ])
  }

  return frame([
    header('runonweb.ai'),
    h('div', { display: 'flex', flexDirection: 'column', flex: 1, justifyContent: 'flex-end', marginTop: 40, paddingBottom: 36 }, [
      h('div', { display: 'flex', alignItems: 'center', gap: 14 }, [led(ACCENT), readout(`${String(models.length).padStart(2, '0')} modules · WebGPU + WASM · $0`, FG2, 24)]),
      h('div', { display: 'flex', fontSize: 96, fontWeight: 600, letterSpacing: -4, lineHeight: 0.95, marginTop: 18, maxWidth: 1000 }, 'AI that runs where your users already are.'),
      h('div', { display: 'flex', fontSize: 30, color: FG2, lineHeight: 1.3, marginTop: 22, maxWidth: 900 }, 'Free, open-source AI modules for the browser. Speech, vision and text. One import per task, zero bytes sent to a server.'),
    ]),
    footer(['Free', 'MIT', 'WebGPU + WASM', 'Nothing leaves the device']),
  ])
}

export async function renderOg(spec: OgSpec): Promise<Uint8Array> {
  const svg = await satori(tree(spec) as never, { width: 1200, height: 630, fonts: await fonts() })
  const png = new Resvg(svg, { fitTo: { mode: 'width', value: 1200 }, background: BG2 }).render().asPng()
  return png
}
