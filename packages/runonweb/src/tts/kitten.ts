import type { ProgressCallback } from '../core/index.ts'
import { splitUtterances } from './split.ts'
import { toFloat32 } from './wav.ts'
import type { SpeakChunk } from './types.ts'

const HF = 'https://huggingface.co/KittenML/kitten-tts-nano-0.8-int8/resolve/main'
const CACHE = 'runonweb-kitten-tts'
const ORT_WASM = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/'
const SAMPLE_RATE = 24_000
const AUDIO_TRIM = 5_000

const VOICE_ALIASES: Record<string, string> = {
  bella: 'expr-voice-2-f',
  jasper: 'expr-voice-2-m',
  luna: 'expr-voice-3-f',
  bruno: 'expr-voice-3-m',
  rosie: 'expr-voice-4-f',
  hugo: 'expr-voice-4-m',
  kiki: 'expr-voice-5-f',
  leo: 'expr-voice-5-m',
}

const SPEED_PRIORS: Record<string, number> = {
  'expr-voice-2-f': 0.8,
  'expr-voice-2-m': 0.8,
  'expr-voice-3-m': 0.8,
  'expr-voice-3-f': 0.8,
  'expr-voice-4-m': 0.9,
  'expr-voice-4-f': 0.8,
  'expr-voice-5-m': 0.8,
  'expr-voice-5-f': 0.8,
}

const PAD = '$'
const PUNCTUATION = ';:,.!?¡¿—…"«»"" '
const LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
const LETTERS_IPA =
  "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘''ᵻ"
const SYMBOLS = [PAD, ...PUNCTUATION, ...LETTERS, ...LETTERS_IPA]
const SYMBOL_TO_ID = new Map(SYMBOLS.map((s, i) => [s, i]))

type OrtApi = {
  env: { wasm: { wasmPaths: string; numThreads: number; simd?: boolean } }
  Tensor: new (type: string, data: Float32Array | BigInt64Array, dims: number[]) => unknown
  InferenceSession: {
    create: (
      model: ArrayBuffer,
      options?: { executionProviders?: string[] }
    ) => Promise<{ run: (feeds: Record<string, unknown>) => Promise<Record<string, { data: Float32Array }>> }>
  }
}

type VoiceEntry = { data: Float32Array; shape: number[] }

export type LoadedKitten = {
  session: Awaited<ReturnType<OrtApi['InferenceSession']['create']>>
  voices: Record<string, VoiceEntry>
  ort: OrtApi
}

export async function loadKitten(onProgress?: ProgressCallback): Promise<LoadedKitten> {
  onProgress?.({ status: 'loading', progress: 0 })
  const [model, voicesBuf] = await Promise.all([
    fetchCached(`${HF}/kitten_tts_nano_v0_8.onnx`, 'kitten.onnx', onProgress),
    fetchCached(`${HF}/voices.npz`, 'voices.npz', onProgress),
  ])
  const [ort, voices] = await Promise.all([loadOrt(), loadNpz(voicesBuf)])
  const session = await ort.InferenceSession.create(model, { executionProviders: ['wasm'] })
  onProgress?.({ status: 'ready', progress: 100 })
  return { session, voices, ort }
}

export async function* streamKitten(
  loaded: LoadedKitten,
  text: string,
  voice: string,
  speed: number
): AsyncGenerator<SpeakChunk> {
  for (const piece of splitUtterances(text)) {
    const audio = await inferKitten(loaded, piece, voice, speed)
    yield { audio, samplingRate: SAMPLE_RATE, text: piece }
  }
}

export function disposeKitten(loaded: LoadedKitten | null) {
  const session = loaded?.session as { release?: () => Promise<void> } | undefined
  void session?.release?.()
}

async function inferKitten(loaded: LoadedKitten, text: string, voiceId: string, speed: number): Promise<Float32Array> {
  const key = VOICE_ALIASES[voiceId] ?? voiceId
  const voice = loaded.voices[key]
  if (!voice) throw new Error(`Unknown Kitten voice: ${voiceId}`)

  const phonemes = await phonemizeEnglish(ensurePunctuation(text))
  const tokenIds = cleanPhonemes(phonemes)
  const [numStyles, styleDim] = voice.shape
  const refId = Math.min(tokenIds.length, (numStyles ?? 1) - 1)
  const dim = styleDim ?? 256
  const style = voice.data.slice(refId * dim, (refId + 1) * dim)
  const scaled = speed * (SPEED_PRIORS[key] ?? 1)

  const feeds = {
    input_ids: new loaded.ort.Tensor('int64', BigInt64Array.from(tokenIds.map(BigInt)), [1, tokenIds.length]),
    style: new loaded.ort.Tensor('float32', new Float32Array(style), [1, dim]),
    speed: new loaded.ort.Tensor('float32', new Float32Array([scaled]), [1]),
  }
  const results = await loaded.session.run(feeds)
  const output = results[Object.keys(results)[0] ?? '']
  const data = toFloat32(output?.data ?? [])
  return data.slice(0, Math.max(0, data.length - AUDIO_TRIM))
}

function cleanPhonemes(phonemes: string): number[] {
  const ids: number[] = []
  for (const ch of phonemes) {
    const id = SYMBOL_TO_ID.get(ch)
    if (id !== undefined) ids.push(id)
  }
  return [0, ...ids, 10, 0]
}

function ensurePunctuation(text: string): string {
  const t = text.trim()
  if (!t) return t
  return /[.!?,;:]$/.test(t) ? t : `${t},`
}

async function phonemizeEnglish(text: string): Promise<string> {
  const { phonemize } = await import('phonemizer')
  const chunks = text.split(/([;:,.!?¡¿—…"«»"()\n]+)/)
  let out = ''
  for (const chunk of chunks) {
    if (!chunk) continue
    if (/^[;:,.!?¡¿—…"«»"()\s]+$/.test(chunk)) {
      out += chunk
      continue
    }
    const ipa = await phonemize(chunk, 'en-us')
    out += ipa.join(' ').replace(/_/g, '')
  }
  return out.trim()
}

async function loadOrt(): Promise<OrtApi> {
  const ort = (await import('onnxruntime-web')) as unknown as OrtApi
  ort.env.wasm.wasmPaths = ORT_WASM
  ort.env.wasm.numThreads = 1
  return ort
}

async function fetchCached(url: string, file: string, onProgress?: ProgressCallback): Promise<ArrayBuffer> {
  const cache = await caches.open(CACHE).catch(() => null)
  const hit = await cache?.match(url)
  if (hit) {
    onProgress?.({ status: 'progress', progress: 100, file })
    return hit.arrayBuffer()
  }

  const res = await fetch(url)
  if (!res.ok) throw new Error(`Could not load ${file} (${res.status})`)

  const total = Number(res.headers.get('content-length') ?? 0)
  if (!res.body || !total) {
    const buf = await res.arrayBuffer()
    await cache?.put(url, new Response(buf.slice(0)))
    return buf
  }

  const reader = res.body.getReader()
  const chunks: Uint8Array[] = []
  let received = 0
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    chunks.push(value)
    received += value.byteLength
    onProgress?.({ status: 'progress', progress: (received / total) * 100, file })
  }

  const out = new Uint8Array(received)
  let offset = 0
  for (const chunk of chunks) {
    out.set(chunk, offset)
    offset += chunk.byteLength
  }
  await cache?.put(url, new Response(out.buffer.slice(0)))
  return out.buffer
}

async function loadNpz(buffer: ArrayBuffer): Promise<Record<string, VoiceEntry>> {
  const view = new DataView(buffer)
  const bytes = new Uint8Array(buffer)
  const voices: Record<string, VoiceEntry> = {}

  const eocd = findEocd(view, bytes.length)
  const count = view.getUint16(eocd + 10, true)
  let central = view.getUint32(eocd + 16, true)

  for (let i = 0; i < count; i++) {
    if (view.getUint32(central, true) !== 0x02014b50) break
    const method = view.getUint16(central + 10, true)
    const compSize = view.getUint32(central + 20, true)
    const rawSize = view.getUint32(central + 24, true)
    const nameLen = view.getUint16(central + 28, true)
    const extraLen = view.getUint16(central + 30, true)
    const commentLen = view.getUint16(central + 32, true)
    const localOff = view.getUint32(central + 42, true)
    const name = new TextDecoder().decode(bytes.subarray(central + 46, central + 46 + nameLen))
    central += 46 + nameLen + extraLen + commentLen
    if (!name.endsWith('.npy')) continue

    const localNameLen = view.getUint16(localOff + 26, true)
    const localExtraLen = view.getUint16(localOff + 28, true)
    const dataStart = localOff + 30 + localNameLen + localExtraLen
    const compressed = bytes.subarray(dataStart, dataStart + compSize)
    const raw =
      method === 0 ? copyBytes(compressed) : method === 8 ? await inflateRaw(compressed, rawSize) : null
    if (!raw) throw new Error(`Unsupported zip method ${method} in voices.npz`)
    const key = name.replace(/\.npy$/, '').split('/').pop() ?? name
    voices[key] = parseNpy(copyBytes(raw).buffer)
  }

  if (!Object.keys(voices).length) throw new Error('voices.npz contained no arrays')
  return voices
}

function findEocd(view: DataView, length: number): number {
  for (let i = length - 22; i >= Math.max(0, length - 22 - 0xffff); i--) {
    if (view.getUint32(i, true) === 0x06054b50) return i
  }
  throw new Error('voices.npz is not a valid zip')
}

function copyBytes(src: Uint8Array): Uint8Array<ArrayBuffer> {
  const out = new Uint8Array(new ArrayBuffer(src.byteLength))
  out.set(src)
  return out
}

async function inflateRaw(data: Uint8Array, outLen: number): Promise<Uint8Array> {
  const stream = new DecompressionStream('deflate-raw')
  const writer = stream.writable.getWriter()
  await writer.write(Uint8Array.from(data))
  await writer.close()
  const reader = stream.readable.getReader()
  const chunks: Uint8Array[] = []
  let received = 0
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    chunks.push(value)
    received += value.byteLength
  }
  const out = new Uint8Array(outLen || received)
  let offset = 0
  for (const chunk of chunks) {
    out.set(chunk, offset)
    offset += chunk.byteLength
  }
  return out
}

function parseNpy(buf: ArrayBuffer): VoiceEntry {
  const bytes = new Uint8Array(buf)
  const major = bytes[6] ?? 1
  const headerLen =
    major >= 2 ? new DataView(buf, 8, 4).getUint32(0, true) : new DataView(buf, 8, 2).getUint16(0, true)
  const headerOffset = major >= 2 ? 12 : 10
  const header = new TextDecoder().decode(bytes.subarray(headerOffset, headerOffset + headerLen))
  const shapeStr = header.match(/'shape'\s*:\s*\(([^)]*)\)/)?.[1]?.trim() ?? ''
  const shape = shapeStr === '' ? [1] : shapeStr.split(',').map((s) => Number(s.trim())).filter((n) => !Number.isNaN(n))
  const start = headerOffset + headerLen
  const avail = buf.byteLength - start
  const usable = avail - (avail % 4)
  const data = new Float32Array(buf.slice(start, start + usable))
  return { data, shape }
}

export { SAMPLE_RATE as KITTEN_SAMPLE_RATE }
