import type { Device, ProgressCallback, ResolvedDevice } from '../core/index.ts'
import { resolveDevice, toProgressInfo } from '../core/index.ts'
import { phonemizeLang } from './phonemes.ts'
import { splitUtterances } from './split.ts'
import { toFloat32 } from './wav.ts'
import { kokoroPhonemeLang } from './voices.ts'
import type { SpeakChunk } from './types.ts'

/**
 * Kokoro 82M (StyleTTS 2) on Transformers.js directly. No `kokoro-js`, so the app ships one
 * copy of Transformers.js. Voice style vectors come from the same Hugging Face repo.
 */
const DEFAULT_MODEL = 'onnx-community/Kokoro-82M-v1.0-ONNX'
const SAMPLE_RATE = 24_000
const STYLE_DIM = 256
const MAX_STYLE_INDEX = 509
const VOICE_CACHE = 'runonweb-kokoro-voices'

type Tensorish = { data: Float32Array | number[]; dims: number[] }

type KokoroModel = {
  (inputs: Record<string, unknown>): Promise<{ waveform: Tensorish }>
  dispose?: () => Promise<void>
}

type KokoroTokenizer = (text: string, options?: { truncation?: boolean }) => { input_ids: { dims: number[] } }

export type LoadedKokoro = {
  model: KokoroModel
  tokenizer: KokoroTokenizer
  modelId: string
  device: ResolvedDevice
}

export async function loadKokoro(options: {
  model?: string
  device?: Device
  onProgress?: ProgressCallback
}): Promise<LoadedKokoro> {
  const device = await resolveDevice(options.device ?? 'auto')
  options.onProgress?.({ status: 'loading', progress: 0 })

  const { StyleTextToSpeech2Model, AutoTokenizer, env } = await import('@huggingface/transformers')
  env.allowLocalModels = false
  const modelId = options.model ?? DEFAULT_MODEL
  const progress_callback = (data: Record<string, unknown>) => {
    options.onProgress?.(toProgressInfo(data))
  }

  const [model, tokenizer] = await Promise.all([
    StyleTextToSpeech2Model.from_pretrained(modelId, {
      device,
      // fp16 / q4f16 produce NaN on the Transformers.js 4 WebGPU runtime; fp32 is clean and fast.
      dtype: device === 'webgpu' ? 'fp32' : 'q8',
      progress_callback,
    }),
    AutoTokenizer.from_pretrained(modelId, { progress_callback }),
  ])

  options.onProgress?.({ status: 'ready', progress: 100 })
  return {
    model: model as unknown as KokoroModel,
    tokenizer: tokenizer as unknown as KokoroTokenizer,
    modelId,
    device,
  }
}

export async function* streamKokoro(
  kokoro: LoadedKokoro,
  text: string,
  voice: string,
  speed: number
): AsyncGenerator<SpeakChunk> {
  const lang = kokoroPhonemeLang(voice) ?? (voice.startsWith('b') ? 'en-gb' : 'en-us')
  for (const piece of splitUtterances(text)) {
    const phonemes = await phonemizeLang(piece, lang)
    if (!phonemes) throw new Error(`Could not phonemize text for ${lang}`)
    const audio = await synthesize(kokoro, phonemes, voice, speed)
    yield { audio, samplingRate: SAMPLE_RATE, text: piece }
  }
}

async function synthesize(kokoro: LoadedKokoro, phonemes: string, voice: string, speed: number): Promise<Float32Array> {
  const { Tensor } = await import('@huggingface/transformers')
  const { input_ids } = kokoro.tokenizer(phonemes, { truncation: true })
  const numTokens = Math.min(Math.max((input_ids.dims.at(-1) ?? 2) - 2, 0), MAX_STYLE_INDEX)
  const styles = await loadVoice(kokoro.modelId, voice)
  const style = styles.slice(numTokens * STYLE_DIM, (numTokens + 1) * STYLE_DIM)

  const { waveform } = await kokoro.model({
    input_ids,
    style: new Tensor('float32', style, [1, STYLE_DIM]),
    speed: new Tensor('float32', [speed], [1]),
  })
  return toFloat32(waveform.data)
}

const voiceCache = new Map<string, Promise<Float32Array>>()

/** Fetch `voices/<id>.bin` (510×256 float32 styles indexed by token count) and keep it in the Cache API. */
function loadVoice(modelId: string, voice: string): Promise<Float32Array> {
  const key = `${modelId}/${voice}`
  let pending = voiceCache.get(key)
  if (!pending) {
    pending = fetchVoice(modelId, voice).catch((err) => {
      voiceCache.delete(key)
      throw err
    })
    voiceCache.set(key, pending)
  }
  return pending
}

async function fetchVoice(modelId: string, voice: string): Promise<Float32Array> {
  const url = `https://huggingface.co/${modelId}/resolve/main/voices/${voice}.bin`
  let cache: Cache | null = null
  try {
    cache = await caches.open(VOICE_CACHE)
    const hit = await cache.match(url)
    if (hit) return new Float32Array(await hit.arrayBuffer())
  } catch {
    cache = null
  }
  const res = await fetch(url)
  if (!res.ok) throw new Error(`Voice "${voice}" not found (${res.status})`)
  const buffer = await res.arrayBuffer()
  try {
    await cache?.put(url, new Response(buffer.slice(0), { headers: { 'content-type': 'application/octet-stream' } }))
  } catch {
    // Cache is best-effort.
  }
  return new Float32Array(buffer)
}

export function disposeKokoro(kokoro: LoadedKokoro | null) {
  void kokoro?.model.dispose?.()
}

export { DEFAULT_MODEL as KOKORO_MODEL, SAMPLE_RATE as KOKORO_SAMPLE_RATE }
