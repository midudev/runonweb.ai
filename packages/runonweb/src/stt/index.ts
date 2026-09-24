import type { Device, ProgressCallback, ProgressInfo } from '../core/index.ts'
import { loadPipeline } from '../core/pipeline.ts'

const DEFAULT_MODEL = 'onnx-community/whisper-tiny.en'
const SAMPLE_RATE = 16_000

export type STTOptions = {
  /** Hugging Face model id. Defaults to a small English Whisper. */
  model?: string
  /** Inference device. Defaults to `auto` (WebGPU when available, else WASM). */
  device?: Device
  /** Language code for multilingual models (e.g. `"es"`, `"en"`). */
  language?: string
  /** Called while model files download / load. */
  onProgress?: ProgressCallback
}

export type STTChunk = {
  text: string
  start: number
  end: number
}

export type STTResult = {
  text: string
  chunks: STTChunk[]
}

export type TranscribeOptions = {
  /** Called with the transcript so far as Whisper emits words. */
  onPartial?: (text: string) => void
  /** Per-call language for multilingual models (e.g. `"es"`, `"ja"`). Falls back to the constructor value. */
  language?: string
  /**
   * Whisper task. Default `transcribe` keeps the source language.
   * `translate` turns speech into English. Never the default here.
   */
  task?: 'transcribe' | 'translate'
  /**
   * Ask Whisper for timestamps. `"segment"` returns each phrase with the model's
   * start and end, in seconds. `"word"` (or `true`) splits those phrases into
   * words and keeps them inside the phrase span. This ONNX build has no
   * cross-attentions, so it cannot time each word on its own. Omit it and
   * `chunks` is empty.
   */
  timestamps?: boolean | 'word' | 'segment'
}

export type STTAudioInput = Blob | File | string | Float32Array

type RawChunk = { text: string; timestamp: [number, number | null] }
type RawResult = { text: string; chunks?: RawChunk[] }

type ASRPipeline = {
  (audio: Float32Array | string, options?: Record<string, unknown>): Promise<RawResult | RawResult[]>
  tokenizer?: unknown
  dispose?: () => Promise<void>
}

/**
 * Speech-to-text that runs entirely in the browser (Whisper via Transformers.js).
 *
 * @example
 * ```ts
 * import { SpeechToText } from 'runonweb/stt'
 *
 * const stt = new SpeechToText()
 * await stt.load()
 * const { text } = await stt.transcribe(audioBlob)
 * console.log(text)
 * ```
 */
export class SpeechToText {
  #model: string
  #device: Device
  #language?: string
  #onProgress?: ProgressCallback
  #pipe: ASRPipeline | null = null
  #loading: Promise<void> | null = null
  #resolvedDevice: 'webgpu' | 'wasm' | null = null

  constructor(options: STTOptions = {}) {
    this.#model = options.model ?? DEFAULT_MODEL
    this.#device = options.device ?? 'auto'
    this.#language = options.language
    this.#onProgress = options.onProgress
  }

  /** Device actually used after `load()`. */
  get device(): 'webgpu' | 'wasm' | null {
    return this.#resolvedDevice
  }

  /** Download and initialize the model. Safe to call multiple times. */
  async load(): Promise<void> {
    if (this.#pipe) return
    if (this.#loading) return this.#loading

    this.#loading = (async () => {
      const { pipe, device } = await loadPipeline({
        task: 'automatic-speech-recognition',
        model: this.#model,
        device: this.#device,
        onProgress: this.#onProgress,
      })
      this.#pipe = pipe as unknown as ASRPipeline
      this.#resolvedDevice = device
    })()

    try {
      await this.#loading
    } finally {
      this.#loading = null
    }
  }

  /**
   * Transcribe audio. Accepts a Blob/File, a URL string, or raw Float32Array samples (16 kHz mono).
   * Pass `onPartial` to receive words as they are generated.
   */
  async transcribe(audio: STTAudioInput, options?: TranscribeOptions): Promise<STTResult> {
    await this.load()
    if (!this.#pipe) throw new Error('SpeechToText model failed to load')

    this.#onProgress?.({ status: 'transcribing' })

    const input = await prepareAudio(audio)
    const onPartial = options?.onPartial
    const timestampMode = resolveTimestamps(options?.timestamps)
    let streamed = ''

    const base: Record<string, unknown> = {
      chunk_length_s: 30,
      stride_length_s: 5,
      // Always transcribe unless asked otherwise. Multilingual Whisper will
      // otherwise slip into "translate to English" when language is unknown.
      task: options?.task ?? 'transcribe',
      // Without this, a clip that ends mid-word throws and the retry can come
      // back empty after the streamer already showed text.
      force_full_sequences: false,
    }
    const language = options?.language ?? this.#language
    if (language) base.language = language

    const run = async (stream: boolean) => {
      const streamer =
        stream && onPartial ? await createStreamer(this.#pipe!, (text) => {
          streamed = text
          onPartial(text)
        }) : undefined
      return this.#pipe!(input, {
        ...base,
        return_timestamps: timestampMode ? true : false,
        ...(streamer ? { streamer } : {}),
      })
    }

    let raw: RawResult | RawResult[]
    try {
      raw = await run(true)
    } catch {
      raw = await run(false)
    }

    const parsed = normalizeRaw(raw)
    const text = parsed.text || streamed
    const chunks =
      timestampMode === 'word'
        ? wordsInsideSegments(parsed.chunks)
        : timestampMode === 'segment'
          ? parsed.chunks
          : []

    this.#onProgress?.({ status: 'done' })
    return { text, chunks }
  }

  /** Release model resources. */
  dispose(): void {
    const pipe = this.#pipe
    this.#pipe = null
    this.#resolvedDevice = null
    void pipe?.dispose?.()
  }
}

/**
 * One-shot helper: load model, transcribe, dispose.
 */
export async function transcribe(
  audio: STTAudioInput,
  options?: STTOptions & TranscribeOptions
): Promise<STTResult> {
  const stt = new SpeechToText(options)
  try {
    return await stt.transcribe(audio, options)
  } finally {
    stt.dispose()
  }
}

function resolveTimestamps(option: TranscribeOptions['timestamps']): 'word' | 'segment' | false {
  if (option === true || option === 'word') return 'word'
  if (option === 'segment') return 'segment'
  return false
}

/** Place words inside the phrase span Whisper actually predicted. */
function wordsInsideSegments(segments: STTChunk[]): STTChunk[] {
  const words: STTChunk[] = []
  for (const segment of segments) {
    const parts = segment.text.trim().split(/\s+/).filter(Boolean)
    if (!parts.length) continue
    const start = segment.start
    const end = Math.max(segment.end, start)
    if (parts.length === 1) {
      words.push({ text: parts[0]!, start, end })
      continue
    }
    const weights = parts.map((part) => Math.max(1, part.length))
    const total = weights.reduce((sum, weight) => sum + weight, 0)
    const span = end - start
    let cursor = start
    parts.forEach((text, i) => {
      const dur = span * (weights[i]! / total)
      const next = i === parts.length - 1 ? end : cursor + dur
      words.push({ text, start: cursor, end: next })
      cursor = next
    })
  }
  return words
}

async function createStreamer(pipe: ASRPipeline, onPartial: (text: string) => void) {
  const { WhisperTextStreamer } = await import('@huggingface/transformers')
  let acc = ''
  return new WhisperTextStreamer(pipe.tokenizer as never, {
    skip_prompt: true,
    skip_special_tokens: true,
    callback_function: (piece: string) => {
      acc += piece
      const next = acc.replace(/\s+/g, ' ').trim()
      if (next) onPartial(next)
    },
  })
}

function normalizeRaw(raw: RawResult | RawResult[]): { text: string; chunks: STTChunk[] } {
  const items = Array.isArray(raw) ? raw : [raw]
  const text = items
    .map((item) => item.text ?? '')
    .join(' ')
    .replace(/\s+/g, ' ')
    .trim()
  return { text, chunks: items.flatMap((item) => parseChunks(item.chunks)) }
}

function parseChunks(raw?: RawChunk[]): STTChunk[] {
  if (!raw?.length) return []
  const chunks: STTChunk[] = []
  for (const item of raw) {
    const text = (item.text ?? '').replace(/\s+/g, ' ').trim()
    if (!text) continue
    const start = item.timestamp?.[0] ?? chunks.at(-1)?.end ?? 0
    const end = item.timestamp?.[1] ?? start
    chunks.push({ text, start, end: end < start ? start : end })
  }
  return chunks
}

export type { ProgressInfo, Device }

async function prepareAudio(audio: STTAudioInput): Promise<Float32Array | string> {
  if (typeof audio === 'string') return audio
  if (audio instanceof Float32Array) return audio

  const arrayBuffer = await audio.arrayBuffer()
  const AudioCtx =
    globalThis.AudioContext ??
    (globalThis as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext
  const ctx = new AudioCtx()
  try {
    const decoded = await ctx.decodeAudioData(arrayBuffer.slice(0))
    const mono = mixMono(decoded)
    return resample(mono, decoded.sampleRate, SAMPLE_RATE)
  } finally {
    await ctx.close()
  }
}

function mixMono(buffer: AudioBuffer): Float32Array {
  if (buffer.numberOfChannels === 1) return new Float32Array(buffer.getChannelData(0))
  const len = buffer.length
  const out = new Float32Array(len)
  const scale = buffer.numberOfChannels === 2 ? Math.SQRT1_2 : 1 / buffer.numberOfChannels
  for (let c = 0; c < buffer.numberOfChannels; c++) {
    const channel = buffer.getChannelData(c)
    for (let i = 0; i < len; i++) out[i] += channel[i]! * scale
  }
  return out
}

function resample(input: Float32Array, from: number, to: number): Float32Array {
  if (from === to) return input
  const ratio = from / to
  const out = new Float32Array(Math.round(input.length / ratio))
  for (let i = 0; i < out.length; i++) {
    const x = i * ratio
    const i0 = Math.min(Math.floor(x), input.length - 1)
    const i1 = Math.min(i0 + 1, input.length - 1)
    const t = x - i0
    out[i] = input[i0]! * (1 - t) + input[i1]! * t
  }
  return out
}
