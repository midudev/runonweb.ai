import type { Device, ProgressCallback, ProgressInfo, ResolvedDevice } from '../core/index.ts'
import { disposeKitten, loadKitten, streamKitten, type LoadedKitten } from './kitten.ts'
import { disposeKokoro, loadKokoro, streamKokoro, type LoadedKokoro } from './kokoro.ts'
import { disposeSupertonic, loadSupertonic, streamSupertonic, type LoadedSupertonic } from './supertonic.ts'
import { DEFAULT_TTS_SIZE, type TTSSize } from './sizes.ts'
import { concatFloat32, float32ToWavBlob } from './wav.ts'
import { defaultVoiceFor } from './voices.ts'
import type { SpeakChunk, TTSResult } from './types.ts'

export {
  DEFAULT_TTS_SIZE,
  TTS_SIZES,
  type TTSSize,
} from './sizes.ts'

export { SUPERTONIC_LANGUAGES, type SupertonicLanguage } from './supertonic.ts'

export {
  DEFAULT_MULTI_VOICE,
  DEFAULT_TINY_VOICE,
  DEFAULT_VOICE,
  TTS_LANGUAGES,
  TTS_VOICE_GROUPS,
  TTS_VOICES,
  defaultVoiceFor,
  getVoice,
  voiceGroupsFor,
  voiceLabel,
  voicesFor,
  type TTSGender,
  type TTSLanguage,
  type TTSLocale,
  type TTSVoice,
} from './voices.ts'

export { float32ToWavBlob } from './wav.ts'
export type { SpeakChunk, TTSResult }

export type TTSOptions = {
  /** `tiny` = KittenTTS (~28 MB). `small` = Kokoro 82M (default). `multi` = Supertonic 2 (5 languages). */
  size?: TTSSize
  /** Hugging Face model id. Kokoro / Supertonic only; ignored for `tiny`. */
  model?: string
  device?: Device
  /** Voice id. Defaults to `af_heart` (small), `bella` (tiny) or `st_f1` (multi). See `TTS_VOICES`. */
  voice?: string
  /** Speaking speed. `1` is natural. */
  speed?: number
  /** Language for `multi` (`en`, `ko`, `es`, `pt`, `fr`). Kokoro / Kitten infer it from the voice. */
  language?: string
  /** Denoising steps for `multi`. Higher is slower and slightly cleaner. Default 5. */
  steps?: number
  onProgress?: ProgressCallback
}

export type SpeakOptions = {
  voice?: string
  speed?: number
  language?: string
  steps?: number
}

/**
 * Text-to-speech that runs entirely in the browser.
 *
 * `small` (default) is Kokoro 82M: WebGPU, English / Spanish / French.
 * `tiny` is KittenTTS nano: ~28 MB, 8 English voices, WASM.
 * `multi` is Supertonic 2: ~262 MB, 44.1 kHz, English / Korean / Spanish / Portuguese / French.
 *
 * @example
 * ```ts
 * import { TextToSpeech } from 'runonweb/tts'
 *
 * const tts = new TextToSpeech({ voice: 'af_heart' })
 * await tts.load()
 * for await (const chunk of tts.speakStream('Hello from the browser')) {
 *   console.log(chunk.text, chunk.audio.length)
 * }
 * ```
 */
export class TextToSpeech {
  #size: TTSSize
  #model?: string
  #device: Device
  #voice: string
  #speed: number
  #language: string
  #steps?: number
  #onProgress?: ProgressCallback
  #kokoro: LoadedKokoro | null = null
  #kitten: LoadedKitten | null = null
  #supertonic: LoadedSupertonic | null = null
  #loading: Promise<void> | null = null
  #resolvedDevice: ResolvedDevice | null = null

  constructor(options: TTSOptions = {}) {
    this.#size = options.size ?? DEFAULT_TTS_SIZE
    this.#model = options.model
    this.#device = options.device ?? 'auto'
    this.#voice = options.voice ?? defaultVoiceFor(this.#size)
    this.#speed = options.speed ?? 1
    this.#language = options.language ?? 'en'
    this.#steps = options.steps
    this.#onProgress = options.onProgress
  }

  get size(): TTSSize {
    return this.#size
  }

  get device(): ResolvedDevice | null {
    return this.#resolvedDevice
  }

  get voice(): string {
    return this.#voice
  }

  async load(): Promise<void> {
    if (this.#kokoro || this.#kitten || this.#supertonic) return
    if (this.#loading) return this.#loading

    this.#loading = this.#loadInternal()
    try {
      await this.#loading
    } finally {
      this.#loading = null
    }
  }

  async #loadInternal(): Promise<void> {
    if (this.#size === 'tiny') {
      this.#kitten = await loadKitten(this.#onProgress)
      this.#resolvedDevice = 'wasm'
      return
    }
    if (this.#size === 'multi') {
      this.#supertonic = await loadSupertonic({
        model: this.#model,
        device: this.#device,
        onProgress: this.#onProgress,
      })
      this.#resolvedDevice = this.#supertonic.device
      return
    }
    this.#kokoro = await loadKokoro({
      model: this.#model,
      device: this.#device,
      onProgress: this.#onProgress,
    })
    this.#resolvedDevice = this.#kokoro.device
  }

  /** Yield PCM per sentence so the caller can play audio as it is generated. */
  async *speakStream(text: string, options: SpeakOptions = {}): AsyncGenerator<SpeakChunk> {
    await this.load()
    if (!text.trim()) throw new Error('Text is empty')

    this.#onProgress?.({ status: 'synthesizing' })
    const voice = options.voice ?? this.#voice
    const speed = options.speed ?? this.#speed

    const language = options.language ?? this.#language
    const steps = options.steps ?? this.#steps

    const stream =
      this.#size === 'tiny' && this.#kitten
        ? streamKitten(this.#kitten, text, voice, speed)
        : this.#size === 'multi' && this.#supertonic
          ? streamSupertonic(this.#supertonic, text, voice, speed, language, steps)
          : this.#kokoro
            ? streamKokoro(this.#kokoro, text, voice, speed)
            : null

    if (!stream) throw new Error('TextToSpeech model failed to load')

    for await (const chunk of stream) {
      yield chunk
    }
    this.#onProgress?.({ status: 'done' })
  }

  /** Synthesize speech from text. Returns PCM samples + sample rate. */
  async speak(text: string, options: SpeakOptions = {}): Promise<TTSResult> {
    const chunks: Float32Array[] = []
    let samplingRate = 24_000
    for await (const chunk of this.speakStream(text, options)) {
      chunks.push(chunk.audio)
      samplingRate = chunk.samplingRate
    }
    return { audio: concatFloat32(chunks), samplingRate }
  }

  /** Synthesize and return a WAV Blob ready for playback / download. */
  async speakToBlob(text: string, options?: SpeakOptions): Promise<Blob> {
    const { audio, samplingRate } = await this.speak(text, options)
    return float32ToWavBlob(audio, samplingRate)
  }

  dispose(): void {
    disposeKokoro(this.#kokoro)
    disposeKitten(this.#kitten)
    disposeSupertonic(this.#supertonic)
    this.#kokoro = null
    this.#kitten = null
    this.#supertonic = null
    this.#resolvedDevice = null
  }
}

export async function speak(text: string, options?: TTSOptions & SpeakOptions): Promise<TTSResult> {
  const tts = new TextToSpeech(options)
  try {
    return await tts.speak(text, options)
  } finally {
    tts.dispose()
  }
}

export type { ProgressInfo, Device }
