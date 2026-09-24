import type { Device, ProgressCallback, ProgressInfo } from '../core/index.ts'
import { loadPipeline } from '../core/pipeline.ts'
import { BergamotEngine, DEFAULT_MODEL_PATH, hasPair, resolveRoute, type Route } from './bergamot.ts'
import { PAIRS, RUNTIME_CDN, RUNTIME_VERSION, type Architecture, type PairEntry } from './registry.ts'

export { PAIRS, RUNTIME_CDN, RUNTIME_VERSION, DEFAULT_MODEL_PATH, hasPair, resolveRoute }
export type { Architecture, PairEntry, Route }

/**
 * Default weights: Mozilla's Firefox Translations models (MPL-2.0), the same ones that power
 * Firefox's built-in translation. Marian NMT students compiled to WASM through Bergamot,
 * 17–44 MB per pair, int8. Pairs without a direct model pivot through English.
 *
 * Pass `model` (a Hugging Face id such as `Xenova/m2m100_418M`) to use a Transformers.js
 * translation pipeline instead: one model for 100 languages, at ~630 MB.
 */
export const MULTILINGUAL_MODEL = 'Xenova/m2m100_418M'

export type TranslateOptions = {
  /**
   * Transformers.js model id (`Xenova/m2m100_418M`, `Xenova/opus-mt-en-es`…). Omit to use the
   * Firefox Translations models, which are smaller and faster.
   */
  model?: string
  /**
   * Base URL the Firefox Translations models are served from. Defaults to the runonweb mirror on
   * the Hugging Face Hub. Self-host with `scripts/translate-models.mjs` and pass `/models/translate/`.
   */
  modelPath?: string
  /** Base URL of the Bergamot WASM runtime. Defaults to jsDelivr; self-host next to the models. */
  runtimePath?: string
  device?: Device
  /** Source language code. Default `en`. */
  from?: string
  /** Target language code. Default `es`. */
  to?: string
  onProgress?: ProgressCallback
}

export type TranslateRunOptions = {
  from?: string
  to?: string
  /** Treat the input as HTML: tags are preserved and only text nodes are translated. */
  html?: boolean
}

export type TranslateResult = {
  text: string
}

type Pipe = CallableFunction & { dispose?: () => Promise<void> }

/**
 * Neural machine translation in the browser.
 *
 * @example
 * ```ts
 * import { Translator } from 'runonweb/translate'
 *
 * const translator = new Translator({ from: 'en', to: 'es' })
 * await translator.load()
 * const { text } = await translator.translate('Hello world')
 * // "Hola mundo"
 * ```
 */
export class Translator {
  #model: string | null
  #device: Device
  #from: string
  #to: string
  #onProgress?: ProgressCallback
  #engine: BergamotEngine | null = null
  #pipe: Pipe | null = null
  #loading: Promise<void> | null = null
  #resolvedDevice: 'webgpu' | 'wasm' | null = null
  #modelPath?: string
  #runtimePath?: string

  constructor(options: TranslateOptions = {}) {
    this.#from = options.from ?? 'en'
    this.#to = options.to ?? 'es'
    this.#model = options.model ?? null
    this.#device = options.device ?? 'auto'
    this.#onProgress = options.onProgress
    this.#modelPath = options.modelPath
    this.#runtimePath = options.runtimePath
  }

  /** True when the loaded model takes `src_lang` / `tgt_lang` (M2M100, NLLB). */
  get multilingual(): boolean {
    return this.#model !== null && !this.#model.includes('opus-mt')
  }

  /** Hugging Face id when a Transformers.js model is used, else `firefox-translations`. */
  get model(): string {
    return this.#model ?? 'firefox-translations'
  }

  get device(): 'webgpu' | 'wasm' | null {
    return this.#resolvedDevice
  }

  async load(): Promise<void> {
    if (this.#engine || this.#pipe) {
      if (this.#engine) await this.#engine.load(this.#from, this.#to)
      return
    }
    if (this.#loading) return this.#loading

    this.#loading = this.#model ? this.#loadTransformers(this.#model) : this.#loadBergamot()
    try {
      await this.#loading
    } finally {
      this.#loading = null
    }
  }

  async #loadBergamot(): Promise<void> {
    if (!hasPair(this.#from, this.#to)) {
      throw new Error(
        `No Firefox Translations model for ${this.#from} → ${this.#to}. See PAIRS in 'runonweb/translate', or pass model: '${MULTILINGUAL_MODEL}'.`
      )
    }
    const engine = new BergamotEngine({
      modelPath: this.#modelPath,
      runtimePath: this.#runtimePath,
      onProgress: this.#onProgress,
    })
    await engine.load(this.#from, this.#to)
    this.#engine = engine
    this.#resolvedDevice = 'wasm'
  }

  async #loadTransformers(model: string): Promise<void> {
    const { pipe, device } = await loadPipeline({
      task: 'translation',
      model,
      device: this.#device,
      // q8 on WASM. Encoder-decoder models fail on WebGPU in ONNX Runtime Web today.
      dtype: 'q8',
      supportedDevices: ['wasm'],
      onProgress: this.#onProgress,
    })
    this.#pipe = pipe
    this.#resolvedDevice = device
  }

  /**
   * Translate text. Optional `from` / `to` override constructor defaults (ISO codes like `en`,
   * `es`, `zh-Hant`). With the default models any supported pair can be requested per call;
   * new pairs are downloaded on demand.
   */
  async translate(text: string, options: TranslateRunOptions = {}): Promise<TranslateResult> {
    if (!text.trim()) throw new Error('Text is empty')
    await this.load()

    const src = options.from ?? this.#from
    const tgt = options.to ?? this.#to

    this.#onProgress?.({ status: 'translating' })

    let result: string
    if (this.#engine) {
      const [out] = await this.#engine.translate(src, tgt, [text], { html: options.html })
      result = out ?? ''
    } else {
      result = await this.#translateTransformers(text, src, tgt)
    }

    this.#onProgress?.({ status: 'done' })
    return { text: result.trim() }
  }

  async #translateTransformers(text: string, src: string, tgt: string): Promise<string> {
    const pipe = this.#pipe
    if (!pipe) throw new Error('Translator model failed to load')

    if (!this.multilingual && (src !== this.#from || tgt !== this.#to)) {
      throw new Error(
        `${this.#model} only translates ${this.#from} → ${this.#to}. Create a new Translator for ${src} → ${tgt}, or use model '${MULTILINGUAL_MODEL}'.`
      )
    }

    const raw = (await pipe(text, this.multilingual ? { src_lang: src, tgt_lang: tgt } : {})) as
      | Array<{ translation_text: string }>
      | { translation_text: string }
    const out = Array.isArray(raw) ? raw[0] : raw
    return out?.translation_text ?? ''
  }

  dispose(): void {
    const engine = this.#engine
    const pipe = this.#pipe
    this.#engine = null
    this.#pipe = null
    this.#resolvedDevice = null
    engine?.dispose()
    void pipe?.dispose?.()
  }
}

export async function translate(text: string, options?: TranslateOptions): Promise<TranslateResult> {
  const translator = new Translator(options)
  try {
    return await translator.translate(text, { from: options?.from, to: options?.to })
  } finally {
    translator.dispose()
  }
}

export type { ProgressInfo, Device }

/** Common language codes. Firefox Translations pairs use BCP 47 (`zh` is Simplified, `zh-Hant` Traditional). */
export const LANGS = {
  english: 'en',
  spanish: 'es',
  french: 'fr',
  german: 'de',
  portuguese: 'pt',
  italian: 'it',
  dutch: 'nl',
  russian: 'ru',
  chinese: 'zh',
  traditionalChinese: 'zh-Hant',
  japanese: 'ja',
  korean: 'ko',
  arabic: 'ar',
  hindi: 'hi',
  catalan: 'ca',
  galician: 'gl',
  basque: 'eu',
} as const

/** Sorted list of every direct pair, `<from>-<to>`. */
export const PAIR_KEYS: readonly string[] = Object.keys(PAIRS).sort()

/** Languages that can be translated to or from English. */
export const LANGUAGES: readonly string[] = [
  ...new Set(Object.values(PAIRS).flatMap((p) => [p.from, p.to])),
].sort()
