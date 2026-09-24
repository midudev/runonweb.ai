import type { Device, ProgressCallback, ProgressInfo } from '../core/index.ts'
import { loadPipeline } from '../core/pipeline.ts'

/**
 * Default weights: `text2emoji-tiny`, a 2.4M-parameter T5 trained from scratch for runonweb
 * on the Text2Emoji dataset. 3.9 MB as q8 (encoder 1.7 MB + decoder 2.1 MB), MIT.
 * Trained with `training/text2emoji` in the runonweb repo.
 */
export const DEFAULT_MODEL = 'text2emoji-tiny'

export type EmojiOptions = {
  /** Model folder name (under `modelPath`) or Hugging Face repo id. */
  model?: string
  /**
   * Base URL the weights are served from, e.g. `/models/`. The runonweb site self-hosts them
   * in `public/models/`. Omit to download `model` from the Hugging Face Hub.
   */
  modelPath?: string
  device?: Device
  /** Maximum number of emojis to generate. Default 12. */
  maxEmojis?: number
  onProgress?: ProgressCallback
}

export type EmojiResult = {
  /** Emojis joined without separators, e.g. "🍕❤️🐶". */
  text: string
  /** One entry per emoji (grapheme cluster). */
  emojis: string[]
}

/**
 * Text → emoji translation in the browser with a 4 MB model.
 *
 * @example
 * ```ts
 * import { Emojifier } from 'runonweb/emoji'
 *
 * const emojifier = new Emojifier({ modelPath: '/models/' })
 * await emojifier.load()
 * const { text } = await emojifier.emojify('I love pizza and my dog')
 * // "🍕❤️🐶"
 * ```
 */
export class Emojifier {
  #model: string
  #modelPath?: string
  #device: Device
  #maxEmojis: number
  #onProgress?: ProgressCallback
  #pipe: (CallableFunction & { dispose?: () => Promise<void> }) | null = null
  #loading: Promise<void> | null = null
  #resolvedDevice: 'webgpu' | 'wasm' | null = null

  constructor(options: EmojiOptions = {}) {
    this.#model = options.model ?? DEFAULT_MODEL
    this.#modelPath = options.modelPath
    this.#device = options.device ?? 'auto'
    this.#maxEmojis = options.maxEmojis ?? 12
    this.#onProgress = options.onProgress
  }

  get device(): 'webgpu' | 'wasm' | null {
    return this.#resolvedDevice
  }

  async load(): Promise<void> {
    if (this.#pipe) return
    if (this.#loading) return this.#loading

    this.#loading = (async () => {
      const { pipe, device } = await loadPipeline({
        task: 'text2text-generation',
        model: this.#model,
        modelPath: this.#modelPath,
        device: this.#device,
        // q8 everywhere: the whole model is 3.9 MB and encoder-decoder graphs fail on WebGPU.
        dtype: 'q8',
        supportedDevices: ['wasm'],
        onProgress: this.#onProgress,
      })
      this.#pipe = pipe
      this.#resolvedDevice = device
    })()

    try {
      await this.#loading
    } finally {
      this.#loading = null
    }
  }

  /** Translate a sentence (English) into a short emoji sequence. */
  async emojify(text: string): Promise<EmojiResult> {
    await this.load()
    if (!this.#pipe) throw new Error('Emoji model failed to load')
    if (!text.trim()) throw new Error('Text is empty')

    this.#onProgress?.({ status: 'translating' })

    const raw = (await this.#pipe(text, {
      max_new_tokens: this.#maxEmojis + 1,
      // Each emoji is one token; never emit the same emoji twice.
      no_repeat_ngram_size: 1,
    })) as Array<{ generated_text: string }> | { generated_text: string }

    const out = Array.isArray(raw) ? raw[0] : raw
    // The tokenizer separates emojis with spaces; drop them and split back into graphemes.
    const emojis = (out?.generated_text ?? '').split(/\s+/).filter(Boolean)

    this.#onProgress?.({ status: 'done' })
    return { text: emojis.join(''), emojis }
  }

  dispose(): void {
    const pipe = this.#pipe
    this.#pipe = null
    this.#resolvedDevice = null
    void pipe?.dispose?.()
  }
}

export async function emojify(text: string, options?: EmojiOptions): Promise<EmojiResult> {
  const emojifier = new Emojifier(options)
  try {
    return await emojifier.emojify(text)
  } finally {
    emojifier.dispose()
  }
}

export type { ProgressInfo, Device }
