import type { Device, ProgressCallback, ProgressInfo, ResolvedDevice } from '../core/index.ts'
import { resolveDevice } from '../core/device.ts'
import { imageToPipelineInput } from '../core/pipeline.ts'
import { toProgressInfo } from '../core/progress.ts'

/**
 * Default: LFM2.5-VL-450M (Liquid AI, Nov 2025). Multilingual vision-language model.
 *
 * License: LFM Open License v1.0. Free for individuals and for companies under
 * USD 10M annual revenue. Larger companies need a commercial license from Liquid AI.
 * https://huggingface.co/LiquidAI/LFM2.5-VL-450M/blob/main/LICENSE
 */
const DEFAULT_MODEL = 'onnx-community/LFM2.5-VL-450M-ONNX'

/** Languages the default model was trained on. Any other value is passed to the prompt as-is. */
export const CAPTION_LANGUAGES = {
  en: 'English',
  es: 'Spanish',
  pt: 'Portuguese',
  fr: 'French',
  de: 'German',
  ar: 'Arabic',
  zh: 'Chinese',
  ja: 'Japanese',
  ko: 'Korean',
} as const

export type CaptionLanguage = keyof typeof CAPTION_LANGUAGES

const PROMPTS = {
  short: 'Describe this image in one short sentence, suitable as alt text. Do not start with "This image" or "The image".',
  detailed: 'Describe this image in two or three sentences.',
  more: 'Describe this image in detail: subjects, setting, colors, composition and any visible text.',
} as const

const MAX_TOKENS = {
  short: 48,
  detailed: 128,
  more: 320,
} as const

export type CaptionDetail = keyof typeof PROMPTS

export type CaptionOptions = {
  model?: string
  device?: Device
  /** Caption verbosity. `short` is alt-text style. */
  detail?: CaptionDetail
  /**
   * Caption language. A code from `CAPTION_LANGUAGES` or a plain language name ("Catalan").
   * Default `en`. The model was trained on the 9 listed languages; others are best-effort.
   */
  language?: CaptionLanguage | (string & {})
  /** Replace the built-in instruction. `language` is still appended. */
  prompt?: string
  /** Max new tokens for generation. Defaults depend on `detail`. */
  maxNewTokens?: number
  onProgress?: ProgressCallback
}

export type CaptionRunOptions = Pick<CaptionOptions, 'detail' | 'language' | 'prompt' | 'maxNewTokens'> & {
  /** Receives the caption as it is generated. */
  onPartial?: (text: string) => void
}

export type CaptionImageInput = Blob | File | string | HTMLImageElement | HTMLCanvasElement | ImageData

export type CaptionResult = {
  text: string
}

type Tensor = {
  dims: number[]
  slice: (...args: unknown[]) => Tensor
}

type VlmModel = {
  generate: (inputs: Record<string, unknown>) => Promise<Tensor>
  dispose?: () => Promise<void>
}

type VlmProcessor = {
  (image: unknown, text: string, options?: Record<string, unknown>): Promise<Record<string, unknown>>
  apply_chat_template: (messages: unknown, options?: Record<string, unknown>) => string
  batch_decode: (ids: Tensor, options: { skip_special_tokens: boolean }) => string[]
  tokenizer: unknown
  image_processor: { do_image_splitting?: boolean }
}

type RawImageLike = {
  width: number
  height: number
}

async function supportsShaderF16(): Promise<boolean> {
  try {
    const gpu = (globalThis as { navigator?: { gpu?: GPU } }).navigator?.gpu
    const adapter = await gpu?.requestAdapter()
    return Boolean(adapter?.features.has('shader-f16'))
  } catch {
    return false
  }
}

function dtypeFor(device: ResolvedDevice, fp16: boolean) {
  if (device === 'webgpu') {
    // Official LFM2.5-VL WebGPU recipe: fp16 vision + embeddings, q4f16 decoder.
    return fp16
      ? ({ embed_tokens: 'fp16', vision_encoder: 'fp16', decoder_model_merged: 'q4f16' } as const)
      : ({ embed_tokens: 'fp32', vision_encoder: 'fp32', decoder_model_merged: 'q4' } as const)
  }
  // WASM: q8/q4 embeddings use GatherBlockQuantized, which ONNX Runtime Web lacks on WASM. fp16 works.
  return { embed_tokens: 'fp16', vision_encoder: 'q8', decoder_model_merged: 'q4' } as const
}

async function toRawImage(image: CaptionImageInput): Promise<RawImageLike> {
  const input = await imageToPipelineInput(image)
  if (typeof input !== 'string') return input as unknown as RawImageLike
  const { RawImage } = await import('@huggingface/transformers')
  return RawImage.fromURL(input) as unknown as Promise<RawImageLike>
}

function languageName(language: string): string {
  return (CAPTION_LANGUAGES as Record<string, string>)[language] ?? language
}

function buildPrompt(detail: CaptionDetail, language: string, custom?: string): string {
  const base = custom ?? PROMPTS[detail]
  return `${base} Answer in ${languageName(language)}.`
}

/**
 * LFM2.5-VL ships a chat template that uses the `{% generation %}` tag, which the Jinja
 * engine bundled with Transformers.js 4.2 does not parse yet. Try the template first (so a
 * custom `model` keeps its own format) and fall back to the documented ChatML layout.
 */
function formatChat(processor: VlmProcessor, prompt: string): string {
  try {
    const messages = [{ role: 'user', content: [{ type: 'image' }, { type: 'text', text: prompt }] }]
    return processor.apply_chat_template(messages, { add_generation_prompt: true })
  } catch {
    return `<|startoftext|><|im_start|>user\n<image>${prompt}<|im_end|>\n<|im_start|>assistant\n`
  }
}

function cleanCaption(text: string): string {
  const t = text.trim().replace(/^["'“”]+|["'“”]+$/g, '').trim()
  return t.charAt(0).toUpperCase() + t.slice(1)
}

/**
 * Image captioning in the browser (LFM2.5-VL-450M, multilingual).
 *
 * @example
 * ```ts
 * import { ImageCaptioner } from 'runonweb/caption'
 *
 * const captioner = new ImageCaptioner({ language: 'es' })
 * await captioner.load()
 * const { text } = await captioner.caption(imageFile)
 * ```
 */
export class ImageCaptioner {
  #modelId: string
  #device: Device
  #detail: CaptionDetail
  #language: string
  #prompt?: string
  #maxNewTokens?: number
  #onProgress?: ProgressCallback
  #model: VlmModel | null = null
  #processor: VlmProcessor | null = null
  #loading: Promise<void> | null = null
  #resolvedDevice: 'webgpu' | 'wasm' | null = null

  constructor(options: CaptionOptions = {}) {
    this.#modelId = options.model ?? DEFAULT_MODEL
    this.#device = options.device ?? 'auto'
    this.#detail = options.detail ?? 'short'
    this.#language = options.language ?? 'en'
    this.#prompt = options.prompt
    this.#maxNewTokens = options.maxNewTokens
    this.#onProgress = options.onProgress
  }

  get device(): 'webgpu' | 'wasm' | null {
    return this.#resolvedDevice
  }

  async load(): Promise<void> {
    if (this.#model) return
    if (this.#loading) return this.#loading

    this.#loading = (async () => {
      const preferred = await resolveDevice(this.#device)
      try {
        await this.#loadOn(preferred)
      } catch (err) {
        if (preferred === 'wasm' || this.#device === 'wasm') throw err
        await this.#loadOn('wasm')
      }
    })()

    try {
      await this.#loading
    } finally {
      this.#loading = null
    }
  }

  async #loadOn(device: ResolvedDevice): Promise<void> {
    this.#onProgress?.({ status: 'loading', progress: 0 })

    const { AutoModelForImageTextToText, AutoProcessor, env } = await import('@huggingface/transformers')
    env.allowLocalModels = false

    const fp16 = device === 'webgpu' && (await supportsShaderF16())
    const progress_callback = (data: Record<string, unknown>) => {
      this.#onProgress?.(toProgressInfo(data))
    }

    const [model, processor] = await Promise.all([
      AutoModelForImageTextToText.from_pretrained(this.#modelId, {
        device,
        dtype: dtypeFor(device, fp16),
        progress_callback,
      }),
      AutoProcessor.from_pretrained(this.#modelId, { progress_callback }),
    ])

    const proc = processor as unknown as VlmProcessor
    // Images are resized to fit 512×512 anyway; tiling only adds tokens and latency for captions.
    if (proc.image_processor) proc.image_processor.do_image_splitting = false

    this.#model = model as unknown as VlmModel
    this.#processor = proc
    this.#resolvedDevice = device

    this.#onProgress?.({ status: 'ready', progress: 100 })
  }

  async caption(image: CaptionImageInput, options: CaptionRunOptions = {}): Promise<CaptionResult> {
    await this.load()
    const model = this.#model
    const processor = this.#processor
    if (!model || !processor) {
      throw new Error('ImageCaptioner model failed to load')
    }

    this.#onProgress?.({ status: 'captioning' })

    const detail = options.detail ?? this.#detail
    const language = options.language ?? this.#language
    const prompt = buildPrompt(detail, language, options.prompt ?? this.#prompt)
    const maxNewTokens = options.maxNewTokens ?? this.#maxNewTokens ?? MAX_TOKENS[detail]

    const chat = formatChat(processor, prompt)

    const rawImage = await toRawImage(image)
    const inputs = await processor(rawImage, chat, { add_special_tokens: false })

    let streamer: unknown
    if (options.onPartial) {
      const { TextStreamer } = await import('@huggingface/transformers')
      let partial = ''
      streamer = new TextStreamer(processor.tokenizer as never, {
        skip_prompt: true,
        skip_special_tokens: true,
        callback_function: (chunk: string) => {
          partial += chunk
          options.onPartial?.(cleanCaption(partial))
        },
      })
    }

    const output = await model.generate({
      ...inputs,
      max_new_tokens: maxNewTokens,
      do_sample: false,
      repetition_penalty: 1.05,
      ...(streamer ? { streamer } : {}),
    })

    const promptLength = (inputs.input_ids as Tensor).dims.at(-1) ?? 0
    const decoded = processor.batch_decode(output.slice(null, [promptLength, null]), { skip_special_tokens: true })
    const text = cleanCaption(decoded[0] ?? '')

    this.#onProgress?.({ status: 'done' })
    return { text }
  }

  dispose(): void {
    const model = this.#model
    this.#model = null
    this.#processor = null
    this.#resolvedDevice = null
    void model?.dispose?.()
  }
}

export async function caption(image: CaptionImageInput, options?: CaptionOptions): Promise<CaptionResult> {
  const captioner = new ImageCaptioner(options)
  try {
    return await captioner.caption(image)
  } finally {
    captioner.dispose()
  }
}

export type { ProgressInfo, Device }
