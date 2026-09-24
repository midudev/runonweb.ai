import type { Device, ProgressCallback, ProgressInfo } from '../core/index.ts'
import { loadPipeline } from '../core/pipeline.ts'

/**
 * Default weights: Superwhisper S1-mini as ONNX (Qwen3-0.6B fine-tune).
 * q4f16 on WebGPU (~339 MB) · q4 on WASM (~385 MB).
 */
export const DEFAULT_MODEL = 'onnx-community/s1-mini-ONNX'

/** Exact system prompt S1-mini was trained with. Do not reword. */
export const SYSTEM_PROMPT =
  'You are a text normalizer for speech-to-text transcripts. The input begins ' +
  'with a control line specifying the styling, structure, and context settings; ' +
  'clean the transcript to match those settings and output only the cleaned text.'

export const STYLING = ['casual', 'semi-casual', 'semi-formal', 'formal'] as const
export const STRUCTURE = ['prose', 'lists'] as const
export const CONTEXT = ['general', 'email'] as const

export type Styling = (typeof STYLING)[number]
export type Structure = (typeof STRUCTURE)[number]
export type Context = (typeof CONTEXT)[number]

export type CleanOptions = {
  /** Hugging Face model id. Defaults to `onnx-community/s1-mini-ONNX`. */
  model?: string
  device?: Device
  /** Written register. Default `semi-formal`. */
  styling?: Styling
  /** `lists` may emit Markdown bullets when there are 3+ items. Default `prose`. */
  structure?: Structure
  /** `email` adds greeting / body / sign-off layout. Default `general`. */
  context?: Context
  onProgress?: ProgressCallback
}

export type CleanRunOptions = {
  styling?: Styling
  structure?: Structure
  context?: Context
  /** Incremental decoded tokens as they arrive (think-block stripped). */
  onPartial?: (text: string) => void
}

export type CleanResult = {
  /** Cleaned written text. Empty when the input was only filler or noise. */
  text: string
}

type Tokenizer = {
  apply_chat_template: (messages: unknown, opts: Record<string, unknown>) => string
  (text: string): { input_ids?: { size?: number; dims?: number[] } }
}

type GenPipe = CallableFunction & {
  tokenizer?: Tokenizer
  dispose?: () => Promise<void>
}

const THINK_BLOCK = /<think>[\s\S]*?<\/think>\s*/g

function controlLine(styling: Styling, structure: Structure, context: Context): string {
  return `[Styling: ${styling}] [Structure: ${structure}] [Context: ${context}]`
}

function tokenCount(tokenizer: Tokenizer | undefined, text: string): number {
  if (!tokenizer) return Math.max(8, text.split(/\s+/).length)
  const ids = tokenizer(text).input_ids
  return ids?.dims?.at(-1) ?? ids?.size ?? Math.max(8, text.split(/\s+/).length)
}

function stripThink(text: string): string {
  return text.replace(THINK_BLOCK, '').trim()
}

function generatedText(raw: unknown): string {
  if (typeof raw === 'string') return raw
  if (Array.isArray(raw)) return generatedText(raw[0])
  if (raw && typeof raw === 'object' && 'generated_text' in raw) {
    const g = (raw as { generated_text: unknown }).generated_text
    if (typeof g === 'string') return g
    if (Array.isArray(g)) {
      const last = g.at(-1) as { content?: string } | undefined
      if (typeof last?.content === 'string') return last.content
    }
  }
  return ''
}

/**
 * Clean a raw speech-to-text transcript in the browser (S1-mini by Superwhisper).
 *
 * @example
 * ```ts
 * import { TranscriptCleaner } from 'runonweb/clean'
 *
 * const cleaner = new TranscriptCleaner()
 * await cleaner.load()
 *
 * const { text } = await cleaner.clean(
 *   'so um i need to like send the the report by uh friday no wait make that thursday'
 * )
 * // "I need to send the report by Thursday."
 * ```
 */
export class TranscriptCleaner {
  #model: string
  #device: Device
  #styling: Styling
  #structure: Structure
  #context: Context
  #onProgress?: ProgressCallback
  #pipe: GenPipe | null = null
  #loading: Promise<void> | null = null
  #resolvedDevice: 'webgpu' | 'wasm' | null = null

  constructor(options: CleanOptions = {}) {
    this.#model = options.model ?? DEFAULT_MODEL
    this.#device = options.device ?? 'auto'
    this.#styling = options.styling ?? 'semi-formal'
    this.#structure = options.structure ?? 'prose'
    this.#context = options.context ?? 'general'
    this.#onProgress = options.onProgress
  }

  get device(): 'webgpu' | 'wasm' | null {
    return this.#resolvedDevice
  }

  async load(): Promise<void> {
    if (this.#pipe) return
    if (this.#loading) return this.#loading

    this.#loading = (async () => {
      const attempts =
        this.#device === 'wasm'
          ? [{ device: 'wasm' as const, dtype: 'q4' as const }]
          : this.#device === 'webgpu'
            ? [
                { device: 'webgpu' as const, dtype: 'q4f16' as const },
                { device: 'webgpu' as const, dtype: 'q4' as const },
              ]
            : [
                { device: 'webgpu' as const, dtype: 'q4f16' as const },
                { device: 'webgpu' as const, dtype: 'q4' as const },
                { device: 'wasm' as const, dtype: 'q4' as const },
              ]

      let lastError: unknown
      for (const attempt of attempts) {
        try {
          const { pipe, device } = await loadPipeline({
            task: 'text-generation',
            model: this.#model,
            device: attempt.device,
            // q4f16 matches the official WebGPU demo; q4 is the compatible fallback.
            dtype: attempt.dtype,
            supportedDevices: [attempt.device],
            onProgress: this.#onProgress,
          })
          this.#pipe = pipe
          this.#resolvedDevice = device
          return
        } catch (err) {
          lastError = err
        }
      }
      const message = lastError instanceof Error ? lastError.message : String(lastError)
      throw new Error(
        /^\d+$/.test(message)
          ? `Could not start S1-mini on this device (${message}). Try again or pass device: "wasm".`
          : message
      )
    })()

    try {
      await this.#loading
    } finally {
      this.#loading = null
    }
  }

  /**
   * Rewrite a raw ASR transcript as clean written text.
   * Empty / whitespace-only input throws. Filler-only input returns `{ text: "" }`.
   */
  async clean(text: string, options: CleanRunOptions = {}): Promise<CleanResult> {
    await this.load()
    if (!this.#pipe) throw new Error('Cleaner model failed to load')
    if (!text.trim()) throw new Error('Text is empty')

    const styling = options.styling ?? this.#styling
    const structure = options.structure ?? this.#structure
    const context = options.context ?? this.#context
    const tokenizer = this.#pipe.tokenizer

    const messages = [
      { role: 'system', content: SYSTEM_PROMPT },
      { role: 'user', content: `${controlLine(styling, structure, context)}\n${text}` },
    ]

    // S1-mini was trained with thinking off. The pipeline does not pass this flag,
    // so the chat template must be applied here or the model emits an empty think block.
    const prompt = tokenizer
      ? tokenizer.apply_chat_template(messages, {
          tokenize: false,
          add_generation_prompt: true,
          enable_thinking: false,
        })
      : `${SYSTEM_PROMPT}\n${controlLine(styling, structure, context)}\n${text}`

    const maxNewTokens = Math.min(1024, Math.max(64, Math.ceil(tokenCount(tokenizer, text) * 1.3) + 32))

    this.#onProgress?.({ status: 'cleaning' })

    let streamed = ''
    let streamer: unknown
    if (options.onPartial && tokenizer) {
      const { TextStreamer } = await import('@huggingface/transformers')
      streamer = new TextStreamer(tokenizer as never, {
        skip_prompt: true,
        skip_special_tokens: true,
        callback_function: (chunk: string) => {
          streamed += chunk
          options.onPartial?.(stripThink(streamed))
        },
      })
    }

    const raw = await this.#pipe(prompt, {
      max_new_tokens: maxNewTokens,
      do_sample: false,
      return_full_text: false,
      ...(streamer ? { streamer } : {}),
    })

    const result = stripThink(generatedText(raw) || streamed)

    this.#onProgress?.({ status: 'done' })
    return { text: result }
  }

  dispose(): void {
    const pipe = this.#pipe
    this.#pipe = null
    this.#resolvedDevice = null
    void pipe?.dispose?.()
  }
}

export async function clean(
  text: string,
  options?: CleanOptions & CleanRunOptions
): Promise<CleanResult> {
  const cleaner = new TranscriptCleaner(options)
  try {
    return await cleaner.clean(text, options)
  } finally {
    cleaner.dispose()
  }
}

export type { ProgressInfo, Device }
