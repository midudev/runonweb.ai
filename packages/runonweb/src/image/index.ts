import type { ProgressCallback, ProgressInfo } from '../core/index.ts'
import { isWebGPUAvailable } from '../core/device.ts'
import type { BonsaiImagePipeline } from './engine.js'
import {
  DEFAULT_IMAGE_SIZE,
  IMAGE_MODELS,
  IMAGE_SIZES,
  type ImageSize,
} from './sizes.ts'

export { DEFAULT_IMAGE_SIZE, IMAGE_MODELS, IMAGE_SIZES }
export type { ImageSize }
export type { ProgressInfo }

const LOAD_WEIGHTS: Record<string, number> = {
  text_encoder: 0.58,
  transformer: 0.38,
  vae: 0.04,
}

const MIN_SIDE = 256
const MAX_SIDE = 1280

export type ImageGeneratorOptions = {
  /** Weight variant. `binary` is smaller; `ternary` is closer to FLUX.2 Klein. */
  size?: ImageSize
  /** Hugging Face repo id. Overrides `size`. Must be an `-mlx-` Bonsai Image variant. */
  model?: string
  /** Default width in pixels (multiple of 16). Default 512. */
  width?: number
  /** Default height in pixels (multiple of 16). Default 512. */
  height?: number
  /** FlowMatch-Euler steps. The model is tuned for 4. */
  steps?: number
  /** Fixed seed. Omit for a random seed on each generate. */
  seed?: number
  onProgress?: ProgressCallback
}

export type GenerateOptions = {
  width?: number
  height?: number
  steps?: number
  seed?: number
  signal?: AbortSignal
  onStep?: (info: { step: number; steps: number }) => void
}

export type ImageResult = {
  /** PNG Blob. */
  image: Blob
  width: number
  height: number
  prompt: string
  seed: number
}

function alignSide(value: number, label: string): number {
  if (!Number.isFinite(value)) throw new Error(`${label} must be a number`)
  const aligned = Math.round(value / 16) * 16
  if (aligned < MIN_SIDE || aligned > MAX_SIDE) {
    throw new Error(`${label} must be between ${MIN_SIDE} and ${MAX_SIDE} (got ${value})`)
  }
  return aligned
}

function randomSeed(): number {
  return Math.floor(Math.random() * 1_000_000_000)
}

function mapLoadProgress(onProgress?: ProgressCallback): (status: Record<string, unknown>) => void {
  const parts: Record<string, number> = {}
  return (status) => {
    const component = typeof status.component === 'string' ? status.component : undefined
    const loaded = typeof status.loaded === 'number' ? status.loaded : 0
    const total = typeof status.total === 'number' ? status.total : 0
    if (component && total > 0) parts[component] = loaded / total

    let progress = 0
    for (const [name, weight] of Object.entries(LOAD_WEIGHTS)) {
      progress += (parts[name] ?? 0) * weight
    }

    onProgress?.({
      status: component ? `loading ${component}` : 'loading',
      progress: progress * 100,
      file: component,
    })
  }
}

/**
 * Text-to-image in the browser (Bonsai Image 4B, WebGPU).
 *
 * @example
 * ```ts
 * import { ImageGenerator } from 'runonweb/image'
 *
 * const gen = new ImageGenerator()
 * await gen.load()
 * const { image } = await gen.generate('A bonsai tree in a ceramic studio')
 * ```
 */
export class ImageGenerator {
  #modelId: string
  #width: number
  #height: number
  #steps: number
  #seed?: number
  #onProgress?: ProgressCallback
  #pipeline: BonsaiImagePipeline | null = null
  #loading: Promise<void> | null = null
  #resolvedDevice: 'webgpu' | null = null

  constructor(options: ImageGeneratorOptions = {}) {
    const size = options.size ?? DEFAULT_IMAGE_SIZE
    this.#modelId = options.model ?? IMAGE_SIZES[size].model
    this.#width = options.width ?? 512
    this.#height = options.height ?? 512
    this.#steps = options.steps ?? 4
    this.#seed = options.seed
    this.#onProgress = options.onProgress
  }

  get device(): 'webgpu' | null {
    return this.#resolvedDevice
  }

  get model(): string {
    return this.#modelId
  }

  async load(): Promise<void> {
    if (this.#pipeline) return
    if (this.#loading) return this.#loading

    this.#loading = (async () => {
      const { BonsaiImagePipeline } = await import('./engine.js')
      if (!(await isWebGPUAvailable()) || !(await BonsaiImagePipeline.isSupported())) {
        throw new Error('ImageGenerator requires WebGPU. This model has no WASM fallback.')
      }

      this.#onProgress?.({ status: 'loading', progress: 0 })

      const pipeline = await BonsaiImagePipeline.from_pretrained(this.#modelId, {
        onProgress: mapLoadProgress(this.#onProgress),
      })

      this.#pipeline = pipeline
      this.#resolvedDevice = 'webgpu'
      this.#onProgress?.({ status: 'ready', progress: 100 })
    })()

    try {
      await this.#loading
    } finally {
      this.#loading = null
    }
  }

  async generate(prompt: string, options?: GenerateOptions): Promise<ImageResult> {
    await this.load()
    const pipeline = this.#pipeline
    if (!pipeline) throw new Error('ImageGenerator model failed to load')

    const text = prompt.trim()
    if (!text) throw new Error('Prompt is empty')

    const width = alignSide(options?.width ?? this.#width, 'width')
    const height = alignSide(options?.height ?? this.#height, 'height')
    const steps = options?.steps ?? this.#steps
    if (!Number.isInteger(steps) || steps < 1 || steps > 50) {
      throw new Error('steps must be an integer between 1 and 50')
    }
    const seed = options?.seed ?? this.#seed ?? randomSeed()

    this.#onProgress?.({ status: 'generating', progress: 0 })

    const result = await pipeline.generate({
      prompt: text,
      width,
      height,
      numInferenceSteps: steps,
      seed,
      signal: options?.signal,
      callbackOnStepEnd: (_pipeline, step) => {
        const done = step + 1
        this.#onProgress?.({ status: 'generating', progress: (done / steps) * 100 })
        options?.onStep?.({ step: done, steps })
      },
    })

    this.#onProgress?.({ status: 'done', progress: 100 })
    return {
      image: result.toBlob(),
      width: result.width,
      height: result.height,
      prompt: result.prompt,
      seed: result.seed,
    }
  }

  dispose(): void {
    const pipeline = this.#pipeline
    this.#pipeline = null
    this.#resolvedDevice = null
    void pipeline?.destroy()
  }
}

export async function generateImage(
  prompt: string,
  options?: ImageGeneratorOptions
): Promise<ImageResult> {
  const generator = new ImageGenerator(options)
  try {
    return await generator.generate(prompt)
  } finally {
    generator.dispose()
  }
}

export async function isImageGenerationSupported(): Promise<boolean> {
  return isWebGPUAvailable()
}

export async function clearImageCache(): Promise<void> {
  const { BonsaiImagePipeline } = await import('./engine.js')
  await BonsaiImagePipeline.clearCache()
}
