import {
  resolveDevice,
  toProgressInfo,
  type Device,
  type ProgressCallback,
  type ResolvedDevice,
} from './index.ts'

export type DTypeName = 'auto' | 'fp32' | 'fp16' | 'q8' | 'int8' | 'uint8' | 'q4' | 'bnb4' | 'q4f16'
export type DType = DTypeName | Record<string, DTypeName>

export type PipelineLoadOptions = {
  task: string
  model: string
  /** Defaults to `auto` (WebGPU when available, else WASM). */
  device?: Device
  /** Quantization. A function receives the resolved device so modules can pick per backend. */
  dtype?: DType | ((device: ResolvedDevice) => DType)
  /**
   * Backends this model is known to work on. If the resolved device is not listed,
   * the loader falls back to WASM instead of failing at session creation.
   */
  supportedDevices?: ResolvedDevice[]
  /**
   * Base URL for self-hosted weights (e.g. `/models/`). When set, `model` is resolved as
   * `<modelPath>/<model>/…` instead of the Hugging Face Hub.
   */
  modelPath?: string
  onProgress?: ProgressCallback
  pipelineOptions?: Record<string, unknown>
}

export type LoadedPipeline = {
  pipe: CallableFunction & { dispose?: () => Promise<void> }
  device: 'webgpu' | 'wasm'
}

/**
 * Shared loader for Transformers.js pipelines with device resolution + progress.
 */
export async function loadPipeline(options: PipelineLoadOptions): Promise<LoadedPipeline> {
  let device = await resolveDevice(options.device ?? 'auto')
  if (options.supportedDevices && !options.supportedDevices.includes(device)) {
    device = 'wasm'
  }
  options.onProgress?.({ status: 'loading', progress: 0 })

  const { pipeline, env } = await import('@huggingface/transformers')
  if (options.modelPath) {
    env.allowLocalModels = true
    env.allowRemoteModels = false
    env.localModelPath = options.modelPath
  } else {
    env.allowLocalModels = false
    env.allowRemoteModels = true
  }

  const dtype =
    typeof options.dtype === 'function'
      ? options.dtype(device)
      : (options.dtype ?? (device === 'webgpu' ? 'fp32' : 'q8'))

  const pipe = await pipeline(options.task as never, options.model, {
    device,
    dtype,
    progress_callback: (data: Record<string, unknown>) => {
      options.onProgress?.(toProgressInfo(data))
    },
    ...options.pipelineOptions,
  })

  options.onProgress?.({ status: 'ready', progress: 100 })
  return { pipe: pipe as LoadedPipeline['pipe'], device }
}

export async function imageToPipelineInput(
  image: Blob | File | string | HTMLImageElement | HTMLCanvasElement | ImageData
): Promise<string | import('@huggingface/transformers').RawImage> {
  if (typeof image === 'string') return image

  const { RawImage } = await import('@huggingface/transformers')

  if (image instanceof HTMLImageElement) {
    if (image.src && !image.src.startsWith('blob:')) return image.src
    const canvas = document.createElement('canvas')
    canvas.width = image.naturalWidth || image.width
    canvas.height = image.naturalHeight || image.height
    const ctx = canvas.getContext('2d')
    if (!ctx) throw new Error('Could not get 2d context')
    ctx.drawImage(image, 0, 0)
    return RawImage.fromCanvas(canvas)
  }

  if (image instanceof HTMLCanvasElement) {
    return RawImage.fromCanvas(image)
  }

  if (image instanceof ImageData) {
    return new RawImage(image.data, image.width, image.height, 4)
  }

  const url = URL.createObjectURL(image)
  try {
    return await RawImage.fromURL(url)
  } finally {
    URL.revokeObjectURL(url)
  }
}
