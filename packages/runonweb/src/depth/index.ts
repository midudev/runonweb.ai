import type { Device, ProgressCallback, ProgressInfo } from '../core/index.ts'
import { imageToPipelineInput, loadPipeline } from '../core/pipeline.ts'

const DEFAULT_MODEL = 'onnx-community/depth-anything-v2-small'

export type DepthOptions = {
  model?: string
  device?: Device
  onProgress?: ProgressCallback
}

export type DepthImageInput = Blob | File | string | HTMLImageElement | HTMLCanvasElement | ImageData

export type DepthResult = {
  /** Grayscale depth visualization as PNG. */
  depth: Blob
  width: number
  height: number
}

/**
 * Monocular depth estimation in the browser (Depth Anything).
 *
 * @example
 * ```ts
 * import { DepthEstimator } from 'runonweb/depth'
 *
 * const depth = new DepthEstimator()
 * await depth.load()
 * const { depth: png } = await depth.estimate(imageFile)
 * ```
 */
export class DepthEstimator {
  #model: string
  #device: Device
  #onProgress?: ProgressCallback
  #pipe: (CallableFunction & { dispose?: () => Promise<void> }) | null = null
  #loading: Promise<void> | null = null
  #resolvedDevice: 'webgpu' | 'wasm' | null = null

  constructor(options: DepthOptions = {}) {
    this.#model = options.model ?? DEFAULT_MODEL
    this.#device = options.device ?? 'auto'
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
        task: 'depth-estimation',
        model: this.#model,
        device: this.#device,
        // fp16 on WebGPU (~50 MB); q8 on WASM (~27 MB)
        dtype: (d) => (d === 'webgpu' ? 'fp16' : 'q8'),
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

  async estimate(image: DepthImageInput): Promise<DepthResult> {
    await this.load()
    if (!this.#pipe) throw new Error('DepthEstimator model failed to load')

    this.#onProgress?.({ status: 'processing' })
    const input = await imageToPipelineInput(image)
    const raw = (await this.#pipe(input)) as {
      depth: {
        width: number
        height: number
        toBlob?: () => Promise<Blob>
        toCanvas?: () => HTMLCanvasElement
      }
    }

    const depthImage = raw.depth
    let blob: Blob
    if (typeof depthImage.toBlob === 'function') {
      blob = await depthImage.toBlob()
    } else if (typeof depthImage.toCanvas === 'function') {
      const canvas = depthImage.toCanvas()
      blob = await new Promise<Blob>((resolve, reject) => {
        canvas.toBlob((b) => (b ? resolve(b) : reject(new Error('toBlob failed'))), 'image/png')
      })
    } else {
      throw new Error('Depth output has no toBlob/toCanvas')
    }

    this.#onProgress?.({ status: 'done' })
    return { depth: blob, width: depthImage.width, height: depthImage.height }
  }

  dispose(): void {
    const pipe = this.#pipe
    this.#pipe = null
    this.#resolvedDevice = null
    void pipe?.dispose?.()
  }
}

export async function estimateDepth(
  image: DepthImageInput,
  options?: DepthOptions
): Promise<DepthResult> {
  const estimator = new DepthEstimator(options)
  try {
    return await estimator.estimate(image)
  } finally {
    estimator.dispose()
  }
}

export type { ProgressInfo, Device }
