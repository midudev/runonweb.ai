import type { Device, ProgressCallback, ProgressInfo } from '../core/index.ts'
import { imageToPipelineInput, loadPipeline } from '../core/pipeline.ts'

/**
 * Default: BEN2 (MIT, 2025, ~219 MB fp16). General background eraser: hair, objects,
 * hard edges. The only ONNX weight is fp16.
 *
 * BEN2 fuses LayerNormalization with an fp16 activation and fp32 scale, bias, and
 * output. onnxruntime-web 1.30.0 through 1.31.0-dev.20260918 assigns `vec4<f16>` to
 * `vec4<f32>` storage, so the first WebGPU `OrtRun` fails with
 * `Invalid ShaderModule "LayerNorm"`. The workspace pins Transformers.js to
 * onnxruntime-web 1.29.0, the last release without the bug, until one ships
 * onnxruntime#32629. https://github.com/microsoft/onnxruntime/issues/32627
 */
const DEFAULT_MODEL = 'onnx-community/BEN2-ONNX'

/** Shader compile failures surface at run time, after the WebGPU session has loaded. */
function isWebGpuShaderError(err: unknown): boolean {
  const message = err instanceof Error ? err.message : String(err)
  return /Invalid ShaderModule|LayerNorm|failed to call OrtRun/i.test(message)
}

export type RemoveBgOptions = {
  /** Hugging Face model id. Defaults to BEN2. */
  model?: string
  /** Inference device. Defaults to `auto` (WebGPU when available, else WASM). */
  device?: Device
  /** Called while model files download / load. */
  onProgress?: ProgressCallback
}

export type RemoveBgImageInput =
  | Blob
  | File
  | string
  | HTMLImageElement
  | ImageData
  | HTMLCanvasElement

type BackgroundRemovalPipeline = {
  (
    images: Array<unknown>,
    options?: Record<string, unknown>
  ): Promise<Array<{ toBlob: () => Promise<Blob>; toCanvas: () => HTMLCanvasElement }>>
  dispose?: () => Promise<void>
}

/**
 * Background removal that runs entirely in the browser.
 *
 * @example
 * ```ts
 * import { RemoveBackground } from 'runonweb/remove-bg'
 *
 * const remover = new RemoveBackground()
 * await remover.load()
 * const png = await remover.remove(imageFile) // PNG Blob with alpha
 * ```
 */
export class RemoveBackground {
  #model: string
  #device: Device
  #onProgress?: ProgressCallback
  #pipe: BackgroundRemovalPipeline | null = null
  #loading: Promise<void> | null = null
  #resolvedDevice: 'webgpu' | 'wasm' | null = null

  constructor(options: RemoveBgOptions = {}) {
    this.#model = options.model ?? DEFAULT_MODEL
    this.#device = options.device ?? 'auto'
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
      const load = (device: Device) =>
        loadPipeline({
          task: 'background-removal',
          model: this.#model,
          device,
          // BEN2 ships fp16 only (~219 MB).
          dtype: 'fp16',
          onProgress: this.#onProgress,
        })

      try {
        const { pipe, device } = await load(this.#device)
        this.#pipe = pipe as unknown as BackgroundRemovalPipeline
        this.#resolvedDevice = device
      } catch (err) {
        if (this.#device === 'wasm') throw err
        const { pipe, device } = await load('wasm')
        this.#pipe = pipe as unknown as BackgroundRemovalPipeline
        this.#resolvedDevice = device
      }
    })()

    try {
      await this.#loading
    } finally {
      this.#loading = null
    }
  }

  /**
   * Remove the background from an image.
   * Returns a PNG `Blob` with an alpha channel.
   */
  async remove(image: RemoveBgImageInput): Promise<Blob> {
    try {
      return await this.#remove(image)
    } catch (err) {
      // A WebGPU session can load and still die in a shader at run time. Retry once on WASM.
      if (this.#resolvedDevice !== 'webgpu' || !isWebGpuShaderError(err)) throw err
      this.dispose()
      this.#device = 'wasm'
      return await this.#remove(image)
    }
  }

  async #remove(image: RemoveBgImageInput): Promise<Blob> {
    await this.load()
    if (!this.#pipe) throw new Error('RemoveBackground model failed to load')

    this.#onProgress?.({ status: 'processing' })

    const input = await imageToPipelineInput(image)
    const output = await this.#pipe([input])
    const result = output[0]
    if (!result) throw new Error('Background removal produced no output')

    const blob = await result.toBlob()
    this.#onProgress?.({ status: 'done' })
    return blob
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
 * One-shot helper: load model, remove background, dispose.
 */
export async function removeBackground(
  image: RemoveBgImageInput,
  options?: RemoveBgOptions
): Promise<Blob> {
  const remover = new RemoveBackground(options)
  try {
    return await remover.remove(image)
  } finally {
    remover.dispose()
  }
}

export type { ProgressInfo, Device }
