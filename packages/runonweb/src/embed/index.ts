import type { Device, ProgressCallback, ProgressInfo } from '../core/index.ts'
import { loadPipeline } from '../core/pipeline.ts'

const DEFAULT_MODEL = 'Xenova/all-MiniLM-L6-v2'

export type EmbedOptions = {
  model?: string
  device?: Device
  /** Pooling strategy. Default `mean`. */
  pooling?: 'mean' | 'cls' | 'none'
  /** L2-normalize vectors. Default true. */
  normalize?: boolean
  onProgress?: ProgressCallback
}

export type EmbedResult = {
  /** Embedding vector(s). One vector if input was a string; one per item if array. */
  embeddings: Float32Array[]
  dimensions: number
}

/**
 * Text embeddings in the browser (sentence-transformers style).
 *
 * @example
 * ```ts
 * import { TextEmbedder } from 'runonweb/embed'
 *
 * const embedder = new TextEmbedder()
 * await embedder.load()
 * const { embeddings } = await embedder.embed('hello world')
 * const sim = cosineSimilarity(embeddings[0], other)
 * ```
 */
export class TextEmbedder {
  #model: string
  #device: Device
  #pooling: 'mean' | 'cls' | 'none'
  #normalize: boolean
  #onProgress?: ProgressCallback
  #pipe: (CallableFunction & { dispose?: () => Promise<void> }) | null = null
  #loading: Promise<void> | null = null
  #resolvedDevice: 'webgpu' | 'wasm' | null = null

  constructor(options: EmbedOptions = {}) {
    this.#model = options.model ?? DEFAULT_MODEL
    this.#device = options.device ?? 'auto'
    this.#pooling = options.pooling ?? 'mean'
    this.#normalize = options.normalize ?? true
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
        task: 'feature-extraction',
        model: this.#model,
        device: this.#device,
        // fp16 on WebGPU (~45 MB); q8 on WASM (~23 MB).
        // q8 on WebGPU produces garbage similarities. Never use it there.
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

  async embed(text: string | string[]): Promise<EmbedResult> {
    await this.load()
    if (!this.#pipe) throw new Error('TextEmbedder model failed to load')

    this.#onProgress?.({ status: 'embedding' })
    const inputs = Array.isArray(text) ? text : [text]
    if (inputs.length === 0) throw new Error('No text provided')

    const embeddings: Float32Array[] = []
    for (const item of inputs) {
      const output = await this.#pipe(item, {
        pooling: this.#pooling,
        normalize: this.#normalize,
      })
      // Transformers.js returns a Tensor-like with .data or nested arrays
      const vector = tensorToFloat32(output)
      embeddings.push(vector)
    }

    this.#onProgress?.({ status: 'done' })
    return {
      embeddings,
      dimensions: embeddings[0]?.length ?? 0,
    }
  }

  dispose(): void {
    const pipe = this.#pipe
    this.#pipe = null
    this.#resolvedDevice = null
    void pipe?.dispose?.()
  }
}

export async function embed(
  text: string | string[],
  options?: EmbedOptions
): Promise<EmbedResult> {
  const embedder = new TextEmbedder(options)
  try {
    return await embedder.embed(text)
  } finally {
    embedder.dispose()
  }
}

/** Cosine similarity between two equal-length vectors. */
export function cosineSimilarity(a: Float32Array, b: Float32Array): number {
  if (a.length !== b.length) throw new Error('Vector length mismatch')
  let dot = 0
  let na = 0
  let nb = 0
  for (let i = 0; i < a.length; i++) {
    const x = a[i]!
    const y = b[i]!
    dot += x * y
    na += x * x
    nb += y * y
  }
  const denom = Math.sqrt(na) * Math.sqrt(nb)
  return denom === 0 ? 0 : dot / denom
}

export type { ProgressInfo, Device }

function tensorToFloat32(output: unknown): Float32Array {
  if (output instanceof Float32Array) return output
  if (output && typeof output === 'object') {
    const o = output as { data?: ArrayLike<number>; tolist?: () => unknown }
    if (o.data) return new Float32Array(o.data as ArrayLike<number>)
    if (typeof o.tolist === 'function') {
      const list = o.tolist()
      return new Float32Array(flattenNumbers(list))
    }
  }
  if (Array.isArray(output)) {
    return new Float32Array(flattenNumbers(output))
  }
  throw new Error('Unexpected embedding output shape')
}

function flattenNumbers(value: unknown): number[] {
  if (typeof value === 'number') return [value]
  if (!Array.isArray(value)) return []
  const out: number[] = []
  for (const item of value) {
    if (typeof item === 'number') out.push(item)
    else out.push(...flattenNumbers(item))
  }
  return out
}
