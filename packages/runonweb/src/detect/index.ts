import type { Device, ProgressCallback, ProgressInfo } from '../core/index.ts'
import { imageToPipelineInput, loadPipeline } from '../core/pipeline.ts'

const DEFAULT_MODEL = 'onnx-community/rfdetr_nano-ONNX'

/** RT-DETR family uses per-class sigmoid, not DETR's softmax + background class. */
const SIGMOID_TYPES = new Set(['rf_detr', 'rt_detr', 'rt_detr_v2', 'd_fine'])

function isSigmoidModelId(model: string): boolean {
  return /rfdetr|rt_detr|rtdetr|dfine|d_fine/i.test(model)
}

export type DetectOptions = {
  model?: string
  device?: Device
  /** Minimum confidence score (0–1). Default 0.5. */
  threshold?: number
  /** Return boxes as fractions of the image (0–1) instead of pixels. Default false. */
  normalized?: boolean
  onProgress?: ProgressCallback
}

export type DetectImageInput = Blob | File | string | HTMLImageElement | HTMLCanvasElement | ImageData

/** Box in pixel coordinates of the input image (or 0–1 fractions with `normalized: true`). */
export type DetectionBox = {
  xmin: number
  ymin: number
  xmax: number
  ymax: number
}

export type Detection = {
  label: string
  score: number
  box: DetectionBox
}

/**
 * Object detection in the browser (RF-DETR Nano by default).
 *
 * @example
 * ```ts
 * import { ObjectDetector } from 'runonweb/detect'
 *
 * const detector = new ObjectDetector()
 * await detector.load()
 * const objects = await detector.detect(imageFile)
 * // [{ label: 'cat', score: 0.98, box: { xmin, ymin, xmax, ymax } }]
 * ```
 */
export class ObjectDetector {
  #model: string
  #device: Device
  #threshold: number
  #normalized: boolean
  #onProgress?: ProgressCallback
  #pipe: DetectionPipe | null = null
  #loading: Promise<void> | null = null
  #resolvedDevice: 'webgpu' | 'wasm' | null = null

  constructor(options: DetectOptions = {}) {
    this.#model = options.model ?? DEFAULT_MODEL
    this.#device = options.device ?? 'auto'
    this.#threshold = options.threshold ?? 0.5
    this.#normalized = options.normalized ?? false
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
        task: 'object-detection',
        model: this.#model,
        device: this.#device,
        // These ONNX exports collapse logits on WebGPU with the current runtime.
        ...(isSigmoidModelId(this.#model)
          ? { supportedDevices: ['wasm'] as const, dtype: 'q8' as const }
          : { dtype: (d: 'webgpu' | 'wasm') => (d === 'webgpu' ? 'fp16' : 'q8') }),
        onProgress: this.#onProgress,
      })
      this.#pipe = pipe as DetectionPipe
      this.#resolvedDevice = device
    })()

    try {
      await this.#loading
    } finally {
      this.#loading = null
    }
  }

  async detect(image: DetectImageInput): Promise<Detection[]> {
    await this.load()
    if (!this.#pipe) throw new Error('ObjectDetector model failed to load')

    this.#onProgress?.({ status: 'processing' })
    const input = await imageToPipelineInput(image)
    const detections = usesSigmoid(this.#pipe)
      ? await detectSigmoid(this.#pipe, input, this.#threshold, this.#normalized)
      : await detectSoftmax(this.#pipe, input, this.#threshold, this.#normalized)

    this.#onProgress?.({ status: 'done' })
    return detections
  }

  dispose(): void {
    const pipe = this.#pipe
    this.#pipe = null
    this.#resolvedDevice = null
    void pipe?.dispose?.()
  }
}

export async function detect(
  image: DetectImageInput,
  options?: DetectOptions
): Promise<Detection[]> {
  const detector = new ObjectDetector(options)
  try {
    return await detector.detect(image)
  } finally {
    detector.dispose()
  }
}

export type { ProgressInfo, Device }

type DetectionPipe = CallableFunction & {
  dispose?: () => Promise<void>
  model?: {
    config?: { model_type?: string; id2label?: Record<string, string> }
    (inputs: Record<string, unknown>): Promise<{ logits: DetTensor; pred_boxes: DetTensor }>
  }
  processor?: (images: unknown) => Promise<{
    pixel_values: unknown
    pixel_mask?: unknown
    original_sizes?: Array<[number, number]>
  }>
}

type DetTensor = {
  dims: number[]
  data: ArrayLike<number>
  [index: number]: DetTensor
}

function usesSigmoid(pipe: DetectionPipe): boolean {
  const type = pipe.model?.config?.model_type
  return Boolean(type && SIGMOID_TYPES.has(type) && pipe.model && pipe.processor)
}

async function detectSoftmax(
  pipe: DetectionPipe,
  input: unknown,
  threshold: number,
  normalized: boolean
): Promise<Detection[]> {
  const raw = (await pipe(input, {
    threshold,
    percentage: normalized,
  })) as Array<{ label: string; score: number; box: DetectionBox }>
  return raw.map((d) => ({ label: d.label, score: d.score, box: d.box }))
}

async function detectSigmoid(
  pipe: DetectionPipe,
  input: unknown,
  threshold: number,
  normalized: boolean
): Promise<Detection[]> {
  const images = Array.isArray(input) ? input : [input]
  const processed = await pipe.processor!(images)
  const output = await pipe.model!({
    pixel_values: processed.pixel_values,
    pixel_mask: processed.pixel_mask,
  })

  const size = normalized
    ? null
    : sizeFrom(input, processed.original_sizes)
  return decodeSigmoid(output.logits, output.pred_boxes, {
    threshold,
    id2label: pipe.model!.config?.id2label ?? {},
    size,
  })
}

function sizeFrom(
  input: unknown,
  originalSizes?: Array<[number, number]>
): { width: number; height: number } | null {
  if (originalSizes?.[0]) {
    const [height, width] = originalSizes[0]
    return { width, height }
  }
  if (input && typeof input === 'object' && 'width' in input && 'height' in input) {
    const { width, height } = input as { width: number; height: number }
    if (width && height) return { width, height }
  }
  return null
}

function decodeSigmoid(
  logits: DetTensor,
  boxes: DetTensor,
  options: {
    threshold: number
    id2label: Record<string, string>
    size: { width: number; height: number } | null
  }
): Detection[] {
  const dims = logits.dims
  const numQueries = dims.length === 3 ? dims[1]! : dims[0]!
  const numClasses = dims.length === 3 ? dims[2]! : dims[1]!
  const batch = dims.length === 3 ? logits[0]! : logits
  const boxBatch = dims.length === 3 ? boxes[0]! : boxes
  const labels = labelList(options.id2label, numClasses)

  const out: Detection[] = []
  for (let j = 0; j < numQueries; j++) {
    const scores = batch[j]!.data
    let best = -Infinity
    let bestIdx = 0
    for (let k = 0; k < numClasses; k++) {
      const p = 1 / (1 + Math.exp(-scores[k]!))
      if (p > best) {
        best = p
        bestIdx = k
      }
    }
    if (best < options.threshold) continue
    const label = labels[bestIdx]
    if (!label) continue

    const [cx, cy, bw, bh] = Array.from(boxBatch[j]!.data)
    let xmin = cx! - bw! / 2
    let ymin = cy! - bh! / 2
    let xmax = cx! + bw! / 2
    let ymax = cy! + bh! / 2
    if (options.size) {
      xmin *= options.size.width
      xmax *= options.size.width
      ymin *= options.size.height
      ymax *= options.size.height
    }
    out.push({
      label,
      score: best,
      box: { xmin, ymin, xmax, ymax },
    })
  }
  return out
}

function labelList(id2label: Record<string, string>, numClasses: number): Array<string | undefined> {
  const byId = Object.entries(id2label)
    .map(([id, name]) => [Number(id), name] as const)
    .sort((a, b) => a[0] - b[0])
  if (byId.length === numClasses) return byId.map(([, name]) => name)

  const sparse: Array<string | undefined> = Array.from({ length: numClasses })
  for (const [id, name] of byId) {
    if (id >= 0 && id < numClasses) sparse[id] = name
  }
  return sparse
}
