import type { Device, ProgressCallback, ProgressInfo, ResolvedDevice } from '../core/index.ts'
import { resolveDevice } from '../core/device.ts'
import { PaddleOcrService } from 'paddleocr'
import { DEFAULT_OCR_SIZE, type OCRSize } from './sizes.ts'

export { DEFAULT_OCR_SIZE, OCR_SIZES, type OCRSize } from './sizes.ts'

const HF = {
  tiny: {
    det: 'https://huggingface.co/PaddlePaddle/PP-OCRv6_tiny_det_onnx/resolve/main/inference.onnx',
    rec: 'https://huggingface.co/PaddlePaddle/PP-OCRv6_tiny_rec_onnx/resolve/main/inference.onnx',
    yml: 'https://huggingface.co/PaddlePaddle/PP-OCRv6_tiny_rec_onnx/resolve/main/inference.yml',
  },
  small: {
    det: 'https://huggingface.co/PaddlePaddle/PP-OCRv6_small_det_onnx/resolve/main/inference.onnx',
    rec: 'https://huggingface.co/PaddlePaddle/PP-OCRv6_small_rec_onnx/resolve/main/inference.onnx',
    yml: 'https://huggingface.co/PaddlePaddle/PP-OCRv6_small_rec_onnx/resolve/main/inference.yml',
  },
  medium: {
    det: 'https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_det_onnx/resolve/main/inference.onnx',
    rec: 'https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_rec_onnx/resolve/main/inference.onnx',
    yml: 'https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_rec_onnx/resolve/main/inference.yml',
  },
} as const

const PRESET = {
  tiny: 'PP-OCRv6_tiny',
  small: 'PP-OCRv6_small',
  medium: 'PP-OCRv6_medium',
} as const

const CACHE = 'runonweb-ppocrv6'
const ORT_WASM = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/'

export type OCROptions = {
  /** Model tier. Defaults to `small` (best size/quality). */
  size?: OCRSize
  /** Inference device. Defaults to `auto` (WebGPU when available, else WASM). */
  device?: Device
  onProgress?: ProgressCallback
}

export type OCRImageInput = Blob | File | string | HTMLImageElement | ImageData | HTMLCanvasElement

export type OCRBox = {
  xmin: number
  ymin: number
  xmax: number
  ymax: number
}

export type OCRLine = {
  text: string
  score: number
  box: OCRBox
}

export type OCRResult = {
  text: string
  lines: OCRLine[]
  confidence: number
}

/**
 * On-device OCR with PP-OCRv6 (PaddleOCR, Apache-2.0).
 *
 * @example
 * ```ts
 * import { OCR } from 'runonweb/ocr'
 *
 * const ocr = new OCR() // size: 'small'
 * await ocr.load()
 * const { text, lines } = await ocr.read(imageFile)
 * ```
 */
export class OCR {
  #size: OCRSize
  #device: Device
  #onProgress?: ProgressCallback
  #service: PaddleOcrService | null = null
  #loading: Promise<void> | null = null
  #pending: Promise<void> = Promise.resolve()
  #resolvedDevice: ResolvedDevice | null = null
  #disposed = false
  #disposeDone: Promise<void> | null = null

  constructor(options: OCROptions = {}) {
    this.#size = options.size ?? DEFAULT_OCR_SIZE
    this.#device = options.device ?? 'auto'
    this.#onProgress = options.onProgress
  }

  get size(): OCRSize {
    return this.#size
  }

  get device(): ResolvedDevice | null {
    return this.#resolvedDevice
  }

  async load(): Promise<void> {
    if (this.#disposed) throw new Error('OCR model was disposed')
    if (this.#service) return
    if (this.#loading) return this.#loading

    this.#loading = (async () => {
      const urls = HF[this.#size]
      this.#onProgress?.({ status: 'loading', progress: 0 })

      const [det, rec, yml] = await Promise.all([
        fetchCached(urls.det, 'det.onnx', this.#onProgress),
        fetchCached(urls.rec, 'rec.onnx', this.#onProgress),
        fetchCachedText(urls.yml),
      ])

      const charactersDictionary = parseCharacterDict(yml)
      if (charactersDictionary[charactersDictionary.length - 1] !== ' ') {
        charactersDictionary.push(' ')
      }

      const ort = await loadOrt()
      const preferred = await resolveDevice(this.#device)
      const service = await createService({
        ort,
        size: this.#size,
        det,
        rec,
        charactersDictionary,
        device: preferred,
      })

      if (this.#disposed) {
        await takeTurn(() => service.ocr.destroy())
        return
      }

      this.#service = service.ocr
      this.#resolvedDevice = service.device
      this.#onProgress?.({ status: 'ready', progress: 100 })
    })()

    try {
      await this.#loading
    } finally {
      this.#loading = null
    }
  }

  read(image: OCRImageInput): Promise<OCRResult> {
    if (this.#disposed) return Promise.reject(new Error('OCR model was disposed'))
    const job = this.#read(image)
    this.#pending = this.#pending.then(
      () => job.then(
        () => undefined,
        () => undefined,
      ),
    )
    return job
  }

  /**
   * Drop the sessions. Waits for an in-flight `read` so `destroy` cannot
   * release a session that is still inside OrtRun.
   */
  async dispose(): Promise<void> {
    if (this.#disposeDone) return this.#disposeDone
    this.#disposed = true
    this.#disposeDone = (async () => {
      await this.#pending
      await this.#loading
      const service = this.#service
      this.#service = null
      this.#resolvedDevice = null
      if (!service) return
      await takeTurn(() => service.destroy())
    })()
    return this.#disposeDone
  }

  async #read(image: OCRImageInput): Promise<OCRResult> {
    if (this.#disposed) throw new Error('OCR model was disposed')
    await this.load()
    if (this.#disposed) throw new Error('OCR model was disposed')
    const service = this.#service
    if (!service) throw new Error('OCR model failed to load')

    this.#onProgress?.({ status: 'recognizing' })
    const pixels = await imageToPixels(image)
    const raw = await takeTurn(() => {
      if (this.#disposed) throw new Error('OCR model was disposed')
      return service.recognize(pixels)
    })
    const processed = service.processRecognition(raw)

    const lines: OCRLine[] = raw.map((item) => ({
      text: item.text,
      score: item.confidence,
      box: {
        xmin: item.box.x,
        ymin: item.box.y,
        xmax: item.box.x + item.box.width,
        ymax: item.box.y + item.box.height,
      },
    }))

    this.#onProgress?.({ status: 'done' })
    return {
      text: processed.text,
      lines,
      confidence: processed.confidence,
    }
  }
}

export async function read(image: OCRImageInput, options?: OCROptions): Promise<OCRResult> {
  const ocr = new OCR(options)
  try {
    return await ocr.read(image)
  } finally {
    await ocr.dispose()
  }
}

/**
 * onnxruntime-web's WebGPU backend keeps a single global OrtRun slot.
 * A second run throws "Session already started", and its `finally` clears
 * that slot, so the first run then throws "Session mismatch".
 */
let ortTurn: Promise<void> = Promise.resolve()

function takeTurn<T>(fn: () => Promise<T>): Promise<T> {
  const run = ortTurn.then(fn, fn)
  ortTurn = run.then(
    () => undefined,
    () => undefined,
  )
  return run
}

export type { ProgressInfo, Device }

type OrtApi = {
  env: { wasm: { wasmPaths: string; numThreads: number } }
  Tensor: unknown
  InferenceSession: {
    create: (model: ArrayBuffer, options?: { executionProviders?: string[] }) => Promise<unknown>
  }
}

let ortModule: OrtApi | null = null

async function loadOrt(): Promise<OrtApi> {
  if (ortModule) return ortModule
  const ort = (await import('onnxruntime-web')) as unknown as OrtApi
  ort.env.wasm.wasmPaths = ORT_WASM
  ort.env.wasm.numThreads = 1
  ortModule = ort
  return ort
}

async function createService(opts: {
  ort: OrtApi
  size: OCRSize
  det: ArrayBuffer
  rec: ArrayBuffer
  charactersDictionary: string[]
  device: ResolvedDevice
}): Promise<{ ocr: PaddleOcrService; device: ResolvedDevice }> {
  const tryDevice = async (device: ResolvedDevice) => {
    const providers = device === 'webgpu' ? ['webgpu', 'wasm'] : ['wasm']
    const wrapped = {
      Tensor: opts.ort.Tensor,
      InferenceSession: {
        create: (modelBuffer: ArrayBuffer) =>
          opts.ort.InferenceSession.create(modelBuffer, { executionProviders: providers }),
      },
    }
    return PaddleOcrService.createInstance({
      ort: wrapped as never,
      modelPreset: PRESET[opts.size],
      detection: { modelBuffer: opts.det.slice(0) },
      recognition: {
        modelBuffer: opts.rec.slice(0),
        charactersDictionary: opts.charactersDictionary,
      },
    })
  }

  try {
    return { ocr: await tryDevice(opts.device), device: opts.device }
  } catch (err) {
    if (opts.device === 'wasm') throw err
    return { ocr: await tryDevice('wasm'), device: 'wasm' }
  }
}

async function fetchCached(
  url: string,
  file: string,
  onProgress?: ProgressCallback
): Promise<ArrayBuffer> {
  const cache = await caches.open(CACHE).catch(() => null)
  const hit = await cache?.match(url)
  if (hit) {
    onProgress?.({ status: 'progress', progress: 100, file })
    return hit.arrayBuffer()
  }

  const res = await fetch(url)
  if (!res.ok) throw new Error(`Could not load ${file} (${res.status})`)

  const total = Number(res.headers.get('content-length') ?? 0)
  if (!res.body || !total) {
    const buf = await res.arrayBuffer()
    await cache?.put(url, new Response(buf.slice(0)))
    return buf
  }

  const reader = res.body.getReader()
  const chunks: Uint8Array[] = []
  let received = 0
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    chunks.push(value)
    received += value.byteLength
    onProgress?.({ status: 'progress', progress: (received / total) * 100, file, loaded: received, total })
  }

  const out = new Uint8Array(received)
  let offset = 0
  for (const chunk of chunks) {
    out.set(chunk, offset)
    offset += chunk.byteLength
  }
  const buf = out.buffer
  await cache?.put(url, new Response(buf.slice(0)))
  return buf
}

async function fetchCachedText(url: string): Promise<string> {
  const cache = await caches.open(CACHE).catch(() => null)
  const hit = await cache?.match(url)
  if (hit) return hit.text()
  const res = await fetch(url)
  if (!res.ok) throw new Error(`Could not load dictionary (${res.status})`)
  const text = await res.text()
  await cache?.put(url, new Response(text))
  return text
}

function parseCharacterDict(yml: string): string[] {
  const start = yml.indexOf('character_dict:')
  if (start < 0) throw new Error('PP-OCRv6 dictionary missing character_dict')
  const chars: string[] = []
  for (const rawLine of yml.slice(start).split('\n').slice(1)) {
    const item = rawLine.match(/^\s*-\s+(.*)$/)
    if (!item) {
      if (chars.length && /^\S/.test(rawLine)) break
      continue
    }
    chars.push(unquoteYaml(item[1] ?? ''))
  }
  if (chars.length < 100) throw new Error('PP-OCRv6 dictionary looks empty')
  return chars
}

function unquoteYaml(value: string): string {
  const v = value.trim()
  if (v === "''" || v === '""') return ''
  if ((v.startsWith("'") && v.endsWith("'")) || (v.startsWith('"') && v.endsWith('"'))) {
    const inner = v.slice(1, -1)
    if (v.startsWith("'")) return inner.replace(/''/g, "'")
    return inner.replace(/\\"/g, '"')
  }
  return v
}

async function imageToPixels(image: OCRImageInput): Promise<{ width: number; height: number; data: Uint8Array }> {
  if (image instanceof ImageData) {
    return { width: image.width, height: image.height, data: new Uint8Array(image.data) }
  }

  let source: CanvasImageSource
  if (typeof image === 'string') {
    source = await loadHtmlImage(image)
  } else if (image instanceof HTMLImageElement) {
    source = image
  } else if (image instanceof HTMLCanvasElement) {
    source = image
  } else {
    source = await createImageBitmap(image)
  }

  const width = 'naturalWidth' in source ? source.naturalWidth || source.width : source.width
  const height = 'naturalHeight' in source ? source.naturalHeight || source.height : source.height
  const canvas = document.createElement('canvas')
  canvas.width = width
  canvas.height = height
  const ctx = canvas.getContext('2d')
  if (!ctx) throw new Error('Could not get 2d context')
  ctx.drawImage(source, 0, 0)
  const { data } = ctx.getImageData(0, 0, width, height)
  if (typeof ImageBitmap !== 'undefined' && source instanceof ImageBitmap) source.close()
  return { width, height, data: new Uint8Array(data) }
}

function loadHtmlImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image()
    img.crossOrigin = 'anonymous'
    img.onload = () => resolve(img)
    img.onerror = () => reject(new Error(`Could not load image: ${src}`))
    img.src = src
  })
}
