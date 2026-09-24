import type { Device, ProgressCallback, ProgressInfo, ResolvedDevice } from '../core/index.ts'
import { resolveDevice } from '../core/device.ts'
import { CACHE_NAME } from '../core/cache.ts'
import {
  toAnswers,
  toRecord,
  validateRequest,
  withDateFacts,
  type Answers,
  type ClassifyRequest,
  type DecisionRecord,
  type Question,
} from './request.ts'

export * from './request.ts'

/**
 * Default weights: Kev-0.8B (Jared Palmer, Apache-2.0), a Jev-style decision model on Qwen3.5-0.8B-Base,
 * with its LoRA merged in and exported to ONNX by runonweb (`training/kev-onnx`). int4 weights with the
 * Gated DeltaNet layers in int8, fp32 activations: ~735 MB, the same file on WebGPU and WASM.
 */
export const DEFAULT_MODEL = 'runonweb/kev-0.8b-ONNX'

const ORT_WASM = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/'

export type ClassifyOptions = {
  /** Hugging Face repo id, or folder name under `modelPath`. Defaults to Kev-0.8B. */
  model?: string
  /**
   * Base URL the weights are served from, e.g. `/models/`: files are read from `<modelPath>/<model>/`.
   * Omit to download `model` from the Hugging Face Hub.
   */
  modelPath?: string
  device?: Device
  /**
   * Append the day count between every pair of absolute dates found in the state
   * ("June 26, 2026 is 8 days before July 4, 2026"). Kev can't subtract dates; it can use a stated count.
   * Same as Kev's `KEV_DATE_FACTS=1`. Default false.
   */
  dateFacts?: boolean
  /**
   * Softmax temperature. Defaults to the one fitted for the checkpoint (~2.35 for Kev-0.8B), which
   * calibrates probabilities without changing any answer. `1` gives the raw logits.
   */
  temperature?: number
  onProgress?: ProgressCallback
}

/** Per-call overrides of the constructor options. */
export type ClassifyRunOptions = Pick<ClassifyOptions, 'dateFacts' | 'temperature'>

export type ClassifyResult<Q extends Record<string, Question> = Record<string, Question>> = {
  model: string
  answers: Answers<Q>
  usage: {
    /** State tokens (counted once) plus every question's tokens. */
    inputTokens: number
  }
  latencyMs: number
}

/** Layout of `kev.json`, written by `training/kev-onnx/assemble.py`. */
type KevConfig = {
  name: string
  temperature: number
  hidden_size: number
  head_dim: number
  head_file: string
  onnx: { model: string; data: string }
  tokens: { state: number; question: number; option: number; option_end: number; decide: number }
  max_state_tokens: number
  max_row_tokens: number
  cache: {
    layer_types: ('linear_attention' | 'full_attention')[]
    num_key_value_heads: number
    attention_head_dim: number
    conv_dim: number
    conv_kernel: number
    linear_heads: number
    linear_key_dim: number
    linear_value_dim: number
  }
}

type Ort = typeof import('onnxruntime-web/webgpu')
type OrtTensor = import('onnxruntime-web').Tensor
type OrtSession = import('onnxruntime-web').InferenceSession
type Tokenizer = { encode: (text: string, opts?: { add_special_tokens?: boolean }) => number[] }

type Encoded = { state: number[]; rows: { ids: number[]; decide: number; options: number[] }[] }

type Runtime = {
  ort: Ort
  session: OrtSession
  tokenizer: Tokenizer
  config: KevConfig
  head: { qW: Float32Array; qB: Float32Array; kW: Float32Array; kB: Float32Array }
  /** Cache inputs for an empty prefix (zero conv/recurrent states, zero-length KV). */
  empty: Record<string, OrtTensor>
  /** present.* output name → past.* input name. */
  pastFor: Map<string, string>
  device: ResolvedDevice
}

/**
 * Typed questions about a text, answered with calibrated probabilities: yes/no (`noul`),
 * multiple choice (`choice`) and ratings (`score`), all in one request. No text is generated.
 *
 * Runs Kev (github.com/jaredpalmer/kev) in the browser. Requests and answers follow TypeSafe's
 * System One API (`POST /v1/systemone`), so the same JSON works against a Kev or Jev server.
 *
 * @example
 * ```ts
 * import { Classifier } from 'runonweb/classify'
 *
 * const classifier = new Classifier()
 * await classifier.load()
 *
 * const { answers } = await classifier.classify({
 *   state: 'Shoes arrived two weeks late and in the wrong size.',
 *   questions: {
 *     department: { type: 'choice', instructions: 'Which team should handle this?',
 *                   criteria: { returns: null, shipping: null, billing: null } },
 *     escalate: { type: 'noul', instructions: 'Does this need urgent human attention?' },
 *     frustration: { type: 'score', instructions: 'How frustrated is the customer?',
 *                    criteria: ['Calm', 'Frustrated', 'Very angry'] },
 *   },
 * })
 * answers.department.choice // 'returns'
 * answers.escalate.noul     // p(yes)
 * ```
 */
export class Classifier {
  #model: string
  #modelPath?: string
  #device: Device
  #dateFacts: boolean
  #temperature?: number
  #onProgress?: ProgressCallback
  #rt: Runtime | null = null
  #loading: Promise<void> | null = null
  #queue: Promise<unknown> = Promise.resolve()

  constructor(options: ClassifyOptions = {}) {
    this.#model = options.model ?? DEFAULT_MODEL
    this.#modelPath = options.modelPath
    this.#device = options.device ?? 'auto'
    this.#dateFacts = options.dateFacts ?? false
    this.#temperature = options.temperature
    this.#onProgress = options.onProgress
  }

  get device(): ResolvedDevice | null {
    return this.#rt?.device ?? null
  }

  /** Calibration temperature in use (the checkpoint's unless overridden). */
  get temperature(): number | null {
    return this.#temperature ?? this.#rt?.config.temperature ?? null
  }

  async load(): Promise<void> {
    if (this.#rt) return
    if (this.#loading) return this.#loading
    this.#loading = (async () => {
      this.#rt = await loadRuntime(this.#baseUrl(), await resolveDevice(this.#device), this.#onProgress)
    })()
    try {
      await this.#loading
    } finally {
      this.#loading = null
    }
  }

  /**
   * Answer every question about `state`. Each question only sees the state and itself;
   * the state is read once and reused for every question.
   */
  async classify<const Q extends Record<string, Question>>(
    request: ClassifyRequest<Q>,
    options: ClassifyRunOptions = {}
  ): Promise<ClassifyResult<Q>> {
    validateRequest(request)
    await this.load()
    const run = this.#queue.then(() => this.#classify(request, options))
    this.#queue = run.catch(() => undefined)
    return run
  }

  async #classify<Q extends Record<string, Question>>(
    request: ClassifyRequest<Q>,
    options: ClassifyRunOptions
  ): Promise<ClassifyResult<Q>> {
    const rt = this.#rt
    if (!rt) throw new Error('Classifier was disposed')
    const t0 = performance.now()
    this.#onProgress?.({ status: 'classifying' })

    const dates = options.dateFacts ?? this.#dateFacts
    const req = dates ? { ...request, state: withDateFacts(request.state) } : request
    const { record, meta } = toRecord(req)
    const enc = encode(rt, record)
    const probs = await scoreRows(rt, enc, options.temperature ?? this.#temperature ?? rt.config.temperature)

    this.#onProgress?.({ status: 'done' })
    return {
      model: rt.config.name,
      answers: toAnswers(probs, meta) as Answers<Q>,
      usage: { inputTokens: enc.state.length + enc.rows.reduce((n, r) => n + r.ids.length, 0) },
      latencyMs: Math.round(performance.now() - t0),
    }
  }

  dispose(): void {
    const rt = this.#rt
    this.#rt = null
    void this.#queue.finally(() => rt?.session.release())
  }

  #baseUrl(): string {
    if (this.#modelPath) {
      const base = new URL(this.#modelPath.replace(/\/?$/, '/'), globalThis.location?.href).href
      return `${base}${this.#model}/`
    }
    return `https://huggingface.co/${this.#model}/resolve/main/`
  }
}

/** One-shot helper: load, classify, dispose. Keep a `Classifier` around for more than one call. */
export async function classify<const Q extends Record<string, Question>>(
  request: ClassifyRequest<Q>,
  options?: ClassifyOptions
): Promise<ClassifyResult<Q>> {
  const classifier = new Classifier(options)
  try {
    return await classifier.classify(request)
  } finally {
    classifier.dispose()
  }
}

export type { ProgressInfo, Device }

// ---------------------------------------------------------------------------------------------------------------
// Encoding (kev/model.py `encode`, row form)

/** Caller text can never produce delimiter tokens: `<|name|>` is rewritten to `<¦name¦>` before tokenizing. */
function userTokens(tok: Tokenizer, text: string): number[] {
  return tok.encode(text.replace(/<\|([A-Za-z0-9_]+)\|>/g, '<¦$1¦>'), { add_special_tokens: false })
}

/**
 * `[<state> …state…]`, then per question one row that continues the state:
 * `<q> instructions <opt> option </opt> … <decide>`. Delimiters are rarely used Qwen special tokens.
 */
function encode(rt: Runtime, record: DecisionRecord): Encoded {
  const { tokens: T, max_state_tokens, max_row_tokens } = rt.config
  const state = [T.state, ...userTokens(rt.tokenizer, record.state)]
  if (state.length > max_state_tokens) {
    throw new Error(`State is ${state.length} tokens; the limit is ${max_state_tokens} (Kev was trained on up to 384)`)
  }
  const rows = record.questions.map((q) => {
    const ids = [T.question, ...userTokens(rt.tokenizer, q.instr)]
    const options: number[] = []
    for (const o of q.options) {
      ids.push(T.option, ...userTokens(rt.tokenizer, o), T.option_end)
      options.push(ids.length - 1)
    }
    ids.push(T.decide)
    if (state.length + ids.length > max_row_tokens) {
      throw new Error(`State plus one question is ${state.length + ids.length} tokens; the limit is ${max_row_tokens}`)
    }
    return { ids, decide: ids.length - 1, options }
  })
  return { state, rows }
}

// ---------------------------------------------------------------------------------------------------------------
// Inference

/**
 * onnxruntime-web's WebGPU backend keeps a single global OrtRun slot; overlapping runs from any session
 * fail with "Session already started". Every run in this module goes through one queue.
 */
let ortTurn: Promise<void> = Promise.resolve()

function takeTurn<T>(fn: () => Promise<T>): Promise<T> {
  const run = ortTurn.then(fn, fn)
  ortTurn = run.then(
    () => undefined,
    () => undefined
  )
  return run
}

function int64(values: ArrayLike<number>, dims: number[], ort: Ort): OrtTensor {
  const data = new BigInt64Array(values.length)
  for (let i = 0; i < values.length; i++) data[i] = BigInt(values[i]!)
  return new ort.Tensor('int64', data, dims)
}

/** input_ids, attention_mask and the 3-axis (M-RoPE) position ids for `ids` continuing `start` cached tokens. */
function tokenFeeds(ort: Ort, ids: number[], start: number): Record<string, OrtTensor> {
  const L = ids.length
  const pos = new Array<number>(3 * L)
  for (let a = 0; a < 3; a++) for (let i = 0; i < L; i++) pos[a * L + i] = start + i
  return {
    input_ids: int64(ids, [1, L], ort),
    attention_mask: int64(new Array<number>(start + L).fill(1), [1, start + L], ort),
    position_ids: int64(pos, [3, 1, L], ort),
  }
}

/** Pointer head: `<decide>` vs each `</opt>`, `(k · q) / sqrt(d) / T`, softmax. */
function readout(rt: Runtime, hidden: Float32Array, row: Encoded['rows'][number], temperature: number): Float64Array {
  const H = rt.config.hidden_size
  const D = rt.config.head_dim
  const project = (W: Float32Array, b: Float32Array, at: number) => {
    const out = new Float64Array(D)
    const off = at * H
    for (let i = 0; i < D; i++) {
      let s = b[i]!
      const w = i * H
      for (let j = 0; j < H; j++) s += W[w + j]! * hidden[off + j]!
      out[i] = s
    }
    return out
  }
  const q = project(rt.head.qW, rt.head.qB, row.decide)
  const scale = 1 / Math.sqrt(D) / temperature
  const z = row.options.map((at) => {
    const k = project(rt.head.kW, rt.head.kB, at)
    let dot = 0
    for (let i = 0; i < D; i++) dot += k[i]! * q[i]!
    return dot * scale
  })
  const max = Math.max(...z)
  const e = z.map((v) => Math.exp(v - max))
  const sum = e.reduce((a, b) => a + b, 0)
  return Float64Array.from(e, (v) => v / sum)
}

/** State pass once, then each question's row on the state's cache (exact: rows are independent). */
async function scoreRows(rt: Runtime, enc: Encoded, temperature: number): Promise<Float64Array[]> {
  const { ort, session } = rt
  const Ls = enc.state.length
  const owned: OrtTensor[] = []
  const dropOutputs = (out: Record<string, OrtTensor>, keep?: Set<string>) => {
    for (const [name, t] of Object.entries(out)) if (name !== 'hidden_states' && !keep?.has(name)) disposeTensor(t)
  }
  try {
    const stateOut = await takeTurn(() => session.run({ ...tokenFeeds(ort, enc.state, 0), ...rt.empty }))
    disposeTensor(stateOut.hidden_states)
    const past: Record<string, OrtTensor> = {}
    for (const [present, pastName] of rt.pastFor) {
      const t = stateOut[present]
      if (!t) throw new Error(`Model output ${present} is missing`)
      past[pastName] = t
      owned.push(t)
    }

    const probs: Float64Array[] = []
    for (const row of enc.rows) {
      const out = await takeTurn(() => session.run({ ...tokenFeeds(ort, row.ids, Ls), ...past }))
      try {
        const hidden = (await out.hidden_states!.getData()) as Float32Array
        probs.push(readout(rt, hidden, row, temperature))
      } finally {
        disposeTensor(out.hidden_states)
        dropOutputs(out)
      }
    }
    return probs
  } finally {
    for (const t of owned) disposeTensor(t)
  }
}

function disposeTensor(t: OrtTensor | undefined): void {
  // GPU-resident outputs hold a GPUBuffer until disposed; CPU tensors ignore this.
  if (t && t.location === 'gpu-buffer') t.dispose()
}

// ---------------------------------------------------------------------------------------------------------------
// Loading

async function loadRuntime(base: string, device: ResolvedDevice, onProgress?: ProgressCallback): Promise<Runtime> {
  onProgress?.({ status: 'loading', progress: 0 })
  const config = (await (await fetchCached(`${base}kev.json`)).json()) as KevConfig

  const files = [
    { key: 'model', url: `${base}${config.onnx.model}`, file: config.onnx.model },
    { key: 'data', url: `${base}${config.onnx.data}`, file: config.onnx.data },
    { key: 'head', url: `${base}${config.head_file}`, file: config.head_file },
    { key: 'tokenizer', url: `${base}tokenizer.json`, file: 'tokenizer.json' },
    { key: 'tokenizerConfig', url: `${base}tokenizer_config.json`, file: 'tokenizer_config.json' },
  ] as const
  const progress = new ProgressTotals(onProgress)
  const [model, data, head, tokenizerJson, tokenizerConfig] = await Promise.all(
    files.map((f) => fetchBytes(f.url, f.file, progress))
  )

  const { PreTrainedTokenizer } = await import('@huggingface/transformers')
  const text = new TextDecoder()
  const tokenizer = new PreTrainedTokenizer(
    JSON.parse(text.decode(tokenizerJson)),
    JSON.parse(text.decode(tokenizerConfig))
  ) as unknown as Tokenizer

  const D = config.head_dim
  const H = config.hidden_size
  const floats = new Float32Array(head!.buffer, head!.byteOffset, head!.byteLength / 4)
  if (floats.length !== 2 * (D * H + D)) throw new Error('head.bin does not match kev.json')
  const headW = {
    qW: floats.subarray(0, D * H),
    qB: floats.subarray(D * H, D * H + D),
    kW: floats.subarray(D * H + D, 2 * D * H + D),
    kB: floats.subarray(2 * D * H + D),
  }

  onProgress?.({ status: 'initializing' })
  const ort = await loadOrt()
  const { cache } = config
  const empty: Record<string, OrtTensor> = {}
  const pastFor = new Map<string, string>()
  cache.layer_types.forEach((type, i) => {
    if (type === 'full_attention') {
      for (const kv of ['key', 'value']) {
        const dims = [1, cache.num_key_value_heads, 0, cache.attention_head_dim]
        empty[`past_key_values.${i}.${kv}`] = new ort.Tensor('float32', new Float32Array(0), dims)
        pastFor.set(`present.${i}.${kv}`, `past_key_values.${i}.${kv}`)
      }
    } else {
      const conv = [1, cache.conv_dim, cache.conv_kernel]
      const rec = [1, cache.linear_heads, cache.linear_key_dim, cache.linear_value_dim]
      empty[`past.${i}.conv`] = new ort.Tensor('float32', new Float32Array(conv.reduce((a, b) => a * b)), conv)
      empty[`past.${i}.recurrent`] = new ort.Tensor('float32', new Float32Array(rec.reduce((a, b) => a * b)), rec)
      pastFor.set(`present.${i}.conv`, `past.${i}.conv`)
      pastFor.set(`present.${i}.recurrent`, `past.${i}.recurrent`)
    }
  })

  const dataPath = config.onnx.data.split('/').pop()!
  const create = (ep: ResolvedDevice) =>
    ort.InferenceSession.create(model!, {
      executionProviders: [ep],
      externalData: [{ path: dataPath, data: data! }],
      graphOptimizationLevel: 'all',
      // Errors only: node-placement warnings ("Some nodes were not assigned…") are expected for shape ops.
      logSeverityLevel: 3,
      // Keep the state's cache on the GPU between the state pass and the question rows.
      ...(ep === 'webgpu'
        ? { preferredOutputLocation: Object.fromEntries([...pastFor.keys()].map((n) => [n, 'gpu-buffer' as const])) }
        : {}),
    })

  let session: OrtSession
  let resolved = device
  try {
    session = await create(device)
  } catch (err) {
    if (device === 'wasm') throw err
    console.warn('[runonweb/classify] WebGPU session failed, falling back to WASM:', err)
    resolved = 'wasm'
    session = await create('wasm')
  }

  onProgress?.({ status: 'ready', progress: 100 })
  return { ort, session, tokenizer, config, head: headW, empty, pastFor, device: resolved }
}

let ortModule: Ort | null = null

async function loadOrt(): Promise<Ort> {
  if (ortModule) return ortModule
  // The native WebGPU EP build: it runs 8-bit MatMulNBits (the JSEP build in the default entry only does 2/4-bit).
  const ort = await import('onnxruntime-web/webgpu')
  if (typeof document !== 'undefined') ort.env.wasm.wasmPaths = ORT_WASM
  ortModule = ort
  return ort
}

class ProgressTotals {
  #loaded = new Map<string, number>()
  #total = new Map<string, number>()
  #cb?: ProgressCallback

  constructor(cb?: ProgressCallback) {
    this.#cb = cb
  }

  update(file: string, loaded: number, total: number) {
    this.#loaded.set(file, loaded)
    this.#total.set(file, total)
    if (!this.#cb) return
    this.#cb({ status: 'progress', file, loaded, total, progress: total ? (loaded / total) * 100 : undefined })
    let l = 0
    let t = 0
    for (const v of this.#loaded.values()) l += v
    for (const v of this.#total.values()) t += v
    if (t) this.#cb({ status: 'progress_total', loaded: l, total: t, progress: (l / t) * 100 })
  }
}

async function openCache(): Promise<Cache | null> {
  if (typeof caches === 'undefined') return null
  return caches.open(CACHE_NAME).catch(() => null)
}

async function fetchCached(url: string): Promise<Response> {
  const cache = await openCache()
  const hit = await cache?.match(url)
  if (hit) return hit
  const res = await fetch(url)
  if (!res.ok) throw new Error(`Could not load ${url} (${res.status})`)
  await cache?.put(url, res.clone()).catch(() => undefined)
  return res
}

/** Download into one buffer with progress, served from and stored in the shared model cache. */
async function fetchBytes(url: string, file: string, progress: ProgressTotals): Promise<Uint8Array> {
  const cache = await openCache()
  const hit = await cache?.match(url)
  if (hit) {
    const bytes = new Uint8Array(await hit.arrayBuffer())
    progress.update(file, bytes.byteLength, bytes.byteLength)
    return bytes
  }

  const res = await fetch(url)
  if (!res.ok) throw new Error(`Could not load ${file} (${res.status})`)
  const total = Number(res.headers.get('content-length') ?? 0)
  let bytes: Uint8Array
  if (!res.body) {
    bytes = new Uint8Array(await res.arrayBuffer())
  } else {
    const reader = res.body.getReader()
    let buf = new Uint8Array(total || 1 << 20)
    let received = 0
    for (;;) {
      const { done, value } = await reader.read()
      if (done) break
      if (received + value.byteLength > buf.byteLength) {
        const grown = new Uint8Array(Math.max(buf.byteLength * 2, received + value.byteLength))
        grown.set(buf.subarray(0, received))
        buf = grown
      }
      buf.set(value, received)
      received += value.byteLength
      progress.update(file, received, total || received)
    }
    bytes = received === buf.byteLength ? buf : buf.slice(0, received)
  }
  progress.update(file, bytes.byteLength, bytes.byteLength)
  await cache?.put(url, new Response(bytes as BodyInit, { headers: { 'content-length': String(bytes.byteLength) } })).catch(() => undefined)
  return bytes
}
