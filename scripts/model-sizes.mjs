#!/usr/bin/env node
// Every model runonweb downloads, with what the browser fetches on first use.
//
//   node scripts/model-sizes.mjs            # regenerate MODELS.md
//   node scripts/model-sizes.mjs --stdout   # print it instead
//
// Sizes come from the Hugging Face Hub file listings: nothing is downloaded.
// MODELS mirrors what each module in packages/runonweb/src loads (repo, ONNX sessions
// and dtype per backend). Update it when a module changes its defaults: the script
// warns about model ids it finds in the source but does not list.

import { readdir, readFile, writeFile } from 'node:fs/promises'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { PAIRS } from '../packages/runonweb/src/translate/registry.ts'

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const SRC = join(ROOT, 'packages/runonweb/src')
const OUT = join(ROOT, 'MODELS.md')
const HUB = 'https://huggingface.co'
const TRANSLATE_MIRROR = 'midudev/firefox-translations'

/** Transformers.js file suffix per dtype (`onnx/<session><suffix>.onnx`). */
const SUFFIX = { fp32: '', fp16: '_fp16', q8: '_quantized', int8: '_int8', uint8: '_uint8', q4: '_q4', q4f16: '_q4f16', bnb4: '_bnb4' }

/** Small files Transformers.js fetches next to the weights, when the repo has them. */
const CONFIG_FILES = [
  'config.json',
  'generation_config.json',
  'preprocessor_config.json',
  'processor_config.json',
  'tokenizer.json',
  'tokenizer_config.json',
  'chat_template.jinja',
]

/**
 * `sessions` + dtype: a Transformers.js repo. `files`: exact paths or RegExps fetched as-is.
 * A backend set to `null` is not used (the module falls back to the other one).
 */
const MODELS = [
  { module: 'stt', repo: 'onnx-community/whisper-tiny.en', sessions: ['encoder_model', 'decoder_model_merged'], webgpu: 'fp32', wasm: 'q8' },
  { module: 'remove-bg', repo: 'onnx-community/BEN2-ONNX', sessions: ['model'], webgpu: 'fp16', wasm: 'fp16' },
  {
    module: 'caption',
    repo: 'onnx-community/LFM2.5-VL-450M-ONNX',
    sessions: ['embed_tokens', 'vision_encoder', 'decoder_model_merged'],
    webgpu: { embed_tokens: 'fp16', vision_encoder: 'fp16', decoder_model_merged: 'q4f16' },
    wasm: { embed_tokens: 'fp16', vision_encoder: 'q8', decoder_model_merged: 'q4' },
  },
  {
    module: 'caption',
    variant: 'no shader-f16',
    repo: 'onnx-community/LFM2.5-VL-450M-ONNX',
    sessions: ['embed_tokens', 'vision_encoder', 'decoder_model_merged'],
    webgpu: { embed_tokens: 'fp32', vision_encoder: 'fp32', decoder_model_merged: 'q4' },
    wasm: null,
  },
  { module: 'depth', repo: 'onnx-community/depth-anything-v2-small', sessions: ['model'], webgpu: 'fp16', wasm: 'q8' },
  // RF-DETR collapses logits on WebGPU, so detect always runs it on WASM.
  { module: 'detect', repo: 'onnx-community/rfdetr_nano-ONNX', sessions: ['model'], webgpu: null, wasm: 'q8' },
  {
    module: 'ocr',
    variant: 'tiny',
    repo: 'PaddlePaddle/PP-OCRv6_tiny_det_onnx',
    extraRepo: 'PaddlePaddle/PP-OCRv6_tiny_rec_onnx',
    files: ['inference.onnx'],
    extraFiles: ['inference.onnx', 'inference.yml'],
  },
  {
    module: 'ocr',
    variant: 'small',
    repo: 'PaddlePaddle/PP-OCRv6_small_det_onnx',
    extraRepo: 'PaddlePaddle/PP-OCRv6_small_rec_onnx',
    files: ['inference.onnx'],
    extraFiles: ['inference.onnx', 'inference.yml'],
  },
  {
    module: 'ocr',
    variant: 'medium',
    repo: 'PaddlePaddle/PP-OCRv6_medium_det_onnx',
    extraRepo: 'PaddlePaddle/PP-OCRv6_medium_rec_onnx',
    files: ['inference.onnx'],
    extraFiles: ['inference.onnx', 'inference.yml'],
  },
  {
    module: 'classify',
    variant: 'small',
    repo: 'midudev/kev-0.8b-ONNX',
    files: ['kev.json', 'head.bin', 'tokenizer.json', 'tokenizer_config.json', /^onnx\/model\.onnx/],
  },
  {
    module: 'classify',
    variant: 'large',
    repo: 'midudev/kev-4b-ONNX',
    files: ['kev.json', 'head.bin', 'tokenizer.json', 'tokenizer_config.json', /^onnx\/model\.onnx/],
  },
  { module: 'embed', repo: 'Xenova/all-MiniLM-L6-v2', sessions: ['model'], webgpu: 'fp16', wasm: 'q8' },
  // Bergamot runs on WASM only. Every pair is listed under "Translation pairs".
  { module: 'translate', variant: 'per pair', repo: TRANSLATE_MIRROR, pairs: true },
  { module: 'translate', variant: 'model: m2m100', repo: 'Xenova/m2m100_418M', sessions: ['encoder_model', 'decoder_model_merged'], webgpu: null, wasm: 'q8' },
  { module: 'clean', repo: 'onnx-community/s1-mini-ONNX', sessions: ['model'], webgpu: 'q4f16', wasm: 'q4' },
  { module: 'clean', variant: 'WebGPU fallback', repo: 'onnx-community/s1-mini-ONNX', sessions: ['model'], webgpu: 'q4', wasm: null },
  { module: 'emoji', repo: 'midudev/text2emoji-tiny', sessions: ['encoder_model', 'decoder_model_merged'], webgpu: null, wasm: 'q8' },
  {
    module: 'tts',
    variant: 'small',
    repo: 'onnx-community/Kokoro-82M-v1.0-ONNX',
    sessions: ['model'],
    webgpu: 'fp32',
    wasm: 'q8',
    extraFiles: ['voices/af_heart.bin'],
  },
  {
    module: 'tts',
    variant: 'multi',
    repo: 'onnx-community/Supertonic-TTS-2-ONNX',
    sessions: ['text_encoder', 'latent_denoiser', 'voice_decoder'],
    webgpu: 'fp32',
    wasm: 'fp32',
    extraFiles: ['voices/F1.bin'],
  },
  {
    module: 'tts',
    variant: 'tiny',
    repo: 'KittenML/kitten-tts-nano-0.8-int8',
    files: ['kitten_tts_nano_v0_8.onnx', 'voices.npz'],
    webgpu: null,
  },
  ...['binary', 'ternary'].map((variant) => ({
    module: 'image',
    variant,
    repo: variant === 'binary' ? 'prism-ml/bonsai-image-binary-4B-mlx-1bit' : 'prism-ml/bonsai-image-ternary-4B-mlx-2bit',
    files: [
      /^text_encoder-mlx-4bit\/(config\.json|model\.safetensors)$/,
      /^(transformer-packed-mflux|vae|scheduler)\//,
      /^tokenizer\/tokenizer(_config)?\.json$/,
    ],
    wasm: null,
  })),
]

/** License of the weights in each repo, as Markdown. The ones with extra terms link to their text. */
const LICENSES = {
  'onnx-community/whisper-tiny.en': 'Apache-2.0',
  'onnx-community/BEN2-ONNX': 'MIT',
  'onnx-community/LFM2.5-VL-450M-ONNX': '[LFM Open License v1.0](https://huggingface.co/LiquidAI/LFM2.5-VL-450M/blob/main/LICENSE)',
  'onnx-community/depth-anything-v2-small': 'Apache-2.0',
  'onnx-community/rfdetr_nano-ONNX': 'Apache-2.0',
  'PaddlePaddle/PP-OCRv6_tiny_det_onnx': 'Apache-2.0',
  'PaddlePaddle/PP-OCRv6_small_det_onnx': 'Apache-2.0',
  'PaddlePaddle/PP-OCRv6_medium_det_onnx': 'Apache-2.0',
  'midudev/kev-0.8b-ONNX': 'Apache-2.0',
  'midudev/kev-4b-ONNX': 'Apache-2.0',
  'Xenova/all-MiniLM-L6-v2': 'Apache-2.0',
  [TRANSLATE_MIRROR]: 'MPL-2.0',
  'Xenova/m2m100_418M': 'MIT',
  'onnx-community/s1-mini-ONNX': '[Apache-2.0 + naming term](https://huggingface.co/superwhisper/s1-mini/blob/main/LICENSE)',
  'midudev/text2emoji-tiny': 'MIT',
  'onnx-community/Kokoro-82M-v1.0-ONNX': 'Apache-2.0',
  'onnx-community/Supertonic-TTS-2-ONNX': '[OpenRAIL-M](https://huggingface.co/Supertone/supertonic-2/blob/main/LICENSE)',
  'KittenML/kitten-tts-nano-0.8-int8': 'Apache-2.0',
  'prism-ml/bonsai-image-binary-4B-mlx-1bit': 'Apache-2.0',
  'prism-ml/bonsai-image-ternary-4B-mlx-2bit': 'Apache-2.0',
}

const NOTES = [
  '`tts` small with Spanish or French also fetches the eSpeak-NG WASM (`espeak-ng@1.0.2` from jsDelivr, ~18 MB) on first use.',
  '`tts` sizes include one voice (`af_heart` for Kokoro, `F1` for Supertonic). Each extra voice is another small download.',
  '`translate` also loads the Bergamot runtime (`@mkljczk/bergamot-translator` from jsDelivr), and pairs without a direct model pivot through English (two pairs).',
  'ONNX Runtime Web and the other JS/WASM runtimes are not counted.',
]

/** Every file in a repo with its size, following the Hub's pagination. */
async function listRepo(repo) {
  const files = new Map()
  let url = `${HUB}/api/models/${repo}/tree/main?recursive=true`
  while (url) {
    const res = await fetch(url)
    if (!res.ok) throw new Error(`${repo}: ${res.status} ${res.statusText}`)
    for (const entry of await res.json()) {
      if (entry.type === 'file') files.set(entry.path, entry.lfs?.size ?? entry.size)
    }
    url = /<([^>]+)>;\s*rel="next"/.exec(res.headers.get('link') ?? '')?.[1]
  }
  return files
}

const repoCache = new Map()
function repoFiles(repo) {
  if (!repoCache.has(repo)) repoCache.set(repo, listRepo(repo))
  return repoCache.get(repo)
}

function pick(files, patterns, repo) {
  const picked = []
  for (const pattern of patterns) {
    const hits = [...files.keys()].filter((p) => (typeof pattern === 'string' ? p === pattern : pattern.test(p)))
    if (!hits.length) throw new Error(`${repo}: nothing matches ${pattern}`)
    picked.push(...hits)
  }
  return picked
}

/** Weights for a Transformers.js dtype: the session graphs plus their external data. */
function onnxFiles(files, sessions, dtype, repo) {
  return sessions.flatMap((session) => {
    const name = `onnx/${session}${SUFFIX[typeof dtype === 'string' ? dtype : dtype[session]]}.onnx`
    return pick(files, [name], repo).concat([...files.keys()].filter((p) => p.startsWith(`${name}_data`)))
  })
}

function dtypeLabel(dtype) {
  if (typeof dtype === 'string') return dtype
  return [...new Set(Object.values(dtype))].join(' + ')
}

/** `{ bytes, dtype }` per backend, or `null` when the module never uses it. */
async function measure(model) {
  const files = await repoFiles(model.repo)
  const extra = model.extraRepo ? await repoFiles(model.extraRepo) : files
  const sum = (from, paths) => paths.reduce((total, p) => total + from.get(p), 0)
  const extraBytes = model.extraFiles ? sum(extra, pick(extra, model.extraFiles, model.extraRepo ?? model.repo)) : 0

  const backend = (dtype) => {
    if (dtype === null) return null
    if (model.files) return { bytes: sum(files, pick(files, model.files, model.repo)) + extraBytes }
    const configs = CONFIG_FILES.filter((p) => files.has(p))
    return { bytes: sum(files, [...onnxFiles(files, model.sessions, dtype, model.repo), ...configs]) + extraBytes, dtype: dtypeLabel(dtype) }
  }
  return { webgpu: backend(model.webgpu), wasm: backend(model.wasm) }
}

/** Firefox Translations pairs as mirrored on the Hub, grouped by `<from>-<to>/`. */
async function measurePairs() {
  const files = await repoFiles(TRANSLATE_MIRROR)
  const bytes = new Map()
  for (const [path, size] of files) {
    const pair = path.split('/')[0]
    if (PAIRS[pair]) bytes.set(pair, (bytes.get(pair) ?? 0) + size)
  }
  const missing = Object.keys(PAIRS).filter((pair) => !bytes.has(pair))
  if (missing.length) console.warn(`warn: pairs missing from ${TRANSLATE_MIRROR}: ${missing.join(', ')}`)
  return [...bytes].map(([pair, size]) => ({ pair, bytes: size, architecture: PAIRS[pair].architecture }))
}

/** Model ids the source references that MODELS does not cover. */
async function unlistedIds() {
  const listed = new Set(MODELS.flatMap((m) => [m.repo, m.extraRepo]))
  const found = new Set()
  const patterns = [
    /(?:DEFAULT_MODEL(?:_ID)?|MULTILINGUAL_MODEL)\s*=\s*['`]([\w.-]+\/[\w.-]+)['`]/g,
    /huggingface\.co\/([\w.-]+\/[\w.-]+)\/resolve\//g,
    /\b(?:model|binary|ternary):\s*'([\w.-]+\/[\w.-]+)'/g,
  ]
  for (const entry of await readdir(SRC, { recursive: true, withFileTypes: true })) {
    if (!entry.isFile() || !/\.(ts|js)$/.test(entry.name)) continue
    const text = await readFile(join(entry.parentPath, entry.name), 'utf8')
    for (const re of patterns) for (const m of text.matchAll(re)) found.add(m[1])
  }
  return [...found].filter((id) => !listed.has(id) && !id.startsWith('${'))
}

function formatBytes(bytes) {
  if (bytes >= 1e9) return `${(bytes / 1e9).toFixed(2)} GB`
  if (bytes >= 1e6) return `${(bytes / 1e6).toFixed(1)} MB`
  return `${(bytes / 1e3).toFixed(0)} KB`
}

function cell(backend) {
  if (!backend) return '—'
  if (backend.range) return backend.range.map(formatBytes).join(' – ')
  return backend.dtype ? `${formatBytes(backend.bytes)} (${backend.dtype})` : formatBytes(backend.bytes)
}

function render(rows, pairs) {
  const lines = [
    '# Models',
    '',
    'Every model runonweb downloads, with the bytes the browser fetches on first use (weights, configs and tokenizers). Everything is cached after that.',
    '',
    `Generated by \`pnpm models:sizes\` (\`scripts/model-sizes.mjs\`) from the Hugging Face Hub on ${new Date().toISOString().slice(0, 10)}. Do not edit by hand. MB and GB are decimal, as on the Hub.`,
    '',
    '| Module | Variant | Model | License | WebGPU | WASM |',
    '| --- | --- | --- | --- | --- | --- |',
    ...rows.map(({ model, size }) => {
      const repos = [model.repo, model.extraRepo].filter(Boolean).map((id) => `[${id}](${HUB}/${id})`)
      return `| \`${model.module}\` | ${model.variant ?? 'default'} | ${repos.join('<br>')} | ${LICENSES[model.repo]} | ${cell(size.webgpu)} | ${cell(size.wasm)} |`
    }),
    '',
    'A dash means the model never runs on that backend: the module uses the other one instead, except `image`, which needs WebGPU.',
    '',
    ...NOTES.map((note) => `- ${note}`),
    '',
    '## Licenses',
    '',
    'The runonweb code is MIT ([LICENSE](./LICENSE)). That license does not cover the models: the package ships no weights, the browser downloads them from the Hugging Face Hub on first use, and each one keeps the license its authors chose (the License column above). Check it before you ship. Three come with extra terms:',
    '',
    '- `caption` (LFM Open License v1.0): free for individuals and companies under USD 10M annual revenue. Above that, Liquid AI requires a commercial license.',
    '- `tts` multi (OpenRAIL-M): open weights with use restrictions, such as no impersonation, deception or illegal use.',
    '- `clean` (Apache-2.0 plus one term): any product that uses the model must keep calling it "S1-mini" by "Superwhisper".',
    '',
    '## Translation pairs',
    '',
    `${pairs.length} Firefox Translations pairs (Mozilla, MPL-2.0), mirrored from Mozilla's registry by \`scripts/translate-models.mjs\`. Sizes include the model, the shortlist and the vocabularies.`,
    '',
    '<details>',
    '<summary>Every pair</summary>',
    '',
    '| Pair | Architecture | Size |',
    '| --- | --- | --- |',
    ...pairs.map((p) => `| \`${p.pair}\` | ${p.architecture} | ${formatBytes(p.bytes)} |`),
    '',
    '</details>',
    '',
  ]
  return lines.join('\n')
}

const unlicensed = MODELS.filter((m) => !LICENSES[m.repo]).map((m) => m.repo)
if (unlicensed.length) throw new Error(`no license in LICENSES for: ${unlicensed.join(', ')}`)

const [pairs, unlisted] = await Promise.all([measurePairs(), unlistedIds()])
const pairBytes = pairs.map((p) => p.bytes)
const pairRange = { webgpu: null, wasm: { range: [Math.min(...pairBytes), Math.max(...pairBytes)] } }
const rows = await Promise.all(
  MODELS.map(async (model) => ({ model, size: model.pairs ? pairRange : await measure(model) }))
)

for (const id of unlisted) console.warn(`warn: ${id} is referenced in packages/runonweb/src but not listed in MODELS`)

const markdown = render(rows, pairs)
if (process.argv.includes('--stdout')) {
  process.stdout.write(markdown)
} else {
  await writeFile(OUT, markdown)
  console.log(`Wrote ${OUT} (${rows.length} models, ${pairs.length} translation pairs)`)
}
