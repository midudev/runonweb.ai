# runonweb

Free browser ML modules. Models run **locally in the browser** (WebGPU preferred, WASM fallback). No API keys, no servers. MIT.

```bash
pnpm add runonweb
npm install runonweb
bun add runonweb
yarn add runonweb
```

Every module follows the same shape:

```ts
const m = new Module({ onProgress })
await m.load()          // downloads + caches weights, safe to call twice
await m.<task>(input)   // the actual work
m.dispose()             // free memory
```

Each module also exports a one-shot function (`transcribe`, `removeBackground`, `caption`, …) that loads, runs and disposes.

## Modules

### Speech-to-text — `runonweb/stt`

```ts
import { SpeechToText } from 'runonweb/stt'

const stt = new SpeechToText()
await stt.load()
const { text } = await stt.transcribe(audioBlob, {
  onPartial: (t) => console.log(t), // words as Whisper emits them
})
```

Base: Whisper tiny.en (OpenAI, Apache-2.0). Pass `model` + `language` for multilingual Whisper variants.

### Background removal — `runonweb/remove-bg`

```ts
import { RemoveBackground } from 'runonweb/remove-bg'

const remover = new RemoveBackground()
const png = await remover.remove(imageFile) // PNG Blob with alpha
```

Base: BEN2 (Prama LLC, MIT, ~219 MB fp16). Hair, objects and hard edges. WASM only until onnxruntime-web ships the LayerNorm shader fix (onnxruntime#32629).

### Image captioning — `runonweb/caption`

```ts
import { ImageCaptioner } from 'runonweb/caption'

const captioner = new ImageCaptioner({ language: 'es', detail: 'short' })
const { text } = await captioner.caption(imageFile)
```

Base: LFM2.5-VL-450M (Liquid AI, ~316 MB on WebGPU). Captions in `en`, `es`, `pt`, `fr`, `de`, `ar`, `zh`, `ja`, `ko`; `detail` is `short` | `detailed` | `more`; `onPartial` streams.

> **License:** LFM Open License v1.0 — free for individuals and companies under USD 10M annual revenue; larger companies need a commercial license from Liquid AI. Not OSI-approved. See <https://huggingface.co/LiquidAI/LFM2.5-VL-450M/blob/main/LICENSE>.

### Depth estimation — `runonweb/depth`

```ts
import { DepthEstimator } from 'runonweb/depth'

const { depth, width, height } = await new DepthEstimator().estimate(imageFile)
// depth: grayscale PNG Blob, brighter = closer
```

Base: Depth Anything V2 Small (Apache-2.0).

### Object detection — `runonweb/detect`

```ts
import { ObjectDetector } from 'runonweb/detect'

const objects = await new ObjectDetector({ threshold: 0.5 }).detect(imageFile)
// [{ label: 'cat', score: 0.98, box: { xmin, ymin, xmax, ymax } }]
```

Base: DETR ResNet-50 (Meta, Apache-2.0). Boxes in pixel coordinates.

### OCR — `runonweb/ocr`

```ts
import { OCR } from 'runonweb/ocr'

const ocr = new OCR() // size: 'small' — best size/quality
// const ocr = new OCR({ size: 'tiny' })   // ~6 MB
// const ocr = new OCR({ size: 'medium' }) // ~139 MB
const { text, lines } = await ocr.read(imageFile)
```

Base: PP-OCRv6 (PaddlePaddle, Apache-2.0). `tiny` / `small` / `medium`. Default is `small` (~31 MB).

### Text embeddings — `runonweb/embed`

```ts
import { TextEmbedder, cosineSimilarity } from 'runonweb/embed'

const { embeddings, dimensions } = await new TextEmbedder().embed(['a', 'b'])
cosineSimilarity(embeddings[0], embeddings[1])
```

Base: all-MiniLM-L6-v2 (Apache-2.0). 384 dims, normalized.

### Translation — `runonweb/translate`

```ts
import { Translator, PAIRS, hasPair } from 'runonweb/translate'

const t = new Translator({ from: 'en', to: 'es' }) // Firefox Translations en-es, ~37 MB
const { text } = await t.translate('Hello world')

await t.translate('Bonjour', { from: 'fr', to: 'en' }) // same instance, downloaded on demand
await t.translate('<b>Hello</b> world', { html: true }) // keeps the markup
hasPair('es', 'fr') // true — pivots through English

// one Transformers.js model for 100 languages (MIT, ~630 MB)
const multi = new Translator({ model: 'Xenova/m2m100_418M', from: 'fr', to: 'en' })
```

Base: Mozilla's [Firefox Translations](https://github.com/mozilla/translations) models (MPL-2.0), the same Marian NMT students Firefox ships. 17–44 MB per pair (int8), 106 direct pairs between English and 58 languages; other pairs pivot through English. They run in a Web Worker through the Bergamot WASM runtime (no COOP/COEP headers needed) and are cached in Cache Storage. `PAIRS` lists every pair with architecture, size and COMET score.

Weights come from the runonweb mirror on the Hugging Face Hub by default. To self-host, run `node scripts/translate-models.mjs fetch en-es es-en` (or `--all`) and pass `modelPath` pointing at wherever you upload that folder; the script also copies the runtime so you can pass `runtimePath: '<modelPath>/runtime/'` instead of loading it from jsDelivr. The registry is regenerated from Mozilla's model list with `node scripts/translate-models.mjs registry`.

### Text-to-speech — `runonweb/tts`

```ts
import { TextToSpeech } from 'runonweb/tts'

const tts = new TextToSpeech({ size: 'small', voice: 'af_heart' })
await tts.load()

for await (const chunk of tts.speakStream('Hello from the browser')) {
  // chunk.audio — 24 kHz PCM, one sentence at a time
}

const wav = await tts.speakToBlob('Hola mundo', { voice: 'ef_dora' })

// Supertonic 2: one model for en · ko · es · pt · fr, 44.1 kHz
const multi = new TextToSpeech({ size: 'multi', voice: 'st_f1' })
const pt = await multi.speakToBlob('Olá do navegador', { language: 'pt' })
```

Base: Kokoro 82M (hexgrad, Apache-2.0) by default — English, Spanish, French, on Transformers.js directly. Pass `size: 'multi'` for Supertonic 2 (Supertone, OpenRAIL-M, ~262 MB): English, Korean, Spanish, Portuguese, French with 10 shared voices `st_f1`…`st_m5` and a `language` option. Pass `size: 'tiny'` for KittenTTS nano (~28 MB, 8 English voices). Audio streams via `speakStream`. Kokoro Spanish/French download a local eSpeak-NG WASM (~18 MB) on first use.

### Core helpers — `runonweb/core`

```ts
import { isWebGPUAvailable, resolveDevice } from 'runonweb/core'

const ok = await isWebGPUAvailable()
const device = await resolveDevice() // 'webgpu' | 'wasm'
```

## Notes

- First run downloads model weights from Hugging Face; the browser caches them afterward.
- Each module picks its quantization per device (`fp16`/`fp32` on WebGPU, `q8` on WASM). `q8` on WebGPU is never used because it produces wrong results.
- Models that fail on WebGPU in ONNX Runtime Web (detect) are pinned to WASM through `supportedDevices` and ignore `device: 'webgpu'`. Translation runs on CPU (Bergamot WASM) by design. Kokoro and Supertonic TTS use WebGPU; KittenTTS (`size: 'tiny'`) is WASM-only.
- Nothing leaves the browser — audio, images and text stay on the user's device.
- All default models are Apache-2.0 or MIT except `runonweb/caption` (LFM Open License v1.0, free under USD 10M revenue). Attribution to the original authors is listed above and on every demo page.
