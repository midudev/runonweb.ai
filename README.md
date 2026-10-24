# runonweb.ai

Free, open-source AI modules that run in the browser. Live demos + the `runonweb` package.

- One import per task: `runonweb/stt`, `runonweb/remove-bg`, `runonweb/embed`…
- WebGPU when available, WASM fallback. Nothing leaves the device.
- Every module is built on an open model. Each demo page credits the base model, author and license. The code is MIT; the models keep their own licenses (see [License](#license)).

```bash
pnpm install
pnpm dev        # astro dev --background is preferred, see AGENTS.md
pnpm check      # astro check (types)
pnpm build
```

Monorepo: the site lives in [`packages/web`](./packages/web) (Astro + Cloudflare Workers) and the library in [`packages/runonweb`](./packages/runonweb). Root scripts forward to `web`. Every push to `main` is built and deployed by Cloudflare Workers Builds on the `runonweb` Worker (root directory `/packages/web`, `pnpm install --frozen-lockfile && pnpm run build`, then `npx wrangler deploy`; build variables `PNPM_VERSION=11.28.0` and `SKIP_DEPENDENCY_INSTALL=1`).

## Package

```ts
import { SpeechToText } from 'runonweb/stt'
import { RemoveBackground } from 'runonweb/remove-bg'
import { ImageCaptioner } from 'runonweb/caption'
import { DepthEstimator } from 'runonweb/depth'
import { ObjectDetector } from 'runonweb/detect'
import { TextEmbedder, cosineSimilarity } from 'runonweb/embed'
import { Classifier } from 'runonweb/classify'
import { Translator, LANGS } from 'runonweb/translate'
import { Emojifier } from 'runonweb/emoji'
import { TextToSpeech } from 'runonweb/tts'
import { isWebGPUAvailable } from 'runonweb/core'
```

Source and API docs: [`packages/runonweb`](./packages/runonweb).

## Modules

| Module | Base model | License | Download (WebGPU / WASM) |
|--------|-----------|---------|--------------------------|
| `runonweb/stt` | Whisper tiny.en (OpenAI) | Apache-2.0 | ~150 MB / ~40 MB |
| `runonweb/remove-bg` | BEN2 | MIT | WASM only · ~219 MB |
| `runonweb/caption` | LFM2.5-VL-450M (Liquid AI) · 9 languages | LFM Open License v1.0 (free < USD 10M revenue) | ~316 MB / ~505 MB |
| `runonweb/depth` | Depth Anything V2 Small | Apache-2.0 | ~50 MB / ~27 MB |
| `runonweb/detect` | RF-DETR Nano (Roboflow) | Apache-2.0 | WASM only · ~29 MB |
| `runonweb/ocr` | PP-OCRv6 tiny / small / medium (PaddlePaddle) | Apache-2.0 | ~6 MB / ~31 MB / ~139 MB (same files on both) |
| `runonweb/embed` | all-MiniLM-L6-v2 (sentence-transformers) | Apache-2.0 | ~45 MB / ~23 MB |
| `runonweb/classify` | Kev-0.8B (Jared Palmer) on Qwen3.5-0.8B-Base, LoRA merged + ONNX by runonweb (`training/kev-onnx`) | Apache-2.0 | ~750 MB · `large` (Kev-4B) ~2.7 GB (same files on both) |
| `runonweb/translate` | Firefox Translations (Mozilla, Marian NMT via Bergamot WASM) · optional M2M100 418M (Meta) | MPL-2.0 · MIT (M2M100) | WASM only · ~21–55 MB per pair · M2M100 ~640 MB |
| `runonweb/clean` | S1-mini (Superwhisper) | Apache-2.0 + naming term | ~365 MB / ~413 MB |
| `runonweb/emoji` | text2emoji-tiny (trained by runonweb, `training/text2emoji`) | MIT | WASM only · ~4 MB |
| `runonweb/tts` | Kokoro 82M (hexgrad) / Supertonic 2 (Supertone) / KittenTTS nano | Apache-2.0 · OpenRAIL-M (Supertonic) | ~326 MB (WebGPU) or ~92 MB (WASM) / ~262 MB / ~28 MB |
| `runonweb/image` | Bonsai Image 4B binary / ternary (Prism ML, on FLUX.2 Klein) | Apache-2.0 | WebGPU only · ~3.4 GB / ~3.9 GB |

Sizes are approximate. The catalog (names, colors, base model, license) lives in [`packages/web/src/data/models.ts`](./packages/web/src/data/models.ts).

### What we learned testing backends (Transformers.js 4.2, ONNX Runtime Web)

- `q8` on WebGPU silently produces garbage (embeddings had ~0.6 similarity for unrelated text). Use `fp16`/`fp32` on WebGPU, `q8` on WASM.
- Encoder-decoder models (Whisper aside) fail at session creation on WebGPU: ViT-GPT2, OPUS-MT, M2M100. They are pinned to WASM via `supportedDevices: ['wasm']`. Kokoro TTS runs on WebGPU.
- Translation left Transformers.js: Mozilla's Firefox Translations models (Bergamot WASM, int8) are 3–5× smaller than OPUS-MT q8 at equal or better COMET, and cover 106 pairs. Weights are fetched with `pnpm models:translate` into `weights/firefox-translations/` (gitignored).
- ORMBG and BiRefNet fail in ONNX Runtime Web (`ceil()` MaxPool unsupported / abort). MODNet works but is portrait-oriented.
- BEN2 (the remove-bg model) loads on WebGPU, then `OrtRun` fails: its fused LayerNorm is fp16 in / fp32 scale+bias+out, and the shader does not compile (`Invalid ShaderModule "LayerNorm"`). Pinned to WASM until `onnxruntime-web` includes [onnxruntime#32629](https://github.com/microsoft/onnxruntime/pull/32629).
- On a laptop GPU, WebGPU fp16 is 5–7× faster than WASM q8 for vision models.
- The default `onnxruntime-web` entry is the JSEP build: its `MatMulNBits` only runs 2/4-bit weights, so 8-bit layers fail at session creation (`nbits_ == 4 || nbits_ == 2 was false`). `onnxruntime-web/webgpu` (the native WebGPU EP, also what Transformers.js uses) runs them, but its CPU side has no `GatherBlockQuantized` (int4 embeddings), so the WASM fallback must load `onnxruntime-web/wasm`. `runonweb/classify` picks the entry per backend.

## Adding a module

1. Create `packages/runonweb/src/<name>/index.ts` using `loadPipeline` from `core/pipeline.ts`. Pick an explicit `dtype` per device and comment the resulting download size.
2. Add the export to `packages/runonweb/package.json`.
3. Add a `Model` entry to `packages/web/src/data/models.ts` with name, color, base model, author, license and size.
4. Add `packages/web/src/pages/models/<slug>.astro` + `packages/web/src/scripts/models/<slug>.ts`. Reuse `ModelShell`, `DemoStatus`, `DemoButton`, `ImageDropzone` and `createDemoUI`.
6. Test the model in the browser on both backends before shipping; see the backend notes above.
5. Only use models whose license allows redistribution and commercial use (Apache-2.0, MIT, BSD, CC-BY). Avoid `-NC` licenses.

## License

The code in this repository and the `runonweb` package is [MIT](./LICENSE).

The MIT license does not cover the models. The package ships no weights: the browser downloads them from the Hugging Face Hub on first use, and each model keeps the license its authors chose. The [Modules](#modules) table lists them per module, and [`MODELS.md`](./MODELS.md#licenses) per variant, with links to the three that add terms of their own:

- `runonweb/caption`: LFM Open License v1.0, free for individuals and companies under USD 10M annual revenue.
- `runonweb/tts` with `size: 'multi'`: OpenRAIL-M, with use restrictions (no impersonation, deception or illegal use).
- `runonweb/clean`: Apache-2.0, plus the duty to keep calling the model "S1-mini" by "Superwhisper".

The fonts in [`packages/web/src/assets/fonts`](./packages/web/src/assets/fonts) are Geist, under the SIL Open Font License 1.1 ([LICENSE](./packages/web/src/assets/fonts/LICENSE)).
