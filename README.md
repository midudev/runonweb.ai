# runonweb.ai

Free, open-source AI modules that run in the browser. Live demos + the `runonweb` package.

- One import per task: `runonweb/stt`, `runonweb/remove-bg`, `runonweb/embed`…
- WebGPU when available, WASM fallback. Nothing leaves the device.
- Every module is built on a permissively licensed open model (Apache-2.0 / MIT). Each demo page credits the base model, author and license.

```bash
pnpm install
pnpm dev        # astro dev --background is preferred, see AGENTS.md
pnpm check      # astro check (types)
pnpm build
pnpm deploy     # astro build && wrangler deploy (Cloudflare Workers)
```

## Package

```ts
import { SpeechToText } from 'runonweb/stt'
import { RemoveBackground } from 'runonweb/remove-bg'
import { ImageCaptioner } from 'runonweb/caption'
import { DepthEstimator } from 'runonweb/depth'
import { ObjectDetector } from 'runonweb/detect'
import { TextEmbedder, cosineSimilarity } from 'runonweb/embed'
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
| `runonweb/detect` | DETR ResNet-50 (Meta) | Apache-2.0 | ~85 MB / ~43 MB |
| `runonweb/embed` | all-MiniLM-L6-v2 (sentence-transformers) | Apache-2.0 | ~45 MB / ~23 MB |
| `runonweb/translate` | Firefox Translations (Mozilla, Marian NMT via Bergamot WASM) | MPL-2.0 | WASM only · ~22–49 MB per pair |
| `runonweb/emoji` | text2emoji-tiny (trained by runonweb, `training/text2emoji`) | MIT | WASM only · ~4 MB |
| `runonweb/tts` | Kokoro 82M (hexgrad) / Supertonic 2 (Supertone) / KittenTTS nano | Apache-2.0 · OpenRAIL-M (Supertonic) | ~326 MB (WebGPU) or ~92 MB (WASM) / ~262 MB / ~28 MB |

Sizes are approximate. The catalog (names, colors, base model, license) lives in [`src/data/models.ts`](./src/data/models.ts).

### What we learned testing backends (Transformers.js 4.2, ONNX Runtime Web)

- `q8` on WebGPU silently produces garbage (embeddings had ~0.6 similarity for unrelated text). Use `fp16`/`fp32` on WebGPU, `q8` on WASM.
- Encoder-decoder models (Whisper aside) fail at session creation on WebGPU: ViT-GPT2, OPUS-MT, M2M100. They are pinned to WASM via `supportedDevices: ['wasm']`. Kokoro TTS runs on WebGPU.
- Translation left Transformers.js: Mozilla's Firefox Translations models (Bergamot WASM, int8) are 3–5× smaller than OPUS-MT q8 at equal or better COMET, and cover 106 pairs. Weights are fetched with `pnpm models:translate` into `weights/firefox-translations/` (gitignored).
- ORMBG and BiRefNet fail in ONNX Runtime Web (`ceil()` MaxPool unsupported / abort). MODNet works but is portrait-oriented.
- BEN2 (the remove-bg model) loads on WebGPU, then `OrtRun` fails: its fused LayerNorm is fp16 in / fp32 scale+bias+out, and the shader does not compile (`Invalid ShaderModule "LayerNorm"`). Pinned to WASM until `onnxruntime-web` includes [onnxruntime#32629](https://github.com/microsoft/onnxruntime/pull/32629).
- On a laptop GPU, WebGPU fp16 is 5–7× faster than WASM q8 for vision models.

## Site

| Path | Description |
|------|-------------|
| `/` | Landing + catalog |
| `/models` | All models grouped by input |
| `/models/<slug>` | Model page: hero, live demo, code, specs, FAQ (`stt`, `remove-bg`, `caption`, `depth`, `detect`, `embed`, `translate`, `emoji`, `tts`) |
| `/docs` | Install, the shared pattern, API for every module |

Sample media for the one-click demos lives in `public/samples/` (images and speech clips taken from the Transformers.js docs dataset `Xenova/transformers.js-docs`; no explicit license is published there, so replace them with owned or CC0 media before a public launch). Sample definitions are the `samples` field in `src/data/models.ts`.

Design: dark "instrument panel". Geist for headings and body, Geist Mono for code, **Geist Pixel as the machine voice** (readouts: sizes, timings, backends, section numbers, status bars). Accent is phosphor amber (#ffb340). One hue per model used for LEDs and a hand-drawn 12×8 pixel icon per task (`icon` rows in `models.ts`, rendered by `PixelIcon.astro`). Favicon set is generated from the logo (`public/favicon.svg`, PNGs via sharp). The landing hero is an ordered-dither field computed by the page every frame (`src/scripts/hero-dither.ts`). Bench numbers in `models.ts` are measured, not estimated.

## Deploy (Cloudflare Workers)

The site is fully static (OG images are prerendered at build time). There is no `@astrojs/cloudflare` adapter — Wrangler serves `dist/` as [Workers static assets](https://developers.cloudflare.com/workers/static-assets/).

```bash
pnpm build
pnpm exec wrangler login    # once
pnpm deploy                 # build + wrangler deploy
```

Local preview of the Worker (same routing as production):

```bash
pnpm preview:worker
```

Attach `runonweb.ai` as a custom domain in the Cloudflare dashboard (Workers & Pages → runonweb → Settings → Domains).

**CI:** push to `main` runs [`.github/workflows/deploy.yml`](./.github/workflows/deploy.yml). Add repository secrets `CLOUDFLARE_API_TOKEN` (Workers edit permission) and `CLOUDFLARE_ACCOUNT_ID`. Alternatively, import the repo in **Workers Builds** with build command `pnpm build` and deploy command `npx wrangler deploy`.

Cloudflare Workers reject static files over 25 MiB. `pnpm build` drops anything larger from `dist/` (`scripts/omit-oversized-assets.mjs`):

- Firefox Translations `.bin` weights (~30 MB) — the translate demo loads them from Hugging Face. `pnpm models:translate` is only for local offline copies.
- ONNX Runtime `jsep.wasm` (~27 MB) — Transformers.js, OCR and Kitten TTS already point `wasmPaths` at jsDelivr.

## Adding a module

1. Create `packages/runonweb/src/<name>/index.ts` using `loadPipeline` from `core/pipeline.ts`. Pick an explicit `dtype` per device and comment the resulting download size.
2. Add the export to `packages/runonweb/package.json`.
3. Add a `Model` entry to `src/data/models.ts` with name, color, base model, author, license and size.
4. Add `src/pages/models/<slug>.astro` + `src/scripts/models/<slug>.ts`. Reuse `ModelShell`, `DemoStatus`, `DemoButton`, `ImageDropzone` and `createDemoUI`.
6. Test the model in the browser on both backends before shipping; see the backend notes above.
5. Only use models whose license allows redistribution and commercial use (Apache-2.0, MIT, BSD, CC-BY). Avoid `-NC` licenses.
