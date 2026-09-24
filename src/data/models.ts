export type WebGPULevel = 'required' | 'recommended' | 'optional' | 'none'
export type ModelInput = 'audio' | 'image' | 'text'
export type ModelStatus = 'available' | 'beta'

/** Where the weights come from. Shown on every model page for transparency. */
export type Weights = {
  /** Hugging Face repo the SDK downloads from. */
  id: string
  /** Original model name. */
  baseModel: string
  /** Original author or organization. */
  author: string
  /** SPDX-style license of the original weights. */
  license: string
  /** Link to the original model card. */
  sourceUrl: string
  /** Approximate first-download size as "WebGPU / WASM". */
  size: string
  /** Quantization actually used per backend. */
  runsOn: string
}

export type Sample =
  | { kind: 'image'; label: string; src: string }
  | { kind: 'audio'; label: string; src: string; duration: string; language?: string }
  | { kind: 'text'; label: string; text: string; extra?: string }

export type Model = {
  slug: string
  /** Short product name (Scribe, Remover…). */
  name: string
  /** Task label (Speech-to-Text…). */
  task: string
  /** One-liner shown on cards. */
  tagline: string
  /** Longer description shown on the model page. */
  description: string
  packagePath: string
  input: ModelInput
  status: ModelStatus
  webgpu: WebGPULevel
  /** LED / icon hue on the dark UI. */
  hue: string
  /**
   * 12×8 pixel icon. '#' = full, '+' = 60%, '-' = 30%, '~' = neutral checker, '.' = empty.
   * Drawn by PixelIcon.astro; keep it readable at 24px.
   */
  icon: string[]
  /** One-click inputs for the live demo. `src` is served from /public. */
  samples: Sample[]
  /** Measured on an Apple M-series laptop, Chrome, after weights are cached. */
  bench: { webgpu?: string; wasm?: string }
  notes: string[]
  weights: Weights
  usageSnippet: string
}

/** Install commands: pnpm first, then npm, bun, yarn. */
export const INSTALL_SNIPPET = `pnpm add runonweb
npm install runonweb
bun add runonweb
yarn add runonweb`

export const INPUT_LABEL: Record<ModelInput, string> = {
  audio: 'Audio',
  image: 'Image',
  text: 'Text',
}

/** Classify samples carry their questions (and demo flags) as JSON in `extra`. */
function kevQuestions(questions: Record<string, unknown>, flags: { dateFacts?: boolean } = {}): string {
  return JSON.stringify({ questions, ...flags })
}

export const models: Model[] = [
  {
    slug: 'ocr',
    name: 'Scan',
    task: 'OCR',
    tagline: 'Read text from any photo. Pick the size.',
    description:
      'Extract printed and scene text in the browser. Three sizes (tiny, small, medium) so you can trade a few megabytes for accuracy. Nothing is uploaded.',
    packagePath: 'runonweb/ocr',
    input: 'image',
    status: 'available',
    webgpu: 'optional',
    hue: '#c8ff4a',
    icon: [
      '############',
      '............',
      '########....',
      '............',
      '##########..',
      '............',
      '######......',
      '............',
    ],
    samples: [
      { kind: 'image', label: 'Alfajor', src: '/samples/ocr-alfajor.jpg' },
      { kind: 'image', label: 'Receipt', src: '/samples/ocr-receipt.png' },
      { kind: 'image', label: 'Carta · ES', src: '/samples/ocr-menu-es.png' },
      { kind: 'image', label: 'Carte · FR', src: '/samples/ocr-menu-fr.png' },
      { kind: 'image', label: 'Document', src: '/samples/ocr-document.png' },
      { kind: 'image', label: 'El Tano', src: '/samples/ocr-el-tano.png' },
      { kind: 'image', label: 'Label', src: '/samples/ocr-sign.jpg' },
      { kind: 'image', label: 'Street', src: '/samples/street.jpg' },
    ],
    bench: { webgpu: '~0.6 s', wasm: '~2 s' },
    notes: [
      'Default size is `small` (~31 MB): the best size/quality trade-off. Pass `tiny` (~6 MB) or `medium` (~139 MB).',
      '`small` and `medium` cover 50 languages (Chinese, English, Japanese, Latin). `tiny` is Chinese + English.',
      'Each line comes back with a box and a confidence score.',
    ],
    weights: {
      id: 'PaddlePaddle/PP-OCRv6_small_det_onnx',
      baseModel: 'PP-OCRv6',
      author: 'PaddlePaddle',
      license: 'Apache-2.0',
      sourceUrl: 'https://huggingface.co/collections/PaddlePaddle/pp-ocrv6',
      size: '~31 MB (small) · tiny ~6 MB · medium ~139 MB',
      runsOn: 'fp32 on WebGPU · fp32 on WASM',
    },
    usageSnippet: `import { OCR } from 'runonweb/ocr'

const ocr = new OCR({
  size: 'small', // tiny | small | medium
  onProgress: (info) => console.log(info.status, info.progress),
})

await ocr.load()

const { text, lines } = await ocr.read(imageFile)
console.log(text)
// lines: [{ text, score, box: { xmin, ymin, xmax, ymax } }]

ocr.dispose()`,
  },
  {
    slug: 'remove-bg',
    name: 'Remover',
    task: 'Background removal',
    tagline: 'Drop a photo, get a transparent PNG.',
    description:
      'Remove image backgrounds client-side. Drop a photo and get a PNG with an alpha channel. Private by default, and offline-capable once the model is cached.',
    packagePath: 'runonweb/remove-bg',
    input: 'image',
    status: 'available',
    webgpu: 'none',
    hue: '#ff8f5e',
    icon: [
      '++++++++++++',
      '+--##--~.~.+',
      '+--##--.~.~+',
      '+------~.~.+',
      '+----#-.~.~+',
      '+---###~.~.+',
      '+--#####.~.+',
      '++++++++++++',
    ],
    samples: [
      { kind: 'image', label: 'Messi', src: '/samples/messi.jpg' },
      { kind: 'image', label: 'Alfajores', src: '/samples/alfajores.jpg' },
      { kind: 'image', label: 'Portrait', src: '/samples/portrait.jpg' },
      { kind: 'image', label: 'Afro', src: '/samples/afro.jpg' },
      { kind: 'image', label: 'Man & car', src: '/samples/man-car.jpg' },
      { kind: 'image', label: 'Corgi', src: '/samples/corgi.jpg' },
    ],
    bench: { wasm: '~15 s' },
    notes: [
      'BEN2 (2025) is a general background eraser: hair, fur, products and hard edges, not just portraits.',
      '~219 MB of fp16 weights, cached after the first download.',
      'Output is a PNG Blob with alpha, same size as the input.',
      'Runs on WASM. WebGPU dies in BEN2’s LayerNorm shader (fp16 activation, fp32 scale and bias) on current onnxruntime-web. The fix is upstream (onnxruntime#32629, 2026-09-22) and not published yet.',
      'Sample photos: Messi by Bryan Berlin (CC BY-SA 4.0), alfajores by V!NZ (CC BY-SA 3.0). Both resized.',
    ],
    weights: {
      id: 'onnx-community/BEN2-ONNX',
      baseModel: 'BEN2',
      author: 'Prama LLC',
      license: 'MIT',
      sourceUrl: 'https://huggingface.co/PramaLLC/BEN2',
      size: '~219 MB',
      runsOn: 'fp16 on WASM',
    },
    usageSnippet: `import { RemoveBackground } from 'runonweb/remove-bg'

const remover = new RemoveBackground({
  onProgress: (info) => console.log(info.status, info.progress),
})

await remover.load()

// Blob, File, URL, HTMLImageElement, ImageData, or canvas
const png = await remover.remove(imageFile)
// png is a PNG Blob with alpha channel

document.querySelector('img').src = URL.createObjectURL(png)

remover.dispose()`,
  },
  {
    slug: 'stt',
    name: 'Scribe',
    task: 'Speech-to-Text',
    tagline: 'Transcribe audio without a server.',
    description:
      'Transcribe audio to text entirely in the browser. Record from the mic or drop a file. Nothing is uploaded, nothing is billed per minute.',
    packagePath: 'runonweb/stt',
    input: 'audio',
    status: 'beta',
    webgpu: 'recommended',
    hue: '#5ee7ff',
    icon: [
      '....#...#...',
      '..#.##..##..',
      '.#####.####.',
      '############',
      '############',
      '.#####.####.',
      '..#.##..##..',
      '....#...#...',
    ],
    samples: [
      { kind: 'audio', label: 'JFK · inaugural', src: '/samples/jfk.wav', duration: '11 s', language: 'en' },
      { kind: 'audio', label: 'MLK · dream', src: '/samples/mlk.wav', duration: '13 s', language: 'en' },
      { kind: 'audio', label: 'TED · talk', src: '/samples/ted.wav', duration: '14 s', language: 'en' },
      { kind: 'audio', label: 'Messi', src: '/samples/messi.wav', duration: '11 s', language: 'es' },
      { kind: 'audio', label: 'Español · ES', src: '/samples/spanish-es.wav', duration: '12 s', language: 'es' },
      { kind: 'audio', label: 'Español · LATAM', src: '/samples/spanish-latam.wav', duration: '14 s', language: 'es' },
      { kind: 'audio', label: 'Japonés · JA', src: '/samples/japanese.wav', duration: '12 s', language: 'ja' },
      { kind: 'audio', label: 'LibriSpeech', src: '/samples/librispeech.wav', duration: '6 s', language: 'en' },
      { kind: 'audio', label: 'Libri · exist', src: '/samples/libri-exist.wav', duration: '10 s', language: 'en' },
      { kind: 'audio', label: 'Libri · brief', src: '/samples/libri-brief.wav', duration: '3 s', language: 'en' },
      { kind: 'audio', label: 'Libri · passage', src: '/samples/libri-passage.wav', duration: '7 s', language: 'en' },
    ],
    bench: { webgpu: '0.4 s / 11 s audio', wasm: '1.6 s / 11 s audio' },
    notes: [
      'English-only by default. Pass `model` (e.g. `onnx-community/whisper-tiny`) and `language` (`es`, `ja`…) for a multilingual Whisper variant. Without `language`, tiny often guesses English and the transcript comes out translated. The live demo always uses `task: "transcribe"` and lets you pick the language.',
      'WebGPU is several times faster; WASM works everywhere else.',
      'Audio is decoded and resampled to 16 kHz mono in the browser.',
      'Pass `timestamps: "segment"` for each phrase with the model\'s start and end, in seconds. `"word"` (or `true`) splits those phrases into words and keeps every word inside its phrase. This ONNX Whisper has no cross-attentions, so it cannot time a word on its own. Omit `timestamps` and `chunks` is empty.',
      'Pass `onPartial` to receive words as they are generated.',
    ],
    weights: {
      id: 'onnx-community/whisper-tiny.en',
      baseModel: 'Whisper tiny.en',
      author: 'OpenAI',
      license: 'Apache-2.0',
      sourceUrl: 'https://huggingface.co/openai/whisper-tiny.en',
      size: '~150 MB / ~40 MB',
      runsOn: 'fp32 on WebGPU · q8 on WASM',
    },
    usageSnippet: `import { SpeechToText } from 'runonweb/stt'

const stt = new SpeechToText({
  onProgress: (info) => console.log(info.status, info.progress),
})

await stt.load()

// Blob, File, URL string, or Float32Array (16 kHz mono)
const { text, chunks } = await stt.transcribe(audioBlob, {
  timestamps: 'word',
  onPartial: (partial) => console.log(partial),
})
console.log(text, chunks[0])

stt.dispose()`,
  },
  {
    slug: 'caption',
    name: 'Alt',
    task: 'Image captioning',
    tagline: 'One sentence for any image, in nine languages.',
    description:
      'Describe any image in one sentence, in English, Spanish, Portuguese, French, German, Arabic, Chinese, Japanese or Korean. Useful for alt text, search indexing and accessibility. Generated on-device, no image ever leaves the page.',
    packagePath: 'runonweb/caption',
    input: 'image',
    status: 'beta',
    webgpu: 'recommended',
    hue: '#4ff0b6',
    icon: [
      '............',
      '#######.....',
      '#...#.#.####',
      '#.....#.....',
      '#..#..#.####',
      '#.###.#.....',
      '#######.###.',
      '............',
    ],
    samples: [
      { kind: 'image', label: 'Cats', src: '/samples/cats.jpg' },
      { kind: 'image', label: 'Street', src: '/samples/street.jpg' },
      { kind: 'image', label: 'Savanna', src: '/samples/savanna.jpg' },
      { kind: 'image', label: 'Corgi', src: '/samples/corgi.jpg' },
    ],
    bench: { webgpu: '0.7–1.2 s', wasm: '~30 s' },
    notes: [
      'LFM2.5-VL-450M (Liquid AI, Nov 2025). A small vision-language model, prompted for captions, not a dedicated captioner.',
      'Pass `language` (`en`, `es`, `pt`, `fr`, `de`, `ar`, `zh`, `ja`, `ko`) to caption in that language. Other languages are best-effort.',
      'Pass `detail: "detailed"` or `"more"` for longer captions. Default is short alt text. `onPartial` streams the text as it is generated.',
      'License: LFM Open License v1.0. Free for individuals and companies under USD 10M annual revenue. Above that, Liquid AI requires a commercial license. Not OSI-approved; the rest of the catalog is Apache-2.0 or MIT.',
      'WebGPU is the fast path (fp16 vision + q4f16 decoder, about a second per caption). WASM works but takes ~30 s per caption.',
      'It is a generative model: captions can hallucinate details. Review before publishing.',
    ],
    weights: {
      id: 'onnx-community/LFM2.5-VL-450M-ONNX',
      baseModel: 'LFM2.5-VL-450M',
      author: 'Liquid AI',
      license: 'LFM Open License v1.0 (free < USD 10M revenue)',
      sourceUrl: 'https://huggingface.co/LiquidAI/LFM2.5-VL-450M',
      size: '~316 MB / ~505 MB',
      runsOn: 'fp16+q4f16 on WebGPU · fp16+q8+q4 on WASM',
    },
    usageSnippet: `import { ImageCaptioner } from 'runonweb/caption'

const captioner = new ImageCaptioner({ language: 'es', detail: 'short' })
await captioner.load()

const { text } = await captioner.caption(imageFile)
// "Un gato sentado sobre una mesa de madera."

// Per call overrides, streamed
await captioner.caption(imageFile, {
  language: 'en',
  detail: 'detailed',
  onPartial: (partial) => console.log(partial),
})

captioner.dispose()`,
  },
  {
    slug: 'depth',
    name: 'Depth',
    task: 'Depth estimation',
    tagline: 'A depth map from a single photo.',
    description:
      'Turn a single photo into a depth map. Great for parallax effects, portrait blur or 3D-ish previews without any server round-trip.',
    packagePath: 'runonweb/depth',
    input: 'image',
    status: 'available',
    webgpu: 'recommended',
    hue: '#b28cff',
    icon: [
      '------------',
      '-++++++++++-',
      '-+########+-',
      '-+#......#+-',
      '-+#......#+-',
      '-+########+-',
      '-++++++++++-',
      '------------',
    ],
    samples: [
      { kind: 'image', label: 'Obelisco', src: '/samples/ba-obelisco.jpg' },
      { kind: 'image', label: 'Congreso', src: '/samples/ba-congreso.jpg' },
      { kind: 'image', label: 'Caminito', src: '/samples/ba-caminito.jpg' },
      { kind: 'image', label: 'Barolo', src: '/samples/ba-barolo.jpg' },
    ],
    bench: { webgpu: '0.8 s · 640×480', wasm: '5.6 s' },
    notes: [
      'Output is a grayscale PNG: brighter means closer.',
      'The Small variant is Apache-2.0; larger Depth Anything models are not.',
      'Sample photos, resized: Obelisco by Roberto Fiadone (CC BY-SA 4.0), Congreso by Jorge Royan (CC BY-SA 3.0), Caminito by Lars Curfs (CC BY-SA 3.0 NL), Palacio Barolo by Beatrice Murch (CC BY 2.0).',
    ],
    weights: {
      id: 'onnx-community/depth-anything-v2-small',
      baseModel: 'Depth Anything V2 Small',
      author: 'DepthAnything (HKU / TikTok)',
      license: 'Apache-2.0',
      sourceUrl: 'https://huggingface.co/depth-anything/Depth-Anything-V2-Small',
      size: '~50 MB / ~27 MB',
      runsOn: 'fp16 on WebGPU · q8 on WASM',
    },
    usageSnippet: `import { DepthEstimator } from 'runonweb/depth'

const estimator = new DepthEstimator()
await estimator.load()

const { depth, width, height } = await estimator.estimate(imageFile)
// depth is a grayscale PNG Blob

document.querySelector('img').src = URL.createObjectURL(depth)

estimator.dispose()`,
  },
  {
    slug: 'detect',
    name: 'Spot',
    task: 'Object detection',
    tagline: 'Boxes and labels for 80 everyday objects.',
    description:
      'Find and label objects in an image with bounding boxes. People, cars, cats, cups, laptops: 80 COCO classes detected locally with RF-DETR Nano.',
    packagePath: 'runonweb/detect',
    input: 'image',
    status: 'available',
    webgpu: 'none',
    hue: '#ff5fa8',
    icon: [
      '##........##',
      '#..........#',
      '.....##.....',
      '....####....',
      '....####....',
      '.....##.....',
      '#..........#',
      '##........##',
    ],
    samples: [
      { kind: 'image', label: 'Pizzas', src: '/samples/pizzas.jpg' },
      { kind: 'image', label: 'Football', src: '/samples/football.jpg' },
      { kind: 'image', label: 'Cats', src: '/samples/cats.jpg' },
      { kind: 'image', label: 'Airport', src: '/samples/airport.jpg' },
    ],
    bench: { wasm: '~1 s' },
    notes: [
      'RF-DETR Nano (2025). 80 COCO classes: people, cars, animals, furniture and more.',
      'WASM only for now. WebGPU is skipped: this ONNX export collapses confidence scores.',
      'Boxes are returned in pixel coordinates of the original image.',
      'Tune `threshold` to trade recall for precision.',
    ],
    weights: {
      id: 'onnx-community/rfdetr_nano-ONNX',
      baseModel: 'RF-DETR Nano',
      author: 'Roboflow',
      license: 'Apache-2.0',
      sourceUrl: 'https://github.com/roboflow/rf-detr',
      size: '~29 MB',
      runsOn: 'q8 on WASM',
    },
    usageSnippet: `import { ObjectDetector } from 'runonweb/detect'

const detector = new ObjectDetector({
  threshold: 0.5, // min confidence
})

await detector.load()

const objects = await detector.detect(imageFile)
// [{ label: 'cat', score: 0.98, box: { xmin, ymin, xmax, ymax } }]

detector.dispose()`,
  },
  {
    slug: 'embed',
    name: 'Vector',
    task: 'Text embeddings',
    tagline: 'Semantic search in 23 MB.',
    description:
      'Turn sentences into 384-dimensional vectors for semantic search, clustering or deduplication. Compare meaning, not keywords, in the browser, in milliseconds.',
    packagePath: 'runonweb/embed',
    input: 'text',
    status: 'beta',
    webgpu: 'optional',
    hue: '#6aa3ff',
    icon: [
      '.+........+.',
      '....+..#....',
      '.+....##.#..',
      '.......###..',
      '..+...##....',
      '+.....+.#...',
      '....+....+..',
      '.+.......+..',
    ],
    samples: [
      {
        kind: 'text',
        label: 'Support tickets',
        text: 'How do I reset my password?',
        extra: 'I forgot my login credentials\nWhere can I change my account email?\nWhat is the weather today?\nThe quick brown fox jumps over the lazy dog\nSteps to recover access to my account',
      },
      {
        kind: 'text',
        label: 'Recipes',
        text: 'Quick vegetarian dinner for two',
        extra: 'Spinach and ricotta pasta in 20 minutes\nSlow-cooked beef brisket\nTofu stir-fry with broccoli\nHow to change a flat tire\nLentil soup with cumin',
      },
      {
        kind: 'text',
        label: 'Code search',
        text: 'function that parses a date string',
        extra: 'parseISO(input: string): Date\nformatCurrency(amount, locale)\nfromTimestamp(ts: number)\nuseDebounce(value, delay)\ntoDate(value: string | number)',
      },
    ],
    bench: { webgpu: '14 ms / sentence', wasm: '40 ms / sentence' },
    notes: [
      'Vectors are L2-normalized by default, so cosine similarity is a dot product.',
      'Works best on English; pass a multilingual model via `model` if needed.',
    ],
    weights: {
      id: 'Xenova/all-MiniLM-L6-v2',
      baseModel: 'all-MiniLM-L6-v2',
      author: 'sentence-transformers',
      license: 'Apache-2.0',
      sourceUrl: 'https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2',
      size: '~45 MB / ~23 MB',
      runsOn: 'fp16 on WebGPU · q8 on WASM',
    },
    usageSnippet: `import { TextEmbedder, cosineSimilarity } from 'runonweb/embed'

const embedder = new TextEmbedder()
await embedder.load()

const { embeddings } = await embedder.embed([
  'How do I reset my password?',
  'I forgot my login credentials',
  'What is the weather today?',
])

cosineSimilarity(embeddings[0], embeddings[1]) // ~0.7
cosineSimilarity(embeddings[0], embeddings[2]) // ~0.0

embedder.dispose()`,
  },
  {
    slug: 'classify',
    name: 'Classify',
    task: 'Text classification',
    tagline: 'Ask typed questions about any text. Get probabilities.',
    description:
      'Route a ticket, rate a review or flag a comment: yes/no, multiple-choice and rating questions about one text, answered with calibrated probabilities in a single call. Runs Kev, an open Jev-style decision model, in the tab. No text is generated.',
    packagePath: 'runonweb/classify',
    input: 'text',
    status: 'available',
    webgpu: 'recommended',
    hue: '#ffb3c7',
    icon: [
      '....##......',
      '....##......',
      '....##......',
      '....##....++',
      '.++.##....++',
      '.++.##.++.++',
      '.++.##.++.++',
      '############',
    ],
    samples: [
      {
        kind: 'text',
        label: 'Ticket',
        text: 'Shoes arrived two weeks late and in the wrong size. Also I see two charges on my card.',
        extra: kevQuestions({
          department: {
            type: 'choice',
            instructions: 'Which team should handle this?',
            criteria: {
              returns: 'Exchanges, refunds, wrong or damaged items',
              shipping: 'Delivery status, delays, lost packages',
              billing: 'Charges, invoices, payment problems',
            },
          },
          escalate: { type: 'noul', instructions: 'Does this need urgent human attention?' },
          frustration: {
            type: 'score',
            instructions: 'How frustrated is the customer?',
            criteria: ['Calm', 'Frustrated', 'Very angry'],
          },
        }),
      },
      {
        kind: 'text',
        label: 'Review',
        text: "The new update is amazing! Battery life doubled and the camera is way sharper. Best phone I've owned.",
        extra: kevQuestions({
          sentiment: {
            type: 'choice',
            instructions: 'What is the sentiment of this review?',
            criteria: { positive: null, neutral: null, negative: null },
          },
          stars: {
            type: 'score',
            instructions: 'How many stars would this reviewer give?',
            criteria: ['1 star', '2 stars', '3 stars', '4 stars', '5 stars'],
          },
          mentions_price: { type: 'noul', instructions: 'Does the review mention the price?' },
        }),
      },
      {
        kind: 'text',
        label: 'Comment',
        text: "You're an idiot and nobody wants you here. Log off before I find where you live.",
        extra: kevQuestions({
          toxic: { type: 'noul', instructions: 'Is this comment toxic?' },
          category: {
            type: 'choice',
            instructions: 'What kind of comment is this?',
            criteria: {
              threat: 'Threatens violence or harm',
              insult: 'Insults or demeans someone',
              spam: 'Ads or repeated links',
              fine: 'Nothing wrong with it',
            },
          },
          severity: {
            type: 'score',
            instructions: 'How severe is it?',
            criteria: ['Harmless', 'Rude', 'Abusive', 'Dangerous'],
          },
        }),
      },
      {
        kind: 'text',
        label: 'Sales lead',
        text: 'Hi, we are a 400-person logistics company evaluating tools to replace our spreadsheets. Budget approved for Q3. Can we get a demo next week?',
        extra: kevQuestions({
          intent: {
            type: 'choice',
            instructions: 'What does the sender want?',
            criteria: {
              demo: 'Wants a demo or sales call',
              support: 'Needs help with an existing account',
              job: 'Applying for a job',
              other: null,
            },
          },
          qualified: { type: 'noul', instructions: 'Is this a qualified sales lead (has budget and a real need)?' },
          urgency: {
            type: 'score',
            instructions: 'How soon do they want to buy?',
            criteria: ['No timeline', 'This year', 'This quarter', 'This month'],
          },
        }),
      },
      {
        kind: 'text',
        label: 'Return window',
        text: 'Policy: items can be returned within 14 days of delivery.\nOrder delivered on June 26, 2026.\nCustomer asked for a refund on July 18, 2026.',
        extra: kevQuestions(
          { eligible: { type: 'noul', instructions: 'Is the refund request inside the return window?' } },
          { dateFacts: true }
        ),
      },
      {
        kind: 'text',
        label: 'Ticket · ES',
        text: 'Hola, desde ayer no puedo iniciar sesión en la app. Me dice contraseña incorrecta pero la acabo de cambiar. ¿Me ayudáis?',
        extra: kevQuestions({
          team: {
            type: 'choice',
            instructions: '¿Qué equipo debe atender este ticket?',
            criteria: {
              billing: 'Pagos y reembolsos',
              shipping: 'Problemas de envío',
              access: 'Acceso a la cuenta e inicio de sesión',
            },
          },
          sentiment: {
            type: 'choice',
            instructions: 'Sentiment of the message',
            criteria: { positive: null, neutral: null, negative: null },
          },
          spam: { type: 'noul', instructions: 'Is this message spam?' },
        }),
      },
    ],
    bench: { webgpu: '~120 ms / 3 questions · large ~450 ms', wasm: '~4 s / 3 questions · large ~29 s' },
    notes: [
      'Kev-0.8B by Jared Palmer: a LoRA and a pointer head on Qwen3.5-0.8B-Base, trained to answer typed questions. Each question scores its options in one forward pass; nothing is generated, so answers are probabilities, not text.',
      'Request and answer shapes are TypeSafe’s System One API (`noul` · `choice` · `score`), the same JSON a Kev or Jev server accepts. Question ids are yours; the model never sees them.',
      'Each question only sees the text and itself. The text is read once and its cache is reused for every question.',
      'Probabilities are calibrated with the temperature fitted for the checkpoint (~2.35). Pass `temperature: 1` for raw logits. Option order can still change an answer.',
      'Kev can’t subtract dates. `dateFacts: true` appends the day count between every pair of absolute dates in the text, as Kev’s `KEV_DATE_FACTS` does.',
      'Two sizes. `small` (default) is Kev-0.8B: 0.70 on sources it wasn’t trained on, weak on general knowledge, best in English. `size: "large"` is Kev-4B: 0.84 (Jev: 0.86), ~2.7 GB and about 4× slower. Test either on your own data before trusting a threshold.',
      'runonweb merged the LoRA and exported the backbone to ONNX: int4 weights with fp32 activations (Kev-0.8B keeps its Gated DeltaNet layers in int8). Probabilities differ from the fp32 export by ~0.03 (small) and ~0.04 (large) on average. Recipe in `training/kev-onnx`.',
    ],
    weights: {
      id: 'midudev/kev-0.8b-ONNX',
      baseModel: 'Kev-0.8B (Qwen3.5-0.8B-Base + LoRA)',
      author: 'Jared Palmer · base: Qwen',
      license: 'Apache-2.0',
      sourceUrl: 'https://huggingface.co/jaredpalmer/kev-0.8b',
      size: '~750 MB · large ~2.7 GB',
      runsOn: 'int4 on WebGPU · same files on WASM',
    },
    usageSnippet: `import { Classifier } from 'runonweb/classify'

const classifier = new Classifier() // size: 'large' for Kev-4B (~2.7 GB)
await classifier.load()

const { answers } = await classifier.classify({
  state: 'Shoes arrived two weeks late and in the wrong size.',
  questions: {
    department: {
      type: 'choice',
      instructions: 'Which team should handle this?',
      criteria: { returns: null, shipping: null, billing: null },
    },
    escalate: { type: 'noul', instructions: 'Does this need urgent human attention?' },
    frustration: {
      type: 'score',
      instructions: 'How frustrated is the customer?',
      criteria: ['Calm', 'Frustrated', 'Very angry'],
    },
  },
})

answers.department.choice        // "returns"
answers.department.probabilities // { returns: 0.50, shipping: 0.44, billing: 0.06 }
answers.escalate.noul            // 0.51 = p(yes)
answers.frustration.score        // 1.21 on a 0–2 scale

classifier.dispose()`,
  },
  {
    slug: 'translate',
    name: 'Lingo',
    task: 'Translation',
    tagline: 'Translate text without a server.',
    description:
      'Translate text entirely in the browser. 17–44 MB per language pair, 58 languages to and from English. Private, offline after the first load, and free of per-character pricing.',
    packagePath: 'runonweb/translate',
    input: 'text',
    status: 'beta',
    webgpu: 'none',
    hue: '#f0abfc',
    icon: [
      '.#####......',
      '.#...#......',
      '.#...#..++++',
      '.#####..+..+',
      '..#.....+..+',
      '........++++',
      '.........+..',
      '............',
    ],
    samples: [
      { kind: 'text', label: 'Product copy', text: 'Browser-side AI keeps your data private and works offline once the model is cached.' },
      { kind: 'text', label: 'Support reply', text: 'Thanks for reaching out. We have refunded your order and you should see it in 3 to 5 business days.' },
      { kind: 'text', label: 'Recipe', text: 'Whisk the eggs with a pinch of salt, then fold in the grated cheese and chopped herbs.' },
      { kind: 'text', label: 'HTML', text: 'Click <a href="/docs">the docs</a> to <strong>get started</strong> in five minutes.', extra: 'html' },
    ],
    bench: { wasm: '~0.1 s / sentence' },
    notes: [
      'Marian NMT students distilled from a larger teacher, run through the Bergamot WASM runtime (int8 GEMM, single worker, no COOP/COEP headers needed).',
      '`base-memory` when available (≈37 MB, on par with cloud translators within ~5% COMET), `tiny` for the rest (≈22 MB). `PAIRS` lists every pair with size and COMET score.',
      'Pairs without a direct model pivot through English: `es → fr` loads `es-en` and `en-fr`.',
      'Pass `html: true` to translate markup while keeping the tags in place.',
      'Codes are BCP 47: `zh` is Simplified Chinese, `zh-Hant` Traditional. One `Translator` can serve many pairs; new pairs download on demand.',
      'Prefer one model for 100 languages? Pass `model: "Xenova/m2m100_418M"` (MIT, ~630 MB, Transformers.js, slower).',
    ],
    weights: {
      id: 'midudev/firefox-translations',
      baseModel: 'Firefox Translations (Marian NMT)',
      author: 'Mozilla',
      license: 'MPL-2.0',
      sourceUrl: 'https://github.com/mozilla/translations',
      size: '~22 MB (tiny) · ~37 MB (base-memory) · ~49 MB (ja, ko, zh)',
      runsOn: 'int8 on WASM (Bergamot)',
    },
    usageSnippet: `import { Translator, PAIRS } from 'runonweb/translate'

// en → es: base-memory, ~37 MB
const translator = new Translator({ from: 'en', to: 'es' })
await translator.load()

const { text } = await translator.translate('Hello world')
// "Hola mundo"

// same instance, another pair (downloaded on demand)
await translator.translate('Bonjour', { from: 'fr', to: 'en' })

// keep the markup, translate the text nodes
await translator.translate('<b>Hello</b> world', { html: true })

translator.dispose()

PAIRS['en-ja'] // { architecture: 'base-memory', bytes: 43849787, comet: 0.90, … }`,
  },
  {
    slug: 'clean',
    name: 'Polish',
    task: 'Transcript cleanup',
    tagline: 'Turn raw dictation into written text.',
    description:
      'Clean a speech-to-text transcript in the tab: drop fillers, resolve self-corrections, write numbers and emails, and apply punctuation. Pair it with Scribe, or paste any rough transcript.',
    packagePath: 'runonweb/clean',
    input: 'text',
    status: 'beta',
    webgpu: 'recommended',
    hue: '#7ef0c8',
    icon: [
      '..........##',
      '.........##.',
      '........##..',
      '.......##...',
      '..#...##....',
      '.###.##.....',
      '..####......',
      '...##.......',
    ],
    samples: [
      {
        kind: 'text',
        label: 'Self-correct',
        text: 'so um i need to like send the the report by uh friday no wait make that thursday',
      },
      {
        kind: 'text',
        label: 'Numbers',
        text: 'the invoice came to twenty three thousand four hundred and fifty dollars and it\'s due on march third twenty twenty six',
      },
      {
        kind: 'text',
        label: 'Meeting',
        text: 'let\'s meet at half past two tomorrow uh actually make it three fifteen p m',
      },
      {
        kind: 'text',
        label: 'Email',
        text: 'hey sarah just wanted to follow up on the proposal can you send the numbers by end of week thanks john',
        extra: 'email',
      },
      {
        kind: 'text',
        label: 'Packing list',
        text: 'so for the trip we need to pack sunscreen and then also a first aid kit and um chargers for everything',
        extra: 'lists',
      },
      {
        kind: 'text',
        label: 'Casual',
        text: 'hmm im gonna be late theres a cute dog outside i cant just walk past him',
      },
    ],
    bench: { webgpu: '~1–2 s / sentence', wasm: '~6–10 s / sentence' },
    notes: [
      'S1-mini by Superwhisper (keep that exact name). Fine-tuned from Qwen3-0.6B to do one job: normalize English ASR output. It is not a chat model.',
      'Pass `styling` (`casual` · `semi-casual` · `semi-formal` · `formal`), `structure` (`prose` · `lists`) and `context` (`general` · `email`). Defaults are semi-formal prose.',
      'English only. Keep a single pass under ~1,000 tokens; chunk longer transcripts at sentence boundaries.',
      'Filler-only input (`um`, `uh`) returns an empty string. That is a valid result, not a failure.',
      'WebGPU prefers q4f16 (~339 MB) and falls back to q4 (~385 MB) if the session fails to start. WASM uses q4.',
    ],
    weights: {
      id: 'onnx-community/s1-mini-ONNX',
      baseModel: 'S1-mini',
      author: 'Superwhisper',
      license: 'Apache-2.0',
      sourceUrl: 'https://huggingface.co/superwhisper/s1-mini',
      size: '~339 MB / ~385 MB',
      runsOn: 'q4f16 on WebGPU · q4 on WASM',
    },
    usageSnippet: `import { TranscriptCleaner } from 'runonweb/clean'

const cleaner = new TranscriptCleaner({
  styling: 'semi-formal',
  onProgress: (info) => console.log(info.status, info.progress),
})

await cleaner.load()

const { text } = await cleaner.clean(
  'so um i need to like send the the report by uh friday no wait make that thursday',
  { onPartial: (partial) => console.log(partial) },
)
// "I need to send the report by Thursday."

cleaner.dispose()`,
  },
  {
    slug: 'emoji',
    name: 'Emojify',
    task: 'Text-to-Emoji',
    tagline: 'Turn a sentence into emojis. 4 MB model.',
    description:
      'Translate any English sentence into a short emoji sequence with a 2.4M-parameter model trained by runonweb. Loads in under a second, runs in a few milliseconds and never leaves the tab.',
    packagePath: 'runonweb/emoji',
    input: 'text',
    status: 'beta',
    webgpu: 'none',
    hue: '#ffd83d',
    icon: [
      '....####....',
      '..##....##..',
      '.#..#..#..#.',
      '.#........#.',
      '.#.#....#.#.',
      '..#.####.#..',
      '...##..##...',
      '....####....',
    ],
    samples: [
      { kind: 'text', label: 'Pizza night', text: 'I love pizza and my dog' },
      { kind: 'text', label: 'Birthday', text: "Happy birthday! Let's party tonight" },
      { kind: 'text', label: 'Rainy day', text: 'It is raining and I forgot my umbrella' },
      { kind: 'text', label: 'Beach', text: 'I am so tired of work, I need a vacation on the beach' },
    ],
    bench: { wasm: '10 ms / sentence' },
    notes: [
      'Trained from scratch: a 3-layer T5 (d_model 128, 8k shared vocab) on ~490k text/emoji pairs from the Text2Emoji dataset. Recipe in `training/text2emoji`.',
      'English input. Output is a short sequence of distinct emojis; `maxEmojis` caps the length (default 12).',
      'Weights download from the Hugging Face Hub (`midudev/text2emoji-tiny`). To self-host, copy the folder and pass `modelPath` plus `model` (the folder name).',
    ],
    weights: {
      id: 'midudev/text2emoji-tiny',
      baseModel: 'text2emoji-tiny (T5, 2.4M params, from scratch)',
      author: 'runonweb · data: Text2Emoji (KomeijiForce)',
      license: 'MIT',
      sourceUrl: 'https://huggingface.co/datasets/KomeijiForce/Text2Emoji',
      size: '~4 MB',
      runsOn: 'q8 on WASM',
    },
    usageSnippet: `import { Emojifier } from 'runonweb/emoji'

// 3.9 MB from the Hugging Face Hub (midudev/text2emoji-tiny)
const emojifier = new Emojifier()
await emojifier.load()

const { text, emojis } = await emojifier.emojify('I love pizza and my dog')
// text   "🍕❤️🐶"
// emojis ["🍕", "❤️", "🐶"]

emojifier.dispose()`,
  },
  {
    slug: 'tts',
    name: 'Voice',
    task: 'Text-to-Speech',
    tagline: 'Natural speech, generated locally.',
    description:
      'Synthesize speech from text on-device. Three sizes: Kokoro 82M for English, Spanish and French, Supertonic 2 for English, Korean, Spanish, Portuguese and French at 44.1 kHz, or KittenTTS nano at ~28 MB. Stream PCM as it is generated, or download a WAV.',
    packagePath: 'runonweb/tts',
    input: 'text',
    status: 'beta',
    webgpu: 'recommended',
    hue: '#e4e4e7',
    icon: [
      '....#.......',
      '...##...+...',
      '.####.+..+..',
      '.####..+..+.',
      '.####..+..+.',
      '.####.+..+..',
      '...##...+...',
      '....#.......',
    ],
    samples: [
      { kind: 'text', label: 'Greeting', text: 'Hello from the browser. This voice was generated on your device.' },
      { kind: 'text', label: 'Weather', text: 'Tomorrow will be partly cloudy with a high of twenty two degrees and light wind from the west.' },
      { kind: 'text', label: 'Countdown', text: 'Ten. Nine. Eight. Seven. Six. Five. Four. Three. Two. One. Liftoff.' },
      { kind: 'text', label: 'Español', text: 'Hola desde el navegador. Esta voz se generó en tu dispositivo, sin enviar nada a un servidor.', extra: 'es' },
      { kind: 'text', label: 'Português', text: 'Olá do navegador. Esta voz foi gerada no seu dispositivo, sem enviar nada para um servidor.', extra: 'pt' },
      { kind: 'text', label: 'Français', text: 'Bonjour depuis le navigateur. Cette voix a été générée sur votre appareil, rien n’a quitté la page.', extra: 'fr' },
      { kind: 'text', label: '한국어', text: '브라우저에서 인사드립니다. 이 목소리는 서버로 아무것도 보내지 않고 기기에서 생성되었습니다.', extra: 'ko' },
    ],
    bench: { webgpu: '0.7 s / sentence', wasm: '~3 s / sentence' },
    notes: [
      'Default `small` is Kokoro 82M (StyleTTS 2): American, British, Spanish and French. Runs on Transformers.js directly, no `kokoro-js`. fp32 on WebGPU (fp16 gives NaN on the v4 runtime), q8 on WASM.',
      'Pass `size: "multi"` for Supertonic 2 (Supertone, 2026): one ~262 MB model for English, Korean, Spanish, Portuguese and French at 44.1 kHz, 10 preset voices (`st_f1`…`st_m5`). Pass `language` per call.',
      'Supertonic 2 is OpenRAIL-M: open weights with use restrictions (no impersonation, no deception, no illegal use). The rest of the TTS sizes are Apache-2.0.',
      'Pass `size: "tiny"` for KittenTTS nano (~28 MB, 8 English voices, WASM).',
      'Audio streams sentence by sentence via `speakStream`. WebGPU is several times faster on `small` and `multi`; WASM works everywhere.',
      'Pass `voice` to pick a speaker (`af_heart` / `bella` / `st_f1`). Kokoro Spanish and French download a local eSpeak-NG WASM (~18 MB) on first use. Nothing is uploaded.',
    ],
    weights: {
      id: 'onnx-community/Kokoro-82M-v1.0-ONNX',
      baseModel: 'Kokoro 82M',
      author: 'hexgrad',
      license: 'Apache-2.0',
      sourceUrl: 'https://huggingface.co/hexgrad/Kokoro-82M',
      size: '~326 MB / ~92 MB (small) · multi ~262 MB · tiny ~28 MB',
      runsOn: 'fp32 on WebGPU · q8 on WASM (multi is fp32, tiny is WASM)',
    },
    usageSnippet: `import { TextToSpeech } from 'runonweb/tts'

const tts = new TextToSpeech({ size: 'small', voice: 'af_heart' })
await tts.load()

for await (const chunk of tts.speakStream('Hello from the browser')) {
  // chunk.audio is 24 kHz PCM. Play as it arrives
}

const wav = await tts.speakToBlob('Hola mundo', { voice: 'ef_dora' })
new Audio(URL.createObjectURL(wav)).play()

// Supertonic 2: one model, five languages, 44.1 kHz
const multi = new TextToSpeech({ size: 'multi', voice: 'st_f1' })
const pt = await multi.speakToBlob('Olá do navegador', { language: 'pt' })

tts.dispose()`,
  },
  {
    slug: 'image',
    name: 'Imagine',
    task: 'Image generation',
    tagline: 'A picture from a sentence. On your GPU.',
    description:
      'Generate images from a text prompt entirely in the tab. Bonsai Image 4B runs on WebGPU. No API key, no upload, nothing leaves the device after the first download.',
    packagePath: 'runonweb/image',
    input: 'text',
    status: 'beta',
    webgpu: 'required',
    hue: '#ff9f1c',
    icon: [
      '############',
      '#..........#',
      '#....##....#',
      '#...####...#',
      '#..#.##.#..#',
      '#.#......#.#',
      '#.########.#',
      '############',
    ],
    samples: [
      { kind: 'text', label: 'Bonsai', text: 'A small bonsai tree in a quiet ceramic studio, soft morning light, photorealistic' },
      { kind: 'text', label: 'Cat', text: 'A fluffy orange cat sitting on a sunlit windowsill, shallow depth of field, 35mm photo' },
      { kind: 'text', label: 'Street', text: 'A rainy neon street at night, reflections on wet asphalt, cinematic cyberpunk' },
      { kind: 'text', label: 'Lake', text: 'A still alpine lake at dawn, mist over the water, pine forest and distant peaks' },
    ],
    bench: { webgpu: '~10–40 s · 512²' },
    notes: [
      'Bonsai Image 4B (Prism ML, 2026) is a 1-bit / 1.58-bit FLUX.2 Klein deployment. WebGPU only. There is no WASM path.',
      'Default `binary` is the smaller payload (~3.4 GB). Pass `size: "ternary"` (~3.9 GB) for the quality-oriented weights.',
      'Tuned for 4 FlowMatch-Euler steps at guidance 1.0. More steps rarely help and can add artifacts. Negative prompts are not used.',
      'Default output is 512×512. Native training resolution is 1024×1024; sides must be multiples of 16 (32 recommended).',
      'First download is several gigabytes and is cached in IndexedDB. Chromium + a recent GPU is required.',
    ],
    weights: {
      id: 'prism-ml/bonsai-image-binary-4B-mlx-1bit',
      baseModel: 'Bonsai Image 4B (FLUX.2 Klein)',
      author: 'Prism ML',
      license: 'Apache-2.0',
      sourceUrl: 'https://huggingface.co/prism-ml/bonsai-image-binary-4B-mlx-1bit',
      size: '~3.4 GB (binary) · ternary ~3.9 GB',
      runsOn: '1-bit / 1.58-bit on WebGPU',
    },
    usageSnippet: `import { ImageGenerator } from 'runonweb/image'

const gen = new ImageGenerator({ size: 'binary' })
await gen.load()

const { image, seed } = await gen.generate(
  'A bonsai tree in a quiet ceramic studio, soft morning light',
)
document.querySelector('img').src = URL.createObjectURL(image)

gen.dispose()`,
  },
]

export function getModel(slug: string): Model | undefined {
  return models.find((m) => m.slug === slug)
}

export const modelsByInput = (input: ModelInput) => models.filter((m) => m.input === input)

/** First download figure only ("~150 MB"), for compact cards and readouts. */
export const shortSize = (m: Model): string => m.weights.size.match(/~?\d+(?:\.\d+)?\s?[MG]B/)?.[0] ?? m.weights.size
