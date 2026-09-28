/** Per-page SEO: titles, descriptions and JSON-LD built from the model catalog. */
import { models, INPUT_LABEL, shortSize, type Model } from '../data/models'
import { SITE, absoluteUrl } from '../data/site'

/**
 * Hand-written search copy per model. Titles lead with what people type
 * ("speech-to-text in the browser", "remove background") and stay ≤52 chars so
 * " · runonweb" still fits in a result; descriptions stay ≤155 chars.
 */
const MODEL_SEO: Record<string, { title: string; description: string }> = {
  ocr: {
    title: 'Free OCR in the browser: image to text with PP-OCRv6',
    description:
      'Extract text from photos, receipts and documents in your browser with PP-OCRv6. 50 languages, three sizes, free, and no image is ever uploaded.',
  },
  'remove-bg': {
    title: 'Remove image background in the browser, free',
    description:
      'Remove the background from any photo in your browser and download a transparent PNG. BEN2 runs locally: free, unlimited, no upload.',
  },
  stt: {
    title: 'Speech-to-text in the browser: free Whisper WebGPU',
    description:
      'Transcribe audio to text in your browser with Whisper on WebGPU. Record or drop a file: free, unlimited, and no audio leaves your device.',
  },
  caption: {
    title: 'AI image captions and alt text in the browser',
    description:
      'Generate image captions and alt text in 9 languages in your browser with LFM2.5-VL. Free, runs on WebGPU, and no image is uploaded.',
  },
  depth: {
    title: 'Depth Anything V2 in the browser: free depth maps',
    description:
      'Turn any photo into a depth map in your browser with Depth Anything V2. Free, runs on WebGPU or WASM, and the image never leaves your device.',
  },
  detect: {
    title: 'Object detection in the browser with RF-DETR',
    description:
      'Detect and label 80 kinds of objects with bounding boxes in your browser. RF-DETR Nano, ~29 MB, free and fully on-device.',
  },
  embed: {
    title: 'Text embeddings in the browser: MiniLM on WebGPU',
    description:
      'Create 384-dimension sentence embeddings for semantic search in your browser with all-MiniLM-L6-v2. ~23 MB, free, no text sent anywhere.',
  },
  classify: {
    title: 'Zero-shot text classification in the browser',
    description:
      'Classify text in your browser: yes/no, multiple-choice and rating questions answered with calibrated probabilities. Free, private, on-device.',
  },
  translate: {
    title: 'Offline translation in the browser: 58 languages',
    description:
      'Translate between English and 58 languages, or between two of them through English, in your browser with Firefox Translations models. Free, private, and offline after the first load.',
  },
  clean: {
    title: 'AI transcript cleanup in the browser, free',
    description:
      'Clean speech-to-text transcripts in your browser: drop filler words, fix self-corrections, format numbers and punctuate. Free and local.',
  },
  emoji: {
    title: 'Text to emoji AI in the browser: a 4 MB model',
    description:
      'Turn any English sentence into emojis in your browser with a 4 MB text-to-emoji model. Instant, free, and nothing leaves the tab.',
  },
  image: {
    title: 'Text-to-image in the browser: FLUX.2 Klein WebGPU',
    description:
      'Generate images from text prompts in your browser with Bonsai Image 4B (FLUX.2 Klein) on WebGPU. No API key, no upload, free.',
  },
  tts: {
    title: 'Text-to-speech in the browser: free Kokoro TTS',
    description:
      'Turn text into natural speech in your browser with Kokoro, Supertonic 2 or KittenTTS. English, Spanish, French and more. Free, on-device.',
  },
}

export function modelFaq(model: Model): Array<{ q: string; a: string }> {
  return [
    {
      q: `Does ${model.name} run on device?`,
      a: 'Yes. Weights download once and stay in the browser. Nothing is sent to a server.',
    },
    {
      q: `Which model does ${model.name} use?`,
      a: `${model.weights.baseModel} by ${model.weights.author}, under ${model.weights.license}. runonweb loads the \`${model.weights.id}\` weights and runs ${model.weights.runsOn}.`,
    },
    {
      q: `How big is the ${model.name} download?`,
      a: `${model.weights.size}, fetched once on first use. The browser caches the weights, so later loads skip the network and keep working offline.`,
    },
    {
      q: `What does ${model.name} cost?`,
      a: 'Nothing. runonweb is free and MIT-licensed, with no device limits, tokens or sign-in.',
    },
    {
      q: `Which browsers support ${model.name}?`,
      a:
        model.slug === 'remove-bg'
          ? 'Any modern browser with WebAssembly. BEN2’s LayerNorm shader fails on the current ONNX Runtime WebGPU build, so runonweb pins this model to WASM.'
          : model.webgpu === 'none'
            ? 'Any modern browser with WebAssembly. This model does not use WebGPU yet: it fails at session creation in ONNX Runtime Web, so runonweb pins it to WASM.'
          : model.webgpu === 'required'
            ? 'A Chromium browser with WebGPU and a recent GPU. This model has no WebAssembly fallback.'
            : 'Chromium browsers use WebGPU. Firefox and Safari fall back to WebAssembly automatically, slower but identical output.',
    },
    {
      q: `Can I use a different model with ${model.packagePath}?`,
      a: 'Yes. See each module’s docs for the options it accepts (`model`, `size`, language…). Check the license and test both backends before shipping.',
    },
  ]
}

const stripTicks = (s: string) => s.replace(/`/g, '')

const BACKEND_REQUIREMENT: Record<Model['webgpu'], string> = {
  required: 'A Chromium browser with WebGPU',
  recommended: 'WebGPU for speed; any modern browser with WebAssembly works',
  optional: 'Any modern browser with WebAssembly; WebGPU when available',
  none: 'Any modern browser with WebAssembly',
}

export const CATALOG_DESCRIPTION =
  'All runonweb models: speech, vision and text AI that runs in your browser. Live demo, copy-paste code, timings and license for each one.'

export const DOCS_DESCRIPTION =
  'Install runonweb and run AI models in the browser with a few lines of JavaScript. API, base model and notes for every speech, vision and text module.'

/** Search title without the brand suffix. Falls back to "Speech-to-Text in the browser with Whisper tiny.en". */
export function modelTitle(model: Model): string {
  return MODEL_SEO[model.slug]?.title ?? `${model.task} in the browser with ${model.weights.baseModel}`
}

export function modelDescription(model: Model): string {
  const seo = MODEL_SEO[model.slug]
  if (seo) return seo.description
  const backend =
    model.webgpu === 'none' ? 'WebAssembly' : model.webgpu === 'required' ? 'WebGPU' : 'WebGPU with WASM fallback'
  const d = `${model.description} Free, ${model.weights.license}, ${shortSize(model).replace('~', '')} download, ${backend}.`
  return d.length <= 160 ? d : model.description
}

export function modelJsonLd(model: Model): Array<Record<string, unknown>> {
  const url = absoluteUrl(`/models/${model.slug}`)
  return [
    {
      '@type': 'SoftwareApplication',
      '@id': `${url}#software`,
      name: `${model.name} (${model.packagePath})`,
      alternateName: model.packagePath,
      description: model.description,
      url,
      applicationCategory: 'DeveloperApplication',
      applicationSubCategory: model.task,
      operatingSystem: 'Web browser (WebGPU, WebAssembly)',
      softwareRequirements: BACKEND_REQUIREMENT[model.webgpu],
      softwareVersion: SITE.version,
      fileSize: shortSize(model).replace('~', ''),
      downloadUrl: SITE.npm,
      installUrl: SITE.npm,
      license: `https://spdx.org/licenses/${SITE.license}.html`,
      isAccessibleForFree: true,
      offers: { '@type': 'Offer', price: '0', priceCurrency: 'USD' },
      mainEntityOfPage: { '@id': `${url}#webpage` },
      isBasedOn: {
        '@type': 'SoftwareSourceCode',
        name: model.weights.baseModel,
        url: model.weights.sourceUrl,
        author: { '@type': 'Organization', name: model.weights.author },
        license: model.weights.license,
      },
      image: absoluteUrl(`/og/model-${model.slug}.png`),
      publisher: { '@id': `${SITE.url}/#organization` },
      keywords: [model.task, `${model.task} in the browser`, INPUT_LABEL[model.input], model.weights.baseModel, 'browser AI', 'on-device AI', 'WebGPU', 'WebAssembly', 'JavaScript'].join(', '),
    },
    {
      '@type': 'BreadcrumbList',
      '@id': `${url}#breadcrumb`,
      itemListElement: [
        { '@type': 'ListItem', position: 1, name: 'Home', item: SITE.url },
        { '@type': 'ListItem', position: 2, name: 'Models', item: absoluteUrl('/models') },
        { '@type': 'ListItem', position: 3, name: `${model.name}: ${model.task}`, item: url },
      ],
    },
    {
      '@type': 'FAQPage',
      '@id': `${url}#faq`,
      mainEntity: modelFaq(model).map(({ q, a }) => ({
        '@type': 'Question',
        name: q,
        acceptedAnswer: { '@type': 'Answer', text: stripTicks(a) },
      })),
    },
  ]
}

/** Everything a model page passes to <Layout>. */
export function modelSeo(model: Model) {
  return {
    title: modelTitle(model),
    description: modelDescription(model),
    ogImage: `/og/model-${model.slug}.png`,
    ogImageAlt: `${model.name}: ${model.task} in the browser, built on ${model.weights.baseModel}`,
    jsonLd: modelJsonLd(model),
  }
}

export function catalogJsonLd(): Array<Record<string, unknown>> {
  return [
    {
      '@type': 'ItemList',
      '@id': `${absoluteUrl('/models')}#list`,
      name: 'runonweb models',
      numberOfItems: models.length,
      itemListOrder: 'https://schema.org/ItemListOrderAscending',
      itemListElement: models.map((m, i) => ({
        '@type': 'ListItem',
        position: i + 1,
        name: `${m.name}: ${m.task}`,
        url: absoluteUrl(`/models/${m.slug}`),
      })),
    },
    {
      '@type': 'BreadcrumbList',
      '@id': `${absoluteUrl('/models')}#breadcrumb`,
      itemListElement: [
        { '@type': 'ListItem', position: 1, name: 'Home', item: SITE.url },
        { '@type': 'ListItem', position: 2, name: 'Models', item: absoluteUrl('/models') },
      ],
    },
  ]
}

export function homeJsonLd(): Array<Record<string, unknown>> {
  return [
    {
      '@type': 'SoftwareApplication',
      '@id': `${SITE.url}/#software`,
      name: SITE.name,
      description: SITE.description,
      url: SITE.url,
      applicationCategory: 'DeveloperApplication',
      operatingSystem: 'Web browser (WebGPU, WebAssembly)',
      softwareRequirements: 'A modern browser with WebAssembly; WebGPU for the fastest path',
      softwareVersion: SITE.version,
      downloadUrl: SITE.npm,
      installUrl: SITE.npm,
      license: `https://spdx.org/licenses/${SITE.license}.html`,
      isAccessibleForFree: true,
      offers: { '@type': 'Offer', price: '0', priceCurrency: 'USD' },
      image: absoluteUrl('/og/site.png'),
      publisher: { '@id': `${SITE.url}/#organization` },
      featureList: models.map((m) => `${m.task} (${m.packagePath})`).join(', '),
    },
    ...catalogJsonLd().slice(0, 1),
  ]
}

export function docsJsonLd(): Array<Record<string, unknown>> {
  const url = absoluteUrl('/docs')
  return [
    {
      '@type': 'TechArticle',
      '@id': `${url}#article`,
      headline: 'runonweb documentation',
      description: DOCS_DESCRIPTION,
      image: absoluteUrl('/og/docs.png'),
      mainEntityOfPage: { '@id': `${url}#webpage` },
      url,
      inLanguage: 'en',
      author: { '@id': `${SITE.url}/#organization` },
      publisher: { '@id': `${SITE.url}/#organization` },
      about: models.map((m) => ({ '@type': 'Thing', name: `${m.name}: ${m.task}`, url: absoluteUrl(`/models/${m.slug}`) })),
      proficiencyLevel: 'Beginner',
    },
    {
      '@type': 'BreadcrumbList',
      '@id': `${url}#breadcrumb`,
      itemListElement: [
        { '@type': 'ListItem', position: 1, name: 'Home', item: SITE.url },
        { '@type': 'ListItem', position: 2, name: 'Docs', item: url },
      ],
    },
  ]
}
