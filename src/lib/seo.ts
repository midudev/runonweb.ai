/** Per-page SEO: titles, descriptions and JSON-LD built from the model catalog. */
import { models, INPUT_LABEL, shortSize, type Model } from '../data/models'
import { SITE, absoluteUrl } from '../data/site'

export function modelFaq(model: Model): Array<{ q: string; a: string }> {
  return [
    {
      q: `Does ${model.name} run on device?`,
      a: 'Yes. Weights download once and stay in the browser. Nothing is sent to a server.',
    },
    {
      q: `What does ${model.name} cost?`,
      a: 'Nothing. runonweb is free and MIT-licensed, with no device limits, tokens or sign-in.',
    },
    {
      q: `Which browsers support ${model.name}?`,
      a:
        model.webgpu === 'none'
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

/** "Scribe: Speech-to-Text in the browser with Whisper tiny.en" — task and base model are the search terms. */
export function modelTitle(model: Model): string {
  return `${model.name}: ${model.task} in the browser with ${model.weights.baseModel}`
}

export function modelDescription(model: Model): string {
  const backend =
    model.webgpu === 'none' ? 'WebAssembly' : model.webgpu === 'required' ? 'WebGPU' : 'WebGPU with WASM fallback'
  const d = `${model.description} Free, ${model.weights.license}, ${shortSize(model).replace('~', '')} download, ${backend}.`
  return d.length <= 160 ? d : model.description
}

export function modelJsonLd(model: Model): Array<Record<string, unknown>> {
  const url = absoluteUrl(`/models/${model.slug}`)
  const index = models.findIndex((m) => m.slug === model.slug)
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
      softwareVersion: '0.0.1',
      license: `https://spdx.org/licenses/${SITE.license}.html`,
      isAccessibleForFree: true,
      offers: { '@type': 'Offer', price: '0', priceCurrency: 'USD' },
      isBasedOn: {
        '@type': 'SoftwareSourceCode',
        name: model.weights.baseModel,
        url: model.weights.sourceUrl,
        author: { '@type': 'Organization', name: model.weights.author },
        license: model.weights.license,
      },
      image: absoluteUrl(`/og/model-${model.slug}.png`),
      publisher: { '@id': `${SITE.url}/#organization` },
      keywords: [model.task, INPUT_LABEL[model.input], model.weights.baseModel, 'browser AI', 'WebGPU', 'Transformers.js'].join(', '),
    },
    {
      '@type': 'BreadcrumbList',
      '@id': `${url}#breadcrumb`,
      itemListElement: [
        { '@type': 'ListItem', position: 1, name: 'Home', item: SITE.url },
        { '@type': 'ListItem', position: 2, name: 'Models', item: absoluteUrl('/models') },
        { '@type': 'ListItem', position: 3, name: `${String(index + 1).padStart(2, '0')} ${model.name}`, item: url },
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
    ogImageAlt: `${model.name} — ${model.task} in the browser, built on ${model.weights.baseModel}`,
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
        name: `${m.name} — ${m.task}`,
        url: absoluteUrl(`/models/${m.slug}`),
      })),
    },
    {
      '@type': 'BreadcrumbList',
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
      license: `https://spdx.org/licenses/${SITE.license}.html`,
      isAccessibleForFree: true,
      offers: { '@type': 'Offer', price: '0', priceCurrency: 'USD' },
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
      description: 'Install runonweb and use any module in a few lines. API for every model.',
      url,
      inLanguage: 'en',
      author: { '@id': `${SITE.url}/#organization` },
      publisher: { '@id': `${SITE.url}/#organization` },
      about: models.map((m) => ({ '@type': 'Thing', name: `${m.name} — ${m.task}`, url: absoluteUrl(`/models/${m.slug}`) })),
      proficiencyLevel: 'Beginner',
    },
    {
      '@type': 'BreadcrumbList',
      itemListElement: [
        { '@type': 'ListItem', position: 1, name: 'Home', item: SITE.url },
        { '@type': 'ListItem', position: 2, name: 'Docs', item: url },
      ],
    },
  ]
}
