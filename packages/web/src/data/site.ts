import pkg from '../../../runonweb/package.json' with { type: 'json' }

/** Site-wide SEO constants. Keep in sync with `site` in astro.config.mjs. */
export const SITE = {
  name: 'runonweb',
  url: 'https://runonweb.ai',
  title: 'runonweb: Free AI models that run in the browser',
  /** ≤155 chars so search results show it whole. */
  description:
    'Free, open-source AI models that run in your browser: speech-to-text, OCR, background removal, translation, TTS and more. WebGPU or WASM, fully private.',
  locale: 'en_US',
  license: 'MIT',
  /** Published version of the `runonweb` package. */
  version: pkg.version,
  npm: 'https://www.npmjs.com/package/runonweb',
} as const

/** Compose a page title. Home keeps the full brand title; other pages get " · runonweb" appended. */
export function pageTitle(title?: string): string {
  if (!title) return SITE.title
  return title.endsWith(SITE.name) ? title : `${title} · ${SITE.name}`
}

/** Absolute URL for a path. */
export function absoluteUrl(path: string): string {
  return new URL(path, SITE.url).toString()
}
