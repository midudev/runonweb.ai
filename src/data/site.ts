/** Site-wide SEO constants. Keep in sync with `site` in astro.config.mjs. */
export const SITE = {
  name: 'runonweb',
  url: 'https://runonweb.ai',
  title: 'runonweb: Free AI models that run in the browser',
  description:
    'Free, open-source AI modules that run in the browser: speech-to-text, transcript cleanup, background removal, captioning, depth, object detection, OCR, embeddings, translation, TTS and image generation. WebGPU or WASM, nothing leaves the device.',
  locale: 'en_US',
  license: 'MIT',
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
