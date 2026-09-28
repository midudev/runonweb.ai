import type { APIRoute, GetStaticPaths } from 'astro'
import { models } from '../../data/models'
import { SITE } from '../../data/site'
import { renderOg, type OgSpec } from '../../lib/og'

const pages: Record<string, OgSpec> = {
  site: { kind: 'site' },
  models: {
    kind: 'page',
    eyebrow: `${String(models.length).padStart(2, '0')} modules`,
    title: 'Models',
    description: 'Speech, vision and text models that run in the browser. Live demos, code, base model and license.',
  },
  docs: {
    kind: 'page',
    eyebrow: 'Documentation',
    title: 'Docs',
    description: `Install ${SITE.name} and use any module in a few lines. API, base models and notes for every task.`,
  },
}

export const getStaticPaths: GetStaticPaths = () => [
  ...Object.keys(pages).map((slug) => ({ params: { slug } })),
  ...models.map((m) => ({ params: { slug: `model-${m.slug}` } })),
]

export const GET: APIRoute = async ({ params }) => {
  const slug = params.slug ?? ''
  let spec = pages[slug]
  if (!spec && slug.startsWith('model-')) {
    const index = models.findIndex((m) => m.slug === slug.slice('model-'.length))
    if (index >= 0) spec = { kind: 'model', model: models[index], index }
  }
  if (!spec) return new Response('Not found', { status: 404 })

  const png = await renderOg(spec)
  return new Response(new Uint8Array(png) as Uint8Array<ArrayBuffer>, {
    headers: { 'Content-Type': 'image/png', 'Cache-Control': 'public, max-age=31536000, immutable' },
  })
}
