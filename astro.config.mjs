// @ts-check
import { execFileSync } from 'node:child_process';
import { defineConfig, fontProviders } from 'astro/config';
import sitemap from '@astrojs/sitemap';

import tailwindcss from '@tailwindcss/vite';

/**
 * Files whose content ends up on a page, so <lastmod> only moves when the page does.
 * @param {string} pathname
 */
function pageSources(pathname) {
  const page = pathname === '/' ? 'src/pages/index.astro' : pathname === '/models' ? 'src/pages/models/index.astro' : `src/pages${pathname}.astro`
  return [page, 'src/data/models.ts', 'src/lib/seo.ts']
}

/**
 * Last commit date touching `files`, or undefined outside a git checkout.
 * @param {string[]} files
 */
function gitLastmod(files) {
  try {
    return execFileSync('git', ['log', '-1', '--format=%cI', '--', ...files], { encoding: 'utf8' }).trim() || undefined
  } catch {
    return undefined
  }
}

// https://astro.build/config
export default defineConfig({
  site: 'https://runonweb.ai',
  trailingSlash: 'never',
  // Self-hosted Geist family (latin subset only), served from the fontsource
  // packages. Astro hashes the files, emits preload links from <Font />, and
  // generates metric-matched fallbacks (size-adjust / ascent-override…) so the
  // system font that shows before the woff2 arrives takes up the same space.
  fonts: [
    {
      provider: fontProviders.local(),
      name: 'Geist',
      cssVariable: '--font-geist',
      fallbacks: ['system-ui'],
      options: {
        variants: [
          { src: ['@fontsource-variable/geist/files/geist-latin-wght-normal.woff2'], weight: '100 900', style: 'normal' },
        ],
      },
    },
    {
      provider: fontProviders.local(),
      name: 'Geist Mono',
      cssVariable: '--font-geist-mono',
      fallbacks: ['ui-monospace', 'monospace'],
      options: {
        variants: [
          { src: ['@fontsource-variable/geist-mono/files/geist-mono-latin-wght-normal.woff2'], weight: '100 900', style: 'normal' },
        ],
      },
    },
    {
      provider: fontProviders.local(),
      name: 'Geist Pixel',
      cssVariable: '--font-geist-pixel',
      fallbacks: ['ui-monospace', 'monospace'],
      options: {
        variants: [
          { src: ['@fontsource/geist-pixel/files/geist-pixel-latin-400-normal.woff2'], weight: '400', style: 'normal' },
        ],
      },
    },
  ],
  prefetch: {
    prefetchAll: true,
    defaultStrategy: 'hover',
  },
  integrations: [
    sitemap({
      filter: (page) => !page.includes('/og/') && !page.endsWith('.txt'),
      changefreq: 'weekly',
      priority: 0.7,
      serialize(item) {
        const url = new URL(item.url)
        item.lastmod = gitLastmod(pageSources(url.pathname))
        if (url.pathname === '/') item.priority = 1
        else if (url.pathname === '/models' || url.pathname.startsWith('/models/')) item.priority = 0.9
        return item
      },
    }),
  ],
  vite: {
    plugins: [tailwindcss()],
    optimizeDeps: {
      exclude: ['@huggingface/transformers', 'onnxruntime-web', 'paddleocr', 'phonemizer', 'runonweb/image'],
    },
  },
});
