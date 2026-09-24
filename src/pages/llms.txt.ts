/** /llms.txt (llmstxt.org): a Markdown map of the site for AI assistants and answer engines. */
import type { APIRoute } from 'astro'
import { models, INPUT_LABEL, shortSize } from '../data/models'
import { SITE, absoluteUrl } from '../data/site'

export const GET: APIRoute = () => {
  const body = `# ${SITE.name}

> ${SITE.description}

runonweb is an MIT-licensed npm package (\`npm install runonweb\`) with one ES module per task. Each module picks WebGPU or WebAssembly, downloads open weights once, caches them in the browser, and returns plain JavaScript values. No server, API key or sign-in. Every module follows the same shape: \`new Module({ onProgress })\`, \`await m.load()\`, run the task, \`m.dispose()\`.

## Models

${models
  .map(
    (m) =>
      `- [${m.name}: ${m.task}](${absoluteUrl(`/models/${m.slug}`)}): ${m.description} Import \`${m.packagePath}\`. ${INPUT_LABEL[m.input]} input, ${m.weights.baseModel} by ${m.weights.author} (${m.weights.license}), ${shortSize(m)} download.`,
  )
  .join('\n')}

## Docs

- [Documentation](${absoluteUrl('/docs')}): install, the load/run/dispose pattern, WebGPU detection, and the API of every module.
- [Full docs as Markdown](${absoluteUrl('/llms-full.txt')}): every module's description, usage code and notes in one file.

## Optional

- [npm package](${SITE.npm}): \`runonweb\` v${SITE.version}.
- [All models](${absoluteUrl('/models')}): catalog with live demos, timings and licenses.
`
  return new Response(body, { headers: { 'Content-Type': 'text/plain; charset=utf-8' } })
}
