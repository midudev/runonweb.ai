/** /llms-full.txt: the docs page as one Markdown file, for AI assistants that answer "how do I…" questions. */
import type { APIRoute } from 'astro'
import { models, INSTALL_SNIPPET } from '../data/models'
import { SITE, absoluteUrl } from '../data/site'
import { PATTERN_SNIPPET, CORE_SNIPPET } from '../data/docs'

export const GET: APIRoute = () => {
  const body = `# ${SITE.name} documentation

> ${SITE.description}

Source: ${absoluteUrl('/docs')}

## Install

\`\`\`bash
${INSTALL_SNIPPET}
\`\`\`

Requires a bundler that understands package \`exports\` (Vite, Next, Astro, Bun…). Weights are fetched at runtime and cached by the browser.

## The pattern: load, run, dispose

\`\`\`ts
${PATTERN_SNIPPET}
\`\`\`

Each module also exports a one-shot function that loads, runs and disposes in one call. Use the class when you will run more than once.

## Core: WebGPU detection

\`\`\`ts
${CORE_SNIPPET}
\`\`\`

${models
  .map(
    (m) => `## ${m.name}: ${m.task} (\`${m.packagePath}\`)

${m.description}

Live demo: ${absoluteUrl(`/models/${m.slug}`)}

\`\`\`ts
${m.usageSnippet}
\`\`\`

- Base model: ${m.weights.baseModel} by ${m.weights.author} (${m.weights.license}), ${m.weights.sourceUrl}
- Download: ${m.weights.size}
- Runs on: ${m.weights.runsOn}
${m.notes.map((n) => `- ${n}`).join('\n')}`,
  )
  .join('\n\n')}
`
  return new Response(body, { headers: { 'Content-Type': 'text/plain; charset=utf-8' } })
}
