/** Snippets shared by /docs and /llms-full.txt. */
export const PATTERN_SNIPPET = `const m = new Module({ onProgress })
await m.load()          // downloads + caches weights, safe to call twice
await m.<task>(input)   // the actual work
m.dispose()             // free memory`

export const CORE_SNIPPET = `import { isWebGPUAvailable, resolveDevice } from 'runonweb/core'

const ok = await isWebGPUAvailable()
const device = await resolveDevice() // 'webgpu' | 'wasm'`
