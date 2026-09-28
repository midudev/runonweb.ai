#!/usr/bin/env node
// Cloudflare Workers static assets reject files over 25 MiB.
// ONNX Runtime's jsep.wasm (~27 MiB) and Firefox Translations `.bin` weights (~30 MiB)
// are loaded from a CDN / Hugging Face at runtime, so they can be dropped from `dist/`.

import { readdir, rm, stat } from 'node:fs/promises'
import { join, relative } from 'node:path'
import { fileURLToPath } from 'node:url'

const LIMIT = 25 * 1024 * 1024
const DIST = fileURLToPath(new URL('../dist', import.meta.url))

async function walk(dir) {
  const entries = await readdir(dir, { withFileTypes: true })
  for (const entry of entries) {
    const path = join(dir, entry.name)
    if (entry.isDirectory()) {
      await walk(path)
      continue
    }
    const { size } = await stat(path)
    if (size <= LIMIT) continue
    console.log(`omit ${(size / 1024 / 1024).toFixed(1)} MiB  ${relative(DIST, path)}`)
    await rm(path)
  }
}

await walk(DIST)
