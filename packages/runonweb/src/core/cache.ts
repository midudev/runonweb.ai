/**
 * Inspect and clear the model weights Transformers.js stores in the browser's Cache Storage.
 * Everything runonweb downloads lives in the `transformers-cache` bucket, keyed by the
 * Hugging Face URL, so a model id maps to a set of cached requests.
 */

export const CACHE_NAME = 'transformers-cache'

/** Cache Storage bucket used by `runonweb/translate` for Firefox Translations models. */
export const TRANSLATE_CACHE_NAME = 'runonweb-translate'

/** Cache Storage bucket used by Bonsai Image (range-request leftovers). */
export const BONSAI_CACHE_NAME = 'bonsai-pipeline-v1'
/** IndexedDB used by the Bonsai Image WebGPU engine for safetensor chunks. */
export const BONSAI_IDB_NAME = 'bonsai-image-v1'

const EXTRA_CACHE_NAMES = [BONSAI_CACHE_NAME, TRANSLATE_CACHE_NAME]

export type CachedFile = {
  url: string
  /** Path inside the repo, e.g. `onnx/encoder_model_fp16.onnx`. */
  path: string
  bytes: number
}

export type CachedModel = {
  /** Hugging Face repo id, e.g. `onnx-community/whisper-tiny.en`. */
  id: string
  files: CachedFile[]
  bytes: number
}

const HF_RE = /^https?:\/\/[^/]+\/(?:datasets\/)?([^/]+\/[^/]+)\/resolve\/[^/]+\/(.+)$/
/** `<base>/<from>-<to>/<model|lex|vocab file>?rev=…` as written by `runonweb/translate`. */
const TRANSLATE_RE = /\/([a-z]{2,3}(?:-Han[st])?-[a-z]{2,3}(?:-Han[st])?)\/((?:model|lex|vocab|srcvocab|trgvocab)\.[^/?]+)(?:\?.*)?$/

/** Split a cached request URL into a model id and a path inside it. */
function identify(url: string): { id: string; path: string } {
  const t = TRANSLATE_RE.exec(url)
  if (t) return { id: `firefox-translations/${t[1]}`, path: t[2] }
  const m = HF_RE.exec(url)
  return { id: m?.[1] ?? 'other', path: m?.[2] ?? url }
}

function cachesAvailable(): boolean {
  return typeof caches !== 'undefined'
}

async function responseSize(res: Response): Promise<number> {
  const len = res.headers.get('content-length')
  if (len && Number(len) > 0) return Number(len)
  try {
    return (await res.clone().blob()).size
  } catch {
    return 0
  }
}

async function collectCacheBucket(cacheName: string, byModel: Map<string, CachedModel>): Promise<void> {
  if (!cachesAvailable()) return
  let cache: Cache
  try {
    cache = await caches.open(cacheName)
  } catch {
    return
  }
  const requests = await cache.keys()
  for (const req of requests) {
    const { id, path } = identify(req.url)
    const res = await cache.match(req)
    const bytes = res ? await responseSize(res) : 0
    const entry = byModel.get(id) ?? { id, files: [], bytes: 0 }
    entry.files.push({ url: req.url, path, bytes })
    entry.bytes += bytes
    byModel.set(id, entry)
  }
}

async function idbExists(name: string): Promise<boolean> {
  if (typeof indexedDB === 'undefined' || !indexedDB.databases) return false
  try {
    const dbs = await indexedDB.databases()
    return dbs.some((db) => db.name === name)
  } catch {
    return false
  }
}

/** Sum Bonsai Image chunks stored in IndexedDB (the engine's real weight cache). */
async function listBonsaiIndexedDb(): Promise<CachedModel | null> {
  if (typeof indexedDB === 'undefined') return null
  if (!(await idbExists(BONSAI_IDB_NAME))) return null

  return new Promise((resolve) => {
    const req = indexedDB.open(BONSAI_IDB_NAME)
    req.onerror = () => resolve(null)
    req.onsuccess = () => {
      const db = req.result
      if (!db.objectStoreNames.contains('chunks')) {
        db.close()
        resolve(null)
        return
      }
      const names = [...db.objectStoreNames]
      const tx = db.transaction(names, 'readonly')
      let bytes = 0
      let id = 'prism-ml/bonsai-image'

      if (names.includes('meta')) {
        const keysReq = tx.objectStore('meta').getAllKeys()
        keysReq.onsuccess = () => {
          for (const key of keysReq.result) {
            const m = HF_RE.exec(String(key))
            if (m?.[1]) {
              id = m[1]
              break
            }
          }
        }
      }

      const cursorReq = tx.objectStore('chunks').openCursor()
      cursorReq.onsuccess = () => {
        const cursor = cursorReq.result
        if (!cursor) return
        const value = cursor.value as unknown
        if (value instanceof Blob) bytes += value.size
        else if (value instanceof ArrayBuffer) bytes += value.byteLength
        cursor.continue()
      }

      tx.oncomplete = () => {
        db.close()
        resolve(
          bytes > 0
            ? { id, files: [{ url: `indexeddb://${BONSAI_IDB_NAME}`, path: 'chunks', bytes }], bytes }
            : null
        )
      }
      tx.onerror = () => {
        db.close()
        resolve(null)
      }
    }
  })
}

/** List every model cached in this origin, largest first. */
export async function listCachedModels(): Promise<CachedModel[]> {
  const byModel = new Map<string, CachedModel>()
  await collectCacheBucket(CACHE_NAME, byModel)
  for (const name of EXTRA_CACHE_NAMES) {
    await collectCacheBucket(name, byModel)
  }

  const bonsai = await listBonsaiIndexedDb()
  if (bonsai) {
    const existing = byModel.get(bonsai.id)
    if (existing) {
      existing.files.push(...bonsai.files)
      existing.bytes += bonsai.bytes
    } else {
      byModel.set(bonsai.id, bonsai)
    }
  }

  return [...byModel.values()].sort((a, b) => b.bytes - a.bytes)
}

async function deleteIndexedDb(name: string): Promise<boolean> {
  if (typeof indexedDB === 'undefined') return false
  if (!(await idbExists(name))) return false
  return new Promise((resolve) => {
    const req = indexedDB.deleteDatabase(name)
    req.onsuccess = () => resolve(true)
    req.onerror = () => resolve(false)
    req.onblocked = () => resolve(true)
  })
}

async function clearCacheBucketForModel(cacheName: string, modelId: string): Promise<number> {
  if (!cachesAvailable()) return 0
  let cache: Cache
  try {
    cache = await caches.open(cacheName)
  } catch {
    return 0
  }
  let removed = 0
  for (const req of await cache.keys()) {
    if (identify(req.url).id === modelId && (await cache.delete(req))) removed++
  }
  return removed
}

/** Delete the cached files of one model. Returns the number of entries removed. */
export async function clearModelCache(modelId: string): Promise<number> {
  let removed = await clearCacheBucketForModel(CACHE_NAME, modelId)
  for (const name of EXTRA_CACHE_NAMES) {
    removed += await clearCacheBucketForModel(name, modelId)
  }
  if (modelId.includes('bonsai-image') && (await deleteIndexedDb(BONSAI_IDB_NAME))) {
    removed += 1
  }
  return removed
}

/** Delete every cached model. Returns the number of entries removed. */
export async function clearAllModelCache(): Promise<number> {
  let removed = 0
  if (cachesAvailable()) {
    for (const name of [CACHE_NAME, ...EXTRA_CACHE_NAMES]) {
      try {
        const cache = await caches.open(name)
        removed += (await cache.keys()).length
        await caches.delete(name)
      } catch {
        // Bucket may not exist.
      }
    }
  }
  if (await deleteIndexedDb(BONSAI_IDB_NAME)) removed += 1
  return removed
}

/** Bytes used / available for this origin, when the browser reports it. */
export async function storageEstimate(): Promise<{ usage: number; quota: number } | null> {
  try {
    const est = await navigator.storage?.estimate?.()
    if (!est) return null
    return { usage: est.usage ?? 0, quota: est.quota ?? 0 }
  } catch {
    return null
  }
}

/** Human-readable bytes: 23 MB, 1.2 GB. */
export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  const units = ['KB', 'MB', 'GB']
  let v = bytes / 1024
  let i = 0
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024
    i++
  }
  return `${v < 10 ? v.toFixed(1) : Math.round(v)} ${units[i]}`
}
