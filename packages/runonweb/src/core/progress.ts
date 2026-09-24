export type ProgressInfo = {
  status: string
  progress?: number
  file?: string
}

export type ProgressCallback = (info: ProgressInfo) => void

/** Normalize Transformers.js progress callbacks into a simple shape. */
export function toProgressInfo(data: Record<string, unknown>): ProgressInfo {
  const status = typeof data.status === 'string' ? data.status : 'progress'
  const progress =
    typeof data.progress === 'number'
      ? data.progress
      : typeof data.loaded === 'number' && typeof data.total === 'number' && data.total > 0
        ? (data.loaded / data.total) * 100
        : undefined
  const file = typeof data.file === 'string' ? data.file : undefined
  return { status, progress, file }
}
