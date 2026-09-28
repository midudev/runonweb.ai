import { formatBytes, isWebGPUAvailable } from 'runonweb/core'
import type { ProgressInfo } from 'runonweb/core'
import { onPage } from '../lifecycle'

/**
 * Small helper shared by every demo page. Binds to `[data-<prefix>-*]` elements:
 * status, progress, device, error.
 */
export type DemoUI = {
  el<T extends Element = HTMLElement>(name: string): T | null
  setStatus(text: string): void
  setError(message: string | null): void
  setProgress(value: number | null): void
  setDevice(text: string): void
  /** Pass straight to a runonweb module's `onProgress`. */
  onProgress(info: ProgressInfo): void
  /** Show "WebGPU available" until a model resolves its device. */
  initDeviceBadge(): void
  /** Wrap an async action: clears error, reports failure, restores state. */
  run<T>(action: () => Promise<T>, opts?: { busy?: (b: boolean) => void }): Promise<T | undefined>
}

type SpeedState = {
  files: Map<string, number>
  useTotal: boolean
  lastBytes: number
  lastTime: number
  speed: number
}

function resetSpeed(state: SpeedState) {
  state.files.clear()
  state.useTotal = false
  state.lastBytes = 0
  state.lastTime = 0
  state.speed = 0
}

/** Smoothed bytes/sec from progress events. `progress_total` is the aggregate and wins over per-file counts. */
function noteDownloadSpeed(state: SpeedState, info: ProgressInfo, now: number): number | null {
  if (typeof info.loaded !== 'number' || !Number.isFinite(info.loaded)) return state.speed || null
  let bytes: number | null = null
  if (info.status === 'progress_total') {
    state.useTotal = true
    bytes = info.loaded
  } else if (!state.useTotal && info.file) {
    state.files.set(info.file, info.loaded)
    bytes = 0
    for (const n of state.files.values()) bytes += n
  } else if (!state.useTotal) {
    bytes = info.loaded
  }
  if (bytes == null) return state.speed || null
  if (state.lastTime === 0) {
    state.lastTime = now
    state.lastBytes = bytes
    return null
  }
  const dt = (now - state.lastTime) / 1000
  if (dt < 0.25) return state.speed || null
  const delta = bytes - state.lastBytes
  if (delta <= 0) {
    state.lastBytes = bytes
    state.lastTime = now
    return state.speed || null
  }
  const instant = delta / dt
  state.speed = state.speed === 0 ? instant : state.speed * 0.65 + instant * 0.35
  state.lastBytes = bytes
  state.lastTime = now
  return state.speed
}

/** Overlay the logo loader on a keycap button. */
export function setButtonLoading(btn: HTMLElement | null | undefined, loading: boolean) {
  if (!btn) return
  if (loading) btn.setAttribute('aria-busy', 'true')
  else btn.removeAttribute('aria-busy')
}

/** Bind a demo only when its page is on screen; dispose when leaving. */
export function onDemoPage(prefix: string, init: () => void | (() => void)) {
  onPage(() => {
    if (!document.querySelector(`[data-${prefix}-status]`)) return
    return init()
  })
}

export function createDemoUI(prefix: string): DemoUI {
  const el = <T extends Element = HTMLElement>(name: string) =>
    document.querySelector<T>(`[data-${prefix}-${name}]`)

  const statusEl = el('status')
  const progressRow = el('progress-row')
  const progressEl = el<HTMLProgressElement>('progress')
  const speedEl = el('speed')
  const deviceEl = el('device')
  const errorEl = el('error')
  const speedState: SpeedState = {
    files: new Map(),
    useTotal: false,
    lastBytes: 0,
    lastTime: 0,
    speed: 0,
  }

  const setSpeed = (bytesPerSec: number | null) => {
    if (!speedEl) return
    if (bytesPerSec == null || bytesPerSec < 1) {
      speedEl.hidden = true
      speedEl.textContent = ''
      return
    }
    speedEl.hidden = false
    speedEl.textContent = `${formatBytes(bytesPerSec)}/s`
  }

  const ui: DemoUI = {
    el,
    setStatus(text) {
      if (statusEl) statusEl.textContent = text
    },
    setError(message) {
      if (!errorEl) return
      errorEl.textContent = message ?? ''
      errorEl.hidden = !message
    },
    setProgress(value) {
      if (value == null) {
        if (progressRow) progressRow.hidden = true
        if (progressEl) progressEl.hidden = true
        setSpeed(null)
        resetSpeed(speedState)
        return
      }
      if (progressRow) progressRow.hidden = false
      if (!progressEl) return
      progressEl.hidden = false
      progressEl.value = Math.max(0, Math.min(100, value))
    },
    setDevice(text) {
      if (deviceEl) deviceEl.textContent = text
    },
    onProgress(info) {
      if (info.status === 'loading' && (info.progress ?? 0) === 0 && info.loaded == null) {
        resetSpeed(speedState)
        setSpeed(null)
      }
      const bytesPerSec = noteDownloadSpeed(speedState, info, performance.now())
      if (typeof info.progress === 'number') {
        ui.setProgress(info.progress)
        ui.setStatus(`${info.status}${info.file ? `: ${info.file}` : ''} (${Math.round(info.progress)}%)`)
        setSpeed(bytesPerSec)
        if (progressEl && bytesPerSec != null && bytesPerSec >= 1) {
          progressEl.setAttribute('aria-valuetext', `${Math.round(info.progress)}%, ${formatBytes(bytesPerSec)}/s`)
        }
      } else {
        ui.setStatus(info.status)
        if (bytesPerSec != null) setSpeed(bytesPerSec)
      }
    },
    initDeviceBadge() {
      void isWebGPUAvailable().then((ok) => {
        if (deviceEl && !deviceEl.textContent) {
          deviceEl.textContent = ok ? 'WebGPU available' : 'WebGPU unavailable. Will use WASM'
        }
      })
    },
    async run(action, opts) {
      const loadingBtn = el<HTMLButtonElement>('run') ?? el<HTMLButtonElement>('transcribe')
      setButtonLoading(loadingBtn, true)
      opts?.busy?.(true)
      ui.setError(null)
      try {
        return await action()
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err)
        ui.setError(message)
        ui.setStatus('Error')
        return undefined
      } finally {
        opts?.busy?.(false)
        setButtonLoading(loadingBtn, false)
      }
    },
  }

  return ui
}

/** Track object URLs so they can be revoked together. */
export function createUrlPool() {
  let urls: string[] = []
  return {
    add(blob: Blob): string {
      const url = URL.createObjectURL(blob)
      urls.push(url)
      return url
    },
    revokeAll() {
      for (const url of urls) URL.revokeObjectURL(url)
      urls = []
    },
  }
}

/** Wire a dropzone + hidden file input. Calls `onFile` with image files only. */
export function bindImagePicker(opts: {
  dropzone: HTMLElement | null
  fileInput: HTMLInputElement | null
  onFile: (file: File) => void
}) {
  const { dropzone, fileInput, onFile } = opts

  fileInput?.addEventListener('change', () => {
    const file = fileInput.files?.[0]
    if (file) onFile(file)
  })

  if (!dropzone) return

  for (const evt of ['dragenter', 'dragover']) {
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault()
      dropzone.dataset.drag = 'true'
    })
  }
  for (const evt of ['dragleave', 'drop']) {
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault()
      dropzone.dataset.drag = 'false'
    })
  }
  dropzone.addEventListener('drop', (e) => {
    const file = (e as DragEvent).dataTransfer?.files?.[0]
    if (file && file.type.startsWith('image/')) onFile(file)
  })
  dropzone.addEventListener('click', () => fileInput?.click())
}

/** Format milliseconds as a short human string. */
export function formatMs(ms: number): string {
  return ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(2)} s`
}

/** Sample shape mirrored from src/data/models.ts (kept loose to avoid importing Astro-side data). */
export type SampleSpec =
  | { kind: 'image'; label: string; src: string }
  | { kind: 'audio'; label: string; src: string; duration: string; language?: string }
  | { kind: 'text'; label: string; text: string; extra?: string }

/**
 * Wire the "Try with" chips. Samples are embedded as JSON on the container so the
 * script does not need to import the catalog. Media samples are fetched and handed
 * over as a File; text samples pass through.
 */
export function bindSamples(
  ui: DemoUI,
  handlers: {
    onFile?: (file: File) => void
    onText?: (sample: Extract<SampleSpec, { kind: 'text' }>) => void
    /** Fires synchronously on click, before any fetch. Use it to start audio playback. */
    onSelect?: (spec: SampleSpec, btn: HTMLButtonElement) => void
  }
) {
  const root = ui.el('samples')
  if (!root) return
  let specs: SampleSpec[] = []
  try {
    specs = JSON.parse(root.dataset.specs ?? '[]') as SampleSpec[]
  } catch {
    return
  }
  const chips = root.querySelectorAll<HTMLButtonElement>('[data-sample]')
  const press = (active: HTMLButtonElement) => {
    chips.forEach((c) => {
      if (c.hasAttribute('aria-pressed')) c.setAttribute('aria-pressed', String(c === active))
    })
  }
  chips.forEach((btn) => {
    btn.addEventListener('click', async () => {
      const spec = specs[Number(btn.dataset.sample)]
      if (!spec) return
      press(btn)
      handlers.onSelect?.(spec, btn)
      if (spec.kind === 'text') {
        handlers.onText?.(spec)
        return
      }
      btn.disabled = true
      setButtonLoading(btn, true)
      ui.setError(null)
      ui.setStatus(`Loading sample: ${spec.label}…`)
      try {
        const res = await fetch(spec.src)
        if (!res.ok) throw new Error(`Could not load ${spec.src} (${res.status})`)
        const blob = await res.blob()
        const name = spec.src.split('/').pop() ?? 'sample'
        handlers.onFile?.(new File([blob], name, { type: blob.type }))
      } catch (err) {
        ui.setError(err instanceof Error ? err.message : String(err))
      } finally {
        btn.disabled = false
        setButtonLoading(btn, false)
      }
    })
  })
}
