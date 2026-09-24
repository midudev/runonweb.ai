import { isWebGPUAvailable } from 'runonweb/core'
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
  const progressEl = el<HTMLProgressElement>('progress')
  const deviceEl = el('device')
  const errorEl = el('error')

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
      if (!progressEl) return
      if (value == null) {
        progressEl.hidden = true
        return
      }
      progressEl.hidden = false
      progressEl.value = Math.max(0, Math.min(100, value))
    },
    setDevice(text) {
      if (deviceEl) deviceEl.textContent = text
    },
    onProgress(info) {
      if (typeof info.progress === 'number') {
        ui.setProgress(info.progress)
        ui.setStatus(`${info.status}${info.file ? `: ${info.file}` : ''} (${Math.round(info.progress)}%)`)
      } else {
        ui.setStatus(info.status)
      }
    },
    initDeviceBadge() {
      void isWebGPUAvailable().then((ok) => {
        if (deviceEl && !deviceEl.textContent) {
          deviceEl.textContent = ok ? 'WebGPU available' : 'WebGPU unavailable — will use WASM'
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
    /** Fires synchronously on click, before any fetch — use it to start audio playback. */
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
