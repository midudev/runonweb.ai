import {
  clearAllModelCache,
  clearModelCache,
  formatBytes,
  isWebGPUAvailable,
  listCachedModels,
  storageEstimate,
  type CachedModel,
} from 'runonweb/core'

const panel = document.querySelector<HTMLElement>('[data-storage-panel]')
const toggle = document.querySelector<HTMLButtonElement>('[data-storage-toggle]')
const dot = document.querySelector<HTMLElement>('[data-storage-dot]')
const list = document.querySelector<HTMLUListElement>('[data-storage-list]')
const count = document.querySelector<HTMLElement>('[data-storage-count]')
const total = document.querySelector<HTMLElement>('[data-storage-total]')
const note = document.querySelector<HTMLElement>('[data-storage-note]')
const clearBtn = document.querySelector<HTMLButtonElement>('[data-storage-clear]')
const backendEl = document.querySelector<HTMLElement>('[data-storage-backend]')
const backendLed = document.querySelector<HTMLElement>('[data-storage-backend-led]')

let armed = false
let armTimer: ReturnType<typeof setTimeout> | null = null

function li(html: string) {
  const el = document.createElement('li')
  el.innerHTML = html
  return el
}

function render(models: CachedModel[]) {
  if (!list) return
  list.replaceChildren()
  const bytes = models.reduce((n, m) => n + m.bytes, 0)
  if (count) count.textContent = String(models.length).padStart(2, '0')
  if (total) total.textContent = bytes ? formatBytes(bytes) : '0 B'
  if (dot) dot.hidden = models.length === 0
  if (clearBtn) clearBtn.disabled = models.length === 0

  if (models.length === 0) {
    list.appendChild(li('<div class="px-4 py-6 text-center"><span class="readout text-fg-3">No models cached yet</span></div>'))
    return
  }

  for (const m of models) {
    const row = li(`
      <div class="flex items-center gap-3 px-4 py-2.5">
        <div class="min-w-0 flex-1">
          <a class="block truncate font-mono text-[12.5px] text-fg hover:underline underline-offset-4" target="_blank" rel="noopener"></a>
          <span class="readout mt-1 block text-fg-3"></span>
        </div>
        <button type="button" class="btn btn-soft btn-sm btn-noled" aria-label="Delete cached files">Delete</button>
      </div>`)
    const link = row.querySelector('a')!
    link.href = `https://huggingface.co/${m.id}`
    link.textContent = m.id
    row.querySelector('span')!.textContent = `${m.files.length} files · ${formatBytes(m.bytes)}`
    row.querySelector('button')!.addEventListener('click', async (e) => {
      const btn = e.currentTarget as HTMLButtonElement
      btn.disabled = true
      btn.textContent = 'Deleting…'
      await clearModelCache(m.id)
      await refresh()
    })
    list.appendChild(row)
  }
}

async function refresh() {
  try {
    const [models, est] = await Promise.all([listCachedModels(), storageEstimate()])
    render(models)
    if (note && est) {
      note.textContent = `Origin uses ${formatBytes(est.usage)} of ${formatBytes(est.quota)}.`
    }
  } catch (err) {
    if (list) {
      list.replaceChildren(
        li(`<div class="px-4 py-4"><span class="readout text-danger">${err instanceof Error ? err.message : String(err)}</span></div>`)
      )
    }
  }
}

function disarm() {
  armed = false
  if (clearBtn) clearBtn.textContent = 'Delete all data'
  if (armTimer) clearTimeout(armTimer)
}

clearBtn?.addEventListener('click', async () => {
  if (!armed) {
    // Two-step confirm without a blocking dialog.
    armed = true
    clearBtn.textContent = 'Confirm delete'
    armTimer = setTimeout(disarm, 4000)
    return
  }
  disarm()
  clearBtn.disabled = true
  clearBtn.textContent = 'Deleting…'
  await clearAllModelCache()
  await refresh()
  clearBtn.textContent = 'Delete all data'
})

panel?.addEventListener('toggle', (e) => {
  const open = (e as ToggleEvent).newState === 'open'
  if (open) void refresh()
  else disarm()
})

toggle?.addEventListener('click', () => {
  // Fallback for browsers without the popover API.
  if (panel && typeof panel.togglePopover !== 'function') {
    panel.hidden = !panel.hidden
    if (!panel.hidden) void refresh()
  }
})

void isWebGPUAvailable().then((ok) => {
  if (backendLed) backendLed.style.setProperty('--hue', ok ? '#3dff7a' : 'var(--color-danger)')
  if (backendEl) backendEl.textContent = ok ? 'WebGPU' : 'WASM'
})

// Light the dot without opening the panel when something is already cached.
void listCachedModels().then((models) => {
  if (dot) dot.hidden = models.length === 0
})
