import { ImageCaptioner, type CaptionDetail, type CaptionLanguage } from 'runonweb/caption'
import { bindImagePicker, bindSamples, createDemoUI, createUrlPool, formatMs, onDemoPage } from './ui.ts'

onDemoPage('cap', () => {
const ui = createDemoUI('cap')
const urls = createUrlPool()
const timingEl = ui.el('timing')
const fileInput = ui.el<HTMLInputElement>('file')
const dropzone = ui.el('dropzone')
const runBtn = ui.el<HTMLButtonElement>('run')
const imageEl = ui.el<HTMLImageElement>('image')
const resultEl = ui.el('result')
const previewRow = ui.el('preview')
const langSelect = ui.el<HTMLSelectElement>('lang')
const detailSelect = ui.el<HTMLSelectElement>('detail')

let captioner: ImageCaptioner | null = null
let selectedFile: File | null = null

function setBusy(busy: boolean) {
  if (runBtn) runBtn.disabled = busy || !selectedFile
  if (fileInput) fileInput.disabled = busy
  if (langSelect) langSelect.disabled = busy
  if (detailSelect) detailSelect.disabled = busy
}

function showFile(file: File) {
  selectedFile = file
  urls.revokeAll()
  if (imageEl) imageEl.src = urls.add(file)
  if (resultEl) resultEl.textContent = 'Caption will appear here…'
  if (previewRow) previewRow.hidden = false
  ui.setStatus(`Selected: ${file.name}`)
  ui.setError(null)
  if (runBtn) runBtn.disabled = false
}

async function ensureModel() {
  if (captioner) return captioner
  const model = new ImageCaptioner({ onProgress: ui.onProgress })
  try {
    await model.load()
  } catch (err) {
    ui.setStatus('Load failed')
    throw err
  }
  captioner = model
  if (model.device) ui.setDevice(`Using ${model.device}`)
  ui.setStatus('Model ready')
  ui.setProgress(null)
  return model
}

async function runCaption() {
  if (!selectedFile) {
    ui.setError('Select an image first')
    return
  }
  const file = selectedFile
  await ui.run(
    async () => {
      const model = await ensureModel()
      ui.setStatus('Captioning…')
      const t0 = performance.now()
      const { text } = await model.caption(file, {
        language: (langSelect?.value as CaptionLanguage) || 'en',
        detail: (detailSelect?.value as CaptionDetail) || 'short',
        onPartial: (partial) => {
          if (resultEl) resultEl.textContent = partial
        },
      })
      if (timingEl) timingEl.textContent = formatMs(performance.now() - t0)
      if (resultEl) resultEl.textContent = text || '(empty caption)'
      ui.setStatus('Done')
      ui.setProgress(null)
    },
    { busy: setBusy }
  )
}

/** Select a file and run immediately: one click, no separate load step. */
function pick(file: File) {
  showFile(file)
  void runCaption()
}

ui.initDeviceBadge()
runBtn?.addEventListener('click', () => void runCaption())
langSelect?.addEventListener('change', () => { if (selectedFile) void runCaption() })
detailSelect?.addEventListener('change', () => { if (selectedFile) void runCaption() })
bindImagePicker({ dropzone, fileInput, onFile: pick })
bindSamples(ui, { onFile: pick })
return () => urls.revokeAll()
})
