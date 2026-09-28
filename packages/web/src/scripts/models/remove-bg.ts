import { RemoveBackground } from 'runonweb/remove-bg'
import { bindImagePicker, bindSamples, createDemoUI, createUrlPool, formatMs, onDemoPage } from './ui.ts'
import { fitCompareToImage, type CompareSlider } from './compare-slider.ts'

onDemoPage('rbg', () => {
const ui = createDemoUI('rbg')
const urls = createUrlPool()
const timingEl = ui.el('timing')
const fileInput = ui.el<HTMLInputElement>('file')
const dropzone = ui.el('dropzone')
const runBtn = ui.el<HTMLButtonElement>('run')
const downloadLink = ui.el<HTMLAnchorElement>('download')
const beforeImg = ui.el<HTMLImageElement>('before')
const afterImg = ui.el<HTMLImageElement>('after')
const previewRow = ui.el('preview')
const stage = ui.el('stage')
const compare = ui.el<CompareSlider>('compare')

let remover: RemoveBackground | null = null
let selectedFile: File | null = null

function setBusy(busy: boolean) {
  if (runBtn) runBtn.disabled = busy || !selectedFile
  if (fileInput) fileInput.disabled = busy
  stage?.toggleAttribute('data-scanning', busy)
}

function showFile(file: File) {
  selectedFile = file
  urls.revokeAll()
  if (beforeImg) {
    beforeImg.src = urls.add(file)
    if (stage) fitCompareToImage(stage, beforeImg)
  }
  if (afterImg) afterImg.removeAttribute('src')
  if (compare) {
    compare.single = true
    compare.position = 100
  }
  if (downloadLink) downloadLink.hidden = true
  if (previewRow) previewRow.hidden = false
  ui.setStatus(`Selected: ${file.name}`)
  ui.setError(null)
  if (runBtn) runBtn.disabled = false
}

async function ensureModel() {
  if (remover) return remover
  const model = new RemoveBackground({ onProgress: ui.onProgress })
  try {
    await model.load()
  } catch (err) {
    ui.setStatus('Load failed')
    throw err
  }
  remover = model
  if (model.device) ui.setDevice(`Using ${model.device}`)
  ui.setStatus('Model ready')
  ui.setProgress(null)
  return model
}

async function runRemove() {
  if (!selectedFile) {
    ui.setError('Select an image first')
    return
  }
  const file = selectedFile
  await ui.run(
    async () => {
      const model = await ensureModel()
      ui.setStatus('Removing background…')
      const t0 = performance.now()
      const png = await model.remove(file)
      if (timingEl) timingEl.textContent = formatMs(performance.now() - t0)
      const url = urls.add(png)
      if (afterImg) afterImg.src = url
      if (compare) {
        compare.single = false
        compare.reveal(50, 600)
      }
      if (downloadLink) {
        downloadLink.href = url
        downloadLink.hidden = false
      }
      ui.setStatus('Done')
      ui.setProgress(null)
    },
    { busy: setBusy }
  )
}

/** Select a file and run immediately: one click, no separate load step. */
function pick(file: File) {
  showFile(file)
  void runRemove()
}

ui.initDeviceBadge()
runBtn?.addEventListener('click', () => void runRemove())
bindImagePicker({ dropzone, fileInput, onFile: pick })
bindSamples(ui, { onFile: pick })
return () => urls.revokeAll()
})
