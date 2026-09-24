import { DepthEstimator } from 'runonweb/depth'
import { bindImagePicker, bindSamples, createDemoUI, createUrlPool, formatMs, onDemoPage } from './ui.ts'
import { fitCompareToImage, type CompareSlider } from './compare-slider.ts'

onDemoPage('dep', () => {
const ui = createDemoUI('dep')
const urls = createUrlPool()
const timingEl = ui.el('timing')
const fileInput = ui.el<HTMLInputElement>('file')
const dropzone = ui.el('dropzone')
const runBtn = ui.el<HTMLButtonElement>('run')
const downloadLink = ui.el<HTMLAnchorElement>('download')
const beforeImg = ui.el<HTMLImageElement>('before')
const afterImg = ui.el<HTMLImageElement>('after')
const previewRow = ui.el('preview')
const compare = ui.el<CompareSlider>('compare')

let estimator: DepthEstimator | null = null
let selectedFile: File | null = null

function setBusy(busy: boolean) {
  if (runBtn) runBtn.disabled = busy || !selectedFile
  if (fileInput) fileInput.disabled = busy
}

function showFile(file: File) {
  selectedFile = file
  urls.revokeAll()
  if (beforeImg) {
    beforeImg.src = urls.add(file)
    if (compare) fitCompareToImage(compare, beforeImg)
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
  if (estimator) return estimator
  const model = new DepthEstimator({ onProgress: ui.onProgress })
  try {
    await model.load()
  } catch (err) {
    ui.setStatus('Load failed')
    throw err
  }
  estimator = model
  if (model.device) ui.setDevice(`Using ${model.device}`)
  ui.setStatus('Model ready')
  ui.setProgress(null)
  return model
}

async function runDepth() {
  if (!selectedFile) {
    ui.setError('Select an image first')
    return
  }
  const file = selectedFile
  await ui.run(
    async () => {
      const model = await ensureModel()
      ui.setStatus('Estimating depth…')
      const t0 = performance.now()
      const { depth, width, height } = await model.estimate(file)
      if (timingEl) timingEl.textContent = `${formatMs(performance.now() - t0)} · ${width}×${height}`
      const url = urls.add(depth)
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
  void runDepth()
}

ui.initDeviceBadge()
runBtn?.addEventListener('click', () => void runDepth())
bindImagePicker({ dropzone, fileInput, onFile: pick })
bindSamples(ui, { onFile: pick })
return () => urls.revokeAll()
})
