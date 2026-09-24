import { OCR, OCR_SIZES, type OCRLine, type OCRSize } from 'runonweb/ocr'
import { bindImagePicker, bindSamples, createDemoUI, createUrlPool, formatMs, onDemoPage } from './ui.ts'

onDemoPage('ocr', () => {
const ui = createDemoUI('ocr')
const urls = createUrlPool()
const timingEl = ui.el('timing')
const fileInput = ui.el<HTMLInputElement>('file')
const dropzone = ui.el('dropzone')
const runBtn = ui.el<HTMLButtonElement>('run')
const imageEl = ui.el<HTMLImageElement>('image')
const overlay = ui.el<SVGSVGElement>('overlay')
const resultEl = ui.el('result')
const previewRow = ui.el('preview')
const sizeButtons = [...document.querySelectorAll<HTMLButtonElement>('[data-ocr-size]')]

let size: OCRSize = 'small'
let ocr: OCR | null = null
let selectedFile: File | null = null

function setBusy(busy: boolean) {
  if (runBtn) runBtn.disabled = busy || !selectedFile
  if (fileInput) fileInput.disabled = busy
  for (const btn of sizeButtons) btn.disabled = busy
}

function clearResults() {
  if (overlay) overlay.replaceChildren()
  if (resultEl) resultEl.textContent = 'Text will appear here…'
}

function showFile(file: File) {
  selectedFile = file
  urls.revokeAll()
  clearResults()
  if (imageEl) imageEl.src = urls.add(file)
  if (previewRow) previewRow.hidden = false
  ui.setStatus(`Selected: ${file.name}`)
  ui.setError(null)
  if (runBtn) runBtn.disabled = false
}

function markSize(next: OCRSize) {
  size = next
  for (const btn of sizeButtons) {
    btn.setAttribute('aria-pressed', btn.dataset.ocrSize === next ? 'true' : 'false')
  }
}

async function ensureModel() {
  if (ocr && ocr.size === size) return ocr
  ocr?.dispose()
  ocr = null
  const model = new OCR({ size, onProgress: ui.onProgress })
  try {
    await model.load()
  } catch (err) {
    ui.setStatus('Load failed')
    throw err
  }
  ocr = model
  if (model.device) ui.setDevice(`Using ${model.device} · ${OCR_SIZES[size].label}`)
  ui.setStatus('Model ready')
  ui.setProgress(null)
  return model
}

function draw(lines: OCRLine[]) {
  if (!overlay || !imageEl) return
  overlay.replaceChildren()
  const w = imageEl.naturalWidth
  const h = imageEl.naturalHeight
  overlay.setAttribute('viewBox', `0 0 ${w} ${h}`)
  overlay.setAttribute('preserveAspectRatio', 'none')
  const stroke = Math.max(2, Math.round(w / 280))
  const ns = 'http://www.w3.org/2000/svg'
  for (const line of lines) {
    const { xmin, ymin, xmax, ymax } = line.box
    const rect = document.createElementNS(ns, 'rect')
    rect.setAttribute('x', String(xmin))
    rect.setAttribute('y', String(ymin))
    rect.setAttribute('width', String(Math.max(1, xmax - xmin)))
    rect.setAttribute('height', String(Math.max(1, ymax - ymin)))
    rect.setAttribute('fill', 'rgb(200 255 74 / 0.18)')
    rect.setAttribute('stroke', '#c8ff4a')
    rect.setAttribute('stroke-width', String(stroke))
    overlay.appendChild(rect)
  }
}

async function runRead() {
  if (!selectedFile) {
    ui.setError('Select an image first')
    return
  }
  const file = selectedFile
  await ui.run(
    async () => {
      const model = await ensureModel()
      ui.setStatus('Reading…')
      const t0 = performance.now()
      const { text, lines } = await model.read(file)
      if (timingEl) timingEl.textContent = formatMs(performance.now() - t0)
      if (resultEl) resultEl.textContent = text || '(no text found)'
      draw(lines)
      ui.setStatus(`Done · ${lines.length} line${lines.length === 1 ? '' : 's'}`)
      ui.setProgress(null)
    },
    { busy: setBusy }
  )
}

function pick(file: File) {
  showFile(file)
  void runRead()
}

ui.initDeviceBadge()
runBtn?.addEventListener('click', () => void runRead())
bindImagePicker({ dropzone, fileInput, onFile: pick })
bindSamples(ui, { onFile: pick })
for (const btn of sizeButtons) {
  btn.addEventListener('click', () => {
    const next = btn.dataset.ocrSize as OCRSize | undefined
    if (!next || next === size) return
    markSize(next)
    ocr?.dispose()
    ocr = null
    ui.setStatus(`Size: ${OCR_SIZES[next].label} ${OCR_SIZES[next].downloadMB}`)
    if (selectedFile) void runRead()
  })
}
return () => urls.revokeAll()
})
