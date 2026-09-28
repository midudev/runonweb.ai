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
const hits = ui.el('hits')
const resultEl = ui.el('result')
const previewRow = ui.el('preview')
const tip = ui.el('tip')
let activeHit: HTMLElement | null = null
const sizeButtons = [...document.querySelectorAll<HTMLButtonElement>('[data-ocr-size]')]

let size: OCRSize = 'small'
let ocr: OCR | null = null
let selectedFile: File | null = null
// Latest photo wins. Overlapping WebGPU OrtRuns make ONNX Runtime throw
// "Session mismatch", and a late result would box the previous photo.
let epoch = 0
let activeEpoch = 0
let job: Promise<void> = Promise.resolve()

function setBusy(busy: boolean) {
  if (runBtn) runBtn.disabled = busy || !selectedFile
  if (fileInput) fileInput.disabled = busy
  for (const btn of sizeButtons) btn.disabled = busy
}

function hideTip() {
  activeHit = null
  if (!tip) return
  tip.hidden = true
  tip.textContent = ''
}

function placeTip() {
  if (!tip || !activeHit || tip.hidden) return
  const box = activeHit.getBoundingClientRect()
  const tipBox = tip.getBoundingClientRect()
  const gap = 10
  let top = box.top - tipBox.height - gap
  if (top < 8) top = box.bottom + gap
  if (top + tipBox.height > window.innerHeight - 8) top = Math.max(8, window.innerHeight - tipBox.height - 8)
  let left = box.left + box.width / 2 - tipBox.width / 2
  left = Math.max(8, Math.min(left, window.innerWidth - tipBox.width - 8))
  tip.style.left = `${Math.round(left)}px`
  tip.style.top = `${Math.round(top)}px`
}

function showTip(hit: HTMLElement) {
  if (!tip) return
  activeHit = hit
  // The demo card clips overflow, so the tip has to leave that box.
  document.body.append(tip)
  tip.textContent = hit.dataset.text?.trim() || 'No text'
  tip.hidden = false
  placeTip()
}

function clearResults() {
  hideTip()
  if (overlay) overlay.replaceChildren()
  if (hits) hits.replaceChildren()
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
}

function markSize(next: OCRSize) {
  size = next
  for (const btn of sizeButtons) {
    btn.setAttribute('aria-pressed', btn.dataset.ocrSize === next ? 'true' : 'false')
  }
}

async function ensureModel() {
  if (ocr && ocr.size === size) return ocr
  const previous = ocr
  ocr = null
  await previous?.dispose()
  const model = new OCR({
    size,
    onProgress: (info) => {
      if (activeEpoch !== epoch) return
      ui.onProgress(info)
    },
  })
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

async function draw(lines: OCRLine[]) {
  if (!overlay || !hits || !imageEl) return
  const token = epoch
  const src = imageEl.src
  // OCR often finishes before the preview has decoded. naturalWidth is then 0,
  // or still the previous photo, and the hit boxes never line up. Wait for this src.
  try {
    await imageEl.decode()
  } catch {
    return
  }
  if (token !== epoch || imageEl.src !== src) return
  const w = imageEl.naturalWidth
  const h = imageEl.naturalHeight
  if (!w || !h) return
  hideTip()
  overlay.replaceChildren()
  hits.replaceChildren()
  overlay.setAttribute('viewBox', `0 0 ${w} ${h}`)
  overlay.setAttribute('preserveAspectRatio', 'none')
  const stroke = Math.max(2, Math.round(w / 280))
  const ns = 'http://www.w3.org/2000/svg'
  for (const line of lines) {
    const { xmin, ymin, xmax, ymax } = line.box
    const width = Math.max(1, xmax - xmin)
    const height = Math.max(1, ymax - ymin)
    const rect = document.createElementNS(ns, 'rect')
    rect.setAttribute('x', String(xmin))
    rect.setAttribute('y', String(ymin))
    rect.setAttribute('width', String(width))
    rect.setAttribute('height', String(height))
    rect.setAttribute('fill', 'rgb(200 255 74 / 0.18)')
    rect.setAttribute('stroke', '#c8ff4a')
    rect.setAttribute('stroke-width', String(stroke))
    overlay.appendChild(rect)

    const label = line.text.trim() || 'No text'
    const hit = document.createElement('button')
    hit.type = 'button'
    hit.className = 'ocr-hit'
    hit.dataset.text = line.text
    hit.setAttribute('aria-label', label)
    hit.style.left = `${(xmin / w) * 100}%`
    hit.style.top = `${(ymin / h) * 100}%`
    hit.style.width = `${(width / w) * 100}%`
    hit.style.height = `${(height / h) * 100}%`
    hit.addEventListener('pointerenter', () => showTip(hit))
    hit.addEventListener('focus', () => showTip(hit))
    hit.addEventListener('pointerleave', hideTip)
    hit.addEventListener('blur', hideTip)
    hits.appendChild(hit)
  }
}

function requestRead() {
  const token = ++epoch
  const file = selectedFile
  job = job
    .then(async () => {
      if (token !== epoch || !file) return
      activeEpoch = token
      await ui.run(
        async () => {
          if (token !== epoch) return
          const model = await ensureModel()
          if (token !== epoch) return
          ui.setStatus('Reading…')
          const t0 = performance.now()
          let text: string
          let lines: OCRLine[]
          try {
            ;({ text, lines } = await model.read(file))
          } catch (err) {
            if (token !== epoch) return
            throw err
          }
          if (token !== epoch) return
          if (timingEl) timingEl.textContent = formatMs(performance.now() - t0)
          if (resultEl) resultEl.textContent = text || '(no text found)'
          await draw(lines)
          if (token !== epoch) return
          ui.setStatus(`Done · ${lines.length} line${lines.length === 1 ? '' : 's'}`)
          ui.setProgress(null)
        },
        { busy: (busy) => { if (token === epoch) setBusy(busy) } },
      )
    })
    .then(
      () => undefined,
      () => undefined,
    )
}

function pick(file: File) {
  showFile(file)
  requestRead()
}

ui.initDeviceBadge()
runBtn?.addEventListener('click', () => {
  if (!selectedFile) {
    ui.setError('Select an image first')
    return
  }
  requestRead()
})
bindImagePicker({ dropzone, fileInput, onFile: pick })
bindSamples(ui, { onFile: pick })
for (const btn of sizeButtons) {
  btn.addEventListener('click', () => {
    const next = btn.dataset.ocrSize as OCRSize | undefined
    if (!next || next === size) return
    markSize(next)
    ui.setStatus(`Size: ${OCR_SIZES[next].label} ${OCR_SIZES[next].downloadMB}`)
    if (selectedFile) requestRead()
  })
}
window.addEventListener('scroll', placeTip, true)
window.addEventListener('resize', placeTip)
return () => {
  epoch += 1
  hideTip()
  tip?.remove()
  window.removeEventListener('scroll', placeTip, true)
  window.removeEventListener('resize', placeTip)
  urls.revokeAll()
  void ocr?.dispose()
}
})
