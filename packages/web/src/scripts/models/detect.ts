import { ObjectDetector, type Detection } from 'runonweb/detect'
import { bindImagePicker, bindSamples, createDemoUI, createUrlPool, formatMs, onDemoPage } from './ui.ts'

onDemoPage('det', () => {
const ui = createDemoUI('det')
const urls = createUrlPool()
const timingEl = ui.el('timing')
const fileInput = ui.el<HTMLInputElement>('file')
const dropzone = ui.el('dropzone')
const runBtn = ui.el<HTMLButtonElement>('run')
const thresholdInput = ui.el<HTMLInputElement>('threshold')
const thresholdValue = ui.el('threshold-value')
const imageEl = ui.el<HTMLImageElement>('image')
const videoEl = ui.el<HTMLVideoElement>('video')
const overlay = ui.el<SVGSVGElement>('overlay')
const listEl = ui.el<HTMLUListElement>('list')
const previewRow = ui.el('preview')
const stage = ui.el('stage')
const webcamBtn = () => ui.el<HTMLButtonElement>('webcam')

const COLORS = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#14b8a6', '#f97316']

let detector: ObjectDetector | null = null
let selectedFile: File | null = null
let threshold = 0.5
let lastDetections: Detection[] = []
let stream: MediaStream | null = null
let live = false
let starting = false
let liveToken = 0
const frameCanvas = document.createElement('canvas')
const frameCtx = frameCanvas.getContext('2d', { willReadFrequently: true })

function setBusy(busy: boolean) {
  if (runBtn) runBtn.disabled = busy || live || !selectedFile
  if (fileInput) fileInput.disabled = busy
}

function setLiveUi(on: boolean) {
  live = on
  const cam = webcamBtn()
  if (cam) {
    cam.toggleAttribute('data-live', on)
    cam.setAttribute('aria-pressed', String(on))
    cam.title = on ? 'Stop webcam' : 'Live webcam'
    cam.setAttribute('aria-label', on ? 'Stop webcam' : 'Live webcam')
  }
  if (stage) stage.toggleAttribute('data-mirror', on)
  if (imageEl) imageEl.classList.toggle('hidden', on)
  if (videoEl) videoEl.classList.toggle('hidden', !on)
  if (runBtn) runBtn.disabled = on || !selectedFile
}

function clearResults() {
  if (overlay) overlay.replaceChildren()
  if (listEl) listEl.replaceChildren()
}

function sourceSize(): { w: number; h: number } | null {
  if (live && videoEl?.videoWidth) return { w: videoEl.videoWidth, h: videoEl.videoHeight }
  if (imageEl?.naturalWidth) return { w: imageEl.naturalWidth, h: imageEl.naturalHeight }
  return null
}

function showFile(file: File) {
  stopWebcam()
  selectedFile = file
  lastDetections = []
  urls.revokeAll()
  clearResults()
  if (imageEl) imageEl.src = urls.add(file)
  if (previewRow) previewRow.hidden = false
  ui.setStatus(`Selected: ${file.name}`)
  ui.setError(null)
  if (runBtn) runBtn.disabled = false
}

function draw(detections: Detection[]) {
  const size = sourceSize()
  if (!overlay || !listEl || !size) return
  clearResults()
  const { w, h } = size
  overlay.setAttribute('viewBox', `0 0 ${w} ${h}`)
  overlay.setAttribute('preserveAspectRatio', 'none')
  const stroke = Math.max(2, Math.round(w / 300))
  const fontSize = Math.max(11, Math.round(w / 52))
  const ns = 'http://www.w3.org/2000/svg'

  detections.forEach((d, i) => {
    const color = COLORS[i % COLORS.length]!
    let { xmin, ymin, xmax, ymax } = d.box
    if (live) {
      const left = w - xmax
      xmax = w - xmin
      xmin = left
    }
    const rect = document.createElementNS(ns, 'rect')
    rect.setAttribute('x', String(xmin))
    rect.setAttribute('y', String(ymin))
    rect.setAttribute('width', String(xmax - xmin))
    rect.setAttribute('height', String(ymax - ymin))
    rect.setAttribute('fill', 'none')
    rect.setAttribute('stroke', color)
    rect.setAttribute('stroke-width', String(stroke))
    overlay.appendChild(rect)

    const text = `${d.label} ${(d.score * 100).toFixed(0)}%`
    const padX = fontSize * 0.45
    const chipH = fontSize * 1.5
    const chipW = text.length * fontSize * 0.64 + padX * 2
    const chipX = Math.min(Math.max(0, xmin), w - chipW)
    const chipY = ymin - chipH >= 0 ? ymin - chipH : ymin
    const chip = document.createElementNS(ns, 'rect')
    chip.setAttribute('x', String(chipX))
    chip.setAttribute('y', String(chipY))
    chip.setAttribute('width', String(chipW))
    chip.setAttribute('height', String(chipH))
    chip.setAttribute('fill', color)
    overlay.appendChild(chip)

    const label = document.createElementNS(ns, 'text')
    label.setAttribute('x', String(chipX + padX))
    label.setAttribute('y', String(chipY + fontSize * 1.1))
    label.setAttribute('fill', '#0a0a0b')
    label.setAttribute('font-size', String(fontSize))
    label.setAttribute('style', 'font-family: var(--font-pixel); text-transform: uppercase; letter-spacing: 0.04em')
    label.textContent = text
    overlay.appendChild(label)

    const li = document.createElement('li')
    li.className = 'border-[1.5px] px-2.5 py-0.5'
    li.style.borderColor = color
    li.style.color = color
    li.textContent = `${d.label} · ${(d.score * 100).toFixed(0)}%`
    listEl.appendChild(li)
  })

  if (detections.length === 0) {
    const li = document.createElement('li')
    li.className = 'text-fg-3'
    li.textContent = 'No objects above the threshold.'
    listEl.appendChild(li)
  }
}

async function ensureModel() {
  if (detector) return detector
  const model = new ObjectDetector({
    threshold: 0.1,
    onProgress: (info) => {
      if (info.status === 'processing' || info.status === 'done') return
      ui.onProgress(info)
    },
  })
  try {
    await model.load()
  } catch (err) {
    ui.setStatus('Load failed')
    throw err
  }
  detector = model
  if (model.device) ui.setDevice(`Using ${model.device}`)
  ui.setStatus('Model ready')
  ui.setProgress(null)
  return model
}

function grabFrame(): HTMLCanvasElement | null {
  if (!videoEl || !frameCtx || !videoEl.videoWidth) return null
  frameCanvas.width = videoEl.videoWidth
  frameCanvas.height = videoEl.videoHeight
  frameCtx.drawImage(videoEl, 0, 0)
  return frameCanvas
}

async function runDetect() {
  if (live) return
  if (!selectedFile) {
    ui.setError('Select an image first')
    return
  }
  const file = selectedFile
  await ui.run(
    async () => {
      const model = await ensureModel()
      ui.setStatus('Detecting…')
      const t0 = performance.now()
      lastDetections = await model.detect(file)
      const detections = lastDetections.filter((d) => d.score >= threshold)
      if (timingEl) timingEl.textContent = `${formatMs(performance.now() - t0)} · ${detections.length} objects`
      draw(detections)
      ui.setStatus('Done')
      ui.setProgress(null)
    },
    { busy: setBusy }
  )
}

async function loopLive(token: number) {
  while (live && token === liveToken) {
    const frame = grabFrame()
    if (!frame || !detector) {
      await new Promise((r) => setTimeout(r, 40))
      continue
    }
    const t0 = performance.now()
    try {
      lastDetections = await detector.detect(frame)
    } catch (err) {
      if (token !== liveToken) return
      ui.setError(err instanceof Error ? err.message : String(err))
      ui.setStatus('Error')
      stopWebcam()
      return
    }
    if (token !== liveToken) return
    const detections = lastDetections.filter((d) => d.score >= threshold)
    const dt = performance.now() - t0
    const fps = dt > 0 ? 1000 / dt : 0
    if (timingEl) timingEl.textContent = `${formatMs(dt)} · ${fps.toFixed(1)} fps · ${detections.length} objects`
    draw(detections)
    ui.setStatus('Live')
    ui.setProgress(null)
  }
}

function isAbortError(err: unknown): boolean {
  return err instanceof DOMException && err.name === 'AbortError'
}

function stopWebcam() {
  liveToken += 1
  live = false
  starting = false
  stream?.getTracks().forEach((t) => t.stop())
  stream = null
  if (videoEl) {
    videoEl.srcObject = null
    videoEl.removeAttribute('src')
    videoEl.load()
  }
  setLiveUi(false)
  if (!selectedFile && previewRow) previewRow.hidden = true
}

function cameraMessage(err: unknown): string {
  const name = err instanceof DOMException ? err.name : ''
  if (name === 'NotAllowedError' || name === 'PermissionDeniedError') {
    return 'Camera permission denied. Allow the camera for this site in the address bar and try again.'
  }
  if (name === 'NotFoundError' || name === 'DevicesNotFoundError') return 'No camera found on this device.'
  if (name === 'NotReadableError' || name === 'TrackStartError') {
    return 'Camera is busy in another app. Close it and try again.'
  }
  if (name === 'SecurityError') return 'Camera is blocked. Open the site on localhost or HTTPS.'
  if (name === 'AbortError') return 'Camera was interrupted. Try again.'
  return err instanceof Error ? err.message : String(err)
}

async function openCamera(): Promise<MediaStream> {
  const media = navigator.mediaDevices
  if (!window.isSecureContext || !media?.getUserMedia) {
    throw new Error('Camera needs a secure context (localhost or HTTPS).')
  }
  try {
    return await media.getUserMedia({ video: true, audio: false })
  } catch (err) {
    return await media.getUserMedia({
      video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } },
      audio: false,
    }).catch(() => {
      throw err
    })
  }
}

async function startWebcam() {
  if (starting || live) return
  starting = true
  const token = ++liveToken
  const cam = webcamBtn()
  if (cam) cam.disabled = true
  try {
    ui.setError(null)
    ui.setStatus('Requesting camera…')
    selectedFile = null
    lastDetections = []
    urls.revokeAll()
    clearResults()
    if (imageEl) imageEl.removeAttribute('src')
    if (runBtn) runBtn.disabled = true

    let media: MediaStream
    try {
      media = await openCamera()
    } catch (err) {
      ui.setError(cameraMessage(err))
      ui.setStatus('Error')
      return
    }
    if (token !== liveToken) {
      media.getTracks().forEach((t) => t.stop())
      return
    }

    stream = media
    if (!videoEl || !previewRow) {
      stopWebcam()
      ui.setError('Webcam preview is missing from the page.')
      ui.setStatus('Error')
      return
    }

    // Unhide first, then attach the stream. Playing a hidden video with
    // autoplay + play() is what triggers Chrome's pause-interrupts-play error.
    previewRow.hidden = false
    setLiveUi(true)
    videoEl.srcObject = media
    ui.setStatus('Starting camera…')

    try {
      if (videoEl.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
        await new Promise<void>((resolve, reject) => {
          const done = () => {
            videoEl.removeEventListener('loadeddata', done)
            videoEl.removeEventListener('error', fail)
            resolve()
          }
          const fail = () => {
            videoEl.removeEventListener('loadeddata', done)
            videoEl.removeEventListener('error', fail)
            reject(new Error('Camera stream failed to load'))
          }
          videoEl.addEventListener('loadeddata', done)
          videoEl.addEventListener('error', fail)
        })
      }
      if (token !== liveToken) return
      await videoEl.play()
    } catch (err) {
      if (token !== liveToken) return
      if (isAbortError(err)) {
        stopWebcam()
        return
      }
      stopWebcam()
      ui.setError(cameraMessage(err))
      ui.setStatus('Error')
      return
    }
    if (token !== liveToken) return

    ui.setStatus('Loading model…')
    try {
      await ensureModel()
    } catch (err) {
      if (token !== liveToken) return
      stopWebcam()
      ui.setError(err instanceof Error ? err.message : String(err))
      ui.setStatus('Error')
      return
    }
    if (token !== liveToken) return

    ui.setStatus('Live')
    void loopLive(token)
  } finally {
    starting = false
    if (cam) cam.disabled = false
  }
}

async function toggleWebcam() {
  if (starting) return
  if (live) {
    stopWebcam()
    ui.setStatus('Camera stopped')
    if (timingEl) timingEl.textContent = ''
    return
  }
  try {
    await startWebcam()
  } catch (err) {
    if (isAbortError(err)) return
    ui.setError(cameraMessage(err))
    ui.setStatus('Error')
  }
}

function pick(file: File) {
  showFile(file)
  void runDetect()
}

ui.initDeviceBadge()
runBtn?.addEventListener('click', () => void runDetect())
ui.el('samples')?.addEventListener('click', (event) => {
  const target = event.target
  if (!(target instanceof Element) || !target.closest('[data-det-webcam]')) return
  event.preventDefault()
  void toggleWebcam()
})
thresholdInput?.addEventListener('input', () => {
  threshold = Number(thresholdInput.value)
  if (thresholdValue) thresholdValue.textContent = threshold.toFixed(2)
  if (lastDetections.length && sourceSize()) draw(lastDetections.filter((d) => d.score >= threshold))
})
bindImagePicker({ dropzone, fileInput, onFile: pick })
bindSamples(ui, { onFile: pick })
return () => {
  stopWebcam()
  urls.revokeAll()
}
})
