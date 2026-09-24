import { ImageGenerator, isImageGenerationSupported, type ImageSize } from 'runonweb/image'
import { createImagineStage } from './imagine-stage.ts'
import { bindSamples, createDemoUI, formatMs, onDemoPage } from './ui.ts'

onDemoPage('img', () => {
const ui = createDemoUI('img')
const timingEl = ui.el('timing')
const inputEl = ui.el<HTMLTextAreaElement>('input')
const runBtn = ui.el<HTMLButtonElement>('run')
const downloadLink = ui.el<HTMLAnchorElement>('download')
const preview = ui.el('preview')
const stageEl = ui.el('stage')
const stage = (() => {
  const canvas = ui.el<HTMLCanvasElement>('canvas')
  return canvas ? createImagineStage(canvas) : null
})()
const resultImg = ui.el<HTMLImageElement>('result')
const seedEl = ui.el('seed')
const sizeBtns = [...document.querySelectorAll<HTMLButtonElement>('[data-img-size]')]
const resBtns = [...document.querySelectorAll<HTMLButtonElement>('[data-img-res]')]

let generator: ImageGenerator | null = null
let loadedSize: ImageSize | null = null
let objectUrl: string | null = null

function currentSize(): ImageSize {
  const pressed = sizeBtns.find((b) => b.getAttribute('aria-pressed') === 'true')
  return (pressed?.dataset.imgSize as ImageSize | undefined) ?? 'binary'
}

function currentRes(): number {
  const pressed = resBtns.find((b) => b.getAttribute('aria-pressed') === 'true')
  return Number(pressed?.dataset.imgRes ?? 512)
}

function showStage(on: boolean) {
  if (on) {
    if (preview) preview.hidden = true
    if (stageEl) stageEl.hidden = false
    stage?.start()
    return
  }
  stage?.stop()
  if (stageEl) stageEl.hidden = true
}

function setBusy(busy: boolean) {
  if (runBtn) runBtn.disabled = busy
  for (const btn of sizeBtns) btn.disabled = busy
  for (const btn of resBtns) btn.disabled = busy
  showStage(busy)
}

function setPressed(buttons: HTMLButtonElement[], active: HTMLButtonElement) {
  for (const btn of buttons) btn.setAttribute('aria-pressed', String(btn === active))
}

function showImage(blob: Blob, seed: number) {
  showStage(false)
  if (objectUrl) URL.revokeObjectURL(objectUrl)
  objectUrl = URL.createObjectURL(blob)
  if (resultImg) resultImg.src = objectUrl
  if (preview) preview.hidden = false
  if (downloadLink) {
    downloadLink.href = objectUrl
    downloadLink.hidden = false
  }
  if (seedEl) seedEl.textContent = `seed ${seed}`
}

async function ensureModel() {
  const size = currentSize()
  if (generator && loadedSize === size) return generator

  generator?.dispose()
  generator = null
  loadedSize = null

  const model = new ImageGenerator({ size, width: currentRes(), height: currentRes(), onProgress: ui.onProgress })
  try {
    await model.load()
  } catch (err) {
    ui.setStatus('Load failed')
    throw err
  }
  generator = model
  loadedSize = size
  ui.setDevice('Using webgpu')
  ui.setStatus('Model ready')
  ui.setProgress(null)
  return model
}

async function runGenerate() {
  const prompt = inputEl?.value.trim() ?? ''
  if (!prompt) {
    ui.setError('Type a prompt first')
    return
  }

  if (!(await isImageGenerationSupported())) {
    ui.setError('WebGPU is required for image generation. Try Chrome or Edge on a recent GPU.')
    return
  }

  const side = currentRes()
  await ui.run(
    async () => {
      const model = await ensureModel()
      ui.setStatus(`Generating ${side}×${side}…`)
      const t0 = performance.now()
      const result = await model.generate(prompt, {
        width: side,
        height: side,
        onStep: ({ step, steps }) => {
          ui.setStatus(`Denoising ${step}/${steps}…`)
          stage?.setProgress(steps > 0 ? step / steps : 0)
        },
      })
      if (timingEl) timingEl.textContent = formatMs(performance.now() - t0)
      showImage(result.image, result.seed)
      ui.setStatus('Done')
      ui.setProgress(null)
    },
    { busy: setBusy }
  )
}

void isImageGenerationSupported().then((ok) => {
  if (ok) {
    ui.setDevice('WebGPU available')
    return
  }
  ui.setDevice('WebGPU unavailable')
  ui.setError('This model needs WebGPU. Open the page in Chrome or Edge on a machine with a recent GPU.')
  if (runBtn) runBtn.disabled = true
})

runBtn?.addEventListener('click', () => void runGenerate())
for (const btn of sizeBtns) {
  btn.addEventListener('click', () => {
    setPressed(sizeBtns, btn)
    ui.setStatus(`${btn.dataset.imgSize} selected. Generate to download weights`)
  })
}
for (const btn of resBtns) {
  btn.addEventListener('click', () => setPressed(resBtns, btn))
}
bindSamples(ui, {
  onText: (s) => {
    if (inputEl) inputEl.value = s.text
    ui.setError(null)
    ui.setStatus('Prompt ready. Press Generate')
    inputEl?.focus()
  },
})

return () => {
  if (objectUrl) URL.revokeObjectURL(objectUrl)
  stage?.stop()
  generator?.dispose()
  generator = null
}
})
