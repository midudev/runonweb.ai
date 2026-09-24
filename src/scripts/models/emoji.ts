import { Emojifier } from 'runonweb/emoji'
import { bindSamples, createDemoUI, formatMs, onDemoPage } from './ui.ts'

onDemoPage('em', () => {
const ui = createDemoUI('em')
const timingEl = ui.el('timing')
const inputEl = ui.el<HTMLTextAreaElement>('input')
const resultEl = ui.el('result')
const runBtn = ui.el<HTMLButtonElement>('run')

let emojifier: Emojifier | null = null
let timer: ReturnType<typeof setTimeout> | null = null

function setBusy(busy: boolean) {
  if (runBtn) runBtn.disabled = busy
}

async function ensureModel() {
  if (emojifier) return emojifier
  // Self-hosted copy (4 MB, public/models/text2emoji-tiny); the SDK default is the Hub repo.
  const model = new Emojifier({ modelPath: '/models/', model: 'text2emoji-tiny', onProgress: ui.onProgress })
  try {
    await model.load()
  } catch (err) {
    ui.setStatus('Load failed')
    throw err
  }
  emojifier = model
  if (model.device) ui.setDevice(`Using ${model.device}`)
  ui.setStatus('Model ready')
  ui.setProgress(null)
  return model
}

async function runEmojify() {
  const text = inputEl?.value.trim() ?? ''
  if (!text) {
    ui.setError('Type a sentence first')
    return
  }

  await ui.run(
    async () => {
      const model = await ensureModel()
      ui.setStatus('Translating…')
      const t0 = performance.now()
      const result = await model.emojify(text)
      if (timingEl) timingEl.textContent = formatMs(performance.now() - t0)
      if (resultEl) resultEl.textContent = result.text || '(empty)'
      ui.setStatus('Done')
      ui.setProgress(null)
    },
    { busy: setBusy }
  )
}

ui.initDeviceBadge()
runBtn?.addEventListener('click', () => void runEmojify())
// Live mode: the model answers in milliseconds, so re-run shortly after typing stops.
inputEl?.addEventListener('input', () => {
  if (!emojifier) return
  if (timer) clearTimeout(timer)
  timer = setTimeout(() => void runEmojify(), 350)
})
bindSamples(ui, {
  onText: (s) => {
    if (inputEl) inputEl.value = s.text
    void runEmojify()
  },
})

return () => {
  if (timer) clearTimeout(timer)
  emojifier?.dispose()
  emojifier = null
}
})
