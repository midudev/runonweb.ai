import {
  TranscriptCleaner,
  type Context,
  type Structure,
  type Styling,
} from 'runonweb/clean'
import { bindSamples, createDemoUI, formatMs, onDemoPage } from './ui.ts'

onDemoPage('cl', () => {
const ui = createDemoUI('cl')
const timingEl = ui.el('timing')
const inputEl = ui.el<HTMLTextAreaElement>('input')
const resultEl = ui.el('result')
const runBtn = ui.el<HTMLButtonElement>('run')
const stylingEl = ui.el<HTMLSelectElement>('styling')
const structureEl = ui.el<HTMLSelectElement>('structure')
const contextEl = ui.el<HTMLSelectElement>('context')

let cleaner: TranscriptCleaner | null = null

function setBusy(busy: boolean) {
  if (runBtn) runBtn.disabled = busy
  if (stylingEl) stylingEl.disabled = busy
  if (structureEl) structureEl.disabled = busy
  if (contextEl) contextEl.disabled = busy
}

function currentControls(): { styling: Styling; structure: Structure; context: Context } {
  return {
    styling: (stylingEl?.value as Styling) || 'semi-formal',
    structure: (structureEl?.value as Structure) || 'prose',
    context: (contextEl?.value as Context) || 'general',
  }
}

async function ensureModel() {
  if (cleaner) return cleaner
  const model = new TranscriptCleaner({ onProgress: ui.onProgress })
  try {
    await model.load()
  } catch (err) {
    ui.setStatus('Load failed')
    throw err
  }
  cleaner = model
  if (model.device) ui.setDevice(`Using ${model.device}`)
  ui.setStatus('Model ready')
  ui.setProgress(null)
  return model
}

async function runClean() {
  const text = inputEl?.value.trim() ?? ''
  if (!text) {
    ui.setError('Paste a transcript to clean')
    return
  }

  await ui.run(
    async () => {
      const model = await ensureModel()
      ui.setStatus('Cleaning…')
      if (resultEl) resultEl.textContent = ''
      const t0 = performance.now()
      const result = await model.clean(text, {
        ...currentControls(),
        onPartial: (partial) => {
          if (resultEl) resultEl.textContent = partial
        },
      })
      if (timingEl) timingEl.textContent = formatMs(performance.now() - t0)
      if (resultEl) {
        resultEl.textContent = result.text || '(empty: the input was only filler or noise)'
      }
      ui.setStatus('Done')
      ui.setProgress(null)
    },
    { busy: setBusy }
  )
}

ui.initDeviceBadge()
runBtn?.addEventListener('click', () => void runClean())
bindSamples(ui, {
  onText: (s) => {
    if (inputEl) inputEl.value = s.text
    if (s.extra === 'email' && contextEl) contextEl.value = 'email'
    else if (s.extra === 'lists' && structureEl) structureEl.value = 'lists'
    else {
      if (contextEl) contextEl.value = 'general'
      if (structureEl) structureEl.value = 'prose'
    }
    void runClean()
  },
})

return () => {
  cleaner?.dispose()
  cleaner = null
}
})
