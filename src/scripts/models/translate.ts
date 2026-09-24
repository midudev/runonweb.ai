import { Translator } from 'runonweb/translate'
import { bindSamples, createDemoUI, formatMs, onDemoPage } from './ui.ts'

onDemoPage('tr', () => {
const ui = createDemoUI('tr')
const timingEl = ui.el('timing')
const pairEl = ui.el<HTMLSelectElement>('pair')
const inputEl = ui.el<HTMLTextAreaElement>('input')
const resultEl = ui.el('result')
const runBtn = ui.el<HTMLButtonElement>('run')
const htmlEl = ui.el<HTMLInputElement>('html')

/** One engine serves every pair; models are downloaded on demand and kept in WASM memory. */
let translator: Translator | null = null

function currentPair(): { from: string; to: string } {
  const option = pairEl?.selectedOptions[0]
  return { from: option?.dataset.from ?? 'en', to: option?.dataset.to ?? 'es' }
}

function setBusy(busy: boolean) {
  if (runBtn) runBtn.disabled = busy
  if (pairEl) pairEl.disabled = busy
}

async function ensureEngine(): Promise<Translator> {
  if (translator) return translator
  const t = new Translator({
    ...currentPair(),
    onProgress: ui.onProgress,
  })
  translator = t
  return t
}

async function runTranslate() {
  const text = inputEl?.value.trim() ?? ''
  if (!text) {
    ui.setError('Type something to translate')
    return
  }

  await ui.run(
    async () => {
      const { from, to } = currentPair()
      const engine = await ensureEngine()
      ui.setStatus(`Loading ${from} → ${to}…`)
      const t0 = performance.now()
      const result = await engine.translate(text, { from, to, html: htmlEl?.checked ?? false })
      if (timingEl) timingEl.textContent = formatMs(performance.now() - t0)
      if (resultEl) resultEl.textContent = result.text || '(empty)'
      if (engine.device) ui.setDevice(`Using ${engine.device}`)
      ui.setStatus(`Done · ${from} → ${to}`)
      ui.setProgress(null)
    },
    { busy: setBusy }
  )
}

ui.initDeviceBadge()
runBtn?.addEventListener('click', () => void runTranslate())
bindSamples(ui, {
  onText: (s) => {
    if (inputEl) inputEl.value = s.text
    if (htmlEl) htmlEl.checked = s.extra === 'html'
    void runTranslate()
  },
})
pairEl?.addEventListener('change', () => {
  const { from, to } = currentPair()
  ui.setStatus(`Pair ${from} → ${to} — translate to load it`)
})

return () => {
  translator?.dispose()
  translator = null
}
})
