import { Translator, resolveRoute } from 'runonweb/translate'
import { bindSamples, createDemoUI, formatMs, onDemoPage } from './ui.ts'

onDemoPage('tr', () => {
const ui = createDemoUI('tr')
const timingEl = ui.el('timing')
const fromEl = ui.el<HTMLSelectElement>('from')
const toEl = ui.el<HTMLSelectElement>('to')
const swapEl = ui.el<HTMLButtonElement>('swap')
const routeEl = ui.el('route')
const inputEl = ui.el<HTMLTextAreaElement>('input')
const resultEl = ui.el('result')
const runBtn = ui.el<HTMLButtonElement>('run')
const htmlEl = ui.el<HTMLInputElement>('html')

/** One engine serves every pair; models are downloaded on demand and kept in WASM memory. */
let translator: Translator | null = null

function currentPair(): { from: string; to: string } {
  return { from: fromEl?.value ?? 'en', to: toEl?.value ?? 'es' }
}

const mb = (bytes: number) => `${Math.round((bytes + 5e6) / 1e6)} MB`

/** Direct model or two hops through English, with the download it needs. */
function showRoute() {
  const { from, to } = currentPair()
  const route = resolveRoute(from, to)
  if (runBtn) runBtn.disabled = !route
  if (!routeEl) return
  if (!route) {
    routeEl.textContent = from === to ? 'Pick two different languages' : 'No model for this pair'
    return
  }
  const size = mb(route.reduce((sum, entry) => sum + entry.bytes, 0))
  routeEl.textContent = route.length > 1 ? `Via English · ${size}` : `Direct · ${size}`
}

function setBusy(busy: boolean) {
  if (runBtn) runBtn.disabled = busy
  if (fromEl) fromEl.disabled = busy
  if (toEl) toEl.disabled = busy
  if (swapEl) swapEl.disabled = busy
  if (!busy) showRoute()
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
function onPairChange() {
  showRoute()
  const { from, to } = currentPair()
  if (resolveRoute(from, to)) ui.setStatus(`Pair ${from} → ${to}. Translate to load it`)
}

fromEl?.addEventListener('change', onPairChange)
toEl?.addEventListener('change', onPairChange)
swapEl?.addEventListener('click', () => {
  if (!fromEl || !toEl) return
  ;[fromEl.value, toEl.value] = [toEl.value, fromEl.value]
  const result = resultEl?.textContent?.trim()
  if (inputEl && result && result !== '(empty)' && result !== 'Translation will appear here…') inputEl.value = result
  onPairChange()
})
showRoute()

return () => {
  translator?.dispose()
  translator = null
}
})
