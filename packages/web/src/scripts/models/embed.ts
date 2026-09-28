import { TextEmbedder, cosineSimilarity } from 'runonweb/embed'
import { bindSamples, createDemoUI, formatMs, onDemoPage } from './ui.ts'

onDemoPage('emb', () => {
const ui = createDemoUI('emb')
const timingEl = ui.el('timing')
const queryEl = ui.el<HTMLTextAreaElement>('query')
const candidatesEl = ui.el<HTMLTextAreaElement>('candidates')
const runBtn = ui.el<HTMLButtonElement>('run')
const resultEl = ui.el<HTMLOListElement>('result')

let embedder: TextEmbedder | null = null

function setBusy(busy: boolean) {
  if (runBtn) runBtn.disabled = busy
}

async function ensureModel() {
  if (embedder) return embedder
  const model = new TextEmbedder({ onProgress: ui.onProgress })
  try {
    await model.load()
  } catch (err) {
    ui.setStatus('Load failed')
    throw err
  }
  embedder = model
  if (model.device) ui.setDevice(`Using ${model.device}`)
  ui.setStatus('Model ready')
  ui.setProgress(null)
  return model
}

function render(rows: Array<{ text: string; score: number }>) {
  if (!resultEl) return
  resultEl.replaceChildren()
  for (const row of rows) {
    const li = document.createElement('li')
    li.className = 'flex items-center gap-3 bg-bg px-4 py-2.5'
    const pct = Math.max(0, Math.min(100, Math.round(row.score * 100)))
    li.innerHTML = `
      <span class="w-12 shrink-0 text-fg-3">${row.score.toFixed(3)}</span>
      <span class="h-1.5 w-24 shrink-0 overflow-hidden bg-bg-4">
        <span class="block h-full bg-fg" style="width:${pct}%"></span>
      </span>
      <span class="text-fg"></span>`
    li.lastElementChild!.textContent = row.text
    resultEl.appendChild(li)
  }
}

async function runRank() {
  const query = queryEl?.value.trim() ?? ''
  const candidates = (candidatesEl?.value ?? '')
    .split('\n')
    .map((s) => s.trim())
    .filter(Boolean)

  if (!query || candidates.length === 0) {
    ui.setError('Enter a query and at least one candidate')
    return
  }

  await ui.run(
    async () => {
      const model = await ensureModel()
      ui.setStatus('Embedding…')
      const t0 = performance.now()
      const { embeddings, dimensions } = await model.embed([query, ...candidates])
      if (timingEl) timingEl.textContent = `${formatMs(performance.now() - t0)} · ${dimensions}d`
      const [q, ...rest] = embeddings
      const rows = rest
        .map((vec, i) => ({ text: candidates[i]!, score: cosineSimilarity(q!, vec) }))
        .sort((a, b) => b.score - a.score)
      render(rows)
      ui.setStatus('Done')
      ui.setProgress(null)
    },
    { busy: setBusy }
  )
}

ui.initDeviceBadge()
runBtn?.addEventListener('click', () => void runRank())
bindSamples(ui, {
  onText: (s) => {
    if (queryEl) queryEl.value = s.text
    if (candidatesEl && s.extra) candidatesEl.value = s.extra
    void runRank()
  },
})
})
