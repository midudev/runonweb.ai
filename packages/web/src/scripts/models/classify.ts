import { Classifier, CLASSIFY_SIZES, type Answer, type ClassifySize, type Question } from 'runonweb/classify'
import { bindSamples, createDemoUI, formatMs, onDemoPage } from './ui.ts'

onDemoPage('cls', () => {
const ui = createDemoUI('cls')
const timingEl = ui.el('timing')
const stateEl = ui.el<HTMLTextAreaElement>('state')
const questionsEl = ui.el<HTMLTextAreaElement>('questions')
const datesEl = ui.el<HTMLInputElement>('dates')
const resultEl = ui.el('result')
const jsonEl = ui.el('json')
const runBtn = ui.el<HTMLButtonElement>('run')
const sizeNote = ui.el('size-note')
const sizeButtons = [...document.querySelectorAll<HTMLButtonElement>('[data-cls-size]')]
let size: ClassifySize = 'small'

let classifier: Classifier | null = null
let classifyTimer = 0
let classifying = false
let queued = false

function setBusy(busy: boolean) {
  if (runBtn) runBtn.disabled = busy
  if (datesEl) datesEl.disabled = busy
  for (const btn of sizeButtons) btn.disabled = busy
}

async function ensureModel() {
  if (classifier && classifier.size === size) return classifier
  classifier?.dispose()
  classifier = null
  const model = new Classifier({ size, onProgress: ui.onProgress })
  try {
    await model.load()
  } catch (err) {
    ui.setStatus('Load failed')
    throw err
  }
  classifier = model
  if (model.device) ui.setDevice(`Using ${model.device} · ${CLASSIFY_SIZES[size].base}`)
  ui.setStatus('Model ready')
  ui.setProgress(null)
  return model
}

function parseQuestions(): Record<string, Question> {
  const raw = questionsEl?.value.trim() ?? ''
  let parsed: unknown
  try {
    parsed = JSON.parse(raw)
  } catch (err) {
    throw new Error(`Questions are not valid JSON: ${err instanceof Error ? err.message : String(err)}`)
  }
  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
    throw new Error('Questions must be a JSON object: { "id": { "type": "noul" | "choice" | "score", … } }')
  }
  return parsed as Record<string, Question>
}

const pct = (p: number) => `${Math.round(p * 100)}%`

function bar(label: string, p: number, top: boolean): HTMLElement {
  const row = document.createElement('div')
  row.className = 'flex items-center gap-3'
  const name = document.createElement('span')
  name.className = `w-28 shrink-0 truncate ${top ? 'text-fg' : 'text-fg-2'}`
  name.textContent = label
  name.title = label
  const track = document.createElement('span')
  track.className = 'h-1.5 min-w-0 flex-1 overflow-hidden bg-bg-4'
  const fill = document.createElement('span')
  fill.className = `cls-bar block h-full ${top ? 'bg-accent' : 'bg-fg-3'}`
  fill.style.width = '0%'
  track.appendChild(fill)
  const value = document.createElement('span')
  value.className = `readout w-12 shrink-0 text-right ${top ? 'text-accent' : 'text-fg-3'}`
  value.textContent = p.toFixed(2)
  row.append(name, track, value)
  requestAnimationFrame(() => {
    fill.style.width = `${Math.max(0, Math.min(100, p * 100))}%`
  })
  return row
}

function optionLabels(question: Question): string[] {
  if (question.type === 'noul') return ['yes', 'no']
  if (question.type === 'choice') return Object.keys(question.criteria)
  return question.criteria.map((item, i) => (typeof item === 'string' ? item : String(i)))
}

function thinkingBar(label: string): HTMLElement {
  const row = document.createElement('div')
  row.className = 'flex items-center gap-3'
  const name = document.createElement('span')
  name.className = 'w-28 shrink-0 truncate text-fg-3'
  name.textContent = label
  name.title = label
  const track = document.createElement('span')
  track.className = 'relative h-1.5 min-w-0 flex-1 overflow-hidden bg-bg-4'
  const sweep = document.createElement('span')
  sweep.className = 'cls-sweep'
  track.appendChild(sweep)
  const value = document.createElement('span')
  value.className = 'readout w-12 shrink-0 text-right text-fg-3'
  value.textContent = '···'
  row.append(name, track, value)
  return row
}

function thinkingCard(id: string, question: Question): HTMLElement {
  const el = document.createElement('section')
  el.className = 'space-y-2.5 border border-line bg-bg px-4 py-3 font-mono text-[12px]'
  const head = document.createElement('header')
  head.className = 'flex items-baseline gap-2'
  const name = document.createElement('span')
  name.className = 'text-[13px] text-fg'
  name.textContent = id
  const type = document.createElement('span')
  type.className = 'readout text-fg-3'
  type.textContent = question.type
  const verdict = document.createElement('span')
  verdict.className = 'readout cls-wait ml-auto text-fg-3'
  verdict.textContent = '…'
  head.append(name, type, verdict)
  el.appendChild(head)
  for (const label of optionLabels(question)) el.appendChild(thinkingBar(label))
  return el
}

function showThinking(questions: Record<string, Question>) {
  resultEl?.replaceChildren(...Object.entries(questions).map(([id, question]) => thinkingCard(id, question)))
}

function card(id: string, answer: Answer): HTMLElement {
  const el = document.createElement('section')
  el.className = 'space-y-2.5 border border-line bg-bg px-4 py-3 font-mono text-[12px]'
  const head = document.createElement('header')
  head.className = 'flex items-baseline gap-2'
  const name = document.createElement('span')
  name.className = 'text-[13px] text-fg'
  name.textContent = id
  const type = document.createElement('span')
  type.className = 'readout text-fg-3'
  type.textContent = answer.type
  const verdict = document.createElement('span')
  verdict.className = 'readout ml-auto text-accent'
  head.append(name, type, verdict)
  el.appendChild(head)

  if (answer.type === 'noul') {
    verdict.textContent = `p(yes) ${answer.noul.toFixed(2)}`
    el.append(bar('yes', answer.noul, answer.noul >= 0.5), bar('no', 1 - answer.noul, answer.noul < 0.5))
  } else if (answer.type === 'choice') {
    verdict.textContent = `${answer.choice} · conf ${pct(answer.confidence)}`
    for (const [key, p] of Object.entries(answer.probabilities)) el.appendChild(bar(key, p, key === answer.choice))
  } else {
    const levels = Object.keys(answer.probabilities).length
    verdict.textContent = `${answer.score.toFixed(2)} / ${levels - 1} · conf ${pct(answer.confidence)}`
    const probs = Object.entries(answer.probabilities)
    const mode = probs.reduce((best, cur) => (cur[1] > best[1] ? cur : best))[0]
    for (const [key, p] of probs) el.appendChild(bar(answer.legend[key] ?? key, p, key === mode))
  }
  return el
}

async function runClassify(source: 'click' | 'type' = 'click') {
  const state = stateEl?.value.trim() ?? ''
  if (!state) {
    if (source === 'click') ui.setError('Write the text to evaluate first')
    return
  }
  let questions: Record<string, Question>
  try {
    questions = parseQuestions()
  } catch (err) {
    if (source === 'click') {
      ui.setError(err instanceof Error ? err.message : String(err))
    }
    return
  }
  // Typing never starts the multi-gigabyte Large download; the button does.
  if (source === 'type' && size === 'large' && classifier?.size !== 'large') return
  if (classifying) {
    queued = true
    return
  }
  classifying = true
  showThinking(questions)

  await ui.run(
    async () => {
      const model = await ensureModel()
      ui.setStatus('Classifying…')
      const t0 = performance.now()
      const result = await model.classify({ state, questions }, { dateFacts: Boolean(datesEl?.checked) })
      if (timingEl) {
        const n = Object.keys(questions).length
        timingEl.textContent = `${formatMs(performance.now() - t0)} · ${n} question${n === 1 ? '' : 's'}`
      }
      resultEl?.replaceChildren(...Object.entries(result.answers).map(([id, a]) => card(id, a as Answer)))
      if (jsonEl) jsonEl.textContent = JSON.stringify(result, null, 2)
      ui.setStatus('Done')
      ui.setProgress(null)
    },
    { busy: setBusy }
  )
  classifying = false
  if (queued) {
    queued = false
    void runClassify('type')
  }
}

function scheduleClassify() {
  window.clearTimeout(classifyTimer)
  classifyTimer = window.setTimeout(() => void runClassify('type'), 500)
}

ui.initDeviceBadge()
runBtn?.addEventListener('click', () => void runClassify())
for (const btn of sizeButtons) {
  btn.addEventListener('click', () => {
    const next = btn.dataset.clsSize as ClassifySize | undefined
    if (!next || next === size) return
    size = next
    for (const b of sizeButtons) b.setAttribute('aria-pressed', b.dataset.clsSize === next ? 'true' : 'false')
    if (sizeNote) sizeNote.textContent = `${CLASSIFY_SIZES[next].downloadMB}, cached after the first load`
    ui.setStatus(`Size: ${CLASSIFY_SIZES[next].label} (${CLASSIFY_SIZES[next].base}, ${CLASSIFY_SIZES[next].downloadMB}). Press Classify`)
  })
}
for (const el of [stateEl, questionsEl]) {
  el?.addEventListener('input', scheduleClassify)
  el?.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
      e.preventDefault()
      window.clearTimeout(classifyTimer)
      void runClassify()
    }
  })
}
bindSamples(ui, {
  onText: (s) => {
    if (stateEl) stateEl.value = s.text
    if (s.extra) {
      const { questions, dateFacts } = JSON.parse(s.extra) as { questions: unknown; dateFacts?: boolean }
      if (questionsEl) questionsEl.value = JSON.stringify(questions, null, 2)
      if (datesEl) datesEl.checked = Boolean(dateFacts)
    }
    void runClassify()
  },
})

return () => {
  window.clearTimeout(classifyTimer)
  classifier?.dispose()
  classifier = null
}
})
