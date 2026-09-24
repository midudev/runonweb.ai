import { SpeechToText, type STTChunk } from 'runonweb/stt'
import { createWaveform } from './waveform.ts'
import { bindSamples, createDemoUI, createUrlPool, formatMs, onDemoPage } from './ui.ts'

onDemoPage('stt', () => {
const ui = createDemoUI('stt')
const urls = createUrlPool()
const textEl = ui.el('text')
const caretEl = ui.el('caret')
const timingEl = ui.el('timing')
const fileInput = ui.el<HTMLInputElement>('file')
const recordBtn = ui.el<HTMLButtonElement>('record')
const transcribeBtn = ui.el<HTMLButtonElement>('transcribe')
const playBtn = ui.el<HTMLButtonElement>('play')
const audioEl = ui.el<HTMLAudioElement>('audio')
const playerEl = ui.el('player')
const samplesEl = ui.el('samples')
const langButtons = [...document.querySelectorAll<HTMLButtonElement>('[data-stt-lang]')]
const canvas = ui.el<HTMLCanvasElement>('wave')
const timeEl = ui.el('time')

const wave =
  canvas && audioEl
    ? createWaveform({ canvas, audio: audioEl, playBtn, timeEl })
    : null

const MULTILINGUAL_MODEL = 'onnx-community/whisper-tiny'

let stt: SpeechToText | null = null
let mediaRecorder: MediaRecorder | null = null
let recordedChunks: Blob[] = []
let selectedFile: File | Blob | null = null
let selectedLanguage: string | undefined
let recording = false
let liveCtx: AudioContext | null = null
let chunks: STTChunk[] = []
let lastPartial = ''
let highlightRaf = 0
let wordEls: HTMLElement[] = []

function setBusy(busy: boolean) {
  if (transcribeBtn) transcribeBtn.disabled = busy || !selectedFile
  if (recordBtn) recordBtn.disabled = busy && !recording
  if (fileInput) fileInput.disabled = busy
  for (const btn of langButtons) btn.disabled = busy
}

function markActiveSample(btn: HTMLButtonElement | null) {
  samplesEl?.querySelectorAll<HTMLButtonElement>('[data-sample]').forEach((el) => {
    el.setAttribute('aria-pressed', el === btn ? 'true' : 'false')
  })
}

function markLanguage(code: string | undefined) {
  selectedLanguage = !code || code === 'auto' ? undefined : code
  const current = selectedLanguage ?? 'auto'
  for (const btn of langButtons) {
    btn.setAttribute('aria-pressed', btn.dataset.sttLang === current ? 'true' : 'false')
  }
}

function setCaret(on: boolean) {
  if (caretEl) caretEl.hidden = !on
}

function audioDuration() {
  const dur = audioEl && Number.isFinite(audioEl.duration) ? audioEl.duration : 0
  return dur > 0 ? dur : (wave?.duration ?? 0)
}

function fitChunks(next: STTChunk[]): STTChunk[] {
  const duration = audioDuration()
  if (!next.length) return next
  const cleaned = next.map((c) => ({
    text: c.text.trim(),
    start: c.start,
    end: Math.max(c.end, c.start),
  }))
  const maxEnd = Math.max(...cleaned.map((c) => c.end))
  if (duration > 0 && maxEnd > 0 && (maxEnd > duration * 1.12 || maxEnd < duration * 0.65)) {
    const scale = duration / maxEnd
    return cleaned.map((c) => ({ ...c, start: c.start * scale, end: c.end * scale }))
  }
  return cleaned
}

function setPlain(text: string, streaming: boolean) {
  chunks = []
  wordEls = []
  if (textEl) {
    textEl.replaceChildren()
    textEl.textContent = text
  }
  setCaret(streaming)
}

function setWords(next: STTChunk[], fallback = '') {
  chunks = fitChunks(next)
  if (!textEl) return
  textEl.replaceChildren()
  if (!chunks.length) {
    textEl.textContent = fallback || '(empty transcript)'
    setCaret(false)
    wordEls = []
    return
  }
  wordEls = chunks.map((chunk, i) => {
    if (i > 0) textEl.appendChild(document.createTextNode(' '))
    const span = document.createElement('span')
    span.className = 'stt-word'
    span.dataset.i = String(i)
    span.textContent = chunk.text
    textEl.appendChild(span)
    return span
  })
  setCaret(false)
  highlight(audioEl?.currentTime ?? 0)
}

function wordAt(time: number) {
  if (!chunks.length) return -1
  if (time < chunks[0]!.start) return 0
  for (let i = chunks.length - 1; i >= 0; i--) {
    if (time >= chunks[i]!.start) return i
  }
  return 0
}

function highlight(time: number) {
  if (!wordEls.length) return
  const active = wordAt(time)
  for (let i = 0; i < wordEls.length; i++) {
    wordEls[i]!.classList.toggle('is-now', i === active)
  }
}

function loopHighlight() {
  highlight(audioEl?.currentTime ?? 0)
  if (audioEl && !audioEl.paused && chunks.length) {
    highlightRaf = requestAnimationFrame(loopHighlight)
  } else {
    highlightRaf = 0
  }
}

function showPlayer() {
  if (playerEl) playerEl.hidden = false
}

async function ensureModel() {
  if (stt) return stt
  const model = new SpeechToText({ model: MULTILINGUAL_MODEL, onProgress: ui.onProgress })
  try {
    await model.load()
  } catch (err) {
    ui.setStatus('Load failed')
    throw err
  }
  stt = model
  if (model.device) ui.setDevice(`Using ${model.device}`)
  ui.setStatus('Model ready')
  ui.setProgress(null)
  return model
}

async function runTranscribe() {
  if (!selectedFile) {
    ui.setError('Select or record audio first')
    return
  }
  const audio = selectedFile
  lastPartial = ''
  setPlain('', true)

  await ui.run(
    async () => {
      const model = await ensureModel()
      ui.setStatus('Transcribing…')
      const t0 = performance.now()
      const result = await model.transcribe(audio, {
        language: selectedLanguage,
        onPartial: (text) => {
          lastPartial = text
          if (!chunks.length) setPlain(text, true)
        },
      })
      if (timingEl) timingEl.textContent = formatMs(performance.now() - t0)
      const text = result.text || lastPartial
      if (result.chunks.length) setWords(result.chunks, text)
      else if (text) setPlain(text, false)
      else setPlain('(empty transcript)', false)
      ui.setStatus('Done')
      ui.setProgress(null)
      if (audioEl && !audioEl.paused && chunks.length) loopHighlight()
    },
    { busy: setBusy }
  )
}

function stopLive() {
  wave?.live(null)
  void liveCtx?.close()
  liveCtx = null
}

async function startLive(stream: MediaStream) {
  const AC =
    globalThis.AudioContext ??
    (globalThis as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext
  liveCtx = new AC()
  const source = liveCtx.createMediaStreamSource(stream)
  const analyser = liveCtx.createAnalyser()
  analyser.fftSize = 1024
  source.connect(analyser)
  showPlayer()
  wave?.live(analyser)
}

async function toggleRecord() {
  if (recording && mediaRecorder) {
    mediaRecorder.stop()
    return
  }

  ui.setError(null)
  audioEl?.pause()
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
    recordedChunks = []
    mediaRecorder = new MediaRecorder(stream)
    await startLive(stream)

    mediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) recordedChunks.push(e.data)
    }

    mediaRecorder.onstop = () => {
      stream.getTracks().forEach((t) => t.stop())
      recording = false
      if (recordBtn) recordBtn.textContent = 'Record'
      stopLive()
      const blob = new Blob(recordedChunks, { type: mediaRecorder?.mimeType || 'audio/webm' })
      markActiveSample(null)
      selectAudio(blob, 'recording')
    }

    mediaRecorder.start()
    recording = true
    if (recordBtn) recordBtn.textContent = 'Stop'
    ui.setStatus('Recording…')
  } catch (err) {
    stopLive()
    const message = err instanceof Error ? err.message : String(err)
    ui.setError(`Microphone access failed: ${message}`)
  }
}

function selectAudio(file: File | Blob, name: string, opts?: { previewSrc?: string; skipPreview?: boolean }) {
  selectedFile = file
  ui.setStatus(`Selected: ${name}`)
  ui.setError(null)
  if (transcribeBtn) transcribeBtn.disabled = false
  showPlayer()
  if (!opts?.skipPreview) {
    if (opts?.previewSrc) void wave?.load(opts.previewSrc, { play: true })
    else {
      urls.revokeAll()
      const src = urls.add(file)
      void wave?.load(src, { play: true })
    }
  }
  void runTranscribe()
}

ui.initDeviceBadge()
transcribeBtn?.addEventListener('click', () => void runTranscribe())
recordBtn?.addEventListener('click', () => void toggleRecord())
for (const btn of langButtons) {
  btn.addEventListener('click', () => markLanguage(btn.dataset.sttLang))
}
fileInput?.addEventListener('change', () => {
  const file = fileInput.files?.[0]
  if (file) {
    markActiveSample(null)
    selectAudio(file, file.name)
  }
})
audioEl?.addEventListener('play', () => {
  if (chunks.length && !highlightRaf) loopHighlight()
})
audioEl?.addEventListener('timeupdate', () => {
  if (!highlightRaf) highlight(audioEl.currentTime)
})
audioEl?.addEventListener('seeked', () => {
  highlight(audioEl.currentTime)
})
bindSamples(ui, {
  onSelect: (spec, btn) => {
    if (spec.kind !== 'audio') return
    markLanguage(spec.language)
    markActiveSample(btn)
    showPlayer()
    void wave?.load(spec.src, { play: true })
  },
  onFile: (file) => selectAudio(file, file.name, { skipPreview: true }),
})
return () => {
  urls.revokeAll()
  if (highlightRaf) cancelAnimationFrame(highlightRaf)
  stopLive()
  if (recording) mediaRecorder?.stop()
}
})
