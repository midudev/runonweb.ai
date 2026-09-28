import {
  DEFAULT_TTS_SIZE,
  TTS_SIZES,
  defaultVoiceFor,
  getVoice,
  voiceGroupsFor,
  voiceLabel,
  type TTSSize,
} from 'runonweb/tts'
import { float32ToWavBlob } from 'runonweb/tts'
import { bindSamples, createDemoUI, createUrlPool, formatMs, onDemoPage } from './ui.ts'
import { createWaveform } from './waveform.ts'
import type { ProgressInfo } from 'runonweb/core'

onDemoPage('tts', () => {
  const ui = createDemoUI('tts')
  const urls = createUrlPool()
  const timingEl = ui.el('timing')
  const inputEl = ui.el<HTMLTextAreaElement>('input')
  const voiceEl = ui.el<HTMLSelectElement>('voice')
  const languageEl = ui.el<HTMLSelectElement>('language')
  const languageWrap = ui.el('language-wrap')
  const runBtn = ui.el<HTMLButtonElement>('run')
  const downloadLink = ui.el<HTMLAnchorElement>('download')
  const audioEl = ui.el<HTMLAudioElement>('audio')
  const playerEl = ui.el('player')
  const playBtn = ui.el<HTMLButtonElement>('play')
  const timeEl = ui.el('time')
  const canvas = ui.el<HTMLCanvasElement>('wave')
  const sizeButtons = [...document.querySelectorAll<HTMLButtonElement>('[data-tts-size]')]

  const wave =
    canvas && audioEl
      ? createWaveform({ canvas, audio: audioEl, playBtn, timeEl })
      : null

  let size: TTSSize = DEFAULT_TTS_SIZE
  let worker: Worker | null = null
  let loadedSize: TTSSize | null = null
  let pending: { resolve: () => void; reject: (err: Error) => void } | null = null
  let pcm: Float32Array[] = []
  let sampleRate = 24_000
  let player: PcmPlayer | null = null

  function showPlayer() {
    if (playerEl) playerEl.hidden = false
  }

  fillVoices(size, defaultVoiceFor(size))

  function selectedVoice() {
    return voiceEl?.value || defaultVoiceFor(size)
  }

  function selectedLanguage() {
    return languageEl?.value || 'en'
  }

  function setLanguage(code: string) {
    if (languageEl) languageEl.value = code
  }

  function setBusy(busy: boolean) {
    if (runBtn) runBtn.disabled = busy
    if (voiceEl) voiceEl.disabled = busy
    if (languageEl) languageEl.disabled = busy
    for (const btn of sizeButtons) btn.disabled = busy
  }

  function markSize(next: TTSSize) {
    size = next
    for (const btn of sizeButtons) {
      btn.setAttribute('aria-pressed', btn.dataset.ttsSize === next ? 'true' : 'false')
    }
    const current = selectedVoice()
    const keep = getVoice(current)?.size === next ? current : defaultVoiceFor(next)
    fillVoices(next, keep)
    if (languageWrap) languageWrap.hidden = next !== 'multi'
  }

  function fillVoices(next: TTSSize, selected: string) {
    if (!voiceEl) return
    voiceEl.replaceChildren()
    for (const group of voiceGroupsFor(next)) {
      const optgroup = document.createElement('optgroup')
      optgroup.label = group.label
      for (const voice of group.voices) {
        const option = document.createElement('option')
        option.value = voice.id
        option.textContent = voiceLabel(voice)
        option.selected = voice.id === selected
        optgroup.appendChild(option)
      }
      voiceEl.appendChild(optgroup)
    }
  }

  function ensureWorker() {
    if (worker) return worker
    worker = new Worker(new URL('./tts.worker.ts', import.meta.url), { type: 'module' })
    worker.onmessage = (event: MessageEvent) => {
      const msg = event.data as
        | { type: 'progress'; info: ProgressInfo }
        | { type: 'ready'; device: string | null }
        | { type: 'chunk'; audio: Float32Array; samplingRate: number; text: string }
        | { type: 'done' }
        | { type: 'error'; message: string }

      if (msg.type === 'progress') {
        ui.onProgress(msg.info)
        return
      }
      if (msg.type === 'ready') {
        loadedSize = size
        if (msg.device) ui.setDevice(`Using ${msg.device} · ${TTS_SIZES[size].label}`)
        ui.setStatus('Model ready')
        ui.setProgress(null)
        pending?.resolve()
        pending = null
        return
      }
      if (msg.type === 'chunk') {
        pcm.push(msg.audio)
        sampleRate = msg.samplingRate
        player ??= new PcmPlayer()
        player.play(msg.audio, msg.samplingRate)
        showPlayer()
        if (player.analyser) wave?.live(player.analyser)
        ui.setStatus(`Speaking… ${msg.text.slice(0, 40)}`)
        return
      }
      if (msg.type === 'done') {
        pending?.resolve()
        pending = null
        return
      }
      pending?.reject(new Error(msg.message))
      pending = null
    }
    worker.onerror = (err) => {
      pending?.reject(new Error(err.message || 'TTS worker failed'))
      pending = null
    }
    return worker
  }

  function request(msg: object) {
    return new Promise<void>((resolve, reject) => {
      pending = { resolve, reject }
      ensureWorker().postMessage(msg)
    })
  }

  async function ensureModel() {
    if (loadedSize === size && worker) return
    ui.setStatus('Loading…')
    await request({ type: 'load', size })
  }

  async function runSpeak() {
    const text = inputEl?.value.trim() ?? ''
    if (!text) {
      ui.setError('Type something to say')
      return
    }

    await ui.run(
      async () => {
        await ensureModel()
        ui.setStatus('Synthesizing…')
        pcm = []
        wave?.live(null)
        wave?.pause()
        player?.stop()
        player = null
        const t0 = performance.now()
        await request({ type: 'speak', text, voice: selectedVoice(), language: selectedLanguage() })
        if (timingEl) timingEl.textContent = formatMs(performance.now() - t0)

        const audio = concat(pcm)
        urls.revokeAll()
        const wav = float32ToWavBlob(audio, sampleRate)
        const url = urls.add(wav)
        if (downloadLink) {
          downloadLink.href = url
          downloadLink.hidden = false
        }
        showPlayer()
        const spoken = player as PcmPlayer | null
        await spoken?.whenIdle()
        wave?.live(null)
        spoken?.stop()
        player = null
        await wave?.load(url)
        ui.setStatus('Done')
        ui.setProgress(null)
      },
      { busy: setBusy }
    )
  }

  ui.initDeviceBadge()
  runBtn?.addEventListener('click', () => void runSpeak())
  bindSamples(ui, {
    onText: (s) => {
      // Non-English samples use Supertonic (`multi`); `extra` carries the language code.
      if (s.extra && s.extra !== 'en') {
        if (size !== 'multi') loadedSize = null
        markSize('multi')
        setLanguage(s.extra)
      } else if (size === 'multi') {
        setLanguage('en')
      }
      if (inputEl) inputEl.value = s.text
      void runSpeak()
    },
  })
  for (const btn of sizeButtons) {
    btn.addEventListener('click', () => {
      const next = btn.dataset.ttsSize as TTSSize | undefined
      if (!next || next === size) return
      markSize(next)
      loadedSize = null
      ui.setStatus(`Size: ${TTS_SIZES[next].label} ${TTS_SIZES[next].downloadMB}`)
    })
  }

  return () => {
    worker?.postMessage({ type: 'dispose' })
    worker?.terminate()
    worker = null
    player?.stop()
    wave?.dispose()
    urls.revokeAll()
  }
})

function concat(chunks: Float32Array[]) {
  const total = chunks.reduce((n, c) => n + c.length, 0)
  const out = new Float32Array(total)
  let offset = 0
  for (const chunk of chunks) {
    out.set(chunk, offset)
    offset += chunk.length
  }
  return out
}

class PcmPlayer {
  #ctx: AudioContext | null = null
  #next = 0
  #analyser: AnalyserNode | null = null

  get analyser() {
    return this.#analyser
  }

  play(samples: Float32Array, rate: number) {
    this.#ctx ??= new AudioContext({ sampleRate: rate })
    const ctx = this.#ctx
    if (!this.#analyser) {
      this.#analyser = ctx.createAnalyser()
      this.#analyser.fftSize = 1024
      this.#analyser.connect(ctx.destination)
    }
    void ctx.resume()
    const buffer = ctx.createBuffer(1, samples.length, rate)
    buffer.getChannelData(0).set(samples)
    const src = ctx.createBufferSource()
    src.buffer = buffer
    src.connect(this.#analyser)
    const now = ctx.currentTime
    if (this.#next < now) this.#next = now
    src.start(this.#next)
    this.#next += buffer.duration
  }

  whenIdle() {
    const ctx = this.#ctx
    if (!ctx) return Promise.resolve()
    const remaining = Math.max(0, this.#next - ctx.currentTime)
    return new Promise<void>((resolve) => {
      setTimeout(resolve, remaining * 1000 + 40)
    })
  }

  stop() {
    void this.#ctx?.close()
    this.#ctx = null
    this.#analyser = null
    this.#next = 0
  }
}
