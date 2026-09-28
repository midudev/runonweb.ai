import { TextToSpeech, type TTSSize } from 'runonweb/tts'
import type { ProgressInfo } from 'runonweb/core'

type InMsg =
  | { type: 'load'; size: TTSSize }
  | { type: 'speak'; text: string; voice: string; speed?: number; language?: string }
  | { type: 'dispose' }

type OutMsg =
  | { type: 'progress'; info: ProgressInfo }
  | { type: 'ready'; device: string | null }
  | { type: 'chunk'; audio: Float32Array; samplingRate: number; text: string }
  | { type: 'done' }
  | { type: 'error'; message: string }

let tts: TextToSpeech | null = null
let size: TTSSize | null = null

function post(msg: OutMsg, transfer?: Transferable[]) {
  self.postMessage(msg, transfer ? { transfer } : undefined)
}

self.onmessage = (event: MessageEvent<InMsg>) => {
  void handle(event.data)
}

async function handle(msg: InMsg) {
  try {
    if (msg.type === 'dispose') {
      tts?.dispose()
      tts = null
      size = null
      return
    }

    if (msg.type === 'load') {
      if (tts && size === msg.size) {
        post({ type: 'ready', device: tts.device })
        return
      }
      tts?.dispose()
      tts = new TextToSpeech({
        size: msg.size,
        onProgress: (info) => post({ type: 'progress', info }),
      })
      await tts.load()
      size = msg.size
      post({ type: 'ready', device: tts.device })
      return
    }

    if (!tts) throw new Error('TTS worker is not loaded')
    for await (const chunk of tts.speakStream(msg.text, { voice: msg.voice, speed: msg.speed, language: msg.language })) {
      post(
        { type: 'chunk', audio: chunk.audio, samplingRate: chunk.samplingRate, text: chunk.text },
        [chunk.audio.buffer]
      )
    }
    post({ type: 'done' })
  } catch (err) {
    post({ type: 'error', message: err instanceof Error ? err.message : String(err) })
  }
}
