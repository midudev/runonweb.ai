export type SpeakChunk = {
  audio: Float32Array
  samplingRate: number
  text: string
}

export type TTSResult = {
  /** Mono PCM samples. 24 kHz for Kokoro / KittenTTS, 44.1 kHz for Supertonic. */
  audio: Float32Array
  samplingRate: number
}
