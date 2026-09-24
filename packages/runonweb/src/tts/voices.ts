import { DEFAULT_TTS_SIZE, type TTSSize } from './sizes.ts'

export type TTSLocale = 'en-US' | 'en-GB' | 'es' | 'fr' | 'multi'
export type TTSGender = 'female' | 'male'

export type TTSVoice = {
  id: string
  name: string
  locale: TTSLocale
  gender: TTSGender
  /** Subjective grade. A is best. Kitten voices are ungraded. */
  grade: string
  size: TTSSize
}

export const DEFAULT_VOICE = 'af_heart'
export const DEFAULT_TINY_VOICE = 'bella'
export const DEFAULT_MULTI_VOICE = 'st_f1'

/** Kokoro v1.0 voices (small) + KittenTTS voices (tiny) + Supertonic 2 presets (multi, any of its 5 languages). */
export const TTS_VOICES: readonly TTSVoice[] = [
  { id: 'af_heart', name: 'Heart', locale: 'en-US', gender: 'female', grade: 'A', size: 'small' },
  { id: 'af_bella', name: 'Bella', locale: 'en-US', gender: 'female', grade: 'A-', size: 'small' },
  { id: 'af_nicole', name: 'Nicole', locale: 'en-US', gender: 'female', grade: 'B-', size: 'small' },
  { id: 'af_aoede', name: 'Aoede', locale: 'en-US', gender: 'female', grade: 'C+', size: 'small' },
  { id: 'af_kore', name: 'Kore', locale: 'en-US', gender: 'female', grade: 'C+', size: 'small' },
  { id: 'af_sarah', name: 'Sarah', locale: 'en-US', gender: 'female', grade: 'C+', size: 'small' },
  { id: 'am_fenrir', name: 'Fenrir', locale: 'en-US', gender: 'male', grade: 'C+', size: 'small' },
  { id: 'am_michael', name: 'Michael', locale: 'en-US', gender: 'male', grade: 'C+', size: 'small' },
  { id: 'am_puck', name: 'Puck', locale: 'en-US', gender: 'male', grade: 'C+', size: 'small' },
  { id: 'af_alloy', name: 'Alloy', locale: 'en-US', gender: 'female', grade: 'C', size: 'small' },
  { id: 'af_nova', name: 'Nova', locale: 'en-US', gender: 'female', grade: 'C', size: 'small' },
  { id: 'af_sky', name: 'Sky', locale: 'en-US', gender: 'female', grade: 'C-', size: 'small' },
  { id: 'af_jessica', name: 'Jessica', locale: 'en-US', gender: 'female', grade: 'D', size: 'small' },
  { id: 'af_river', name: 'River', locale: 'en-US', gender: 'female', grade: 'D', size: 'small' },
  { id: 'am_echo', name: 'Echo', locale: 'en-US', gender: 'male', grade: 'D', size: 'small' },
  { id: 'am_eric', name: 'Eric', locale: 'en-US', gender: 'male', grade: 'D', size: 'small' },
  { id: 'am_liam', name: 'Liam', locale: 'en-US', gender: 'male', grade: 'D', size: 'small' },
  { id: 'am_onyx', name: 'Onyx', locale: 'en-US', gender: 'male', grade: 'D', size: 'small' },
  { id: 'am_santa', name: 'Santa', locale: 'en-US', gender: 'male', grade: 'D-', size: 'small' },
  { id: 'am_adam', name: 'Adam', locale: 'en-US', gender: 'male', grade: 'F+', size: 'small' },
  { id: 'bf_emma', name: 'Emma', locale: 'en-GB', gender: 'female', grade: 'B-', size: 'small' },
  { id: 'bf_isabella', name: 'Isabella', locale: 'en-GB', gender: 'female', grade: 'C', size: 'small' },
  { id: 'bm_george', name: 'George', locale: 'en-GB', gender: 'male', grade: 'C', size: 'small' },
  { id: 'bm_fable', name: 'Fable', locale: 'en-GB', gender: 'male', grade: 'C', size: 'small' },
  { id: 'bm_lewis', name: 'Lewis', locale: 'en-GB', gender: 'male', grade: 'D+', size: 'small' },
  { id: 'bf_alice', name: 'Alice', locale: 'en-GB', gender: 'female', grade: 'D', size: 'small' },
  { id: 'bf_lily', name: 'Lily', locale: 'en-GB', gender: 'female', grade: 'D', size: 'small' },
  { id: 'bm_daniel', name: 'Daniel', locale: 'en-GB', gender: 'male', grade: 'D', size: 'small' },
  { id: 'ef_dora', name: 'Dora', locale: 'es', gender: 'female', grade: 'D', size: 'small' },
  { id: 'em_alex', name: 'Alex', locale: 'es', gender: 'male', grade: 'D', size: 'small' },
  { id: 'em_santa', name: 'Santa', locale: 'es', gender: 'male', grade: 'D', size: 'small' },
  { id: 'ff_siwis', name: 'Siwis', locale: 'fr', gender: 'female', grade: 'B-', size: 'small' },
  { id: 'bella', name: 'Bella', locale: 'en-US', gender: 'female', grade: '', size: 'tiny' },
  { id: 'luna', name: 'Luna', locale: 'en-US', gender: 'female', grade: '', size: 'tiny' },
  { id: 'rosie', name: 'Rosie', locale: 'en-US', gender: 'female', grade: '', size: 'tiny' },
  { id: 'kiki', name: 'Kiki', locale: 'en-US', gender: 'female', grade: '', size: 'tiny' },
  { id: 'jasper', name: 'Jasper', locale: 'en-US', gender: 'male', grade: '', size: 'tiny' },
  { id: 'bruno', name: 'Bruno', locale: 'en-US', gender: 'male', grade: '', size: 'tiny' },
  { id: 'hugo', name: 'Hugo', locale: 'en-US', gender: 'male', grade: '', size: 'tiny' },
  { id: 'leo', name: 'Leo', locale: 'en-US', gender: 'male', grade: '', size: 'tiny' },
  { id: 'st_f1', name: 'F1', locale: 'multi', gender: 'female', grade: '', size: 'multi' },
  { id: 'st_f2', name: 'F2', locale: 'multi', gender: 'female', grade: '', size: 'multi' },
  { id: 'st_f3', name: 'F3', locale: 'multi', gender: 'female', grade: '', size: 'multi' },
  { id: 'st_f4', name: 'F4', locale: 'multi', gender: 'female', grade: '', size: 'multi' },
  { id: 'st_f5', name: 'F5', locale: 'multi', gender: 'female', grade: '', size: 'multi' },
  { id: 'st_m1', name: 'M1', locale: 'multi', gender: 'male', grade: '', size: 'multi' },
  { id: 'st_m2', name: 'M2', locale: 'multi', gender: 'male', grade: '', size: 'multi' },
  { id: 'st_m3', name: 'M3', locale: 'multi', gender: 'male', grade: '', size: 'multi' },
  { id: 'st_m4', name: 'M4', locale: 'multi', gender: 'male', grade: '', size: 'multi' },
  { id: 'st_m5', name: 'M5', locale: 'multi', gender: 'male', grade: '', size: 'multi' },
]

/** Languages Supertonic 2 (`multi`) can speak with any of its voices. */
export const TTS_LANGUAGES = {
  en: 'English',
  ko: 'Korean',
  es: 'Spanish',
  pt: 'Portuguese',
  fr: 'French',
} as const

export type TTSLanguage = keyof typeof TTS_LANGUAGES

const LOCALE_LABEL: Record<TTSLocale, string> = {
  'en-US': 'American English',
  'en-GB': 'British English',
  es: 'Spanish',
  fr: 'French',
  multi: 'Supertonic · English, Korean, Spanish, Portuguese, French',
}

export function voicesFor(size: TTSSize = DEFAULT_TTS_SIZE): TTSVoice[] {
  return TTS_VOICES.filter((v) => v.size === size)
}

export function defaultVoiceFor(size: TTSSize = DEFAULT_TTS_SIZE): string {
  return size === 'tiny' ? DEFAULT_TINY_VOICE : size === 'multi' ? DEFAULT_MULTI_VOICE : DEFAULT_VOICE
}

export function getVoice(id: string): TTSVoice | undefined {
  return TTS_VOICES.find((v) => v.id === id)
}

export const TTS_VOICE_GROUPS = voiceGroupsFor(DEFAULT_TTS_SIZE)

export function voiceGroupsFor(size: TTSSize = DEFAULT_TTS_SIZE) {
  const voices = voicesFor(size)
  const locales = [...new Set(voices.map((v) => v.locale))]
  return locales.map((locale) => ({
    locale,
    label: LOCALE_LABEL[locale],
    voices: voices.filter((v) => v.locale === locale),
  }))
}

export function voiceLabel(voice: TTSVoice): string {
  if (voice.locale === 'multi') return `${voice.name} · ${voice.gender}`
  const region =
    voice.locale === 'en-GB' ? 'British' : voice.locale === 'es' ? 'Spanish' : voice.locale === 'fr' ? 'French' : 'American'
  const grade = voice.grade ? `${voice.gender} · ${voice.grade}` : voice.gender
  return `${voice.name} · ${region} ${grade}`
}

/** eSpeak language for Kokoro voices that `phonemizer` cannot handle (Spanish, French). */
export function kokoroPhonemeLang(voiceId: string): 'es' | 'fr-fr' | null {
  const prefix = voiceId.slice(0, 2)
  if (prefix === 'ef' || prefix === 'em') return 'es'
  if (prefix === 'ff') return 'fr-fr'
  return null
}
