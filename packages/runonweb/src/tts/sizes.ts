export type TTSSize = 'tiny' | 'small' | 'multi'

export const TTS_SIZES: Record<
  TTSSize,
  { label: string; engine: string; params: string; downloadMB: string; quality: string }
> = {
  tiny: {
    label: 'Tiny',
    engine: 'KittenTTS',
    params: '15M',
    downloadMB: '~28 MB',
    quality: '8 English voices',
  },
  small: {
    label: 'Small',
    engine: 'Kokoro 82M',
    params: '82M',
    downloadMB: '~326 MB · WASM ~92 MB',
    quality: 'English · Spanish · French',
  },
  multi: {
    label: 'Multi',
    engine: 'Supertonic 2',
    params: '66M',
    downloadMB: '~262 MB',
    quality: 'en · ko · es · pt · fr · 44.1 kHz',
  },
}

export const DEFAULT_TTS_SIZE: TTSSize = 'small'
