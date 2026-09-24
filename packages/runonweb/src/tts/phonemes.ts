import { kokoroPhonemeLang } from './voices.ts'

const ESPEAK_JS = 'https://cdn.jsdelivr.net/npm/espeak-ng@1.0.2/dist/espeak-ng.js'
const ESPEAK_WASM = 'https://cdn.jsdelivr.net/npm/espeak-ng@1.0.2/dist/espeak-ng.wasm'

type ESpeakInstance = {
  FS: { readFile: (name: string, opts: { encoding: string }) => string }
}

type ESpeakFactory = (opts: {
  locateFile?: (file: string) => string
  arguments?: string[]
}) => Promise<ESpeakInstance>

let factory: ESpeakFactory | null = null

/**
 * Convert text to IPA for Kokoro. English uses `phonemizer` with Kokoro's text normalization
 * and phoneme fixes (ported from kokoro-js); ES/FR load eSpeak-NG WASM on demand.
 */
export async function phonemizeLang(text: string, lang: string): Promise<string> {
  if (lang === 'en-us' || lang === 'en-gb' || lang === 'en') {
    return phonemizeEnglish(text, lang === 'en-us' ? 'a' : 'b')
  }
  return phonemizeEspeak(text, lang)
}

const PUNCT = ';:,.!?¡¿—…"«»“”(){}[]'
const PUNCT_RE = new RegExp(`(\\s*[${PUNCT.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}]+\\s*)+`, 'g')

async function phonemizeEnglish(text: string, variant: 'a' | 'b'): Promise<string> {
  const { phonemize } = await import('phonemizer')
  const normalized = normalizeEnglish(text)
  const lang = variant === 'a' ? 'en-us' : 'en'

  // Phonemize the text between punctuation runs; keep the punctuation verbatim.
  const parts: string[] = []
  let last = 0
  for (const m of normalized.matchAll(PUNCT_RE)) {
    const idx = m.index ?? 0
    if (last < idx) parts.push((await phonemize(normalized.slice(last, idx), lang)).join(' '))
    if (m[0].length > 0) parts.push(m[0])
    last = idx + m[0].length
  }
  if (last < normalized.length) parts.push((await phonemize(normalized.slice(last), lang)).join(' '))

  let ps = parts
    .join('')
    .replace(/kəkˈoːɹoʊ/g, 'kˈoʊkəɹoʊ')
    .replace(/kəkˈɔːɹəʊ/g, 'kˈəʊkəɹəʊ')
    .replace(/ʲ/g, 'j')
    .replace(/r/g, 'ɹ')
    .replace(/x/g, 'k')
    .replace(/ɬ/g, 'l')
    .replace(/(?<=[a-zɹː])(?=hˈʌndɹɪd)/g, ' ')
    .replace(/ z(?=[;:,.!?¡¿—…"«»“” ]|$)/g, 'z')
  if (variant === 'a') ps = ps.replace(/(?<=nˈaɪn)ti(?!ː)/g, 'di')
  return ps.replace(/\s+/g, ' ').trim()
}

function normalizeEnglish(text: string): string {
  return text
    .replace(/[‘’]/g, "'")
    .replace(/«/g, '“')
    .replace(/»/g, '”')
    .replace(/[“”]/g, '"')
    .replace(/\(/g, '«')
    .replace(/\)/g, '»')
    .replace(/、/g, ', ')
    .replace(/。/g, '. ')
    .replace(/！/g, '! ')
    .replace(/，/g, ', ')
    .replace(/：/g, ': ')
    .replace(/；/g, '; ')
    .replace(/？/g, '? ')
    .replace(/[^\S \n]/g, ' ')
    .replace(/  +/, ' ')
    .replace(/(?<=\n) +(?=\n)/g, '')
    .replace(/\bD[Rr]\.(?= [A-Z])/g, 'Doctor')
    .replace(/\b(?:Mr\.|MR\.(?= [A-Z]))/g, 'Mister')
    .replace(/\b(?:Ms\.|MS\.(?= [A-Z]))/g, 'Miss')
    .replace(/\b(?:Mrs\.|MRS\.(?= [A-Z]))/g, 'Mrs')
    .replace(/\betc\.(?! [A-Z])/gi, 'etc')
    .replace(/\b(y)eah?\b/gi, "$1e'a")
    .replace(/\d*\.\d+|\b\d{4}s?\b|(?<!:)\b(?:[1-9]|1[0-2]):[0-5]\d\b(?!:)/g, splitNum)
    .replace(/(?<=\d),(?=\d)/g, '')
    .replace(/[$£]\d+(?:\.\d+)?(?: hundred| thousand| (?:[bm]|tr)illion)*\b|[$£]\d+\.\d\d?\b/gi, flipMoney)
    .replace(/\d*\.\d+/g, pointNum)
    .replace(/(?<=\d)-(?=\d)/g, ' to ')
    .replace(/(?<=\d)S/g, ' S')
    .replace(/(?<=[BCDFGHJ-NP-TV-Z])'?s\b/g, "'S")
    .replace(/(?<=X')S\b/g, 's')
    .replace(/(?:[A-Za-z]\.){2,} [a-z]/g, (m) => m.replace(/\./g, '-'))
    .replace(/(?<=[A-Z])\.(?=[A-Z])/gi, '-')
    .trim()
}

function splitNum(match: string): string {
  if (match.includes('.')) return match
  if (match.includes(':')) {
    const [h, m] = match.split(':').map(Number)
    if (m === 0) return `${h} o'clock`
    if (m! < 10) return `${h} oh ${m}`
    return `${h} ${m}`
  }
  const year = parseInt(match.slice(0, 4), 10)
  if (year < 1100 || year % 1000 < 10) return match
  const left = match.slice(0, 2)
  const right = parseInt(match.slice(2, 4), 10)
  const suffix = match.endsWith('s') ? 's' : ''
  if (year % 1000 >= 100 && year % 1000 <= 999) {
    if (right === 0) return `${left} hundred${suffix}`
    if (right < 10) return `${left} oh ${right}${suffix}`
  }
  return `${left} ${right}${suffix}`
}

function flipMoney(match: string): string {
  const unit = match[0] === '$' ? 'dollar' : 'pound'
  if (isNaN(Number(match.slice(1)))) return `${match.slice(1)} ${unit}s`
  if (!match.includes('.')) {
    const s = match.slice(1) === '1' ? '' : 's'
    return `${match.slice(1)} ${unit}${s}`
  }
  const [whole, frac] = match.slice(1).split('.')
  const cents = parseInt((frac ?? '').padEnd(2, '0'), 10)
  const centUnit = match[0] === '$' ? (cents === 1 ? 'cent' : 'cents') : cents === 1 ? 'penny' : 'pence'
  return `${whole} ${unit}${whole === '1' ? '' : 's'} and ${cents} ${centUnit}`
}

function pointNum(match: string): string {
  const [whole, frac] = match.split('.')
  return `${whole} point ${(frac ?? '').split('').join(' ')}`
}

async function phonemizeEspeak(text: string, lang: string): Promise<string> {
  if (!factory) {
    const mod = (await import(/* @vite-ignore */ ESPEAK_JS)) as { default: ESpeakFactory }
    factory = mod.default
  }
  const instance = await factory({
    locateFile: (file) => (file.endsWith('.wasm') ? ESPEAK_WASM : file),
    arguments: ['--phonout', 'generated', '-q', '-b=1', '--ipa=3', '-v', lang, text],
  })
  return instance.FS.readFile('generated', { encoding: 'utf8' }).replace(/\s+/g, ' ').trim()
}

export function needsExternalPhonemes(voiceId: string): boolean {
  return kokoroPhonemeLang(voiceId) != null
}
